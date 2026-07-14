#!/usr/bin/env python
"""
main.py — 向量检索实验主入口

职责：
  - 单机/多机分布式 HNSW 建图与分区
  - 通过 SSH/srun 在集群各节点启动 node_start.py worker 和 query.py 客户端
  - 文件屏障同步多机并发查询，聚合结果并保存

运行模式：
  1. 分布式模式 (--distributed)：通过 SDK 远程操控各 worker 节点，建图/分区/查询均在远端
  2. 单机模拟模式（无 --distributed）：本地创建 Cluster + 多 HNSW 节点，本地查询
"""

import os
import time
import argparse
import numpy as np
import json
import subprocess
import shlex
from tqdm import tqdm
import threading

import sys
sys.setrecursionlimit(5000)

# 集群部署参数统一从 config.json 读取；并行 Slurm job 可通过 VR_CONFIG_PATH 隔离配置文件
_CONFIG_PATH = os.environ.get(
    "VR_CONFIG_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json"),
)
with open(_CONFIG_PATH, "r") as _f:
    _CLUSTER_CONFIG = json.load(_f)

class QueryStallError(Exception):
    """查询进度停滞异常：当进度条超过指定时间没有变化时抛出"""
    pass

select_parser = argparse.ArgumentParser(add_help=False)
select_parser.add_argument("--use_cpp", type=bool, default=True,  
                           help="whether to use the C++ implementation")
select_args, rest = select_parser.parse_known_args()
use_cpp = select_args.use_cpp

project_root = os.path.abspath(os.path.dirname(__file__))
py_source_dir = os.path.join(project_root, 'source/python')
cpp_source_dir = os.path.join(project_root, 'source/cpp')
build_dir = os.path.join(project_root, 'build')

_dataset_root = os.path.join(project_root, 'dataset')
TIMING_FIELDS = (
    "total_ms",
    "search_ms",
    "serialize_ms",
    "deserialize_ms",
    "queue_ms",
    "result_ms",
    "network_ms",
)

DATASET_PRESETS = {
    "siftsmall": {
        "data_path":        os.path.join(_dataset_root, "siftsmall/siftsmall_base.fvecs"),
        "query_path":       os.path.join(_dataset_root, "siftsmall/siftsmall_query.fvecs"),
        "groundtruth_path": os.path.join(_dataset_root, "siftsmall/siftsmall_groundtruth.ivecs"),
        "dtype":            "float",
        "dim":              128,
    },
    "sift": {
        "data_path":        os.path.join(_dataset_root, "sift/sift_base.fvecs"),
        "query_path":       os.path.join(_dataset_root, "sift/sift_query.fvecs"),
        "groundtruth_path": os.path.join(_dataset_root, "sift/sift_groundtruth.ivecs"),
        "dtype":            "float",
        "dim":              128,
    },
    "sift100m": {
        "data_path":        os.path.join(_dataset_root, "sift100m/base.u8bin"),
        "query_path":       os.path.join(_dataset_root, "sift100m/queries.u8bin"),
        "groundtruth_path": os.path.join(_dataset_root, "sift100m/gt_100.bin"),
        "dtype":            "float",
        "dim":              128,
    },
    "sift100m-u8": {
        "data_path":        os.path.join(_dataset_root, "sift100m/base.u8bin"),
        "query_path":       os.path.join(_dataset_root, "sift100m/queries.u8bin"),
        "groundtruth_path": os.path.join(_dataset_root, "sift100m/gt_100.bin"),
        "dtype":            "uint8",
        "dim":              128,
    },
    "deep100m": {
        "data_path":        os.path.join(_dataset_root, "deep100m/base.fbin"),
        "query_path":       os.path.join(_dataset_root, "deep100m/queries.fbin"),
        "groundtruth_path": os.path.join(_dataset_root, "deep100m/gt_100.bin"),
        "dtype":            "float",
        "dim":              96,
    },
    "uint8_test": {
        "data_path":        os.path.join(_dataset_root, "uint8_test/base.u8bin"),
        "query_path":       os.path.join(_dataset_root, "uint8_test/queries.u8bin"),
        "groundtruth_path": os.path.join(_dataset_root, "uint8_test/gt_100.bin"),
        "dtype":            "uint8",
        "dim":              128,
    },
}

_DTYPE_CLASS_MAP = {
    "float":  ("PayloadFloat",  "HNSWGraphFloat",  "ClusterFloat",  "ClusterSDKClientFloat"),
    "double": ("PayloadDouble", "HNSWGraphDouble", "ClusterDouble", "ClusterSDKClientDouble"),
    "int32":  ("PayloadInt32",  "HNSWGraphInt32",  "ClusterInt32",  "ClusterSDKClientInt32"),
    "uint32": ("PayloadUInt32", "HNSWGraphUInt32", "ClusterUInt32", "ClusterSDKClientUInt32"),
    # uint8 存储 + int32 距离：复用 PayloadInt32（HNSW/Cluster/SDK 内部 DistT=int32_t）
    "uint8":  ("PayloadInt32",  "HNSWGraphUInt8",  "ClusterUInt8",  "ClusterSDKClientUInt8"),
}

# -------------------- 模块级初始化 --------------------
# dtype 两阶段解析：先尝试命令行 --dtype，否则从 preset 推断。
# 必须在 import C++ 模块之前确定类型，以加载正确的模板实例化。
_dtype_parser = argparse.ArgumentParser(add_help=False)
_dtype_parser.add_argument("--dtype", type=str, default=None)
_dtype_parser.add_argument("--data_name", type=str, default="sift")
_dtype_args, _ = _dtype_parser.parse_known_args()

# 优先使用显式 --dtype，否则从 preset 推断
_active_dtype = _dtype_args.dtype
if _active_dtype is None:
    _preset = DATASET_PRESETS.get(_dtype_args.data_name, {})
    _active_dtype = _preset.get("dtype", "float")

if use_cpp:
    if os.path.isdir(build_dir):
        os.environ['PYTHONPATH'] = f"{build_dir}:{os.environ.get('PYTHONPATH','')}"
        if build_dir not in sys.path:
            sys.path.insert(0, build_dir)

    try:
        import vector_retrieve as _vr
        _payload_cls, _hnsw_cls, _cluster_cls, _sdk_cls = _DTYPE_CLASS_MAP[_active_dtype]
        Payload = getattr(_vr, _payload_cls)
        HNSWGraph = getattr(_vr, _hnsw_cls)
        Cluster = getattr(_vr, _cluster_cls)
        ClusterSDKClient = getattr(_vr, _sdk_cls)
    except ImportError:
        # query.py 等其他脚本 import main.py 时可能不需要 C++ 模块
        if __name__ == "__main__":
            raise
else:
    if os.path.isdir(py_source_dir):
        if py_source_dir not in sys.path:
            sys.path.insert(0, py_source_dir)

    from payload import Payload
    from hnsw import HNSWGraph
    from cluster import Cluster

def read_fvecs(filename, c_contiguous=True):
    """读取 .fvecs 格式的文件"""
    fv = np.fromfile(filename, dtype=np.float32)
    dim = fv.view(np.int32)[0]
    fv = fv.reshape(-1, 1 + dim)
    vectors = fv[:, 1:]
    if c_contiguous:
        return np.ascontiguousarray(vectors)
    return vectors

def read_ivecs(filename, c_contiguous=True):
    """读取 .ivecs 格式的文件"""
    iv = np.fromfile(filename, dtype=np.int32)
    dim = iv[0]
    iv = iv.reshape(-1, 1 + dim)
    vectors = iv[:, 1:]
    if c_contiguous:
        return np.ascontiguousarray(vectors)
    return vectors

def read_u8bin(filename, c_contiguous=True, target_dtype=None):
    """读取 BigANN .u8bin 格式：8字节头(nrows, ncols) + uint8数据。
    根据 _active_dtype 决定返回数组的 dtype：
      - dtype=='uint8'  → 原样保留 uint8（零拷贝路径，节省内存带宽）
      - dtype=='float'  → 转为 float
      - dtype=='double' → 转为 double
      - dtype=='int32'/'uint32' → 整数再扩展
    显式传 target_dtype 可覆盖 _active_dtype。"""
    with open(filename, 'rb') as f:
        nrows, ncols = np.frombuffer(f.read(8), dtype=np.uint32)
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(nrows, ncols)
    if target_dtype is None:
        dtype_map = {"int32": np.int32, "uint32": np.uint32, "float": np.float32, "double": np.float64, "uint8": np.uint8}
        target_dtype = dtype_map.get(_active_dtype, np.float32)
    # uint8 → uint8：astype 在 dtype 一致时返回视图/副本，但 numpy 通常会 copy；
    # 下面的 ascontiguousarray 也会拷贝一次，足够保证连续布局。
    vectors = data.astype(target_dtype, copy=False) if target_dtype == np.uint8 else data.astype(target_dtype)
    if c_contiguous:
        return np.ascontiguousarray(vectors)
    return vectors

def read_fbin(filename, c_contiguous=True):
    """读取 BigANN .fbin 格式：8字节头(nrows, ncols) + float32数据"""
    with open(filename, 'rb') as f:
        nrows, ncols = np.frombuffer(f.read(8), dtype=np.uint32)
        data = np.frombuffer(f.read(), dtype=np.float32).reshape(nrows, ncols)
    if c_contiguous:
        return np.ascontiguousarray(data)
    return data

def read_ibin(filename, c_contiguous=True):
    """读取 BigANN .ibin/.bin 格式的 ground truth：8字节头(nrows, ncols) + int32 IDs [+ float32 distances]"""
    with open(filename, 'rb') as f:
        nrows, ncols = np.frombuffer(f.read(8), dtype=np.uint32)
        raw = np.frombuffer(f.read(), dtype=np.int32)
    total = int(nrows) * int(ncols)
    if raw.size == total:
        # 仅包含 IDs
        data = raw.reshape(nrows, ncols)
    elif raw.size == total * 2:
        # 包含 IDs + distances：先存所有 IDs (nrows*ncols)，再存所有 distances (nrows*ncols)
        data = raw[:total].reshape(nrows, ncols)
    else:
        raise ValueError(f"Unexpected data size {raw.size} for ibin (nrows={nrows}, ncols={ncols})")
    if c_contiguous:
        return np.ascontiguousarray(data)
    return data

def read_vectors(filename, c_contiguous=True):
    """根据文件扩展名自动选择读取函数（向量数据）"""
    ext = os.path.splitext(filename)[1].lower()
    if ext == '.fvecs':
        return read_fvecs(filename, c_contiguous)
    elif ext == '.u8bin':
        return read_u8bin(filename, c_contiguous)
    elif ext == '.fbin':
        return read_fbin(filename, c_contiguous)
    else:
        raise ValueError(f"Unsupported vector format: {ext} ({filename})")

def read_groundtruth(filename, c_contiguous=True):
    """根据文件扩展名自动选择读取函数（ground truth）"""
    ext = os.path.splitext(filename)[1].lower()
    if ext == '.ivecs':
        return read_ivecs(filename, c_contiguous)
    elif ext in ('.bin', '.ibin'):
        return read_ibin(filename, c_contiguous)
    else:
        raise ValueError(f"Unsupported groundtruth format: {ext} ({filename})")

class VectorRetrieveRunner:
    """向量检索实验运行器，封装建图、分区、查询全流程。

    通过 config.json 获取集群拓扑，支持：
      - 协调器模式（host 不在 work_nodes 中）：仅调度，不运行本地 worker
      - 混合模式（host 在 work_nodes 中）：调度 + 本地 worker"""

    def __init__(self, args, use_cpp):
        self.args = args
        self.use_cpp = use_cpp

        self.num_construction_threads = args.num_construction_threads
        self.num_query_threads = args.num_query_threads  # 全局查询线程总数
        self.num_threads = args.num_threads  # 服务端每 node 搜索线程数（msg_pool_size）
        self.num_cpu = args.num_cpu
        self.query_worker_rss_samples = []
        self.run_id = os.environ.get("SLURM_JOB_ID") or f"manual-{os.getpid()}"
        self.runtime_dir = os.path.abspath(os.path.join(project_root, "tmp", f"run-{self.run_id}"))
        os.makedirs(self.runtime_dir, exist_ok=True)
        self.worker_dim = args.dim or self._resolve_worker_dim(args)

        # 集群拓扑从 config.json 读取。约定：
        #   host_node — 运行 main.py 的机器（可不在 work_nodes 中，此时仅作协调器）
        #   work_nodes — 参与 HNSW 存储/搜索的节点列表
        #   query_nodes — 发查询的机器列表（可选，用于多机分布式压测）
        self.host_info = _CLUSTER_CONFIG["host_node"]
        self.all_nodes = list(_CLUSTER_CONFIG["work_nodes"])
        # query_nodes：负责发查询的机器列表（可选，用于多机分布式 query）
        self.query_nodes = list(_CLUSTER_CONFIG.get("query_nodes", []))
        # --num_query_nodes 截断：仅使用前 N 个 query 节点
        num_query_nodes = getattr(args, 'num_query_nodes', 0) or len(self.query_nodes)
        self.query_nodes = self.query_nodes[:num_query_nodes]
        for key in ("ip", "port", "user_name"):
            if key not in self.host_info:
                raise ValueError(f"config.json: host_node 缺少字段 '{key}'")
        for idx, nd in enumerate(self.all_nodes):
            for key in ("ip", "port", "user_name"):
                if key not in nd:
                    raise ValueError(f"config.json: work_nodes[{idx}] 缺少字段 '{key}'")
        for idx, nd in enumerate(self.query_nodes):
            for key in ("ip", "port", "user_name"):
                if key not in nd:
                    raise ValueError(f"config.json: query_nodes[{idx}] 缺少字段 '{key}'")

        self.host_ip = self.host_info["ip"]
        self.host_port = int(self.host_info["port"])
        self.host_user_name = self.host_info["user_name"]

        # 定位 host 在 nodes 列表中的位置（未找到则 -1，代表协调器模式）
        self._host_index_in_nodes = -1
        for idx, nd in enumerate(self.all_nodes):
            if nd["ip"] == self.host_ip and int(nd["port"]) == self.host_port:
                self._host_index_in_nodes = idx
                break
        self.host_is_node = (self._host_index_in_nodes >= 0)

        # 兼容旧字段
        self.user_name = self.host_user_name

    # 选择实际使用的节点子集：按 work_nodes 顺序取前 node_num 个。
    # 若 host 在 work_nodes 中，保证子集始终包含 host。
        if self.args.distributed:
            requested = self.args.node_num if self.args.node_num and self.args.node_num > 0 else len(self.all_nodes)
            requested = min(requested, len(self.all_nodes))
            if requested <= 0:
                raise ValueError("config.json: nodes 为空，分布式模式无法启动")

            if self.host_is_node:
                host_entry = self.all_nodes[self._host_index_in_nodes]
                if requested <= 1:
                    selected = [host_entry]
                elif self._host_index_in_nodes < requested:
                    # host 已在前 requested 个，直接取前缀
                    selected = list(self.all_nodes[:requested])
                else:
                    # host 在后段，用前 requested-1 个 + host 占最后一个槽
                    selected = list(self.all_nodes[:requested - 1]) + [host_entry]
            else:
                # 协调器模式：host 不在集群中，按前缀选取
                selected = list(self.all_nodes[:requested])

            self.node_ips = [nd["ip"] for nd in selected]
            self.node_ports = [int(nd["port"]) for nd in selected]
            self.node_user_names = [nd["user_name"] for nd in selected]
            self.node_hostnames = [nd.get("hostname", nd["ip"]) for nd in selected]
            self.node_num = len(self.node_ips)
            if self.host_is_node:
                # host_id 为 host 在子集中的实际位置（不一定是 0）
                self.host_id = next(
                    i for i, nd in enumerate(selected)
                    if nd["ip"] == self.host_ip and int(nd["port"]) == self.host_port
                )
            else:
                self.host_id = -1  # 协调器模式：无本地 worker
        else:
            # 单机多进程模拟：不使用真实 IP 列表，保持原行为
            self.node_ips = [self.host_ip]
            self.node_ports = [self.host_port]
            self.node_user_names = [self.host_user_name]
            self.node_num = self.args.node_num
            self.host_id = 0

        self.data_name = args.data_name
        self.data_path = args.data_path
        self.query_path = args.query_path
        self.groundtruth_path = args.groundtruth_path

        self.graph_path = args.graph_path
        self.cluster_path = args.cluster_path
        self.loaded_partitioned_graphs = False
        self.loaded_full_graph_direct = False
        self.full_hnsw_release_needed = False
        self.full_hnsw_released = False

        self.k = args.k
        self.m = args.m
        self.ef_construction = args.ef_construction
        self.ef_search = args.ef_search
        self.limited_level = args.limited_level
        self.partition_mode = args.partition_mode
        self.partition_distribution = args.partition_distribution
        self.use_layer = args.layer
        # 可选分区路径标签（VR_PART_TAG）：区分同 (mode,rep) 但不同复制策略的分区
        # 例如加权复制 weighted budget=0.05 -> VR_PART_TAG=w005，避免覆盖 uniform 分区
        _pt = os.environ.get("VR_PART_TAG", "").strip()
        self._part_tag_sfx = f"-{_pt}" if _pt else ""

        # layer_config 与 layer_replicate_flags 从 config.json 读取，缺失时回退默认
        _default_layer_config = {l: list(range(self.node_num)) for l in range(self.limited_level + 1)}
        _default_layer_flags = [l >= 2 for l in range(self.limited_level + 1)]

        raw_layer_config = _CLUSTER_CONFIG.get("layer_config")
        if raw_layer_config:
            self.layer_config = {int(k): list(v) for k, v in raw_layer_config.items()}
        else:
            self.layer_config = _default_layer_config

        raw_layer_flags = _CLUSTER_CONFIG.get("layer_replicate_flags")
        if raw_layer_flags:
            self.layer_replicate_flags = [bool(v) for v in raw_layer_flags]
        else:
            self.layer_replicate_flags = _default_layer_flags

        self.replication = args.replication
        # 格式化为文件名友好的字符串，避免 f"rep{2.0}" 变成 "rep2.0"
        self._rep_fmt = str(int(self.replication)) if self.replication == int(self.replication) else str(self.replication)

        # 通信模式参数
        self.comm_mode = getattr(args, 'comm_mode', 'tcp')
        self.ib_dev = getattr(args, 'ib_dev', '')
        self.ib_port = getattr(args, 'ib_port', 1)
        self.gid_index = getattr(args, 'gid_index', -1)

        # 折返检测窗口大小（0 关闭）
        self.max_track_num = getattr(args, 'max_track_num', 10)

        self.cluster = None
        self.hnsw = None
        self.sdk_config = None
        self.worker_processes = []
        self.num_queries = 0

    @staticmethod
    def _resolve_worker_dim(args):
        ext = os.path.splitext(args.data_path or "")[1].lower()
        if ext in ('.fbin', '.u8bin', '.ibin', '.bin'):
            with open(args.data_path, 'rb') as f:
                header = np.frombuffer(f.read(8), dtype=np.uint32)
            if header.size == 2:
                return int(header[1])
        if ext in ('.fvecs', '.ivecs'):
            with open(args.data_path, 'rb') as f:
                header = np.frombuffer(f.read(4), dtype=np.int32)
            if header.size == 1:
                return int(header[0])
        return 128

    def setup_distributed_nodes(self):
        """在集群各节点启动 worker 进程。

        流程：
          1. 清理所有节点上的旧 node_start.py 进程（两阶段：TERM + KILL）
          2. 等待端口释放（TCP TIME_WAIT）
          3. 通过 srun（SLURM）或 SSH 在各节点启动 node_start.py
          4. TCP 端口探测确认所有 worker 就绪"""
        if not self.args.distributed:
            return
        if self.args.skip_prepare:
            print("Skipping distributed node setup (--skip_prepare)")
            self.worker_processes = []
            return

        print(f"Distributed mode enabled. Node IPs: {self.node_ips}")
        print(f"Using cluster config: {_CONFIG_PATH}")

        # Clean up old worker processes on all remote nodes and the local host
        print("Cleaning up old worker processes...")
        use_slurm = bool(self.args.slurm)

        # SLURM 模式：用 pkill 清理旧 worker（不依赖 scancel，避免误杀）

        def _kill_local_node_start():
            # Two-stage local kill: TERM, wait, then KILL if still present
            try:
                p = subprocess.run(["pgrep", "-f", "node_start.py"], capture_output=True, text=True)
                if p.returncode == 0 and p.stdout.strip():
                    pids = p.stdout.strip().split()
                    print(f"Local node_start.py PIDs: {pids}")
                    subprocess.run(["kill", "-TERM"] + pids, check=False)
                    time.sleep(1)
                    p2 = subprocess.run(["pgrep", "-f", "node_start.py"], capture_output=True, text=True)
                    if p2.returncode == 0 and p2.stdout.strip():
                        pids2 = p2.stdout.strip().split()
                        print(f"Local force-killing PIDs: {pids2}")
                        subprocess.run(["kill", "-KILL"] + pids2, check=False)
            except Exception as e:
                print(f"Local cleanup error: {e}")

        # First, clean local processes (host)
        _kill_local_node_start()

        cleanup_procs = []
        for i in range(len(self.node_ips)):
            ip = self.node_ips[i]
            if ip == self.host_ip:
                print(f"Host machine identified as node {self.host_id} with IP {self.host_ip}.")
                continue
            # Remote two-stage kill: find PIDs, TERM, wait, then KILL if still present
            kill_script = (
                f"pids=$(pgrep -f node_start.py || true); "
                f"if [ -n \"$pids\" ]; then echo \"Killing PIDs: $pids on {ip}\"; kill -TERM $pids 2>/dev/null || true; sleep 1; "
                f"pids2=$(pgrep -f node_start.py || true); if [ -n \"$pids2\" ]; then echo \"Force killing PIDs: $pids2 on {ip}\"; kill -KILL $pids2 2>/dev/null || true; fi; fi"
            )
            if use_slurm:
                hostname = self.node_hostnames[i]
                cmd = ["srun", "-N1", "-n1", f"--cpus-per-task={self.num_cpu if self.num_cpu and self.num_cpu > 0 else self.num_threads}", "--overlap", "--overcommit",
                       "-w", hostname, "bash", "-lc", kill_script]
            else:
                cmd = [
                    "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30",
                    f"{self.node_user_names[i]}@{ip}",
                    kill_script
                ]
            print(f"  Cleaning node {i} on {ip}...")
            proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            cleanup_procs.append((proc, ip))
        # Wait for cleanup commands to finish (with timeout) and ensure remote processes are killed
        for proc, ip in cleanup_procs:
            try:
                stdout, stderr = proc.communicate(timeout=120)
                if stdout:
                    print(f"  [{ip}] {stdout.decode().strip()}")
                if stderr:
                    print(f"  [{ip}] ERROR: {stderr.decode().strip()}")
            except subprocess.TimeoutExpired:
                proc.kill()
                print(f"  [{ip}] Cleanup command timed out")
        print("Waiting for ports to be released...")
        time.sleep(5)  # 等待端口释放（TCP TIME_WAIT）

        # Launch all workers (including local host machine)
        current_dir = os.getcwd()
        worker_processes = []
        node_ips_str = ','.join(self.node_ips)
        for i in range(len(self.node_ips)):
            ip = self.node_ips[i]
            if ip == self.host_ip:
                print(f"Host machine identified as node {self.host_id} with IP {self.host_ip}.")
                # Launch local worker
                print(f"Launching worker node {i} on localhost...")
                log_file = os.path.join(self.runtime_dir, f"worker_{i}.log")

                distributed_flag = "--distributed" if self.args.distributed else ""
                shard_flag = "--shard" if self.args.shard else ""
                global_pull_flag = "--global-pull" if self.args.is_global_pull else ""

                config_env = os.environ.copy()
                config_env["VR_CONFIG_PATH"] = _CONFIG_PATH

                # Build local command
                local_cmd = [
                    "python", "-u", "node_start.py",
                    "--node_id", str(i),
                    "--node_ips", node_ips_str,
                    "--dtype", _active_dtype,
                    "--m", str(self.args.m),
                    "--ef_construction", str(self.args.ef_construction),
                    "--ef_search", str(self.ef_search),
                    "--limited_level", str(self.args.limited_level),
                    "--dim", str(self.worker_dim),
                    "--comm_mode", self.comm_mode,
                    "--num_threads", str(self.num_threads),
                    "--max_track_num", str(self.max_track_num),
                ]
                if self.num_cpu and self.num_cpu > 0:
                    local_cmd = ["taskset", "-c", f"0-{self.num_cpu - 1}"] + local_cmd
                if distributed_flag:
                    local_cmd.append(distributed_flag)
                if shard_flag:
                    local_cmd.append(shard_flag)
                if global_pull_flag:
                    local_cmd.append(global_pull_flag)

                # Launch local worker in background
                with open(log_file, 'w') as log_f:
                    proc = subprocess.Popen(local_cmd, stdout=log_f, stderr=log_f, env=config_env)
                worker_processes.append(proc)
            else:
                # Launch remote worker
                log_file = os.path.join(self.runtime_dir, f"worker_{i}.log")

                distributed_flag = "--distributed" if self.args.distributed else ""
                shard_flag = "--shard" if self.args.shard else ""
                global_pull_flag = "--global-pull" if self.args.is_global_pull else ""

                # slurm 自己通过 cgroups 做 CPU 绑定，不需要 taskset
                cpu_prefix = f"taskset -c 0-{self.num_cpu - 1} " if (self.num_cpu and self.num_cpu > 0 and not use_slurm) else ""

                if use_slurm:
                    # srun 方式启动远程 worker
                    hostname = self.node_hostnames[i]
                    srun_cmd = [
                        "srun", "-N1", "-n1", f"--cpus-per-task={self.num_cpu if self.num_cpu and self.num_cpu > 0 else self.num_threads}",
                        "--overlap", "--overcommit",
                        "--export=ALL",
                        "-w", hostname,
                        "-o", log_file,
                        "-e", log_file,
                        "bash", "-lc",
                        f"export VR_CONFIG_PATH={shlex.quote(_CONFIG_PATH)} && "
                        + (f"export VR_DROP_OUTER_LINKS={shlex.quote(os.environ['VR_DROP_OUTER_LINKS'])} && " if os.environ.get('VR_DROP_OUTER_LINKS') else "")
                        + f"cd {current_dir} && eval \"$(conda shell.bash hook)\" && conda activate vector-retrieve && "
                        f"{cpu_prefix}python -u node_start.py "
                        f"--node_id {i} --node_ips {node_ips_str} "
                        f"--dtype {_active_dtype} "
                        f"--m {self.args.m} --ef_construction {self.args.ef_construction} "
                        f"--ef_search {self.ef_search} --limited_level {self.args.limited_level} "
                        f"--dim {self.worker_dim} "
                        f"--comm_mode {self.comm_mode} "
                        f"--num_threads {self.num_threads} "
                        f"--max_track_num {self.max_track_num} "
                        f"{distributed_flag} {shard_flag} {global_pull_flag}"
                    ]
                    print(f"  CMD: {' '.join(srun_cmd)}")
                    proc = subprocess.Popen(srun_cmd, stdin=subprocess.DEVNULL,
                                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                else:
                    # SSH 方式启动远程 worker
                    print(f"Launching worker node {i} on {self.node_user_names[i]}@{ip}...")
                    remote_cmd = (
                        f"export VR_CONFIG_PATH={shlex.quote(_CONFIG_PATH)} && "
                        f"cd {current_dir} && "
                        f"conda activate vector-retrieve && "
                        f"nohup {cpu_prefix}python -u node_start.py "
                        f"--node_id {i} --node_ips {node_ips_str} "
                        f"--dtype {_active_dtype} "
                        f"--m {self.args.m} --ef_construction {self.args.ef_construction} "
                        f"--ef_search {self.ef_search} --limited_level {self.args.limited_level} "
                        f"--dim {self.worker_dim} "
                        f"--comm_mode {self.comm_mode} "
                        f"--num_threads {self.num_threads} "
                        f"--max_track_num {self.max_track_num} "
                        f"{distributed_flag} {shard_flag} {global_pull_flag} "
                        f"> {log_file} 2>&1 &"
                    )
                    cmd = ["ssh", f"{self.node_user_names[i]}@{ip}", remote_cmd]
                    print(f"  CMD: {' '.join(cmd)}")
                    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL,
                                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                worker_processes.append(proc)

        # 等待所有 worker 节点真正启动并可连接
        print(f"Waiting for {len(worker_processes)} worker nodes to start (including local node {self.host_id})...")
        max_wait = 300  # 最长等待 300 秒
        check_interval = 2
        waited = 0
        all_ready = False
        while waited < max_wait:
            time.sleep(check_interval)
            waited += check_interval
            all_ready = True
            for i in range(len(self.node_ips)):
                port = self.node_ports[i]
                connect_ip = "127.0.0.1" if i == self.host_id else self.node_ips[i]
                try:
                    import socket
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.settimeout(2)
                    s.connect((connect_ip, port))
                    s.close()
                except Exception:
                    all_ready = False
                    break
            if all_ready:
                print(f"All worker nodes are ready after {waited}s.")
                break
        if not all_ready:
            raise RuntimeError(f"Not all worker nodes are ready after {max_wait}s. Aborting to avoid query failures.")

        self.worker_processes = worker_processes

    def build_index(self, data, num_elements):
        """通过 SDK 将向量添加到远端/本地 HNSW 图。
        多线程模式下，每个线程独立持有 SDK 连接，分担建图负载。"""
        mode_label = "shard" if self.args.shard else "normal"
        if self.num_construction_threads > 1:
            print(f"Building in {mode_label} with {self.num_construction_threads} threads via SDK...")
            start_time = time.time()

            data_with_ids = [(idx, vector) for idx, vector in enumerate(data[:num_elements])]

            chunk_size = (len(data_with_ids) + self.num_construction_threads - 1) // self.num_construction_threads
            threads = []
            progress_counter = [0]
            progress_lock = threading.Lock()

            def add_chunk_via_sdk(chunk, counter, lock, config):
                thread_client = ClusterSDKClient(ip=config["ip"], port=config["port"], comm_mode=self.comm_mode)
                if not thread_client.connect():
                    raise RuntimeError("Thread failed to connect to local node for building")
                try:
                    local_count = 0
                    for vid, vec in chunk:
                        thread_client.add_vector(vid, vec)
                        local_count += 1
                        if local_count % 10 == 0:
                            with lock:
                                counter[0] += 10
                                local_count = 0
                    if local_count > 0:
                        with lock:
                            counter[0] += local_count
                finally:
                    thread_client.disconnect()

            for t in range(self.num_construction_threads):
                start_idx = t * chunk_size
                end_idx = min(start_idx + chunk_size, len(data_with_ids))
                if start_idx >= len(data_with_ids):
                    break
                chunk = data_with_ids[start_idx:end_idx]
                thread = threading.Thread(target=add_chunk_via_sdk, args=(chunk, progress_counter, progress_lock, self.sdk_config))
                thread.start()
                threads.append(thread)

            with tqdm(total=len(data_with_ids), desc="Building", unit="vec") as pbar:
                last_value = 0
                while any(t.is_alive() for t in threads):
                    with progress_lock:
                        current = progress_counter[0]
                    pbar.update(current - last_value)
                    last_value = current
                    time.sleep(0.1)

                for thread in threads:
                    thread.join()

                with progress_lock:
                    current = progress_counter[0]
                pbar.update(current - last_value)

            build_time = time.time() - start_time
            print(f"Building Time: {build_time:.4f} seconds (using {self.num_construction_threads} threads via SDK).")

            # 分布式模式下打印各节点的向量数量
            if self.args.distributed and self.args.shard:
                self._print_node_vector_counts()
            return build_time

        print(f"Building in {mode_label} via SDK...")
        start_time = time.time()
        thread_client = ClusterSDKClient(ip=self.sdk_config["ip"], port=self.sdk_config["port"], comm_mode=self.comm_mode)
        if not thread_client.connect():
            raise RuntimeError("Failed to connect SDK to local node for building")
        try:
            for idx, vector in tqdm(enumerate(data[:num_elements]), total=num_elements, desc="Building", unit="vec"):
                thread_client.add_vector(idx, vector)
            build_time = time.time() - start_time
            print(f"Building Time: {build_time:.4f} seconds.")

            # 分布式模式下打印各节点的向量数量
            if self.args.distributed and self.args.shard:
                self._print_node_vector_counts()
            return build_time
        finally:
            thread_client.disconnect()

    def _print_node_vector_counts(self):
        """打印各节点的向量数量（分布式shard模式）"""
        print("Node vector counts:")
        # 通过 SDK 获取每个节点的信息
        for i, ip in enumerate(self.node_ips):
            try:
                port = self.node_ports[i]
                # 对于本地节点使用 127.0.0.1
                connect_ip = "127.0.0.1" if i == self.host_id else ip
                client = ClusterSDKClient(ip=connect_ip, port=port, comm_mode=self.comm_mode)
                if client.connect():
                    info = client.get_information()
                    client.disconnect()
                    if info and "cluster" in info and "nodes" in info["cluster"]:
                        nodes = info["cluster"]["nodes"]
                        # 找到当前连接节点的信息（is_remote_node=False 的那个）
                        count = None
                        for node_info in nodes:
                            if node_info.get("is_remote_node") == False:
                                count = node_info.get("vector_count", 0)
                                break
                        if count is not None:
                            print(f"  Node {i}: {count} vectors")
                            continue
                print(f"  Node {i}: (failed to get info)")
            except Exception as e:
                print(f"  Node {i}: (error: {e})")

    def _parallel_build(self, add_fn, data, num_elements, num_threads, build_desc):
        """多线程建图/建簇的公共循环：按线程切分数据，逐条调用 add_fn(vid, vec) 添加，
        并用 tqdm 显示进度。add_fn 传入 self.hnsw.add_vector（传统模式）或
        self.cluster.add_vector（shard 模式）。build_desc 为日志前缀。"""
        print(f"{build_desc} with {num_threads} threads...")
        start_time = time.time()

        # 准备数据：(id, vector) pairs（直接传numpy数组，避免需要GIL的类型转换）
        data_with_ids = [(idx, vector) for idx, vector in enumerate(data[:num_elements])]

        chunk_size = (len(data_with_ids) + num_threads - 1) // num_threads
        threads = []

        # 使用列表存储计数器（避免nonlocal作用域问题）
        progress_counter = [0]
        progress_lock = threading.Lock()

        def add_chunk(chunk, counter, lock):
            local_count = 0
            for vid, vec in chunk:
                add_fn(vid, vec)
                local_count += 1
                # 更频繁更新，提高速度显示准确性
                if local_count % 10 == 0:
                    with lock:
                        counter[0] += 10
                    local_count = 0
            # 更新剩余的
            if local_count > 0:
                with lock:
                    counter[0] += local_count

        # 启动所有线程
        for t in range(num_threads):
            start_idx = t * chunk_size
            end_idx = min(start_idx + chunk_size, len(data_with_ids))
            if start_idx >= len(data_with_ids):
                break
            chunk = data_with_ids[start_idx:end_idx]
            thread = threading.Thread(target=add_chunk, args=(chunk, progress_counter, progress_lock))
            thread.start()
            threads.append(thread)

        # 显示进度条
        with tqdm(total=len(data_with_ids), desc="Building", unit="vec") as pbar:
            last_value = 0
            while any(t.is_alive() for t in threads):
                with progress_lock:
                    current = progress_counter[0]
                pbar.update(current - last_value)
                last_value = current
                time.sleep(0.1)

            # 等待所有线程完成
            for thread in threads:
                thread.join()

            # 确保进度条到达100%
            with progress_lock:
                current = progress_counter[0]
            pbar.update(current - last_value)

        build_time = time.time() - start_time
        print(f"Building Machine Time: {build_time:.4f} seconds (using {num_threads} threads).")

    def prepare(self):
        """准备阶段：加载/构建 HNSW 图并分区。

        分支逻辑：
          1. --skip_prepare：跳过，仅设置 SDK 配置
          2. --distributed + 有 graph_path：通过 SDK 远程加载已有图
          3. --distributed + 无图：加载数据，通过 SDK 远程建图
          4. 非分布式 + 有 cluster_path：本地加载 Cluster
          5. 非分布式 + 有 graph_path：本地加载图 → 分区
          6. 非分布式 + shard：轮询建各节点独立图
          7. 非分布式 + 传统：先建统一图 → 再分区"""
        if self.args.skip_prepare:
            print("Skipping graph loading and partitioning (--skip_prepare)")
            if self.args.distributed:
                self.cluster = None
                self.hnsw = None
                sdk_idx = max(0, self.host_id)
                self.sdk_config = {
                    'ip': '127.0.0.1' if self.host_is_node else self.node_ips[0],
                    'port': self.node_ports[sdk_idx]
                }
            return

        # 在分布式模式下，不创建cluster和hnsw，直接使用SDK
        if self.args.distributed:
            # 分布式模式：所有操作通过SDK完成
            print("Distributed mode: Using SDK client for all operations")
            self.cluster = None
            self.hnsw = None

            # 分布式模式：如有 graph_path 则直接通过 SDK 加载
            sdk_idx = max(0, self.host_id)
            print(f"Configuring SDK for {'local node' if self.host_is_node else 'remote node'} {sdk_idx} ({self.node_ips[sdk_idx]}:{self.node_ports[sdk_idx]}) for building...")
            self.sdk_config = {
                'ip': '127.0.0.1' if self.host_is_node else self.node_ips[0],
                'port': self.node_ports[sdk_idx]
            }

            direct_full_graph = (
                self.node_num == 1
                and self.graph_path is not None
                and os.path.exists(self.graph_path)
                and not self.args.shard
                and not self.args.build_only
            )

            all_saved = False
            if not self.args.shard and not direct_full_graph:
                # 生成 cluster_path 和分区文件名
                layer_sfx = "-layer" if self.use_layer else ""
                if self.cluster_path:
                    cluster_base = self.cluster_path
                elif self.graph_path and os.path.isdir(self.graph_path):
                    cluster_base = os.path.join(self.graph_path,
                        f"m{self.m}-efc{self.ef_construction}-ll{self.limited_level}-"
                        f"{self.partition_mode}-rep{self._rep_fmt}{layer_sfx}{self._part_tag_sfx}")
                else:
                    cluster_base = (f"./save/{self.data_name}/node{self.node_num}/"
                        f"m{self.m}-efc{self.ef_construction}-ll{self.limited_level}-"
                        f"{self.partition_mode}-rep{self._rep_fmt}{layer_sfx}{self._part_tag_sfx}")
                self.cluster_path = cluster_base

                # 检查是否已有分区文件（优先加载，跳过建图和分区）
                all_saved = all(os.path.exists(f"{cluster_base}-{i}.bin") for i in range(self.node_num))

            if all_saved:
                # 已有分区文件：直接加载，无需加载完整 HNSW 图
                print(f"Found partitioned graphs: {cluster_base}-*.bin, loading...")
                route_client = ClusterSDKClient(ip=self.sdk_config['ip'], port=self.sdk_config['port'], comm_mode=self.comm_mode)
                route_client.init_routes()  # init_routes 内部会用临时连接获取路由，不需先 connect
                route_client.load_cluster_all(cluster_base)
                route_client.disconnect()
                self.loaded_partitioned_graphs = True
                print("Partitioned graphs loaded, skipping partition.")
                # 加载后更新所有节点的 ef_search（分区图可能用不同的 ef 构建）
                print(f"Setting ef_search={self.ef_search} on all worker nodes...")
                for i in range(len(self.node_ips)):
                    connect_ip = "127.0.0.1" if i == self.host_id else self.node_ips[i]
                    port = self.node_ports[i]
                    ef_client = ClusterSDKClient(ip=connect_ip, port=port, comm_mode=self.comm_mode)
                    if ef_client.connect():
                        ef_client.set_hnsw_params(ef_search=self.ef_search)
                        ef_client.disconnect()
            elif self.graph_path is not None and os.path.exists(self.graph_path) and not self.args.shard:
                # shard 模式需要每个节点独立建图，不能复用统一的 HNSW 图
                print(f"Loading HNSW graph via SDK from {self.graph_path}...")
                sdk_client = ClusterSDKClient(ip=self.sdk_config['ip'], port=self.sdk_config['port'], comm_mode=self.comm_mode)
                if not sdk_client.connect():
                    raise RuntimeError("Failed to connect SDK to local node for load_hnsw")
                try:
                    sdk_client.load_hnsw(self.graph_path)
                    # 图加载后通过专用接口覆盖运行时查询参数，避免依赖 load 时的历史参数。
                    sdk_client.set_hnsw_params(ef_search=self.ef_search)
                    if direct_full_graph:
                        self.loaded_full_graph_direct = True
                        print("node_num=1: full graph loaded directly; skipping partition.")
                finally:
                    sdk_client.disconnect()
            else:
                # 分布式模式：加载数据并构建
                print("Loading data for distributed mode...")
                data = read_vectors(self.data_path)
                print(f"Data shape: {data.shape}")

                max_num_elements = data.shape[0]
                num_elements = max_num_elements

                self.build_index(data=data, num_elements=num_elements)

            if not self.args.shard and not all_saved and not self.loaded_full_graph_direct:
                self.full_hnsw_release_needed = True
                if self.args.save and self.graph_path and not os.path.exists(self.graph_path):
                    sdk_client = ClusterSDKClient(ip=self.sdk_config['ip'], port=self.sdk_config['port'], comm_mode=self.comm_mode)
                    if not sdk_client.connect():
                        raise RuntimeError("Failed to connect SDK to local node for full HNSW save")
                    try:
                        print(f"Saving full HNSW to {self.graph_path} before partitioning...")
                        sdk_client.save_hnsw(self.graph_path)
                        print(f"Full HNSW saved: {os.path.abspath(self.graph_path)}")
                    finally:
                        sdk_client.disconnect()

                if self.args.build_only and not os.path.exists(self.graph_path):
                    print("[build_only] Built and saved full HNSW graph; skipping partition.")
                    return
                # graph existed before, proceed to partition below

                # 需要分区：发送 partition 命令到 worker
                partition_client = ClusterSDKClient(ip=self.sdk_config['ip'], port=self.sdk_config['port'], comm_mode=self.comm_mode)
                if not partition_client.connect(): raise RuntimeError("Failed to connect for partition")
                try:
                    print("Partitioning data across cluster nodes via SDK...")
                    start_time = time.time()
                    print(f"Partition distribution: {self.partition_distribution}")
                    if self.use_layer:
                        print(f"Partitioning in layer mode (replication={self.replication})...")
                        partition_client.partition(mode="layer", replication=self.replication,
                            layer_config=self.layer_config, layer_partition_mode=self.partition_mode,
                            layer_replicate_flags=self.layer_replicate_flags,
                            num_threads=self.num_construction_threads,
                            partition_distribution=self.partition_distribution,
                            partition_base=cluster_base)
                    else:
                        print(f"Partitioning in {self.partition_mode} mode (replication={self.replication})...")
                        partition_client.partition(mode=self.partition_mode, replication=self.replication,
                            num_threads=self.num_construction_threads,
                            partition_distribution=self.partition_distribution,
                            partition_base=cluster_base)
                    print(f"Partitioning Time: {time.time() - start_time:.4f} seconds.")
                finally:
                    partition_client.disconnect()

                if self.partition_distribution == "nfs-load":
                    print(f"Loading NFS-materialized partitioned graphs from {cluster_base}-*.bin...")
                    route_client = ClusterSDKClient(ip=self.sdk_config['ip'], port=self.sdk_config['port'], comm_mode=self.comm_mode)
                    route_client.init_routes()
                    route_client.load_cluster_all(cluster_base)
                    route_client.disconnect()
                    self.loaded_partitioned_graphs = True
                    print("Partitioned graphs loaded from NFS.")
                elif self.args.save:
                    print(f"Saving partitioned graphs to {cluster_base}-*.bin...")
                    save_client = ClusterSDKClient(ip=self.sdk_config['ip'], port=self.sdk_config['port'], comm_mode=self.comm_mode)
                    if not save_client.connect(): raise RuntimeError("Failed to connect for save")
                    save_client.init_routes()
                    try:
                        save_client.save_cluster_all(cluster_base)
                    finally:
                        save_client.disconnect()
                    print(f"Saved {self.node_num} node graphs.")
            return

        # 非分布式模式：传统的cluster和hnsw创建方式
        if self.cluster_path is not None and os.path.exists(self.cluster_path):
            print("Loading cluster from disk...")
            self.cluster = Cluster.load(self.cluster_path)
            self.cluster.reset()
            if self.args.is_global_pull:
                self.cluster.is_global_pull = True
        elif self.graph_path is not None and os.path.exists(self.graph_path) and not self.args.shard and not self.args.build_only:
            # shard 模式需要每个节点独立建图，不能复用统一的 HNSW 图
            print("Loading HNSW graph from disk...")
            self.hnsw = HNSWGraph.load(self.graph_path)
            self.hnsw.ef_search = self.ef_search
            print("Partitioning data across cluster nodes...")
            self.cluster = Cluster(self.hnsw, self.node_num, comm_mode=self.comm_mode, ib_dev=self.ib_dev, ib_port=self.ib_port, gid_index=self.gid_index, max_track_num=self.max_track_num)
            if self.args.is_global_pull:
                self.cluster.is_global_pull = True

            layer_sfx = "-layer" if self.use_layer else ""
            cluster_base = (f"./save/{self.data_name}/node{self.node_num}/"
                f"m{self.m}-efc{self.ef_construction}-ll{self.limited_level}-"
                f"{self.partition_mode}-rep{self._rep_fmt}{layer_sfx}{self._part_tag_sfx}")
            start_time = time.time()
            if self.use_layer:
                self.cluster.partition(mode="layer", replication=int(self.replication), layer_config=self.layer_config, layer_partition_mode=self.partition_mode, layer_replicate_flags=self.layer_replicate_flags, num_threads=self.num_construction_threads, partition_distribution="nfs-load", partition_base=cluster_base)
            else:
                self.cluster.partition(mode=self.partition_mode, replication=int(self.replication), layer_config=self.layer_config, num_threads=self.num_construction_threads, partition_distribution="nfs-load", partition_base=cluster_base)
            partition_time = time.time() - start_time
            print(f"Partitioning Time: {partition_time:.4f} seconds.")
        else:
            print("Loading data...")
            data = read_vectors(self.data_path)
            print(f"Data shape: {data.shape}")

            dim = data.shape[1]
            max_num_elements = data.shape[0]
            num_elements = max_num_elements

            self.hnsw = HNSWGraph(dim=dim, m=self.m, ef_construction=self.ef_construction, ef_search=self.ef_search, limited_level=self.limited_level, max_elements=num_elements)
            self.cluster = Cluster(self.hnsw, self.node_num, comm_mode=self.comm_mode, ib_dev=self.ib_dev, ib_port=self.ib_port, gid_index=self.gid_index, max_track_num=self.max_track_num)
            if self.args.is_global_pull:
                self.cluster.is_global_pull = True

            if self.args.shard:
                # Reduce模式：先创建cluster，轮询添加向量到各节点独立建图
                self.cluster.is_shard = True

                num_threads = self.num_construction_threads

                if num_threads > 1:
                    self._parallel_build(self.cluster.add_vector, data, num_elements,
                                         num_threads, "Building cluster in shard mode")

                    # 打印每个节点的向量数量
                    if not self.args.distributed:
                        print("Node vector counts:")
                        for i, node in enumerate(self.cluster.nodes):
                            count = node.local_hnsw.get_cur_element_count() if node.local_hnsw else 0
                            print(f"  Node {i}: {count} vectors")
                else:
                    print("Building cluster in shard mode...")
                    start_time = time.time()
                    for idx, vector in tqdm(enumerate(data[:num_elements]), total=num_elements, desc="Building", unit="vec"):
                        self.cluster.add_vector(idx, vector)  # 轮询添加到各节点
                    build_time = time.time() - start_time
                    print(f"Building Machine Time: {build_time:.4f} seconds.")

                    # 打印每个节点的向量数量
                    if not self.args.distributed:
                        print("Node vector counts:")
                        for i, node in enumerate(self.cluster.nodes):
                            count = node.local_hnsw.get_cur_element_count() if node.local_hnsw else 0
                            print(f"  Node {i}: {count} vectors")
            else:
                # 传统模式（非分布式）：先建统一图，再分区
                num_threads = self.num_construction_threads

                if num_threads > 1:
                    self._parallel_build(self.hnsw.add_vector, data, num_elements,
                                         num_threads, "Building HNSW index")

                    # 打印每个节点的向量数量
                    if not self.args.distributed:
                        print("Node vector counts:")
                        for i, node in enumerate(self.cluster.nodes):
                            count = node.local_hnsw.get_cur_element_count() if node.local_hnsw else 0
                            print(f"  Node {i}: {count} vectors")
                else:
                    print("Building HNSW index (single-threaded)...")
                    start_time = time.time()
                    for idx, vector in tqdm(enumerate(data[:num_elements]), total=num_elements, desc="Building", unit="vec"):
                        self.hnsw.add_vector(idx, vector)
                    build_time = time.time() - start_time
                    print(f"Building Machine Time: {build_time:.4f} seconds.")

            if self.args.build_only:
                print("[build_only] Built full HNSW graph; skipping partition.")
                return

            # 传统模式需要分区，shard模式不需要（因为每个节点已经有自己的图）
            if not self.args.shard:
                print("Partitioning data across cluster nodes...")
                start_time = time.time()
                if self.use_layer:
                    self.cluster.partition(mode="layer", replication=int(self.replication), layer_config=self.layer_config, layer_partition_mode=self.partition_mode, layer_replicate_flags=self.layer_replicate_flags, num_threads=self.num_construction_threads)
                else:
                    self.cluster.partition(mode=self.partition_mode, replication=int(self.replication), layer_config=self.layer_config, num_threads=self.num_construction_threads)
                partition_time = time.time() - start_time
                print(f"Partitioning Time: {partition_time:.4f} seconds.")

    def release_full_hnsw_before_query(self):
        if not self.args.distributed or self.args.shard or self.args.build_only:
            return
        if self.loaded_full_graph_direct or self.full_hnsw_released:
            return
        if not (self.full_hnsw_release_needed or self.loaded_partitioned_graphs):
            return

        print("Releasing full HNSW graphs on worker nodes before query...")
        route_client = ClusterSDKClient(ip=self.sdk_config['ip'], port=self.sdk_config['port'], comm_mode=self.comm_mode)
        if not route_client.init_routes():
            raise RuntimeError("Failed to initialize routes before releasing full HNSW graphs")
        try:
            route_client.release_full_hnsw_all()
        finally:
            route_client.disconnect()
        self.full_hnsw_released = True
        print("Full HNSW graphs released; workers retain only local partition graphs.")

    def save_graph(self):
        """保存已建好的 HNSW 图。分布式模式通过 SDK 在远端保存，单机模式本地保存。"""
        if not self.args.save:
            return
        save_dir = "./save"
        os.makedirs(save_dir, exist_ok=True)
        if self.args.distributed:
            if self.loaded_partitioned_graphs:
                print("Skipping full HNSW save: partitioned graphs were loaded directly.")
                return
            if self.loaded_full_graph_direct:
                print("Skipping full HNSW save: node_num=1 loaded the full graph directly.")
                return
            if not self.args.shard:
                print("Skipping full HNSW save: distributed partition/save already wrote partitioned graphs.")
                return

            base = (f"{save_dir}/hnsw-{self.data_name}-m{self.m}-"
                    f"efc{self.ef_construction}-ll{self.limited_level}")
            hnsw_path = f"{base}.bin"

            sdk_client = ClusterSDKClient(ip=self.sdk_config['ip'], port=self.sdk_config['port'], comm_mode=self.comm_mode)
            if not sdk_client.connect():
                raise RuntimeError("Failed to connect SDK to local node for save")
            try:
                print(f"Saving local HNSW to {hnsw_path}...")
                sdk_client.save_hnsw(hnsw_path)
                self.graph_path = hnsw_path
                print(f"HNSW saved: {os.path.abspath(hnsw_path)}")
            finally:
                sdk_client.disconnect()
        elif self.graph_path is None or self.args.build_only:
            if self.graph_path is None:
                self.graph_path = (f"{save_dir}/hnsw-{self.data_name}-m{self.m}-"
                                   f"efc{self.ef_construction}-ll{self.limited_level}.bin")
            print(f"Saving graph state to {self.graph_path}...")
            self.hnsw.save(self.graph_path)

    def orchestrate_distributed_query(self):
        """
        协调多台机器并行发 query。
        通过 SSH 在 query_nodes 列表（来自 config.json）的每台机器上启动 query.py，
        使用文件屏障保证同时开始，最后聚合每个 rank 的结果。
        """
        if not self.query_nodes:
            print("No query_nodes in config.json, skipping distributed query orchestration")
            return None

        world_size = len(self.query_nodes)
        # self.num_query_threads 是全局总线程数；每个 rank 平分
        per_rank_threads = max(1, self.num_query_threads // world_size)
        print(f"[orchestrate] Global query threads: {self.num_query_threads}, per-rank: {per_rank_threads}")

        # 创建 job-specific 屏障目录 + query 结果目录，避免并行 Slurm job 互相删除运行时文件
        barrier_dir = os.path.join(self.runtime_dir, "barrier")
        tmp_query_dir = os.path.join(self.runtime_dir, "query_results")
        import shutil
        if os.path.exists(barrier_dir):
            shutil.rmtree(barrier_dir)
        os.makedirs(barrier_dir, exist_ok=True)
        if os.path.exists(tmp_query_dir):
            shutil.rmtree(tmp_query_dir)
        os.makedirs(tmp_query_dir, exist_ok=True)

        # 启动 query.py：默认 SSH，--slurm 时用 srun --overlap
        use_slurm = bool(self.args.slurm)
        ssh_procs = []
        for rank, qnode in enumerate(self.query_nodes):
            host = qnode["ip"]
            srun_target = qnode.get("hostname") or host  # srun -w 优先用 hostname
            user = qnode.get("user_name", self.host_user_name)
            log_file = os.path.join(self.runtime_dir, f"query_rank{rank}.log")
            ready_timeout_value = 7200
            query_args = (
                f"--data_name {self.data_name} "
                f"--dtype {_active_dtype} "
                f"--k {self.k} "
                f"--ef_search {self.ef_search} "
                f"--num_query_threads {per_rank_threads} "
                f"--num_runs {self.args.num_runs} "
                f"--query_rank {rank} "
                f"--query_world_size {world_size} "
                f"--comm_mode {self.comm_mode} "
                f"--barrier_dir {barrier_dir} "
                f"--results_dir {tmp_query_dir} "
                f"--barrier_timeout {ready_timeout_value} "
                f"--query_path {self.query_path} "
                f"--groundtruth_path {self.groundtruth_path} "
            )

            if use_slurm:
                # srun --overlap：在 sbatch 已分配的节点上叠加 job step（被 SLURM 跟踪）
                srun_cmd = [
                    "srun", "-N1", "-n1", f"--cpus-per-task={per_rank_threads}",
                    "--overlap",
                    "--overcommit",
                    "--export=ALL",
                    "-w", srun_target,
                    "-o", log_file,
                    "-e", log_file,
                    "bash", "-lc",
                    f"export VR_CONFIG_PATH={shlex.quote(_CONFIG_PATH)} && "
                    f"cd {project_root} && eval \"$(conda shell.bash hook)\" && conda activate vector-retrieve && "
                    f"python -u query.py {query_args}"
                ]
                print(f"[orchestrate] srun launch query.py on rank {rank} ({host})")
                proc = subprocess.Popen(srun_cmd,
                                        stdin=subprocess.DEVNULL,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE)
            else:
                cmd_str = (
                    f"export VR_CONFIG_PATH={shlex.quote(_CONFIG_PATH)} && "
                    f"cd {project_root} && conda activate vector-retrieve && "
                    f"python -u query.py {query_args} > {log_file} 2>&1"
                )
                ssh_cmd = ["ssh", "-o", "BatchMode=yes",
                           f"{user}@{host}", cmd_str]
                print(f"[orchestrate] ssh launch query.py on rank {rank} ({host})")
                proc = subprocess.Popen(ssh_cmd,
                                        stdin=subprocess.DEVNULL,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE)
            ssh_procs.append((rank, host, proc, log_file))

        def _sample_worker_rss(stage):
            if not use_slurm:
                return []
            print(f"[rss] {stage}: sampling worker node_start.py RSS")
            samples = []
            for wid, ip in enumerate(self.node_ips):
                host = self.node_hostnames[wid] if wid < len(self.node_hostnames) else ip
                cmd = [
                    "srun", "-N1", "-n1", "--cpus-per-task=1",
                    "--overlap", "--overcommit", "--export=ALL",
                    "-w", host,
                    "bash", "-lc",
                    "ps -C python -o pid=,rss=,args= | awk '/node_start.py/ {print $1, $2}'"
                ]
                try:
                    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=10).strip()
                except Exception as e:
                    print(f"[rss] {stage}: worker {wid} ({host}) sample failed: {e}")
                    samples.append({"stage": stage, "worker_id": wid, "host": host, "ip": ip, "error": str(e)})
                    continue
                if not out:
                    print(f"[rss] {stage}: worker {wid} ({host}) node_start.py not found")
                    samples.append({"stage": stage, "worker_id": wid, "host": host, "ip": ip, "error": "node_start.py not found"})
                    continue
                values = []
                for line in out.splitlines():
                    parts = line.split()
                    if len(parts) >= 2:
                        values.append((parts[0], int(parts[1])))
                if values:
                    rss_values = [rss for _, rss in values]
                    max_rss = max(rss_values)
                    rss_gib = max_rss / 1024 / 1024
                    pids = ",".join(pid for pid, _ in values)
                    sample = {
                        "stage": stage,
                        "worker_id": wid,
                        "host": host,
                        "ip": ip,
                        "pids": [pid for pid, _ in values],
                        "rss_kb_values": rss_values,
                        "max_rss_kb": max_rss,
                        "max_rss_gib": rss_gib,
                    }
                    samples.append(sample)
                    print(f"[rss] {stage}: worker {wid} ({host}) pids={pids} max_rss={max_rss}K ({rss_gib:.2f} GiB)")
            self.query_worker_rss_samples.extend(samples)
            return samples

        # 多轮文件屏障协议：
        #   1. 等所有 rank 写 ready_<rank>_<run>
        #   2. 写 go_<run>（含 3s 后的 scheduled_start 时间戳）
        #   3. 各 rank 同时读到 go 文件，等待到 scheduled_start 之后开始并发查询
        # 进程死亡或超时写 abort 文件通知其他 rank 终止。
        num_runs = self.args.num_runs
        first_round_scheduled_start = None
        last_round_scheduled_start = None
        for run_idx in range(num_runs):
            print(f"[orchestrate] Round {run_idx+1}/{num_runs}: waiting for {world_size} ready files...")
            ready_timeout = 7200
            wait_start = time.time()
            while True:
                ready_count = sum(1 for r in range(world_size)
                                  if os.path.exists(os.path.join(barrier_dir, f"ready_{r}_{run_idx}")))
                if ready_count >= world_size:
                    print(f"[orchestrate]   round {run_idx+1}: all ready after {time.time()-wait_start:.2f}s")
                    break
                # 检查 SSH/srun 进程是否死亡
                for rank, host, proc, log_file in ssh_procs:
                    if proc.poll() is not None and proc.returncode != 0:
                        try:
                            stderr = proc.stderr.read().decode(errors='replace') if proc.stderr else ""
                        except Exception:
                            stderr = ""
                        print(f"[orchestrate] ERROR: query rank {rank} on {host} died (rc={proc.returncode}). stderr: {stderr[:500]}")
                        with open(os.path.join(barrier_dir, "abort"), "w") as f:
                            f.write("aborted")
                        raise QueryStallError(
                            f"query rank {rank} on {host} died at round {run_idx+1} "
                            f"(rc={proc.returncode}); stderr: {stderr[:200]}"
                        )
                if time.time() - wait_start > ready_timeout:
                    print(f"[orchestrate] TIMEOUT round {run_idx+1}: only {ready_count}/{world_size} ready after {ready_timeout}s")
                    with open(os.path.join(barrier_dir, "abort"), "w") as f:
                        f.write("timeout")
                    raise QueryStallError(
                        f"ready barrier timeout at round {run_idx+1}: "
                        f"only {ready_count}/{world_size} ready after {ready_timeout}s"
                    )
                time.sleep(0.1)

            # 写 go_<run_idx> 文件，3s 后让所有 rank 同步开始
            go_file = os.path.join(barrier_dir, f"go_{run_idx}")
            scheduled_start = time.time() + 3.0
            if run_idx == 0:
                first_round_scheduled_start = scheduled_start
            last_round_scheduled_start = scheduled_start
            with open(go_file + ".tmp", "w") as f:
                f.write(f"{scheduled_start:.6f}\n")
            os.replace(go_file + ".tmp", go_file)
            print(f"[orchestrate]   round {run_idx+1}: GO sent, scheduled start at {time.strftime('%H:%M:%S', time.localtime(scheduled_start))}")
            time.sleep(max(0.0, scheduled_start + 0.1 - time.time()))
            _sample_worker_rss(f"round {run_idx+1} query")

        # 等所有 query.py 完成；记录每个 rank 的完成时刻
        print(f"[orchestrate] Waiting for all query.py instances to finish...")
        rank_finish_times = []
        failed_ranks = []
        for rank, host, proc, log_file in ssh_procs:
            try:
                stdout, stderr = proc.communicate()
                finish_t = time.time()
                rank_finish_times.append(finish_t)
                rc = proc.returncode
                tail_log = ""
                try:
                    with open(log_file, "r") as f:
                        lines = f.readlines()
                        tail_log = "".join(lines[-15:])
                except Exception:
                    pass
                print(f"[orchestrate] rank {rank} ({host}) finished at +{finish_t - first_round_scheduled_start:.2f}s, rc={rc}")
                if tail_log:
                    print(f"  log tail:\n{tail_log}")
                if rc != 0:
                    failed_ranks.append((rank, host, rc))
            except Exception as e:
                print(f"[orchestrate] rank {rank} ({host}): error waiting: {e}")
                failed_ranks.append((rank, host, -1))
        if failed_ranks:
            detail = ", ".join(f"rank{r}@{h}(rc={c})" for r, h, c in failed_ranks)
            raise QueryStallError(f"query rank(s) exited non-zero: {detail}")

        # main.py 视角的 wall-clock：从第一轮 scheduled_start 到最后一个 rank 完成
        # （含 num_runs 轮 + 各轮间的同步等待）
        wallclock_total_time = (max(rank_finish_times) - first_round_scheduled_start) if rank_finish_times and first_round_scheduled_start else 0.0

        # 聚合每个 rank 的结果（从 tmp 临时目录读取，聚合完后删除临时目录）
        print(f"[orchestrate] Aggregating results from {world_size} ranks...")
        tmp_dir = tmp_query_dir

        # 合并 per-query 字典
        recall = {}
        remote_transfer_cnt = {}
        remote_transfer_data = {}
        distance_cnt = {}
        pull_cnt = {}
        pull_data = {}
        backtrack_cnt = {}
        timing_stats = {field: {} for field in TIMING_FIELDS}
        per_rank_qps = []
        per_rank_qps_runs_all = []
        per_rank_query_time = []
        per_rank_assigned = []
        per_rank_query_times_runs = []  # 每 rank 每轮的 query_time

        for rank in range(world_size):
            rpath = os.path.join(tmp_dir, f"rank{rank}-of{world_size}.json")
            if not os.path.exists(rpath):
                print(f"  rank {rank}: NO RESULTS FILE FOUND ({rpath})")
                continue
            with open(rpath) as f:
                rj = json.load(f)
            per_rank_qps.append(rj.get("qps", 0.0))
            per_rank_assigned.append(rj.get("num_queries_assigned", 0))
            per_rank_qps_runs_all.append(rj.get("qps_runs", []))
            per_rank_query_time.append(rj.get("last_query_time", 0.0))
            per_rank_query_times_runs.append(rj.get("query_times_runs", []))
            # 合并字典（key 已是字符串，转回 int 以统一）
            for k_str, v in rj.get("recall", {}).items():
                recall[int(k_str)] = v
            for k_str, v in rj.get("remote_transfer_cnt", {}).items():
                remote_transfer_cnt[int(k_str)] = v
            for k_str, v in rj.get("remote_transfer_data", {}).items():
                remote_transfer_data[int(k_str)] = v
            for k_str, v in rj.get("distance_cnt", {}).items():
                distance_cnt[int(k_str)] = v
            for k_str, v in rj.get("backtrack_cnt", {}).items():
                backtrack_cnt[int(k_str)] = v
            for k_str, v in rj.get("pull_cnt", {}).items():
                pull_cnt[int(k_str)] = v
            for k_str, v in rj.get("pull_data", {}).items():
                pull_data[int(k_str)] = v
            for field in TIMING_FIELDS:
                for k_str, v in rj.get(field, {}).items():
                    timing_stats[field][int(k_str)] = v
            print(f"  rank {rank}: QPS={rj.get('qps', 0):.2f}, queries={rj.get('num_queries_assigned', 0)}")

        # num_queries 为聚合后的总数（所有 rank 的并集）
        num_queries = len(recall)
        if num_queries == 0:
            print("[orchestrate] ERROR: no query results aggregated")
            raise QueryStallError("no query results aggregated from any rank")

        avg_recall = sum(recall.values()) / num_queries
        all_remote_transfer_cnt = sum(remote_transfer_cnt.values())
        avg_remote_transfer_cnt = all_remote_transfer_cnt / num_queries
        avg_remote_transfer_data_per_transfer = sum(remote_transfer_data.values()) / all_remote_transfer_cnt if all_remote_transfer_cnt > 0 else 0.0
        avg_remote_transfer_data_per_query = sum(remote_transfer_data.values()) / num_queries
        all_distance_cnt = sum(distance_cnt.values())
        avg_distance_cnt = all_distance_cnt / num_queries

        all_backtrack_cnt = sum(backtrack_cnt.values()) if backtrack_cnt else 0
        avg_backtrack_cnt = all_backtrack_cnt / num_queries if num_queries else 0.0
        backtrack_ratio = all_backtrack_cnt / all_remote_transfer_cnt if all_remote_transfer_cnt > 0 else 0.0

        all_pull_cnt = sum(pull_cnt.values()) if pull_cnt else 0
        avg_pull_cnt = all_pull_cnt / num_queries if num_queries else 0.0
        all_pull_data = sum(pull_data.values()) if pull_data else 0
        avg_pull_data_per_pull = all_pull_data / all_pull_cnt if all_pull_cnt > 0 else 0.0
        avg_pull_data_per_query = all_pull_data / num_queries if num_queries else 0.0

        avg_timing_stats = {
            f"avg_{field}": (sum(values.values()) / len(values) if values else 0.0)
            for field, values in timing_stats.items()
        }

        # 总 QPS 计算：
        #   每轮 wall-clock = max(各 rank query_time)（文件屏障保证同步起始，最慢 rank 决定该轮耗时）
        #   每轮总查询数 = Σ(per_rank_assigned) × per_rank_threads（所有 rank 线程并发处理不同 query）
        #   总 wall-clock = Σ(每轮 wall-clock)，不包含轮间同步等待时间
        max_runs = max((len(r) for r in per_rank_query_times_runs), default=0)
        per_run_wallclock = []
        per_run_total_queries_count = sum(per_rank_assigned) * per_rank_threads
        qps_runs = []
        for run_i in range(max_runs):
            run_max_t = max(
                (times[run_i] for times in per_rank_query_times_runs if run_i < len(times)),
                default=0.0
            )
            per_run_wallclock.append(run_max_t)
            qps_runs.append(per_run_total_queries_count / run_max_t if run_max_t > 0 else 0.0)

        # 最终 QPS = 各轮平均（去除 MAD 离群值，至少保留 1 轮；若全离群则取所有轮均值）
        def _trim_outliers(runs):
            """基于 MAD (Median Absolute Deviation) 去除离群值。
            modified Z-score = 0.6745 * |x - median| / MAD，> 3.5 视为离群。
            若 MAD=0 或样本数 < 3，不去除。"""
            if len(runs) < 3:
                return list(runs)
            sorted_r = sorted(runs)
            n = len(sorted_r)
            median = sorted_r[n // 2] if n % 2 == 1 else (sorted_r[n // 2 - 1] + sorted_r[n // 2]) / 2
            abs_dev = sorted([abs(x - median) for x in runs])
            mad = abs_dev[n // 2] if n % 2 == 1 else (abs_dev[n // 2 - 1] + abs_dev[n // 2]) / 2
            if mad == 0:
                return list(runs)
            kept = [x for x in runs if 0.6745 * abs(x - median) / mad <= 3.5]
            return kept if len(kept) >= 1 else list(runs)

        if len(qps_runs) >= 3:
            kept_runs = _trim_outliers(qps_runs)
            total_qps = sum(kept_runs) / len(kept_runs)
            trim_note = f", outliers removed: {len(qps_runs) - len(kept_runs)} of {len(qps_runs)}" if len(kept_runs) != len(qps_runs) else ""
        elif qps_runs:
            total_qps = sum(qps_runs) / len(qps_runs)
            trim_note = ""
        else:
            total_qps = 0.0
            trim_note = ""
        total_query_time = sum(per_run_wallclock)

        print("-" * 30)
        print("Per-round breakdown (wallclock = max query_time across ranks):")
        for i, (wc, q) in enumerate(zip(per_run_wallclock, qps_runs)):
            print(f"  Round {i+1}: wallclock={wc:.4f}s, QPS={q:.2f}")
        print("-" * 30)
        print("Evaluation Results:")
        print(f"Recall@{self.k}: {avg_recall:.4f}")
        print(f"QPS: {total_qps:.4f} (mean of {len(qps_runs)} rounds{trim_note})")
        print(f"Query Time Total: {total_query_time:.4f} seconds (sum of per-round wallclock)")
        print(f"Average Remote Transfer Count per Query: {avg_remote_transfer_cnt:.4f}")
        print(f"Average Remote Transfer Data per Transfer: {avg_remote_transfer_data_per_transfer:.0f} bits ({avg_remote_transfer_data_per_transfer/8:.0f} bytes)")
        print(f"Average Remote Transfer Data per Query: {avg_remote_transfer_data_per_query:.0f} bits ({avg_remote_transfer_data_per_query/8:.0f} bytes)")
        print(f"Average Distance Computations per Query: {avg_distance_cnt:.4f}")
        print(f"Average Backtrack Count per Query: {avg_backtrack_cnt:.4f} (window={self.max_track_num}, ratio={backtrack_ratio:.4f})")
        for field in TIMING_FIELDS:
            print(f"Average {field} per Query: {avg_timing_stats[f'avg_{field}']:.4f} ms")
        if self.args.is_global_pull:
            print(f"Average Pull Count per Query: {avg_pull_cnt:.4f}")
            print(f"Average Pull Data per Pull: {avg_pull_data_per_pull:.0f} bits ({avg_pull_data_per_pull/8:.0f} bytes)")
            print(f"Average Pull Data per Query: {avg_pull_data_per_query:.0f} bits ({avg_pull_data_per_query/8:.0f} bytes)")
        print("-" * 30)

        # ----- 通过 SDK 获取 hnsw / cluster 参数（与原 save_results 保持一致） -----
        sdk_idx = max(0, self.host_id)
        info_ip = '127.0.0.1' if self.host_is_node else self.node_ips[0]
        info_port = self.node_ports[sdk_idx]
        info_client = ClusterSDKClient(ip=info_ip, port=info_port, comm_mode=self.comm_mode)
        if not info_client.connect():
            raise RuntimeError("Failed to connect SDK for get_information")
        try:
            info = info_client.get_information()
        finally:
            info_client.disconnect()

        hnsw_info = info.get("hnsw") if isinstance(info, dict) else None
        cluster_info = info.get("cluster") if isinstance(info, dict) else None

        hnsw_params = {
            "data_name": self.data_name,
            "dim": hnsw_info.get("dim") if hnsw_info else None,
            "m": hnsw_info.get("m") if hnsw_info else self.m,
            "ef_construction": hnsw_info.get("ef_construction") if hnsw_info else self.ef_construction,
            "ef_search": hnsw_info.get("ef_search") if hnsw_info else self.ef_search,
            "limited_level": hnsw_info.get("limited_level") if hnsw_info else self.limited_level,
            "max_level": hnsw_info.get("max_level") if hnsw_info else self.limited_level,
        }
        info_partition_mode = cluster_info.get("partition_mode") if cluster_info else None
        info_layer_partition_mode = cluster_info.get("layer_partition_mode") if cluster_info else None
        info_replication_level = cluster_info.get("replication_level") if cluster_info else None
        info_replication_num = cluster_info.get("replication_num") if cluster_info else None

        effective_partition_mode = info_partition_mode or ("layer" if self.use_layer else self.partition_mode)
        effective_layer_partition_mode = info_layer_partition_mode or (self.partition_mode if effective_partition_mode == "layer" else "")
        import math
        effective_replication_level = info_replication_level if info_replication_level not in (None, 0) else int(math.ceil(self.replication))
        effective_replication_num = info_replication_num if info_replication_num is not None else 0
        effective_layer_config = self.layer_config
        effective_layer_replicate_flags = self.layer_replicate_flags
        if cluster_info:
            info_layer_config = cluster_info.get("layer_config")
            if info_layer_config:
                if isinstance(info_layer_config, dict):
                    effective_layer_config = {int(k): list(v) for k, v in info_layer_config.items()}
                else:
                    effective_layer_config = info_layer_config
            info_layer_replicate_flags = cluster_info.get("layer_replicate_flags")
            if info_layer_replicate_flags:
                effective_layer_replicate_flags = info_layer_replicate_flags

        cluster_params = {
            "node_num": cluster_info.get("node_num") if cluster_info else self.node_num,
            "partition_mode": effective_partition_mode,
            "replication_level": effective_replication_level,
            "replication_num": effective_replication_num,
            "requested_replication": self.replication,
            "bandwidth": cluster_info.get("bandwidth") if cluster_info else 0,
            "is_distributed": cluster_info.get("is_distributed") if cluster_info else True,
            "is_shard": cluster_info.get("is_shard") if cluster_info else self.args.shard,
            "is_global_pull": cluster_info.get("is_global_pull") if cluster_info else self.args.is_global_pull,
            "max_track_num": self.max_track_num,
        }
        if cluster_params["partition_mode"] == "layer":
            cluster_params["layer_partition_mode"] = effective_layer_partition_mode
            cluster_params["layer_config"] = effective_layer_config
            cluster_params["layer_replicate_flags"] = effective_layer_replicate_flags

        # self.num_query_threads 已经是全局总数
        total_query_threads = self.num_query_threads

        valid_rss_samples = [s for s in self.query_worker_rss_samples if "max_rss_kb" in s]
        avg_query_worker_rss_gib = (
            sum(s["max_rss_gib"] for s in valid_rss_samples) / len(valid_rss_samples)
            if valid_rss_samples else 0.0
        )

        stats = {
            "hnsw_params": hnsw_params,
            "cluster_params": cluster_params,
            "num_queries": num_queries,
            "num_query_threads": total_query_threads,
            "num_threads": self.num_threads,
            "k": self.k,
            "qps": total_qps,
            "qps_runs": qps_runs if qps_runs else [total_qps],
            "query_times_runs": per_run_wallclock,
            "total_query_time": total_query_time,
            f"avg_recall@{self.k}": avg_recall,
            "avg_remote_transfer_cnt": avg_remote_transfer_cnt,
            "avg_remote_transfer_data_per_transfer": avg_remote_transfer_data_per_transfer,
            "avg_remote_transfer_data_per_query": avg_remote_transfer_data_per_query,
            "avg_distance_cnt": avg_distance_cnt,
            "avg_backtrack_cnt": avg_backtrack_cnt,
            "backtrack_ratio": backtrack_ratio,
            "avg_query_worker_rss_gib": avg_query_worker_rss_gib,
            "recall": recall,
            "remote_transfer_cnt": remote_transfer_cnt,
            "remote_transfer_data": remote_transfer_data,
            "distance_cnt": distance_cnt,
            "backtrack_cnt": backtrack_cnt,
        }
        stats.update(avg_timing_stats)
        for field in TIMING_FIELDS:
            stats[field] = timing_stats[field]
        if self.args.is_global_pull:
            stats["avg_pull_cnt"] = avg_pull_cnt
            stats["avg_pull_data_per_pull"] = avg_pull_data_per_pull
            stats["avg_pull_data_per_query"] = avg_pull_data_per_query
            stats["pull_cnt"] = pull_cnt
            stats["pull_data"] = pull_data

        # 与原 save_results 文件名格式完全一致
        results_dir = "./results/cpp" if self.use_cpp else "./results/python"
        os.makedirs(results_dir, exist_ok=True)
        is_node1 = self.node_num == 1
        distributed_suffix = "-distributed" if self.args.distributed and not is_node1 else ""
        cpu_suffix = f"-cpus{self.args.num_cpu}" if self.args.distributed and self.args.num_cpu != 0 else ""
        shard_suffix = "-shard" if self.args.shard else ""
        global_pull_suffix = "-global_pull" if self.args.is_global_pull else ""
        layer_suffix = "-layer" if self.use_layer and not is_node1 else ""
        # 文件名里的 query_threads 用全局总数（各 rank 之和）
        qt_suffix = f"-query_threads{total_query_threads}"

        if self.args.results_path:
            results_path = self.args.results_path
        elif self.args.shard:
            results_path = (f"{results_dir}/results-{self.data_name}-m{self.m}-"
                            f"efc{self.ef_construction}-efs{self.ef_search}-ll{self.limited_level}-"
                            f"node{self.node_num}-"
                            f"threads{self.num_threads}{qt_suffix}{distributed_suffix}{cpu_suffix}{shard_suffix}.json")
        elif self.args.is_global_pull:
            results_path = (f"{results_dir}/results-{self.data_name}-m{self.m}-"
                            f"efc{self.ef_construction}-efs{self.ef_search}-ll{self.limited_level}-"
                            f"node{self.node_num}-{self.partition_mode}-rep{self._rep_fmt}-"
                            f"threads{self.num_threads}{qt_suffix}{distributed_suffix}{cpu_suffix}{global_pull_suffix}.json")
        else:
            if is_node1:
                results_path = (f"{results_dir}/results-{self.data_name}-m{self.m}-"
                                f"efc{self.ef_construction}-efs{self.ef_search}-ll{self.limited_level}-"
                                f"node{self.node_num}-"
                                f"threads{self.num_threads}{qt_suffix}{cpu_suffix}.json")
            else:
                results_path = (f"{results_dir}/results-{self.data_name}-m{self.m}-"
                                f"efc{self.ef_construction}-efs{self.ef_search}-ll{self.limited_level}-"
                                f"node{self.node_num}-{self.partition_mode}-rep{self._rep_fmt}{self._part_tag_sfx}-"
                                f"threads{self.num_threads}{qt_suffix}{distributed_suffix}{cpu_suffix}{layer_suffix}.json")

        with open(results_path, 'w') as f:
            json.dump(stats, f, indent=4)

        print("\n" + "=" * 50)
        print(f"Results: {os.path.abspath(results_path)}")
        if self.args.distributed:
            print("Distributed mode: Graph and Cluster managed by worker nodes")
        print("=" * 50 + "\n")

        # 删除临时文件
        try:
            import shutil
            shutil.rmtree(tmp_dir)
            print(f"[orchestrate] Removed tmp dir: {tmp_dir}")
        except Exception as e:
            print(f"[orchestrate] Warning: failed to remove tmp dir: {e}")

        return results_path

    def cleanup_workers(self, force=False):
        """清理所有节点上的 worker 进程。两阶段：本地进程句柄 → 本地 PID → 远程 SSH。"""
        if self.args.no_cleanup:
            print("Skipping worker cleanup (--no_cleanup)")
            return
        # 终止查询 worker 进程（避免成为孤儿进程继续尝试连接已关闭的节点）
        for proc in getattr(self, 'query_worker_processes', []):
            try:
                if proc.is_alive():
                    proc.terminate()
            except Exception:
                pass
        for proc in getattr(self, 'query_worker_processes', []):
            try:
                proc.join(timeout=2)
                if proc.is_alive():
                    proc.kill()
            except Exception:
                pass
        self.query_worker_processes = []

        if self.args.distributed and self.worker_processes:
            print("\nCleaning up worker processes...")
            # 1. 先 terminate 本地管理的进程句柄（本地 worker + SSH 进程）
            for proc in self.worker_processes:
                if proc.poll() is None:
                    proc.terminate()
            time.sleep(1)
            for proc in self.worker_processes:
                if proc.poll() is None:
                    proc.kill()

        # 2. 清理本地 node_start.py 进程
            try:
                p = subprocess.run(["pgrep", "-f", "node_start.py"], capture_output=True, text=True)
                if p.returncode == 0 and p.stdout.strip():
                    pids = p.stdout.strip().split()
                    print(f"  Killing local node_start.py PIDs: {pids}")
                    subprocess.run(["kill", "-TERM"] + pids, check=False)
                    time.sleep(1)
                    p2 = subprocess.run(["pgrep", "-f", "node_start.py"], capture_output=True, text=True)
                    if p2.returncode == 0 and p2.stdout.strip():
                        subprocess.run(["kill", "-KILL"] + p2.stdout.strip().split(), check=False)
            except Exception as e:
                print(f"  Local cleanup error: {e}")

            # 3. 清理远程节点上的 node_start.py 进程
            cleanup_procs = []
            for i, ip in enumerate(self.node_ips):
                if ip == self.host_ip:
                    continue
                remote_cmd = (
                    f"pids=$(pgrep -f node_start.py || true); "
                    f"if [ -n \"$pids\" ]; then kill -TERM $pids 2>/dev/null || true; sleep 1; "
                    f"pids2=$(pgrep -f node_start.py || true); if [ -n \"$pids2\" ]; then kill -KILL $pids2 2>/dev/null || true; fi; fi"
                )
                cmd = [
                    "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30",
                    f"{self.node_user_names[i]}@{ip}", remote_cmd
                ]
                proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                cleanup_procs.append((proc, ip))
            for proc, ip in cleanup_procs:
                try:
                    proc.communicate(timeout=120)
                except subprocess.TimeoutExpired:
                    proc.kill()

            # 等待端口完全释放，避免下一个任务启动时端口仍被占用
            print("Waiting for ports to be released after cleanup...")
            time.sleep(5)
            print("Worker processes cleaned up.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="HNSW Vector Retrieval")
    parser.add_argument("--data_name", type=str, default="sift",
                        help=f"Name of the dataset. Presets available: {list(DATASET_PRESETS.keys())}")
    parser.add_argument("--dtype", type=str, default=None,
                        choices=list(_DTYPE_CLASS_MAP.keys()),
                        help="Vector element type (auto-resolved from --data_name if omitted)")
    parser.add_argument("--data_path", type=str, default=None, help="Path to base vectors file (auto-resolved from --data_name if omitted)")
    parser.add_argument("--query_path", type=str, default=None, help="Path to query vectors file (auto-resolved from --data_name if omitted)")
    parser.add_argument("--groundtruth_path", type=str, default=None, help="Path to groundtruth file (auto-resolved from --data_name if omitted)")
    parser.add_argument("--graph_path", type=str, default=None, help="Path to save/load the graph")
    parser.add_argument("--cluster_path", type=str, default=None, help="Path to save/load the cluster")
    parser.add_argument("--results_path", type=str, default=None, help="Explicit path for results JSON (overrides auto-generated filename)")
    parser.add_argument("--k", type=int, default=10, help="Number of nearest neighbors to retrieve")
    parser.add_argument("--m", type=int, default=10, help="Max number of connections per element")
    parser.add_argument("--ef_construction", type=int, default=200, help="Size of the dynamic candidate list for construction")
    parser.add_argument("--ef_search", type=int, default=15, help="Size of the dynamic candidate list for search")
    parser.add_argument("--limited_level", type=int, default=5, help="Level limit for HNSW")
    parser.add_argument("--dim", type=int, default=None, help="Vector dimension; defaults to the dataset preset")
    parser.add_argument("--node_num", type=int, default=6, help="Number of cluster nodes")
    parser.add_argument("--partition_mode", type=str, default="k-means", help="Data partition mode across nodes")
    parser.add_argument("--partition_distribution", type=str, default="nfs-load", choices=["nfs-load", "network-send"], help="How fresh distributed partitions are materialized: write files to shared storage then load them, or send graph payloads over the network")
    parser.add_argument("--layer", action='store_true', default=False, help="Enable layer-based partitioning")
    parser.add_argument("--replication", type=float, default=2.0, help="Replication level for data replication across nodes (supports fractional values like 0.4, 1.5)")
    parser.add_argument("--distributed", default=False, action='store_true', help="Enable distributed mode")
    parser.add_argument("--shard", default=False, action='store_true', help="Enable shard mode (independent graphs per node)")
    parser.add_argument("--global-pull", dest="is_global_pull", default=False, action='store_true', help="Enable global-pull mode: every node holds the full graph skeleton, only vector data is partitioned; remote vectors are pulled on demand during search")
    parser.add_argument("--num_construction_threads", type=int, default=64, help="Number of threads for parallel graph construction (0=auto detect)")
    parser.add_argument("--num_query_nodes", type=int, default=0, help="Number of query nodes to use (0=use all from config.json)")
    parser.add_argument("--num_query_threads", type=int, default=40, help="Number of client query processes (each process does all queries)")
    parser.add_argument("--num_threads", type=int, default=8, help="Number of server-side search threads per node (msg_pool_size)")
    parser.add_argument("--num_cpu", type=int, default=0, help="Limit CPU cores per node process (0=unlimited)")
    parser.add_argument("--save", action='store_true', default=False, help="Save the built graph to disk after construction")
    # 通信模式和 RDMA 参数
    parser.add_argument("--comm_mode", type=str, default="tcp", choices=["tcp", "rdma"], help="Communication mode: tcp or rdma")
    parser.add_argument("--ib_dev", type=str, default="", help="RDMA IB device name (auto-detect if empty)")
    parser.add_argument("--ib_port", type=int, default=1, help="RDMA IB port number")
    parser.add_argument("--gid_index", type=int, default=-1, help="RDMA GID index (-1 for auto-detect)")
    parser.add_argument("--max_track_num", type=int, default=10, help="Sliding window size for backtrack detection (transfer 回折返检测的目标节点窗口，0 关闭)")
    parser.add_argument("--num_runs", type=int, default=1, help="Number of query runs (>=3: trim max/min then average QPS)")
    parser.add_argument("--skip_prepare", action='store_true', default=False, help="Skip graph loading and partitioning (assume servers are already prepared)")
    parser.add_argument("--no_cleanup", action='store_true', default=False, help="Do not cleanup worker processes on exit")
    parser.add_argument("--skip_queries", action='store_true', default=False, help="Skip the query phase (only build/partition/save). Use when queries run on separate machines via query.py.")
    parser.add_argument("--build_only", action='store_true', default=False, help="Build and save the full HNSW graph only; skip partition and queries.")
    parser.add_argument("--slurm", action='store_true', default=False, help="Use srun (SLURM) instead of ssh to launch query.py on each query node. Default: ssh.")

    args = parser.parse_args(rest)

    # Auto-resolve dataset paths from preset when not explicitly provided
    if args.data_name in DATASET_PRESETS:
        preset = DATASET_PRESETS[args.data_name]
        if args.data_path is None:
            args.data_path = preset["data_path"]
        if args.query_path is None:
            args.query_path = preset["query_path"]
        if args.groundtruth_path is None:
            args.groundtruth_path = preset["groundtruth_path"]
        if args.dim is None:
            args.dim = preset.get("dim")
    else:
        missing = [k for k in ("data_path", "query_path", "groundtruth_path") if getattr(args, k) is None]
        if missing:
            parser.error(f"--data_name '{args.data_name}' is not a preset ({list(DATASET_PRESETS.keys())}). "
                         f"Please provide: {missing}")

    if args.shard and args.is_global_pull:
        parser.error("--shard and --global-pull are mutually exclusive")

    runner = VectorRetrieveRunner(args, use_cpp)
    try:
        runner.setup_distributed_nodes()
        runner.prepare()
        runner.save_graph()  # 建图完成后立即保存

        num_runs = args.num_runs

        # skip_prepare 模式下需要动态设置所有节点的 ef_search
        if args.skip_prepare:
            if runner.args.distributed:
                print(f"Setting ef_search={runner.ef_search} on all nodes...")
                for i, ip in enumerate(runner.node_ips):
                    port = runner.node_ports[i]
                    connect_ip = "127.0.0.1" if i == runner.host_id else ip
                    sdk_client = ClusterSDKClient(ip=connect_ip, port=port, comm_mode=runner.comm_mode)
                    if not sdk_client.connect():
                        raise RuntimeError(f"Failed to connect to node {i} ({ip}:{port}) for set_hnsw_params")
                    try:
                        sdk_client.set_hnsw_params(ef_search=runner.ef_search)
                        # 验证
                        sdk_verify = ClusterSDKClient(ip=connect_ip, port=port, comm_mode=runner.comm_mode)
                        if sdk_verify.connect():
                            info = sdk_verify.get_information()
                            sdk_verify.disconnect()
                            actual_efs = info.get("hnsw", {}).get("ef_search") if isinstance(info, dict) else None
                            if actual_efs != runner.ef_search:
                                raise RuntimeError(f"Node {i}: ef_search verification failed (expected {runner.ef_search}, got {actual_efs})")
                            print(f"  Node {i} ({ip}): ef_search={actual_efs} ✓")
                    finally:
                        sdk_client.disconnect()
            elif runner.hnsw is not None:
                runner.hnsw.ef_search = runner.ef_search

        runner.release_full_hnsw_before_query()

        # 查询前验证所有节点的 ef_search
        if runner.args.distributed:
            print(f"[Verify] Expected ef_search={runner.ef_search} on all nodes:")
            for i, ip in enumerate(runner.node_ips):
                port = runner.node_ports[i]
                connect_ip = "127.0.0.1" if i == runner.host_id else ip
                try:
                    sdk_v = ClusterSDKClient(ip=connect_ip, port=port, comm_mode=runner.comm_mode)
                    if sdk_v.connect():
                        info = sdk_v.get_information()
                        sdk_v.disconnect()
                        actual = info.get("hnsw", {}).get("ef_search") if isinstance(info, dict) else "N/A"
                        status = "✓" if actual == runner.ef_search else f"✗ MISMATCH!"
                        print(f"  Node {i} ({ip}): ef_search={actual} {status}")
                    else:
                        print(f"  Node {i} ({ip}): connect failed")
                except Exception as e:
                    print(f"  Node {i} ({ip}): error: {e}")

        if args.build_only:
            print("\n[build_only] Skipping query phase after full graph build/save.")
        elif args.skip_queries:
            print("\n[skip_queries] Skipping query phase entirely (build/partition/save only).")
        elif runner.query_nodes:
            # 多机分布式查询模式：通过 SSH 在每台 query 主机上启动 query.py，文件屏障同步
            print(f"\n{'='*60}\n  Distributed Query Mode (query_nodes from config.json)\n{'='*60}")
            runner.orchestrate_distributed_query()
        else:
            print("\n[main.py] No query_nodes in config.json and --skip_queries not set.")
            print("  main.py only orchestrates: workers + build + partition + save.")
            print("  To run queries, either add 'query_nodes' to config.json (multi-machine via query.py)")
            print("  or run query.py manually on each query machine.")
    except (QueryStallError, RuntimeError) as e:
        if os.environ.get('NO_RESTART') or args.build_only:
            print(f"[NO_RESTART] {type(e).__name__}: {e}. Skipping auto-restart for debugging.")
            print("Worker logs preserved for inspection.")
            sys.exit(2)
        print(f"[RESTART] {type(e).__name__}: {e}. Cleaning up and restarting main.py...")
        runner.cleanup_workers(force=True)
        # 重启策略：
        #   1. 去掉 --skip_prepare（需重新启动 worker 服务）
        #   2. 保留 --no_cleanup（batch 模式复用已启动的服务）
        #   3. 若已有保存的图，自动加 --graph_path 跳过重建
        restart_argv = [a for a in sys.argv if a != '--skip_prepare']
        graph_path = runner.graph_path
        if not graph_path:
            graph_path = f"./save/hnsw-{runner.data_name}-m{runner.m}-efc{runner.ef_construction}-ll{runner.limited_level}.bin"
        if graph_path and os.path.exists(graph_path):
            if '--graph_path' not in restart_argv:
                restart_argv += ['--graph_path', graph_path]
        python = sys.executable
        os.execv(python, [python] + restart_argv)
    finally:
        if not os.environ.get('NO_RESTART'):
            runner.cleanup_workers()
