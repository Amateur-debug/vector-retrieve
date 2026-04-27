import os
import time
import argparse
import numpy as np
import json
import subprocess
from tqdm import tqdm
import threading
import multiprocessing as mp
from collections import defaultdict

import sys
sys.setrecursionlimit(5000)

# 集群部署参数统一从 config.json 读取；新环境只需编辑 config.json（ALL_IPS、HOST_IP、BASE_PORT、USER_NAME）
_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
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
DATASET_PRESETS = {
    "siftsmall": {
        "data_path":        os.path.join(_dataset_root, "siftsmall/siftsmall_base.fvecs"),
        "query_path":       os.path.join(_dataset_root, "siftsmall/siftsmall_query.fvecs"),
        "groundtruth_path": os.path.join(_dataset_root, "siftsmall/siftsmall_groundtruth.ivecs"),
        "dtype":            "float",
    },
    "sift": {
        "data_path":        os.path.join(_dataset_root, "sift/sift_base.fvecs"),
        "query_path":       os.path.join(_dataset_root, "sift/sift_query.fvecs"),
        "groundtruth_path": os.path.join(_dataset_root, "sift/sift_groundtruth.ivecs"),
        "dtype":            "float",
    },
    "sift100m": {
        "data_path":        os.path.join(_dataset_root, "sift100m/base.u8bin"),
        "query_path":       os.path.join(_dataset_root, "sift100m/queries.u8bin"),
        "groundtruth_path": os.path.join(_dataset_root, "sift100m/gt_100.bin"),
        "dtype":            "float",
    },
    "deep100m": {
        "data_path":        os.path.join(_dataset_root, "deep100m/base.fbin"),
        "query_path":       os.path.join(_dataset_root, "deep100m/queries.fbin"),
        "groundtruth_path": os.path.join(_dataset_root, "deep100m/gt_100.bin"),
        "dtype":            "float",
    },
    "uint8_test": {
        "data_path":        os.path.join(_dataset_root, "uint8_test/base.u8bin"),
        "query_path":       os.path.join(_dataset_root, "uint8_test/queries.u8bin"),
        "groundtruth_path": os.path.join(_dataset_root, "uint8_test/gt_100.bin"),
        "dtype":            "uint8",
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

# 从命令行提前解析 --dtype 和 --data_name 以确定数据类型
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

    import vector_retrieve as _vr
    _payload_cls, _hnsw_cls, _cluster_cls, _sdk_cls = _DTYPE_CLASS_MAP[_active_dtype]
    Payload = getattr(_vr, _payload_cls)
    HNSWGraph = getattr(_vr, _hnsw_cls)
    Cluster = getattr(_vr, _cluster_cls)
    ClusterSDKClient = getattr(_vr, _sdk_cls)
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
    def __init__(self, args, use_cpp):
        self.args = args
        self.use_cpp = use_cpp

        self.num_construction_threads = args.num_construction_threads
        self.num_query_threads = args.num_query_threads
        self.num_threads = args.num_threads  # 服务端每 node 搜索线程数（msg_pool_size）
        self.num_cpu = args.num_cpu

        # 集群部署参数统一从 config.json 读取；新环境只需编辑 config.json。
        # 约定：host 大括号描述本机（运行 main.py 的机器）；nodes 为集群节点有序列表，
        # 节点按任意顺序排列均可。若 host 也参与集群，其信息须同时出现在 nodes 中
        # （位置任意，子集选取逻辑会保证包含 host）；若 host 不在 nodes 中，则 main.py
        # 仅作协调器，无本地 worker。
        self.host_info = _CLUSTER_CONFIG["host"]
        self.all_nodes = list(_CLUSTER_CONFIG["nodes"])
        for key in ("ip", "port", "user_name"):
            if key not in self.host_info:
                raise ValueError(f"config.json: host 缺少字段 '{key}'")
        for idx, nd in enumerate(self.all_nodes):
            for key in ("ip", "port", "user_name"):
                if key not in nd:
                    raise ValueError(f"config.json: nodes[{idx}] 缺少字段 '{key}'")

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

        # 选择实际使用的节点子集：按 nodes 顺序取前 node_num 个；
        # 若 host 在 nodes 中，子集必须包含 host（host 不在前 node_num 时，替换末位为 host）。
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

        self.k = args.k
        self.m = args.m
        self.ef_construction = args.ef_construction
        self.ef_search = args.ef_search
        self.limited_level = args.limited_level
        self.partition_mode = args.partition_mode
        self.use_layer = args.layer

        self.layer_config = {
            0: [0, 1, 2, 3, 4],
            1: [0, 1, 2, 3, 4],
            2: [0, 1, 2, 3, 4],
            3: [0, 1, 2, 3, 4],
            4: [0, 1, 2, 3, 4],
            5: [0, 1, 2, 3, 4]
        }
        
        self.layer_replicate_flags = [
            False, False, True, True, True, True
        ]
        if hasattr(args, "layer_replicate_flags") and args.layer_replicate_flags:
            try:
                parsed_flags = json.loads(args.layer_replicate_flags)
                if isinstance(parsed_flags, list):
                    if len(parsed_flags) > 0:
                        self.layer_replicate_flags = [bool(v) for v in parsed_flags]
                else:
                    print(f"Warning: --layer_replicate_flags should be a JSON list, fallback to default {self.layer_replicate_flags}")
            except Exception as e:
                print(f"Warning: failed to parse --layer_replicate_flags ({e}), fallback to default {self.layer_replicate_flags}")

        self.replication = args.replication

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

    def setup_distributed_nodes(self):
        if not self.args.distributed:
            return
        if self.args.skip_prepare:
            print("Skipping distributed node setup (--skip_prepare)")
            self.worker_processes = []
            return

        print(f"Distributed mode enabled. Node IPs: {self.node_ips}")

        # Clean up old worker processes on all remote nodes and the local host
        print("Cleaning up old worker processes...")

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
                # We already attempted local cleanup; continue to next
                continue
            # Remote two-stage kill: find PIDs, TERM, wait, then KILL if still present
            remote_cmd = (
                f"pids=$(pgrep -f node_start.py || true); "
                f"if [ -n \"$pids\" ]; then echo \"Killing PIDs: $pids on {ip}\"; kill -TERM $pids 2>/dev/null || true; sleep 1; "
                f"pids2=$(pgrep -f node_start.py || true); if [ -n \"$pids2\" ]; then echo \"Force killing PIDs: $pids2 on {ip}\"; kill -KILL $pids2 2>/dev/null || true; fi; fi"
            )
            cmd = [
                "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30",
                f"{self.node_user_names[i]}@{ip}",
                remote_cmd
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
                log_file = f"{current_dir}/worker_{i}.log"

                distributed_flag = "--distributed" if self.args.distributed else ""
                shard_flag = "--shard" if self.args.shard else ""
                global_pull_flag = "--global-pull" if self.args.is_global_pull else ""

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
                    proc = subprocess.Popen(local_cmd, stdout=log_f, stderr=log_f)
                worker_processes.append(proc)
            else:
                # Launch remote worker
                print(f"Launching worker node {i} on {self.node_user_names[i]}@{ip}...")
                log_file = f"{current_dir}/worker_{i}.log"

                distributed_flag = "--distributed" if self.args.distributed else ""
                shard_flag = "--shard" if self.args.shard else ""
                global_pull_flag = "--global-pull" if self.args.is_global_pull else ""

                # Build remote command as a single string
                cpu_prefix = f"taskset -c 0-{self.num_cpu - 1} " if self.num_cpu and self.num_cpu > 0 else ""
                remote_cmd = (
                    f"cd {current_dir} && "
                    f"conda activate vector-retrieve && "
                    f"nohup {cpu_prefix}python -u node_start.py "
                    f"--node_id {i} --node_ips {node_ips_str} "
                    f"--dtype {_active_dtype} "
                    f"--m {self.args.m} --ef_construction {self.args.ef_construction} "
                    f"--ef_search {self.ef_search} --limited_level {self.args.limited_level} "
                    f"--comm_mode {self.comm_mode} "
                    f"--num_threads {self.num_threads} "
                    f"--max_track_num {self.max_track_num} "
                    f"{distributed_flag} {shard_flag} {global_pull_flag} "
                    f"> {log_file} 2>&1 &"
                )

                cmd = ["ssh", f"{self.node_user_names[i]}@{ip}", remote_cmd]
                print(f"  CMD: {' '.join(cmd)}")

                # Launch worker process in background (no output redirection needed, handled by nohup)
                proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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

    def prepare(self):
        if self.args.skip_prepare:
            print("Skipping graph loading and partitioning (--skip_prepare)")
            if self.args.distributed:
                self.cluster = None
                self.hnsw = None
                self.sdk_config = {
                    'ip': '127.0.0.1',
                    'port': self.node_ports[self.host_id]
                }
            return

        # 在分布式模式下，不创建cluster和hnsw，直接使用SDK
        if self.args.distributed:
            # 分布式模式：所有操作通过SDK完成
            print("Distributed mode: Using SDK client for all operations")
            self.cluster = None
            self.hnsw = None

            # 分布式模式：如有 graph_path 则直接通过 SDK 加载
            print(f"Configuring SDK for local node {self.host_id} ({self.node_ips[self.host_id]}:{self.node_ports[self.host_id]}) for building...")
            self.sdk_config = {
                'ip': '127.0.0.1',
                'port': self.node_ports[self.host_id]
            }

            if self.graph_path is not None and os.path.exists(self.graph_path) and not self.args.shard:
                # shard 模式需要每个节点独立建图，不能复用统一的 HNSW 图
                print(f"Loading HNSW graph via SDK from {self.graph_path}...")
                sdk_client = ClusterSDKClient(ip=self.sdk_config['ip'], port=self.sdk_config['port'], comm_mode=self.comm_mode)
                if not sdk_client.connect():
                    raise RuntimeError("Failed to connect SDK to local node for load_hnsw")
                try:
                    sdk_client.load_hnsw(self.graph_path)
                    # 图加载后通过专用接口覆盖运行时查询参数，避免依赖 load 时的历史参数。
                    sdk_client.set_hnsw_params(ef_search=self.ef_search)
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

            if not self.args.shard:
                # 通过 SDK 调用 partition
                print("Partitioning data across cluster nodes via SDK...")
                start_time = time.time()
                # 创建 SDK 客户端连接到本机
                partition_client = ClusterSDKClient(ip=self.sdk_config['ip'], port=self.sdk_config['port'], comm_mode=self.comm_mode)
                if not partition_client.connect():
                    raise RuntimeError("Failed to connect SDK to local node for partition")
                try:
                    if self.use_layer:
                        print(f"Partitioning in layer mode (replication={self.replication})...")
                        partition_client.partition(
                            mode="layer",
                            replication=self.replication,
                            layer_config=self.layer_config,
                            layer_partition_mode=self.partition_mode,
                            layer_replicate_flags=self.layer_replicate_flags,
                            num_threads=self.num_construction_threads,
                        )
                    else:
                        print(f"Partitioning in {self.partition_mode} mode (replication={self.replication})...")
                        partition_client.partition(
                            mode=self.partition_mode,
                            replication=self.replication,
                            num_threads=self.num_construction_threads,
                        )
                    partition_time = time.time() - start_time
                    print(f"Partitioning Time: {partition_time:.4f} seconds.")
                finally:
                    partition_client.disconnect()
            return

        # 非分布式模式：传统的cluster和hnsw创建方式
        if self.cluster_path is not None and os.path.exists(self.cluster_path):
            print("Loading cluster from disk...")
            self.cluster = Cluster.load(self.cluster_path)
            self.cluster.reset()
            if self.args.is_global_pull:
                self.cluster.is_global_pull = True
        elif self.graph_path is not None and os.path.exists(self.graph_path) and not self.args.shard:
            # shard 模式需要每个节点独立建图，不能复用统一的 HNSW 图
            print("Loading HNSW graph from disk...")
            self.hnsw = HNSWGraph.load(self.graph_path)
            self.hnsw.ef_search = self.ef_search
            print("Partitioning data across cluster nodes...")
            self.cluster = Cluster(self.hnsw, self.node_num, comm_mode=self.comm_mode, ib_dev=self.ib_dev, ib_port=self.ib_port, gid_index=self.gid_index, max_track_num=self.max_track_num)
            if self.args.is_global_pull:
                self.cluster.is_global_pull = True

            start_time = time.time()
            if self.use_layer:
                self.cluster.partition(mode="layer", replication=self.replication, layer_config=self.layer_config, layer_partition_mode=self.partition_mode, layer_replicate_flags=self.layer_replicate_flags, num_threads=self.num_construction_threads)
            else:
                self.cluster.partition(mode=self.partition_mode, replication=self.replication, layer_config=self.layer_config, num_threads=self.num_construction_threads)
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
                    print(f"Building cluster in shard mode with {num_threads} threads...")
                    start_time = time.time()

                    # 准备数据（直接传numpy数组，避免需要GIL的类型转换）
                    data_with_ids = [(idx, vector) for idx, vector in enumerate(data[:num_elements])]

                    import threading
                    chunk_size = (len(data_with_ids) + num_threads - 1) // num_threads
                    threads = []

                    # 使用列表存储计数器（避免nonlocal作用域问题）
                    progress_counter = [0]
                    progress_lock = threading.Lock()

                    def add_chunk(chunk, counter, lock):
                        local_count = 0
                        for vid, vec in chunk:
                            self.cluster.add_vector(vid, vec)
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

                    # 显示进度条（Reduce模式）
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
                    print(f"Building HNSW index with {num_threads} threads...")
                    start_time = time.time()

                    # 准备数据：(id, vector) pairs（直接传numpy数组，避免需要GIL的类型转换）
                    data_with_ids = [(idx, vector) for idx, vector in enumerate(data[:num_elements])]

                    # 使用多线程建图
                    import threading
                    chunk_size = (len(data_with_ids) + num_threads - 1) // num_threads
                    threads = []

                    # 使用列表存储计数器（避免nonlocal作用域问题）
                    progress_counter = [0]
                    progress_lock = threading.Lock()

                    def add_chunk(chunk, counter, lock):
                        local_count = 0
                        for vid, vec in chunk:
                            self.hnsw.add_vector(vid, vec)
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

                    # 显示进度条（HNSW建图模式）
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

            # 传统模式需要分区，shard模式不需要（因为每个节点已经有自己的图）
            if not self.args.shard:
                print("Partitioning data across cluster nodes...")
                start_time = time.time()
                if self.use_layer:
                    self.cluster.partition(mode="layer", replication=self.replication, layer_config=self.layer_config, layer_partition_mode=self.partition_mode, layer_replicate_flags=self.layer_replicate_flags, num_threads=self.num_construction_threads)
                else:
                    self.cluster.partition(mode=self.partition_mode, replication=self.replication, layer_config=self.layer_config, num_threads=self.num_construction_threads)
                partition_time = time.time() - start_time
                print(f"Partitioning Time: {partition_time:.4f} seconds.")

    def run_queries(self):
        queries = read_vectors(self.query_path)
        groundtruth = read_groundtruth(self.groundtruth_path)
        print(f"Queries shape: {queries.shape}")
        print(f"Groundtruth shape: {groundtruth.shape}")
        max_num_queries = queries.shape[0]
        self.num_queries = max_num_queries

        print("Performing queries...")

        # 分布式模式下，查询入口路由由 SDK 负责并进行一次性初始化。
        if self.args.distributed:
            if self.sdk_config is None:
                self.sdk_config = {
                    'ip': "127.0.0.1",
                    'port': self.node_ports[self.host_id]
                }
            print("Distributed query entry nodes are initialized by SDK from cluster info.")

        # 各线程查询循环结束的时间戳，用于准确计算 QPS（不包含离线召回率计算时间）
        thread_query_end_times = [0.0] * self.num_query_threads

        def query(worker_id, stats_list, progress_counter=None):
            # 查询结果暂存，召回率等 Python 密集计算推迟到查询循环外，减少 GIL 竞争
            local_results = {}          # i -> list[int]
            local_payloads = {}         # i -> (remote_transfer_cnt, remote_transfer_data, distance_cnt, pull_cnt, pull_data, backtrack_cnt)

            # 每个进程创建独立的 SDK 客户端实例。
            thread_sdk_client = None
            if self.args.distributed:
                thread_sdk_client = ClusterSDKClient(
                    ip=self.sdk_config['ip'],
                    port=self.sdk_config['port'],
                    comm_mode=self.comm_mode
                )
                # 所有分布式模式都需要初始化路由（包括 shard 模式）
                if not thread_sdk_client.init_routes():
                    # init_routes 失败时回退到默认地址
                    print(f"[Worker {worker_id}] Warning: init_routes failed, using default endpoint")

            try:
                # 打乱查询顺序，每个线程使用不同的随机顺序
                import random
                query_indices = list(range(self.num_queries))
                random.seed(worker_id)  # 每个线程使用不同的种子
                random.shuffle(query_indices)

                local_count = 0
                if self.args.distributed:
                    # 流水线模式：先发后收，充分利用远程转发等待时间
                    PIPELINE_DEPTH = 16
                    pending = {}  # query_id -> (original_index, payload)
                    sent = 0
                    received = 0

                    for idx, i in enumerate(query_indices):
                        payload = Payload()
                        qid = i + worker_id * self.num_queries
                        thread_sdk_client.query_send(qid, queries[i], payload, k=self.k)
                        pending[qid] = (i, payload)
                        sent += 1

                        # 管道满时收一个结果
                        while sent - received >= PIPELINE_DEPTH:
                            ret_qid, results, ret_payload = thread_sdk_client.query_recv()
                            orig_idx, _ = pending.pop(ret_qid)
                            local_results[orig_idx] = results
                            local_payloads[orig_idx] = (ret_payload.remote_transfer_cnt, ret_payload.remote_transfer_data, ret_payload.distance_cnt, ret_payload.pull_cnt, ret_payload.pull_data, ret_payload.backtrack_cnt)
                            received += 1
                            local_count += 1
                            if progress_counter is not None and local_count % 10 == 0:
                                progress_counter[0] += 10
                                local_count = 0

                    # 收完剩余结果
                    while received < sent:
                        ret_qid, results, ret_payload = thread_sdk_client.query_recv()
                        orig_idx, _ = pending.pop(ret_qid)
                        local_results[orig_idx] = results
                        local_payloads[orig_idx] = (ret_payload.remote_transfer_cnt, ret_payload.remote_transfer_data, ret_payload.distance_cnt, ret_payload.pull_cnt, ret_payload.pull_data, ret_payload.backtrack_cnt)
                        received += 1
                        local_count += 1
                        if progress_counter is not None and local_count % 10 == 0:
                            progress_counter[0] += 10
                            local_count = 0
                else:
                    for idx, i in enumerate(query_indices):
                        payload = Payload()
                        if self.args.shard:
                            results = self.cluster.query(i + worker_id * self.num_queries, queries[i], payload, self.K)
                        else:
                            results = self.cluster.nodes[self.host_id].query(i + worker_id * self.num_queries, queries[i], payload, k=self.k)

                        local_results[i] = results
                        local_payloads[i] = (payload.remote_transfer_cnt, payload.remote_transfer_data, payload.distance_cnt, payload.pull_cnt, payload.pull_data, payload.backtrack_cnt)

                        local_count += 1
                        if progress_counter is not None:
                            if local_count % 10 == 0:
                                progress_counter[0] += 10
                                local_count = 0

                # 更新剩余的进度
                if progress_counter is not None and local_count > 0:
                    progress_counter[0] += local_count

            finally:
                # 每个进程完成查询后主动释放连接。
                if thread_sdk_client:
                    thread_sdk_client.disconnect()

            # 记录查询循环结束时间（不含后续召回率计算），用于准确计算 QPS
            thread_query_end_times[worker_id] = time.time()

            # ---- 查询循环结束后，在线程内离线计算召回率等统计 ----
            # 此时所有线程的查询已完成（或即将完成），GIL 竞争大幅降低
            local_recall = {}
            local_remote_transfer_cnt = {}
            local_remote_transfer_data = {}
            local_distance_cnt = {}
            local_pull_cnt = {}
            local_pull_data = {}
            local_backtrack_cnt = {}

            for i, results in local_results.items():
                results_set = set(results)
                true_ids = set(groundtruth[i][:self.k])
                correct = len(results_set.intersection(true_ids))
                local_recall[i] = correct / self.k

                rtc, rtl, dc, pc, pd, bt = local_payloads[i]
                local_remote_transfer_cnt[i] = rtc
                local_remote_transfer_data[i] = rtl
                local_distance_cnt[i] = dc
                local_pull_cnt[i] = pc
                local_pull_data[i] = pd
                local_backtrack_cnt[i] = bt

            stats_list[worker_id] = {
                "recall": local_recall,
                "remote_transfer_cnt": local_remote_transfer_cnt,
                "remote_transfer_data": local_remote_transfer_data,
                "distance_cnt": local_distance_cnt,
                "pull_cnt": local_pull_cnt,
                "pull_data": local_pull_data,
                "backtrack_cnt": local_backtrack_cnt,
            }

        total_query_start_time = time.time()
        total_queries = self.num_queries * self.num_query_threads

        if self.num_query_threads > 1:
            # 使用多进程（绕过 GIL，真正的 CPU 并行）
            import multiprocessing as mp

            # 子进程入口函数（必须是顶层可 pickle 的 target）
            # 通过 mp.Queue 传回结果，通过 mp.Value 共享进度
            progress_counter_mp = mp.Value('i', 0)
            result_queue = mp.Queue()

            build_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'build'))

            def query_worker_process(worker_id, num_queries, queries_np, groundtruth, k,
                                     distributed, shard, sdk_config, comm_mode,
                                     host_id, cluster_ref, progress_val, result_q,
                                     build_dir_for_subprocess,
                                     sdk_cls_name, payload_cls_name):
                """子进程入口：独立创建 SDK 客户端，执行查询，通过 Queue 返回结果"""
                import random
                import sys
                if build_dir_for_subprocess not in sys.path:
                    sys.path.insert(0, build_dir_for_subprocess)

                local_results = {}
                local_payloads = {}

                thread_sdk_client = None
                if distributed:
                    import vector_retrieve as _vr_sub
                    ClusterSDKClient = getattr(_vr_sub, sdk_cls_name)
                    Payload = getattr(_vr_sub, payload_cls_name)
                    thread_sdk_client = ClusterSDKClient(
                        ip=sdk_config['ip'],
                        port=sdk_config['port'],
                        comm_mode=comm_mode
                    )
                    if not thread_sdk_client.init_routes():
                        print(f"[Worker {worker_id}] Warning: init_routes failed, using default endpoint")

                try:
                    query_indices = list(range(num_queries))
                    random.seed(worker_id)
                    random.shuffle(query_indices)

                    local_count = 0
                    if distributed:
                        PIPELINE_DEPTH = 16
                        pending = {}
                        sent = 0
                        received = 0

                        for idx, i in enumerate(query_indices):
                            payload = Payload()
                            qid = i + worker_id * num_queries
                            thread_sdk_client.query_send(qid, queries_np[i].tolist(), payload, k=k)
                            pending[qid] = i
                            sent += 1

                            while sent - received >= PIPELINE_DEPTH:
                                ret_qid, results, ret_payload = thread_sdk_client.query_recv()
                                orig_idx = pending.pop(ret_qid)
                                local_results[orig_idx] = results
                                local_payloads[orig_idx] = (ret_payload.remote_transfer_cnt, ret_payload.remote_transfer_data, ret_payload.distance_cnt, ret_payload.pull_cnt, ret_payload.pull_data, ret_payload.backtrack_cnt)
                                received += 1
                                local_count += 1
                                if local_count % 10 == 0:
                                    with progress_val.get_lock():
                                        progress_val.value += 10
                                    local_count = 0

                        while received < sent:
                            ret_qid, results, ret_payload = thread_sdk_client.query_recv()
                            orig_idx = pending.pop(ret_qid)
                            local_results[orig_idx] = results
                            local_payloads[orig_idx] = (ret_payload.remote_transfer_cnt, ret_payload.remote_transfer_data, ret_payload.distance_cnt, ret_payload.pull_cnt, ret_payload.pull_data, ret_payload.backtrack_cnt)
                            received += 1
                            local_count += 1
                            if local_count % 10 == 0:
                                with progress_val.get_lock():
                                    progress_val.value += 10
                                local_count = 0
                    else:
                        # 非分布式模式：子进程无法共享 cluster 对象，回退到线程模式
                        # 此分支不应被多进程调用，仅作防御
                        pass

                    if local_count > 0:
                        with progress_val.get_lock():
                            progress_val.value += local_count

                finally:
                    if thread_sdk_client:
                        thread_sdk_client.disconnect()

                # 查询循环结束时间
                query_end_time = time.time()

                # 离线计算召回率
                local_recall = {}
                local_remote_transfer_cnt = {}
                local_remote_transfer_data = {}
                local_distance_cnt = {}
                local_pull_cnt = {}
                local_pull_data = {}
                local_backtrack_cnt = {}

                for i, results in local_results.items():
                    results_set = set(results)
                    true_ids = set(groundtruth[i][:k])
                    correct = len(results_set.intersection(true_ids))
                    local_recall[i] = correct / k

                    rtc, rtl, dc, pc, pd, bt = local_payloads[i]
                    local_remote_transfer_cnt[i] = rtc
                    local_remote_transfer_data[i] = rtl
                    local_distance_cnt[i] = dc
                    local_pull_cnt[i] = pc
                    local_pull_data[i] = pd
                    local_backtrack_cnt[i] = bt

                result_q.put((worker_id, query_end_time, {
                    "recall": local_recall,
                    "remote_transfer_cnt": local_remote_transfer_cnt,
                    "remote_transfer_data": local_remote_transfer_data,
                    "distance_cnt": local_distance_cnt,
                    "pull_cnt": local_pull_cnt,
                    "pull_data": local_pull_data,
                    "backtrack_cnt": local_backtrack_cnt,
                }))

            if self.args.distributed:
                # 多进程模式（分布式查询，绕过 GIL）
                processes = []
                for i in range(self.num_query_threads):
                    p = mp.Process(target=query_worker_process, args=(
                        i, self.num_queries, queries, groundtruth, self.k,
                        True, self.args.shard,
                        self.sdk_config, self.comm_mode,
                        self.host_id, None,
                        progress_counter_mp, result_queue,
                        build_dir_path,
                        _sdk_cls, _payload_cls
                    ))
                    p.start()
                    processes.append(p)
                self.query_worker_processes = processes

                # 进度条显示
                # 进度条：只跟踪查询发送进度，到 100% 后立即关闭
                with tqdm(total=total_queries, desc="Querying", unit="qry",
                          bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}, QPS: {postfix}]') as pbar:
                    pbar.set_postfix_str("0.0")
                    last_value = 0
                    last_update_time = time.time()
                    stall_last_value = 0
                    stall_check_time = time.time()
                    STALL_TIMEOUT = 60

                    while True:
                        current = progress_counter_mp.value
                        now = time.time()

                        if now - last_update_time >= 0.5 or current >= total_queries:
                            elapsed = now - total_query_start_time
                            current_qps = current / elapsed if elapsed > 0 else 0.0
                            pbar.update(current - last_value)
                            pbar.set_postfix_str(f"{current_qps:.1f}")
                            last_value = current
                            last_update_time = now

                        # 查询全部完成，退出进度条循环
                        if current >= total_queries:
                            break

                        if current != stall_last_value:
                            stall_last_value = current
                            stall_check_time = now
                        elif now - stall_check_time >= STALL_TIMEOUT:
                            print(f"\n[WARNING] Query progress stalled for {STALL_TIMEOUT}s (stuck at {current}/{total_queries}). Restarting main.py...")
                            raise QueryStallError(f"Query progress stalled for {STALL_TIMEOUT}s")

                        time.sleep(0.1)

                # 先从 Queue 读结果（必须在 join 之前，否则管道满时会死锁）
                stats_list = [None] * self.num_query_threads
                for _ in range(self.num_query_threads):
                    worker_id, query_end_time, stats = result_queue.get()
                    stats_list[worker_id] = stats
                    thread_query_end_times[worker_id] = query_end_time

                # 再 join 进程
                for p in processes:
                    p.join()

                thread_stats = stats_list
            else:
                # 非分布式模式：仍使用线程（cluster 对象无法跨进程共享）
                stats_list = [None] * self.num_query_threads
                progress_counter = [0]
                threads = []
                for i in range(self.num_query_threads):
                    t = threading.Thread(target=query, args=(i, stats_list, progress_counter))
                    t.start()
                    threads.append(t)

                with tqdm(total=total_queries, desc="Querying", unit="qry",
                          bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] QPS: {postfix}') as pbar:
                    pbar.set_postfix_str("0.0")
                    last_value = 0
                    last_update_time = time.time()
                    stall_last_value = 0
                    stall_check_time = time.time()
                    STALL_TIMEOUT = 60

                    while any(t.is_alive() for t in threads):
                        current = progress_counter[0]
                        now = time.time()
                        if now - last_update_time >= 0.5:
                            elapsed = now - total_query_start_time
                            current_qps = current / elapsed if elapsed > 0 else 0.0
                            pbar.update(current - last_value)
                            pbar.set_postfix_str(f"{current_qps:.1f}")
                            last_value = current
                            last_update_time = now

                        if current != stall_last_value:
                            stall_last_value = current
                            stall_check_time = now
                        elif now - stall_check_time >= STALL_TIMEOUT:
                            print(f"\n[WARNING] Query progress stalled for {STALL_TIMEOUT}s (stuck at {current}/{total_queries}). Restarting main.py...")
                            raise QueryStallError(f"Query progress stalled for {STALL_TIMEOUT}s")

                        time.sleep(0.1)

                    for t in threads:
                        t.join()

                    current = progress_counter[0]
                    elapsed = time.time() - total_query_start_time
                    current_qps = current / elapsed if elapsed > 0 else 0.0
                    pbar.update(current - last_value)
                    pbar.set_postfix_str(f"{current_qps:.1f}")

                thread_stats = stats_list
        else:
            # 单线程模式也显示进度条
            progress_counter = [0]
            progress_lock = threading.Lock()
            
            # 包装query函数以显示进度
            stats_list = [None]

            # 单线程模式的看门狗线程：检测进度停滞
            stall_detected = [False]
            def _single_thread_watchdog(counter, stall_flag):
                STALL_TIMEOUT = 10
                last_val = counter[0]
                last_change_time = time.time()
                while not stall_flag[0]:
                    time.sleep(1)
                    cur = counter[0]
                    if cur != last_val:
                        last_val = cur
                        last_change_time = time.time()
                    elif time.time() - last_change_time >= STALL_TIMEOUT:
                        print(f"\n[WARNING] Single-thread query progress stalled for {STALL_TIMEOUT}s (stuck at {cur}/{total_queries}). Restarting main.py...")
                        stall_flag[0] = True
                        # 中断主线程
                        import ctypes
                        ctypes.pythonapi.PyThreadState_SetAsyncExc(
                            ctypes.c_ulong(threading.main_thread().ident),
                            ctypes.py_object(QueryStallError)
                        )
                        return

            watchdog_thread = threading.Thread(target=_single_thread_watchdog, args=(progress_counter, stall_detected), daemon=True)
            watchdog_thread.start()
            
            with tqdm(total=total_queries, desc="Querying", unit="qry",
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] QPS: {postfix}') as pbar:
                pbar.set_postfix_str("0.0")
                last_update_time = time.time()
                
                # 在主线程中执行查询
                thread_sdk_client = None
                if self.args.distributed:
                    thread_sdk_client = ClusterSDKClient(
                        ip=self.sdk_config['ip'],
                        port=self.sdk_config['port'],
                        comm_mode=self.comm_mode
                    )
                    # 所有分布式模式都需要初始化路由（包括 shard 模式）
                    if not thread_sdk_client.init_routes():
                        print("[Query] Warning: init_routes failed, using default endpoint")

                local_recall = {}
                local_remote_transfer_cnt = {}
                local_remote_transfer_data = {}
                local_distance_cnt = {}
                local_backtrack_cnt = {}

                try:
                    import random
                    query_indices = list(range(self.num_queries))
                    random.seed(0)
                    random.shuffle(query_indices)

                    for idx, i in enumerate(query_indices):
                        query_vec = queries[i]
                        start_time = time.time()

                        payload = Payload()
                        if self.args.distributed:
                            thread_sdk_client.query_send(i, query_vec, payload, k=self.k)
                            ret_qid, results, ret_payload = thread_sdk_client.query_recv()
                            results = set(results)
                            local_remote_transfer_cnt[i] = ret_payload.remote_transfer_cnt
                            local_remote_transfer_data[i] = ret_payload.remote_transfer_data
                            local_distance_cnt[i] = ret_payload.distance_cnt
                            local_backtrack_cnt[i] = ret_payload.backtrack_cnt
                        else:
                            if self.args.shard:
                                results = self.cluster.query(i, query_vec, payload, self.k)
                                results = set(results)
                                local_remote_transfer_cnt[i] = 0
                                local_remote_transfer_data[i] = 0.0
                            else:
                                results = self.cluster.nodes[self.host_id].query(i, query_vec, payload, k=self.k)
                                results = set(results)
                                local_remote_transfer_cnt[i] = payload.remote_transfer_cnt
                                local_remote_transfer_data[i] = payload.remote_transfer_data
                            local_distance_cnt[i] = payload.distance_cnt
                            local_backtrack_cnt[i] = payload.backtrack_cnt
                        true_ids = set(groundtruth[i][:self.k])
                        correct = len(results.intersection(true_ids))
                        local_recall[i] = correct / self.k

                        # 更新进度条
                        pbar.update(1)
                        progress_counter[0] = pbar.n  # 同步给看门狗线程
                        now = time.time()
                        if now - last_update_time >= 0.5:
                            elapsed = now - total_query_start_time
                            current_qps = pbar.n / elapsed if elapsed > 0 else 0.0
                            pbar.set_postfix_str(f"{current_qps:.1f}")
                            last_update_time = now
                finally:
                    if thread_sdk_client:
                        thread_sdk_client.disconnect()

                stats_list[0] = {
                    "recall": local_recall,
                    "remote_transfer_cnt": local_remote_transfer_cnt,
                    "remote_transfer_data": local_remote_transfer_data,
                    "distance_cnt": local_distance_cnt,
                    "backtrack_cnt": local_backtrack_cnt,
                }
            
            # 查询正常完成，停止看门狗线程
            stall_detected[0] = True
            thread_stats = stats_list

        # 注意：每个进程自己的连接已经在 query() 函数结束时断开

        # 用各线程查询结束时间的平均值计算 QPS
        valid_times = [t for t in thread_query_end_times if t > 0]
        if valid_times:
            avg_query_end = sum(valid_times) / len(valid_times)
            total_query_time = avg_query_end - total_query_start_time
            # 打印最快/最慢线程的 QPS
            fastest_time = min(valid_times) - total_query_start_time
            slowest_time = max(valid_times) - total_query_start_time
            fastest_qps = self.num_queries / fastest_time if fastest_time > 0 else 0.0
            slowest_qps = self.num_queries / slowest_time if slowest_time > 0 else 0.0
            print(f"Fastest thread: {fastest_time:.4f}s, QPS={fastest_qps:.2f} | "
                  f"Slowest thread: {slowest_time:.4f}s, QPS={slowest_qps:.2f} | "
                  f"Gap: {slowest_time - fastest_time:.4f}s")
        else:
            total_query_time = time.time() - total_query_start_time
        qps = self.num_queries * self.num_query_threads / total_query_time if total_query_time > 0 else 0.0

        return thread_stats, total_query_time, qps

    def save_graph(self):
        if not self.args.save:
            return
        save_dir = "./save"
        os.makedirs(save_dir, exist_ok=True)
        if self.args.distributed:
            base = (f"{save_dir}/hnsw-{self.data_name}-m{self.m}-"
                    f"efc{self.ef_construction}-ll{self.limited_level}")
            cluster_path = f"{base}-cluster.json"
            hnsw_path = f"{base}.json"

            sdk_client = ClusterSDKClient(ip=self.sdk_config['ip'], port=self.sdk_config['port'], comm_mode=self.comm_mode)
            if not sdk_client.connect():
                raise RuntimeError("Failed to connect SDK to local node for save")
            try:
                # print(f"Saving cluster state to {cluster_path}...")
                # sdk_client.save_cluster(cluster_path)
                # self.cluster_path = cluster_path
                # print(f"Cluster saved: {os.path.abspath(cluster_path)}")

                print(f"Saving local HNSW to {hnsw_path}...")
                sdk_client.save_hnsw(hnsw_path)
                self.graph_path = hnsw_path
                print(f"HNSW saved: {os.path.abspath(hnsw_path)}")
            finally:
                sdk_client.disconnect()
        elif self.graph_path is None:
            self.graph_path = (f"{save_dir}/hnsw-{self.data_name}-m{self.m}-"
                               f"efc{self.ef_construction}-ll{self.limited_level}.json")
            print(f"Saving graph state to {self.graph_path}...")
            self.hnsw.save(self.graph_path)

    def save_results(self, thread_stats, total_query_time, qps, qps_runs=None):
        # --- 计算并打印最终结果 ---
        recall = {}
        remote_transfer_cnt = {}
        remote_transfer_data = {}
        distance_cnt = {}
        pull_cnt = {}
        pull_data = {}
        backtrack_cnt = {}
        for s in thread_stats:
            if not s:
                continue
            recall.update(s["recall"])
            remote_transfer_cnt.update(s["remote_transfer_cnt"])
            remote_transfer_data.update(s["remote_transfer_data"])
            distance_cnt.update(s["distance_cnt"])
            pull_cnt.update(s.get("pull_cnt", {}))
            pull_data.update(s.get("pull_data", {}))
            backtrack_cnt.update(s.get("backtrack_cnt", {}))

        avg_recall = sum(recall.values()) / self.num_queries
        all_remote_transfer_cnt = sum(remote_transfer_cnt.values())
        avg_remote_transfer_cnt = all_remote_transfer_cnt / self.num_queries
        avg_remote_transfer_data_per_transfer = sum(remote_transfer_data.values()) / all_remote_transfer_cnt if all_remote_transfer_cnt > 0 else 0.0
        avg_remote_transfer_data_per_query = sum(remote_transfer_data.values()) / self.num_queries
        all_distance_cnt = sum(distance_cnt.values())
        avg_distance_cnt = all_distance_cnt / self.num_queries

        # 折返统计：backtrack_cnt 为 transfer 命中最近 max_track_num 窗口的次数，
        # backtrack_ratio 以 remote_transfer_cnt 为分母，反映整体折返占比。
        all_backtrack_cnt = sum(backtrack_cnt.values()) if backtrack_cnt else 0
        avg_backtrack_cnt = all_backtrack_cnt / self.num_queries if self.num_queries else 0.0
        backtrack_ratio = all_backtrack_cnt / all_remote_transfer_cnt if all_remote_transfer_cnt > 0 else 0.0

        # global-pull 模式拉取统计
        all_pull_cnt = sum(pull_cnt.values()) if pull_cnt else 0
        avg_pull_cnt = all_pull_cnt / self.num_queries if self.num_queries else 0.0
        all_pull_data = sum(pull_data.values()) if pull_data else 0
        avg_pull_data_per_pull = all_pull_data / all_pull_cnt if all_pull_cnt > 0 else 0.0
        avg_pull_data_per_query = all_pull_data / self.num_queries if self.num_queries else 0.0

        print("-" * 30)
        print("Evaluation Results:")
        print(f"Recall@{self.k}: {avg_recall:.4f}")
        print(f"QPS: {qps:.4f}")
        print(f"Query Time Total: {total_query_time:.4f} seconds")
        print(f"Average Remote Transfer Count per Query: {avg_remote_transfer_cnt:.4f}")
        print(f"Average Remote Transfer Data per Transfer: {avg_remote_transfer_data_per_transfer:.0f} bits ({avg_remote_transfer_data_per_transfer/8:.0f} bytes)")
        print(f"Average Remote Transfer Data per Query: {avg_remote_transfer_data_per_query:.0f} bits ({avg_remote_transfer_data_per_query/8:.0f} bytes)")
        print(f"Average Distance Computations per Query: {avg_distance_cnt:.4f}")
        print(f"Average Backtrack Count per Query: {avg_backtrack_cnt:.4f} (window={self.max_track_num}, ratio={backtrack_ratio:.4f})")
        if self.args.is_global_pull:
            print(f"Average Pull Count per Query: {avg_pull_cnt:.4f}")
            print(f"Average Pull Data per Pull: {avg_pull_data_per_pull:.0f} bits ({avg_pull_data_per_pull/8:.0f} bytes)")
            print(f"Average Pull Data per Query: {avg_pull_data_per_query:.0f} bits ({avg_pull_data_per_query/8:.0f} bytes)")
        print("-" * 30)

        # --- 保存所有结果 ---
        print("-" * 30)

        # 构建统计参数
        if self.args.distributed:
            # 分布式模式：通过 SDK 获取与非分布式一致的参数
            info_client = ClusterSDKClient(ip=self.sdk_config['ip'], port=self.sdk_config['port'], comm_mode=self.comm_mode)
            if not info_client.connect():
                raise RuntimeError("Failed to connect SDK to local node for get_information")
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
                "max_level": hnsw_info.get("max_level") if hnsw_info else self.limited_level
            }
            cluster_params = {
                "node_num": cluster_info.get("node_num") if cluster_info else self.node_num,
                "partition_mode": cluster_info.get("partition_mode") if cluster_info else self.partition_mode,
                "replication_level": cluster_info.get("replication_level") if cluster_info else self.replication,
                "replication_num": cluster_info.get("replication_num") if cluster_info else 0,
                "bandwidth": cluster_info.get("bandwidth") if cluster_info else 0,
                "is_distributed": cluster_info.get("is_distributed") if cluster_info else True,
                "is_shard": cluster_info.get("is_shard") if cluster_info else self.args.shard,
                "is_global_pull": cluster_info.get("is_global_pull") if cluster_info else self.args.is_global_pull,
                "max_track_num": self.max_track_num,
            }

            if cluster_params["partition_mode"] == "layer":
                cluster_params["layer_partition_mode"] = self.partition_mode
                cluster_params["layer_config"] = cluster_info.get("layer_config") if cluster_info else self.layer_config
        else:
            # 非分布式模式：保存完整的hnsw和cluster参数
            hnsw_params = {
                "data_name": self.data_name,
                "dim": self.hnsw.dim,
                "m": self.hnsw.m,
                "ef_construction": self.hnsw.ef_construction,
                "ef_search": self.hnsw.ef_search,
                "limited_level": self.hnsw.limited_level,
                "max_level": self.hnsw.max_level
            }
            cluster_params = {
                "node_num": self.cluster.node_num,
                "partition_mode": self.cluster.partition_mode,
                "replication_level": self.cluster.replication_level,
                "replication_num": self.cluster.replication_num,
                "bandwidth": self.cluster.bandwidth,
                "is_distributed": self.cluster.is_distributed,
                "is_shard": self.cluster.is_shard,
                "is_global_pull": self.cluster.is_global_pull,
                "max_track_num": self.max_track_num,
            }

            if self.cluster.partition_mode == "layer":
                cluster_params["layer_partition_mode"] = self.partition_mode
                cluster_params["layer_config"] = self.layer_config

        stats = {
            "hnsw_params": hnsw_params,
            "cluster_params": cluster_params,
            "num_queries": self.num_queries,
            "num_query_threads": self.num_query_threads,
            "num_threads": self.num_threads,
            "k": self.k,
            "qps": qps,
            "qps_runs": qps_runs if qps_runs else [qps],
            f"avg_recall@{self.k}": avg_recall,
            "avg_remote_transfer_cnt": avg_remote_transfer_cnt,
            "avg_remote_transfer_data_per_transfer": avg_remote_transfer_data_per_transfer,
            "avg_remote_transfer_data_per_query": avg_remote_transfer_data_per_query,
            "avg_distance_cnt": avg_distance_cnt,
            "avg_pull_cnt": avg_pull_cnt,
            "avg_pull_data_per_pull": avg_pull_data_per_pull,
            "avg_pull_data_per_query": avg_pull_data_per_query,
            "avg_backtrack_cnt": avg_backtrack_cnt,
            "backtrack_ratio": backtrack_ratio,
            "recall": recall,
            "remote_transfer_cnt": remote_transfer_cnt,
            "remote_transfer_data": remote_transfer_data,
            "distance_cnt": distance_cnt,
            "pull_cnt": pull_cnt,
            "pull_data": pull_data,
            "backtrack_cnt": backtrack_cnt,
        }
        if self.use_cpp:
            results_dir = "./results/cpp"
        else:
            results_dir = "./results/python"
        os.makedirs(results_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        distributed_suffix = "-distributed" if self.args.distributed else ""
        cpu_suffix = f"-cpus{self.args.num_cpu}" if self.args.distributed and self.args.num_cpu != 0 else ""
        shard_suffix = "-shard" if self.args.shard else ""
        global_pull_suffix = "-global_pull" if self.args.is_global_pull else ""
        layer_suffix = "-layer" if self.use_layer else ""

        qt_suffix = f"-query_threads{self.num_query_threads}"
        if self.args.shard:
            results_path = (f"{results_dir}/results-{self.data_name}-m{self.m}-"
                            f"efc{self.ef_construction}-efs{self.ef_search}-ll{self.limited_level}-"
                            f"node{self.node_num}-"
                            f"threads{self.num_threads}{qt_suffix}{distributed_suffix}{cpu_suffix}{shard_suffix}.json")
        elif self.args.is_global_pull:
            results_path = (f"{results_dir}/results-{self.data_name}-m{self.m}-"
                            f"efc{self.ef_construction}-efs{self.ef_search}-ll{self.limited_level}-"
                            f"node{self.node_num}-{self.partition_mode}-rep{self.replication}-"
                            f"threads{self.num_threads}{qt_suffix}{distributed_suffix}{cpu_suffix}{global_pull_suffix}.json")
        else:
            results_path = (f"{results_dir}/results-{self.data_name}-m{self.m}-"
                            f"efc{self.ef_construction}-efs{self.ef_search}-ll{self.limited_level}-"
                            f"node{self.node_num}-{self.partition_mode}-rep{self.replication}-"
                            f"threads{self.num_threads}{qt_suffix}{distributed_suffix}{cpu_suffix}{layer_suffix}.json")

        with open(results_path, 'w') as f:
            json.dump(stats, f, indent=4)

        # Print all saved paths
        print("\n" + "=" * 50)
        print(f"Results: {os.path.abspath(results_path)}")
        if not self.args.distributed and self.graph_path:
            print(f"Graph: {os.path.abspath(self.graph_path)}")
        if not self.args.distributed and self.cluster_path:
            print(f"Cluster: {os.path.abspath(self.cluster_path)}")
        if self.args.distributed:
            print("Distributed mode: Graph and Cluster managed by worker nodes")
        print("=" * 50 + "\n")

        return results_path

    def cleanup_workers(self, force=False):
        if not force and self.args.no_cleanup:
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
    parser.add_argument("--k", type=int, default=10, help="Number of nearest neighbors to retrieve")
    parser.add_argument("--m", type=int, default=10, help="Max number of connections per element")
    parser.add_argument("--ef_construction", type=int, default=200, help="Size of the dynamic candidate list for construction")
    parser.add_argument("--ef_search", type=int, default=15, help="Size of the dynamic candidate list for search")
    parser.add_argument("--limited_level", type=int, default=5, help="Level limit for HNSW")
    parser.add_argument("--node_num", type=int, default=6, help="Number of cluster nodes")
    parser.add_argument("--partition_mode", type=str, default="k-means", help="Data partition mode across nodes")
    parser.add_argument("--layer", action='store_true', default=False, help="Enable layer-based partitioning")
    parser.add_argument("--layer_config", type=str, default="{}", help="Layer configuration in JSON format")
    parser.add_argument("--layer_replicate_flags", type=str, default="[]", help="Layer replicate flags in JSON list; default behavior is [false,false,true,...] based on limited_level")
    parser.add_argument("--replication", type=int, default=2, help="Replication level for data replication across nodes")
    parser.add_argument("--distributed", default=False, action='store_true', help="Enable distributed mode")
    parser.add_argument("--shard", default=False, action='store_true', help="Enable shard mode (independent graphs per node)")
    parser.add_argument("--global-pull", dest="is_global_pull", default=False, action='store_true', help="Enable global-pull mode: every node holds the full graph skeleton, only vector data is partitioned; remote vectors are pulled on demand during search")
    parser.add_argument("--num_construction_threads", type=int, default=64, help="Number of threads for parallel graph construction (0=auto detect)")
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

        if num_runs <= 1:
            # 单次运行
            thread_stats, total_query_time, qps = runner.run_queries()
            runner.save_results(thread_stats, total_query_time, qps)
        else:
            # 多次运行取 QPS 均值
            qps_list = []
            last_thread_stats = None
            last_total_query_time = None
            for run_idx in range(num_runs):
                print(f"\n{'='*60}")
                print(f"  Query Run {run_idx + 1}/{num_runs}")
                print(f"{'='*60}")
                if run_idx > 0:
                    # 等待上一轮 worker 进程完全退出，让服务端 msg_pool_ 排空
                    time.sleep(3)
                thread_stats, total_query_time, qps = runner.run_queries()
                qps_list.append(qps)
                last_thread_stats = thread_stats
                last_total_query_time = total_query_time
                print(f"  Run {run_idx + 1}/{num_runs} QPS: {qps:.2f}")

            # 去掉最高最低取均值
            if len(qps_list) >= 3:
                sorted_qps = sorted(qps_list)
                trimmed = sorted_qps[1:-1]
                avg_qps = sum(trimmed) / len(trimmed)
            else:
                avg_qps = sum(qps_list) / len(qps_list)

            print(f"\n{'='*60}")
            print(f"  QPS runs: {[round(q, 2) for q in qps_list]}")
            print(f"  QPS avg (trimmed): {avg_qps:.2f}")
            print(f"{'='*60}")

            # 用最后一次的 thread_stats 保存结果，QPS 用均值
            runner.save_results(last_thread_stats, last_total_query_time, avg_qps, qps_runs=qps_list)
    except (QueryStallError, RuntimeError) as e:
        if os.environ.get('NO_RESTART'):
            print(f"[NO_RESTART] {type(e).__name__}: {e}. Skipping auto-restart for debugging.")
            print("Worker logs preserved for inspection.")
            sys.exit(2)
        print(f"[RESTART] {type(e).__name__}: {e}. Cleaning up and restarting main.py...")
        runner.cleanup_workers(force=True)
        # 重启时去掉 --skip_prepare，重新启动服务；保留 --no_cleanup 以维持 batch 的服务复用逻辑
        restart_argv = [a for a in sys.argv if a != '--skip_prepare']
        # 如果有已保存的图，加上 --graph_path 避免重新建图
        graph_path = runner.graph_path
        if not graph_path:
            # 根据参数推算默认保存路径
            graph_path = f"./save/hnsw-{runner.data_name}-m{runner.m}-efc{runner.ef_construction}-ll{runner.limited_level}.json"
        if graph_path and os.path.exists(graph_path):
            if '--graph_path' not in restart_argv:
                restart_argv += ['--graph_path', graph_path]
        python = sys.executable
        os.execv(python, [python] + restart_argv)
    finally:
        if not os.environ.get('NO_RESTART'):
            runner.cleanup_workers()
