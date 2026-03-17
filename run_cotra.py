import os
import sys
import time
import json
import argparse
import subprocess
import tempfile
import threading

import numpy as np


def fvecs_to_fbin(fvecs_path, fbin_path):
    """
    将 .fvecs 格式转换为 BigANN .fbin 格式。
    .fvecs: 每个向量前有一个 int32 维度值，格式为 [dim, v1, v2, ..., vD, dim, v1, ...]
    .fbin:  文件头为 [num_points(uint32), dim(uint32)]，后跟连续的 float32 向量数据
    """
    print(f"Converting {fvecs_path} -> {fbin_path}")
    fv = np.fromfile(fvecs_path, dtype=np.float32)
    dim = fv.view(np.int32)[0]
    fv = fv.reshape(-1, 1 + dim)
    vectors = fv[:, 1:]
    num_points = vectors.shape[0]
    print(f"  {num_points} vectors, dim={dim}")

    with open(fbin_path, 'wb') as f:
        np.array([num_points, dim], dtype=np.uint32).tofile(f)
        vectors.tofile(f)
    print(f"  Saved to {fbin_path}")
    return fbin_path


def ivecs_to_ibin(ivecs_path, ibin_path):
    """
    将 .ivecs 格式转换为 BigANN .ibin 格式（用于 groundtruth）。
    """
    print(f"Converting {ivecs_path} -> {ibin_path}")
    iv = np.fromfile(ivecs_path, dtype=np.int32)
    dim = iv[0]
    iv = iv.reshape(-1, 1 + dim)
    vectors = iv[:, 1:]
    num_points = vectors.shape[0]
    print(f"  {num_points} vectors, dim={dim}")

    with open(ibin_path, 'wb') as f:
        np.array([num_points, dim], dtype=np.uint32).tofile(f)
        vectors.tofile(f)
    print(f"  Saved to {ibin_path}")
    return ibin_path


def auto_convert_if_needed(path, target_ext):
    """
    如果路径是 .fvecs/.ivecs 格式，自动转换为 .fbin/.ibin。
    如果已经是正确格式或转换后的文件已存在，直接返回。
    """
    if path is None:
        return path

    base, ext = os.path.splitext(path)

    if ext == target_ext:
        return path

    converted_path = base + target_ext

    if os.path.exists(converted_path):
        print(f"Using existing converted file: {converted_path}")
        return converted_path

    if ext == ".fvecs" and target_ext == ".fbin":
        return fvecs_to_fbin(path, converted_path)
    elif ext == ".ivecs" and target_ext == ".ibin":
        return ivecs_to_ibin(path, converted_path)
    else:
        print(f"WARNING: Cannot convert {ext} to {target_ext}, using original path")
        return path


class CoTraRunner:
    """
    CoTra 分布式向量检索运行器。
    参考 main.py 的分布式模式，通过 SSH 在各节点上同时执行 CoTra 的 C++ 可执行文件。
    CoTra 使用 memcached 协调节点，每个节点运行相同的命令。
    """

    def __init__(self, args):
        self.args = args

        self.user_name = r"CONNECT\xchen740"
        self.default_ips = "192.168.200.204,192.168.200.205,192.168.200.206,192.168.200.207,192.168.200.208"
        self.host_ip = "192.168.200.206"
        self.host_id = 2

        self.node_ips = args.node_ips.split(',') if args.node_ips else self.default_ips.split(',')
        self.node_num = len(self.node_ips)

        # CoTra 项目路径
        self.project_root = os.path.abspath(os.path.dirname(__file__))
        self.cotra_dir = os.path.join(self.project_root, "CoTra")
        self.cotra_build_dir = os.path.join(self.cotra_dir, "build")
        self.cotra_scripts_dir = os.path.join(self.cotra_dir, "scripts")

        # 确保 build 目录存在
        if not os.path.isdir(self.cotra_build_dir):
            raise RuntimeError(f"CoTra build directory not found: {self.cotra_build_dir}. "
                               f"Please build CoTra first: cd CoTra && mkdir build && cd build && cmake .. && make -j")

        # memcached 端口
        self.memcached_port = args.memcached_port

        # 数据参数
        self.data_name = args.data_name
        self.data_path = args.data_path
        self.query_path = args.query_path
        self.groundtruth_path = args.groundtruth_path
        self.data_type = args.data_type
        self.dist_fn = args.dist_fn
        self.million = args.million

        # 索引参数
        self.graph_type = args.graph_type
        self.index_save_dir = args.index_save_dir
        self.R = args.R
        self.L = args.L
        self.B = args.B
        self.M = args.M
        self.num_threads = args.num_threads
        self.num_cpu = args.num_cpu

        # 搜索参数
        self.app_type = args.app_type
        self.k = args.k

        # RDMA 设备
        self.ib_dev = args.ib_dev

        # 节点 hostname 映射（通过 SSH 获取或使用 IP）
        self.node_hostnames = {}
        self.worker_processes = []

        # 生成的 conf 文件路径
        self.conf_file = None

    def _resolve_hostnames(self):
        """
        直接使用 IP 地址作为 conf 文件中的标识符。
        CoTra 内部通过 getaddrinfo 解析 conf 中的名称并与本机 IP 匹配，
        使用 hostname 可能因为 FQDN 后缀或多网卡导致匹配失败，
        所以直接使用 IP 最可靠。
        """
        print("Resolving node identities...")
        for i, ip in enumerate(self.node_ips):
            if ip == self.host_ip:
                self.host_id = i
            self.node_hostnames[i] = ip
            label = "(local)" if ip == self.host_ip else ""
            print(f"  Node {i}: {ip} {label}")

    def _generate_conf_file(self):
        """生成 CoTra 使用的 conf 文件"""
        # Leader 是 node 0（host_id 对应的节点作为 leader）
        leader_hostname = self.node_hostnames[0]

        conf_path = os.path.join(self.cotra_scripts_dir, "ip_auto.conf")
        lines = [leader_hostname, str(self.memcached_port)]
        for i in range(self.node_num):
            lines.append(f"{self.node_hostnames[i]}={i}")

        with open(conf_path, 'w') as f:
            f.write('\n'.join(lines) + '\n')

        self.conf_file = conf_path
        print(f"Generated conf file: {conf_path}")
        print(f"  Leader: {leader_hostname}:{self.memcached_port}")
        for i in range(self.node_num):
            print(f"  Node {i}: {self.node_hostnames[i]} ({self.node_ips[i]})")

    def _build_index_cmd(self):
        """构建建索引命令"""
        # 相对于 CoTra/build 目录的路径
        conf_rel = os.path.relpath(self.conf_file, self.cotra_build_dir)
        index_prefix = os.path.join(self.index_save_dir, "merged_index")

        cmd = (
            f"./tests/scala_index "
            f"--config_file {conf_rel} "
            f"--graph_type {self.graph_type} "
            f"--data_type {self.data_type} --dist_fn {self.dist_fn} "
            f"--data_path {self.data_path} "
            f"--index_path_prefix {index_prefix} "
            f"-R {self.R} -L {self.L} -B {self.B} -M {self.M} "
            f"-T {self.num_threads} -d {self.ib_dev} "
            f"--scala_v3 --scalagraph_v2 -s {self.million} -t {self.num_threads}"
        )
        return cmd

    def _build_query_cmd(self):
        """构建查询命令"""
        conf_rel = os.path.relpath(self.conf_file, self.cotra_build_dir)
        index_prefix = os.path.join(self.index_save_dir, "merged_index")

        cmd = (
            f"./tests/scala_anns "
            f"--config_file {conf_rel} "
            f"--app_type {self.app_type} "
            f"--graph_type {self.graph_type} "
            f"--data_type {self.data_type} --dist_fn {self.dist_fn} "
            f"--data_path {self.data_path} "
            f"--query_path {self.query_path} "
            f"--gt_path {self.groundtruth_path} "
            f"--index_path_prefix {index_prefix} "
            f"-R {self.R} -L {self.L} -B {self.B} -M {self.M} "
            f"-T {self.num_threads} -d {self.ib_dev} "
            f"--scala_v3 -s {self.million} -t {self.num_threads}"
        )
        return cmd

    def _build_restart_memcache_cmd(self):
        """构建在 leader 节点上重启 memcached 的命令"""
        leader_hostname = self.node_hostnames[0]
        port = self.memcached_port
        cmd = (
            # 杀掉旧的 memcached
            f"if [ -f /tmp/memcached.pid ]; then kill $(cat /tmp/memcached.pid) 2>/dev/null; rm -f /tmp/memcached.pid; fi; "
            # 启动新的 memcached
            f"memcached -u root -l {leader_hostname} -p {port} -c 10000 -d -P /tmp/memcached.pid; "
            f"sleep 1; "
            # 初始化计数器
            f'echo -e "set serverNum 0 0 1\\r\\n0\\r\\nquit\\r" | nc {leader_hostname} {port}; '
            f'echo -e "set clientNum 0 0 1\\r\\n0\\r\\nquit\\r" | nc {leader_hostname} {port}; '
            f"echo 'memcache cleared and restarted'"
        )
        return cmd

    def setup(self):
        """初始化：解析 hostname、生成 conf 文件、转换数据格式"""
        self._resolve_hostnames()
        self._generate_conf_file()

        # CoTra 使用 BigANN 二进制格式（.fbin/.ibin），自动转换 .fvecs/.ivecs
        self.data_path = auto_convert_if_needed(self.data_path, ".fbin")
        self.query_path = auto_convert_if_needed(self.query_path, ".fbin")
        self.groundtruth_path = auto_convert_if_needed(self.groundtruth_path, ".ibin")

    def _restart_memcached_on_leader(self):
        """在 leader 节点（node 0）上重启 memcached"""
        leader_ip = self.node_ips[0]
        memcache_cmd = self._build_restart_memcache_cmd()

        print(f"Restarting memcached on leader node 0 ({leader_ip})...")
        if leader_ip == self.host_ip:
            # 本地执行
            result = subprocess.run(["bash", "-c", memcache_cmd],
                                    capture_output=True, text=True, timeout=15)
            print(f"  {result.stdout.strip()}")
            if result.stderr.strip():
                print(f"  STDERR: {result.stderr.strip()}")
        else:
            cmd = [
                "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
                f"{self.user_name}@{leader_ip}", memcache_cmd
            ]
            result = subprocess.run(cmd, stdin=subprocess.DEVNULL,
                                    capture_output=True, text=True, timeout=15)
            print(f"  {result.stdout.strip()}")
            if result.stderr.strip():
                print(f"  STDERR: {result.stderr.strip()}")

    def _run_on_all_nodes(self, cotra_cmd, phase_name):
        """
        在所有节点上同时执行 CoTra 命令。
        CoTra 要求所有节点同时运行相同的命令，通过 memcached 协调。
        """
        self._restart_memcached_on_leader()
        time.sleep(2)  # 等待 memcached 完全启动

        print(f"\n{'='*50}")
        print(f"Starting {phase_name} on {self.node_num} nodes...")
        print(f"Command: {cotra_cmd}")
        print(f"{'='*50}\n")

        current_dir = os.getcwd()
        log_dir = os.path.join(current_dir, "cotra_logs")
        os.makedirs(log_dir, exist_ok=True)

        # 确保索引目录存在（在所有节点上）
        mkdir_cmd = f"mkdir -p {self.index_save_dir}"

        worker_processes = []
        log_files = []

        for i in range(self.node_num):
            ip = self.node_ips[i]
            log_path = os.path.join(log_dir, f"cotra_{phase_name}_node{i}.log")

            # CPU 限制前缀
            cpu_prefix = f"taskset -c 0-{self.num_cpu - 1} " if self.num_cpu and self.num_cpu > 0 else ""

            # 完整命令：cd 到 build 目录，创建索引目录，执行 CoTra
            full_cmd = f"cd {self.cotra_build_dir} && {mkdir_cmd} && {cpu_prefix}{cotra_cmd}"

            if ip == self.host_ip:
                self.host_id = i
                print(f"Launching node {i} on localhost ({ip})...")
                log_f = open(log_path, 'w')
                proc = subprocess.Popen(
                    ["bash", "-c", full_cmd],
                    stdout=log_f, stderr=log_f
                )
                worker_processes.append(proc)
                log_files.append(log_f)
            else:
                print(f"Launching node {i} on {self.user_name}@{ip}...")
                # SSH 执行，需要 source conda 环境
                remote_cmd = f"source ~/.bashrc && {full_cmd}"
                ssh_cmd = [
                    "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                    f"{self.user_name}@{ip}",
                    remote_cmd
                ]
                log_f = open(log_path, 'w')
                proc = subprocess.Popen(
                    ssh_cmd, stdin=subprocess.DEVNULL,
                    stdout=log_f, stderr=log_f
                )
                worker_processes.append(proc)
                log_files.append(log_f)

        self.worker_processes = worker_processes

        # 等待所有节点完成
        print(f"\nWaiting for all {self.node_num} nodes to complete {phase_name}...")
        start_time = time.time()

        # 轮询检查进度
        completed = [False] * self.node_num
        while not all(completed):
            for i, proc in enumerate(worker_processes):
                if not completed[i] and proc.poll() is not None:
                    completed[i] = True
                    elapsed = time.time() - start_time
                    status = "OK" if proc.returncode == 0 else f"FAILED (rc={proc.returncode})"
                    print(f"  Node {i} ({self.node_ips[i]}): {status} ({elapsed:.1f}s)")
            time.sleep(1)

        # 关闭日志文件
        for f in log_files:
            f.close()

        total_time = time.time() - start_time
        print(f"\n{phase_name} completed in {total_time:.1f}s")

        # 检查是否有节点失败
        failed_nodes = []
        for i, proc in enumerate(worker_processes):
            if proc.returncode != 0:
                failed_nodes.append(i)

        if failed_nodes:
            print(f"\nWARNING: Nodes {failed_nodes} failed during {phase_name}!")
            print("Check logs for details:")
            for i in failed_nodes:
                log_path = os.path.join(log_dir, f"cotra_{phase_name}_node{i}.log")
                print(f"  Node {i}: {log_path}")
                # 打印最后几行日志
                try:
                    with open(log_path, 'r') as f:
                        lines = f.readlines()
                        tail = lines[-20:] if len(lines) > 20 else lines
                        for line in tail:
                            print(f"    {line.rstrip()}")
                except Exception:
                    pass
            raise RuntimeError(f"{phase_name} failed on nodes {failed_nodes}")

        # 打印各节点日志摘要
        print(f"\nLog summary for {phase_name}:")
        for i in range(self.node_num):
            log_path = os.path.join(log_dir, f"cotra_{phase_name}_node{i}.log")
            try:
                with open(log_path, 'r') as f:
                    lines = f.readlines()
                    # 打印最后 5 行
                    tail = lines[-5:] if len(lines) > 5 else lines
                    print(f"  --- Node {i} ({self.node_ips[i]}) ---")
                    for line in tail:
                        print(f"    {line.rstrip()}")
            except Exception as e:
                print(f"  Node {i}: (failed to read log: {e})")

        return total_time

    def build_index(self):
        """在所有节点上构建索引"""
        cmd = self._build_index_cmd()
        return self._run_on_all_nodes(cmd, "build_index")

    def run_queries(self):
        """在所有节点上执行查询"""
        cmd = self._build_query_cmd()
        return self._run_on_all_nodes(cmd, "query")

    def cleanup_workers(self):
        """清理所有工作进程"""
        if not self.worker_processes:
            return
        print("\nCleaning up worker processes...")
        for proc in self.worker_processes:
            if proc.poll() is None:
                proc.terminate()
        time.sleep(1)
        for proc in self.worker_processes:
            if proc.poll() is None:
                proc.kill()
        self.worker_processes = []
        print("Worker processes cleaned up.")


# ── graph_type → app_type 默认映射 ──────────────────────────────────
GRAPH_TYPE_TO_APP_TYPE = {
    "scalagraph_v3": "scala_v3",
    "shared_nothing": "b2",
    "Kshared_nothing": "b2",
    "vamana": "single",
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CoTra 分布式向量检索 Python 启动器")

    # 节点配置
    parser.add_argument("--node_ips", type=str,
                        default="192.168.200.204,192.168.200.205,192.168.200.206,192.168.200.207,192.168.200.208",
                        help="Comma-separated list of node IPs")
    parser.add_argument("--memcached_port", type=int, default=2378,
                        help="Memcached port for CoTra coordination")

    # 数据集参数
    parser.add_argument("--data_name", type=str, default="sift", help="Name of the dataset")
    parser.add_argument("--data_path", type=str,
                        default="/home/CONNECT/xchen740/workspace/vector-retrieve/dataset/sift/sift_base.fvecs",
                        help="Path to base vectors file")
    parser.add_argument("--query_path", type=str,
                        default="/home/CONNECT/xchen740/workspace/vector-retrieve/dataset/sift/sift_query.fvecs",
                        help="Path to query vectors file")
    parser.add_argument("--groundtruth_path", type=str,
                        default="/home/CONNECT/xchen740/workspace/vector-retrieve/dataset/sift/sift_groundtruth.ivecs",
                        help="Path to groundtruth file")
    parser.add_argument("--data_type", type=str, default="float",
                        choices=["float", "uint8", "int8"], help="Vector data type")
    parser.add_argument("--dist_fn", type=str, default="l2",
                        choices=["l2", "ip"], help="Distance function")
    parser.add_argument("--million", type=int, default=1,
                        help="Dataset size in millions")

    # CoTra 索引参数
    parser.add_argument("--graph_type", type=str, default="scalagraph_v3",
                        choices=["scalagraph_v3", "shared_nothing", "Kshared_nothing", "vamana"],
                        help="Graph index type")
    parser.add_argument("--app_type", type=str, default=None,
                        choices=["scala_v3", "b2", "single"],
                        help="Search app type (auto-detected from graph_type if not set)")
    parser.add_argument("--index_save_dir", type=str, default=None,
                        help="Index save directory (auto-generated if not set)")
    parser.add_argument("-R", type=int, default=48, help="Max degree of graph index")
    parser.add_argument("-L", type=int, default=500, help="Build/search queue size (efConstruction)")
    parser.add_argument("-B", type=float, default=100, help="Memory budget (GB)")
    parser.add_argument("-M", type=float, default=120, help="Memory limit (GB)")
    parser.add_argument("-k", type=int, default=10, help="Number of nearest neighbors to retrieve")
    parser.add_argument("--num_threads", type=int, default=32,
                        help="Number of threads per node")
    parser.add_argument("--num_cpu", type=int, default=0,
                        help="Limit CPU cores per node process (0=unlimited)")

    # RDMA 设备
    parser.add_argument("--ib_dev", type=str, default="mlx5_0",
                        help="RDMA IB device name (e.g. mlx5_0, mlx4_0)")

    # 运行模式
    parser.add_argument("--mode", type=str, default="all",
                        choices=["all", "index", "query"],
                        help="Run mode: index only, query only, or both")
    parser.add_argument("--single", action='store_true', default=False,
                        help="Single-machine mode (no RDMA needed). "
                             "Overrides to --graph_type vamana --app_type single with 1 node.")

    args = parser.parse_args()

    # 单机模式：使用 shared_nothing + b2，只用本机
    # shared_nothing 支持 MACHINE_NUM=1（所有数据分配到节点 0），不需要 RDMA
    if args.single:
        args.graph_type = "shared_nothing"
        args.app_type = "single"
        args.node_ips = args.node_ips.split(',')[0]  # 只用第一个节点
        print(f"Single-machine mode: graph_type=shared_nothing, app_type=single, node={args.node_ips}")

    # 自动推断 app_type
    if args.app_type is None:
        args.app_type = GRAPH_TYPE_TO_APP_TYPE.get(args.graph_type, "scala_v3")
        print(f"Auto-detected app_type={args.app_type} for graph_type={args.graph_type}")

    # 自动生成 index_save_dir
    if args.index_save_dir is None:
        node_ips_list = args.node_ips.split(',') if ',' in args.node_ips else [args.node_ips]
        node_num = len(node_ips_list)
        args.index_save_dir = os.path.abspath(
            f"./cotra_index/{args.graph_type}/{args.data_name}_{args.million}M_{node_num}P"
        )
        print(f"Auto-generated index_save_dir={args.index_save_dir}")

    runner = CoTraRunner(args)
    try:
        runner.setup()

        if args.mode in ("all", "index"):
            print("\n" + "=" * 60)
            print("Phase 1: Building Index")
            print("=" * 60)
            build_time = runner.build_index()
            print(f"Index build completed in {build_time:.1f}s")

        if args.mode in ("all", "query"):
            print("\n" + "=" * 60)
            print("Phase 2: Running Queries")
            print("=" * 60)
            query_time = runner.run_queries()
            print(f"Query completed in {query_time:.1f}s")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except RuntimeError as e:
        print(f"\nError: {e}")
    finally:
        runner.cleanup_workers()
