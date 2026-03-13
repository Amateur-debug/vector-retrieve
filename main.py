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

if use_cpp:
    if os.path.isdir(build_dir):
        os.environ['PYTHONPATH'] = f"{build_dir}:{os.environ.get('PYTHONPATH','')}"
        if build_dir not in sys.path:
            sys.path.insert(0, build_dir)

    from vector_retrieve import PayloadFloat as Payload
    from vector_retrieve import HNSWGraphFloat as HNSWGraph
    from vector_retrieve import ClusterFloat as Cluster
    from vector_retrieve import ClusterSDKClientFloat as ClusterSDKClient
else:
    if os.path.isdir(py_source_dir):
        if py_source_dir not in sys.path:
            sys.path.insert(0, py_source_dir)

    from payload import Payload
    from hnsw import HNSWGraph
    from cluster import Cluster

def read_fvecs(filename, c_contiguous=True):
    """读取 .fvecs 格式的文件"""
    # 将文件内容读入一个 (N, D+1) 的浮点数数组
    # 其中 D 是向量维度。第一列是维度，可以忽略。
    fv = np.fromfile(filename, dtype=np.float32)

    # 获取第一个向量的维度
    dim = fv.view(np.int32)[0]

    # 重塑数组
    fv = fv.reshape(-1, 1 + dim)

    # 去掉维度列
    vectors = fv[:, 1:]

    if c_contiguous:
        return np.ascontiguousarray(vectors)
    
    return vectors

def read_ivecs(filename, c_contiguous=True):
    """读取 .ivecs 格式的文件"""
    # 将文件内容读入一个 (N, D+1) 的整数数组
    iv = np.fromfile(filename, dtype=np.int32)

    # 获取第一个向量的维度
    dim = iv[0]

    # 重塑数组
    iv = iv.reshape(-1, 1 + dim)

    # 去掉维度列
    vectors = iv[:, 1:]

    if c_contiguous:
        return np.ascontiguousarray(vectors)
    
    return vectors

class VectorRetrieveRunner:
    def __init__(self, args, use_cpp):
        self.args = args
        self.use_cpp = use_cpp

        self.num_construction_threads = args.num_construction_threads
        self.num_query_threads = args.num_query_threads
        self.num_cpu = args.num_cpu

        self.user_name = r"CONNECT\xchen740"
        self.default_ips = "192.168.200.204,192.168.200.205,192.168.200.206,192.168.200.207,192.168.200.208"
        self.host_ip = "192.168.200.206"
        self.host_id = 2
        self.node_ips = self.default_ips.split(',')
        self.node_num = len(self.node_ips) if self.args.distributed else self.args.node_num

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

        self.redundancy = args.redundancy

        # 通信模式参数
        self.comm_mode = getattr(args, 'comm_mode', 'tcp')
        self.ib_dev = getattr(args, 'ib_dev', '')
        self.ib_port = getattr(args, 'ib_port', 1)
        self.gid_index = getattr(args, 'gid_index', -1)

        self.cluster = None
        self.hnsw = None
        self.sdk_config = None
        self.worker_processes = []
        self.num_queries = 0

    def setup_distributed_nodes(self):
        if not self.args.distributed:
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
                self.host_id = i
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
                "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
                f"{self.user_name}@{ip}",
                remote_cmd
            ]
            print(f"  Cleaning node {i} on {ip}...")
            proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            cleanup_procs.append((proc, ip))
        # Wait for cleanup commands to finish (with timeout) and ensure remote processes are killed
        for proc, ip in cleanup_procs:
            try:
                stdout, stderr = proc.communicate(timeout=30)
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
        for i in range(len(self.node_ips)):
            ip = self.node_ips[i]
            if ip == self.host_ip:
                self.host_id = i
                print(f"Host machine identified as node {self.host_id} with IP {self.host_ip}.")
                # Launch local worker
                print(f"Launching worker node {i} on localhost...")
                log_file = f"{current_dir}/worker_{i}.log"

                distributed_flag = "--distributed" if self.args.distributed else ""
                reduce_flag = "--reduce" if self.args.reduce else ""

                # Build local command
                local_cmd = [
                    "python", "-u", "node_start.py",
                    "--node_id", str(i),
                    "--node_ips", self.default_ips,
                    "--m", str(self.args.m),
                    "--ef_construction", str(self.args.ef_construction),
                    "--ef_search", str(self.args.ef_search),
                    "--limited_level", str(self.args.limited_level),
                    "--comm_mode", self.comm_mode,
                ]
                if self.num_cpu and self.num_cpu > 0:
                    local_cmd = ["taskset", "-c", f"0-{self.num_cpu - 1}"] + local_cmd
                if distributed_flag:
                    local_cmd.append(distributed_flag)
                if reduce_flag:
                    local_cmd.append(reduce_flag)

                # Launch local worker in background
                with open(log_file, 'w') as log_f:
                    proc = subprocess.Popen(local_cmd, stdout=log_f, stderr=log_f)
                worker_processes.append(proc)
            else:
                # Launch remote worker
                print(f"Launching worker node {i} on {self.user_name}@{ip}...")
                log_file = f"{current_dir}/worker_{i}.log"

                distributed_flag = "--distributed" if self.args.distributed else ""
                reduce_flag = "--reduce" if self.args.reduce else ""

                # Build remote command as a single string
                cpu_prefix = f"taskset -c 0-{self.num_cpu - 1} " if self.num_cpu and self.num_cpu > 0 else ""
                remote_cmd = (
                    f"cd {current_dir} && "
                    f"conda activate vector-retrieve && "
                    f"nohup {cpu_prefix}python -u node_start.py "
                    f"--node_id {i} --node_ips {self.default_ips} "
                    f"--m {self.args.m} --ef_construction {self.args.ef_construction} "
                    f"--ef_search {self.args.ef_search} --limited_level {self.args.limited_level} "
                    f"--comm_mode {self.comm_mode} "
                    f"{distributed_flag} {reduce_flag} "
                    f"> {log_file} 2>&1 &"
                )

                cmd = ["ssh", f"{self.user_name}@{ip}", remote_cmd]
                print(f"  CMD: {' '.join(cmd)}")

                # Launch worker process in background (no output redirection needed, handled by nohup)
                proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                worker_processes.append(proc)

        # 等待所有 worker 节点真正启动并可连接
        print(f"Waiting for {len(worker_processes)} worker nodes to start (including local node {self.host_id})...")
        max_wait = 60  # 最长等待 60 秒
        check_interval = 2
        waited = 0
        all_ready = False
        while waited < max_wait:
            time.sleep(check_interval)
            waited += check_interval
            all_ready = True
            for i in range(len(self.node_ips)):
                port = 19000 + i
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
        mode_label = "reduce" if self.args.reduce else "normal"
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
            if self.args.distributed and self.args.reduce:
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
            if self.args.distributed and self.args.reduce:
                self._print_node_vector_counts()
            return build_time
        finally:
            thread_client.disconnect()

    def _print_node_vector_counts(self):
        """打印各节点的向量数量（分布式reduce模式）"""
        print("Node vector counts:")
        # 通过 SDK 获取每个节点的信息
        for i, ip in enumerate(self.node_ips):
            try:
                port = 19000 + i
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
        # 在分布式模式下，不创建cluster和hnsw，直接使用SDK
        if self.args.distributed:
            # 分布式模式：所有操作通过SDK完成
            print("Distributed mode: Using SDK client for all operations")
            self.cluster = None
            self.hnsw = None

            # 分布式模式：如有 graph_path 则直接通过 SDK 加载
            print(f"Configuring SDK for local node {self.host_id} ({self.node_ips[self.host_id]}:{19000 + self.host_id}) for building...")
            self.sdk_config = {
                'ip': '127.0.0.1',
                'port': 19000 + self.host_id
            }

            if self.graph_path is not None and os.path.exists(self.graph_path):
                print(f"Loading HNSW graph via SDK from {self.graph_path}...")
                sdk_client = ClusterSDKClient(ip=self.sdk_config['ip'], port=self.sdk_config['port'], comm_mode=self.comm_mode)
                if not sdk_client.connect():
                    raise RuntimeError("Failed to connect SDK to local node for load_hnsw")
                try:
                    sdk_client.load_hnsw(self.graph_path)
                finally:
                    sdk_client.disconnect()
            else:
                # 分布式模式：加载数据并构建
                print("Loading data for distributed mode...")
                data = read_fvecs(self.data_path)
                print(f"Data shape: {data.shape}")

                max_num_elements = data.shape[0]
                num_elements = max_num_elements

                self.build_index(data=data, num_elements=num_elements)

            if not self.args.reduce:
                # 通过 SDK 调用 partition
                print("Partitioning data across cluster nodes via SDK...")
                start_time = time.time()
                # 创建 SDK 客户端连接到本机
                partition_client = ClusterSDKClient(ip=self.sdk_config['ip'], port=self.sdk_config['port'], comm_mode=self.comm_mode)
                if not partition_client.connect():
                    raise RuntimeError("Failed to connect SDK to local node for partition")
                try:
                    if self.use_layer:
                        print(f"Partitioning in layer mode (redundancy={self.redundancy})...")
                        partition_client.partition(
                            mode="layer",
                            redundancy=self.redundancy,
                            layer_config=self.layer_config,
                            layer_partition_mode=self.partition_mode,
                            layer_replicate_flags=self.layer_replicate_flags,
                        )
                    else:
                        print(f"Partitioning in {self.partition_mode} mode (redundancy={self.redundancy})...")
                        partition_client.partition(
                            mode=self.partition_mode,
                            redundancy=self.redundancy,
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
        elif self.graph_path is not None and os.path.exists(self.graph_path):
            print("Loading HNSW graph from disk...")
            self.hnsw = HNSWGraph.load(self.graph_path)
            self.hnsw.ef_search = self.ef_search
            print("Partitioning data across cluster nodes...")
            self.cluster = Cluster(self.hnsw, self.node_num, comm_mode=self.comm_mode, ib_dev=self.ib_dev, ib_port=self.ib_port, gid_index=self.gid_index)

            start_time = time.time()
            if self.use_layer:
                self.cluster.partition(mode="layer", redundancy=self.redundancy, layer_config=self.layer_config, layer_partition_mode=self.partition_mode, layer_replicate_flags=self.layer_replicate_flags)
            else:
                self.cluster.partition(mode=self.partition_mode, redundancy=self.redundancy, layer_config=self.layer_config)
            partition_time = time.time() - start_time
            print(f"Partitioning Time: {partition_time:.4f} seconds.")
        else:
            print("Loading data...")
            data = read_fvecs(self.data_path)
            print(f"Data shape: {data.shape}")

            dim = data.shape[1]
            max_num_elements = data.shape[0]
            num_elements = max_num_elements

            self.hnsw = HNSWGraph(dim=dim, m=self.m, ef_construction=self.ef_construction, ef_search=self.ef_search, limited_level=self.limited_level, max_elements=num_elements)
            self.cluster = Cluster(self.hnsw, self.node_num, comm_mode=self.comm_mode, ib_dev=self.ib_dev, ib_port=self.ib_port, gid_index=self.gid_index)

            if self.args.reduce:
                # Reduce模式：先创建cluster，轮询添加向量到各节点独立建图
                self.cluster.is_reduce = True

                num_threads = self.num_construction_threads

                if num_threads > 1:
                    print(f"Building cluster in reduce mode with {num_threads} threads...")
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
                    print("Building cluster in reduce mode...")
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

            # 传统模式需要分区，reduce模式不需要（因为每个节点已经有自己的图）
            if not self.args.reduce:
                print("Partitioning data across cluster nodes...")
                start_time = time.time()
                if self.use_layer:
                    self.cluster.partition(mode="layer", redundancy=self.redundancy, layer_config=self.layer_config, layer_partition_mode=self.partition_mode, layer_replicate_flags=self.layer_replicate_flags)
                else:
                    self.cluster.partition(mode=self.partition_mode, redundancy=self.redundancy, layer_config=self.layer_config)
                partition_time = time.time() - start_time
                print(f"Partitioning Time: {partition_time:.4f} seconds.")

    def run_queries(self):
        queries = read_fvecs(self.query_path)
        groundtruth = read_ivecs(self.groundtruth_path)
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
                    'port': 19000 + self.host_id
                }
            print("Distributed query entry nodes are initialized by SDK from cluster info.")

        def query(worker_id, stats_list, progress_counter=None):
            local_recall = {}
            local_remote_transfer_cnt = {}
            local_remote_transfer_latency = {}
            local_distance_cnt = {}

            # 每个进程创建独立的 SDK 客户端实例。
            thread_sdk_client = None
            if self.args.distributed:
                thread_sdk_client = ClusterSDKClient(
                    ip=self.sdk_config['ip'],
                    port=self.sdk_config['port'],
                    comm_mode=self.comm_mode
                )
                # 所有分布式模式都需要初始化路由（包括 reduce 模式）
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
                for idx, i in enumerate(query_indices):
                    query_vec = queries[i]
                    start_time = time.time()

                    payload = Payload()
                    if self.args.distributed:
                        # 分布式模式：通过当前进程的 SDK 客户端发起 TCP 查询。
                        results = thread_sdk_client.query(i + worker_id * self.num_queries, query_vec, payload, k=self.k)
                        results = set(results)
                        local_remote_transfer_cnt[i] = payload.remote_transfer_cnt
                        local_remote_transfer_latency[i] = payload.remote_transfer_latency
                        avg_remote_transfer_latency = (payload.remote_transfer_latency / payload.remote_transfer_cnt) if payload.remote_transfer_cnt > 0 else 0.0
                    else:
                        # 非分布式模式。
                        if self.args.reduce:
                            # Reduce 模式：向所有节点查询并聚合结果。
                            results = self.cluster.query(i + worker_id * self.num_queries, query_vec, payload, self.K)
                            results = set(results)
                            local_remote_transfer_cnt[i] = 0
                            local_remote_transfer_latency[i] = 0.0
                            avg_remote_transfer_latency = 0.0
                        else:
                            # 常规模式：基于 payload 执行跨节点搜索。
                            results = self.cluster.nodes[self.host_id].query(i + worker_id * self.num_queries, query_vec, payload, k=self.k)
                            results = set(results)
                            local_remote_transfer_cnt[i] = payload.remote_transfer_cnt
                            local_remote_transfer_latency[i] = payload.remote_transfer_latency
                            avg_remote_transfer_latency = (payload.remote_transfer_latency / payload.remote_transfer_cnt) if payload.remote_transfer_cnt > 0 else 0.0

                    local_distance_cnt[i] = payload.distance_cnt

                    query_time = time.time() - start_time

                    # 召回率计算。
                    # 从 groundtruth 中读取真实近邻 ID 集合。
                    true_ids = set(groundtruth[i][:self.k])
                    # 统计检索结果与真实结果的交集数量。
                    correct = len(results.intersection(true_ids))
                    local_recall[i] = correct / self.k

                    # 更新进度（无锁，丢失少量计数不影响显示准确性）
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

            stats_list[worker_id] = {
                "recall": local_recall,
                "remote_transfer_cnt": local_remote_transfer_cnt,
                "remote_transfer_latency": local_remote_transfer_latency,
                "distance_cnt": local_distance_cnt,
            }

        total_query_start_time = time.time()
        total_queries = self.num_queries * self.num_query_threads

        if self.num_query_threads > 1:
            # 使用多线程（C++层已释放GIL，可以并行）
            stats_list = [None] * self.num_query_threads
            progress_counter = [0]
            threads = []
            for i in range(self.num_query_threads):
                t = threading.Thread(target=query, args=(i, stats_list, progress_counter))
                t.start()
                threads.append(t)
            
            # 进度条显示
            with tqdm(total=total_queries, desc="Querying", unit="qry", 
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] QPS: {postfix}') as pbar:
                pbar.set_postfix_str("0.0")
                last_value = 0
                last_update_time = time.time()
                
                # 用于检测进度停滞的变量
                stall_last_value = 0
                stall_check_time = time.time()
                STALL_TIMEOUT = 10  # 进度停滞超时阈值（秒）

                while any(t.is_alive() for t in threads):
                    current = progress_counter[0]
                    now = time.time()
                    # 每0.5秒更新一次进度条
                    if now - last_update_time >= 0.5:
                        elapsed = now - total_query_start_time
                        current_qps = current / elapsed if elapsed > 0 else 0.0
                        pbar.update(current - last_value)
                        pbar.set_postfix_str(f"{current_qps:.1f}")
                        last_value = current
                        last_update_time = now

                    # 检测进度停滞：如果进度有变化则重置计时器
                    if current != stall_last_value:
                        stall_last_value = current
                        stall_check_time = now
                    elif now - stall_check_time >= STALL_TIMEOUT:
                        print(f"\n[WARNING] Query progress stalled for {STALL_TIMEOUT}s (stuck at {current}/{total_queries}). Restarting main.py...")
                        raise QueryStallError(f"Query progress stalled for {STALL_TIMEOUT}s")

                    time.sleep(0.1)
                
                # 等待所有线程完成
                for t in threads:
                    t.join()
                
                # 最终更新
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
                    # 所有分布式模式都需要初始化路由（包括 reduce 模式）
                    if not thread_sdk_client.init_routes():
                        print("[Query] Warning: init_routes failed, using default endpoint")

                local_recall = {}
                local_remote_transfer_cnt = {}
                local_remote_transfer_latency = {}
                local_distance_cnt = {}

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
                            results = thread_sdk_client.query(i, query_vec, payload, k=self.k)
                            results = set(results)
                            local_remote_transfer_cnt[i] = payload.remote_transfer_cnt
                            local_remote_transfer_latency[i] = payload.remote_transfer_latency
                        else:
                            if self.args.reduce:
                                results = self.cluster.query(i, query_vec, payload, self.k)
                                results = set(results)
                                local_remote_transfer_cnt[i] = 0
                                local_remote_transfer_latency[i] = 0.0
                            else:
                                results = self.cluster.nodes[self.host_id].query(i, query_vec, payload, k=self.k)
                                results = set(results)
                                local_remote_transfer_cnt[i] = payload.remote_transfer_cnt
                                local_remote_transfer_latency[i] = payload.remote_transfer_latency

                        local_distance_cnt[i] = payload.distance_cnt
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
                    "remote_transfer_latency": local_remote_transfer_latency,
                    "distance_cnt": local_distance_cnt,
                }
            
            # 查询正常完成，停止看门狗线程
            stall_detected[0] = True
            thread_stats = stats_list

        # 注意：每个进程自己的连接已经在 query() 函数结束时断开

        total_query_end_time = time.time()
        total_query_time = total_query_end_time - total_query_start_time
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

    def save_results(self, thread_stats, total_query_time, qps):
        # --- 计算并打印最终结果 ---
        recall = {}
        remote_transfer_cnt = {}
        remote_transfer_latency = {}
        distance_cnt = {}
        for s in thread_stats:
            if not s:
                continue
            recall.update(s["recall"])
            remote_transfer_cnt.update(s["remote_transfer_cnt"])
            remote_transfer_latency.update(s["remote_transfer_latency"])
            distance_cnt.update(s["distance_cnt"])

        avg_recall = sum(recall.values()) / self.num_queries
        all_remote_transfer_cnt = sum(remote_transfer_cnt.values())
        avg_remote_transfer_cnt = all_remote_transfer_cnt / self.num_queries
        avg_remote_transfer_latency_query = sum(remote_transfer_latency.values()) / self.num_queries
        avg_remote_transfer_latency_trans = sum(remote_transfer_latency.values()) / all_remote_transfer_cnt if all_remote_transfer_cnt > 0 else 0.0
        all_distance_cnt = sum(distance_cnt.values())
        avg_distance_cnt = all_distance_cnt / self.num_queries

        print("-" * 30)
        print("Evaluation Results:")
        print(f"Recall@{self.k}: {avg_recall:.4f}")
        print(f"QPS: {qps:.4f}")
        print(f"Query Time Total: {total_query_time:.4f} seconds")
        print(f"Average Remote Transfer Count per Query: {avg_remote_transfer_cnt:.4f}")
        print(f"Average Remote Transfer Latency per Query: {avg_remote_transfer_latency_query:.4f}")
        print(f"Average Remote Transfer Latency per Transfer: {avg_remote_transfer_latency_trans:.4f}")
        print(f"Average Distance Computations per Query: {avg_distance_cnt:.4f}")
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
                "redundancy_level": cluster_info.get("redundancy_level") if cluster_info else self.redundancy,
                "redundancy_num": cluster_info.get("redundancy_num") if cluster_info else 0,
                "bandwidth": cluster_info.get("bandwidth") if cluster_info else 0,
                "is_distributed": cluster_info.get("is_distributed") if cluster_info else True,
                "is_reduce": cluster_info.get("is_reduce") if cluster_info else self.args.reduce,
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
                "redundancy_level": self.cluster.redundancy_level,
                "redundancy_num": self.cluster.redundancy_num,
                "bandwidth": self.cluster.bandwidth,
                "is_distributed": self.cluster.is_distributed,
                "is_reduce": self.cluster.is_reduce,
            }

            if self.cluster.partition_mode == "layer":
                cluster_params["layer_partition_mode"] = self.partition_mode
                cluster_params["layer_config"] = self.layer_config

        stats = {
            "hnsw_params": hnsw_params,
            "cluster_params": cluster_params,
            "num_queries": self.num_queries,
            "k": self.k,
            "qps": qps,
            f"avg_recall@{self.k}": avg_recall,
            "all_remote_transfer_cnt": all_remote_transfer_cnt,
            "avg_remote_transfer_cnt": avg_remote_transfer_cnt,
            "avg_remote_transfer_latency_query": avg_remote_transfer_latency_query,
            "avg_remote_transfer_latency_trans": avg_remote_transfer_latency_trans,
            "all_distance_cnt": all_distance_cnt,
            "avg_distance_cnt": avg_distance_cnt,
            "recall": recall,
            "remote_transfer_cnt": remote_transfer_cnt,
            "remote_transfer_latency": remote_transfer_latency,
            "distance_cnt": distance_cnt,
        }
        if self.use_cpp:
            results_dir = "./results/cpp"
        else:
            results_dir = "./results/python"
        os.makedirs(results_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        distributed_suffix = "-distributed" if self.args.distributed else ""
        cpu_suffix = f"-cpus{self.args.num_cpu}" if self.args.distributed and self.args.num_cpu != 0 else ""
        reduce_suffix = "-reduce" if self.args.reduce else ""
        layer_suffix = "-layer" if self.use_layer else ""

        if self.args.reduce:
            results_path = (f"{results_dir}/results-{self.data_name}-m{self.m}-"
                            f"efc{self.ef_construction}-efs{self.ef_search}-ll{self.limited_level}-"
                            f"node{self.node_num}-"
                            f"threads{self.num_query_threads}{distributed_suffix}{cpu_suffix}{reduce_suffix}.json")
        else:
            results_path = (f"{results_dir}/results-{self.data_name}-m{self.m}-"
                            f"efc{self.ef_construction}-efs{self.ef_search}-ll{self.limited_level}-"
                            f"node{self.node_num}-{self.partition_mode}-red{self.redundancy}-"
                            f"threads{self.num_query_threads}{distributed_suffix}{cpu_suffix}{layer_suffix}.json")

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

    def cleanup_workers(self):
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
                    "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
                    f"{self.user_name}@{ip}", remote_cmd
                ]
                proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                cleanup_procs.append((proc, ip))
            for proc, ip in cleanup_procs:
                try:
                    proc.communicate(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()

            # 等待端口完全释放，避免下一个任务启动时端口仍被占用
            print("Waiting for ports to be released after cleanup...")
            time.sleep(5)
            print("Worker processes cleaned up.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="HNSW Vector Retrieval")
    parser.add_argument("--data_name", type=str, default="sift", help="Name of the dataset")
    parser.add_argument("--data_path", type=str, default="/home/CONNECT/xchen740/workspace/vector-retrieve/dataset/sift/sift_base.fvecs", help="Path to base vectors file")
    parser.add_argument("--query_path", type=str, default="/home/CONNECT/xchen740/workspace/vector-retrieve/dataset/sift/sift_query.fvecs", help="Path to query vectors file")
    parser.add_argument("--groundtruth_path", type=str, default="/home/CONNECT/xchen740/workspace/vector-retrieve/dataset/sift/sift_groundtruth.ivecs", help="Path to groundtruth file")
    parser.add_argument("--graph_path", type=str, default=None, help="Path to save/load the graph")
    parser.add_argument("--cluster_path", type=str, default=None, help="Path to save/load the cluster")
    parser.add_argument("--k", type=int, default=10, help="Number of nearest neighbors to retrieve")
    parser.add_argument("--m", type=int, default=16, help="Max number of connections per element")
    parser.add_argument("--ef_construction", type=int, default=200, help="Size of the dynamic candidate list for construction")
    parser.add_argument("--ef_search", type=int, default=16, help="Size of the dynamic candidate list for search")
    parser.add_argument("--limited_level", type=int, default=5, help="Level limit for HNSW")
    parser.add_argument("--node_num", type=int, default=6, help="Number of cluster nodes")
    parser.add_argument("--partition_mode", type=str, default="k-means", help="Data partition mode across nodes")
    parser.add_argument("--layer", action='store_true', default=False, help="Enable layer-based partitioning")
    parser.add_argument("--layer_config", type=str, default="{}", help="Layer configuration in JSON format")
    parser.add_argument("--layer_replicate_flags", type=str, default="[]", help="Layer replicate flags in JSON list; default behavior is [false,false,true,...] based on limited_level")
    parser.add_argument("--redundancy", type=int, default=2, help="Redundancy level for data replication across nodes")
    parser.add_argument("--distributed", default=False, action='store_true', help="Enable distributed mode")
    parser.add_argument("--reduce", default=False, action='store_true', help="Enable reduce mode (independent graphs per node)")
    parser.add_argument("--num_construction_threads", type=int, default=64, help="Number of threads for parallel graph construction (0=auto detect)")
    parser.add_argument("--num_query_threads", type=int, default=32, help="Number of threads for querying (each thread does all queries)")
    parser.add_argument("--num_cpu", type=int, default=8, help="Limit CPU cores per node process (0=unlimited)")
    parser.add_argument("--save", action='store_true', default=False, help="Save the built graph to disk after construction")
    # 通信模式和 RDMA 参数
    parser.add_argument("--comm_mode", type=str, default="tcp", choices=["tcp", "rdma"], help="Communication mode: tcp or rdma")
    parser.add_argument("--ib_dev", type=str, default="", help="RDMA IB device name (auto-detect if empty)")
    parser.add_argument("--ib_port", type=int, default=1, help="RDMA IB port number")
    parser.add_argument("--gid_index", type=int, default=-1, help="RDMA GID index (-1 for auto-detect)")

    args = parser.parse_args(rest)

    runner = VectorRetrieveRunner(args, use_cpp)
    try:
        runner.setup_distributed_nodes()
        runner.prepare()
        runner.save_graph()  # 建图完成后立即保存
        thread_stats, total_query_time, qps = runner.run_queries()
        runner.save_results(thread_stats, total_query_time, qps)
    except QueryStallError:
        print("[RESTART] Cleaning up and restarting main.py...")
        runner.cleanup_workers()
        # 使用 os.execv 重新启动整个进程，保留原始命令行参数
        python = sys.executable
        os.execv(python, [python] + sys.argv)
    finally:
        runner.cleanup_workers()
