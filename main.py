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
            # Escape backslashes in username (e.g., DOMAIN\user) for safe usage
            esc_user = self.user_name.replace("\\", "\\\\") if "\\" in self.user_name else self.user_name
            # Remote two-stage kill: find PIDs, TERM, wait, then KILL if still present
            remote_cmd = (
                "pids=$(pgrep -f node_start.py || true); "
                "if [ -n \"$pids\" ]; then kill -TERM $pids 2>/dev/null || true; sleep 1; "
                "pids2=$(pgrep -f node_start.py || true); if [ -n \"$pids2\" ]; then kill -KILL $pids2 2>/dev/null || true; fi; fi"
            )
            cmd = [
                "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
                f"{esc_user}@{ip}",
                remote_cmd
            ]
            proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            cleanup_procs.append(proc)
        # Wait for cleanup commands to finish (with timeout) and ensure remote processes are killed
        for proc in cleanup_procs:
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
        time.sleep(1)  # Short wait to ensure remote kills completed

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
                    f"cd {current_dir} && conda activate base && "
                    f"nohup {cpu_prefix}python -u node_start.py "
                    f"--node_id {i} --node_ips {self.default_ips} "
                    f"--m {self.args.m} --ef_construction {self.args.ef_construction} "
                    f"--ef_search {self.args.ef_search} --limited_level {self.args.limited_level} "
                    f"{distributed_flag} {reduce_flag} "
                    f"> {log_file} 2>&1 &"
                )

                cmd = ["ssh", f"{self.user_name}@{ip}", remote_cmd]
                print(f"  CMD: {' '.join(cmd)}")

                # Launch worker process in background (no output redirection needed, handled by nohup)
                proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                worker_processes.append(proc)

        # Give workers some time to start
        print(f"Waiting for {len(worker_processes)} worker nodes to start (including local node {self.host_id})...")
        time.sleep(5)

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
                thread_client = ClusterSDKClient(ip=config["ip"], port=config["port"])
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
            return build_time

        print(f"Building in {mode_label} via SDK...")
        start_time = time.time()
        thread_client = ClusterSDKClient(ip=self.sdk_config["ip"], port=self.sdk_config["port"])
        if not thread_client.connect():
            raise RuntimeError("Failed to connect SDK to local node for building")
        try:
            for idx, vector in tqdm(enumerate(data[:num_elements]), total=num_elements, desc="Building", unit="vec"):
                thread_client.add_vector(idx, vector)
            build_time = time.time() - start_time
            print(f"Building Time: {build_time:.4f} seconds.")
            return build_time
        finally:
            thread_client.disconnect()

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
                sdk_client = ClusterSDKClient(ip=self.sdk_config['ip'], port=self.sdk_config['port'])
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
                partition_client = ClusterSDKClient(ip=self.sdk_config['ip'], port=self.sdk_config['port'])
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
            self.cluster = Cluster(self.hnsw, self.node_num)

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

            self.hnsw = HNSWGraph(dim=dim, m=self.m, ef_construction=self.ef_construction, ef_search=self.ef_search, limited_level=self.limited_level)
            self.cluster = Cluster(self.hnsw, self.node_num)

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
                else:
                    print("Building cluster in reduce mode...")
                    start_time = time.time()
                    for idx, vector in tqdm(enumerate(data[:num_elements]), total=num_elements, desc="Building", unit="vec"):
                        self.cluster.add_vector(idx, vector)  # 轮询添加到各节点
                    build_time = time.time() - start_time
                    print(f"Building Machine Time: {build_time:.4f} seconds.")
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

        def query(worker_id, stats_list):
            local_recall = {}
            local_remote_transfer_cnt = {}
            local_remote_transfer_latency = {}
            local_distance_cnt = {}

            # 每个进程创建独立的 SDK 客户端实例。
            thread_sdk_client = None
            if self.args.distributed:
                thread_sdk_client = ClusterSDKClient(
                    ip=self.sdk_config['ip'],
                    port=self.sdk_config['port']
                )
                if self.use_layer:
                    if not thread_sdk_client.init_routes():
                        raise RuntimeError("Failed to initialize distributed query routes in SDK")

            try:
                # 打乱查询顺序，每个线程使用不同的随机顺序
                import random
                query_indices = list(range(self.num_queries))
                random.seed(worker_id)  # 每个线程使用不同的种子
                random.shuffle(query_indices)
                
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
                            results = self.cluster.query(i + worker_id * self.num_queries, query_vec, payload, self.k)
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

                    # print(f"\nQuery({worker_id}, {i}) Machine time: {query_time * 1000:.4f} ms")
                    # print(f"  - Network latency: {local_remote_transfer_latency[i]:.4f} ms")
                    # print(f"  - Remote transfers cnt: {local_remote_transfer_cnt[i]}")
                    # print(f"  - Average remote transfer latency: {avg_remote_transfer_latency:.4f} ms")
                    # print(f"  - Distance computations: {local_distance_cnt[i]}")
                    # print(f"  - Recall@{self.k}: {local_recall[i]:.4f}")
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

        if self.num_query_threads > 1:
            ctx = mp.get_context("fork")
            manager = ctx.Manager()
            stats_list = manager.list([None] * self.num_query_threads)
            procs = []
            for i in range(self.num_query_threads):
                p = ctx.Process(target=query, args=(i, stats_list))
                p.start()
                procs.append(p)
            for p in procs:
                p.join()
            thread_stats = list(stats_list)
        else:
            thread_stats = [None]
            query(0, thread_stats)

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

            sdk_client = ClusterSDKClient(ip=self.sdk_config['ip'], port=self.sdk_config['port'])
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
            info_client = ClusterSDKClient(ip=self.sdk_config['ip'], port=self.sdk_config['port'])
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

        results_path = (f"{results_dir}/results-{self.data_name}-m{self.m}-"
                        f"efc{self.ef_construction}-efs{self.ef_search}-ll{self.limited_level}-"
                        f"node{self.node_num}-{self.partition_mode}-red{self.redundancy}-"
                        f"threads{self.num_query_threads}{distributed_suffix}{cpu_suffix}{reduce_suffix}{layer_suffix}.json")

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
            for proc in self.worker_processes:
                if proc.poll() is None:  # Still running
                    proc.terminate()
            # Wait a bit for graceful shutdown
            time.sleep(1)
            for proc in self.worker_processes:
                if proc.poll() is None:  # Still running after terminate
                    proc.kill()
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
    parser.add_argument("--ef_construction", type=int, default=32, help="Size of the dynamic candidate list for construction")
    parser.add_argument("--ef_search", type=int, default=16, help="Size of the dynamic candidate list for search")
    parser.add_argument("--limited_level", type=int, default=5, help="Level limit for HNSW")
    parser.add_argument("--node_num", type=int, default=6, help="Number of cluster nodes")
    parser.add_argument("--partition_mode", type=str, default="k-means", help="Data partition mode across nodes")
    parser.add_argument("--layer", action='store_true', default=False, help="Enable layer-based partitioning")
    parser.add_argument("--layer_config", type=str, default="{}", help="Layer configuration in JSON format")
    parser.add_argument("--layer_replicate_flags", type=str, default="[]", help="Layer replicate flags in JSON list; default behavior is [false,false,true,...] based on limited_level")
    parser.add_argument("--redundancy", type=int, default=1, help="Redundancy level for data replication across nodes")
    parser.add_argument("--distributed", default=False, action='store_true', help="Enable distributed mode")
    parser.add_argument("--reduce", default=False, action='store_true', help="Enable reduce mode (independent graphs per node)")
    parser.add_argument("--num_construction_threads", type=int, default=64, help="Number of threads for parallel graph construction (0=auto detect)")
    parser.add_argument("--num_query_threads", type=int, default=64, help="Number of threads for querying (each thread does all queries)")
    parser.add_argument("--num_cpu", type=int, default=8, help="Limit CPU cores per node process (0=unlimited)")
    parser.add_argument("--save", action='store_true', default=False, help="Save the built graph to disk after construction")

    args = parser.parse_args(rest)

    runner = VectorRetrieveRunner(args, use_cpp)
    runner.setup_distributed_nodes()
    runner.prepare()
    runner.save_graph()  # 建图完成后立即保存
    thread_stats, total_query_time, qps = runner.run_queries()
    runner.save_results(thread_stats, total_query_time, qps)
    runner.cleanup_workers()
