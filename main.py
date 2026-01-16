import os
import time
import argparse
import numpy as np
import json
import subprocess
from tqdm import tqdm
import threading
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
    parser.add_argument("--partition_mode", type=str, default="round_robin", help="Data partition mode across nodes")
    parser.add_argument("--redundancy", type=int, default=1, help="Redundancy level for data replication across nodes")
    parser.add_argument("--distributed", default=False, action='store_true', help="Enable distributed mode")
    parser.add_argument("--reduce", default=False, action='store_true', help="Enable reduce mode (independent graphs per node)")
    parser.add_argument("--num_construction_threads", type=int, default=32, help="Number of threads for parallel graph construction (0=auto detect)")
    parser.add_argument("--num_query_threads", type=int, default=1, help="Number of threads for querying (each thread does all queries)")
    parser.add_argument("--num_workers", type=int, default=2048, help="Number of worker threads for each node")

    args = parser.parse_args(rest)

    num_construction_threads = args.num_construction_threads
    num_query_threads = args.num_query_threads
    num_workers = args.num_workers

    user_name = r"CONNECT\xchen740"
    default_ips = "192.168.200.204,192.168.200.205,192.168.200.206,192.168.200.207,192.168.200.209"
    host_ip = "192.168.200.206"
    host_id = 2
    node_ips = []
    node_ips = default_ips.split(',')
    if args.distributed:
        node_num = len(node_ips)
    else:
        node_num = args.node_num

    if args.distributed:
        print(f"Distributed mode enabled. Node IPs: {node_ips}")
        
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
        for i in range(len(node_ips)):
            ip = node_ips[i]
            if ip == host_ip:
                host_id = i
                print(f"Host machine identified as node {host_id} with IP {host_ip}.")
                # We already attempted local cleanup; continue to next
                continue
            # Escape backslashes in username (e.g., DOMAIN\user) for safe usage
            esc_user = user_name.replace("\\", "\\\\") if "\\" in user_name else user_name
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
        
        # Launch remote workers
        current_dir = os.getcwd()
        worker_processes = []
        for i in range(len(node_ips)):
            ip = node_ips[i]
            if ip == host_ip:
                continue  # Skip the host machine
            print(f"Launching worker node {i} on {user_name}@{ip}...")
            log_file = f"{current_dir}/worker_{i}.log"
            
            # Use subprocess to launch worker in background
            distributed_flag = "--distributed" if args.distributed else ""
            reduce_flag = "--reduce" if args.reduce else ""
            
            # Build remote command as a single string
            remote_cmd = (
                f"cd {current_dir} && conda activate base && "
                f"nohup python -u node_start.py "
                f"--node_id {i} --node_ips {default_ips} "
                f"--m {args.m} --ef_construction {args.ef_construction} "
                f"--ef_search {args.ef_search} --limited_level {args.limited_level} "
                f"--num_workers {num_workers} "
                f"{distributed_flag} {reduce_flag} "
                f"> {log_file} 2>&1 &"
            )
            
            cmd = ["ssh", f"{user_name}@{ip}", remote_cmd]
            print(f"  CMD: {' '.join(cmd)}")
            
            # Launch worker process in background (no output redirection needed, handled by nohup)
            proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            worker_processes.append(proc)
        
        # Give workers some time to start
        print(f"Waiting for {len(worker_processes)} worker nodes to start...")
        time.sleep(5)

    data_name = args.data_name
    data_path = args.data_path
    query_path = args.query_path
    groundtruth_path = args.groundtruth_path

    graph_path = args.graph_path
    cluster_path = args.cluster_path
    
    k = args.k
    m = args.m
    ef_construction = args.ef_construction
    ef_search = args.ef_search
    limited_level = args.limited_level
    partition_mode = args.partition_mode
    redundancy = args.redundancy

    if cluster_path is not None and os.path.exists(cluster_path):
        print("Loading cluster from disk...")
        cluster = Cluster.load(cluster_path)
        cluster.reset()
    elif graph_path is not None and os.path.exists(graph_path):
        print("Loading HNSW graph from disk...")
        hnsw = HNSWGraph.load(graph_path)
        hnsw.ef_search = ef_search
        print("Partitioning data across cluster nodes...")
        cluster = Cluster(hnsw, node_num)

        if args.distributed:
            for i, ip in enumerate(node_ips):
                is_remote = (i != host_id)
                port = 19000 + i
                cluster.set_node_info(i, ip, port, is_remote)
            cluster.nodes[host_id].num_worker = int(num_workers)
            cluster.start_node_server(host_id)

        start_time = time.time()
        cluster.partition(mode=partition_mode, redundancy=redundancy)
        partition_time = time.time() - start_time
        print(f"Partitioning Time: {partition_time:.4f} seconds.")
    else:
        print("Loading data...")
        data = read_fvecs(data_path)
        
        print(f"Data shape: {data.shape}")
    
        dim = data.shape[1]
        max_num_elements = data.shape[0]
        num_elements = max_num_elements

        hnsw = HNSWGraph(dim=dim, m=m, ef_construction=ef_construction, ef_search=ef_search, limited_level=limited_level)
        cluster = Cluster(hnsw, node_num)

        if args.distributed:
            for i, ip in enumerate(node_ips):
                is_remote = (i != host_id)
                port = 19000 + i
                cluster.set_node_info(i, ip, port, is_remote)
            cluster.nodes[host_id].num_worker = int(num_workers)
            cluster.start_node_server(host_id)

        if args.reduce:
            # Reduce模式：先创建cluster，轮询添加向量到各节点独立建图
            cluster.is_reduce = True
            num_threads = num_construction_threads
            
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
                        cluster.add_vector(vid, vec)
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
                start_time = time.time()
                for idx, vector in tqdm(enumerate(data[:num_elements]), total=num_elements, desc="Building", unit="vec"):
                    cluster.add_vector(idx, vector)  # 轮询添加到各节点
                build_time = time.time() - start_time
                print(f"Building Machine Time: {build_time:.4f} seconds.")
        else:
            # 传统模式：先建统一图，再分区
            num_threads = num_construction_threads
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
                        hnsw.add_vector(vid, vec)
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
                    hnsw.add_vector(idx, vector)
                build_time = time.time() - start_time
                print(f"Building Machine Time: {build_time:.4f} seconds.")
            print("Partitioning data across cluster nodes...")

            start_time = time.time()
            cluster.partition(mode=partition_mode, redundancy=redundancy)
            partition_time = time.time() - start_time
            print(f"Partitioning Time: {partition_time:.4f} seconds.")

    queries = read_fvecs(query_path)
    groundtruth = read_ivecs(groundtruth_path)
    print(f"Queries shape: {queries.shape}")
    print(f"Groundtruth shape: {groundtruth.shape}")
    max_num_queries = queries.shape[0]
    num_queries = max_num_queries

    print("Performing queries...")
    thread_stats = [None] * num_query_threads

    def query(worker_id):
        local_recall = {}
        local_remote_transfer_cnt = {}
        local_remote_transfer_latency = {}
        local_distance_cnt = {}

        for i, query in enumerate(queries[:num_queries]):
            start_time = time.time()

            payload = Payload()
            if cluster.is_reduce:
                # Reduce模式：向所有节点查询并合并结果
                results = cluster.query(i + worker_id * num_queries, query, payload, k)
                results = set(results)
                local_remote_transfer_cnt[i] = 0
                local_remote_transfer_latency[i] = 0.0
                avg_remote_transfer_latency = 0.0
            else:
                # 传统模式：使用payload进行跨节点搜索
                results = cluster.nodes[host_id].query(i + worker_id * num_queries, query, payload, k=k)
                results = set(results)
                local_remote_transfer_cnt[i] = payload.remote_transfer_cnt
                local_remote_transfer_latency[i] = payload.remote_transfer_latency
                avg_remote_transfer_latency = (payload.remote_transfer_latency / payload.remote_transfer_cnt) if payload.remote_transfer_cnt > 0 else 0.0

            local_distance_cnt[i] = payload.distance_cnt

            query_time = time.time() - start_time

            # --- 召回率计算 ---
            # 从 groundtruth 数据中获取真实的最近邻 ID 集合
            true_ids = set(groundtruth[i][:k])
            # 计算两个集合的交集，即正确找到的邻居数量
            correct = len(results.intersection(true_ids))
            local_recall[i] = correct / k

            # print(f"\nQuery({worker_id}, {i}) Machine time: {query_time * 1000:.4f} ms")
            # print(f"  - Network latency: {local_remote_transfer_latency[i]:.4f} ms")
            # print(f"  - Remote transfers cnt: {local_remote_transfer_cnt[i]}")
            # print(f"  - Average remote transfer latency: {avg_remote_transfer_latency:.4f} ms")
            # print(f"  - Distance computations: {local_distance_cnt[i]}")
            # print(f"  - Recall@{k}: {local_recall[i]:.4f}")

        thread_stats[worker_id] = {
            "recall": local_recall,
            "remote_transfer_cnt": local_remote_transfer_cnt,
            "remote_transfer_latency": local_remote_transfer_latency,
            "distance_cnt": local_distance_cnt,
        }

    total_query_start_time = time.time()

    if num_query_threads > 1:
        threads = []
        for i in range(num_query_threads):
            t = threading.Thread(target=query, args=(i,))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        # 多线程模式下，主线程不做统计合并，结果由各线程各自输出
    else:
        query(0)
    
    total_query_end_time = time.time()
    total_query_time = total_query_end_time - total_query_start_time
    qps = num_queries * num_query_threads / total_query_time if total_query_time > 0 else 0.0

    # --- 保存cluster或graph状态 ---
    # print("-" * 30)
    # if cluster_path is None:
    #     print("Saving cluster state to disk...")
    #     save_dir = "./save"
    #     os.makedirs(save_dir, exist_ok=True)
    #     cluster_path = (f"{save_dir}/cluster-{data_name}-m{m}-"
    #                     f"efc{ef_construction}-efs{ef_search}-ll{limited_level}-"
    #                     f"node{node_num}-{partition_mode}-red{redundancy}.pkl")
    #     cluster.save(cluster_path)
    if graph_path is None:
        print("Saving graph state to disk...")
        save_dir = "./save"
        os.makedirs(save_dir, exist_ok=True)
        graph_path = (f"{save_dir}/hnsw-{data_name}-m{m}-"
                      f"efc{ef_construction}-ll{limited_level}.json")
        hnsw.save(graph_path)
    # print("-" * 30)

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

    avg_recall = sum(recall.values()) / num_queries
    all_remote_transfer_cnt = sum(remote_transfer_cnt.values())
    avg_remote_transfer_cnt = all_remote_transfer_cnt / num_queries
    avg_remote_transfer_latency_query = sum(remote_transfer_latency.values()) / num_queries
    avg_remote_transfer_latency_trans = sum(remote_transfer_latency.values()) / all_remote_transfer_cnt if all_remote_transfer_cnt > 0 else 0.0
    all_distance_cnt = sum(distance_cnt.values())
    avg_distance_cnt = all_distance_cnt / num_queries

    print("-" * 30)
    print("Evaluation Results:")
    print(f"Recall@{k}: {avg_recall:.4f}")
    print(f"QPS: {qps:.4f}")
    print(f"Query Time Total: {total_query_time:.4f} seconds")
    print(f"Average Remote Transfer Count per Query: {avg_remote_transfer_cnt:.4f}")
    print(f"Average Remote Transfer Latency per Query: {avg_remote_transfer_latency_query:.4f}")
    print(f"Average Remote Transfer Latency per Transfer: {avg_remote_transfer_latency_trans:.4f}")
    print(f"Average Distance Computations per Query: {avg_distance_cnt:.4f}")
    print("-" * 30)

    # --- 保存所有结果 ---
    print("-" * 30)
    hnsw_params = {
        "data_name": data_name,
        "dim": hnsw.dim,
        "m": hnsw.m,
        "ef_construction": hnsw.ef_construction,
        "ef_search": hnsw.ef_search,
        "limited_level": hnsw.limited_level,
        "max_level": hnsw.max_level
    }
    cluster_params = {
        "node_num": cluster.node_num,
        "partition_mode": cluster.partition_mode,
        "redundancy_level": cluster.redundancy_level,
        "redundancy_num": cluster.redundancy_num,
        "bandwidth": cluster.bandwidth,
        "is_distributed": cluster.is_distributed,
        "is_reduce": cluster.is_reduce,
    }
    stats = {
        "hnsw_params": hnsw_params,
        "cluster_params": cluster_params,
        "num_queries": num_queries,
        "k": k,
        "qps": qps,
        f"avg_recall@{k}": avg_recall,
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
    if use_cpp:
        results_dir = "./results/cpp"
    else:
        results_dir = "./results/python"
    os.makedirs(results_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    distributed_suffix = "-distributed" if args.distributed else ""
    reduce_suffix = "-reduce" if args.reduce else ""
    results_path = (f"{results_dir}/results-{data_name}-m{m}-"
                    f"efc{ef_construction}-efs{ef_search}-ll{limited_level}-"
                    f"node{node_num}-{partition_mode}-red{redundancy}-"
                    f"threads{num_query_threads}{reduce_suffix}{distributed_suffix}.json")

    with open(results_path, 'w') as f:
        json.dump(stats, f, indent=4)

    # Print all saved paths
    print("\n" + "=" * 50)
    print(f"Results: {os.path.abspath(results_path)}")
    if graph_path:
        print(f"Graph: {os.path.abspath(graph_path)}")
    if cluster_path:
        print(f"Cluster: {os.path.abspath(cluster_path)}")
    print("=" * 50 + "\n")

    # Clean up worker processes if in distributed mode
    if args.distributed and 'worker_processes' in locals():
        print("\nCleaning up worker processes...")
        for proc in worker_processes:
            if proc.poll() is None:  # Still running
                proc.terminate()
        # Wait a bit for graceful shutdown
        time.sleep(1)
        for proc in worker_processes:
            if proc.poll() is None:  # Still running after terminate
                proc.kill()
        print("Worker processes cleaned up.")
