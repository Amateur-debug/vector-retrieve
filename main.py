import os
import time
import argparse
import numpy as np
import json

import sys
sys.setrecursionlimit(5000) 

from hnsw import HNSWGraph
from node import Cluster, Node
from payload import Payload

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
    parser.add_argument("--data_name", type=str, default="siftsmall", help="Name of the dataset")
    parser.add_argument("--data_path", type=str, default="/home/CONNECT/xchen740/workspace/vector-retrieve/dataset/siftsmall/siftsmall_base.fvecs", help="Path to base vectors file")
    parser.add_argument("--query_path", type=str, default="/home/CONNECT/xchen740/workspace/vector-retrieve/dataset/siftsmall/siftsmall_query.fvecs", help="Path to query vectors file")
    parser.add_argument("--groundtruth_path", type=str, default="/home/CONNECT/xchen740/workspace/vector-retrieve/dataset/siftsmall/siftsmall_groundtruth.ivecs", help="Path to groundtruth file")
    parser.add_argument("--graph_path", type=str, default=None, help="Path to save/load the graph")
    parser.add_argument("--cluster_path", type=str, default=None, help="Path to save/load the cluster")
    parser.add_argument("--k", type=int, default=10, help="Number of nearest neighbors to retrieve")
    parser.add_argument("--m", type=int, default=8, help="Max number of connections per element")
    parser.add_argument("--ef_construction", type=int, default=16, help="Size of the dynamic candidate list for construction")
    parser.add_argument("--ef_search", type=int, default=8, help="Size of the dynamic candidate list for search")
    parser.add_argument("--limited_level", type=int, default=5, help="Level limit for HNSW")
    parser.add_argument("--node_num", type=int, default=4, help="Number of cluster nodes")
    parser.add_argument("--partition_mode", type=str, default="round_robin", help="Data partition mode across nodes")
    parser.add_argument("--redundancy", type=int, default=1, help="Redundancy level for data replication across nodes")

    args = parser.parse_args()

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
    node_num = args.node_num
    partition_mode = args.partition_mode
    redundancy = args.redundancy

    if cluster_path is not None and os.path.exists(cluster_path):
        print("Loading cluster from disk...")
        cluster = Cluster.load(cluster_path)
        cluster.reset()
    elif graph_path is not None and os.path.exists(graph_path):
        print("Loading HNSW graph from disk...")
        hnsw = HNSWGraph.load(graph_path)
    else:
        print("Loading data...")
        data = read_fvecs(data_path)
        
        print(f"Data shape: {data.shape}")
    
        dim = data.shape[1]
        max_num_elements = data.shape[0]
        num_elements = max_num_elements

        print("Building HNSW index...")
        hnsw = HNSWGraph(dim=dim, m=m, ef_construction=ef_construction, ef_search=ef_search, limited_level=limited_level)
        start_time = time.time()
        for idx, vector in enumerate(data[:num_elements]):
            hnsw.add_vector(idx, vector) 
        build_time = time.time() - start_time
        print(f"Building Machine Time: {build_time:.4f} seconds.")

        print("Partitioning data across cluster nodes...")
        cluster = Cluster(hnsw, node_num)
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
    recall = {}

    for i, query in enumerate(queries[:num_queries]):
        start_time = time.time()
        
        payload = Payload()
        
        results = cluster.nodes[0].query(i, query, payload, k=k)
        query_time = time.time() - start_time

        # --- 召回率计算 ---
        # 从 groundtruth 数据中获取真实的最近邻 ID 集合
        true_ids = set(groundtruth[i][:k])
        # 计算两个集合的交集，即正确找到的邻居数量
        correct = len(results.intersection(true_ids))
        recall[i] = correct / k

        print(f"\nQuery({i}) Machine time: {query_time * 1000:.4f} ms")
        print(f"Recall@{k}({i}): {recall[i]:.4f}")

    # --- 保存cluster或graph状态 ---
    print("-" * 30)
    if cluster_path is None:
        print("Saving cluster state to disk...")
        save_dir = "./save"
        os.makedirs(save_dir, exist_ok=True)
        cluster_path = f"{save_dir}/cluster-{data_name}-node{node_num}-{partition_mode}.pkl"
        cluster.save(cluster_path)
    if graph_path is None:
        print("Saving graph state to disk...")
        save_dir = "./save"
        os.makedirs(save_dir, exist_ok=True)
        graph_path = f"{save_dir}/hnsw-{data_name}-m{m}-efc{ef_construction}-efs{ef_search}-ll{limited_level}.pkl"
        hnsw.save(graph_path)
    print("-" * 30)

    # --- 计算并打印最终结果 ---
    avg_recall = sum(recall.values()) / (num_queries)
    all_remote_transfer_cnt = sum(cluster.remote_transfer_cnt.values())
    avg_remote_transfer_cnt = all_remote_transfer_cnt / num_queries
    avg_remote_transfer_latency_query = sum(cluster.remote_transfer_latency.values()) / num_queries
    avg_remote_transfer_latency_trans = sum(cluster.remote_transfer_latency.values()) / all_remote_transfer_cnt if all_remote_transfer_cnt > 0 else 0.0

    print("-" * 30)
    print("Evaluation Results:")
    print(f"Recall@{k}: {avg_recall:.4f}")
    print(f"Average Remote Transfer Count per Query: {avg_remote_transfer_cnt:.4f}")
    print(f"Average Remote Transfer Latency per Query: {avg_remote_transfer_latency_query:.4f}")
    print(f"Average Remote Transfer Latency per Transfer: {avg_remote_transfer_latency_trans:.4f}")
    print("-" * 30)

    # --- 保存所有结果 ---
    print("-" * 30)
    hnsw_params = {
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
    }
    stats = {
        "hnsw_params": hnsw_params,
        "cluster_params": cluster_params,
        "num_queries": num_queries,
        "k": k,
        f"avg_recall@{k}": avg_recall,
        "all_remote_transfer_cnt": all_remote_transfer_cnt,
        "avg_remote_transfer_latency_query": avg_remote_transfer_latency_query,
        "avg_remote_transfer_latency_trans": avg_remote_transfer_latency_trans,
        "recall": recall,
        "remote_transfer_cnt": cluster.remote_transfer_cnt,
        "remote_transfer_latency": cluster.remote_transfer_latency,
    }

    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_path = (f"{results_dir}/results-{data_name}-m{m}-"
                    f"efc{ef_construction}-efs{ef_search}-ll{limited_level}-"
                    f"node{node_num}-{partition_mode}-red{redundancy}.json")

    with open(results_path, 'w') as f:
        json.dump(stats, f, indent=4)
