import os
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
import pickle
import json

from hnsw import HNSWGraph
from payload import Payload

class Cluster:
    """
    模拟一个分布式集群，管理所有节点和数据的分布。
    """
    def __init__(self, hnsw: HNSWGraph, node_num=4):
        self.hnsw = hnsw
        self.node_num = node_num 
        
        self.nodes = {} # cluster下属的所有节点
        self.vector_node_map = {} # 记录每个向量属于哪个节点

        self.bandwidth = 1000  # 模拟网络带宽，单位为 Mb/s
        
        for i in range(node_num):
            node = Node(node_id=i, cluster=self, local_hnsw=None)
            self.nodes[i] = node

        self.redundancy_num = 0 

        self.remote_transfer_cnt = {}
        self.remote_transfer_latency = {} # 记录每个查询的远程传输延迟, 单位为 ms

    def reset(self):
        self.remote_transfer_cnt = {}
        self.remote_transfer_latency = {}
        for node in self.nodes.values():
            node.reset()

    def partition(self, mode, redundancy=1):
        max_level = self.hnsw.max_level 
        vectors = self.hnsw.vectors
        neighbors = self.hnsw.neighbors

        local_vectors = defaultdict(lambda: defaultdict(dict))
        # local_neighbors = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

        match mode:
            case "round_robin":
                for i, (vector_id, vector) in enumerate(vectors.items()):
                    target_node_id = i % self.node_num
                    self.vector_node_map[vector_id] = target_node_id
                    local_vectors[target_node_id][vector_id] = vector

            case "k-means":
                vector_ids = list(vectors.keys())
                # 将向量转换为 numpy 数组供 sklearn 使用
                data_matrix = np.array([vectors[vid] for vid in vector_ids])

                # 执行聚类
                kmeans = KMeans(n_clusters=self.node_num, random_state=123, n_init=10)
                kmeans.fit(data_matrix)
                labels = kmeans.labels

                # 根据聚类结果分配“主节点”
                for idx, label in enumerate(labels):
                    vector_id = vector_ids[idx]
                    target_node_id = label  # 簇 ID 即为 Node ID

                    # 记录全局路由表：这个向量真正属于哪个节点
                    self.vector_node_map[vector_id] = target_node_id
                    # 将向量数据放入该节点的本地存储
                    local_vectors[target_node_id][vector_id] = vectors[vector_id]

        # 添加冗余
        if redundancy > 0:
            for r in range(redundancy):
                print(f"Processing redundancy pass {r+1}/{redundancy}...")
                
                # 记录本轮需要添加的数据，避免在遍历时修改字典导致的问题
                updates = defaultdict(dict)

                # 遍历整张图的所有连接关系
                # neighbors 结构: {level: {src_id: [neighbor_ids]}}
                for level, level_neighbors in neighbors.items():
                    for vector_id, neighbor_ids in level_neighbors.items():
                        for node_id in range(self.node_num):
                            if vector_id in local_vectors[node_id]:
                                for neighbor_id in neighbor_ids:
                                    # 如果邻居尚未在本地
                                    if neighbor_id not in local_vectors[node_id]:
                                        if neighbor_id not in updates[node_id]:
                                            if neighbor_id in vectors:
                                                updates[node_id][neighbor_id] = vectors[neighbor_id]                                      
                                                self.redundancy_num += 1
                
                # 应用本轮更新
                for node_id, new_vectors in updates.items():
                    local_vectors[node_id].update(new_vectors)

        for i in range(self.node_num):
            local_hnsw = HNSWGraph(self.hnsw.dim, self.hnsw.m, self.hnsw.ef_construction, 
                                   self.hnsw.ef_search, self.hnsw.limited_level, self.hnsw.seed)
            local_hnsw.vectors = local_vectors[i]
            local_hnsw.neighbors = neighbors

            local_hnsw.entry_point_id = self.hnsw.entry_point_id
            self.nodes[i].local_hnsw = local_hnsw


    def transfer_to_remote(self, query_id, query_vector, payload: Payload, k):

        print("Transferring payload to remote node...")
        self.remote_transfer_cnt[query_id] = self.remote_transfer_cnt.get(query_id, 0) + 1
        self.remote_transfer_latency[query_id] = self.remote_transfer_latency.get(query_id, 0) + payload.size_bits / (self.bandwidth * 1e3) # 单位为 ms
        print(f"Remote transfer count: {self.remote_transfer_cnt[query_id]}")

        require_id = payload.require_id
        target_node_id = self.vector_node_map[require_id]
        target_node = self.nodes[target_node_id]
        
        target_node.query(query_id, query_vector, payload, k)

    def save(self, filepath):
        """将整个 Cluster 对象保存到磁盘"""
        print(f"Saving cluster to {filepath}...")
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print("Cluster saved successfully.")

    @staticmethod
    def load(filepath):
        """从磁盘加载 Cluster 对象"""
        print(f"Loading cluster from {filepath}...")
        with open(filepath, 'rb') as f:
            cluster = pickle.load(f)
        print("Cluster loaded successfully.")
        return cluster

    def save_stats(self, filepath):
        """保存统计数据 (transfer count & latency) 以及配置参数 为 JSON 格式"""
        print(f"Saving stats to {filepath}...")

        # 收集 HNSW 的配置参数
        hnsw_params = {
            "dim": self.hnsw.dim,
            "m": self.hnsw.m,
            "ef_construction": self.hnsw.ef_construction,
            "ef_search": self.hnsw.ef_search,
            "limited_level": self.hnsw.limited_level,
            "max_level": self.hnsw.max_level
        }
        # 收集 Cluster 的配置参数
        cluster_params = {
            "node_num": self.node_num,
            "bandwidth": self.bandwidth,
            "redundancy_num": self.redundancy_num
        }
        num_queries = sum(node.num_queries for node in self.nodes.values())
        avg_remote_transfer_latency_query = sum(self.remote_transfer_latency.values()) / num_queries
        avg_remote_transfer_latency_trans = sum(self.remote_transfer_latency.values()) / sum(self.remote_transfer_cnt.values()) if sum(self.remote_transfer_cnt.values()) > 0 else 0.0
        stats = {
            "hnsw_params": hnsw_params,
            "cluster_params": cluster_params,
            "num_queries": num_queries,
            "avg_remote_transfer_latency_query": avg_remote_transfer_latency_query,
            "avg_remote_transfer_latency_trans": avg_remote_transfer_latency_trans,
            "remote_transfer_cnt": self.remote_transfer_cnt,
            "remote_transfer_latency": self.remote_transfer_latency,
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'w') as f:
            # indent=4 使输出格式化，方便阅读
            json.dump(stats, f, indent=4)
        print("Stats saved successfully.")

class Node:
    """
    模拟一个分布式节点, 它拥有自己的一部分数据
    """
    def __init__(self, node_id, cluster: Cluster, local_hnsw: HNSWGraph):
        self.node_id = node_id
        self.cluster = cluster 
        self.local_hnsw = local_hnsw
        self.num_queries = 0

    def reset(self):
        self.num_queries = 0

    def query(self, query_id, query_vector, payload: Payload, k=10):
        self.local_hnsw.query(query_vector, payload, k)
        
        if payload.continue_search is True:
            self.cluster.transfer_to_remote(query_id, query_vector, payload, k)
        else:
            self.num_queries += 1

        return payload.results
