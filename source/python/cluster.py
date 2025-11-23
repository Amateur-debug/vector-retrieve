import os
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
import pickle
import fcntl

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

        self.partition_mode = None
        self.outer_ring_max_iterations = 0

        self.redundancy_level = 0
        self.redundancy_num = 0 

        self.remote_transfer_cnt = {}
        self.remote_transfer_latency = {} # 记录每个查询的远程传输延迟, 单位为 ms

    def reset(self):
        self.remote_transfer_cnt = {}
        self.remote_transfer_latency = {}
        for node in self.nodes.values():
            node.reset()

    def partition(self, mode, redundancy=1):

        self.partition_mode = mode
        self.redundancy_level = redundancy
        self.redundancy_num = 0

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
                labels = kmeans.labels_

                # 根据聚类结果分配“主节点”
                for idx, label in enumerate(labels):
                    vector_id = vector_ids[idx]
                    target_node_id = label  # 簇 ID 即为 Node ID

                    # 记录全局路由表：这个向量真正属于哪个节点
                    self.vector_node_map[vector_id] = target_node_id
                    # 将向量数据放入该节点的本地存储
                    local_vectors[target_node_id][vector_id] = vectors[vector_id]
            case "outer_ring":
                # 1. 先用 k-means 分配
                vector_ids = list(vectors.keys())
                # 将向量转换为 numpy 数组供 sklearn 使用
                data_matrix = np.array([vectors[vid] for vid in vector_ids])

                # 执行聚类
                kmeans = KMeans(n_clusters=self.node_num, random_state=123, n_init=10)
                kmeans.fit(data_matrix)
                labels = kmeans.labels_

                # 根据聚类结果分配“主节点”
                for idx, label in enumerate(labels):
                    vector_id = vector_ids[idx]
                    target_node_id = label  # 簇 ID 即为 Node ID

                    # 记录全局路由表：这个向量真正属于哪个节点
                    self.vector_node_map[vector_id] = target_node_id
                    # 将向量数据放入该节点的本地存储
                    local_vectors[target_node_id][vector_id] = vectors[vector_id]
                
                # 2. 挑选边界向量，重新分配（迭代，支持提前停止）
                removed_cnt_before = 0
                for i in range(self.outer_ring_max_iterations):
                    removed_cnt = self.outer_ring(local_vectors)
                    removed_add_ratio = (removed_cnt - removed_cnt_before) / removed_cnt_before if removed_cnt_before > 0 else 1.0
                    # 提前停止条件
                    if removed_cnt == 0 or removed_add_ratio < 0.001:
                        break
                    removed_cnt_before = removed_cnt

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
            local_hnsw.max_level = max_level
            local_hnsw.vectors = local_vectors[i]
            local_hnsw.neighbors = neighbors

            local_hnsw.entry_point_id = self.hnsw.entry_point_id
            self.nodes[i].local_hnsw = local_hnsw
    
    def outer_ring(self, local_vectors):
        # 1. 找出所有有跨节点邻居的候选 boundary vectors（及其跨节点邻居数）
        neighbors = self.hnsw.neighbors
        boundary_candidates = defaultdict(list)  # node_id -> [(vid, cross_cnt), ...]
        for node_id in range(self.node_num):
            for vid in local_vectors[node_id].keys():
                cross_cnt = 0
                for neighbor_id in neighbors[0][vid]:
                    if self.vector_node_map.get(neighbor_id, node_id) != node_id:
                        cross_cnt += 1
                if cross_cnt > 0:
                    boundary_candidates[node_id].append((vid, cross_cnt))
        
        # 2. 从候选中挑选尽量少的外圈向量（按跨节点邻居数降序），可调参数：
        outer_fraction = 0.02   # 每个节点保留的比例（可调，越小越少）
        max_outer = 50          # 每个节点保留的上限（可调）
        outer_ring = defaultdict(set)
        for node_id in range(self.node_num):
            candidates = boundary_candidates.get(node_id, [])
            if not candidates:
                continue
            # 按跨节点邻居数降序
            candidates.sort(key=lambda x: -x[1])
            # 选择数量：至少1个（若存在候选），或按比例取，且不超过 max_outer
            choose_n = max(1, int(len(local_vectors[node_id]) * outer_fraction))
            choose_n = min(choose_n, max_outer, len(candidates))
            for vid, _ in candidates[:choose_n]:
                outer_ring[node_id].add(vid)
                                        
        # 3. 从每个 node 的本地存储中删除未被选为外圈的 boundary vectors
        removed = set()
        for node_id in range(self.node_num):
            for vid, _ in boundary_candidates.get(node_id, []):
                if vid not in outer_ring[node_id]:
                    local_vectors[node_id].pop(vid, None)
                    removed.add(vid)

        # 4. 重新分配local_vectors
        for vid in list(removed):
            # 统计该向量在各节点的邻居数量（只计入当前 local_vectors 中仍存在的邻居）
            counts = defaultdict(int)
            for neighbor_id in neighbors[0].get(vid, ()):
                neighbor_node = self.vector_node_map.get(neighbor_id)
                if neighbor_id in local_vectors.get(neighbor_node, {}):
                    counts[neighbor_node] += 1
            if counts:
                # 选择邻居数最多的节点；如并列，选择当前存量最小的那个作为 tiebreak
                best_node = max(counts.items(), key=lambda x: (x[1], -len(local_vectors[x[0]])))[0]
            else:
                # 没有可用邻居，放到当前最小负载节点
                best_node = min(range(self.node_num), key=lambda nid: len(local_vectors[nid]))
            
            # 将向量数据放回到选定节点
            vector = self.hnsw.vectors.get(vid)
            local_vectors[best_node][vid] = vector
            self.vector_node_map[vid] = best_node

        return len(removed)

    def transfer_to_remote(self, query_id, query_vector, payload: Payload, k):

        self.remote_transfer_cnt[query_id] = self.remote_transfer_cnt.get(query_id, 0) + 1
        self.remote_transfer_latency[query_id] = self.remote_transfer_latency.get(query_id, 0) + payload.size_bits / (self.bandwidth * 1e3) # 单位为 ms

        require_id = payload.require_id
        target_node_id = self.vector_node_map[require_id]
        target_node = self.nodes[target_node_id]
        
        target_node.query(query_id, query_vector, payload, k)

    def save(self, filepath):
        """将整个 Cluster 对象保存到磁盘"""
        print(f"Saving cluster to {filepath}...")
        with open(filepath, 'wb') as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print(f"{filepath} is locked by another process, skipping save.")
                return
            pickle.dump(self, f)
            fcntl.flock(f, fcntl.LOCK_UN)
            print("Cluster saved successfully.")

    @staticmethod
    def load(filepath):
        """从磁盘加载 Cluster 对象"""
        print(f"Loading cluster from {filepath}...")
        with open(filepath, 'rb') as f:
            cluster = pickle.load(f)
        print("Cluster loaded successfully.")
        return cluster

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
