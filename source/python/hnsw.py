import sys
import math
import random
import heapq
from collections import defaultdict
import pickle
import fcntl

from payload import Payload

class HNSWGraph:
    def __init__(self, dim, m=16, ef_construction=128, ef_search=64, limited_level=5, seed=123):
        self.dim = dim
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search  
        self.limited_level = limited_level
        self.max_level = 0
        if isinstance(seed, random.Random):
            self.seed = seed
        else:
            self.seed = random.Random(seed)

        self.entry_point_id = None
        self.vectors = {}
        self.neighbors = self.neighbors = defaultdict(lambda: defaultdict(set))

    def add_vector(self, vector_id, vector):
        """向图中添加一个新向量"""
        # 生成新节点的层级
        r = self.seed.random()
        if r == 0.0:
            r = sys.float_info.min
        level = int(-math.log(r) * (1 / math.log(self.m)))
        if self.limited_level is not None:
            level = min(level, self.limited_level)
        
        # 存储向量和节点信息
        self.vectors[vector_id] = vector
        for i in range(level + 1):
            self.neighbors[i][vector_id] = set()

        # 如果是第一个节点
        if self.entry_point_id is None:
            self.entry_point_id = vector_id
            self.max_level = level
            return
        
        entry_point_id = self.entry_point_id

        for l in range(self.max_level, -1, -1):
            payload = Payload()
            payload.entry_point_id = entry_point_id
            self.search_layer_ef(vector, l, self.ef_construction, payload)
            entry_point_id = payload.results[0][1]

            # 使用启发式算法从候选者中挑选邻居
            selected_neighbors = self.select_neighbors_heuristic(payload.results)

            # 为新节点建立连接
            self.neighbors[l][vector_id] = selected_neighbors

            # 为被选中的邻居建立反向连接
            for neighbor_id in selected_neighbors:
                neighbor_connections = self.neighbors[l][neighbor_id]
                 # 如果邻居的连接数超限
                if len(neighbor_connections) + 1 > self.m:
                    # 将超出的邻居的现有连接和新节点一起放入候选池
                    new_candidates_for_neighbor = [(self.distance_id(neighbor_id, nid), nid) for nid in neighbor_connections]
                    
                    # 重新用启发式算法为这个邻居挑选最佳连接
                    new_neighbors = self.select_neighbors_heuristic(new_candidates_for_neighbor)
                    self.neighbors[l][neighbor_id] = new_neighbors
                else:
                    # 如果未超限，直接添加
                    neighbor_connections.add(vector_id)
            
        # 如果新节点的层级更高，更新图的全局入口点和最高层
        if level > self.max_level:
            self.max_level = level
            self.entry_point_id = vector_id

    def select_neighbors_heuristic(self, candidates):
        """
        使用启发式算法从候选者中选择邻居
        Args:
            candidates: 候选邻居列表, 按距离从小到大排序, 格式为 [(距离, 节点ID), ...]
            level: 当前层级
        Returns:
            一个包含最终选定邻居ID的集合, 最大为 self.m 个
        """
        results = set()
        
        for e_dist, e_id in candidates:
            if len(results) >= self.m:
                break
            
            is_good_neighbor = True

            for r_id in results:
                er_dist = self.distance_id(e_id, r_id)
                if er_dist < e_dist:
                    is_good_neighbor = False
                    break
            
            if is_good_neighbor:
                results.add(e_id)

        return results

    def query(self, query_vector, payload: Payload, k):
        """在图中搜索与查询向量最相似的 k 个向量"""

        if not payload.continue_search:
            payload.start_level = self.max_level
            payload.entry_point_id = self.entry_point_id

        for l in range(payload.start_level, -1, -1):
            payload.start_level = l

            self.search_layer_ef(query_vector, l, self.ef_search, payload)
            if payload.continue_search:
                return payload.results
    
            payload.entry_point_id = payload.results[0][1]
            payload.visited = set()
            payload.candidates = []

            if l != 0:
                payload.results = []
            
        payload.results = {res_id for dist, res_id in payload.results[:k]}

        return payload.results
            
    def search_layer_ef(self, query_vector, level, ef, payload: Payload):
        """在指定层使用候选列表进行搜索 (由 ef 控制), 返回最近的ef个邻居, 格式为 [(距离, 节点ID), ...]"""

        entry_point_id = payload.entry_point_id
        payload.entry_point_id = -1
        
        if not payload.continue_search:
            if self.vectors.get(entry_point_id) is None:
                payload.continue_search = True
                payload.require_id = entry_point_id
                payload.start_level = level
                payload.entry_point_id = entry_point_id
                return payload.results

            dist = self.distance(query_vector, self.vectors.get(entry_point_id))
            payload.visited = {entry_point_id}
            payload.candidates = [(dist, entry_point_id)]
            payload.results = [(-dist, entry_point_id)]
        else:
            if entry_point_id != -1:
                dist = self.distance(query_vector, self.vectors.get(entry_point_id))
                payload.visited = {entry_point_id}
                payload.candidates = [(dist, entry_point_id)]
                payload.results = [(-dist, entry_point_id)]
                
        while payload.candidates:
            candidate_dist, candidate_id = heapq.heappop(payload.candidates)

            # 如果当前候选者比结果中最差的还差，就不用继续了
            if candidate_dist > -payload.results[0][0]:
                break

            for neighbor_id in self.neighbors[level][candidate_id]:
                if neighbor_id not in payload.visited:
                    neighbor_vector = self.vectors.get(neighbor_id)

                    if neighbor_vector is None:
                        payload.continue_search = True
                        payload.require_id = neighbor_id
                        payload.start_level = level
                        heapq.heappush(payload.candidates, (candidate_dist, candidate_id))

                        return payload.results
                    
                    payload.visited.add(neighbor_id)
                    neighbor_dist = self.distance(query_vector, neighbor_vector)
                    
                    if len(payload.results) < ef or neighbor_dist < -payload.results[0][0]:
                        heapq.heappush(payload.results, (-neighbor_dist, neighbor_id))
                        if len(payload.results) > ef:
                            heapq.heappop(payload.results)
                        heapq.heappush(payload.candidates, (neighbor_dist, neighbor_id))
        
        payload.continue_search = False
        payload.results = sorted([(-d, nid) for d, nid in payload.results])

        return payload.results

    def distance_id(self, v1_id, v2_id):
        """根据id计算两个向量之间的欧氏距离"""
        v1 = self.vectors[v1_id]
        v2 = self.vectors[v2_id]
    
        return self.distance(v1, v2)

    def distance(self, v1, v2):
        """计算两个向量之间的欧氏距离"""
    
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))
    
    # --- 新增以下两个方法用于支持 pickle ---
    def __getstate__(self):
        """序列化时调用：将 defaultdict 转换为普通 dict"""
        state = self.__dict__.copy()
        # 将 neighbors 转换为普通字典，因为 lambda 无法被 pickle
        # neighbors 结构: {level: {node_id: set(neighbors)}}
        state['neighbors'] = {k: dict(v) for k, v in self.neighbors.items()}
        # 如果 seed 是 random.Random 对象，pickle 通常能处理，但为了保险也可以只存状态
        return state

    def __setstate__(self, state):
        """反序列化时调用：恢复 defaultdict 结构"""
        self.__dict__.update(state)
        # 恢复 neighbors 为 defaultdict(lambda: defaultdict(set))
        raw_neighbors = state['neighbors']
        self.neighbors = defaultdict(lambda: defaultdict(set))
        for level, nodes in raw_neighbors.items():
            for node_id, neighbors_set in nodes.items():
                self.neighbors[level][node_id] = neighbors_set

    def save(self, filepath):
        """将 HNSWGraph 对象保存到磁盘"""
        print(f"Saving HNSW to {filepath}...")
        with open(filepath, 'wb') as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print(f"{filepath} is locked by another process, skipping save.")
                return
            pickle.dump(self, f)
            fcntl.flock(f, fcntl.LOCK_UN)
            print("HNSW saved successfully.")

    @staticmethod
    def load(filepath):
        """从磁盘加载 HNSWGraph 对象"""
        print(f"Loading HNSW from {filepath}...")
        with open(filepath, 'rb') as f:
            hnsw = pickle.load(f)
        print("HNSW index loaded successfully.")
        return hnsw