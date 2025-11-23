from dataclasses import dataclass, field
from typing import Set, Dict, Any

@dataclass
class Payload:
    """
    封装查询过程中的状态和数据载荷
    """
    results: Set[int] = field(default_factory=set)  # 当前找到的最近邻ID集合
    visited: Set[int] = field(default_factory=set)  # 已访问节点集合
    candidates: list = field(default_factory=list)  # 候选节点堆列表
    
    continue_search: bool = False                   # 是否是跳转来的继续搜索
    require_id: int = -1                            # 下一步需要的节点/向量ID
    start_level: int = -1                           # 继续跳转时的起始层级
    entry_point_id: int = -1                        # 继续跳转时的入口点
    
    @property
    def size_bits(self) -> int:
        """估算当前 Payload 的大小 (bits)，用于计算网络延迟"""
        # 1. Results 大小 (假设每个 ID 32 bits)
        results_size = len(self.results) * 32
        
        # 2. Visited 集合大小
        visited_size = len(self.visited) * 32
        
        # 3. Candidates 列表大小 (假设每个元素是 (dist, id) 元组，约 64 bits)
        candidates_size = len(self.candidates) * 64 
        
        # 4. 固定开销 (continue_search, require_id, start_level, entry_point_id 等)
        overhead = 256 
        
        return results_size + visited_size + candidates_size + overhead

