#pragma once

#include <vector>
#include <queue>
#include <utility>
#include <functional>
#include <cstdint>
#include <algorithm>

using namespace std;

template <typename T>
struct Payload {
    bool continue_search = false;   // 是否是跳转来的继续搜索
    int origin_node_id = -1;        // 链式转发时的入口节点（由最终节点回传结果）
    int require_id = -1;            // 下一步需要的节点/向量ID 
    int start_level = -1;           // 继续跳转时的起始层级
    int entry_point_id = -1;        // 继续跳转时的入口点 
    int remote_transfer_cnt = 0;
    double remote_transfer_latency = 0.0;     // 总延迟(毫秒)
    double send_timestamp = 0.0;    // 发送时间戳(毫秒)
    int distance_cnt = 0;        // 计算距离次数

    // 跨节点序列化传输使用的 visited vid 列表（外部 ID）
    // 搜索内部的 visited 逻辑由 VisitedListPool 管理（基于 internal_id）
    std::vector<int> visited_vids;

    // 待处理的 vid 列表（逐个 pop 处理）
    // 本地遍历邻居时收集到的所有未访问远程 vid，transfer 后远程节点逐个处理：
    // 在本地的计算距离加入堆并 pop，不在本地的继续 transfer，直到列表为空
    std::vector<int> pending_vids;

    // candidates as min-heap: (distance, id) with smallest distance on top
    priority_queue<pair<T,int>, vector<pair<T,int>>, greater<pair<T,int>>> candidates;
    
    // results as max-heap: (distance, id) with largest distance on top
    priority_queue<pair<T,int>> results;

    void clear() {
        visited_vids.clear();
        pending_vids.clear();
        candidates = decltype(candidates)();
        results = decltype(results)();
        send_timestamp = 0.0;
    }

    // 预留 visited_vids 容量，减少动态扩容开销
    void reserve_visited(size_t capacity) {
        visited_vids.reserve(capacity);
    }

    // 估算当前 Payload 的大小 (bits)，用于计算网络延迟
    uint64_t size_bits() const {

        uint64_t visited_size = visited_vids.size() * sizeof(int) * 8;

        uint64_t pending_size = pending_vids.size() * sizeof(int) * 8;

        uint64_t candidates_size = candidates.size() * (sizeof(T) + sizeof(int)) * 8;

        uint64_t results_size = results.size() * (sizeof(T) + sizeof(int)) * 8;

        uint64_t overhead = 256;

        return visited_size + pending_size + candidates_size + results_size + overhead;
    }

    // 快速获取 results 中距离最小的单个 vid（不拷贝堆）
    // 用于取下一层入口点，避免 topk_ids(1) 的整堆拷贝开销
    // 通过访问 priority_queue 底层容器实现 O(N) 无拷贝扫描
    int top1_id() const
    {
        if (results.empty())
            return -1;

        // hack: 访问 priority_queue 底层 vector 容器（标准布局，c 是第一个成员）
        // 参考: priority_queue 在各主流标准库中底层容器名为 c
        struct PQHack : public decltype(results) {
            static const auto& get_container(const decltype(results)& pq) {
                return reinterpret_cast<const PQHack&>(pq).c;
            }
        };
        const auto& container = PQHack::get_container(results);

        auto best = container[0];
        for (size_t i = 1; i < container.size(); i++)
        {
            if (container[i].first < best.first)
                best = container[i];
        }
        return best.second;
    }

    std::vector<int> topk_ids(size_t k) const
    {
        std::vector<int> ids;
        ids.reserve(k);
        auto pq = results; // 复制以保留原队列

        while (pq.size() > k) pq.pop();

        std::vector<pair<T, int>> items;
        items.reserve(pq.size());
        while (!pq.empty())
        {
            items.push_back(pq.top());
            pq.pop();
        }
        sort(items.begin(), items.end(), [](const pair<T, int> &a, const pair<T, int> &b)
             { return a.first < b.first; });

        size_t limit = std::min(k, items.size());
        for (size_t i = 0; i < limit; i++)
        {
            ids.push_back(items[i].second);
        }
        return ids;
    }
};