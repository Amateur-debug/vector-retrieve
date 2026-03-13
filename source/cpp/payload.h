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
    bool continue_search = false;   // 标记本次搜索是否为跨节点跳转后的续搜状态
    int origin_node_id = -1;        // 链式转发场景下的原始发起节点编号（由最终处理节点将结果回传至此节点）
    int require_id = -1;            // 下一步所需访问的远程节点或向量 ID（用于路由跳转）
    int start_level = -1;           // 续搜时的起始层级编号
    int entry_point_id = -1;        // 续搜时使用的入口点 vid
    int remote_transfer_cnt = 0;    // 跨节点远程传输次数累计
    double remote_transfer_latency = 0.0;     // 跨节点传输的累计延迟（单位：毫秒）
    double send_timestamp = 0.0;    // 最近一次发送消息的时间戳（单位：毫秒，用于计算单跳延迟）
    int distance_cnt = 0;           // 本次搜索过程中计算距离的总次数

    // 跨节点序列化传输使用的已访问向量 vid 列表（外部 ID）。
    // 在分布式搜索中，各节点通过该列表同步已访问状态，避免重复计算距离。
    // 注意：节点内部的 visited 判重由 VisitedListPool（基于 internal_id）独立管理，
    // 该列表仅在跨节点边界传输时使用。
    std::vector<int> visited_vids;

    // 待处理的远程向量 vid 队列（按顺序逐个弹出处理）。
    // 本地节点在遍历邻居时，会将所有未访问的远程 vid 收集至此列表，
    // 然后通过 transfer 发送至远程节点。远程节点依次处理每个 vid：
    //   - 若该 vid 在当前节点本地，则计算距离并加入候选堆，然后弹出继续下一个；
    //   - 若该 vid 仍不在本地，则继续 transfer 到目标节点，直到列表清空。
    std::vector<int> pending_vids;

    // 候选集（最小堆）：按距离升序排列，堆顶为距离最近的候选向量。
    // 搜索过程中不断从此堆弹出最近候选进行邻居展开。
    priority_queue<pair<T,int>, vector<pair<T,int>>, greater<pair<T,int>>> candidates;

    // 结果集（最大堆）：按距离降序排列，堆顶为距离最远的结果向量。
    // 堆大小维持在 ef 以内，堆顶距离作为搜索终止的 lower_bound 阈值。
    priority_queue<pair<T,int>> results;

    void clear() {
        visited_vids.clear();
        pending_vids.clear();
        candidates = decltype(candidates)();
        results = decltype(results)();
        send_timestamp = 0.0;
    }

    // 预分配 visited_vids 的内存容量，减少搜索过程中 push_back 触发的动态扩容开销。
    void reserve_visited(size_t capacity) {
        visited_vids.reserve(capacity);
    }

    // 估算当前 Payload 的序列化大小（单位：比特），用于模拟网络传输延迟。
    // 统计 visited_vids、pending_vids、candidates、results 四个容器的内存占用，
    // 加上固定开销（协议头、标量字段等），得到近似的传输数据量。
    uint64_t size_bits() const {

        uint64_t visited_size = visited_vids.size() * sizeof(int) * 8;

        uint64_t pending_size = pending_vids.size() * sizeof(int) * 8;

        uint64_t candidates_size = candidates.size() * (sizeof(T) + sizeof(int)) * 8;

        uint64_t results_size = results.size() * (sizeof(T) + sizeof(int)) * 8;

        uint64_t overhead = 256;

        return visited_size + pending_size + candidates_size + results_size + overhead;
    }

    // 从 results 最大堆中获取距离最小的单个 vid（不拷贝堆）。
    // 用途：取下一层搜索的入口点。相比 topk_ids(1) 需要拷贝整个堆，
    // 此方法通过直接访问 priority_queue 底层 vector 容器进行 O(N) 线性扫描，
    // 避免了堆拷贝开销，在频繁调用的场景下显著提升性能。
    int top1_id() const
    {
        if (results.empty())
            return -1;

        // 技巧：通过继承 hack 访问 priority_queue 的底层 vector 容器。
        // priority_queue 在 libstdc++/libc++/MSVC STL 中底层容器均命名为 c，
        // 作为 protected 成员，可通过派生类的 reinterpret_cast 直接访问。
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
        auto pq = results; // 拷贝副本以保留原始结果集不被修改

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