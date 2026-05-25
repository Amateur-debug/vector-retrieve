#pragma once

#include <vector>
#include <queue>
#include <utility>
#include <functional>
#include <cstdint>
#include <algorithm>

using namespace std;

// 序列化用的标量字段打包结构体（POD），一次 memcpy 读写所有标量字段。
struct PayloadHeader {
    bool continue_search;
    int origin_node_id;
    int require_id;
    int start_level;
    int entry_point_id;
    int remote_transfer_cnt;
    uint64_t remote_transfer_data;
    double send_timestamp;
    int distance_cnt;
    int pull_cnt;
    uint64_t pull_data;
    int backtrack_cnt;
    double query_start_ms;
    double total_ms;
    double search_ms;
    double serialize_ms;
    double deserialize_ms;
    double send_ms;
    double recv_ms;
    double queue_ms;
    double result_ms;
};

// 通用 PQHack：直接访问 priority_queue 的底层 vector 容器（protected 成员 c）。
// 避免通过 pop/push 逐个操作的 O(n log n) 开销，实现 O(n) 的序列化/反序列化。
template <typename PQ>
struct PQAccessor : public PQ {
    static const auto& get_container(const PQ& pq) {
        return reinterpret_cast<const PQAccessor&>(pq).c;
    }
    static auto& get_container_mut(PQ& pq) {
        return reinterpret_cast<PQAccessor&>(pq).c;
    }
};

template <typename T>
struct Payload {
    bool continue_search = false;   // 标记本次搜索是否为跨节点跳转后的续搜状态
    int origin_node_id = -1;        // 链式转发场景下的原始发起节点编号（由最终处理节点将结果回传至此节点）
    int require_id = -1;            // 下一步所需访问的远程节点或向量 ID（用于路由跳转）
    int start_level = -1;           // 续搜时的起始层级编号
    int entry_point_id = -1;        // 续搜时使用的入口点 vid
    int remote_transfer_cnt = 0;    // 跨节点远程传输次数累计
    uint64_t remote_transfer_data = 0;   // 跨节点传输的累计数据量（单位：比特）
    double send_timestamp = 0.0;    // 最近一次发送消息的时间戳（单位：毫秒，用于计算单跳延迟）
    int distance_cnt = 0;           // 本次搜索过程中计算距离的总次数

    // global-pull 模式：按 vid 远程拉取向量的次数与累计数据量（单位：比特）。
    // 与 remote_transfer_cnt/remote_transfer_data 区分：后者统计 forward 模式下的 payload 转发，
    // pull 仅用于拉取单个向量数据（dim * sizeof(T) 字节/次）。两种模式在 JSON 中分开呈现。
    int pull_cnt = 0;
    uint64_t pull_data = 0;

    // 折返检测：最近 max_track_num 次 transfer 的目标 node_id 滑动窗口（FIFO）。
    // 每次 transfer_to_remote 选定 target_node_id 后，若该 node_id 已在窗口内，
    // 视为一次折返（backtrack），否则入窗；超过 max_track_num 则丢弃最旧元素。
    // 该窗口跨节点随 payload 序列化传输，保持单条查询的全链路视角。
    std::vector<int> recent_nodes;

    // 折返次数累计：target_node_id 命中 recent_nodes 的次数。
    int backtrack_cnt = 0;

    double query_start_ms = 0.0;
    double total_ms = 0.0;
    double search_ms = 0.0;
    double serialize_ms = 0.0;
    double deserialize_ms = 0.0;
    double send_ms = 0.0;
    double recv_ms = 0.0;
    double queue_ms = 0.0;
    double result_ms = 0.0;

    // 跨节点序列化传输的已访问向量 vid 列表（外部 ID），表示距离已计算。
    // 选用 vector<int> 而非 set：序列化走单次 bulk memcpy，成本最低。
    // 节点内部本地向量的 visited 判重仍由 VisitedListPool（基于 internal_id）独立管理；
    // 远程邻居的 O(1) 去重由 search_layer_query 内部的局部 remote_visited_set 承担。
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

    // 将标量字段打包为 PayloadHeader（用于高效序列化）
    PayloadHeader to_header() const {
        return {continue_search, origin_node_id, require_id, start_level,
                entry_point_id, remote_transfer_cnt, remote_transfer_data,
                send_timestamp, distance_cnt, pull_cnt, pull_data, backtrack_cnt,
                query_start_ms, total_ms, search_ms, serialize_ms, deserialize_ms,
                send_ms, recv_ms, queue_ms, result_ms};
    }

    // 从 PayloadHeader 恢复标量字段
    void from_header(const PayloadHeader& h) {
        continue_search = h.continue_search;
        origin_node_id = h.origin_node_id;
        require_id = h.require_id;
        start_level = h.start_level;
        entry_point_id = h.entry_point_id;
        remote_transfer_cnt = h.remote_transfer_cnt;
        remote_transfer_data = h.remote_transfer_data;
        send_timestamp = h.send_timestamp;
        distance_cnt = h.distance_cnt;
        pull_cnt = h.pull_cnt;
        pull_data = h.pull_data;
        backtrack_cnt = h.backtrack_cnt;
        query_start_ms = h.query_start_ms;
        total_ms = h.total_ms;
        search_ms = h.search_ms;
        serialize_ms = h.serialize_ms;
        deserialize_ms = h.deserialize_ms;
        send_ms = h.send_ms;
        recv_ms = h.recv_ms;
        queue_ms = h.queue_ms;
        result_ms = h.result_ms;
    }

    void clear() {
        visited_vids.clear();
        pending_vids.clear();
        recent_nodes.clear();
        candidates = decltype(candidates)();
        results = decltype(results)();
        send_timestamp = 0.0;
    }

    // 预分配 visited_vids 的向量容量，减少搜索过程中 push_back 触发的动态扩容开销。
    void reserve_visited(size_t capacity) {
        visited_vids.reserve(capacity);
    }

    // 估算当前 Payload 的序列化大小（单位：比特），用于模拟网络传输延迟。
    // 统计 visited_vids、pending_vids、candidates、results 四个容器的内存占用，
    // 加上固定开销（协议头、标量字段等），得到近似的传输数据量。
    uint64_t size_bits() const {

        uint64_t visited_size = visited_vids.size() * sizeof(int) * 8;

        uint64_t pending_size = pending_vids.size() * sizeof(int) * 8;

        // 注：recent_nodes 和 backtrack_cnt 为折返统计信息，不计入模拟传输数据量

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

        const auto& container = PQAccessor<decltype(results)>::get_container(results);

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
