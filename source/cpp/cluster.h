#pragma once

#include <sys/file.h>
#include <poll.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <sstream>
#include <map>
#include <mutex>
#include <deque>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#if defined(__GLIBC__)
#include <malloc.h>
#endif

#include <condition_variable>
#include <cstring>
#include <thread>
#include <atomic>
#include <future>
#include <poll.h>
#include <queue>

#include "distance.h"
#include "hnsw.h"
#include "log.h"
#include "payload.h"
#include "tcp_transport.h"

#include "json.hpp"
using json = nlohmann::json;

using namespace std;

// 注意：TcpConnectionPool 已迁移到 tcp_transport.h 中统一管理
// g_connection_pool 全局变量已被 Cluster::transport 成员变量替代

// 消息处理线程池：固定大小 + 双端任务队列。
// submit() 把任务追加到队尾（FIFO，普通查询）；
// submit_priority() 把任务插到队首（用于 MSG_SEARCH_RESULT/MSG_SHARD_RESULT 这类
// 处理极轻但需要尽快返回给 SDK 的响应消息，避免被前面慢的 SEARCH 任务堵几十秒）。
class MessageThreadPool
{
public:
    explicit MessageThreadPool(size_t pool_size)
        : stop_(false)
    {
        for (size_t i = 0; i < pool_size; i++)
        {
            workers_.emplace_back([this] { worker_loop(); });
        }
    }

    ~MessageThreadPool()
    {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto &w : workers_)
        {
            if (w.joinable())
                w.join();
        }
    }

    // 提交普通任务到队尾（FIFO）
    void submit(std::function<void()> task)
    {
        {
            std::lock_guard<std::mutex> lk(mu_);
            tasks_.push_back(std::move(task));
        }
        cv_.notify_one();
    }

    // 提交优先任务到队首：跳过排队的普通任务，下一个空闲 worker 立即处理。
    // 注意：仍受限于 worker 数量；如果所有 worker 都在执行长任务（如 transfer_to_remote
    // 的 send_raw 阻塞），priority 任务也得等其中一个 worker 先 idle。
    void submit_priority(std::function<void()> task)
    {
        {
            std::lock_guard<std::mutex> lk(mu_);
            tasks_.push_front(std::move(task));
        }
        cv_.notify_one();
    }

    MessageThreadPool(const MessageThreadPool &) = delete;
    MessageThreadPool &operator=(const MessageThreadPool &) = delete;

private:
    void worker_loop()
    {
        while (true)
        {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [this] { return stop_ || !tasks_.empty(); });
                if (stop_ && tasks_.empty())
                    return;
                task = std::move(tasks_.front());
                tasks_.pop_front();
            }
            task();
        }
    }

    bool stop_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<std::function<void()>> tasks_;
    std::vector<std::thread> workers_;
};

template <typename T>
class Cluster; // 前向声明

inline double steady_now_ms()
{
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// 异步 SDK 查询完成队列：每个 SDK 连接一个，支持 reader/writer 双线程模型
template <typename T>
struct SDKCompletionQueue {
    using DistT = DistTOf<T>;
    struct Result {
        int query_id;
        std::vector<int> results;
        Payload<DistT> payload;
    };
    std::queue<Result> completed;
    std::mutex mu;
    std::condition_variable cv;
    bool closed = false;

    void push(Result&& r) {
        {
            std::lock_guard<std::mutex> lk(mu);
            completed.push(std::move(r));
        }
        cv.notify_one();
    }

    bool pop(Result& out) {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [this] { return !completed.empty() || closed; });
        if (completed.empty()) return false;
        out = std::move(completed.front());
        completed.pop();
        return true;
    }

    void close() {
        {
            std::lock_guard<std::mutex> lk(mu);
            closed = true;
        }
        cv.notify_all();
    }
};

// ==================== Shard 模式异步查询累积器 ====================
// 收集所有节点的查询结果，全部到齐后合并取 top-k 并推入完成队列。
template <typename T>
struct ReduceQueryAccumulator {
    using DistT = DistTOf<T>;
    int query_id;
    int k;
    int total_nodes;                    // 期望收到的结果数（所有节点）
    std::atomic<int> received_count{0};
    std::mutex mu;
    std::vector<std::pair<DistT, int>> all_results;
    Payload<DistT> merged_payload;
    double query_start_ms = 0.0;
    std::shared_ptr<SDKCompletionQueue<T>> completion_queue;
    std::function<void()> cleanup_fn;   // 完成后从 map 中移除自己

    void add_node_results(const std::vector<std::pair<DistT, int>>& results, const Payload<DistT>& payload) {
        int count;
        {
            std::lock_guard<std::mutex> lk(mu);
            all_results.insert(all_results.end(), results.begin(), results.end());
            merged_payload.distance_cnt += payload.distance_cnt;
            merged_payload.remote_transfer_cnt += payload.remote_transfer_cnt;
            merged_payload.remote_transfer_data += payload.remote_transfer_data;
            merged_payload.backtrack_cnt += payload.backtrack_cnt;
            merged_payload.search_ms += payload.search_ms;
            merged_payload.serialize_ms += payload.serialize_ms;
            merged_payload.deserialize_ms += payload.deserialize_ms;
            merged_payload.send_ms += payload.send_ms;
            merged_payload.recv_ms += payload.recv_ms;
            merged_payload.queue_ms += payload.queue_ms;
            merged_payload.result_ms += payload.result_ms;
            count = received_count.fetch_add(1) + 1;
        }
        if (count == total_nodes) {
            // 所有节点结果到齐，合并取 top-k
            size_t limit = std::min((size_t)k, all_results.size());
            std::partial_sort(all_results.begin(), all_results.begin() + limit, all_results.end());
            std::vector<int> final_ids;
            final_ids.reserve(limit);
            for (size_t i = 0; i < limit; i++)
                final_ids.push_back(all_results[i].second);
            if (query_start_ms != 0.0)
                merged_payload.total_ms = steady_now_ms() - query_start_ms;
            completion_queue->push({query_id, std::move(final_ids), std::move(merged_payload)});
            if (cleanup_fn) cleanup_fn();
        }
    }
};

enum MessageType
{
    MSG_SEARCH = 0,
    MSG_INIT_GRAPH = 1,
    MSG_ADD_VECTOR = 2,
    MSG_QUERY = 3,
    MSG_SEARCH_RESULT = 4,  // 异步查询结果回传
    MSG_PARTITION = 5,
    MSG_LOAD_CLUSTER = 6,
    MSG_LOAD_HNSW = 7,
    MSG_GET_INFORMATION = 8,
    MSG_SAVE_CLUSTER = 9,
    MSG_SAVE_HNSW = 10,
    MSG_QUERY_LOCAL = 11,  // Shard模式下只执行本地查询（避免递归）
    MSG_ADD_VECTOR_LOCAL = 12,  // Shard模式下直接添加到本地（避免递归路由）
    MSG_SET_HNSW_PARAMS = 13,   // 运行时更新 HNSW 参数（如 ef_search）
    MSG_QUERY_SHARD = 14,      // 异步 shard 查询请求（远程节点执行本地查询后通过 MSG_SHARD_RESULT 回传）
    MSG_SHARD_RESULT = 15,     // shard 查询结果异步回传（携带 distance+vid 对，用于合并排序）
    MSG_FETCH_VECTOR = 16,     // global-pull 模式：按 vid 同步拉取远程节点的向量数据
    MSG_RELEASE_FULL_HNSW = 17
};

// 序列化/反序列化辅助缓冲区（用于 HNSW 图的网络传输、分布式同步和持久化存储）
class Buffer
{
public:
    std::vector<char> data;
    size_t read_pos = 0;

    void write(const void *ptr, size_t size)
    {
        const char *cptr = static_cast<const char *>(ptr);
        data.insert(data.end(), cptr, cptr + size);
    }

    template <typename U>
    void write_val(const U &val)
    {
        write(&val, sizeof(U));
    }

    template <typename U>
    void write_vec(const std::vector<U> &vec)
    {
        int size = (int)vec.size();
        write_val(size);
        if (size > 0)
        {
            write(vec.data(), size * sizeof(U));
        }
    }

    void write_string(const std::string &str)
    {
        uint64_t size = (uint64_t)str.size();
        write_val(size);
        if (size > 0)
        {
            write(str.data(), size);
        }
    }

    void read(void *ptr, size_t size)
    {
        if (read_pos + size > data.size())
            throw std::runtime_error("Buffer underflow");
        std::memcpy(ptr, data.data() + read_pos, size);
        read_pos += size;
    }

    template <typename U>
    U read_val()
    {
        U val;
        read(&val, sizeof(U));
        return val;
    }

    template <typename U>
    std::vector<U> read_vec()
    {
        int size = read_val<int>();
        std::vector<U> vec(size);
        if (size > 0)
        {
            read(vec.data(), size * sizeof(U));
        }
        return vec;
    }

    std::string read_string()
    {
        uint64_t size = read_val<uint64_t>();
        if (size == 0)
            return std::string();
        if (read_pos + size > data.size())
            throw std::runtime_error("Buffer underflow");
        std::string str(data.data() + read_pos, data.data() + read_pos + size);
        read_pos += size;
        return str;
    }
};

    // --- 序列化/反序列化辅助工具函数 ---

// 前向声明：HNSWGraph 定义于 hnsw.h，避免循环依赖。
template <typename T> class HNSWGraph;

// serialize_payload / deserialize_payload 的 graph 参数语义：
//   - 非空（节点侧，本地持有 HNSWGraph）：
//       serialize 把堆内 iid 编码转成 wire 全局 vid（get_external_label 指针解引用，~1ns/项）。
//       deserialize 把 wire vid 批量转回本节点 iid 编码（label_lookup_.find，命中→iid；未命中→-(vid+1)）。
//   - 空（SDK 客户端，无本地 HNSW）：
//       serialize 跳过转换（客户端侧堆本来就空），
//       deserialize 将堆项统一置为 -(vid+1) 编码（客户端不再用来搜索，仅作为响应保留）。
template <typename T>
void serialize_payload(Buffer &buf, Payload<DistTOf<T>> &payload, const HNSWGraph<T> *graph)
{
    double timing_start_ms = steady_now_ms();
    // 标量字段打包：先写占位 header，完成后回填本次 serialize_ms。
    size_t header_pos = buf.data.size();
    PayloadHeader header = payload.to_header();
    buf.write_val(header);

    // visited_vids (vector<int> of vids)：单次 bulk memcpy
    buf.write_vec(payload.visited_vids);
    buf.write_vec(payload.pending_vids);
    buf.write_vec(payload.recent_nodes);

    // 候选/结果堆：thread_local tmp 做 iid→vid 转换 + 单次 bulk memcpy
    using DistT = DistTOf<T>;
    auto write_pq = [&](const auto& vec) {
        int n = (int)vec.size();
        buf.write_val(n);
        if (n == 0) return;
        static thread_local std::vector<std::pair<DistT, int>> tmp;
        tmp.clear();
        tmp.reserve(n);
        if (graph)
        {
            for (const auto& item : vec)
            {
                int key = item.second;
                int vid = (key >= 0) ? graph->get_external_label(key) : -(key + 1);
                tmp.emplace_back(item.first, vid);
            }
        }
        else
        {
            // 客户端：堆本应为空；即便非空，视为已是 vid 直接透传
            for (const auto& item : vec) tmp.emplace_back(item);
        }
        buf.write(tmp.data(), (size_t)n * sizeof(std::pair<DistT, int>));
        if (tmp.capacity() > 4096)
            std::vector<std::pair<DistT, int>>().swap(tmp);
    };
    write_pq(PQAccessor<decltype(payload.candidates)>::get_container(payload.candidates));
    write_pq(PQAccessor<decltype(payload.results)>::get_container(payload.results));

    payload.serialize_ms += steady_now_ms() - timing_start_ms;
    header = payload.to_header();
    std::memcpy(buf.data.data() + header_pos, &header, sizeof(PayloadHeader));
}

template <typename T>
void deserialize_payload(Buffer &buf, Payload<DistTOf<T>> &payload, const HNSWGraph<T> *graph)
{
    double timing_start_ms = steady_now_ms();
    auto timing_guard = [&]() { payload.deserialize_ms += steady_now_ms() - timing_start_ms; };
    struct Guard { decltype(timing_guard) &fn; ~Guard() { fn(); } } guard{timing_guard};
    // 标量字段打包：一次 memcpy 读出所有标量
    PayloadHeader header = buf.read_val<PayloadHeader>();
    payload.from_header(header);

    // visited_vids：单次 bulk read
    payload.visited_vids = buf.read_vec<int>();
    payload.pending_vids = buf.read_vec<int>();
    payload.recent_nodes = buf.read_vec<int>();

    // 候选/结果堆：bulk read 到 vec 拿到 (dist, vid)，原地做 vid→iid 编码 + make_heap。
    // 用 software prefetch 掩盖 label_lookup_ bucket 查找的 cache miss。
    using DistT = DistTOf<T>;
    auto read_pq = [&](auto& vec, auto cmp) {
        int n = buf.read_val<int>();
        vec.resize(n);
        if (n == 0) return;
        buf.read(vec.data(), (size_t)n * sizeof(std::pair<DistT, int>));
        if (graph)
        {
            const auto& lut = graph->label_lookup();
            constexpr size_t PF_DIST = 4;
            for (size_t i = 0; i < (size_t)n; ++i)
            {
                if (i + PF_DIST < (size_t)n)
                {
                    size_t bk = lut.bucket(vec[i + PF_DIST].second);
                    auto bit = lut.begin(bk);
                    if (bit != lut.end(bk))
                        __builtin_prefetch(&(*bit), 0, 0);
                }
                int vid = vec[i].second;
                auto it = lut.find(vid);
                vec[i].second = (it != lut.end()) ? it->second : -(vid + 1);
            }
        }
        else
        {
            // 客户端：全部按远程编码存，客户端不再用来搜索
            for (auto& item : vec) item.second = -(item.second + 1);
        }
        std::make_heap(vec.begin(), vec.end(), cmp);
    };
    read_pq(PQAccessor<decltype(payload.candidates)>::get_container_mut(payload.candidates),
            std::greater<std::pair<DistT, int>>());
    read_pq(PQAccessor<decltype(payload.results)>::get_container_mut(payload.results),
            std::less<std::pair<DistT, int>>());
}

template <typename T>
void serialize_hnsw(Buffer &buf, const HNSWGraph<T> *hnsw)
{
    size_t cur_count = hnsw->cur_element_count_.load();

    // 写入 HNSW 图的基本参数头（dim、m、ef_construction 等）
    buf.write_val(hnsw->dim);
    buf.write_val(hnsw->m);
    buf.write_val(hnsw->ef_construction);
    buf.write_val(hnsw->ef_search);
    buf.write_val(hnsw->limited_level);
    buf.write_val(hnsw->max_level);
    buf.write_val(hnsw->entry_point_id);
    buf.write_val(hnsw->seed);
    buf.write_val((uint64_t)hnsw->max_elements_);
    buf.write_val((uint64_t)cur_count);
    buf.write_val((uint64_t)hnsw->size_per_element_);
    buf.write_val((uint64_t)hnsw->links_level0_size_);
    buf.write_val((uint64_t)hnsw->data_offset_);
    buf.write_val((uint64_t)hnsw->label_offset_);
    buf.write_val((uint64_t)hnsw->size_links_per_element_);
    buf.write_val(hnsw->max_m_);
    buf.write_val(hnsw->max_m0_);
    buf.write_val(hnsw->level_mult_);

    LOG_DEBUG_STREAM("Serializing HNSW: " << cur_count << " elements.");

    // 写入第 0 层连续内存块（邻居链表 + 向量数据 + label）
    buf.write(hnsw->data_level0_, cur_count * hnsw->size_per_element_);

    // 写入各节点的高层邻居链表数据
    for (size_t i = 0; i < cur_count; i++)
    {
        int link_level = hnsw->element_levels_[i];
        buf.write_val(link_level);
        if (link_level > 0)
        {
            size_t link_size = hnsw->size_links_per_element_ * link_level;
            buf.write(hnsw->link_lists_[i], link_size);
        }
    }
}

template <typename T>
HNSWGraph<T> *deserialize_hnsw(Buffer &buf)
{
    int dim = buf.read_val<int>();
    int m = buf.read_val<int>();
    int ef_c = buf.read_val<int>();
    int ef_s = buf.read_val<int>();
    int lim = buf.read_val<int>();
    int max_l = buf.read_val<int>();
    int ep = buf.read_val<int>();
    unsigned int seed = buf.read_val<unsigned int>();
    uint64_t max_elements = buf.read_val<uint64_t>();
    uint64_t cur_count = buf.read_val<uint64_t>();
    uint64_t size_per_element = buf.read_val<uint64_t>();
    uint64_t links_level0_size = buf.read_val<uint64_t>();
    uint64_t data_offset = buf.read_val<uint64_t>();
    uint64_t label_offset = buf.read_val<uint64_t>();
    uint64_t size_links_per_element = buf.read_val<uint64_t>();
    int max_m = buf.read_val<int>();
    int max_m0 = buf.read_val<int>();
    double level_mult = buf.read_val<double>();

    HNSWGraph<T> *hnsw = new HNSWGraph<T>(dim, m, ef_c, ef_s, lim, max_elements, seed);
    hnsw->max_level = max_l;
    hnsw->entry_point_id = ep;
    hnsw->cur_element_count_ = (size_t)cur_count;

    // 读取第 0 层连续内存块（与写入格式对称）
    buf.read(hnsw->data_level0_, cur_count * hnsw->size_per_element_);

    // 读取高层链接数据并重建 label_lookup_
    for (uint64_t i = 0; i < cur_count; i++)
    {
        int link_level = buf.read_val<int>();
        hnsw->element_levels_[i] = link_level;

        if (link_level > 0)
        {
            size_t link_size = hnsw->size_links_per_element_ * link_level;
            hnsw->link_lists_[i] = (char *)malloc(link_size);
            if (!hnsw->link_lists_[i])
                throw std::runtime_error("deserialize_hnsw: failed to allocate link_lists_");
            buf.read(hnsw->link_lists_[i], link_size);
        }

        // 重建 label_lookup_
        int vid = hnsw->get_external_label((int)i);
        hnsw->label_lookup_[vid] = (int)i;
    }

    return hnsw;
}

// 序列化 HNSW 子集：保留完整的邻接结构（图骨架），但只传输指定子集的向量数据
// 非本地向量的数据区清零以节省带宽，接收方通过 local_vids_ 区分本地/远程节点
template <typename T>
void serialize_hnsw_subset(Buffer &buf, const HNSWGraph<T> *hnsw, const std::unordered_set<int> &local_vids, const std::unordered_map<int, int> *vid_ring = nullptr, int replication_level = 0)
{
    size_t cur_count = hnsw->cur_element_count_.load();
    size_t vec_bytes = hnsw->dim * sizeof(T);

    // 写入参数头（格式与 serialize_hnsw 完全一致，确保反序列化时通用）
    buf.write_val(hnsw->dim);
    buf.write_val(hnsw->m);
    buf.write_val(hnsw->ef_construction);
    buf.write_val(hnsw->ef_search);
    buf.write_val(hnsw->limited_level);
    buf.write_val(hnsw->max_level);
    buf.write_val(hnsw->entry_point_id);
    buf.write_val(hnsw->seed);
    buf.write_val((uint64_t)hnsw->max_elements_);
    buf.write_val((uint64_t)cur_count);
    buf.write_val((uint64_t)hnsw->size_per_element_);
    buf.write_val((uint64_t)hnsw->links_level0_size_);
    buf.write_val((uint64_t)hnsw->data_offset_);
    buf.write_val((uint64_t)hnsw->label_offset_);
    buf.write_val((uint64_t)hnsw->size_links_per_element_);
    buf.write_val(hnsw->max_m_);
    buf.write_val(hnsw->max_m0_);
    buf.write_val(hnsw->level_mult_);

    LOG_DEBUG_STREAM("Serializing HNSW subset: " << cur_count << " elements, " << local_vids.size() << " local vids.");

    // 写入第 0 层骨架数据（邻居链表 + label，跳过向量数据区以节省带宽）
    for (size_t i = 0; i < cur_count; i++)
    {
        char *elem_ptr = hnsw->data_level0_ + i * hnsw->size_per_element_;
        // 写入邻居链表区域（完整保留图的邻接结构）
        buf.write(elem_ptr, hnsw->links_level0_size_);
        // 写入节点的外部标签（vid）
        buf.write(elem_ptr + hnsw->label_offset_, sizeof(int));
    }

    // 仅写入本地持有的向量数据：以 (internal_id, vec_data) 对的形式传输
    buf.write_val((uint64_t)local_vids.size());
    for (size_t i = 0; i < cur_count; i++)
    {
        char *elem_ptr = hnsw->data_level0_ + i * hnsw->size_per_element_;
        int vid = *(int *)(elem_ptr + hnsw->label_offset_);
        if (local_vids.count(vid))
        {
            buf.write_val((uint64_t)i);  // internal_id
            buf.write(elem_ptr + hnsw->data_offset_, vec_bytes);  // 向量数据
        }
    }

    // 写入各节点的高层邻居链表数据（完整保留图结构）
    for (size_t i = 0; i < cur_count; i++)
    {
        int link_level = hnsw->element_levels_[i];
        buf.write_val(link_level);
        if (link_level > 0)
        {
            size_t link_size = hnsw->size_links_per_element_ * link_level;
            buf.write(hnsw->link_lists_[i], link_size);
        }
    }

    uint32_t ring_magic = 0x474e4952;
    buf.write_val(ring_magic);
    int stored_replication_level = replication_level ? replication_level : hnsw->replication_level_;
    buf.write_val(stored_replication_level);
    buf.write_val((uint64_t)local_vids.size());
    for (int vid : local_vids)
    {
        int ring = 0;
        if (vid_ring)
        {
            auto it = vid_ring->find(vid);
            if (it != vid_ring->end() && stored_replication_level > 0 && it->second == stored_replication_level)
                ring = stored_replication_level;
        }
        else
        {
            auto it = hnsw->label_lookup_.find(vid);
            if (it != hnsw->label_lookup_.end() && stored_replication_level > 0 && hnsw->is_iid_outermost(it->second))
                ring = stored_replication_level;
        }
        buf.write_val(vid);
        buf.write_val(ring);
    }
}

// 反序列化 HNSW 子集：读取图结构骨架和稀疏分布的本地向量数据
// do_compact=true（默认）：紧凑重排，邻居列表中远程 vid 以负数编码（forward/shard 模式）
// do_compact=false：保留完整图骨架，所有 vid 保持原始 internal_id；本地持有的向量数据
//                   由 local_vids_ 标记（global-pull 模式用，搜索时按需远程拉取）
template <typename T>
HNSWGraph<T> *deserialize_hnsw_subset(Buffer &buf, bool do_compact = true)
{
    int dim = buf.read_val<int>();
    int m = buf.read_val<int>();
    int ef_c = buf.read_val<int>();
    int ef_s = buf.read_val<int>();
    int lim = buf.read_val<int>();
    int max_l = buf.read_val<int>();
    int ep = buf.read_val<int>();
    unsigned int seed = buf.read_val<unsigned int>();
    uint64_t max_elements = buf.read_val<uint64_t>();
    uint64_t cur_count = buf.read_val<uint64_t>();
    uint64_t size_per_element = buf.read_val<uint64_t>();
    uint64_t links_level0_size = buf.read_val<uint64_t>();
    uint64_t data_offset = buf.read_val<uint64_t>();
    uint64_t label_offset = buf.read_val<uint64_t>();
    uint64_t size_links_per_element = buf.read_val<uint64_t>();
    int max_m = buf.read_val<int>();
    int max_m0 = buf.read_val<int>();
    double level_mult = buf.read_val<double>();

    size_t vec_bytes = dim * sizeof(T);

    HNSWGraph<T> *hnsw = new HNSWGraph<T>(dim, m, ef_c, ef_s, lim, max_elements, seed);
    hnsw->max_level = max_l;
    hnsw->entry_point_id = ep;
    hnsw->cur_element_count_ = (size_t)cur_count;

    // 读取第 0 层骨架（邻居链表 + label，向量数据区保持构造函数的零初始化状态）
    for (uint64_t i = 0; i < cur_count; i++)
    {
        char *elem_ptr = hnsw->data_level0_ + i * hnsw->size_per_element_;
        // 读取邻居链表数据
        buf.read(elem_ptr, hnsw->links_level0_size_);
        // 读取节点标签（写入 label_offset_ 对应位置）
        buf.read(elem_ptr + hnsw->label_offset_, sizeof(int));
    }

    // 读取本地向量数据：以 (internal_id, vec_data) 对的形式恢复
    uint64_t local_count = buf.read_val<uint64_t>();
    for (uint64_t i = 0; i < local_count; i++)
    {
        uint64_t internal_id = buf.read_val<uint64_t>();
        char *vec_ptr = hnsw->data_level0_ + internal_id * hnsw->size_per_element_ + hnsw->data_offset_;
        buf.read(vec_ptr, vec_bytes);
        // 将该向量对应的 vid 加入 local_vids_ 集合，标记为本地持有
        int vid = hnsw->get_external_label((int)internal_id);
        hnsw->local_vids_.insert(vid);
    }

    // 读取高层邻居链表并重建 label_lookup_ 映射表
    for (uint64_t i = 0; i < cur_count; i++)
    {
        int link_level = buf.read_val<int>();
        hnsw->element_levels_[i] = link_level;

        if (link_level > 0)
        {
            size_t link_size = hnsw->size_links_per_element_ * link_level;
            hnsw->link_lists_[i] = (char *)malloc(link_size);
            if (!hnsw->link_lists_[i])
                throw std::runtime_error("deserialize_hnsw_subset: failed to allocate link_lists_");
            buf.read(hnsw->link_lists_[i], link_size);
        }

        int vid = hnsw->get_external_label((int)i);
        hnsw->label_lookup_[vid] = (int)i;
    }

    LOG_DEBUG_STREAM("Deserialized HNSW subset: " << cur_count << " elements, " << hnsw->local_vids_.size() << " local vids.");

    uint32_t ring_magic = buf.read_val<uint32_t>();
    if (ring_magic != 0x474e4952)
        throw std::runtime_error("deserialize_hnsw_subset: missing RING section");
    hnsw->replication_level_ = buf.read_val<int>();
    uint64_t ring_count = buf.read_val<uint64_t>();
    hnsw->local_vid_ring_.reserve(ring_count);
    for (uint64_t i = 0; i < ring_count; i++)
    {
        int vid = buf.read_val<int>();
        int ring = buf.read_val<int>();
        hnsw->local_vid_ring_[vid] = ring;
    }

    if (do_compact && hnsw->local_vids_.size() < cur_count)
    {
        // 紧凑重排：仅保留本地持有的向量，远程邻居以负数编码 -(vid+1) 表示
        hnsw->compact();
        LOG_DEBUG_STREAM("Compacted HNSW subset: " << hnsw->get_cur_element_count() << " local elements, max_elements=" << hnsw->max_elements_);
    }
    else if (do_compact)
    {
        // 所有元素都是本地向量，图已经是紧凑格式，无需 compact
        hnsw->is_global_pull_mode_ = false;
        hnsw->is_compacted_graph_ = true;
        hnsw->resize_outermost_bits(hnsw->max_elements_);
        if (hnsw->replication_level_ > 0)
        {
            for (uint64_t i = 0; i < cur_count; i++)
            {
                int vid = hnsw->get_external_label((int)i);
                auto ring_it = hnsw->local_vid_ring_.find(vid);
                if (ring_it != hnsw->local_vid_ring_.end() && ring_it->second == hnsw->replication_level_)
                    hnsw->set_iid_outermost((int)i);
            }
        }
        std::unordered_set<int>().swap(hnsw->local_vids_);
        std::unordered_map<int, int>().swap(hnsw->local_vid_ring_);
        LOG_DEBUG_STREAM("Skipped compact (already compacted): " << cur_count << " local elements.");
    }
    else
    {
        // global-pull 模式：保留完整图骨架，不做 compact；label_lookup_ 已在上方循环中建立，
        // local_vids_ 标记本地持有的 vid 集合，搜索时通过 pull_fetcher_ 按需拉取远程向量。
        hnsw->is_global_pull_mode_ = true;
        LOG_DEBUG_STREAM("Skipped compact (global-pull mode): " << cur_count << " graph elements, " << hnsw->local_vids_.size() << " locally held vectors.");
    }

    return hnsw;
}

void serialize_routing(Buffer &buf,
                       const std::vector<std::unordered_map<int, std::vector<int>>> &node_maps)
{
    uint32_t magic = 0x54554f52;
    buf.write_val(magic);
    buf.write_val((uint64_t)node_maps.size());
    for (const auto &map : node_maps)
    {
        buf.write_val((uint64_t)map.size());
        for (const auto &kv : map)
        {
            buf.write_val(kv.first);
            buf.write_vec(kv.second);
        }
    }
}

void deserialize_routing(Buffer &buf,
                         std::vector<std::unordered_map<int, std::vector<int>>> &node_maps)
{
    uint32_t magic = buf.read_val<uint32_t>();
    if (magic != 0x54554f52)
        throw std::runtime_error("deserialize_routing: missing ROUT section");

    uint64_t levels = buf.read_val<uint64_t>();
    node_maps.clear();
    node_maps.resize(levels);
    for (uint64_t l = 0; l < levels; ++l)
    {
        uint64_t size = buf.read_val<uint64_t>();
        for (uint64_t i = 0; i < size; ++i)
        {
            int k = buf.read_val<int>();
            node_maps[l][k] = buf.read_vec<int>();
        }
    }
}

void serialize_layer_config(Buffer &buf, const std::unordered_map<int, std::vector<int>> &layer_config)
{
    buf.write_val((uint64_t)layer_config.size());
    for (const auto &kv : layer_config)
    {
        buf.write_val(kv.first);
        buf.write_vec(kv.second);
    }
}

void deserialize_layer_config(Buffer &buf, std::unordered_map<int, std::vector<int>> &layer_config)
{
    uint64_t size = buf.read_val<uint64_t>();
    layer_config.clear();
    for (uint64_t i = 0; i < size; ++i)
    {
        int level = buf.read_val<int>();
        std::vector<int> nodes = buf.read_vec<int>();
        layer_config[level] = std::move(nodes);
    }
}

void serialize_layer_replicate_flags(Buffer &buf, const std::vector<bool> &flags)
{
    std::vector<uint8_t> raw;
    raw.reserve(flags.size());
    for (bool flag : flags)
    {
        raw.push_back(flag ? 1 : 0);
    }
    buf.write_vec(raw);
}

void deserialize_layer_replicate_flags(Buffer &buf, std::vector<bool> &flags)
{
    std::vector<uint8_t> raw = buf.read_vec<uint8_t>();
    flags.clear();
    flags.reserve(raw.size());
    for (uint8_t v : raw)
    {
        flags.push_back(v != 0);
    }
}

    // 底层发送辅助函数：确保将指定大小的数据完整发送，处理部分发送的情况
void send_all(int sock, const void *data, size_t size)
{
    const char *ptr = static_cast<const char *>(data);
    size_t total_sent = 0;
    while (total_sent < size)
    {
        ssize_t sent = send(sock, ptr + total_sent, size - total_sent, 0);
        if (sent <= 0)
        {
            throw std::runtime_error("send failed");
        }
        total_sent += sent;
    }
}

    // 底层接收辅助函数：确保接收指定大小的完整数据，处理部分接收的情况
void recv_all(int sock, void *data, size_t size)
{
    char *ptr = static_cast<char *>(data);
    size_t total_read = 0;
    while (total_read < size)
    {
        ssize_t r = recv(sock, ptr + total_read, size - total_read, 0);
        if (r == 0)
        {
            // 对端正常关闭连接（EOF），用专门的消息让上层区分静默处理
            throw std::runtime_error("peer closed");
        }
        if (r < 0)
        {
            // SDK 端 disconnect() 走 SO_LINGER=0 → 发 RST，服务端 recv 返回 ECONNRESET。
            // 这是 SDK 短连接（init_routes/save/load_cluster/set_ef/partition）正常断开的副作用，
            // 与 FIN 等效，按 peer closed 静默处理；其他 errno 才报真错。
            int err = errno;
            if (err == ECONNRESET || err == EPIPE)
                throw std::runtime_error("peer closed");
            throw std::runtime_error(std::string("recv failed: ") + std::strerror(err));
        }
        total_read += r;
    }
}

    // --- 序列化/反序列化辅助工具函数结束 ---

template <typename T>
class Node
{
public:
    using DistT = DistTOf<T>;
    int node_id;
    Cluster<T> *cluster;
    HNSWGraph<T> *local_hnsw;
    uint64_t num_queries = 0;

    bool is_remote_node = false;

    int sockfd = -1;          // 节点间通信 server fd（transport）
    int sdk_sockfd = -1;      // SDK 通信 server fd（纯 TCP）
    struct sockaddr_in sock_addr;
    std::string ip_addr = "127.0.0.1";
    int port = 9000;
    static constexpr int NODE_PORT_OFFSET = 1000;  // 节点间端口 = port + NODE_PORT_OFFSET

    std::thread server_thread;      // 节点间通信线程
    std::thread sdk_server_thread;  // SDK 通信线程
    std::atomic<bool> stop_server{false};

    std::atomic<int> active_handlers{0};

    // 消息处理线程池：tcp_server_loop 收到完整消息后立即分派到此线程池处理，
    // 自己继续接收下一条消息，避免处理阻塞接收。
    std::unique_ptr<MessageThreadPool> msg_pool_;
    int msg_pool_size_ = 32;  // 默认线程池大小，可通过构造参数或外部设置

    // 保护 local_hnsw 初始化过程的互斥锁（防止多线程同时初始化同一节点）
    std::mutex local_hnsw_mutex;

    // 异步查询结果机制（流水线模式）
    // query_id → 对应 SDK 连接的完成队列。远程结果返回时直接推入队列，
    // Writer 线程从队列取出并发送回 SDK 客户端。无阻塞等待。
    std::mutex sdk_cq_map_mu_;
    std::unordered_map<int, std::shared_ptr<SDKCompletionQueue<T>>> sdk_cq_map_;  // query_id -> completion queue

    // Shard 模式异步累积器：query_id -> accumulator
    std::mutex reduce_accum_mu_;
    std::unordered_map<int, std::shared_ptr<ReduceQueryAccumulator<T>>> reduce_accum_map_;

    // 异步查询辅助方法：将结果推入 origin 节点的完成队列
    void send_result_to_origin(int query_id, const std::vector<int>& results, Payload<DistT>& payload);

    Node(int node_id, Cluster<T> *cluster, HNSWGraph<T> *local_hnsw = nullptr, bool is_remote_node = false, int msg_pool_size = 32)
        : node_id(node_id), cluster(cluster), local_hnsw(local_hnsw), is_remote_node(is_remote_node), msg_pool_size_(msg_pool_size)
    {
        port = 9500 + node_id;
    }

    // 禁止拷贝构造和拷贝赋值（包含线程、socket 等不可拷贝资源）
    Node(const Node &) = delete;
    Node &operator=(const Node &) = delete;

    // 移动构造函数（支持在 STL 容器中存储 Node 对象）
    Node(Node &&other) noexcept
        : node_id(other.node_id), cluster(other.cluster), local_hnsw(other.local_hnsw),
          num_queries(other.num_queries), is_remote_node(other.is_remote_node),
          sockfd(other.sockfd), sdk_sockfd(other.sdk_sockfd), sock_addr(other.sock_addr), ip_addr(std::move(other.ip_addr)),
          port(other.port), server_thread(std::move(other.server_thread)),
          sdk_server_thread(std::move(other.sdk_server_thread)), stop_server(other.stop_server.load()),
          msg_pool_(std::move(other.msg_pool_)), msg_pool_size_(other.msg_pool_size_)
    {
        other.sockfd = -1;
        other.sdk_sockfd = -1;
        other.local_hnsw = nullptr;
    }

    ~Node();

    void start_server();

    void reset()
    {
        num_queries = 0;
    }

    void tcp_server_loop();      // 节点间通信 accept 循环
    void sdk_server_loop();      // SDK 通信 accept 循环

    void close_connection_hard(int fd);
    void close_connection_soft(int fd);

    void handle_sdk_client(int client_fd);  // 处理 SDK 消息（裸 TCP 收发）

    std::vector<int> query(int query_id, const vector<T> &query_vector, Payload<DistT> &payload, int k = 10);

    void add_vector(int vid, const std::vector<T> &vec);
};

template <typename T>
class Cluster
{
public:
    using DistT = DistTOf<T>;
    HNSWGraph<T> *hnsw;

    int node_num = 4;

    // 集群中所有节点的状态集合（索引为节点编号）
    std::vector<Node<T>> nodes;

    // 向量到路由节点的映射关系（按层级组织，用于分布式查询时确定目标节点）
    std::vector<std::unordered_map<int, std::vector<int>>> vector_node_map;

    bool dense_level0_routes_enabled_ = false;
    int dense_level0_min_vid_ = 0;
    uint64_t dense_level0_count_ = 0;
    uint8_t dense_level0_owner_width_ = 0;
    std::vector<uint8_t> dense_level0_owner_bytes_;

    // 模拟网络带宽限制（单位：Mb/s，用于分布式场景下的性能评估和延迟模拟）
    double bandwidth = 1000.0;

    // 分区模式标识（控制向量在节点间的分配策略，如 round_robin、kmeans 等）
    std::string partition_mode;
    std::string layer_partition_mode;
    std::unordered_map<int, std::vector<int>> layer_config;
    std::vector<bool> layer_replicate_flags;

    int outer_ring_max_iterations = 0;

    int replication_level = 0;
    uint64_t replication_num = 0;

    bool is_distributed = false;
    bool is_shard = false;
    bool is_global_pull = false;  // global-pull 模式：每节点保留完整图骨架，仅向量数据按节点分布，查询时按需远程拉取
    bool owns_hnsw = false;

    std::atomic<int> add_round_robin_counter{0};

    // 通信中间层的配置参数和传输层实例（支持 TCP 和 RDMA 两种传输模式）
    TransportConfig transport_config;
    std::shared_ptr<ITransport> transport;

    // 折返检测窗口大小：transfer_to_remote 时保留最近 max_track_num 次目标节点 id，
    // 若下一次 transfer 的目标已出现在窗口内则计为一次 backtrack。0 表示关闭统计。
    int max_track_num = 10;

    Cluster(HNSWGraph<T> *hnsw, int node_num = 4, const TransportConfig& config = TransportConfig(), int max_track_num = 10)
        : hnsw(hnsw), node_num(node_num), transport_config(config), max_track_num(max_track_num)
    {
        transport = TransportFactory::create(transport_config);
        nodes.reserve(node_num);
        for (int i = 0; i < node_num; i++)
        {
            nodes.emplace_back(Node<T>(i, this, nullptr));
        }
    }

    void release_full_hnsw()
    {
        if (!hnsw)
            return;
        for (const auto &node : nodes)
            if (node.local_hnsw == hnsw)
                return;
        size_t released = hnsw->estimated_owned_bytes();
        if (owns_hnsw)
            delete hnsw;
        else
            hnsw->release_storage();
        hnsw = nullptr;
#if defined(__GLIBC__)
        malloc_trim(0);
#endif
        LOG_INFO_STREAM("[release_full_hnsw] released full graph estimate=" << HNSWGraph<T>::format_bytes(released));
    }

    uint64_t infer_replication_num_from_routes() const
    {
        return 0;
    }

    size_t estimate_route_map_bytes(const std::vector<std::unordered_map<int, std::vector<int>>> &maps) const
    {
        size_t total = maps.capacity() * sizeof(std::unordered_map<int, std::vector<int>>);
        for (const auto &level_map : maps)
        {
            total += level_map.bucket_count() * sizeof(void *);
            for (const auto &kv : level_map)
                total += sizeof(std::pair<const int, std::vector<int>>) + sizeof(void *)
                       + kv.second.capacity() * sizeof(int);
        }
        return total;
    }

    size_t dense_level0_route_bytes() const
    {
        return dense_level0_owner_bytes_.capacity();
    }

    void clear_dense_level0_routes()
    {
        dense_level0_routes_enabled_ = false;
        dense_level0_min_vid_ = 0;
        dense_level0_count_ = 0;
        dense_level0_owner_width_ = 0;
        std::vector<uint8_t>().swap(dense_level0_owner_bytes_);
    }

    void normalize_route_owners(std::vector<int> &owners) const
    {
        std::vector<int> cleaned;
        cleaned.reserve(owners.size());
        for (int node_id : owners)
        {
            if (node_id < 0 || node_id >= node_num)
                continue;
            if (std::find(cleaned.begin(), cleaned.end(), node_id) == cleaned.end())
                cleaned.push_back(node_id);
        }
        owners.swap(cleaned);
    }

    std::vector<int> normalized_route_owners(const std::vector<int> &owners) const
    {
        std::vector<int> cleaned = owners;
        normalize_route_owners(cleaned);
        return cleaned;
    }

    int primary_from_route(const std::vector<int> &candidates) const
    {
        for (int node_id : candidates)
            if (node_id >= 0 && node_id < node_num)
                return node_id;
        return -1;
    }

    void append_route_owner(int level, int vid, int node_id)
    {
        if (level < 0 || node_id < 0 || node_id >= node_num)
            return;
        if (level >= (int)vector_node_map.size())
            vector_node_map.resize((size_t)level + 1);
        auto &owners = vector_node_map[level][vid];
        if (std::find(owners.begin(), owners.end(), node_id) == owners.end())
            owners.push_back(node_id);
    }

    int current_local_node_id() const
    {
        for (int i = 0; i < node_num; i++)
            if (!nodes[i].is_remote_node)
                return i;
        return -1;
    }

    bool choose_from_route(const std::vector<int> &owners, int current_node_id, uint64_t salt, int &owner_node_id) const
    {
        int valid_count = 0;
        int nonlocal_count = 0;
        for (int node_id : owners)
        {
            if (node_id < 0 || node_id >= node_num)
                continue;
            valid_count++;
            if (node_id != current_node_id)
                nonlocal_count++;
        }
        int target_count = nonlocal_count > 0 ? nonlocal_count : valid_count;
        if (target_count <= 0)
            return false;
        int pick = (int)(salt % (uint64_t)target_count);
        for (int node_id : owners)
        {
            if (node_id < 0 || node_id >= node_num)
                continue;
            if (nonlocal_count > 0 && node_id == current_node_id)
                continue;
            if (pick-- == 0)
            {
                owner_node_id = node_id;
                return true;
            }
        }
        return false;
    }

    bool choose_route_owner(int level, int vid, int current_node_id, uint64_t salt, int &owner_node_id) const
    {
        if (level >= 0 && level < (int)vector_node_map.size())
        {
            auto it = vector_node_map[level].find(vid);
            if (it != vector_node_map[level].end() && choose_from_route(it->second, current_node_id, salt, owner_node_id))
                return true;
        }
        if (level == 0)
            return lookup_dense_level0_owner(vid, owner_node_id);
        return false;
    }

    bool lookup_dense_level0_owner(int vid, int &owner_node_id) const
    {
        if (!dense_level0_routes_enabled_ || dense_level0_owner_width_ == 0)
            return false;
        int64_t offset = (int64_t)vid - (int64_t)dense_level0_min_vid_;
        if (offset < 0 || (uint64_t)offset >= dense_level0_count_)
            return false;
        size_t pos = (size_t)offset * dense_level0_owner_width_;
        if (pos + dense_level0_owner_width_ > dense_level0_owner_bytes_.size())
            return false;

        uint32_t owner = 0;
        if (dense_level0_owner_width_ == 1)
        {
            uint8_t v = dense_level0_owner_bytes_[pos];
            if (v == std::numeric_limits<uint8_t>::max())
                return false;
            owner = v;
        }
        else if (dense_level0_owner_width_ == 2)
        {
            uint16_t v;
            std::memcpy(&v, dense_level0_owner_bytes_.data() + pos, sizeof(v));
            if (v == std::numeric_limits<uint16_t>::max())
                return false;
            owner = v;
        }
        else if (dense_level0_owner_width_ == 4)
        {
            uint32_t v;
            std::memcpy(&v, dense_level0_owner_bytes_.data() + pos, sizeof(v));
            if (v == std::numeric_limits<uint32_t>::max())
                return false;
            owner = v;
        }
        else
        {
            return false;
        }

        if (owner >= (uint32_t)node_num)
            return false;
        owner_node_id = (int)owner;
        return true;
    }

    bool lookup_primary_owner(int level, int vid, int &owner_node_id) const
    {
        if (level == 0 && lookup_dense_level0_owner(vid, owner_node_id))
            return true;
        if (level < 0 || level >= (int)vector_node_map.size())
            return false;
        auto it = vector_node_map[level].find(vid);
        if (it == vector_node_map[level].end())
            return false;
        int owner = primary_from_route(it->second);
        if (owner < 0)
            return false;
        owner_node_id = owner;
        return true;
    }

    void log_route_memory_breakdown(const std::string &prefix) const
    {
        auto log_maps = [&](const char *name, const std::vector<std::unordered_map<int, std::vector<int>>> &maps) {
            size_t entries = 0;
            size_t values = 0;
            size_t buckets = 0;
            size_t max_level_entries = 0;
            for (const auto &level_map : maps)
            {
                entries += level_map.size();
                buckets += level_map.bucket_count();
                max_level_entries = std::max(max_level_entries, level_map.size());
                for (const auto &kv : level_map)
                    values += kv.second.size();
            }
            LOG_INFO_STREAM(prefix << " " << name << " memory estimate: total="
                            << HNSWGraph<T>::format_bytes(estimate_route_map_bytes(maps))
                            << ", levels=" << maps.size()
                            << ", entries=" << entries
                            << ", route_values=" << values
                            << ", buckets=" << buckets
                            << ", max_level_entries=" << max_level_entries);
        };
        log_maps("vector_node_map", vector_node_map);
        if (dense_level0_routes_enabled_)
        {
            LOG_INFO_STREAM(prefix << " dense_level0_routes memory estimate: total="
                            << HNSWGraph<T>::format_bytes(dense_level0_route_bytes())
                            << ", min_vid=" << dense_level0_min_vid_
                            << ", count=" << dense_level0_count_
                            << ", owner_width=" << (int)dense_level0_owner_width_);
        }
    }

    void log_cluster_memory_breakdown(const std::string &prefix) const
    {
        log_route_memory_breakdown(prefix);
        size_t local_graph_total = 0;
        for (const auto &node : nodes)
        {
            if (node.local_hnsw)
            {
                local_graph_total += node.local_hnsw->estimated_owned_bytes();
                node.local_hnsw->log_memory_breakdown(prefix + " node" + std::to_string(node.node_id));
            }
        }
        LOG_INFO_STREAM(prefix << " local_hnsw total estimate on this process=" << HNSWGraph<T>::format_bytes(local_graph_total));
    }

    static constexpr uint32_t partition_file_magic()
    {
        return 0x46505256; // VRPF
    }

    static constexpr uint32_t partition_route_magic()
    {
        return 0x54525250; // PRRT
    }

    uint8_t route_owner_width() const
    {
        if (node_num <= (int)std::numeric_limits<uint8_t>::max())
            return 1;
        if (node_num <= (int)std::numeric_limits<uint16_t>::max())
            return 2;
        return 4;
    }

    void write_owner_value(std::vector<uint8_t> &bytes, uint8_t width, size_t idx, uint32_t owner) const
    {
        size_t pos = idx * width;
        if (width == 1)
        {
            bytes[pos] = (uint8_t)owner;
        }
        else if (width == 2)
        {
            uint16_t v = (uint16_t)owner;
            std::memcpy(bytes.data() + pos, &v, sizeof(v));
        }
        else
        {
            uint32_t v = owner;
            std::memcpy(bytes.data() + pos, &v, sizeof(v));
        }
    }

    void serialize_partition_routing(Buffer &buf) const
    {
        buf.write_string(partition_mode);
        buf.write_string(layer_partition_mode);
        buf.write_val(replication_level);
        buf.write_val(replication_num);
        buf.write_val(node_num);
        serialize_layer_config(buf, layer_config);
        serialize_layer_replicate_flags(buf, layer_replicate_flags);

        buf.write_val(partition_route_magic());
        buf.write_val((uint32_t)2);
        uint64_t levels = std::max<uint64_t>((uint64_t)vector_node_map.size(), dense_level0_routes_enabled_ ? 1 : 0);
        buf.write_val(levels);

        const auto *level0 = (!vector_node_map.empty()) ? &vector_node_map[0] : nullptr;
        bool wrote_dense = false;
        if (dense_level0_routes_enabled_)
        {
            buf.write_val((uint8_t)1);
            buf.write_val(dense_level0_min_vid_);
            buf.write_val(dense_level0_count_);
            buf.write_val(dense_level0_owner_width_);
            buf.write_val((uint64_t)dense_level0_owner_bytes_.size());
            if (!dense_level0_owner_bytes_.empty())
                buf.write(dense_level0_owner_bytes_.data(), dense_level0_owner_bytes_.size());
            buf.write_val((uint64_t)0);
            wrote_dense = true;
        }
        else if (level0 && !level0->empty())
        {
            int min_vid = std::numeric_limits<int>::max();
            int max_vid = std::numeric_limits<int>::min();
            uint64_t valid_entries = 0;
            for (const auto &kv : *level0)
            {
                int owner = primary_from_route(kv.second);
                if (owner < 0)
                    continue;
                min_vid = std::min(min_vid, kv.first);
                max_vid = std::max(max_vid, kv.first);
                valid_entries++;
            }

            uint8_t width = route_owner_width();
            bool can_dense = valid_entries > 0 && min_vid >= 0 && max_vid >= min_vid;
            uint64_t range = can_dense ? (uint64_t)((int64_t)max_vid - (int64_t)min_vid + 1) : 0;
            if (can_dense)
                can_dense = range <= valid_entries + valid_entries / 4 + 1024;
            if (can_dense && range > (std::numeric_limits<size_t>::max() / width))
                can_dense = false;

            if (can_dense)
            {
                std::vector<uint8_t> dense_bytes((size_t)range * width, 0xff);
                for (const auto &kv : *level0)
                {
                    int owner = primary_from_route(kv.second);
                    if (owner < 0)
                        continue;
                    size_t idx = (size_t)((int64_t)kv.first - (int64_t)min_vid);
                    write_owner_value(dense_bytes, width, idx, (uint32_t)owner);
                }
                buf.write_val((uint8_t)1);
                buf.write_val(min_vid);
                buf.write_val(range);
                buf.write_val(width);
                buf.write_val((uint64_t)dense_bytes.size());
                buf.write(dense_bytes.data(), dense_bytes.size());
                buf.write_val((uint64_t)0);
                wrote_dense = true;
            }
        }

        if (!wrote_dense)
        {
            buf.write_val((uint8_t)0);
            uint64_t count = 0;
            if (level0)
            {
                for (const auto &kv : *level0)
                    if (!normalized_route_owners(kv.second).empty())
                        count++;
            }
            buf.write_val(count);
            if (level0)
            {
                for (const auto &kv : *level0)
                {
                    std::vector<int> owners = normalized_route_owners(kv.second);
                    int owner = primary_from_route(owners);
                    if (owner < 0)
                        continue;
                    buf.write_val(kv.first);
                    buf.write_vec(std::vector<int>{owner});
                }
            }
        }

        for (uint64_t level = 1; level < levels; level++)
        {
            const auto *level_map = (level < vector_node_map.size()) ? &vector_node_map[level] : nullptr;
            uint64_t count = 0;
            if (level_map)
            {
                for (const auto &kv : *level_map)
                    if (!normalized_route_owners(kv.second).empty())
                        count++;
            }
            buf.write_val(count);
            if (!level_map)
                continue;
            for (const auto &kv : *level_map)
            {
                std::vector<int> owners = normalized_route_owners(kv.second);
                if (owners.empty())
                    continue;
                buf.write_val(kv.first);
                buf.write_vec(owners);
            }
        }
    }

    void deserialize_partition_routing(Buffer &buf)
    {
        partition_mode = buf.read_string();
        layer_partition_mode = buf.read_string();
        replication_level = buf.read_val<int>();
        replication_num = buf.read_val<uint64_t>();
        int file_node_num = buf.read_val<int>();
        if (file_node_num != node_num)
            throw std::runtime_error("load_local: node_num mismatch in partition routing");
        deserialize_layer_config(buf, layer_config);
        deserialize_layer_replicate_flags(buf, layer_replicate_flags);

        uint32_t route_magic = buf.read_val<uint32_t>();
        if (route_magic != partition_route_magic())
            throw std::runtime_error("load_local: missing PRRT route section");
        uint32_t route_version = buf.read_val<uint32_t>();
        if (route_version != 2)
            throw std::runtime_error("load_local: unsupported partition route version");
        uint64_t levels = buf.read_val<uint64_t>();
        vector_node_map.clear();
        vector_node_map.resize(levels);
        clear_dense_level0_routes();

        uint8_t level0_encoding = buf.read_val<uint8_t>();
        if (level0_encoding == 1)
        {
            dense_level0_min_vid_ = buf.read_val<int>();
            dense_level0_count_ = buf.read_val<uint64_t>();
            dense_level0_owner_width_ = buf.read_val<uint8_t>();
            if (dense_level0_owner_width_ != 1 && dense_level0_owner_width_ != 2 && dense_level0_owner_width_ != 4)
                throw std::runtime_error("load_local: invalid dense owner width");
            uint64_t bytes = buf.read_val<uint64_t>();
            if (bytes != dense_level0_count_ * dense_level0_owner_width_)
                throw std::runtime_error("load_local: dense route byte size mismatch");
            if (bytes > (uint64_t)std::numeric_limits<size_t>::max())
                throw std::runtime_error("load_local: dense route too large for this process");
            dense_level0_owner_bytes_.resize((size_t)bytes);
            if (bytes > 0)
                buf.read(dense_level0_owner_bytes_.data(), (size_t)bytes);
            dense_level0_routes_enabled_ = true;
            uint64_t extra_count = buf.read_val<uint64_t>();
            if (!vector_node_map.empty())
                vector_node_map[0].reserve((size_t)std::min<uint64_t>(extra_count, (uint64_t)std::numeric_limits<size_t>::max()));
            for (uint64_t i = 0; i < extra_count; i++)
            {
                int vid = buf.read_val<int>();
                std::vector<int> owners = buf.read_vec<int>();
                normalize_route_owners(owners);
                if (!owners.empty() && !vector_node_map.empty())
                    vector_node_map[0][vid] = std::move(owners);
            }
        }
        else if (level0_encoding == 0)
        {
            uint64_t count = buf.read_val<uint64_t>();
            if (!vector_node_map.empty())
                vector_node_map[0].reserve((size_t)std::min<uint64_t>(count, (uint64_t)std::numeric_limits<size_t>::max()));
            for (uint64_t i = 0; i < count; i++)
            {
                int vid = buf.read_val<int>();
                std::vector<int> owners = buf.read_vec<int>();
                normalize_route_owners(owners);
                if (!owners.empty() && !vector_node_map.empty())
                    vector_node_map[0][vid] = std::move(owners);
            }
        }
        else
        {
            throw std::runtime_error("load_local: unknown level0 route encoding");
        }

        for (uint64_t level = 1; level < levels; level++)
        {
            uint64_t count = buf.read_val<uint64_t>();
            vector_node_map[level].reserve((size_t)std::min<uint64_t>(count, (uint64_t)std::numeric_limits<size_t>::max()));
            for (uint64_t i = 0; i < count; i++)
            {
                int vid = buf.read_val<int>();
                std::vector<int> owners = buf.read_vec<int>();
                normalize_route_owners(owners);
                if (!owners.empty())
                    vector_node_map[level][vid] = std::move(owners);
            }
        }
        filter_routes_for_runtime();
        if (buf.read_pos != buf.data.size())
            throw std::runtime_error("load_local: trailing bytes in partition routing section");
    }

    std::string build_routing_json_string() const
    {
        json routing_json_obj;
        routing_json_obj["partition_mode"] = partition_mode;
        routing_json_obj["layer_partition_mode"] = layer_partition_mode;
        routing_json_obj["replication_level"] = replication_level;
        routing_json_obj["replication_num"] = replication_num;
        routing_json_obj["layer_config"] = layer_config;
        routing_json_obj["layer_replicate_flags"] = layer_replicate_flags;
        routing_json_obj["vector_node_map"] = json::array();
        for (const auto &level_map : vector_node_map)
        {
            json level_json = json::object();
            for (const auto &kv : level_map)
                level_json[std::to_string(kv.first)] = kv.second;
            routing_json_obj["vector_node_map"].push_back(std::move(level_json));
        }
        return routing_json_obj.dump();
    }

    bool write_partition_file(const std::string &filepath,
                              const std::unordered_set<int> &local_vids,
                              const std::unordered_map<int, int> *local_vid_ring)
    {
        std::filesystem::path p(filepath);
        if (p.has_parent_path())
            std::filesystem::create_directories(p.parent_path());

        Buffer routing_buf;
        serialize_partition_routing(routing_buf);
        Buffer subset_buf;
        serialize_hnsw_subset(subset_buf, hnsw, local_vids, local_vid_ring, replication_level);

        std::ofstream output_file(filepath, std::ios::binary);
        if (!output_file)
        {
            LOG_INFO_STREAM("[partition nfs-load] Failed to open " << filepath << " for writing");
            return false;
        }
        uint32_t magic = partition_file_magic();
        uint32_t version = 2;
        uint64_t routing_size = routing_buf.data.size();
        output_file.write((char *)&magic, sizeof(uint32_t));
        output_file.write((char *)&version, sizeof(uint32_t));
        output_file.write((char *)&routing_size, sizeof(uint64_t));
        output_file.write(routing_buf.data.data(), routing_buf.data.size());
        output_file.write(subset_buf.data.data(), subset_buf.data.size());
        if (!output_file)
        {
            LOG_INFO_STREAM("[partition nfs-load] Write failed for " << filepath);
            return false;
        }
        LOG_INFO_STREAM("[partition nfs-load] Wrote " << filepath
                        << " routing=" << HNSWGraph<T>::format_bytes(routing_buf.data.size())
                        << " subset=" << HNSWGraph<T>::format_bytes(subset_buf.data.size())
                        << " total=" << HNSWGraph<T>::format_bytes(sizeof(uint32_t) * 2 + sizeof(uint64_t) + routing_buf.data.size() + subset_buf.data.size()));
        return true;
    }

    bool should_keep_full_route_level(int level) const
    {
        return level >= 0 && level < (int)layer_replicate_flags.size() && layer_replicate_flags[level];
    }

    void normalize_routes_primary_first()
    {
        for (auto &level_map : vector_node_map)
        {
            for (auto it = level_map.begin(); it != level_map.end(); )
            {
                normalize_route_owners(it->second);
                if (it->second.empty())
                    it = level_map.erase(it);
                else
                    ++it;
            }
        }
    }

    void filter_routes_for_runtime()
    {
        for (size_t level = 0; level < vector_node_map.size(); level++)
        {
            auto &level_map = vector_node_map[level];
            if (should_keep_full_route_level((int)level))
            {
                for (auto &kv : level_map)
                    normalize_route_owners(kv.second);
                continue;
            }
            for (auto &kv : level_map)
            {
                normalize_route_owners(kv.second);
                if (kv.second.size() > 1)
                    kv.second.resize(1);
            }
            if (level == 0 && dense_level0_routes_enabled_)
                std::unordered_map<int, std::vector<int>>().swap(level_map);
        }
    }

    ~Cluster()
    {
        // 停止所有节点的服务器线程并释放关联资源
        for (int i = 0; i < node_num; i++)
        {
            nodes[i].stop_server.store(true);
        }
        // 通过通信中间层向服务端发送停止信号
        if (transport)
        {
            transport->stop_server();
        }
        // 关闭 SDK 服务端的监听 socket，使 accept 循环优雅退出
        for (int i = 0; i < node_num; i++)
        {
            nodes[i].sockfd = -1;
            if (nodes[i].sdk_sockfd != -1)
            {
                shutdown(nodes[i].sdk_sockfd, SHUT_RDWR);
                nodes[i].sdk_sockfd = -1;
            }
        }
        // 阻塞等待所有服务器线程优雅退出
        for (int i = 0; i < node_num; i++)
        {
            if (nodes[i].server_thread.joinable())
            {
                nodes[i].server_thread.join();
            }
            if (nodes[i].sdk_server_thread.joinable())
            {
                nodes[i].sdk_server_thread.join();
            }
        }
        // 释放各节点的 local_hnsw 资源（由 partition 方法动态创建，在析构时统一释放）
        for (int i = 0; i < node_num; i++)
        {
            if (nodes[i].local_hnsw != nullptr && nodes[i].local_hnsw != hnsw)
            {
                delete nodes[i].local_hnsw;
                nodes[i].local_hnsw = nullptr;
            }
        }
        if (hnsw && owns_hnsw)
        {
            delete hnsw;
            hnsw = nullptr;
        }
    }

    void add_vector(int vid, const std::vector<T> &vec)
    {
        if (is_shard)
        {
            // 采用轮询策略将向量均匀分配到各个节点
            int target_node = add_round_robin_counter % node_num;
            add_round_robin_counter++;

            nodes[target_node].add_vector(vid, vec);
        }
        else
        {
            if (hnsw)
            {
                hnsw->add_vector(vid, vec);
            }
            else
            {
                LOG_DEBUG_STREAM("Error: HNSW graph is null, cannot add vector.");
            }
        }
    }

    void reset()
    {
        for (int i = 0; i < node_num; i++)
        {
            nodes[i].reset();
        }
    }

    // 配置指定节点的网络通信地址信息（IP、端口及远程标志）
    void set_node_info(int node_id, const std::string &ip, int port, bool is_remote)
    {
        if (node_id >= 0 && node_id < node_num)
        {
            nodes[node_id].ip_addr = ip;
            nodes[node_id].port = port;
            nodes[node_id].is_remote_node = is_remote;
            // 节点间内部通信使用 port + NODE_PORT_OFFSET 作为端口（与客户端服务端口区分）
            transport->register_node(node_id, ip, port + Node<T>::NODE_PORT_OFFSET);
        }
    }

    // 启动指定节点的全部服务（包括节点间通信服务和客户端 SDK 服务）
    // msg_pool_size: 消息处理线程池大小，建议设为与查询线程数相同
    void start_node_server(int node_id, int msg_pool_size = 32)
    {
        if (node_id >= 0 && node_id < node_num)
        {
            nodes[node_id].msg_pool_size_ = msg_pool_size;
            nodes[node_id].start_server();
        }
    }

    // 阻塞等待指定节点的服务线程退出（供 Worker 节点保持运行直到收到停止信号）
    void join_node_server(int node_id)
    {
        if (node_id >= 0 && node_id < node_num)
        {
            if (nodes[node_id].server_thread.joinable())
            {
                nodes[node_id].server_thread.join();
            }
        }
    }

    // 简单 K-Means 聚类实现（基于欧氏距离，支持指定随机种子确保可复现性）
    // 返回值：与 data 迭代顺序一致的聚类标签数组（labels[i] 对应 data 中第 i 个向量的簇编号）
    // kmeans 核心实现：接受指向向量数据的指针数组，避免不必要的内存拷贝
    static std::vector<int> kmeans_core(const std::vector<const T*> &ptrs, int n, int dim,
                                        int k, int max_iters = 100, int seed = 123,
                                        int num_threads = 0)
    {
        std::vector<int> labels(n, 0);
        if (n == 0 || k <= 0)
            return labels;

        std::mt19937 rng(seed);

        // 质心始终用 double 存储，避免整数类型截断导致所有质心收敛到全局均值
        std::vector<std::vector<double>> centers(k, std::vector<double>(dim));
        std::unordered_set<int> chosen;
        std::uniform_int_distribution<int> uid(0, n - 1);
        int ci = 0;
        while (ci < k)
        {
            int idx = uid(rng);
            if (chosen.insert(idx).second)
            {
                for (int d = 0; d < dim; d++)
                    centers[ci][d] = (double)ptrs[idx][d];
                ci++;
            }
        }

        // 距离计算用 SIMD 优化的 dist_func（与 HNSW 查询共用同一套 SIMD 检测逻辑）
        auto space = create_l2_space<T>((size_t)dim);
        auto dist_func = space->get_dist_func();
        auto dist_param = space->get_dist_func_param();

        int nthreads = (num_threads > 0) ? num_threads
                     : std::max(1, (int)std::thread::hardware_concurrency());
        LOG_INFO_STREAM("[kmeans] n=" << n << " k=" << k << " dim=" << dim
                        << " num_threads=" << num_threads << " nthreads=" << nthreads
                        << " hardware_concurrency=" << std::thread::hardware_concurrency());

        // 调试：打印前几个向量和初始质心的前几维
        {
            auto print_vec = [&](const char* label, const T* v, int len) {
                std::ostringstream oss;
                oss << label << " [";
                for (int d = 0; d < std::min(len, 8); d++)
                    oss << (d ? "," : "") << +v[d];  // unary + 防止 uint8_t 被当字符打印
                if (len > 8) oss << ",...";
                oss << "]";
                LOG_INFO_STREAM(oss.str());
            };
            for (int i = 0; i < std::min(3, n); i++)
                print_vec(("  vec[" + std::to_string(i) + "]").c_str(), ptrs[i], dim);
            for (int j = 0; j < k; j++) {
                std::ostringstream oss;
                oss << "  center[" << j << "] [";
                for (int d = 0; d < std::min(dim, 8); d++)
                    oss << (d ? "," : "") << centers[j][d];
                if (dim > 8) oss << ",...";
                oss << "]";
                LOG_INFO_STREAM(oss.str());
            }
        }

        for (int iter = 0; iter < max_iters; iter++)
        {
            std::atomic<bool> changed{false};

            // 将 double 质心截断为 T 类型的临时副本，用于 SIMD 距离计算
            std::vector<std::vector<T>> centers_t(k, std::vector<T>(dim));
            for (int j = 0; j < k; j++)
                for (int d = 0; d < dim; d++)
                    centers_t[j][d] = (T)centers[j][d];

            // 分配阶段：多线程并行，每个向量独立计算到各质心的距离
            {
                std::vector<std::thread> threads;
                int chunk = (n + nthreads - 1) / nthreads;
                for (int t = 0; t < nthreads; t++)
                {
                    int start = t * chunk;
                    int end = std::min(start + chunk, n);
                    if (start >= n) break;
                    threads.emplace_back([&, start, end]() {
                        for (int i = start; i < end; i++)
                        {
                            DistT best_dist = numeric_limits<DistT>::max();
                            int best_center = 0;
                            const T* vec = ptrs[i];
                            for (int j = 0; j < k; j++)
                            {
                                DistT dist = dist_func(vec, centers_t[j].data(), dist_param);
                                if (dist < best_dist)
                                {
                                    best_dist = dist;
                                    best_center = j;
                                }
                            }
                            if (labels[i] != best_center)
                            {
                                labels[i] = best_center;
                                changed.store(true, std::memory_order_relaxed);
                            }
                        }
                    });
                }
                for (auto &t : threads) t.join();
            }

            if (!changed.load())
                break;

            // 更新阶段：per-thread 局部累加后归约，避免锁竞争
            std::vector<std::vector<std::vector<double>>> thread_sums(nthreads,
                std::vector<std::vector<double>>(k, std::vector<double>(dim, 0.0)));
            std::vector<std::vector<int>> thread_counts(nthreads, std::vector<int>(k, 0));

            {
                std::vector<std::thread> threads;
                int chunk = (n + nthreads - 1) / nthreads;
                for (int t = 0; t < nthreads; t++)
                {
                    int start = t * chunk;
                    int end = std::min(start + chunk, n);
                    if (start >= n) break;
                    threads.emplace_back([&, t, start, end]() {
                        auto &local_sums = thread_sums[t];
                        auto &local_counts = thread_counts[t];
                        for (int i = start; i < end; i++)
                        {
                            int cid = labels[i];
                            local_counts[cid]++;
                            for (int j = 0; j < dim; j++)
                                local_sums[cid][j] += (double)ptrs[i][j];
                        }
                    });
                }
                for (auto &t : threads) t.join();
            }

            // 归约各线程的局部结果
            std::vector<std::vector<double>> sums(k, std::vector<double>(dim, 0.0));
            std::vector<int> counts(k, 0);
            for (int t = 0; t < nthreads; t++)
            {
                for (int j = 0; j < k; j++)
                {
                    counts[j] += thread_counts[t][j];
                    for (int d = 0; d < dim; d++)
                        sums[j][d] += thread_sums[t][j][d];
                }
            }

            for (int j = 0; j < k; j++)
            {
                if (counts[j] > 0)
                {
                    for (int t = 0; t < dim; t++)
                        centers[j][t] = sums[j][t] / counts[j];
                }
                else
                {
                    // 空簇：重新随机选一个点作为质心
                    int idx = uid(rng);
                    for (int d = 0; d < dim; d++)
                        centers[j][d] = (double)ptrs[idx][d];
                }
            }

            // 打印每轮迭代各簇大小
            {
                std::ostringstream oss;
                oss << "[kmeans] iter=" << iter << " counts=[";
                for (int j = 0; j < k; j++)
                    oss << (j ? "," : "") << counts[j];
                oss << "]";
                LOG_INFO_STREAM(oss.str());
            }
        }
        return labels;
    }

    // 兼容接口：接受 unordered_map（供 pybind 和 layer partition 的 subset_vectors 使用）
    static std::vector<int> kmeans(const std::unordered_map<int, vector<T>> &data,
                                   int k, int max_iters = 100, int seed = 123,
                                   int num_threads = 0)
    {
        int n = (int)data.size();
        if (n == 0) return std::vector<int>();
        int dim = (int)data.begin()->second.size();
        std::vector<const T*> ptrs;
        ptrs.reserve(n);
        for (const auto &kv : data)
            ptrs.push_back(kv.second.data());
        return kmeans_core(ptrs, n, dim, k, max_iters, seed, num_threads);
    }

    // 直接从 HNSW 图执行 kmeans，零拷贝
    static std::vector<int> kmeans_hnsw(const HNSWGraph<T> *hnsw, int k,
                                        int max_iters = 100, int seed = 123,
                                        int num_threads = 0)
    {
        int n = (int)hnsw->get_cur_element_count();
        int dim = hnsw->dim;
        std::vector<const T*> ptrs;
        ptrs.reserve(n);
        for (int i = 0; i < n; i++)
            ptrs.push_back((const T*)hnsw->get_data_by_id(i));
        return kmeans_core(ptrs, n, dim, k, max_iters, seed, num_threads);
    }

    // Fennel 流式图分区（Tsourakakis et al., WSDM 2014），单遍 + restreaming。
    // 直接零拷贝读 HNSW linklist，额外内存仅 O(n) 标签数组（~4 字节/顶点），永不 OOM。
    // members 为空 => 全图（level 0，local idx == internal_id）；非空 => 该层子图（用 level 层 linklist）。
    // 返回 labels：members 为空时按 internal_id 0..n-1，非空时按 members 顺序，取值 [0, k)。
    // 参数：passes restream 遍数；gamma Fennel 指数（默认 1.5）；nu 平衡硬上限系数（每分区 <= nu*n/k）。
    // seed_labels（可选）：初始标签种子（大小须 == n,取值 [0,k)）。用于 kmeans-fennel：
    //   以 k-means 结果作初始分区,再用 Fennel restream 沿图边精修。nullptr=冷启动(纯 fennel)。
    static std::vector<int> fennel_partition(const HNSWGraph<T> *hnsw,
                                             const std::vector<int> &members,
                                             int level, int k,
                                             int passes = 3, double gamma = 1.5,
                                             double nu = 1.1, int seed = 123,
                                             const std::vector<int> *seed_labels = nullptr)
    {
        // 环境变量覆盖（VR_FENNEL_PASSES / VR_FENNEL_BALANCE）
        if (const char *e = std::getenv("VR_FENNEL_PASSES")) { int v = std::atoi(e); if (v > 0) passes = v; }
        if (const char *e = std::getenv("VR_FENNEL_BALANCE")) { double v = std::atof(e); if (v >= 1.0) nu = v; }

        const bool full_graph = members.empty();
        const int n = full_graph ? (int)hnsw->get_cur_element_count() : (int)members.size();
        std::vector<int> labels(n, -1);
        if (n == 0 || k <= 0)
            return labels;
        if (k == 1)
        {
            std::fill(labels.begin(), labels.end(), 0);
            return labels;
        }

        // 种子初始化（kmeans-fennel）：用 k-means 标签作为 Fennel 的起点（clamp 到 [0,k)）
        const bool seeded = (seed_labels != nullptr && (int)seed_labels->size() == n);
        if (seeded)
        {
            for (int i = 0; i < n; i++)
            {
                int v = (*seed_labels)[i];
                labels[i] = (v >= 0 && v < k) ? v : 0;
            }
            LOG_INFO_STREAM("[fennel] seeded with k-means labels (kmeans-fennel), n=" << n << " k=" << k);
        }

        // internal_id -> 本地下标（仅子图需要；全图为 identity）
        std::unordered_map<int, int> id2local;
        if (!full_graph)
        {
            id2local.reserve(n * 2);
            for (int i = 0; i < n; i++)
                id2local[members[i]] = i;
        }
        auto local_of = [&](int internal_id) -> int {
            if (full_graph) return internal_id;
            auto it = id2local.find(internal_id);
            return it == id2local.end() ? -1 : it->second;
        };
        auto internal_of = [&](int i) -> int { return full_graph ? i : members[i]; };
        auto get_links = [&](int internal_id, int &sz) -> const int * {
            const int *ll = (level == 0) ? hnsw->get_linklist0(internal_id)
                                         : hnsw->get_linklist(internal_id, level);
            sz = hnsw->get_list_count(ll);
            return ll + 1;
        };

        // 统计有向边数 m，用于 alpha = m * k^(gamma-1) / n^gamma
        long long m = 0;
        for (int i = 0; i < n; i++)
        {
            int sz;
            get_links(internal_of(i), sz);
            m += sz;
        }
        const double alpha = (double)m * std::pow((double)k, gamma - 1.0) / std::pow((double)n, gamma);
        const double cap = nu * (double)n / (double)k; // 每分区硬平衡上限
        const bool gamma_is_1_5 = std::fabs(gamma - 1.5) < 1e-9;

        LOG_INFO_STREAM("[fennel] n=" << n << " k=" << k << " m(directed)=" << m
                        << " alpha=" << alpha << " passes=" << passes << " gamma=" << gamma
                        << " nu=" << nu << " cap=" << cap << " seed=" << seed
                        << " scope=" << (full_graph ? "full" : "layer") << " level=" << level);

        std::vector<long long> psize(k, 0);
        // psize 初始化为当前 labels 的分布：冷启动全 -1 => 全 0；种子启动 => k-means 分布
        // （这样 restream 的「先减回旧标签」逻辑对种子启动也成立）
        for (int i = 0; i < n; i++)
            if (labels[i] >= 0 && labels[i] < k)
                psize[labels[i]]++;
        std::vector<int> cnt(k, 0); // 复用缓冲，避免每顶点堆分配

        for (int pass = 0; pass < passes; pass++)
        {
            for (int i = 0; i < n; i++)
            {
                const int internal_id = internal_of(i);
                if (labels[i] >= 0) // restream：先把旧标签计数减回
                    psize[labels[i]]--;

                std::fill(cnt.begin(), cnt.end(), 0);
                int sz;
                const int *nb = get_links(internal_id, sz);
                for (int j = 0; j < sz; j++)
                {
                    int li = local_of(nb[j]);
                    if (li < 0)
                        continue;
                    int p = labels[li];
                    if (p >= 0)
                        cnt[p]++;
                }

                // argmax score(p) = cnt[p] - alpha*gamma*psize[p]^(gamma-1)，受 cap 约束
                double best_score = -std::numeric_limits<double>::infinity();
                int best_p = -1;
                for (int p = 0; p < k; p++)
                {
                    if ((double)psize[p] >= cap)
                        continue;
                    double pg = gamma_is_1_5 ? std::sqrt((double)psize[p])
                                             : std::pow((double)psize[p], gamma - 1.0);
                    double score = (double)cnt[p] - alpha * gamma * pg;
                    if (score > best_score)
                    {
                        best_score = score;
                        best_p = p;
                    }
                }
                if (best_p < 0) // 所有分区都达 cap（极少见）→ 放当前最小的
                    best_p = (int)(std::min_element(psize.begin(), psize.end()) - psize.begin());

                labels[i] = best_p;
                psize[best_p]++;
            }
        }

        // 质量日志：各分区大小 + 跨分区边数（cut）
        long long pmin = psize[0], pmax = psize[0];
        for (int p = 0; p < k; p++)
        {
            pmin = std::min(pmin, psize[p]);
            pmax = std::max(pmax, psize[p]);
        }
        long long cut = 0;
        for (int i = 0; i < n; i++)
        {
            int sz;
            const int *nb = get_links(internal_of(i), sz);
            int pi = labels[i];
            for (int j = 0; j < sz; j++)
            {
                int li = local_of(nb[j]);
                if (li >= 0 && labels[li] != pi)
                    cut++;
            }
        }
        LOG_INFO_STREAM("[fennel] done: psize_min=" << pmin << " psize_max=" << pmax
                        << " balance=" << (n > 0 ? (double)pmax * k / n : 0.0)
                        << " edge_cut(directed)=" << cut
                        << " cut_ratio=" << (m > 0 ? (double)cut / m : 0.0));
        return labels;
    }

    int outer_ring(vector<unordered_map<int, vector<T>>> &local_vectors)
    {
        // 缓存全局邻居关系和向量数据用于 outer_ring 操作
        auto all_neighbors = hnsw->get_all_neighbors();
        auto all_vectors = hnsw->get_all_vectors();

        // 1. 找出所有有跨节点邻居的候选 boundary vectors（及其跨节点邻居数）
    unordered_map<int, vector<pair<int, int>>> boundary_candidates; // 节点 -> [(vid, 跨节点连接数)]
        for (int node_id = 0; node_id < node_num; node_id++)
        {
            for (const auto &kv : local_vectors[node_id])
            {
                int vid = kv.first;
                int cross_cnt = 0;
                auto level_it = all_neighbors.find(0);
                if (level_it != all_neighbors.end())
                {
                    auto vid_it = level_it->second.find(vid);
                    if (vid_it != level_it->second.end())
                    {
                        for (int nid : vid_it->second)
                        {
                            auto route_it = vector_node_map[0].find(nid);
                            if (route_it == vector_node_map[0].end() || route_it->second.empty())
                                continue;
                            if (route_it->second[0] != node_id)
                            {
                                cross_cnt++;
                            }
                        }
                    }
                }
                if (cross_cnt > 0)
                    boundary_candidates[node_id].emplace_back(vid, cross_cnt);
            }
        }

        // 2. 按跨节点邻居数降序挑选外圈
        double outer_fraction = 0.02;
        int max_outer = 50;
        vector<unordered_set<int>> outer_ring(node_num);
        for (int node_id = 0; node_id < node_num; node_id++)
        {
            auto &candidates = boundary_candidates[node_id];

            if (candidates.empty())
                continue;

            sort(candidates.begin(), candidates.end(), [](auto &a, auto &b)
                 { return a.second > b.second; });

            int choose_n = max(1, int(local_vectors[node_id].size() * outer_fraction));
            choose_n = min({choose_n, max_outer, (int)candidates.size()});

            for (int i = 0; i < choose_n; ++i)
            {
                outer_ring[node_id].insert(candidates[i].first);
            }
        }

        // 3. 从每个 node 的本地存储中删除未被选为外圈的 boundary vectors
        unordered_set<int> removed;
        for (int node_id = 0; node_id < node_num; node_id++)
        {
            for (auto &p : boundary_candidates[node_id])
            {
                int vid = p.first;
                if (!outer_ring[node_id].count(vid))
                {
                    local_vectors[node_id].erase(vid);
                    removed.insert(vid);
                }
            }
        }

        // 4. 重新分配local_vectors
        for (int vid : removed)
        {
    unordered_map<int, int> counts; // 节点 -> 分配的向量数量
            auto level_it = all_neighbors.find(0);
            std::unordered_set<int> neighbor_set;
            if (level_it != all_neighbors.end())
            {
                auto vid_it = level_it->second.find(vid);
                if (vid_it != level_it->second.end())
                    neighbor_set = vid_it->second;
            }
            for (int nid : neighbor_set)
            {
                auto route_it = vector_node_map[0].find(nid);
                if (route_it == vector_node_map[0].end() || route_it->second.empty())
                    continue;
                int neighbor_node = route_it->second[0];
                if (local_vectors[neighbor_node].count(nid))
                    counts[neighbor_node]++;
            }
            int best_node = -1;
            if (!counts.empty())
            {
                // 选择邻居最多的节点；并列时选择当前存量最小的
                pair<int, int> best_pair = {-1, -1}; // (count, -size)
                for (auto &kv : counts)
                {
                    int node_id = kv.first;
                    int cnt = kv.second;
                    int node_size = local_vectors[node_id].size();
                    if (best_pair.first == -1 || make_pair(cnt, -node_size) > best_pair)
                    {
                        best_node = node_id;
                        best_pair = {cnt, -node_size};
                    }
                }
            }
            else
            {
                // 放到当前最小负载节点
                best_node = 0;
                size_t min_cnt = local_vectors[0].size();
                for (int nid = 1; nid < node_num; nid++)
                {
                    if (local_vectors[nid].size() < min_cnt)
                    {
                        min_cnt = local_vectors[nid].size();
                        best_node = nid;
                    }
                }
            }
            // 将向量数据放回到选定节点
            local_vectors[best_node][vid] = all_vectors[vid];
            vector_node_map[0][vid] = {best_node};
        }

        return (int)removed.size();
    }

    void partition(const string &mode, int replication = 1, const std::unordered_map<int, std::vector<int>> &layer_config = {}, const std::string &layer_partition_mode = "round_robin", const std::vector<bool> &layer_replicate_flags = {}, int num_threads = 0, const std::string &partition_distribution = "network-send", const std::string &partition_base = "")
    {
        partition_mode = mode;
        replication_level = replication;
        if (hnsw)
            hnsw->replication_level_ = replication;
        replication_num = 0;
        this->layer_partition_mode = layer_partition_mode;
        this->layer_replicate_flags = layer_replicate_flags;

        int max_level = hnsw->max_level;
        size_t cur_count = hnsw->get_cur_element_count();

        // local_vids: 节点编号 -> 该节点持有的 vid 集合（不拷贝向量数据，零内存开销）
        vector<unordered_set<int>> local_vids(node_num);
        vector<unordered_map<int, int>> local_vid_ring(node_num);
        auto add_local_vid = [&](int node_id, int vid, int ring) {
            if (node_id < 0 || node_id >= node_num)
                return;
            local_vids[node_id].insert(vid);
            auto it = local_vid_ring[node_id].find(vid);
            if (it == local_vid_ring[node_id].end() || ring < it->second)
                local_vid_ring[node_id][vid] = ring;
        };

        vector_node_map.clear();
        if (mode == "layer")
            vector_node_map.resize(max_level + 1);
        else
            vector_node_map.resize(1);

        if (mode == "round_robin")
        {
            for (size_t i = 0; i < cur_count; i++)
            {
                int vid = hnsw->get_external_label((int)i);
                int target = vid % node_num;
                vector_node_map[0][vid] = {target};
                add_local_vid(target, vid, 0);
            }
        }
        else if (mode == "k-means")
        {
            LOG_INFO_STREAM("[partition] Running k-means on " << cur_count << " vectors (zero-copy from HNSW)...");
            auto labels = kmeans_hnsw(hnsw, node_num, 100, hnsw->seed, num_threads);
            // kmeans_hnsw 按 internal_id 0..n-1 顺序返回 labels
            for (size_t i = 0; i < cur_count; i++)
            {
                int vid = hnsw->get_external_label((int)i);
                int target = labels[i];
                vector_node_map[0][vid] = {target};
                add_local_vid(target, vid, 0);
            }
        }
        else if (mode == "fennel")
        {
            LOG_INFO_STREAM("[partition] Running Fennel streaming partition on " << cur_count << " vectors (zero-copy from HNSW)...");
            auto labels = fennel_partition(hnsw, {}, 0, node_num, 3, 1.5, 1.1, hnsw->seed);
            // fennel_partition 全图模式按 internal_id 0..n-1 顺序返回 labels
            for (size_t i = 0; i < cur_count; i++)
            {
                int vid = hnsw->get_external_label((int)i);
                int target = labels[i];
                vector_node_map[0][vid] = {target};
                add_local_vid(target, vid, 0);
            }
        }
        else if (mode == "kmeans-fennel")
        {
            // 先 k-means 得到初始分区(向量局部性),再用 Fennel restream 沿图边精修
            LOG_INFO_STREAM("[partition] Running k-means seed for kmeans-fennel on " << cur_count << " vectors...");
            auto seed_labels = kmeans_hnsw(hnsw, node_num, 100, hnsw->seed, num_threads);
            LOG_INFO_STREAM("[partition] Refining k-means seed with Fennel restream...");
            auto labels = fennel_partition(hnsw, {}, 0, node_num, 3, 1.5, 1.1, hnsw->seed, &seed_labels);
            for (size_t i = 0; i < cur_count; i++)
            {
                int vid = hnsw->get_external_label((int)i);
                int target = labels[i];
                vector_node_map[0][vid] = {target};
                add_local_vid(target, vid, 0);
            }
        }
        else if (mode == "outer_ring")
        {
            // 先用 k-means 分配
            auto labels = kmeans_hnsw(hnsw, node_num, 100, hnsw->seed, num_threads);
            // 需要 local_vectors（含向量数据）给 outer_ring 使用
            vector<unordered_map<int, vector<T>>> local_vectors_tmp(node_num);
            for (size_t i = 0; i < cur_count; i++)
            {
                int vid = hnsw->get_external_label((int)i);
                int target = labels[i];
                vector_node_map[0][vid] = {target};
                add_local_vid(target, vid, 0);
                local_vectors_tmp[target][vid] = hnsw->get_vector(vid);
            }

            // 挑选边界向量，重新分配
            int removed_cnt_before = 0;
            for (int it = 0; it < outer_ring_max_iterations; it++)
            {
                int removed_cnt = outer_ring(local_vectors_tmp);
                double removed_add_ratio = (removed_cnt_before > 0) ? double(removed_cnt - removed_cnt_before) / removed_cnt_before : 1.0;
                if (removed_cnt == 0 || removed_add_ratio < 0.001)
                    break;
                removed_cnt_before = removed_cnt;
            }
            // 从 outer_ring 结果重建 local_vids
            for (int i = 0; i < node_num; i++)
            {
                local_vids[i].clear();
                local_vid_ring[i].clear();
                for (const auto &kv : local_vectors_tmp[i])
                    add_local_vid(i, kv.first, 0);
            }
        }
        else if (mode == "layer")
        {
            this->layer_config = layer_config;

            // 直接从 HNSW 内部结构构建 level → vids 映射（不调 get_all_neighbors）
            LOG_INFO_STREAM("[partition] Building level-vid mapping from HNSW (" << cur_count << " elements)...");
            std::unordered_map<int, std::vector<std::pair<int, int>>> level_vids; // level → [(internal_id, vid)]
            for (size_t i = 0; i < cur_count; i++)
            {
                int vid = hnsw->get_external_label((int)i);
                int elem_level = hnsw->element_levels_[i];
                for (int level = 0; level <= elem_level; level++)
                    level_vids[level].emplace_back((int)i, vid);
            }

            // 初始化路由表（先给第 0 层占位）
            for (const auto &p : level_vids[0])
                vector_node_map[0][p.second] = {0};

            for (const auto &level_kv : level_vids)
            {
                int level = level_kv.first;
                const auto &vids_at_level = level_kv.second;
                std::vector<int> target_nodes;

                if (layer_config.count(level))
                {
                    target_nodes = layer_config.at(level);
                }
                else
                {
                    for (int i = 0; i < node_num; i++)
                        target_nodes.push_back(i);
                }

                if (target_nodes.empty())
                    continue;

                bool is_replicate_level = (level >= 0 && level < (int)this->layer_replicate_flags.size())
                                              ? this->layer_replicate_flags[level]
                                              : false;

                std::unordered_map<int, int> vid_partition_map;
                if (!is_replicate_level && layer_partition_mode == "k-means" && target_nodes.size() > 1)
                {
                    // 收集当前层的向量指针用于 kmeans（零拷贝）
                    std::vector<const T*> subset_ptrs;
                    std::vector<int> subset_vids;
                    subset_ptrs.reserve(vids_at_level.size());
                    subset_vids.reserve(vids_at_level.size());
                    for (const auto &p : vids_at_level)
                    {
                        subset_ptrs.push_back((const T*)hnsw->get_data_by_id(p.first));
                        subset_vids.push_back(p.second);
                    }

                    int sub_cnt = (int)subset_ptrs.size();
                    if (sub_cnt > 0)
                    {
                        int k = std::min((int)target_nodes.size(), sub_cnt);
                        auto labels = kmeans_core(subset_ptrs, sub_cnt, hnsw->dim, k, 100, hnsw->seed, num_threads);
                        for (int i = 0; i < sub_cnt; i++)
                            vid_partition_map[subset_vids[i]] = labels[i];
                    }
                }
                else if (!is_replicate_level && layer_partition_mode == "fennel" && target_nodes.size() > 1)
                {
                    // 用该层 internal_id 作 members，按 level 层 linklist 跑 Fennel 流式分区（零拷贝）
                    std::vector<int> members;
                    std::vector<int> subset_vids;
                    members.reserve(vids_at_level.size());
                    subset_vids.reserve(vids_at_level.size());
                    for (const auto &p : vids_at_level)
                    {
                        members.push_back(p.first);      // internal_id
                        subset_vids.push_back(p.second); // vid
                    }

                    int sub_cnt = (int)members.size();
                    if (sub_cnt > 0)
                    {
                        int k = std::min((int)target_nodes.size(), sub_cnt);
                        auto labels = fennel_partition(hnsw, members, level, k, 3, 1.5, 1.1, hnsw->seed);
                        for (int i = 0; i < sub_cnt; i++)
                            vid_partition_map[subset_vids[i]] = labels[i];
                    }
                }
                else if (!is_replicate_level && layer_partition_mode == "kmeans-fennel" && target_nodes.size() > 1)
                {
                    // 该层先 k-means(向量)播种,再 Fennel(图边)restream 精修
                    std::vector<int> members;
                    std::vector<int> subset_vids;
                    std::vector<const T*> subset_ptrs;
                    members.reserve(vids_at_level.size());
                    subset_vids.reserve(vids_at_level.size());
                    subset_ptrs.reserve(vids_at_level.size());
                    for (const auto &p : vids_at_level)
                    {
                        members.push_back(p.first);
                        subset_vids.push_back(p.second);
                        subset_ptrs.push_back((const T*)hnsw->get_data_by_id(p.first));
                    }

                    int sub_cnt = (int)members.size();
                    if (sub_cnt > 0)
                    {
                        int k = std::min((int)target_nodes.size(), sub_cnt);
                        auto seed_labels = kmeans_core(subset_ptrs, sub_cnt, hnsw->dim, k, 100, hnsw->seed, num_threads);
                        auto labels = fennel_partition(hnsw, members, level, k, 3, 1.5, 1.1, hnsw->seed, &seed_labels);
                        for (int i = 0; i < sub_cnt; i++)
                            vid_partition_map[subset_vids[i]] = labels[i];
                    }
                }

                LOG_INFO_STREAM("[layer partition] level="<< level << " vids=" << vids_at_level.size() << " mode=" << (is_replicate_level ? "replicate" : "split") << " target_nodes=[" << [&]{ std::string s; for(int n:target_nodes) s+=std::to_string(n)+","; return s; }() << "]");
                for (const auto &p : vids_at_level)
                {
                    int vid = p.second;

                    if (is_replicate_level)
                    {
                        int primary_node = -1;
                        int replica_count = 0;
                        for (int target_node : target_nodes)
                        {
                            if (target_node < 0 || target_node >= node_num)
                                continue;
                            add_local_vid(target_node, vid, 0);
                            if (primary_node < 0)
                                primary_node = target_node;
                            replica_count++;
                        }
                        if (primary_node >= 0)
                        {
                            if (replica_count > 1)
                                replication_num += (uint64_t)(replica_count - 1);
                            auto &owners = vector_node_map[level][vid];
                            owners.clear();
                            for (int target_node : target_nodes)
                                if (target_node >= 0 && target_node < node_num)
                                    owners.push_back(target_node);
                            normalize_route_owners(owners);
                        }
                    }
                    else
                    {
                        int target_idx = 0;
                        if (!vid_partition_map.empty() && vid_partition_map.count(vid))
                            target_idx = vid_partition_map.at(vid);
                        else
                            target_idx = vid % target_nodes.size();

                        if (target_idx < 0 || target_idx >= (int)target_nodes.size())
                            target_idx = 0;

                        int target_node = target_nodes[target_idx];
                        if (target_node >= 0 && target_node < node_num)
                        {
                            add_local_vid(target_node, vid, 0);
                            vector_node_map[level][vid] = {target_node};
                        }
                    }
                }
            }

            // 修正 entry_point 的路由
            if (layer_config.count(max_level) && !layer_config.at(max_level).empty()) {
                bool ep_replicate = (max_level >= 0 && max_level < (int)this->layer_replicate_flags.size())
                                        ? this->layer_replicate_flags[max_level]
                                        : false;
                if (ep_replicate)
                {
                    auto &owners = vector_node_map[max_level][hnsw->entry_point_id];
                    owners.clear();
                    for (int target_node : layer_config.at(max_level))
                        if (target_node >= 0 && target_node < node_num)
                            owners.push_back(target_node);
                    normalize_route_owners(owners);
                }
                else
                {
                    int target_node = layer_config.at(max_level)[hnsw->entry_point_id % layer_config.at(max_level).size()];
                    vector_node_map[max_level][hnsw->entry_point_id] = {target_node};
                }
            }
        }

        // 冗余机制：直接遍历 HNSW linklist，将远程邻居的 vid 添加到本地节点
        if (replication > 0)
        {
            int nthreads = (num_threads > 0) ? num_threads
                         : std::max(1, (int)std::thread::hardware_concurrency());
            size_t n = hnsw->get_cur_element_count();
            LOG_INFO_STREAM("[partition] Adding replication (direct HNSW traversal)... n=" << n
                            << " num_threads=" << num_threads << " nthreads=" << nthreads);

            // 复制策略：VR_REPL_BUDGET>0 启用加权(targeted)复制——按 cross-in-degree
            // (该节点本地有多少向量指向该候选 = 复制它能消除多少潜在 transfer)排序,
            // 仅取 top-(budget*|local|) 个高价值候选;未设/<=0 则均匀复制(原行为)。
            // VR_REPL_FULL_ROUNDS：前 N 轮仍全量复制(1-hop ring 几乎全必需),只对之后的
            // 高跳(2-hop+)轮做预算加权。默认 1 = 保 rep1 完整 + 只对 2-hop 切片加权。
            double repl_budget = 0.0;
            if (const char *e = std::getenv("VR_REPL_BUDGET")) repl_budget = std::atof(e);
            int full_rounds = 1;
            if (const char *e = std::getenv("VR_REPL_FULL_ROUNDS")) full_rounds = std::atoi(e);
            const bool weighted_repl = repl_budget > 0.0;
            // VR_REPL_INDEG_THRESH=τ：全局阈值模式(repN-wM 的实现)——复制所有 cross-in-degree>=τ
            // 的候选,不分跳;连通由逐跳 frontier 保证。这是"整体按价值挑"的等价物,优先级最高。
            int global_thresh = 0;
            if (const char *e = std::getenv("VR_REPL_INDEG_THRESH")) global_thresh = std::atoi(e);
            // VR_REPL_W=M：repN-wM 的全局预算(0<M<1)。在 round0 扫候选 cross-in-degree 直方图,
            // 自动选 τ 使保留 ≈ top M%(整数 τ 粒度,取 frac>=M 的最小 τ),再对各跳统一用该 τ。
            double repl_w = 0.0;
            if (const char *e = std::getenv("VR_REPL_W")) repl_w = std::atof(e);
            const bool w_mode = repl_w > 0.0 && repl_w < 1.0 && global_thresh == 0;
            // VR_REPL_BUDGETS="0.85,0.1"：每跳独立预算(>=1.0 即该跳全量),优先于上面单值逻辑。
            // 表达"两层都部分",如 rep0.85-0.1。被丢的低价值点会作为已复制点的邻居在下一轮
            // 重新成为候选(=「下一轮可再争取」语义,非严格 frontier)。
            std::vector<double> repl_budgets;
            if (const char *e = std::getenv("VR_REPL_BUDGETS"))
            {
                std::string s(e), tok;
                std::stringstream ss(s);
                while (std::getline(ss, tok, ','))
                    if (!tok.empty()) repl_budgets.push_back(std::atof(tok.c_str()));
            }
            {
                std::ostringstream bs;
                for (size_t x = 0; x < repl_budgets.size(); x++) bs << (x ? "," : "") << repl_budgets[x];
                LOG_INFO_STREAM("[partition] replication mode="
                                << (weighted_repl || !repl_budgets.empty() ? "weighted" : "uniform")
                                << " budget_ratio=" << repl_budget << " full_rounds=" << full_rounds
                                << " per_round_budgets=[" << bs.str() << "] rounds=" << replication);
            }

            // 外层 replication 轮串行（下一轮依赖上一轮扩张后的 local_vids/vector_node_map）；
            // 内层 vid 维度并行：每线程持有 per-thread 计数缓冲(候选 -> cross-in-degree)，循环结束归约。
            for (int r = 0; r < replication; r++)
            {
                // 本轮预算:优先用 VR_REPL_BUDGETS[r](每跳独立);否则回退 full_rounds+单值。
                double round_budget;
                bool round_weighted;
                if (!repl_budgets.empty())
                {
                    round_budget = repl_budgets[std::min((size_t)r, repl_budgets.size() - 1)];
                    round_weighted = round_budget < 1.0 - 1e-9; // >=1.0 即该跳全量
                }
                else
                {
                    round_budget = repl_budget;
                    round_weighted = weighted_repl && (r >= full_rounds);
                }
                std::vector<std::vector<std::unordered_map<int, int>>> thread_counts(
                    nthreads, std::vector<std::unordered_map<int, int>>(node_num));
                std::vector<std::vector<std::unordered_map<int, std::vector<int>>>> thread_route_updates(
                    nthreads, std::vector<std::unordered_map<int, std::vector<int>>>(vector_node_map.size()));

                {
                    std::vector<std::thread> threads;
                    size_t chunk = (n + (size_t)nthreads - 1) / (size_t)nthreads;
                    for (int t = 0; t < nthreads; t++)
                    {
                        size_t start = (size_t)t * chunk;
                        size_t end = std::min(start + chunk, n);
                        if (start >= n) break;
                        threads.emplace_back([&, t, start, end]() {
                            auto &my_counts = thread_counts[t];
                            auto &my_route = thread_route_updates[t];
                            for (size_t i = start; i < end; i++)
                            {
                                int vid = hnsw->get_external_label((int)i);
                                int elem_level = hnsw->element_levels_[i];

                                for (int level = 0; level <= elem_level; level++)
                                {
                                    if (level < 0 || level >= (int)vector_node_map.size())
                                        continue;

                                    const auto &route_map = vector_node_map[level];
                                    auto it = route_map.find(vid);
                                    if (it == route_map.end())
                                        continue;

                                    int *ll = (level == 0) ? hnsw->get_linklist0((int)i) : hnsw->get_linklist((int)i, level);
                                    int sz = hnsw->get_list_count(ll);
                                    int *data = ll + 1;

                                    for (int node_id : it->second)
                                    {
                                        if (node_id < 0 || node_id >= node_num)
                                            continue;
                                        for (int j = 0; j < sz; j++)
                                        {
                                            int neighbor_vid = hnsw->get_external_label(data[j]);
                                            if (local_vids[node_id].count(neighbor_vid))
                                                continue; // 已在该节点,无需复制
                                            // cross-in-degree：每条 (本地向量 -> 该远程候选) 边计 1 次
                                            int prev = my_counts[node_id][neighbor_vid]++;
                                            if (prev == 0)
                                            {
                                                auto &route = my_route[level][neighbor_vid];
                                                if (std::find(route.begin(), route.end(), node_id) == route.end())
                                                    route.push_back(node_id);
                                            }
                                        }
                                    }
                                }
                            }
                        });
                    }
                    for (auto &th : threads) th.join();
                }

                // 归约：合并每节点候选计数(cross-in-degree) + 路由候选
                std::vector<std::unordered_map<int, int>> counts(node_num);
                vector<unordered_map<int, std::vector<int>>> route_updates(vector_node_map.size());
                for (int t = 0; t < nthreads; t++)
                {
                    for (int node_id = 0; node_id < node_num; node_id++)
                        for (const auto &kv : thread_counts[t][node_id])
                            counts[node_id][kv.first] += kv.second;
                    for (size_t lvl = 0; lvl < thread_route_updates[t].size(); lvl++)
                        for (const auto &kv : thread_route_updates[t][lvl])
                        {
                            auto &route = route_updates[lvl][kv.first];
                            for (int node_id : kv.second)
                                if (std::find(route.begin(), route.end(), node_id) == route.end())
                                    route.push_back(node_id);
                        }
                }

                // VR_REPL_W：用 round0 候选直方图自动标定全局阈值 τ(只标定一次,后续轮沿用)
                if (w_mode && global_thresh == 0)
                {
                    std::map<int, long long> hist; // cross-in-degree -> 候选数
                    long long tot = 0;
                    for (int node_id = 0; node_id < node_num; node_id++)
                        for (const auto &kv : counts[node_id]) { hist[kv.second]++; tot++; }
                    long long target = (long long)(repl_w * (double)tot);
                    long long acc = 0; int tau = 1;
                    for (auto it = hist.rbegin(); it != hist.rend(); ++it)
                    {
                        acc += it->second;
                        tau = it->first;
                        if (acc >= target) break;
                    }
                    global_thresh = std::max(1, tau);
                    LOG_INFO_STREAM("[partition] VR_REPL_W=" << repl_w << " → global τ=" << global_thresh
                                    << " (round0 calib: keep value>=τ, achieved_frac≈"
                                    << (tot ? (double)acc / tot : 0.0) << ")");
                }

                // 选择：均匀=全选；加权=每节点按 count 降序取 top-(budget*|local|)
                std::vector<std::unordered_set<int>> selected(node_num);
                size_t total_sel = 0, total_cand = 0;
                for (int node_id = 0; node_id < node_num; node_id++)
                {
                    auto &cmap = counts[node_id];
                    total_cand += cmap.size();
                    if (global_thresh > 0)
                    {
                        // 全局阈值：复制所有 cross-in-degree >= τ 的候选(整体按价值,跨跳统一)
                        for (const auto &kv : cmap)
                            if (kv.second >= global_thresh)
                                selected[node_id].insert(kv.first);
                    }
                    else if (!round_weighted)
                    {
                        for (const auto &kv : cmap)
                            selected[node_id].insert(kv.first);
                    }
                    else
                    {
                        // budget = 保留该跳候选中价值最高的比例(rep0.85 = top 85% 候选)
                        size_t B = std::max<size_t>(1, (size_t)(round_budget * (double)cmap.size()));
                        if (cmap.size() <= B)
                        {
                            for (const auto &kv : cmap)
                                selected[node_id].insert(kv.first);
                        }
                        else
                        {
                            std::vector<std::pair<int, int>> cand; // (count, vid)
                            cand.reserve(cmap.size());
                            for (const auto &kv : cmap)
                                cand.emplace_back(kv.second, kv.first);
                            std::partial_sort(cand.begin(), cand.begin() + B, cand.end(),
                                              [](const std::pair<int, int> &a, const std::pair<int, int> &b)
                                              { return a.first > b.first; });
                            for (size_t x = 0; x < B; x++)
                                selected[node_id].insert(cand[x].second);
                        }
                    }
                    total_sel += selected[node_id].size();
                }

                // 提交：加本地 + 仅为 selected 的候选写路由
                int new_ring = r + 1;
                for (int node_id = 0; node_id < node_num; node_id++)
                    for (int vid : selected[node_id])
                    {
                        add_local_vid(node_id, vid, new_ring);
                        replication_num++;
                    }
                for (size_t lvl = 0; lvl < route_updates.size(); lvl++)
                    for (const auto &kv : route_updates[lvl])
                    {
                        int vid = kv.first;
                        for (int node_id : kv.second)
                            if (selected[node_id].count(vid))
                                append_route_owner((int)lvl, vid, node_id);
                    }

                LOG_INFO_STREAM("[partition] Redundancy round " << r << " done, mode="
                                << (global_thresh > 0 ? "thresh" : (round_weighted ? "weighted" : "uniform"))
                                << " thresh=" << global_thresh
                                << " selected=" << total_sel << "/" << total_cand
                                << " frac=" << (total_cand ? (double)total_sel / total_cand : 0.0)
                                << " replication_num=" << replication_num);
            }
        }

        // 打印各节点分到的向量数
        for (int i = 0; i < node_num; i++)
            LOG_INFO_STREAM("[partition] node " << i << " local_vectors=" << local_vids[i].size());
        log_route_memory_breakdown("[partition before distribute]");

        // 构建每个节点的 local_vids 集合
        std::vector<std::unordered_set<int>> node_local_vids(node_num);
        for (int i = 0; i < node_num; i++)
        {
            if (hnsw->entry_point_id >= 0)
                add_local_vid(i, hnsw->get_external_label(hnsw->entry_point_id), 0);
            node_local_vids[i] = std::move(local_vids[i]);
        }

        normalize_routes_primary_first();

        if (partition_distribution == "nfs-load")
        {
            if (partition_base.empty())
                throw std::runtime_error("partition nfs-load requires non-empty partition_base");
            LOG_INFO_STREAM("[partition] Materializing partition files to NFS: " << partition_base << "-*.bin");
            for (int i = 0; i < node_num; i++)
            {
                std::string filepath = partition_base + "-" + std::to_string(i) + ".bin";
                if (!write_partition_file(filepath, node_local_vids[i], &local_vid_ring[i]))
                    throw std::runtime_error("failed to write partition file: " + filepath);
            }
            log_route_memory_breakdown("[partition nfs-load done]");
            return;
        }
        if (partition_distribution != "network-send")
            throw std::runtime_error("unknown partition_distribution: " + partition_distribution);

        // 释放旧的 local_hnsw
        for (int i = 0; i < node_num; i++)
        {
            if (nodes[i].local_hnsw != nullptr && nodes[i].local_hnsw != hnsw)
            {
                delete nodes[i].local_hnsw;
                nodes[i].local_hnsw = nullptr;
            }
        }
        LOG_INFO_STREAM("[partition] Distributing subset HNSW graphs (full adjacency + local vectors only)");

    // 将分区数据分发到各远程工作节点
        for (int i = 0; i < node_num; i++)
        {
            Node<T> &target_node = nodes[i];
            if (target_node.is_remote_node)
            {
                LOG_INFO_STREAM("[partition] Connecting to remote node " << i << " (" << target_node.ip_addr << ":" << (target_node.port + Node<T>::NODE_PORT_OFFSET) << ")...");
                // 使用通信中间层获取连接
                int sock = transport->acquire_connection(i);
                if (sock < 0)
                {
                    LOG_DEBUG_STREAM("Failed to get connection to node " << i);
                    continue;
                }
                LOG_INFO_STREAM("[partition] Connected to node " << i << ", sock=" << sock);

                Buffer buf;
                buf.write_val((int)MSG_INIT_GRAPH);
                buf.write_val(is_global_pull);  // 通知远程节点是否为 global-pull 模式（跳过 compact）
                serialize_routing(buf, vector_node_map);
                serialize_hnsw_subset(buf, hnsw, node_local_vids[i], &local_vid_ring[i], replication_level);

                uint64_t msg_size = buf.data.size();
                LOG_INFO_STREAM("[partition] Sending MSG_INIT_GRAPH to node " << i << ", msg_size=" << msg_size << " bytes");
                transport->send_raw(sock, &msg_size, sizeof(uint64_t));
                LOG_INFO_STREAM("[partition] Sent msg_size header to node " << i << ", now sending payload...");
                transport->send_raw(sock, buf.data.data(), msg_size);
                LOG_INFO_STREAM("[partition] Sent payload to node " << i << ", waiting for ACK...");

    // 等待工作节点返回确认响应
                char ack = 0;
                try
                {
                    transport->recv_raw(sock, &ack, 1);
                    if (ack == 1)
                    {
                        LOG_INFO_STREAM("[partition] Node " << i << " ACK received OK");
                    }
                    else
                    {
                        LOG_DEBUG_STREAM("Warning: Invalid ACK from node " << i << ", ack=" << (int)ack);
                    }
                }
                catch (const std::exception &e)
                {
                    LOG_DEBUG_STREAM("Warning: No ACK from node " << i << ": " << e.what());
                }
                catch (...)
                {
                    LOG_DEBUG_STREAM("Warning: No ACK from node " << i);
                }
    // MSG_INIT_GRAPH 响应后服务器会关闭连接，此处不复用
                transport->close_connection(sock);
            }
            else
            {
                // 本地节点：通过序列化/反序列化深拷贝子集图
                Buffer buf;
                serialize_routing(buf, vector_node_map);
                serialize_hnsw_subset(buf, hnsw, node_local_vids[i], &local_vid_ring[i], replication_level);
                deserialize_routing(buf, vector_node_map);

                target_node.local_hnsw = deserialize_hnsw_subset<T>(buf, /*do_compact=*/!is_global_pull);
                if (target_node.local_hnsw)
                    target_node.local_hnsw->log_memory_breakdown("[partition local node " + std::to_string(i) + "]");
                log_route_memory_breakdown("[partition local node " + std::to_string(i) + "]");

                // global-pull 模式下注入远程向量拉取回调
                if (is_global_pull && target_node.local_hnsw)
                {
                    Cluster<T> *cl = this;
                    target_node.local_hnsw->pull_fetcher_ = [cl](int vid) -> std::vector<T> {
                        return cl->fetch_remote_vector(vid);
                    };
                }
            }
        }
    }

    std::vector<int> query(int query_id, const vector<T> &query_vector, Payload<DistT> &payload, int k)
    {
        if (is_shard)
        {
            std::vector<pair<DistT, int>> all_results;

    // Shard 模式查询：并行向所有节点发送查询请求并合并结果
            struct NodeQueryOutput
            {
                std::vector<pair<DistT, int>> results;
                Payload<DistT> payload;
            };

            std::vector<std::future<NodeQueryOutput>> futs;
            futs.reserve(node_num);
 
            for (int i = 0; i < node_num; i++)
            {
                futs.emplace_back(std::async(std::launch::async, [this, query_id, &query_vector, k, i]() -> NodeQueryOutput {
                    NodeQueryOutput out;
                    Node<T> &node = nodes[i];

                    // 每个节点使用独立payload，避免并发写同一个payload
                    Payload<DistT> node_payload;

                    if (is_distributed && node.is_remote_node)
                    {
                        // 使用通信中间层获取连接
                        int sock = transport->acquire_connection(i);
                        if (sock < 0)
                        {
                            out.payload = std::move(node_payload);
                            return out;
                        }

                        LOG_DEBUG_STREAM("[Cluster] Sending query " << query_id << " to remote node " << i);

                        Buffer buf;
                        buf.write_val((int)MSG_QUERY_LOCAL);  // 使用 MSG_QUERY_LOCAL 避免递归
                        buf.write_val(query_id);
                        buf.write_val(k);
                        buf.write_vec(query_vector);
                        // node_payload 是空堆，传 nullptr 即可（serialize 会跳过转换）
                        serialize_payload(buf, node_payload, (const HNSWGraph<T>*)nullptr);

                        uint64_t msg_size = buf.data.size();
                        try
                        {
                            transport->send_raw(sock, &msg_size, sizeof(uint64_t));
                            transport->send_raw(sock, buf.data.data(), msg_size);

                            uint64_t resp_size = 0;
                            transport->recv_raw(sock, &resp_size, sizeof(uint64_t));

                            std::vector<char> resp_data(resp_size);
                            transport->recv_raw(sock, resp_data.data(), resp_size);

                            LOG_DEBUG_STREAM("[Cluster] Received response for query " << query_id << " from remote node " << i);

                            // 持久连接：远程节点 handle_client 继续等待下一个消息，归还连接池复用
                            transport->release_connection(i, sock, true);

                            Buffer resp_buf;
                            resp_buf.data = std::move(resp_data);

                            int result_cnt = resp_buf.read_val<int>();
                            out.results.reserve(result_cnt);
                            for (int j = 0; j < result_cnt; j++)
                            {
                                DistT dist = resp_buf.read_val<DistT>();
                                int id = resp_buf.read_val<int>();
                                out.results.push_back({dist, id});
                            }
                            // 堆内容对本节点不需要；传 nullptr，项按 -(vid+1) 编码（不再使用）
                            deserialize_payload(resp_buf, node_payload, (const HNSWGraph<T>*)nullptr);
                        }
                        catch (...)
                        {
                            transport->release_connection(i, sock, false);
                            out.payload = std::move(node_payload);
                            return out;
                        }
                    }
                    else
                    {
                        if (node.local_hnsw)
                        {
                            node.local_hnsw->query(query_vector, node_payload, k);

                            // 堆内存 iid 编码，pop 时解码成 vid 再入 out.results
                            auto pq = node_payload.results;
                            out.results.reserve(pq.size());
                            while (!pq.empty())
                            {
                                auto item = pq.top();
                                out.results.push_back({item.first, node.local_hnsw->decode_key(item.second)});
                                pq.pop();
                            }
                        }
                    }

                    out.payload = std::move(node_payload);
                    return out;
                }));
            }

            for (auto &f : futs)
            {
                NodeQueryOutput out;
                try
                {
                    out = f.get();
                }
                catch (...)
                {
                    continue;
                }

                all_results.insert(all_results.end(), out.results.begin(), out.results.end());
                payload.distance_cnt += out.payload.distance_cnt;
                payload.remote_transfer_cnt += out.payload.remote_transfer_cnt;
                payload.remote_transfer_data += out.payload.remote_transfer_data;
                payload.backtrack_cnt += out.payload.backtrack_cnt;
            }

            // 对所有结果排序并返回top-k
            std::sort(all_results.begin(), all_results.end());
            std::vector<int> final_results;
            size_t limit = std::min((size_t)k, all_results.size());
            for (size_t i = 0; i < limit; i++)
            {
                final_results.push_back(all_results[i].second);
            }

            return final_results;
        }

        // 非reduce模式，返回空向量（或可以抛出异常）
        return std::vector<int>();
    }

    // global-pull 模式：按 vid 从拥有该向量数据的节点同步拉取原始向量数据。
    // 路由依据 vector_node_map[0]（底层分区）。若拥有者是本地节点，直接从 local_hnsw->get_vector 读取；
    // 否则通过 MSG_FETCH_VECTOR 发送同步 RPC，阻塞等待响应。
    // 注意：此函数不更新 payload 统计；调用方（HNSW 搜索层）负责记账 remote_transfer_cnt / remote_transfer_data。
    std::vector<T> fetch_remote_vector(int vid)
    {
        int owner_node_id = 0;
        if (!lookup_primary_owner(0, vid, owner_node_id))
        {
            LOG_DEBUG_STREAM("[Cluster] fetch_remote_vector: vid " << vid << " not in routing map");
            return {};
        }

        Node<T> &owner = nodes[owner_node_id];

        // 拥有者是本地节点：直接读取，不走网络
        if (!is_distributed || !owner.is_remote_node)
        {
            if (owner.local_hnsw)
                return owner.local_hnsw->get_vector(vid);
            return {};
        }

        // 远程节点：通过 MSG_FETCH_VECTOR 同步拉取
        int sock = transport->acquire_connection(owner_node_id);
        if (sock < 0)
        {
            LOG_DEBUG_STREAM("[Cluster] fetch_remote_vector: failed to get connection to node " << owner_node_id);
            return {};
        }

        try
        {
            Buffer buf;
            buf.write_val((int)MSG_FETCH_VECTOR);
            buf.write_val(vid);

            uint64_t msg_size = buf.data.size();
            transport->send_raw(sock, &msg_size, sizeof(uint64_t));
            transport->send_raw(sock, buf.data.data(), msg_size);

            uint64_t resp_size = 0;
            transport->recv_raw(sock, &resp_size, sizeof(uint64_t));

            std::vector<char> resp_data(resp_size);
            if (resp_size > 0)
                transport->recv_raw(sock, resp_data.data(), resp_size);

            transport->release_connection(owner_node_id, sock, true);

            Buffer resp_buf;
            resp_buf.data = std::move(resp_data);
            return resp_buf.read_vec<T>();
        }
        catch (...)
        {
            transport->release_connection(owner_node_id, sock, false);
            return {};
        }
    }

    // 转发链路任何失败时调用：把空结果送达 origin，避免 SDK 端 query_recv 永久阻塞。
    // 自动处理 origin 是本地（push 到本机 sdk_cq_map_）或远程（network send）两种情况。
    // 用空 payload 构造，统计字段全为 0；该 query 召回为 0，但全局 drain 不会卡住。
    void deliver_empty_result(int query_id, int origin_node_id)
    {
        if (origin_node_id < 0 || origin_node_id >= node_num)
        {
            LOG_INFO_STREAM("[Cluster] deliver_empty_result: invalid origin_node_id="
                            << origin_node_id << " for query " << query_id);
            return;
        }
        Payload<DistT> empty_payload;
        empty_payload.origin_node_id = origin_node_id;
        // 通过任意本机 Node 调用 send_result_to_origin —— 路由由 origin_node_id 决定，
        // 调用者节点是哪个不影响投递结果
        for (int i = 0; i < node_num; i++)
        {
            if (!nodes[i].is_remote_node && nodes[i].local_hnsw)
            {
                nodes[i].send_result_to_origin(query_id, std::vector<int>{}, empty_payload);
                return;
            }
        }
        LOG_INFO_STREAM("[Cluster] deliver_empty_result: no local node found for query " << query_id);
    }

    // 跨节点传输：统计并将查询转发到目标节点
    std::vector<int> transfer_to_remote(int query_id, const vector<T> &query_vector, Payload<DistT> &payload, int k)
    {
        payload.remote_transfer_cnt += 1;

        int require_id = payload.require_id;
        int route_level = payload.start_level;
        if (route_level < 0 || route_level >= (int)vector_node_map.size())
            route_level = 0;

        int current_node_id = current_local_node_id();
        uint64_t route_salt = ((uint64_t)(uint32_t)query_id << 32) ^ (uint32_t)require_id ^ ((uint64_t)(uint32_t)route_level << 16);
        int target_node_id = 0;
        if (!choose_route_owner(route_level, require_id, current_node_id, route_salt, target_node_id))
        {
            if (route_level != 0)
                choose_route_owner(0, require_id, current_node_id, route_salt, target_node_id);
            else
                lookup_primary_owner(0, require_id, target_node_id);
        }

        Node<T> &target_node = nodes[target_node_id];

        // 折返检测：检查 target_node_id 是否出现在最近 max_track_num 次 transfer 的目标节点窗口中。
        // 命中则累加 backtrack_cnt；然后 FIFO 更新窗口。max_track_num == 0 时关闭统计。
        if (max_track_num > 0)
        {
            auto &recent = payload.recent_nodes;
            if (std::find(recent.begin(), recent.end(), target_node_id) != recent.end())
            {
                payload.backtrack_cnt += 1;
            }
            if ((int)recent.size() >= max_track_num)
            {
                recent.erase(recent.begin());
            }
            recent.push_back(target_node_id);
        }

        // 累计本次 transfer 的数据量（在序列化之前计算）
        payload.remote_transfer_data += payload.size_bits();

        if (is_distributed && target_node.is_remote_node)
        {
            // 记录发送时间
            auto now = std::chrono::system_clock::now();
            payload.send_timestamp = std::chrono::duration<double, std::milli>(now.time_since_epoch()).count();

            // 使用通信中间层获取连接
            int sock = transport->acquire_connection(target_node_id);
            if (sock < 0)
            {
                LOG_INFO_STREAM("[Cluster] transfer_to_remote: acquire_connection failed for query "
                                << query_id << " to node " << target_node_id
                                << "; delivering empty result to origin " << payload.origin_node_id);
                deliver_empty_result(query_id, payload.origin_node_id);
                return {};
            }

            Buffer buf;
            buf.write_val((int)MSG_SEARCH);
            buf.write_val(query_id);
            buf.write_val(k);
            buf.write_vec(query_vector);
            // 本地搜索产出的堆以 iid 编码，此处转回 wire vid
            // 注意：必须用本机持有的 local_hnsw（partition 后的子图）做 iid→vid 映射，
            // 不能用 cluster->hnsw（worker 节点上是 node_start.py 创建的空 dummy 图）。
            HNSWGraph<T> *src_local_hnsw = nullptr;
            for (int i = 0; i < node_num; i++)
            {
                if (!nodes[i].is_remote_node)
                {
                    src_local_hnsw = nodes[i].local_hnsw;
                    break;
                }
            }
            serialize_payload(buf, payload, src_local_hnsw);

            uint64_t msg_size = buf.data.size();

            LOG_DEBUG_STREAM("[Cluster] Sending MSG_SEARCH to node " << target_node_id << " for query " << query_id << ", transfer_cnt=" << payload.remote_transfer_cnt << ", msg_size=" << msg_size);

            try
            {
                double send_start_ms = steady_now_ms();
                transport->send_raw(sock, &msg_size, sizeof(uint64_t));
                transport->send_raw(sock, buf.data.data(), msg_size);
                payload.send_ms += steady_now_ms() - send_start_ms;

                transport->release_connection(target_node_id, sock, true);

                LOG_DEBUG_STREAM("[Cluster] MSG_SEARCH successfully sent to node " << target_node_id << " for query " << query_id);

                return {};
            }
            catch (...)
            {
                transport->release_connection(target_node_id, sock, false);
                LOG_INFO_STREAM("[Cluster] transfer_to_remote: send_raw threw for query "
                                << query_id << " to node " << target_node_id
                                << "; delivering empty result to origin " << payload.origin_node_id);
                deliver_empty_result(query_id, payload.origin_node_id);
                return {};
            }
        }
        else
        {
            return target_node.query(query_id, query_vector, payload, k);
        }
    }

    // 保存 cluster 状态到文件
    // 保存当前节点的分区子图 + routing（二进制紧凑格式）
    bool save(const string &filepath)
    {
        try
        {
            // 父目录可能不存在（例如新 node_num 的 save/.../nodeN/），先建出来；
            // 否则 ofstream 静默 fail 但不抛异常，老代码会误报 "saved" 实际没落盘。
            std::filesystem::path p(filepath);
            if (p.has_parent_path())
                std::filesystem::create_directories(p.parent_path());

            Buffer routing_buf;
            serialize_partition_routing(routing_buf);

            Buffer subset_buf;
            for (int i = 0; i < node_num; i++) {
                if (!nodes[i].is_remote_node && nodes[i].local_hnsw) {
                    std::unordered_set<int> local_vids_to_save;
                    size_t local_count = nodes[i].local_hnsw->get_cur_element_count();
                    local_vids_to_save.reserve(local_count);
                    for (size_t iid = 0; iid < local_count; iid++)
                        local_vids_to_save.insert(nodes[i].local_hnsw->get_external_label((int)iid));
                    serialize_hnsw_subset(subset_buf, nodes[i].local_hnsw,
                                           local_vids_to_save);
                    break;
                }
            }

            std::ofstream output_file(filepath, std::ios::binary);
            if (!output_file) {
                LOG_INFO_STREAM("[save] Failed to open " << filepath << " for writing");
                return false;
            }
            uint32_t magic = partition_file_magic();
            uint32_t version = 2;
            uint64_t routing_size = routing_buf.data.size();
            output_file.write((char*)&magic, sizeof(uint32_t));
            output_file.write((char*)&version, sizeof(uint32_t));
            output_file.write((char*)&routing_size, sizeof(uint64_t));
            output_file.write(routing_buf.data.data(), routing_buf.data.size());
            output_file.write(subset_buf.data.data(), subset_buf.data.size());
            if (!output_file) {
                LOG_INFO_STREAM("[save] Write failed for " << filepath);
                return false;
            }
            LOG_INFO_STREAM("[save] Wrote " << filepath
                            << " routing=" << HNSWGraph<T>::format_bytes(routing_buf.data.size())
                            << " subset=" << HNSWGraph<T>::format_bytes(subset_buf.data.size())
                            << " total=" << HNSWGraph<T>::format_bytes(sizeof(uint32_t) * 2 + sizeof(uint64_t) + routing_buf.data.size() + subset_buf.data.size()));
            return true;
        }
        catch (const std::exception &e) {
            LOG_INFO_STREAM("[save] Exception while saving " << filepath << ": " << e.what());
            return false;
        }
        catch (...) { return false; }
    }

    bool load_local(const string &filepath)
    {
        try
        {
            LOG_INFO_STREAM("[load_local] Opening file: " << filepath);
            std::ifstream input_file(filepath, std::ios::binary);
            if (!input_file) { LOG_INFO_STREAM("[load_local] Failed to open file"); return false; }

            input_file.seekg(0, std::ios::end);
            size_t total_size = input_file.tellg();
            input_file.seekg(0);
            if (total_size < sizeof(uint32_t))
                throw std::runtime_error("load_local: partition file too small");

            uint32_t magic = 0;
            input_file.read((char *)&magic, sizeof(uint32_t));
            if (!input_file)
                throw std::runtime_error("load_local: failed to read partition magic");
            size_t subset_offset = 0;
            if (magic == partition_file_magic())
            {
                constexpr size_t header_size = sizeof(uint32_t) * 2 + sizeof(uint64_t);
                if (total_size < header_size)
                    throw std::runtime_error("load_local: truncated binary partition header");
                uint32_t version = 0;
                uint64_t routing_size = 0;
                input_file.read((char *)&version, sizeof(uint32_t));
                input_file.read((char *)&routing_size, sizeof(uint64_t));
                if (!input_file)
                    throw std::runtime_error("load_local: failed to read binary partition header");
                if (version != 2)
                    throw std::runtime_error("load_local: unsupported partition file version");
                if (routing_size > total_size - header_size)
                    throw std::runtime_error("load_local: invalid routing section size");
                std::vector<char> routing_data((size_t)routing_size);
                if (routing_size > 0)
                    input_file.read(routing_data.data(), (size_t)routing_size);
                if (!input_file)
                    throw std::runtime_error("load_local: failed to read binary routing section");
                Buffer routing_buf;
                routing_buf.data = std::move(routing_data);
                deserialize_partition_routing(routing_buf);
                subset_offset = header_size + (size_t)routing_size;
                LOG_INFO_STREAM("[load_local] Parsed binary routing: routing="
                                << HNSWGraph<T>::format_bytes((size_t)routing_size));
            }
            else
            {
                input_file.seekg(0);
                if (total_size < sizeof(uint64_t))
                    throw std::runtime_error("load_local: truncated legacy partition header");
                uint64_t routing_json_length = 0;
                input_file.read((char*)&routing_json_length, sizeof(uint64_t));
                if (!input_file)
                    throw std::runtime_error("load_local: failed to read legacy routing JSON length");
                if (routing_json_length > total_size - sizeof(uint64_t))
                    throw std::runtime_error("load_local: invalid legacy routing JSON length");
                LOG_INFO_STREAM("[load_local] Loading legacy JSON routing: "
                                << HNSWGraph<T>::format_bytes((size_t)routing_json_length));
                std::string routing_json_string((size_t)routing_json_length, '\0');
                input_file.read(&routing_json_string[0], (size_t)routing_json_length);
                if (!input_file)
                    throw std::runtime_error("load_local: failed to read legacy routing JSON");
                json routing_json_obj = json::parse(routing_json_string);
                partition_mode = routing_json_obj.value("partition_mode", partition_mode);
                layer_partition_mode = routing_json_obj.value("layer_partition_mode", layer_partition_mode);
                replication_level = routing_json_obj.value("replication_level", replication_level);
                replication_num = routing_json_obj.value("replication_num", replication_num);
                if (!routing_json_obj.contains("layer_config") || !routing_json_obj["layer_config"].is_object())
                    throw std::runtime_error("load_local: missing layer_config");
                layer_config.clear();
                for (auto const &kv : routing_json_obj["layer_config"].items())
                    layer_config[std::stoi(kv.key())] = kv.value().get<std::vector<int>>();

                if (!routing_json_obj.contains("layer_replicate_flags") || !routing_json_obj["layer_replicate_flags"].is_array())
                    throw std::runtime_error("load_local: missing layer_replicate_flags");
                layer_replicate_flags = routing_json_obj["layer_replicate_flags"].get<std::vector<bool>>();

                if (!routing_json_obj.contains("vector_node_map") || !routing_json_obj["vector_node_map"].is_array())
                    throw std::runtime_error("load_local: missing vector_node_map");
                clear_dense_level0_routes();
                vector_node_map.clear();
                vector_node_map.resize(routing_json_obj["vector_node_map"].size());
                for (size_t level = 0; level < routing_json_obj["vector_node_map"].size(); level++)
                    for (auto const &kv : routing_json_obj["vector_node_map"][level].items())
                        vector_node_map[level][std::stoi(kv.key())] = kv.value().get<std::vector<int>>();
                normalize_routes_primary_first();
                filter_routes_for_runtime();
                subset_offset = sizeof(uint64_t) + (size_t)routing_json_length;
            }

            if (subset_offset >= total_size)
                throw std::runtime_error("load_local: missing HNSW subset section");
            size_t subset_size = total_size - subset_offset;
            input_file.seekg(subset_offset);
            std::vector<char> subset_data(subset_size);
            input_file.read(subset_data.data(), subset_size);
            if (!input_file)
                throw std::runtime_error("load_local: failed to read HNSW subset section");
            input_file.close();
            LOG_INFO_STREAM("[load_local] File read complete: " << subset_size << " bytes, deserializing...");

            Buffer subset_buf; subset_buf.data = std::move(subset_data);
            HNSWGraph<T> *loaded_graph = deserialize_hnsw_subset<T>(subset_buf, /*do_compact=*/true);
            LOG_INFO_STREAM("[load_local] Deserialized, elements=" << loaded_graph->get_cur_element_count() << ", assigning to node...");
            log_route_memory_breakdown("[load_local]");
            loaded_graph->log_memory_breakdown("[load_local]");
            for (int i = 0; i < node_num; i++)
            {
                if (!nodes[i].is_remote_node)
                {
                    if (nodes[i].local_hnsw) delete nodes[i].local_hnsw;
                    nodes[i].local_hnsw = loaded_graph;
                    return true;
                }
            }
            delete loaded_graph;
            return false;
        }
        catch (const std::exception &e)
        {
            LOG_INFO_STREAM("[load_local] Exception while loading " << filepath << ": " << e.what());
            return false;
        }
        catch (...)
        {
            LOG_INFO_STREAM("[load_local] Unknown exception while loading " << filepath);
            return false;
        }
    }

    // 从文件加载 cluster 状态
    static Cluster<T> *load(const string &filepath)
    {
        std::ifstream ifs(filepath, std::ios::binary);
        if (!ifs)
            throw std::runtime_error("failed to open cluster file: " + filepath);
        json j;
        ifs >> j;

        // 加载主 HNSW 图
        HNSWGraph<T> *hnsw = nullptr;
        if (j.contains("hnsw_path") && !j["hnsw_path"].get<string>().empty())
        {
            string hnsw_path = j["hnsw_path"].get<string>();
            hnsw = HNSWGraph<T>::load(hnsw_path);
        }

        int node_num = j.value("node_num", 4);
        Cluster<T> *cluster = new Cluster<T>(hnsw, node_num);
        cluster->owns_hnsw = (hnsw != nullptr);

        cluster->bandwidth = j.value("bandwidth", 1000.0);
        cluster->partition_mode = j.value("partition_mode", "");
        cluster->layer_partition_mode = j.value("layer_partition_mode", "");
        cluster->outer_ring_max_iterations = j.value("outer_ring_max_iterations", 0);
        cluster->replication_level = j.value("replication_level", 0);
        cluster->replication_num = j.value("replication_num", (uint64_t)0);
        cluster->is_distributed = j.value("is_distributed", false);
        cluster->is_shard = j.value("is_shard", false);
        cluster->is_global_pull = j.value("is_global_pull", false);
        cluster->add_round_robin_counter = j.value("add_round_robin_counter", 0);

        if (cluster->partition_mode == "layer")
        {
            if (!j.contains("layer_config") || !j["layer_config"].is_object())
                throw std::runtime_error("cluster load: missing layer_config");
            cluster->layer_config.clear();
            for (auto const &kv : j["layer_config"].items())
                cluster->layer_config[std::stoi(kv.key())] = kv.value().get<std::vector<int>>();

            if (!j.contains("layer_replicate_flags") || !j["layer_replicate_flags"].is_array())
                throw std::runtime_error("cluster load: missing layer_replicate_flags");
            cluster->layer_replicate_flags = j["layer_replicate_flags"].get<std::vector<bool>>();
        }

        if (!j.contains("vector_node_map") || !j["vector_node_map"].is_array())
            throw std::runtime_error("cluster load: missing vector_node_map");
        cluster->vector_node_map.clear();
        cluster->vector_node_map.resize(j["vector_node_map"].size());
        for (size_t l = 0; l < j["vector_node_map"].size(); l++)
            for (auto const &kv : j["vector_node_map"][l].items())
                cluster->vector_node_map[l][std::stoi(kv.key())] = kv.value().get<std::vector<int>>();

        cluster->normalize_routes_primary_first();
        cluster->filter_routes_for_runtime();

        // 加载每个节点的状态
        if (j.contains("nodes"))
        {
            const json &jnodes = j["nodes"];
            for (size_t i = 0; i < jnodes.size() && i < cluster->nodes.size(); i++)
            {
                const json &jnode = jnodes[i];
                cluster->nodes[i].num_queries = jnode.value("num_queries", (uint64_t)0);
                cluster->nodes[i].is_remote_node = jnode.value("is_remote_node", false);
                cluster->nodes[i].ip_addr = jnode.value("ip_addr", "127.0.0.1");
                cluster->nodes[i].port = jnode.value("port", 9500 + (int)i);

                if (jnode.contains("local_hnsw_path") && !jnode["local_hnsw_path"].get<string>().empty())
                {
                    string local_hnsw_path = jnode["local_hnsw_path"].get<string>();
                    cluster->nodes[i].local_hnsw = HNSWGraph<T>::load(local_hnsw_path);
                }
            }
        }

        return cluster;
    }
};

// ==================== Shard 模式查询会话 ====================
// 持久连接 + 流水线发送，避免每次查询都 std::async + acquire/release 连接。
// 用法：每个查询线程创建一个 session，复用到所有查询，结束后 close。
// 查询流程：
//   1. 向所有远程节点并行发送 MSG_QUERY_LOCAL（流水线，不等回复）
//   2. 本地节点直接调用 HNSW query（与网络传输重叠）
//   3. 依次接收所有远程节点的响应
//   4. 合并排序取 top-k
template <typename T>
class ShardQuerySession
{
public:
    using DistT = DistTOf<T>;
    Cluster<T> *cluster_;
    std::vector<int> node_conns_; // node_conns_[i] = 到节点 i 的持久连接，本地节点为 -1
    bool valid_ = false;

    ShardQuerySession() : cluster_(nullptr) {}

    explicit ShardQuerySession(Cluster<T> *cluster) : cluster_(cluster)
    {
        node_conns_.resize(cluster->node_num, -1);
        for (int i = 0; i < cluster->node_num; i++)
        {
            if (cluster->is_distributed && cluster->nodes[i].is_remote_node)
            {
                node_conns_[i] = cluster->transport->acquire_connection(i);
                if (node_conns_[i] < 0)
                {
                    LOG_DEBUG_STREAM("[ShardQuerySession] Failed to acquire connection to node " << i);
                }
            }
        }
        valid_ = true;
    }

    ~ShardQuerySession()
    {
        close();
    }

    // 禁止拷贝
    ShardQuerySession(const ShardQuerySession &) = delete;
    ShardQuerySession &operator=(const ShardQuerySession &) = delete;

    // 支持移动
    ShardQuerySession(ShardQuerySession &&other) noexcept
        : cluster_(other.cluster_), node_conns_(std::move(other.node_conns_)), valid_(other.valid_)
    {
        other.valid_ = false;
    }
    ShardQuerySession &operator=(ShardQuerySession &&other) noexcept
    {
        if (this != &other)
        {
            close();
            cluster_ = other.cluster_;
            node_conns_ = std::move(other.node_conns_);
            valid_ = other.valid_;
            other.valid_ = false;
        }
        return *this;
    }

    void close()
    {
        if (!valid_ || !cluster_)
            return;
        for (int i = 0; i < (int)node_conns_.size(); i++)
        {
            if (node_conns_[i] >= 0)
            {
                cluster_->transport->release_connection(i, node_conns_[i], true);
                node_conns_[i] = -1;
            }
        }
        valid_ = false;
    }

    std::vector<int> query(int query_id, const vector<T> &query_vector, Payload<DistT> &payload, int k)
    {
        // Phase 1: 向所有远程节点发送查询（流水线，不等回复）
        for (int i = 0; i < cluster_->node_num; i++)
        {
            if (node_conns_[i] >= 0)
            {
                Buffer buf;
                buf.write_val((int)MSG_QUERY_LOCAL);
                buf.write_val(query_id);
                buf.write_val(k);
                buf.write_vec(query_vector);
                Payload<DistT> empty_payload;
                serialize_payload(buf, empty_payload, (const HNSWGraph<T>*)nullptr);

                uint64_t msg_size = buf.data.size();
                cluster_->transport->send_raw(node_conns_[i], &msg_size, sizeof(uint64_t));
                cluster_->transport->send_raw(node_conns_[i], buf.data.data(), msg_size);
            }
        }

        // Phase 2: 本地节点直接查询（与网络传输重叠）
        std::vector<pair<DistT, int>> all_results;
        for (int i = 0; i < cluster_->node_num; i++)
        {
            if (node_conns_[i] < 0)
            {
                // 本地节点
                Node<T> &node = cluster_->nodes[i];
                if (node.local_hnsw)
                {
                    Payload<DistT> local_payload;
                    node.local_hnsw->query(query_vector, local_payload, k);
                    // 堆内存 iid 编码，pop 时解码成 vid
                    auto pq = local_payload.results;
                    while (!pq.empty())
                    {
                        auto item = pq.top();
                        all_results.push_back({item.first, node.local_hnsw->decode_key(item.second)});
                        pq.pop();
                    }
                    payload.distance_cnt += local_payload.distance_cnt;
                }
            }
        }

        // Phase 3: 依次接收远程节点响应
        for (int i = 0; i < cluster_->node_num; i++)
        {
            if (node_conns_[i] >= 0)
            {
                try
                {
                    uint64_t resp_size = 0;
                    cluster_->transport->recv_raw(node_conns_[i], &resp_size, sizeof(uint64_t));

                    std::vector<char> resp_data(resp_size);
                    cluster_->transport->recv_raw(node_conns_[i], resp_data.data(), resp_size);

                    Buffer resp_buf;
                    resp_buf.data = std::move(resp_data);

                    int result_cnt = resp_buf.read_val<int>();
                    for (int j = 0; j < result_cnt; j++)
                    {
                        DistT dist = resp_buf.read_val<DistT>();
                        int id = resp_buf.read_val<int>();
                        all_results.push_back({dist, id});
                    }

                    Payload<DistT> node_payload;
                    // 堆内容不使用，传 nullptr
                    deserialize_payload(resp_buf, node_payload, (const HNSWGraph<T>*)nullptr);
                    payload.distance_cnt += node_payload.distance_cnt;
                    payload.remote_transfer_cnt += node_payload.remote_transfer_cnt;
                    payload.remote_transfer_data += node_payload.remote_transfer_data;
                    payload.backtrack_cnt += node_payload.backtrack_cnt;
                }
                catch (const std::exception &e)
                {
                    LOG_DEBUG_STREAM("[ShardQuerySession] Error receiving from node " << i << ": " << e.what());
                }
            }
        }

        // Phase 4: 合并排序取 top-k
        std::sort(all_results.begin(), all_results.end());
        std::vector<int> final_results;
        size_t limit = std::min((size_t)k, all_results.size());
        for (size_t i = 0; i < limit; i++)
        {
            final_results.push_back(all_results[i].second);
        }

        return final_results;
    }
};

// 以下为 Node 成员方法的实现
template <typename T>
Node<T>::~Node()
{
    stop_server.store(true);
    // 先等待线程池中所有任务完成（它们可能需要通过 transport 发送响应）
    msg_pool_.reset();
    if (sockfd != -1)
        close(sockfd);
    if (sdk_sockfd != -1)
        close(sdk_sockfd);
    if (server_thread.joinable())
        server_thread.join();
    if (sdk_server_thread.joinable())
        sdk_server_thread.join();
}

template <typename T>
void Node<T>::close_connection_soft(int fd)
{
    if (fd < 0)
        return;
    shutdown(fd, SHUT_RDWR);
    close(fd);
}

// 彻底关闭连接，避免 TIME-WAIT 状态
template <typename T>
void Node<T>::close_connection_hard(int fd)
{
    if (fd < 0) return;
    
    // 通过通信中间层关闭连接（硬关闭，发送 RST）
    cluster->transport->close_connection(fd);
}

template <typename T>
void Node<T>::start_server()
{
    cluster->is_distributed = true;
    if (is_remote_node)
        return;

    // 初始化消息处理线程池
    msg_pool_ = std::make_unique<MessageThreadPool>(msg_pool_size_);
    LOG_INFO_STREAM("Node " << node_id << " message thread pool initialized with " << msg_pool_size_ << " threads");

    // 1. 启动节点间通信端口（transport），端口 = port + NODE_PORT_OFFSET
    int node_port = port + NODE_PORT_OFFSET;
    sockfd = cluster->transport->start_server(node_port);
    if (sockfd < 0)
    {
        LOG_DEBUG_STREAM("Failed to start node server on port " << node_port);
        return;
    }
    LOG_DEBUG_STREAM("Node " << node_id << " node-to-node server started on port " << node_port);

    try
    {
        server_thread = std::thread(&Node<T>::tcp_server_loop, this);
    }
    catch (const std::exception &e)
    {
        LOG_DEBUG_STREAM("Failed to start node server thread: " << e.what());
        cluster->transport->stop_server();
        sockfd = -1;
        return;
    }

    // 2. 启动 SDK 通信端口（纯 TCP），端口 = port
    sdk_sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sdk_sockfd < 0)
    {
        LOG_DEBUG_STREAM("Failed to create SDK server socket");
        return;
    }
    int opt = 1;
    setsockopt(sdk_sockfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);

    if (bind(sdk_sockfd, (struct sockaddr*)&addr, sizeof(addr)) < 0)
    {
        LOG_DEBUG_STREAM("Failed to bind SDK server on port " << port);
        close(sdk_sockfd);
        sdk_sockfd = -1;
        return;
    }
    if (listen(sdk_sockfd, 1024) < 0)
    {
        LOG_DEBUG_STREAM("Failed to listen SDK server on port " << port);
        close(sdk_sockfd);
        sdk_sockfd = -1;
        return;
    }
    LOG_DEBUG_STREAM("Node " << node_id << " SDK server started on port " << port);

    try
    {
        sdk_server_thread = std::thread(&Node<T>::sdk_server_loop, this);
    }
    catch (const std::exception &e)
    {
        LOG_DEBUG_STREAM("Failed to start SDK server thread: " << e.what());
        close(sdk_sockfd);
        sdk_sockfd = -1;
    }
}

// 节点间通信 accept 循环（transport 端口）
//
// 设计：单 poll 线程 + 非阻塞 socket + per-fd 状态机
//   - listen socket 阻塞，仅在 POLLIN 时 accept
//   - 已连接 socket 设 O_NONBLOCK
//   - 每个 fd 维护 ConnState（READING_HEADER / READING_BODY，部分接收的进度）
//   - 收到完整一条消息后立刻 dispatch，状态重置继续读下一条（drain 直到 EAGAIN）
//   - 重消息处理（query / 反序列化）一律走 msg_pool_，poll 线程不做 CPU 重活
template <typename T>
void Node<T>::tcp_server_loop()
{
    int node_port = port + NODE_PORT_OFFSET;
    LOG_INFO_STREAM("Node " << node_id << " poll server loop on port " << node_port);

    struct ConnState {
        int fd = -1;
        enum Phase { READING_HEADER, READING_BODY };
        Phase phase = READING_HEADER;
        char header_buf[8] = {0};
        size_t header_recv = 0;
        uint64_t msg_size = 0;
        std::shared_ptr<Buffer> body;
        size_t body_recv = 0;
    };

    std::vector<struct pollfd> pfds;
    std::vector<ConnState> conns; // conns[i] 对应 pfds[i+1]
    pfds.push_back({sockfd, POLLIN, 0});

    auto drop_at = [&](size_t i) {
        // i 是 pfds 索引（>=1），删掉对应 fd 与 ConnState
        ::close(pfds[i].fd);
        pfds[i] = pfds.back(); pfds.pop_back();
        conns[i - 1] = conns.back(); conns.pop_back();
    };

    auto try_send_ack = [](int fd, char ack) {
        // 1 字节 ack。fd 非阻塞，正常情况下立刻成功；EAGAIN 时短暂 poll 再试。
        for (int retries = 0; retries < 5; retries++) {
            ssize_t s = ::send(fd, &ack, 1, MSG_NOSIGNAL);
            if (s == 1) return true;
            if (s < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
                struct pollfd pp{fd, POLLOUT, 0};
                ::poll(&pp, 1, 100);
                continue;
            }
            return false;
        }
        return false;
    };

    auto dispatch = [&](int fd, std::shared_ptr<Buffer> buf) -> bool {
        // 返回 true = 保持连接；false = 关闭（仅 MSG_INIT_GRAPH 一次性连接）
        buf->read_pos = 0;
        int msg_type = buf->read_val<int>();

        if (msg_type == MSG_INIT_GRAPH) {
            bool remote_global = buf->read_val<bool>();
            cluster->is_global_pull = remote_global;
            deserialize_routing(*buf, cluster->vector_node_map);
            cluster->normalize_routes_primary_first();
            cluster->filter_routes_for_runtime();
            if (local_hnsw) delete local_hnsw;
            local_hnsw = deserialize_hnsw_subset<T>(*buf, /*do_compact=*/!remote_global);
            cluster->log_route_memory_breakdown("[MSG_INIT_GRAPH node " + std::to_string(node_id) + "]");
            if (local_hnsw)
                local_hnsw->log_memory_breakdown("[MSG_INIT_GRAPH node " + std::to_string(node_id) + "]");
            if (remote_global && local_hnsw) {
                Cluster<T>* cl = cluster;
                local_hnsw->pull_fetcher_ = [cl](int vid){ return cl->fetch_remote_vector(vid); };
            }
            try_send_ack(fd, 1);
            return false; // 关闭连接
        }

        // MSG_SEARCH_RESULT / MSG_SHARD_RESULT 走 msg_pool_ 但用 submit_priority 插队到前面。
        // 原因：这两条消息处理极轻（反序列化 + map 查找 + push），
        // 如果跟 MSG_SEARCH（含 transfer_to_remote 的网络 send_raw 阻塞）一起 FIFO 排队，
        // 慢的 SEARCH 任务会把响应消息堵在队尾几十秒，最终触发 SDK 超时。
        if (msg_type == MSG_SEARCH_RESULT) {
            msg_pool_->submit_priority([this, buf]() {
                int qid = -1;
                try {
                    qid = buf->read_val<int>();
                    std::vector<int> rs = buf->read_vec<int>();
                    Payload<DistT> pl; deserialize_payload(*buf, pl, local_hnsw);
                    if (pl.query_start_ms != 0.0)
                        pl.total_ms = steady_now_ms() - pl.query_start_ms;
                    std::shared_ptr<SDKCompletionQueue<T>> cq;
                    { std::lock_guard<std::mutex> lk(sdk_cq_map_mu_);
                      auto it = sdk_cq_map_.find(qid);
                      if (it != sdk_cq_map_.end()) { cq = it->second; sdk_cq_map_.erase(it); } }
                    if (cq) cq->push({qid, std::move(rs), std::move(pl)});
                } catch (const std::exception &e) {
                    LOG_INFO_STREAM("[Node " << node_id << "] MSG_SEARCH_RESULT handler threw for query " << qid
                                    << ": " << e.what() << "; pushing empty result");
                    if (qid >= 0) {
                        std::shared_ptr<SDKCompletionQueue<T>> cq;
                        { std::lock_guard<std::mutex> lk(sdk_cq_map_mu_);
                          auto it = sdk_cq_map_.find(qid);
                          if (it != sdk_cq_map_.end()) { cq = it->second; sdk_cq_map_.erase(it); } }
                        if (cq) {
                            Payload<DistT> empty;
                            try { cq->push({qid, std::vector<int>{}, std::move(empty)}); } catch (...) {}
                        }
                    }
                } catch (...) {
                    LOG_INFO_STREAM("[Node " << node_id << "] MSG_SEARCH_RESULT handler unknown exception for query " << qid);
                }
            });
            return true;
        }

        if (msg_type == MSG_SHARD_RESULT) {
            msg_pool_->submit_priority([this, buf]() {
                try {
                    int qid = buf->read_val<int>();
                    int cnt = buf->read_val<int>();
                    std::vector<std::pair<DistT,int>> rs; rs.reserve(cnt);
                    for (int j=0;j<cnt;j++) rs.push_back({buf->read_val<DistT>(), buf->read_val<int>()});
                    Payload<DistT> pl; deserialize_payload(*buf, pl, local_hnsw);
                    std::shared_ptr<ReduceQueryAccumulator<T>> acc;
                    { std::lock_guard<std::mutex> lk(reduce_accum_mu_);
                      auto it = reduce_accum_map_.find(qid);
                      if (it != reduce_accum_map_.end()) acc = it->second; }
                    if (acc) acc->add_node_results(rs, pl);
                } catch (const std::exception &e) {
                    LOG_INFO_STREAM("[Node " << node_id << "] MSG_SHARD_RESULT handler threw: " << e.what());
                } catch (...) {}
            });
            return true;
        }

        if (msg_type == MSG_SEARCH) {
            double enqueue_ms = steady_now_ms();
            msg_pool_->submit([this, buf, enqueue_ms]() {
                int qid = -1;
                int origin = -1;
                try {
                    qid = buf->read_val<int>();
                    int k = buf->read_val<int>();
                    std::vector<T> qv = buf->read_vec<T>();
                    Payload<DistT> pl; deserialize_payload(*buf, pl, local_hnsw);
                    pl.queue_ms += steady_now_ms() - enqueue_ms;
                    if (pl.query_start_ms == 0.0)
                        pl.query_start_ms = steady_now_ms();
                    origin = pl.origin_node_id;
                    this->query(qid, qv, pl, k);
                } catch (const std::exception &e) {
                    // 任何异常（反序列化、local_hnsw->query、转发）都要把空结果送回 origin，
                    // 否则 origin 节点的 sdk_cq_map_[qid] 永远 orphan
                    LOG_INFO_STREAM("[Node " << node_id << "] MSG_SEARCH handler threw for query " << qid
                                    << " (origin=" << origin << "): " << e.what()
                                    << "; delivering empty result");
                    if (qid >= 0 && origin >= 0)
                        cluster->deliver_empty_result(qid, origin);
                } catch (...) {
                    LOG_INFO_STREAM("[Node " << node_id << "] MSG_SEARCH handler unknown exception for query " << qid
                                    << " (origin=" << origin << "); delivering empty result");
                    if (qid >= 0 && origin >= 0)
                        cluster->deliver_empty_result(qid, origin);
                }
            });
            return true;
        }

        if (msg_type == MSG_ADD_VECTOR_LOCAL) {
            int vid = buf->read_val<int>(); std::vector<T> vec = buf->read_vec<T>();
            char ack = 1;
            try {
                if (!local_hnsw && cluster && cluster->is_shard) {
                    std::lock_guard<std::mutex> lk(local_hnsw_mutex);
                    if (!local_hnsw) local_hnsw = new HNSWGraph<T>(cluster->hnsw->dim, cluster->hnsw->m,
                        cluster->hnsw->ef_construction, cluster->hnsw->ef_search,
                        cluster->hnsw->limited_level, 100000, cluster->hnsw->seed);
                }
                if (local_hnsw) local_hnsw->add_vector(vid, vec);
            } catch (...) { ack = 0; }
            try_send_ack(fd, ack);
            return true;
        }

        // 未知类型：保持连接，丢弃此消息
        return true;
    };

    while (!stop_server.load()) {
        int ret = ::poll(pfds.data(), pfds.size(), 100);
        if (ret < 0) { if (stop_server.load()) break; continue; }

        // 注意：循环中可能修改 pfds（drop_at），故用索引并谨慎处理删除
        for (size_t i = 0; i < pfds.size(); ) {
            short re = pfds[i].revents;
            if (!(re & (POLLIN | POLLERR | POLLHUP))) { i++; continue; }

            // listen socket
            if (pfds[i].fd == sockfd) {
                if (re & POLLIN) {
                    struct sockaddr_in addr; socklen_t len = sizeof(addr);
                    int cfd = ::accept(sockfd, (struct sockaddr*)&addr, &len);
                    if (cfd >= 0) {
                        int fl = ::fcntl(cfd, F_GETFL, 0);
                        if (fl >= 0) ::fcntl(cfd, F_SETFL, fl | O_NONBLOCK);
                        int one = 1;
                        ::setsockopt(cfd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
                        pfds.push_back({cfd, POLLIN, 0});
                        conns.push_back(ConnState{cfd});
                    }
                }
                i++; continue;
            }

            // 数据 socket：错误 / 挂起 → 关闭
            if (re & (POLLERR | POLLHUP | POLLNVAL)) { drop_at(i); continue; }

            // POLLIN：drain 直到 EAGAIN 或读到完整消息后继续读下一条
            ConnState &cs = conns[i - 1];
            bool dropped = false;
            bool exit_loop = false;
            while (!exit_loop) {
                if (cs.phase == ConnState::READING_HEADER) {
                    ssize_t r = ::recv(cs.fd, cs.header_buf + cs.header_recv,
                                       sizeof(cs.header_buf) - cs.header_recv, 0);
                    if (r > 0) {
                        cs.header_recv += static_cast<size_t>(r);
                        if (cs.header_recv == sizeof(cs.header_buf)) {
                            std::memcpy(&cs.msg_size, cs.header_buf, sizeof(uint64_t));
                            if (cs.msg_size == 0) { dropped = true; break; }
                            cs.body = std::make_shared<Buffer>();
                            cs.body->data.resize(cs.msg_size);
                            cs.body_recv = 0;
                            cs.phase = ConnState::READING_BODY;
                        }
                    } else if (r == 0) { dropped = true; break; }
                    else {
                        if (errno == EAGAIN || errno == EWOULDBLOCK) { exit_loop = true; break; }
                        if (errno == EINTR) continue;
                        dropped = true; break;
                    }
                } else { // READING_BODY
                    ssize_t r = ::recv(cs.fd,
                                       cs.body->data.data() + cs.body_recv,
                                       cs.msg_size - cs.body_recv, 0);
                    if (r > 0) {
                        cs.body_recv += static_cast<size_t>(r);
                        if (cs.body_recv == cs.msg_size) {
                            auto body = std::move(cs.body);
                            cs.phase = ConnState::READING_HEADER;
                            cs.header_recv = 0;
                            cs.msg_size = 0;
                            cs.body_recv = 0;
                            int fd_for_dispatch = cs.fd;
                            bool keep = dispatch(fd_for_dispatch, body);
                            if (!keep) { dropped = true; break; }
                            // 继续 while：可能下一条 header 已经在内核缓冲中
                        }
                    } else if (r == 0) { dropped = true; break; }
                    else {
                        if (errno == EAGAIN || errno == EWOULDBLOCK) { exit_loop = true; break; }
                        if (errno == EINTR) continue;
                        dropped = true; break;
                    }
                }
            }

            if (dropped) { drop_at(i); continue; }
            i++;
        }
    }

    // 退出清理
    for (size_t i = 1; i < pfds.size(); i++) ::close(pfds[i].fd);
}

// SDK 通信 accept 循环（纯 TCP 端口，不走传输层抽象）
template <typename T>
void Node<T>::sdk_server_loop()
{
    LOG_DEBUG_STREAM("Node " << node_id << " SDK server loop started on port " << port);

    while (!stop_server.load())
    {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(sdk_sockfd, (struct sockaddr*)&client_addr, &client_len);
        if (client_fd < 0)
        {
            if (stop_server.load())
                break;
            continue;
        }

        int flag = 1;
        setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));

        std::thread(&Node<T>::handle_sdk_client, this, client_fd).detach();
    }
}


// ==================== 处理 SDK 消息（裸 TCP 收发） ====================
// 异步流水线模型：Reader 线程读取查询并处理，Writer 线程发送结果。
// Forward 模式下，Reader 线程处理完本地 HNSW 搜索后立即转发，不等待远程结果，
// 继续读取下一条查询。远程结果返回后由 MSG_SEARCH_RESULT 处理器推入完成队列，
// Writer 线程取出并发送回 SDK 客户端。
template <typename T>
void Node<T>::handle_sdk_client(int client_fd)
{
    active_handlers++;

    // 创建本连接的完成队列
    auto completion_queue = std::make_shared<SDKCompletionQueue<T>>();

    // 记录有多少查询尚未完成（已发送但未收到结果）
    std::atomic<int> inflight_count{0};

    // Writer 线程：从完成队列取出结果，发送回 SDK 客户端
    // 写操作由 Writer 线程独占 client_fd 的写端，Reader 线程独占读端，
    // TCP 支持不同线程并发读写同一 fd。
    std::thread writer_thread([&, client_fd, completion_queue]() {
        while (true)
        {
            typename SDKCompletionQueue<T>::Result result;
            if (!completion_queue->pop(result))
                break;  // 队列已关闭且为空

            try
            {
                double result_start_ms = steady_now_ms();
                Buffer resp_buf;
                resp_buf.write_val(result.query_id);
                resp_buf.write_val((int)result.results.size());
                for (const auto &id : result.results)
                {
                    resp_buf.write_val((DistT)0.0);
                    resp_buf.write_val(id);
                }
                result.payload.result_ms += steady_now_ms() - result_start_ms;
                serialize_payload(resp_buf, result.payload, local_hnsw);

                uint64_t resp_size = resp_buf.data.size();
                send_all(client_fd, &resp_size, sizeof(uint64_t));
                send_all(client_fd, resp_buf.data.data(), resp_size);
                inflight_count--;
            }
            catch (...)
            {
                LOG_DEBUG_STREAM("[Node " << node_id << "] Writer thread: send failed for query " << result.query_id);
                inflight_count--;
                break;
            }
        }
    });

    // Reader 循环（主线程）
    while (true)
    {
        try
        {
            uint64_t msg_size = 0;
            recv_all(client_fd, &msg_size, sizeof(uint64_t));
            if (msg_size == 0)
                break;

            Buffer buf;
            buf.data.resize(msg_size);
            recv_all(client_fd, buf.data.data(), msg_size);
            buf.read_pos = 0;
            int msg_type = buf.read_val<int>();

            if (msg_type == MSG_ADD_VECTOR)
            {
                int vid = buf.read_val<int>();
                std::vector<T> vec = buf.read_vec<T>();

                char ack = 1;
                try {
                    if (cluster && cluster->is_shard)
                        cluster->add_vector(vid, vec);
                    else if (cluster && cluster->hnsw)
                        cluster->hnsw->add_vector(vid, vec);
                } catch (...) {
                    LOG_DEBUG_STREAM("[Node " << node_id << "] Error adding vector " << vid);
                    ack = 0;
                }

                send_all(client_fd, &ack, 1);
                continue;
            }
            else if (msg_type == MSG_QUERY)
            {
                int query_id = buf.read_val<int>();
                int k = buf.read_val<int>();
                std::vector<T> query_vector = buf.read_vec<T>();

                Payload<DistT> payload;
                deserialize_payload(buf, payload, local_hnsw);

                // 查询执行提交到 msg_pool_，reader 线程只负责接收
                auto qv = std::make_shared<std::vector<T>>(std::move(query_vector));
                auto pl = std::make_shared<Payload<DistT>>(std::move(payload));
                double enqueue_ms = steady_now_ms();

                msg_pool_->submit([this, query_id, k, qv, pl, completion_queue, &inflight_count, enqueue_ms]() {
                    pl->queue_ms += steady_now_ms() - enqueue_ms;
                    if (pl->query_start_ms == 0.0)
                        pl->query_start_ms = steady_now_ms();
                    if (cluster && cluster->is_shard)
                    {
                        // Shard 模式：异步流水线
                        inflight_count++;

                        auto accum = std::make_shared<ReduceQueryAccumulator<T>>();
                        accum->query_id = query_id;
                        accum->k = k;
                        accum->total_nodes = cluster->node_num;
                        accum->query_start_ms = pl->query_start_ms;
                        accum->completion_queue = completion_queue;
                        accum->cleanup_fn = [this, query_id]() {
                            std::lock_guard<std::mutex> lk(reduce_accum_mu_);
                            reduce_accum_map_.erase(query_id);
                        };

                        {
                            std::lock_guard<std::mutex> lk(reduce_accum_mu_);
                            reduce_accum_map_[query_id] = accum;
                        }

                        // Phase 1: 向所有远程节点发送 MSG_QUERY_SHARD
                        for (int i = 0; i < cluster->node_num; i++)
                        {
                            if (cluster->is_distributed && cluster->nodes[i].is_remote_node)
                            {
                                int sock = cluster->transport->acquire_connection(i);
                                if (sock >= 0) {
                                    try {
                                        Buffer sbuf;
                                        sbuf.write_val((int)MSG_QUERY_SHARD);
                                        sbuf.write_val(query_id);
                                        sbuf.write_val(k);
                                        sbuf.write_vec(*qv);
                                        sbuf.write_val(node_id);

                                        uint64_t msg_size = sbuf.data.size();
                                        cluster->transport->send_raw(sock, &msg_size, sizeof(uint64_t));
                                        cluster->transport->send_raw(sock, sbuf.data.data(), msg_size);
                                        cluster->transport->release_connection(i, sock, true);
                                    } catch (...) {
                                        cluster->transport->release_connection(i, sock, false);
                                        Payload<DistT> empty_payload;
                                        accum->add_node_results({}, empty_payload);
                                    }
                                } else {
                                    Payload<DistT> empty_payload;
                                    accum->add_node_results({}, empty_payload);
                                }
                            }
                        }

                        // Phase 2: 本地节点直接查询
                        for (int i = 0; i < cluster->node_num; i++)
                        {
                            if (!cluster->is_distributed || !cluster->nodes[i].is_remote_node)
                            {
                                Node<T> &nd = cluster->nodes[i];
                                if (nd.local_hnsw) {
                                    Payload<DistT> local_payload;
                                    nd.local_hnsw->query(*qv, local_payload, k);
                                    std::vector<std::pair<DistT, int>> local_results;
                                    // 堆内存 iid 编码，pop 时解码成 vid
                                    auto pq = local_payload.results;
                                    while (!pq.empty()) {
                                        auto item = pq.top();
                                        local_results.push_back({item.first, nd.local_hnsw->decode_key(item.second)});
                                        pq.pop();
                                    }
                                    accum->add_node_results(local_results, local_payload);
                                } else {
                                    Payload<DistT> empty_payload;
                                    accum->add_node_results({}, empty_payload);
                                }
                            }
                        }
                    }
                    else
                    {
                        // Forward 模式：先 inflight_count++ 再做工作，保证任何抛错路径都能被 writer 平衡 dec
                        inflight_count++;
                        try
                        {
                            pl->origin_node_id = node_id;
                            std::vector<int> results;
                            if (local_hnsw)
                                results = local_hnsw->query(*qv, *pl, k);

                            if (!cluster->is_distributed || !pl->continue_search)
                            {
                                if (pl->query_start_ms != 0.0)
                                    pl->total_ms = steady_now_ms() - pl->query_start_ms;
                                completion_queue->push({query_id, std::move(results), std::move(*pl)});
                            }
                            else
                            {
                                {
                                    std::lock_guard<std::mutex> lk(sdk_cq_map_mu_);
                                    sdk_cq_map_[query_id] = completion_queue;
                                }
                                cluster->transfer_to_remote(query_id, *qv, *pl, k);
                            }
                        }
                        catch (const std::exception &e)
                        {
                            // 任何异常（local_hnsw->query 抛、transfer_to_remote 抛、push 抛）都不能让 SDK hang
                            // 强制把空结果推到 completion_queue，让 SDK 收到 query_id 对应的回包
                            LOG_INFO_STREAM("[Node " << node_id << "] MSG_QUERY lambda threw for query "
                                            << query_id << ": " << e.what()
                                            << "; pushing empty result to keep SDK unblocked");
                            // 清理可能已 register 的 sdk_cq_map_ 条目（避免后续 MSG_SEARCH_RESULT 看到 stale）
                            {
                                std::lock_guard<std::mutex> lk(sdk_cq_map_mu_);
                                sdk_cq_map_.erase(query_id);
                            }
                            Payload<DistT> empty_payload;
                            try {
                                completion_queue->push({query_id, std::vector<int>{}, std::move(empty_payload)});
                            } catch (...) {}
                        }
                        catch (...)
                        {
                            LOG_INFO_STREAM("[Node " << node_id << "] MSG_QUERY lambda unknown exception for query "
                                            << query_id << "; pushing empty result");
                            {
                                std::lock_guard<std::mutex> lk(sdk_cq_map_mu_);
                                sdk_cq_map_.erase(query_id);
                            }
                            Payload<DistT> empty_payload;
                            try {
                                completion_queue->push({query_id, std::vector<int>{}, std::move(empty_payload)});
                            } catch (...) {}
                        }
                    }
                });
                continue;
            }
            else if (msg_type == MSG_PARTITION)
            {
                std::string mode = buf.read_string();
                int replication = buf.read_val<int>();
                std::unordered_map<int, std::vector<int>> layer_config;
                std::string layer_partition_mode = "round_robin";
                std::vector<bool> layer_replicate_flags;
                // SDK 端始终序列化全部字段，服务端也必须始终反序列化
                deserialize_layer_config(buf, layer_config);
                layer_partition_mode = buf.read_string();
                deserialize_layer_replicate_flags(buf, layer_replicate_flags);
                int num_threads = buf.read_val<int>();
                std::string partition_distribution = buf.read_string();
                std::string partition_base = buf.read_string();

                char ack = 1;
                try
                {
                    if (cluster)
                        cluster->partition(mode, replication, layer_config, layer_partition_mode, layer_replicate_flags,
                                           num_threads, partition_distribution, partition_base);
                }
                catch (const std::exception &e)
                {
                    LOG_INFO_STREAM("[Node " << node_id << "] MSG_PARTITION failed: " << e.what());
                    ack = 0;
                }
                catch (...)
                {
                    LOG_INFO_STREAM("[Node " << node_id << "] MSG_PARTITION failed with unknown exception");
                    ack = 0;
                }
                send_all(client_fd, &ack, 1);
                break;
            }
            else if (msg_type == MSG_LOAD_CLUSTER)
            {
                std::string filepath = buf.read_string();
                LOG_INFO_STREAM("[Node " << node_id << "] MSG_LOAD_CLUSTER: " << filepath);
                auto t0 = std::chrono::steady_clock::now();
                bool ok = cluster ? cluster->load_local(filepath) : false;
                auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - t0).count();
                LOG_INFO_STREAM("[Node " << node_id << "] load_local " << (ok ? "OK" : "FAILED")
                               << " in " << dt << "ms, sending ACK");
                char ack = ok ? 1 : 0;
                send_all(client_fd, &ack, 1);
                LOG_INFO_STREAM("[Node " << node_id << "] MSG_LOAD_CLUSTER ACK sent");
                break;
            }
            else if (msg_type == MSG_LOAD_HNSW)
            {
                std::string filepath = buf.read_string();
                bool ok = false;
                if (cluster)
                {
                    HNSWGraph<T> *loaded_hnsw = nullptr;
                    try { loaded_hnsw = HNSWGraph<T>::load(filepath); }
                    catch (...) { loaded_hnsw = nullptr; }

                    if (loaded_hnsw)
                    {
                        if (cluster->hnsw && cluster->hnsw != loaded_hnsw)
                        {
                            if (cluster->owns_hnsw)
                                delete cluster->hnsw;
                            else
                                cluster->hnsw->release_storage();
                        }
                        cluster->hnsw = loaded_hnsw;
                        cluster->owns_hnsw = true;
                        if (cluster->node_num == 1 && !cluster->nodes.empty() && !cluster->nodes[0].is_remote_node)
                            cluster->nodes[0].local_hnsw = loaded_hnsw;
                        ok = true;
                    }
                }
                char ack = ok ? 1 : 0;
                send_all(client_fd, &ack, 1);
                break;
            }
            else if (msg_type == MSG_SET_HNSW_PARAMS)
            {
                int ef_search = buf.read_val<int>();
                int ef_construction = buf.read_val<int>();
                int limited_level = buf.read_val<int>();

                bool ok = false;
                if (cluster)
                {
                    auto apply_params = [&](HNSWGraph<T> *graph) {
                        if (!graph)
                            return;
                        if (ef_search > 0)
                            graph->ef_search = ef_search;
                        if (ef_construction > 0)
                            graph->ef_construction = ef_construction;
                        // limited_level = -1 表示不限制；-2 表示保持不变（由 SDK 默认值约定）
                        if (limited_level >= -1)
                            graph->limited_level = limited_level;
                    };

                    bool has_graph = false;

                    if (cluster->hnsw)
                    {
                        apply_params(cluster->hnsw);
                        has_graph = true;
                    }

                    for (auto &node : cluster->nodes)
                    {
                        if (!node.local_hnsw)
                            continue;
                        // 避免对同一指针重复设置
                        if (node.local_hnsw == cluster->hnsw)
                            continue;
                        apply_params(node.local_hnsw);
                        has_graph = true;
                    }

                    ok = has_graph;
                }

                char ack = ok ? 1 : 0;
                send_all(client_fd, &ack, 1);
                break;
            }
            else if (msg_type == MSG_SAVE_HNSW)
            {
                std::string filepath = buf.read_string();
                bool ok = false;
                if (cluster && cluster->hnsw)
                {
                    try { cluster->hnsw->save(filepath); ok = true; }
                    catch (...) {}
                }
                char ack = ok ? 1 : 0;
                send_all(client_fd, &ack, 1);
                break;
            }
            else if (msg_type == MSG_RELEASE_FULL_HNSW)
            {
                bool ok = false;
                if (cluster)
                {
                    cluster->release_full_hnsw();
                    ok = true;
                }
                char ack = ok ? 1 : 0;
                send_all(client_fd, &ack, 1);
                break;
            }
            else if (msg_type == MSG_GET_INFORMATION)
            {
                json info;
                if (cluster)
                {
                    json cj;
                    cj["node_num"] = cluster->node_num;
                    cj["bandwidth"] = cluster->bandwidth;
                    cj["partition_mode"] = cluster->partition_mode;
                    cj["layer_partition_mode"] = cluster->layer_partition_mode;
                    cj["replication_level"] = cluster->replication_level;
                    cj["replication_num"] = cluster->replication_num;
                    cj["is_distributed"] = cluster->is_distributed;
                    cj["is_shard"] = cluster->is_shard;
                    cj["is_global_pull"] = cluster->is_global_pull;
                    cj["layer_config"] = cluster->layer_config;
                    cj["layer_replicate_flags"] = cluster->layer_replicate_flags;

                    json nodes_json = json::array();
                    for (int i = 0; i < cluster->node_num && i < (int)cluster->nodes.size(); i++)
                    {
                        json nj;
                        nj["node_id"] = i;
                        nj["ip_addr"] = cluster->nodes[i].ip_addr;
                        nj["port"] = cluster->nodes[i].port;
                        nj["is_remote_node"] = cluster->nodes[i].is_remote_node;
                        if (!cluster->nodes[i].is_remote_node && cluster->nodes[i].local_hnsw)
                            nj["vector_count"] = (int)cluster->nodes[i].local_hnsw->get_cur_element_count();
                        nodes_json.push_back(std::move(nj));
                    }
                    cj["nodes"] = std::move(nodes_json);

                    // 入口点所在节点：找到 vector_node_map 最高层中所有节点
                    json entry_nodes_json = json::array();
                    bool entry_nodes_found = false;
                    if (!cluster->vector_node_map.empty())
                    {
                        // vector_node_map[level] 按层级从 0 开始，找最高非空层
                        for (int lvl = (int)cluster->vector_node_map.size() - 1; lvl >= 0; lvl--)
                        {
                            if (!cluster->vector_node_map[lvl].empty())
                            {
                                std::set<int> entry_node_set;
                                for (const auto &kv : cluster->vector_node_map[lvl])
                                {
                                    for (int nid : kv.second)
                                        entry_node_set.insert(nid);
                                }
                                for (int nid : entry_node_set)
                                    entry_nodes_json.push_back(nid);
                                entry_nodes_found = true;
                                break;
                            }
                        }
                    }
                    if (!entry_nodes_found && cluster->dense_level0_routes_enabled_)
                    {
                        for (int i = 0; i < cluster->node_num && i < (int)cluster->nodes.size(); i++)
                        {
                            HNSWGraph<T> *lh = cluster->nodes[i].local_hnsw;
                            if (!cluster->nodes[i].is_remote_node && lh && lh->entry_point_id >= 0)
                            {
                                int ep_vid = lh->get_external_label(lh->entry_point_id);
                                int owner = -1;
                                if (cluster->lookup_primary_owner(0, ep_vid, owner))
                                    entry_nodes_json.push_back(owner);
                                break;
                            }
                        }
                    }
                    cj["entry_node_ids"] = std::move(entry_nodes_json);

                    info["cluster"] = std::move(cj);

                    HNSWGraph<T> *info_hnsw = cluster->hnsw;
                    if (!info_hnsw)
                    {
                        for (auto &node : cluster->nodes)
                        {
                            if (!node.is_remote_node && node.local_hnsw)
                            {
                                info_hnsw = node.local_hnsw;
                                break;
                            }
                        }
                    }
                    if (info_hnsw)
                    {
                        json hj;
                        hj["dim"] = info_hnsw->dim;
                        hj["m"] = info_hnsw->m;
                        hj["ef_construction"] = info_hnsw->ef_construction;
                        hj["ef_search"] = info_hnsw->ef_search;
                        hj["limited_level"] = info_hnsw->limited_level;
                        hj["max_level"] = info_hnsw->max_level;
                        info["hnsw"] = std::move(hj);
                    }
                    else
                        info["hnsw"] = nullptr;
                }
                else
                {
                    info["cluster"] = nullptr;
                    info["hnsw"] = nullptr;
                }

                std::string out = info.dump();
                Buffer resp_buf;
                resp_buf.write_string(out);
                uint64_t resp_size = resp_buf.data.size();
                send_all(client_fd, &resp_size, sizeof(uint64_t));
                send_all(client_fd, resp_buf.data.data(), resp_size);
                break;
            }
            else if (msg_type == MSG_SAVE_CLUSTER)
            {
                std::string filepath = buf.read_string();
                char ack = 0;
                if (cluster)
                    ack = cluster->save(filepath) ? 1 : 0;
                send_all(client_fd, &ack, 1);
                break;
            }
            else
            {
                LOG_DEBUG_STREAM("[Node " << node_id << "] Unknown SDK message type: " << msg_type);
                break;
            }
        }
        catch (const std::exception &e)
        {
            // 对端正常关闭（peer closed）是良性事件 —— SDK 一次性请求完毕断开都会走这里
            // 仅在真正的错误时打日志
            std::string msg = e.what();
            if (msg != "peer closed")
            {
                LOG_INFO_STREAM("[Node " << node_id << "] Error in handle_sdk_client (fd=" << client_fd << "): " << msg);
            }
            break;
        }
        catch (...)
        {
            break;
        }
    }

    // 等待所有 inflight 查询完成，然后关闭完成队列
    // 注意：对于非 MSG_QUERY break 出来的情况（如 MSG_PARTITION），
    // 这里仍需等待已发出的查询返回结果
    while (inflight_count.load() > 0)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // 关闭完成队列，Writer 线程会退出
    completion_queue->close();
    if (writer_thread.joinable())
        writer_thread.join();

    // SDK 连接是纯 TCP fd，直接关闭即可。
    // 不能使用 close_connection_hard（经过 transport 层），因为 RDMA transport
    // 找不到该 fd 后会以 SO_LINGER=0 强制关闭，RST 可能丢弃未送达的 ACK 数据。
    if (client_fd >= 0) {
        close(client_fd);
    }
    active_handlers--;
}

template <typename T>
void Node<T>::send_result_to_origin(int query_id, const std::vector<int>& results, Payload<DistT>& payload)
{
    if (payload.origin_node_id < 0 || payload.origin_node_id >= cluster->node_num)
    {
        LOG_DEBUG_STREAM("[Node " << node_id << "] ERROR: Invalid origin_node_id=" << payload.origin_node_id);
        return;
    }
    
    Node<T>& origin_node = cluster->nodes[payload.origin_node_id];

    // 如果 origin_node 是本地节点，直接推入完成队列
    if (!origin_node.is_remote_node)
    {
        LOG_DEBUG_STREAM("[Node " << node_id << "] Setting result locally for query " << query_id);
        std::shared_ptr<SDKCompletionQueue<T>> cq;
        {
            std::lock_guard<std::mutex> lk(origin_node.sdk_cq_map_mu_);
            auto it = origin_node.sdk_cq_map_.find(query_id);
            if (it != origin_node.sdk_cq_map_.end()) {
                cq = it->second;
                origin_node.sdk_cq_map_.erase(it);
            }
        }
        if (cq) {
            if (payload.query_start_ms != 0.0)
                payload.total_ms = steady_now_ms() - payload.query_start_ms;
            cq->push({query_id, results, std::move(payload)});
        } else {
            LOG_DEBUG_STREAM("[Node " << node_id << "] Warning: No completion queue for local query " << query_id);
        }
        LOG_DEBUG_STREAM("[Node " << node_id << "] Result set for query " << query_id);
        return;
    }
    
    // origin_node 在远程节点上，通过网络发送查询结果。
    // 重试若干次以容忍连接池/瞬时网络抖动，避免 SDK 端 query_recv 永久阻塞。
    LOG_DEBUG_STREAM("[Node " << node_id << "] Sending result to remote origin node " << payload.origin_node_id);

    double result_start_ms = steady_now_ms();
    // 序列化一次，重试时复用
    Buffer buf;
    buf.write_val((int)MSG_SEARCH_RESULT);
    buf.write_val(query_id);
    buf.write_vec(results);
    payload.result_ms += steady_now_ms() - result_start_ms;
    serialize_payload(buf, payload, local_hnsw);
    uint64_t msg_size = buf.data.size();

    constexpr int MAX_RETRIES = 3;
    constexpr int BASE_DELAY_MS = 50;
    for (int attempt = 0; attempt < MAX_RETRIES; ++attempt)
    {
        int sock = cluster->transport->acquire_connection(payload.origin_node_id);
        if (sock < 0)
        {
            if (attempt + 1 < MAX_RETRIES)
                std::this_thread::sleep_for(std::chrono::milliseconds(BASE_DELAY_MS << attempt));
            continue;
        }

        try
        {
            double send_start_ms = steady_now_ms();
            cluster->transport->send_raw(sock, &msg_size, sizeof(uint64_t));
            cluster->transport->send_raw(sock, buf.data.data(), msg_size);
            payload.send_ms += steady_now_ms() - send_start_ms;
            cluster->transport->release_connection(payload.origin_node_id, sock, true);
            LOG_DEBUG_STREAM("[Node " << node_id << "] Result sent to origin node " << payload.origin_node_id);
            return;
        }
        catch (...)
        {
            cluster->transport->release_connection(payload.origin_node_id, sock, false);
            if (attempt + 1 < MAX_RETRIES)
                std::this_thread::sleep_for(std::chrono::milliseconds(BASE_DELAY_MS << attempt));
        }
    }

    // 所有重试失败：origin 节点的 sdk_cq_map_[query_id] 仍在等，但我们已尽力。
    // 大声日志报警，让运维知道有 query 真的丢了（这种情况下 origin 端 SDK 仍会 hang，
    // 需要 SDK 侧加 query_recv 超时来保底；此 log 是定位线索）。
    LOG_INFO_STREAM("[Node " << node_id << "] CRITICAL: failed to deliver result for query "
                    << query_id << " to remote origin " << payload.origin_node_id
                    << " after " << MAX_RETRIES << " retries");
}

// Node::query() — 用于两种场景：
// 1. 中间/目标节点收到 MSG_SEARCH 后处理（is_distributed=true, origin 已设置）
// 2. Python 直接调用（非分布式或本地模式）
// 注意：SDK 流水线模式不再走此函数，而是由 handle_sdk_client 内联处理。
template <typename T>
std::vector<int> Node<T>::query(int query_id, const vector<T> &query_vector, Payload<DistT> &payload, int k)
{
    // 如果 origin_node_id 未设置，说明是 Python 直接调用
    if (payload.origin_node_id < 0)
    {
        payload.origin_node_id = node_id;
        LOG_DEBUG_STREAM("[Node " << node_id << "] Starting query " << query_id << ", set origin_node_id=" << node_id);
    }
    else
    {
        LOG_DEBUG_STREAM("[Node " << node_id << "] Processing query " << query_id << ", origin_node_id=" << payload.origin_node_id);
    }

    std::vector<int> results;

    if (local_hnsw)
    {
        LOG_DEBUG_STREAM("[Node " << node_id << "] Before HNSW query " << query_id << ", continue_search=" << payload.continue_search);
        results = local_hnsw->query(query_vector, payload, k);
        LOG_DEBUG_STREAM("[Node " << node_id << "] After HNSW query " << query_id << ", continue_search=" << payload.continue_search << ", require_id=" << payload.require_id << ", transfer_cnt=" << payload.remote_transfer_cnt);
    }

    if (cluster->is_distributed)
    {
        if (payload.continue_search)
        {
            LOG_DEBUG_STREAM("[Node " << node_id << "] Intermediate node forwarding query " << query_id);
            // 转发到远程节点，不等待（中间节点行为）
            cluster->transfer_to_remote(query_id, query_vector, payload, k);
            return {};
        }
        else
        {
            num_queries++;
            LOG_DEBUG_STREAM("[Node " << node_id << "] Query " << query_id << " COMPLETED, sending to origin " << payload.origin_node_id);
            send_result_to_origin(query_id, results, payload);
        }
    }
    else
    {
        if (payload.continue_search)
        {
            return cluster->transfer_to_remote(query_id, query_vector, payload, k);
        }
        else
        {
            num_queries++;
            LOG_DEBUG_STREAM("[Node " << node_id << "] Query " << query_id << " completed locally");
            return results;
        }
    }
    LOG_DEBUG_STREAM("[Node " << node_id << "] Returning result for query " << query_id);
    return results;
}

template <typename T>
void Node<T>::add_vector(int vid, const std::vector<T> &vec)
{
    if (is_remote_node)
    {
        // 使用通信中间层获取连接
        int sock = cluster->transport->acquire_connection(node_id);
        if (sock < 0)
        {
            LOG_DEBUG_STREAM("Failed to get connection for add_vector");
            return;
        }

        Buffer buf;
        buf.write_val((int)MSG_ADD_VECTOR_LOCAL);  // 使用 LOCAL 消息类型，避免递归路由
        buf.write_val(vid);
        buf.write_vec(vec);

        uint64_t msg_size = buf.data.size();
        try
        {
            cluster->transport->send_raw(sock, &msg_size, sizeof(uint64_t));
            cluster->transport->send_raw(sock, buf.data.data(), msg_size);

    // 等待远程节点的确认响应
            char ack = 0;
            cluster->transport->recv_raw(sock, &ack, 1);
            // 持久连接：远程节点不会关闭连接，归还到连接池复用
            cluster->transport->release_connection(node_id, sock, true);
        }
        catch (const std::exception &e)
        {
            LOG_DEBUG_STREAM("Error sending add_vector: " << e.what());
            cluster->transport->release_connection(node_id, sock, false);
        }
    }
    else
    {
        // 确保 local_hnsw 已初始化（reduce模式下按需初始化）
        // 双重检查锁定：只保护指针创建，add_vector 内部已有线程安全保护
        if (local_hnsw == nullptr && cluster->is_shard)
        {
            std::lock_guard<std::mutex> lk(local_hnsw_mutex);
            if (local_hnsw == nullptr)
            {
                local_hnsw = new HNSWGraph<T>(
                    cluster->hnsw->dim, cluster->hnsw->m, cluster->hnsw->ef_construction,
                    cluster->hnsw->ef_search, cluster->hnsw->limited_level, 100000, cluster->hnsw->seed);
            }
        }

        if (local_hnsw)
        {
            local_hnsw->add_vector(vid, vec);
        }
        else
        {
            LOG_DEBUG_STREAM("Local HNSW is not initialized for node " << node_id);
        }
    }
}
