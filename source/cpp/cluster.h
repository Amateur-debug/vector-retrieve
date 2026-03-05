#pragma once

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <exception>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <condition_variable>
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

#include <cstring>
#include <thread>
#include <atomic>
#include <future>
#include <poll.h>

#include "distance.h"
#include "hnsw.h"
#include "log.h"
#include "payload.h"
#include "tcp_transport.h"

#include "json.hpp"
using json = nlohmann::json;

using namespace std;

// 注意: TcpConnectionPool 已迁移到 tcp_transport.h
// g_connection_pool 已被 Cluster::transport 成员替代

template <typename T>
class Cluster; // Forward declaration

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
    MSG_QUERY_LOCAL = 11,  // Reduce模式下只执行本地查询（避免递归）
    MSG_ADD_VECTOR_LOCAL = 12  // Reduce模式下直接添加到本地（避免递归路由）
};

// Buffer helper for serialization
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

// --- Serialization Helpers ---

template <typename T>
void serialize_payload(Buffer &buf, Payload<T> &payload)
{
    buf.write_val(payload.continue_search);
    buf.write_val(payload.origin_node_id);
    buf.write_val(payload.require_id);
    buf.write_val(payload.start_level);
    buf.write_val(payload.entry_point_id);
    buf.write_val(payload.remote_transfer_cnt);
    buf.write_val(payload.remote_transfer_latency);
    buf.write_val(payload.send_timestamp);
    buf.write_val(payload.distance_cnt);

    buf.write_vec(payload.visited_vids);

    auto cand_copy = payload.candidates;
    std::vector<pair<T, int>> cand_vec;
    while (!cand_copy.empty())
    {
        cand_vec.push_back(cand_copy.top());
        cand_copy.pop();
    }
    buf.write_val((int)cand_vec.size());
    for (auto &p : cand_vec)
    {
        buf.write_val(p.first);
        buf.write_val(p.second);
    }

    auto res_copy = payload.results;
    std::vector<pair<T, int>> res_vec;
    while (!res_copy.empty())
    {
        res_vec.push_back(res_copy.top());
        res_copy.pop();
    }
    buf.write_val((int)res_vec.size());
    for (auto &p : res_vec)
    {
        buf.write_val(p.first);
        buf.write_val(p.second);
    }
}

template <typename T>
void deserialize_payload(Buffer &buf, Payload<T> &payload)
{
    payload.continue_search = buf.read_val<bool>();
    payload.origin_node_id = buf.read_val<int>();
    payload.require_id = buf.read_val<int>();
    payload.start_level = buf.read_val<int>();
    payload.entry_point_id = buf.read_val<int>();
    payload.remote_transfer_cnt = buf.read_val<int>();
    payload.remote_transfer_latency = buf.read_val<double>();
    payload.send_timestamp = buf.read_val<double>();
    payload.distance_cnt = buf.read_val<int>();

    payload.visited_vids = buf.read_vec<int>();

    int cand_size = buf.read_val<int>();
    payload.candidates = decltype(payload.candidates)();
    for (int i = 0; i < cand_size; ++i)
    {
        T dist = buf.read_val<T>();
        int id = buf.read_val<int>();
        payload.candidates.push({dist, id});
    }

    int res_size = buf.read_val<int>();
    payload.results = decltype(payload.results)();
    for (int i = 0; i < res_size; ++i)
    {
        T dist = buf.read_val<T>();
        int id = buf.read_val<int>();
        payload.results.push({dist, id});
    }
}

template <typename T>
void serialize_hnsw(Buffer &buf, const HNSWGraph<T> *hnsw)
{
    size_t cur_count = hnsw->cur_element_count_.load();

    // 写入参数头
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

    // 写入 level 0 数据块
    buf.write(hnsw->data_level0_, cur_count * hnsw->size_per_element_);

    // 写入高层链接数据
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

    // 读取 level 0 数据块
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

void serialize_routing(Buffer &buf, const std::vector<std::unordered_map<int, std::vector<int>>> &maps)
{
    buf.write_val((uint64_t)maps.size());
    for (const auto &map : maps)
    {
        buf.write_val((uint64_t)map.size());
        for (const auto &kv : map)
        {
            buf.write_val(kv.first);
            buf.write_vec(kv.second);
        }
    }
}

void deserialize_routing(Buffer &buf, std::vector<std::unordered_map<int, std::vector<int>>> &maps)
{
    uint64_t levels = buf.read_val<uint64_t>();
    maps.clear();
    maps.resize(levels);
    for (uint64_t l = 0; l < levels; ++l)
    {
        uint64_t size = buf.read_val<uint64_t>();
        for (uint64_t i = 0; i < size; ++i)
        {
            int k = buf.read_val<int>();
            std::vector<int> v = buf.read_vec<int>();
            maps[l][k] = std::move(v);
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

// Helper for sending all data
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

// Helper for receiving all data
void recv_all(int sock, void *data, size_t size)
{
    char *ptr = static_cast<char *>(data);
    size_t total_read = 0;
    while (total_read < size)
    {
        ssize_t r = recv(sock, ptr + total_read, size - total_read, 0);
        if (r <= 0)
        {
            throw std::runtime_error("recv failed");
        }
        total_read += r;
    }
}

// --- End Serialization Helpers ---

template <typename T>
class Node
{
public:
    int node_id;
    Cluster<T> *cluster;
    HNSWGraph<T> *local_hnsw;
    uint64_t num_queries = 0;

    bool is_remote_node = false;

    int sockfd = -1;
    struct sockaddr_in sock_addr;
    std::string ip_addr = "127.0.0.1";
    int port = 9000;

    std::thread server_thread;
    std::atomic<bool> stop_server{false};

    std::atomic<int> active_handlers{0};
    
    // 保护 local_hnsw 初始化的互斥锁
    std::mutex local_hnsw_mutex;

    // 异步查询结果等待
    struct PendingQuery {
        std::vector<int> results;
        Payload<T> payload;
        bool completed = false;
    };
    std::mutex pending_queries_mu;
    std::condition_variable pending_queries_cv;
    std::unordered_map<int, PendingQuery> pending_queries;  // query_id -> PendingQuery

    // 异步查询辅助方法
    void send_result_to_origin(int query_id, const std::vector<int>& results, Payload<T>& payload);
    void wait_for_async_result(int query_id, std::vector<int>& results, Payload<T>& payload);

    Node(int node_id, Cluster<T> *cluster, HNSWGraph<T> *local_hnsw = nullptr, bool is_remote_node = false)
        : node_id(node_id), cluster(cluster), local_hnsw(local_hnsw), is_remote_node(is_remote_node)
    {
        port = 9500 + node_id;
    }

    // 禁止拷贝
    Node(const Node &) = delete;
    Node &operator=(const Node &) = delete;

    // 移动构造函数
    Node(Node &&other) noexcept
        : node_id(other.node_id), cluster(other.cluster), local_hnsw(other.local_hnsw),
          num_queries(other.num_queries), is_remote_node(other.is_remote_node),
          sockfd(other.sockfd), sock_addr(other.sock_addr), ip_addr(std::move(other.ip_addr)),
          port(other.port), server_thread(std::move(other.server_thread)), stop_server(other.stop_server.load())
    {
        other.sockfd = -1;
        other.local_hnsw = nullptr;
    }

    ~Node();

    void start_server();

    void reset()
    {
        num_queries = 0;
    }

    void tcp_server_loop();

    void close_connection_hard(int fd);
    void close_connection_soft(int fd);

    void handle_client(int client_fd);

    std::vector<int> query(int query_id, const vector<T> &query_vector, Payload<T> &payload, int k = 10);

    void add_vector(int vid, const std::vector<T> &vec);
};

template <typename T>
class Cluster
{
public:
    HNSWGraph<T> *hnsw;

    int node_num = 4;

    // 节点集合
    std::vector<Node<T>> nodes;

    // 向量 -> 路由节点集合（按层）
    std::vector<std::unordered_map<int, std::vector<int>>> vector_node_map;

    // 模拟网络带宽 Mb/s
    double bandwidth = 1000.0;

    // 分区模式
    std::string partition_mode;
    std::unordered_map<int, std::vector<int>> layer_config;
    std::vector<bool> layer_replicate_flags;

    int outer_ring_max_iterations = 0;

    int redundancy_level = 0;
    uint64_t redundancy_num = 0;

    bool is_distributed = false;
    bool is_reduce = false;

    std::atomic<int> add_round_robin_counter{0};

    // 通信中间层配置和实例
    TransportConfig transport_config;
    std::shared_ptr<ITransport> transport;

    Cluster(HNSWGraph<T> *hnsw, int node_num = 4, const TransportConfig& config = TransportConfig())
        : hnsw(hnsw), node_num(node_num), transport_config(config)
    {
        transport = TransportFactory::create(transport_config);
        nodes.reserve(node_num);
        for (int i = 0; i < node_num; i++)
        {
            nodes.emplace_back(Node<T>(i, this, nullptr));
        }
    }

    ~Cluster()
    {
        // 停止所有服务器并清理资源
        for (int i = 0; i < node_num; i++)
        {
            nodes[i].stop_server.store(true);
        }
        // 通过通信中间层停止服务端
        if (transport)
        {
            transport->stop_server();
        }
        // 重置节点的 sockfd
        for (int i = 0; i < node_num; i++)
        {
            nodes[i].sockfd = -1;
        }
        // 等待所有服务器线程退出
        for (int i = 0; i < node_num; i++)
        {
            if (nodes[i].server_thread.joinable())
            {
                nodes[i].server_thread.join();
            }
        }
        // 清理 local_hnsw（由 partition 创建）
        for (int i = 0; i < node_num; i++)
        {
            if (nodes[i].local_hnsw != nullptr && nodes[i].local_hnsw != hnsw)
            {
                delete nodes[i].local_hnsw;
                nodes[i].local_hnsw = nullptr;
            }
        }
    }

    void add_vector(int vid, const std::vector<T> &vec)
    {
        if (is_reduce)
        {
            // 轮询分配到各个节点
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

    // 新增：配置节点网络信息
    void set_node_info(int node_id, const std::string &ip, int port, bool is_remote)
    {
        if (node_id >= 0 && node_id < node_num)
        {
            nodes[node_id].ip_addr = ip;
            nodes[node_id].port = port;
            nodes[node_id].is_remote_node = is_remote;
            // 注册到通信中间层
            transport->register_node(node_id, ip, port);
        }
    }

    // 新增：启动指定节点的服务
    void start_node_server(int node_id)
    {
        if (node_id >= 0 && node_id < node_num)
        {
            nodes[node_id].start_server();
        }
    }

    // 新增：阻塞等待指定节点的服务（用于 Worker 节点保持运行）
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

    // 简单 KMeans 实现（欧氏距离）
    // 返回值：与 data 迭代顺序一致的 label 数组（labels[i] 对应 data 第 i 个元素）
    static std::vector<int> kmeans(const std::unordered_map<int, vector<T>> &data,
                                   int k, int max_iters = 100, int seed = 123)
    {
        int n = (int)data.size();
        std::vector<int> labels(n, 0);
        if (n == 0 || k <= 0)
            return labels;

        // 将 unordered_map 转为连续数组，避免用整数 key 当下标访问
        std::vector<const std::vector<T>*> ptrs;
        ptrs.reserve(n);
        for (const auto &kv : data)
            ptrs.push_back(&kv.second);

        int dim = (int)ptrs[0]->size();
        std::mt19937 rng(seed);

        // 初始化：随机选 k 个质心
        std::vector<vector<T>> centers;
        centers.reserve(k);
        std::unordered_set<int> chosen;
        std::uniform_int_distribution<int> uid(0, n - 1);
        while ((int)centers.size() < k)
        {
            int idx = uid(rng);
            if (chosen.insert(idx).second)
                centers.push_back(*ptrs[idx]);
        }

        for (int iter = 0; iter < max_iters; iter++)
        {
            bool changed = false;

            for (int i = 0; i < n; i++)
            {
                T best_dist = numeric_limits<T>::max();
                int best_center = 0;
                for (int j = 0; j < (int)centers.size(); j++)
                {
                    T dist = L2Sqr(ptrs[i]->data(), centers[j].data(), dim);

                    if (dist < best_dist)
                    {
                        best_dist = dist;
                        best_center = j;
                    }
                }
                if (labels[i] != best_center)
                {
                    labels[i] = best_center;
                    changed = true;
                }
            }
            if (!changed)
                break;

            // recompute centers
            std::vector<std::vector<T>> sums(k, std::vector<T>(dim, 0.0f));
            std::vector<int> counts(k, 0);
            for (int i = 0; i < n; i++)
            {
                int current_id = labels[i];
                counts[current_id]++;
                for (int j = 0; j < dim; j++)
                {
                    sums[current_id][j] += (*ptrs[i])[j];
                }
            }
            for (int j = 0; j < k; j++)
            {
                if (counts[j] > 0)
                {
                    for (int t = 0; t < dim; t++)
                    {
                        centers[j][t] = sums[j][t] / counts[j];
                    }
                }
                else
                {
                    // 重新随机选一个点作为质心
                    centers[j] = *ptrs[uid(rng)];
                }
            }
        }
        return labels;
    }

    int outer_ring(vector<unordered_map<int, vector<T>>> &local_vectors)
    {
        // 缓存全局邻居关系和向量数据用于 outer_ring 操作
        auto all_neighbors = hnsw->get_all_neighbors();
        auto all_vectors = hnsw->get_all_vectors();

        // 1. 找出所有有跨节点邻居的候选 boundary vectors（及其跨节点邻居数）
        unordered_map<int, vector<pair<int, int>>> boundary_candidates; // node -> [(vid, cross_cnt)]
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
            unordered_map<int, int> counts; // node -> count
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

    void partition(const string &mode, int redundancy = 1, const std::unordered_map<int, std::vector<int>> &layer_config = {}, const std::string &layer_partition_mode = "round_robin", const std::vector<bool> &layer_replicate_flags = {})
    {
        partition_mode = mode;
        redundancy_level = redundancy;
        redundancy_num = 0;
        this->layer_replicate_flags = layer_replicate_flags;

        int max_level = hnsw->max_level;
        auto vectors = hnsw->get_all_vectors();
        auto neighbors = hnsw->get_all_neighbors();

        // local_vectors: node_id -> map<vid, vector<T>>
        vector<unordered_map<int, vector<T>>> local_vectors(node_num);

        vector_node_map.clear();
        if (mode == "layer")
            vector_node_map.resize(max_level + 1);
        else
            vector_node_map.resize(1);

        if (mode == "round_robin")
        {
            for (const auto &kv : vectors)
            {
                int vid = kv.first;
                int target = vid % node_num;
                vector_node_map[0][vid] = {target};
                local_vectors[target][vid] = kv.second;
            }
        }
        else if (mode == "k-means")
        {
            auto labels = kmeans(vectors, node_num, 100, hnsw->seed);
            int idx = 0;
            for (const auto &kv : vectors)
            {
                int vid = kv.first;
                int target = labels[idx++];
                vector_node_map[0][vid] = {target};
                local_vectors[target][vid] = kv.second;
            }
        }
        else if (mode == "outer_ring")
        {
            // 先用 k-means 分配
            auto labels = kmeans(vectors, node_num, 100, hnsw->seed);
            int idx = 0;
            for (const auto &kv : vectors)
            {
                int vid = kv.first;
                int target = labels[idx++];
                vector_node_map[0][vid] = {target};
                local_vectors[target][vid] = kv.second;
            }

            // 挑选边界向量，重新分配（迭代，支持提前停止）
            int removed_cnt_before = 0;
            for (int it = 0; it < outer_ring_max_iterations; it++)
            {
                int removed_cnt = outer_ring(local_vectors);
                double removed_add_ratio = (removed_cnt_before > 0) ? double(removed_cnt - removed_cnt_before) / removed_cnt_before : 1.0;
                if (removed_cnt == 0 || removed_add_ratio < 0.001)
                    break;
                removed_cnt_before = removed_cnt;
            }
        }
        else if (mode == "layer")
        {
            this->layer_config = layer_config;

            // 初始化路由表（先给第 0 层占位）
            for (const auto &kv : vectors)
            {
                vector_node_map[0][kv.first] = {0};
            }

            for (const auto &level_kv : neighbors)
            {
                int level = level_kv.first;
                std::vector<int> target_nodes;

                if (layer_config.count(level))
                {
                    target_nodes = layer_config.at(level);
                }
                else
                {
                    // 默认分配给所有节点
                    for (int i = 0; i < node_num; i++)
                        target_nodes.push_back(i);
                }

                if (target_nodes.empty())
                    continue;

                bool is_replicate_level = (level >= 0 && level < (int)this->layer_replicate_flags.size())
                                              ? this->layer_replicate_flags[level]
                                              : false;

                // split 模式才需要计算层内分区策略 (如 KMeans)
                std::unordered_map<int, int> vid_partition_map;
                if (!is_replicate_level && layer_partition_mode == "k-means" && target_nodes.size() > 1)
                {
                    // 收集当前层的所有向量构建子集（key 使用真实 vid）
                    std::unordered_map<int, std::vector<T>> subset_vectors;
                    for (const auto &node_kv : level_kv.second)
                    {
                        int vid = node_kv.first;
                        if (vectors.count(vid))
                            subset_vectors[vid] = vectors.at(vid);
                    }

                    int sub_cnt = (int)subset_vectors.size();
                    if (sub_cnt > 0)
                    {
                        // 执行 KMeans (k 为目标节点数)
                        int k = std::min((int)target_nodes.size(), sub_cnt);
                        auto labels = kmeans(subset_vectors, k, 100, hnsw->seed);
                        // kmeans() 内部用 for(const auto &kv : data) 迭代，
                        // 此处用相同顺序遍历 subset_vectors 来对齐 labels
                        int i = 0;
                        for (const auto &kv : subset_vectors)
                        {
                            vid_partition_map[kv.first] = labels[i++];
                        }
                    }
                }

                const auto &node_map = level_kv.second;
                LOG_INFO_STREAM("[layer partition] level=" << level << " vids=" << node_map.size() << " mode=" << (is_replicate_level ? "replicate" : "split") << " target_nodes=[" << [&]{ std::string s; for(int n:target_nodes) s+=std::to_string(n)+","; return s; }() << "]");
                for (const auto &node_kv : node_map)
                {
                    int vid = node_kv.first;

                    if (is_replicate_level)
                    {
                        std::vector<int> route_nodes;
                        route_nodes.reserve(target_nodes.size());
                        for (int target_node : target_nodes)
                        {
                            if (target_node < 0 || target_node >= node_num)
                                continue;
                            if (vectors.count(vid))
                            {
                                local_vectors[target_node][vid] = vectors.at(vid);
                            }
                            route_nodes.push_back(target_node);
                        }
                        if (!route_nodes.empty())
                        {
                            vector_node_map[level][vid] = std::move(route_nodes);
                        }
                    }
                    else
                    {
                        // 确定目标节点索引
                        int target_idx = 0;
                        if (!vid_partition_map.empty() && vid_partition_map.count(vid))
                        {
                            target_idx = vid_partition_map.at(vid);
                        }
                        else
                        {
                            // 默认层内 Round Robin 分配
                            target_idx = vid % target_nodes.size();
                        }

                        // 边界检查
                        if (target_idx < 0 || target_idx >= (int)target_nodes.size())
                            target_idx = 0;

                        int target_node = target_nodes[target_idx];

                        if (target_node >= 0 && target_node < node_num)
                        {
                            // 复制向量数据
                            if (vectors.count(vid))
                            {
                                local_vectors[target_node][vid] = vectors.at(vid);
                            }

                            // 路由表按层记录
                            vector_node_map[level][vid] = {target_node};
                        }
                    }
                }
            }
            
            // 修正 entry_point 的路由，指向最高层所在的节点（replicate层写入全部节点）
            if (layer_config.count(max_level) && !layer_config.at(max_level).empty()) {
                bool ep_replicate = (max_level >= 0 && max_level < (int)this->layer_replicate_flags.size())
                                        ? this->layer_replicate_flags[max_level]
                                        : false;
                if (ep_replicate)
                {
                    std::vector<int> route_nodes;
                    for (int target_node : layer_config.at(max_level))
                    {
                        if (target_node >= 0 && target_node < node_num)
                            route_nodes.push_back(target_node);
                    }
                    if (!route_nodes.empty())
                        vector_node_map[max_level][hnsw->entry_point_id] = std::move(route_nodes);
                }
                else
                {
                    int target_node = layer_config.at(max_level)[hnsw->entry_point_id % layer_config.at(max_level).size()];
                    vector_node_map[max_level][hnsw->entry_point_id] = {target_node};
                }
            }
        }

        // redundancy: 将邻居复制到本地
        if (redundancy > 0)
        {
            for (int r = 0; r < redundancy; r++)
            {
                // 收集更新，避免遍历时修改
                vector<unordered_map<int, vector<T>>> updates(node_num);
                // 收集路由表更新
                vector<unordered_map<int, std::vector<int>>> route_updates(vector_node_map.size());

                for (const auto &level_kv : neighbors)
                {
                    int level = level_kv.first;
                    if (level < 0 || level >= (int)vector_node_map.size())
                        continue;

                    const auto &route_map = vector_node_map[level];

                    for (const auto &neighbor_kv : level_kv.second)
                    {
                        int vid = neighbor_kv.first;
                        const auto &neighbor_set = neighbor_kv.second;

                        // 查找该向量在该层的路由
                        auto it = route_map.find(vid);
                        if (it == route_map.end())
                            continue;

                        // 对路由覆盖的所有节点复制邻居
                        for (int node_id : it->second)
                        {
                            if (node_id < 0 || node_id >= node_num)
                                continue;
                            for (int neighbor_id : neighbor_set)
                            {
                                if (!local_vectors[node_id].count(neighbor_id) &&
                                    !updates[node_id].count(neighbor_id) &&
                                    vectors.count(neighbor_id))
                                {
                                    updates[node_id][neighbor_id] = vectors.at(neighbor_id);
                                    redundancy_num++;

                                    // 更新路由表：将 neighbor_id 添加到 node_id
                                    auto &route = route_updates[level][neighbor_id];
                                    if (std::find(route.begin(), route.end(), node_id) == route.end())
                                    {
                                        route.push_back(node_id);
                                    }
                                }
                            }
                        }
                    }
                }

                // 应用向量更新
                for (int node_id = 0; node_id < node_num; node_id++)
                {
                    for (const auto &kv : updates[node_id])
                        local_vectors[node_id][kv.first] = kv.second;
                }

                // 应用路由表更新
                for (size_t lvl = 0; lvl < route_updates.size(); lvl++)
                {
                    for (const auto &kv : route_updates[lvl])
                    {
                        int vid = kv.first;
                        const auto &nodes = kv.second;
                        auto &existing = vector_node_map[lvl][vid];
                        for (int node_id : nodes)
                        {
                            if (std::find(existing.begin(), existing.end(), node_id) == existing.end())
                            {
                                existing.push_back(node_id);
                            }
                        }
                    }
                }
            }
        }

        // 打印各节点分到的向量数
        for (int i = 0; i < node_num; i++)
            LOG_INFO_STREAM("[partition] node " << i << " local_vectors=" << local_vectors[i].size());

        // 为每个 node 创建 local_hnsw（浅拷贝/新实例），并设置 vectors/neighbors/entrypoint/max_level
        for (int i = 0; i < node_num; i++)
        {
            // 释放旧的 local_hnsw（避免重复 partition 时内存泄漏）
            if (nodes[i].local_hnsw != nullptr && nodes[i].local_hnsw != hnsw)
            {
                delete nodes[i].local_hnsw;
                nodes[i].local_hnsw = nullptr;
            }
            size_t local_max_elements = std::max(local_vectors[i].size(), (size_t)1000);
            HNSWGraph<T> *local_hnsw = new HNSWGraph<T>(hnsw->dim, hnsw->m, hnsw->ef_construction, hnsw->ef_search, hnsw->limited_level, local_max_elements, hnsw->seed);
            // 通过 add_vector 逐个添加（建立连续内存布局和邻居关系）
            for (const auto &kv : local_vectors[i])
            {
                local_hnsw->add_vector(kv.first, kv.second);
            }
            nodes[i].local_hnsw = local_hnsw;
        }

        // Distribute partitions to remote nodes
        for (int i = 0; i < node_num; i++)
        {
            Node<T> &target_node = nodes[i];
            if (target_node.is_remote_node)
            {
                // 使用通信中间层获取连接
                int sock = transport->acquire_connection(i);
                if (sock < 0)
                {
                    LOG_DEBUG_STREAM("Failed to get connection to node " << i);
                    continue;
                }

                Buffer buf;
                buf.write_val((int)MSG_INIT_GRAPH);
                serialize_routing(buf, vector_node_map);
                serialize_hnsw(buf, target_node.local_hnsw);

                uint64_t msg_size = buf.data.size();
                transport->send_raw(sock, &msg_size, sizeof(uint64_t));
                transport->send_raw(sock, buf.data.data(), msg_size);

                // Wait for ACK from worker node
                char ack = 0;
                try
                {
                    transport->recv_raw(sock, &ack, 1);
                    if (ack == 1)
                    {
                        LOG_DEBUG_STREAM("Partition sent to node " << i << " - ACK received");
                    }
                    else
                    {
                        LOG_DEBUG_STREAM("Warning: Invalid ACK from node " << i);
                    }
                }
                catch (...)
                {
                    LOG_DEBUG_STREAM("Warning: No ACK from node " << i);
                }
                // MSG_INIT_GRAPH 后服务器会关闭连接，不复用
                transport->close_connection(sock);
            }
            else
            {
                Buffer buf;
                serialize_routing(buf, vector_node_map);
                serialize_hnsw(buf, target_node.local_hnsw);
                deserialize_routing(buf, vector_node_map);

                if (target_node.local_hnsw)
                    delete target_node.local_hnsw;
                target_node.local_hnsw = deserialize_hnsw<T>(buf);
            }
        }
    }

    std::vector<int> query(int query_id, const vector<T> &query_vector, Payload<T> &payload, int k)
    {
        if (is_reduce)
        {
            std::vector<pair<T, int>> all_results;

            // Reduce模式查询：并行向所有节点发送查询并合并结果
            struct NodeQueryOutput
            {
                std::vector<pair<T, int>> results;
                Payload<T> payload;
            };

            std::vector<std::future<NodeQueryOutput>> futs;
            futs.reserve(node_num);
 
            for (int i = 0; i < node_num; i++)
            {
                futs.emplace_back(std::async(std::launch::async, [this, query_id, &query_vector, k, i]() -> NodeQueryOutput {
                    NodeQueryOutput out;
                    Node<T> &node = nodes[i];

                    // 每个节点使用独立payload，避免并发写同一个payload
                    Payload<T> node_payload;

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
                        serialize_payload(buf, node_payload);

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

                            // 非持久连接：远程节点已关闭，标记为无效
                            transport->release_connection(i, sock, false);

                            Buffer resp_buf;
                            resp_buf.data = std::move(resp_data);

                            int result_cnt = resp_buf.read_val<int>();
                            out.results.reserve(result_cnt);
                            for (int j = 0; j < result_cnt; j++)
                            {
                                T dist = resp_buf.read_val<T>();
                                int id = resp_buf.read_val<int>();
                                out.results.push_back({dist, id});
                            }
                            deserialize_payload(resp_buf, node_payload);
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

                            auto pq = node_payload.results;
                            out.results.reserve(pq.size());
                            while (!pq.empty())
                            {
                                out.results.push_back(pq.top());
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
                payload.remote_transfer_latency += out.payload.remote_transfer_latency;
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

    // 跨节点传输：统计并将查询转发到目标节点
    std::vector<int> transfer_to_remote(int query_id, const vector<T> &query_vector, Payload<T> &payload, int k)
    {
        payload.remote_transfer_cnt += 1;

        int require_id = payload.require_id;
        int route_level = payload.start_level;
        if (route_level < 0 || route_level >= (int)vector_node_map.size())
            route_level = 0;

        auto choose_target_node = [this](const std::vector<int> &candidates) -> int {
            std::vector<int> valid;
            valid.reserve(candidates.size());
            for (int node_id : candidates)
            {
                if (node_id >= 0 && node_id < this->node_num)
                    valid.push_back(node_id);
            }
            if (valid.empty())
                return 0;
            if (valid.size() == 1)
                return valid[0];
            static thread_local std::mt19937 gen(std::random_device{}());
            std::uniform_int_distribution<size_t> dist(0, valid.size() - 1);
            return valid[dist(gen)];
        };

        int target_node_id = 0;
        if (!vector_node_map.empty())
        {
            auto it = vector_node_map[route_level].find(require_id);
            if (it != vector_node_map[route_level].end())
            {
                target_node_id = choose_target_node(it->second);
            }
            else if (route_level != 0)
            {
                auto it0 = vector_node_map[0].find(require_id);
                if (it0 != vector_node_map[0].end())
                    target_node_id = choose_target_node(it0->second);
            }
        }

        Node<T> &target_node = nodes[target_node_id];

        if (is_distributed && target_node.is_remote_node)
        {
            // 记录发送时间
            auto now = std::chrono::system_clock::now();
            payload.send_timestamp = std::chrono::duration<double, std::milli>(now.time_since_epoch()).count();

            // 使用通信中间层获取连接
            int sock = transport->acquire_connection(target_node_id);
            if (sock < 0)
            {
                LOG_DEBUG_STREAM("[Cluster] ERROR: Failed to get connection for query " << query_id << " to node " << target_node_id);
                return {};
            }

            Buffer buf;
            buf.write_val((int)MSG_SEARCH);
            buf.write_val(query_id);
            buf.write_val(k);
            buf.write_vec(query_vector);
            serialize_payload(buf, payload);

            uint64_t msg_size = buf.data.size();
            
            LOG_DEBUG_STREAM("[Cluster] Sending MSG_SEARCH to node " << target_node_id << " for query " << query_id << ", transfer_cnt=" << payload.remote_transfer_cnt << ", msg_size=" << msg_size);
            
            try
            {
                transport->send_raw(sock, &msg_size, sizeof(uint64_t));
                transport->send_raw(sock, buf.data.data(), msg_size);

                // 非持久连接：远程节点已关闭，标记为无效
                transport->release_connection(target_node_id, sock, false);
                
                LOG_DEBUG_STREAM("[Cluster] MSG_SEARCH successfully sent to node " << target_node_id << " for query " << query_id);
                
                return {};
            }
            catch (...)
            {
                transport->release_connection(target_node_id, sock, false);
                return {};
            }
        }
        else
        {
            payload.remote_transfer_latency += payload.size_bits() / (bandwidth * 1000) + 0.5; // ms
            // 本地查询不计入网络延迟
            return target_node.query(query_id, query_vector, payload, k);
        }
    }

    // 保存 cluster 状态到文件
    bool save(const string &filepath)
    {
        try
        {
            json j;
            j["node_num"] = node_num;
            j["bandwidth"] = bandwidth;
            j["partition_mode"] = partition_mode;
            j["outer_ring_max_iterations"] = outer_ring_max_iterations;
            j["redundancy_level"] = redundancy_level;
            j["redundancy_num"] = redundancy_num;
            j["is_distributed"] = is_distributed;
            j["is_reduce"] = is_reduce;
            j["add_round_robin_counter"] = add_round_robin_counter.load();

            if (partition_mode == "layer")
            {
                j["layer_config"] = layer_config;
                j["layer_replicate_flags"] = layer_replicate_flags;
            }

            // 保存 vector_node_map（layer: 多层；非 layer: 仅第0层）
            if (partition_mode == "layer")
            {
                json jmaps = json::array();
                for (const auto &level_map : vector_node_map)
                {
                    json jmap = json::object();
                    for (const auto &kv : level_map)
                    {
                        jmap[std::to_string(kv.first)] = kv.second;
                    }
                    jmaps.push_back(std::move(jmap));
                }
                j["vector_node_map"] = std::move(jmaps);
            }
            else
            {
                json jmap = json::object();
                if (!vector_node_map.empty())
                {
                    for (const auto &kv : vector_node_map[0])
                    {
                        if (!kv.second.empty())
                            jmap[std::to_string(kv.first)] = kv.second[0];
                    }
                }
                j["vector_node_map"] = std::move(jmap);
            }

            // 保存主 HNSW 图
            string hnsw_path = filepath + ".hnsw.bin";
            if (hnsw)
            {
                hnsw->save(hnsw_path);
                j["hnsw_path"] = hnsw_path;
            }
            else
            {
                j["hnsw_path"] = "";
            }

            // 保存每个节点的 local_hnsw（如果存在）
            json jnodes = json::array();
            for (int i = 0; i < node_num; i++)
            {
                json jnode;
                jnode["node_id"] = nodes[i].node_id;
                jnode["num_queries"] = nodes[i].num_queries;
                jnode["is_remote_node"] = nodes[i].is_remote_node;
                jnode["ip_addr"] = nodes[i].ip_addr;
                jnode["port"] = nodes[i].port;

                if (nodes[i].local_hnsw && nodes[i].local_hnsw != hnsw)
                {
                    string local_hnsw_path = filepath + ".node" + std::to_string(i) + ".bin";
                    nodes[i].local_hnsw->save(local_hnsw_path);
                    jnode["local_hnsw_path"] = local_hnsw_path;
                }
                else
                {
                    jnode["local_hnsw_path"] = "";
                }
                jnodes.push_back(std::move(jnode));
            }
            j["nodes"] = std::move(jnodes);

            std::string out = j.dump(4);
            int fd = ::open(filepath.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
            if (fd == -1)
                return false;

            if (flock(fd, LOCK_EX | LOCK_NB) == -1)
            {
                ::close(fd);
                return false;
            }

            const char *buf = out.data();
            std::size_t total = out.size();
            std::size_t written = 0;
            while (written < total)
            {
                ssize_t w = ::write(fd, buf + written, total - written);
                if (w <= 0)
                {
                    flock(fd, LOCK_UN);
                    ::close(fd);
                    return false;
                }
                written += (std::size_t)w;
            }

            ::fsync(fd);
            flock(fd, LOCK_UN);
            ::close(fd);
            return true;
        }
        catch (...)
        {
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

        cluster->bandwidth = j.value("bandwidth", 1000.0);
        cluster->partition_mode = j.value("partition_mode", "");
        cluster->outer_ring_max_iterations = j.value("outer_ring_max_iterations", 0);
        cluster->redundancy_level = j.value("redundancy_level", 0);
        cluster->redundancy_num = j.value("redundancy_num", (uint64_t)0);
        cluster->is_distributed = j.value("is_distributed", false);
        cluster->is_reduce = j.value("is_reduce", false);
        cluster->add_round_robin_counter = j.value("add_round_robin_counter", 0);

        if (cluster->partition_mode == "layer")
        {
            if (j.contains("layer_config") && j["layer_config"].is_object())
            {
                cluster->layer_config.clear();
                for (auto const &kv : j["layer_config"].items())
                {
                    int level = std::stoi(kv.key());
                    if (kv.value().is_array())
                    {
                        cluster->layer_config[level] = kv.value().get<std::vector<int>>();
                    }
                }
            }
            if (j.contains("layer_replicate_flags") && j["layer_replicate_flags"].is_array())
            {
                cluster->layer_replicate_flags = j["layer_replicate_flags"].get<std::vector<bool>>();
            }
        }

        // 加载 vector_node_map（layer: 多层；非 layer: 仅第0层）
        if (j.contains("vector_node_map"))
        {
            cluster->vector_node_map.clear();
            if (cluster->partition_mode == "layer")
            {
                if (!j["vector_node_map"].is_array())
                    throw std::runtime_error("vector_node_map format error for layer mode");
                cluster->vector_node_map.resize(j["vector_node_map"].size());
                for (size_t l = 0; l < j["vector_node_map"].size(); l++)
                {
                    const auto &jmap = j["vector_node_map"][l];
                    for (auto const &kv : jmap.items())
                    {
                        int vid = std::stoi(kv.key());
                        if (kv.value().is_array())
                        {
                            cluster->vector_node_map[l][vid] = kv.value().get<std::vector<int>>();
                        }
                        else
                        {
                            int node_id = kv.value().get<int>();
                            cluster->vector_node_map[l][vid] = {node_id};
                        }
                    }
                }
            }
            else
            {
                if (!j["vector_node_map"].is_object())
                    throw std::runtime_error("vector_node_map format error for non-layer mode");
                cluster->vector_node_map.resize(1);
                for (auto const &kv : j["vector_node_map"].items())
                {
                    int vid = std::stoi(kv.key());
                    if (kv.value().is_array())
                    {
                        cluster->vector_node_map[0][vid] = kv.value().get<std::vector<int>>();
                    }
                    else
                    {
                        int node_id = kv.value().get<int>();
                        cluster->vector_node_map[0][vid] = {node_id};
                    }
                }
            }
        }

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

// Implement Node methods
template <typename T>
Node<T>::~Node()
{
    stop_server.store(true);
    if (sockfd != -1)
        close(sockfd);
    if (server_thread.joinable())
        server_thread.join();
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

    // 使用通信中间层启动服务端
    sockfd = cluster->transport->start_server(port);
    if (sockfd < 0)
    {
        LOG_DEBUG_STREAM("Failed to start server on port " << port);
        return;
    }

    try
    {
        server_thread = std::thread(&Node<T>::tcp_server_loop, this);
    }
    catch (const std::exception &e)
    {
        LOG_DEBUG_STREAM("Failed to start server thread: " << e.what());
        cluster->transport->stop_server();
        sockfd = -1;
    }
}

template <typename T>
void Node<T>::tcp_server_loop()
{
    LOG_DEBUG_STREAM("Node " << node_id << " tcp_server_loop started on port " << port);

    while (!stop_server.load())
    {
        int client_fd = cluster->transport->accept_client();
        if (client_fd < 0)
        {
            LOG_DEBUG_STREAM("[Node " << node_id << "] Failed to accept");
            if (stop_server.load())
                break;
            continue;
        }

        // int buf = 4 * 1024 * 1024;
        // setsockopt(client_fd, SOL_SOCKET, SO_SNDBUF, &buf, sizeof(buf));
        // setsockopt(client_fd, SOL_SOCKET, SO_RCVBUF, &buf, sizeof(buf));

        std::thread(&Node<T>::handle_client, this, client_fd).detach();
    }
}

template <typename T>
void Node<T>::handle_client(int client_fd)
{
    active_handlers++;

    // 持久连接模式：循环处理多个消息
    while (true)
    {
        try
        {
            uint64_t msg_size = 0;
            ssize_t r = recv(client_fd, &msg_size, sizeof(uint64_t), MSG_WAITALL);
            if (r <= 0)
            {
                // 连接关闭或错误
                break;
            }

            std::vector<char> buffer(msg_size);
            recv_all(client_fd, buffer.data(), msg_size);

            Buffer buf;
            buf.data = std::move(buffer);

            int msg_type = buf.read_val<int>();

            if (msg_type == MSG_SEARCH)
            {
                int query_id = buf.read_val<int>();
                int k = buf.read_val<int>();
                std::vector<T> query_vector = buf.read_vec<T>();

                Payload<T> payload;
                deserialize_payload(buf, payload);

                LOG_DEBUG_STREAM("[Node " << node_id << "] Received MSG_SEARCH for query " << query_id << ", transfer_cnt=" << payload.remote_transfer_cnt << ", origin=" << payload.origin_node_id);

                // 计算网络延迟
                auto now = std::chrono::system_clock::now();
                double recv_timestamp = std::chrono::duration<double, std::milli>(now.time_since_epoch()).count();
                double latency = recv_timestamp - payload.send_timestamp;
                payload.remote_transfer_latency += latency;

                std::vector<int> results = this->query(query_id, query_vector, payload, k);

                LOG_DEBUG_STREAM("[Node " << node_id << "] MSG_SEARCH query " << query_id << " returned, results.size=" << results.size() << ", continue_search=" << payload.continue_search << ", transfer_cnt=" << payload.remote_transfer_cnt);

                // 持久连接：不关闭，继续等待下一个消息
                continue;
            }
            else if (msg_type == MSG_INIT_GRAPH)
            {
                LOG_DEBUG_STREAM("Node " << node_id << " receiving graph partition...");
                deserialize_routing(buf, cluster->vector_node_map);

                if (local_hnsw)
                    delete local_hnsw;
                local_hnsw = deserialize_hnsw<T>(buf);

                LOG_DEBUG_STREAM("Node " << node_id << " initialized with " << local_hnsw->get_cur_element_count() << " vectors.");

                if (buf.read_pos != buf.data.size())
                {
                    LOG_DEBUG_STREAM("Warning: Buffer not fully consumed! read_pos=" << buf.read_pos << " size=" << buf.data.size());
                }

                // Send ACK back to master
                char ack = 1;
                send_all(client_fd, &ack, 1);
                break;
            }
            else if (msg_type == MSG_ADD_VECTOR)
            {
                int vid = buf.read_val<int>();
                std::vector<T> vec = buf.read_vec<T>();

                char ack = 1;  // Success by default
                try {
                    if (cluster && cluster->is_reduce)
                    {
                        // Reduce 模式：调用 cluster->add_vector 进行轮询分发到各节点
                        cluster->add_vector(vid, vec);
                    }
                    else if (cluster && !cluster->is_reduce)
                    {
                        if (cluster->hnsw)
                        {
                            cluster->hnsw->add_vector(vid, vec);
                        }
                    }
                } catch (const std::exception &e) {
                    LOG_DEBUG_STREAM("[Node " << node_id << "] Error adding vector " << vid << ": " << e.what());
                    ack = 0;  // Failure
                } catch (...) {
                    LOG_DEBUG_STREAM("[Node " << node_id << "] Unknown error adding vector " << vid);
                    ack = 0;  // Failure
                }

                // 发送 ACK 确认 (always send, even on error)
                send_all(client_fd, &ack, 1);
                // 非持久连接：主动关闭
                break;
            }
            else if (msg_type == MSG_ADD_VECTOR_LOCAL)
            {
                // 来自其他节点的添加请求，直接添加到本地 local_hnsw（不再路由）
                int vid = buf.read_val<int>();
                std::vector<T> vec = buf.read_vec<T>();

                char ack = 1;  // Success by default
                try {
                    if (local_hnsw == nullptr && cluster && cluster->is_reduce)
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
                } catch (const std::exception &e) {
                    LOG_DEBUG_STREAM("[Node " << node_id << "] Error adding local vector " << vid << ": " << e.what());
                    ack = 0;  // Failure
                } catch (...) {
                    LOG_DEBUG_STREAM("[Node " << node_id << "] Unknown error adding local vector " << vid);
                    ack = 0;  // Failure
                }

                // 发送 ACK 确认 (always send, even on error)
                send_all(client_fd, &ack, 1);
                // 非持久连接：主动关闭
                break;
            }
            else if (msg_type == MSG_QUERY)
            {
                int query_id = buf.read_val<int>();
                int k = buf.read_val<int>();
                std::vector<T> query_vector = buf.read_vec<T>();

                Payload<T> payload;
                deserialize_payload(buf, payload);

                std::vector<int> results;
                
                // Reduce模式：调用cluster级别查询，并行向所有节点发送查询并合并结果
                if (cluster && cluster->is_reduce)
                {
                    results = cluster->query(query_id, query_vector, payload, k);
                }
                else
                {
                    // 非Reduce模式：执行本地查询
                    this->query(query_id, query_vector, payload, k);
                    results = payload.topk_ids(k);
                }
                
                LOG_DEBUG_STREAM("[Node " << node_id << "] MSG_QUERY: results.size=" << results.size());

                Buffer resp_buf;
                resp_buf.write_val((int)results.size());
                for (const auto &id : results)
                {
                    resp_buf.write_val((T)0.0); // distance placeholder
                    resp_buf.write_val(id);
                }
                serialize_payload(resp_buf, payload);

                uint64_t resp_size = resp_buf.data.size();
                send_all(client_fd, &resp_size, sizeof(uint64_t));
                send_all(client_fd, resp_buf.data.data(), resp_size);
                continue;
            }
            else if (msg_type == MSG_QUERY_LOCAL)
            {
                // Reduce模式下的本地查询请求（来自其他节点，避免递归）
                int query_id = buf.read_val<int>();
                int k = buf.read_val<int>();
                std::vector<T> query_vector = buf.read_vec<T>();

                Payload<T> payload;
                deserialize_payload(buf, payload);

                std::vector<pair<T, int>> results;
                
                // 只执行本地HNSW查询
                if (local_hnsw)
                {
                    local_hnsw->query(query_vector, payload, k);
                    auto pq = payload.results;
                    while (!pq.empty())
                    {
                        results.push_back(pq.top());
                        pq.pop();
                    }
                }
                
                LOG_DEBUG_STREAM("[Node " << node_id << "] MSG_QUERY_LOCAL: results.size=" << results.size());

                Buffer resp_buf;
                resp_buf.write_val((int)results.size());
                for (const auto &p : results)
                {
                    resp_buf.write_val(p.first);   // distance
                    resp_buf.write_val(p.second);  // id
                }
                serialize_payload(resp_buf, payload);

                uint64_t resp_size = resp_buf.data.size();
                send_all(client_fd, &resp_size, sizeof(uint64_t));
                send_all(client_fd, resp_buf.data.data(), resp_size);
                continue;
            }
            else if (msg_type == MSG_PARTITION)
            {
                std::string mode = buf.read_string();
                int redundancy = buf.read_val<int>();
                std::unordered_map<int, std::vector<int>> layer_config;
                std::string layer_partition_mode = "round_robin";
                std::vector<bool> layer_replicate_flags;
                if (mode == "layer")
                {
                    deserialize_layer_config(buf, layer_config);
                    layer_partition_mode = buf.read_string();
                    deserialize_layer_replicate_flags(buf, layer_replicate_flags);
                }

                if (cluster)
                {
                    cluster->partition(mode, redundancy, layer_config, layer_partition_mode, layer_replicate_flags);
                }

                char ack = 1;
                send_all(client_fd, &ack, 1);
                // 一次性操作，退出循环
                break;
            }
            else if (msg_type == MSG_LOAD_CLUSTER)
            {
                std::string filepath = buf.read_string();
                bool ok = false;
                if (cluster)
                {
                    Cluster<T> *loaded = nullptr;
                    try
                    {
                        loaded = Cluster<T>::load(filepath);
                    }
                    catch (...)
                    {
                        loaded = nullptr;
                    }

                    if (loaded && loaded->node_num == cluster->node_num)
                    {
                        cluster->bandwidth = loaded->bandwidth;
                        cluster->partition_mode = loaded->partition_mode;
                        cluster->outer_ring_max_iterations = loaded->outer_ring_max_iterations;
                        cluster->redundancy_level = loaded->redundancy_level;
                        cluster->redundancy_num = loaded->redundancy_num;
                        cluster->is_distributed = loaded->is_distributed;
                        cluster->is_reduce = loaded->is_reduce;
                        cluster->add_round_robin_counter.store(loaded->add_round_robin_counter.load());
                        cluster->vector_node_map = std::move(loaded->vector_node_map);
                        cluster->layer_config = std::move(loaded->layer_config);
                        cluster->layer_replicate_flags = std::move(loaded->layer_replicate_flags);

                        if (cluster->hnsw && cluster->hnsw != loaded->hnsw)
                        {
                            delete cluster->hnsw;
                        }
                        cluster->hnsw = loaded->hnsw;
                        loaded->hnsw = nullptr;

                        for (int i = 0; i < cluster->node_num && i < (int)loaded->nodes.size(); i++)
                        {
                            cluster->nodes[i].num_queries = loaded->nodes[i].num_queries;
                            cluster->nodes[i].is_remote_node = loaded->nodes[i].is_remote_node;
                            cluster->nodes[i].ip_addr = loaded->nodes[i].ip_addr;
                            cluster->nodes[i].port = loaded->nodes[i].port;

                            if (cluster->nodes[i].local_hnsw != nullptr && cluster->nodes[i].local_hnsw != cluster->hnsw)
                            {
                                delete cluster->nodes[i].local_hnsw;
                            }
                            cluster->nodes[i].local_hnsw = loaded->nodes[i].local_hnsw;
                            loaded->nodes[i].local_hnsw = nullptr;
                        }

                        ok = true;
                    }

                    if (loaded)
                    {
                        delete loaded;
                    }
                }
                char ack = ok ? 1 : 0;
                send_all(client_fd, &ack, 1);
                // 一次性操作，退出循环
                break;
            }
            else if (msg_type == MSG_LOAD_HNSW)
            {
                std::string filepath = buf.read_string();
                bool ok = false;
                if (cluster)
                {
                    HNSWGraph<T> *loaded_hnsw = nullptr;
                    try
                    {
                        loaded_hnsw = HNSWGraph<T>::load(filepath);
                    }
                    catch (...)
                    {
                        loaded_hnsw = nullptr;
                    }

                    if (loaded_hnsw)
                    {
                        if (cluster->hnsw && cluster->hnsw != loaded_hnsw)
                        {
                            delete cluster->hnsw;
                        }
                        cluster->hnsw = loaded_hnsw;

                        ok = true;
                    }
                }
                char ack = ok ? 1 : 0;
                send_all(client_fd, &ack, 1);
                // 一次性操作，退出循环
                break;
            }
            else if (msg_type == MSG_SAVE_HNSW)
            {
                std::string filepath = buf.read_string();
                bool ok = false;
                if (cluster && cluster->hnsw)
                {
                    try
                    {
                        cluster->hnsw->save(filepath);
                        ok = true;
                    }
                    catch (...) {}
                }
                char ack = ok ? 1 : 0;
                send_all(client_fd, &ack, 1);
                // 一次性操作，退出循环
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
                    cj["redundancy_level"] = cluster->redundancy_level;
                    cj["redundancy_num"] = cluster->redundancy_num;
                    cj["is_distributed"] = cluster->is_distributed;
                    cj["is_reduce"] = cluster->is_reduce;
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
                        // 只有本地节点才有 vector_count
                        if (!cluster->nodes[i].is_remote_node && cluster->nodes[i].local_hnsw)
                        {
                            nj["vector_count"] = (int)cluster->nodes[i].local_hnsw->get_cur_element_count();
                        }
                        nodes_json.push_back(std::move(nj));
                    }
                    cj["nodes"] = std::move(nodes_json);
                    info["cluster"] = std::move(cj);

                    if (cluster->hnsw)
                    {
                        json hj;
                        hj["dim"] = cluster->hnsw->dim;
                        hj["m"] = cluster->hnsw->m;
                        hj["ef_construction"] = cluster->hnsw->ef_construction;
                        hj["ef_search"] = cluster->hnsw->ef_search;
                        hj["limited_level"] = cluster->hnsw->limited_level;
                        hj["max_level"] = cluster->hnsw->max_level;
                        info["hnsw"] = std::move(hj);
                    }
                    else
                    {
                        info["hnsw"] = nullptr;
                    }
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
                // 一次性操作，退出循环
                break;
            }
            else if (msg_type == MSG_SAVE_CLUSTER)
            {
                std::string filepath = buf.read_string();
                char ack = 0;
                if (cluster)
                {
                    ack = cluster->save(filepath) ? 1 : 0;
                }
                send_all(client_fd, &ack, 1);
                // 一次性操作，退出循环
                break;
            }
            else if (msg_type == MSG_SEARCH_RESULT)
            {
                // 接收异步查询结果
                int query_id = buf.read_val<int>();
                std::vector<int> results = buf.read_vec<int>();

                LOG_DEBUG_STREAM("[Node " << node_id << "] Received MSG_SEARCH_RESULT for query " << query_id);

                Payload<T> payload;
                deserialize_payload(buf, payload);

                // 设置到 pending_queries
                std::lock_guard<std::mutex> lk(pending_queries_mu);
                auto& pq = pending_queries[query_id];
                pq.results = std::move(results);
                pq.payload = std::move(payload);
                pq.completed = true;
                pending_queries_cv.notify_all();

                LOG_DEBUG_STREAM("[Node " << node_id << "] Notified waiting thread for query " << query_id);
                // 持久连接：继续等待下一个消息
                continue;
            }
        }
        catch (const std::exception &e)
        {
            LOG_DEBUG_STREAM("Error handling client: " << e.what());
            break;
        }
        catch (...)
        {
            break;
        }
    }

    close_connection_hard(client_fd);
    active_handlers--;
}

template <typename T>
void Node<T>::send_result_to_origin(int query_id, const std::vector<int>& results, Payload<T>& payload)
{
    if (payload.origin_node_id < 0 || payload.origin_node_id >= cluster->node_num)
    {
        LOG_DEBUG_STREAM("[Node " << node_id << "] ERROR: Invalid origin_node_id=" << payload.origin_node_id);
        return;
    }
    
    Node<T>& origin_node = cluster->nodes[payload.origin_node_id];
    
    // 如果 origin_node 是本地节点，直接设置结果
    if (!origin_node.is_remote_node)
    {
        LOG_DEBUG_STREAM("[Node " << node_id << "] Setting result locally for query " << query_id);
        std::lock_guard<std::mutex> lk(origin_node.pending_queries_mu);
        auto& pq = origin_node.pending_queries[query_id];
        pq.results = results;
        pq.payload = payload;
        pq.completed = true;
        origin_node.pending_queries_cv.notify_all();
        LOG_DEBUG_STREAM("[Node " << node_id << "] Result set and notified for query " << query_id);
        return;
    }
    
    // origin_node 是远程节点，通过网络发送结果
    LOG_DEBUG_STREAM("[Node " << node_id << "] Sending result to remote origin node " << payload.origin_node_id);
    
    // 使用通信中间层获取连接
    int sock = cluster->transport->acquire_connection(payload.origin_node_id);
    if (sock < 0)
    {
        LOG_DEBUG_STREAM("[Node " << node_id << "] Failed to get connection to origin node");
        return;
    }
    
    try
    {
        Buffer buf;
        buf.write_val((int)MSG_SEARCH_RESULT);
        buf.write_val(query_id);
        buf.write_vec(results);
        serialize_payload(buf, payload);
        
        uint64_t msg_size = buf.data.size();
        cluster->transport->send_raw(sock, &msg_size, sizeof(uint64_t));
        cluster->transport->send_raw(sock, buf.data.data(), msg_size);
        
        // 非持久连接：远程节点已关闭，标记为无效
        cluster->transport->release_connection(payload.origin_node_id, sock, false);
        LOG_DEBUG_STREAM("[Node " << node_id << "] Result sent to origin node " << payload.origin_node_id);
    }
    catch (...)
    {
        LOG_DEBUG_STREAM("[Node " << node_id << "] Exception sending result to origin node");
        cluster->transport->release_connection(payload.origin_node_id, sock, false);
    }
}

template <typename T>
void Node<T>::wait_for_async_result(int query_id, std::vector<int>& results, Payload<T>& payload)
{
    LOG_DEBUG_STREAM("[Node " << node_id << "] Waiting for async result, query_id=" << query_id);
    std::unique_lock<std::mutex> lk(pending_queries_mu);
    pending_queries_cv.wait(lk, [this, query_id]() {
        return pending_queries[query_id].completed;
    });
    
    LOG_DEBUG_STREAM("[Node " << node_id << "] Received async result, query_id=" << query_id);
    auto& pq = pending_queries[query_id];
    results = std::move(pq.results);
    payload = std::move(pq.payload);
    pending_queries.erase(query_id);
}

template <typename T>
std::vector<int> Node<T>::query(int query_id, const vector<T> &query_vector, Payload<T> &payload, int k)
{
    // 如果 origin_node_id 未设置，说明这是起始节点
    bool is_origin = false;
    if (payload.origin_node_id < 0)
    {
        payload.origin_node_id = node_id;
        is_origin = true;
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
            LOG_DEBUG_STREAM("[Node " << node_id << "] Query " << query_id << " needs transfer, require_id=" << payload.require_id);
            // 需要 transfer，检查是否是起始节点
            if (is_origin)
            {
                LOG_DEBUG_STREAM("[Node " << node_id << "] Origin node transferring query " << query_id);
                
                // 先初始化 pending_queries,避免竞态条件
                {
                    std::lock_guard<std::mutex> lk(pending_queries_mu);
                    pending_queries[query_id] = PendingQuery();
                    LOG_DEBUG_STREAM("[Node " << node_id << "] Initialized pending_queries for query " << query_id);
                }
                
                // 起始节点：异步转发后等待结果
                cluster->transfer_to_remote(query_id, query_vector, payload, k);

                // 等待异步结果
                wait_for_async_result(query_id, results, payload);
            }
            else
            {
                LOG_DEBUG_STREAM("[Node " << node_id << "] Intermediate node forwarding query " << query_id);
                // 中间节点：异步转发，不等待
                cluster->transfer_to_remote(query_id, query_vector, payload, k);
                return {};
            }
        }
        else
        {
            num_queries++;
            LOG_DEBUG_STREAM("[Node " << node_id << "] Query " << query_id << " COMPLETED (continue_search=false), sending to origin " << payload.origin_node_id << ", transfer_cnt=" << payload.remote_transfer_cnt);
            std::vector<int> topk_results = payload.topk_ids(k);
            send_result_to_origin(query_id, topk_results, payload);
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
    return payload.topk_ids(k);
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

            // Wait for ACK
            char ack = 0;
            cluster->transport->recv_raw(sock, &ack, 1);
            // 远程节点在发送ACK后会关闭连接，所以这里标记为无效
            cluster->transport->release_connection(node_id, sock, false);
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
        if (local_hnsw == nullptr && cluster->is_reduce)
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
