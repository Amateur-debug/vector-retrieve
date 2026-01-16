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
#include <sys/socket.h>

#include <cstring>
#include <thread>
#include <atomic>
#include <future>

#include "distance.h"
#include "hnsw.h"
#include "log.h"
#include "payload.h"

using namespace std;

template <typename T>
class Cluster; // Forward declaration

enum MessageType
{
    MSG_SEARCH = 0,
    MSG_INIT_GRAPH = 1,
    MSG_ADD_VECTOR = 2,
    MSG_QUERY = 3,
    MSG_SEARCH_RESULT = 4  // 异步查询结果回传
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

    std::vector<int> visited_vec = payload.visited.to_vector();
    buf.write_vec(visited_vec);

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

    std::vector<int> visited_vec = buf.read_vec<int>();
    payload.visited.clear();
    for (int v : visited_vec)
        payload.visited.insert(v);

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
    buf.write_val(hnsw->dim);
    buf.write_val(hnsw->m);
    buf.write_val(hnsw->ef_construction);
    buf.write_val(hnsw->ef_search);
    buf.write_val(hnsw->limited_level);
    buf.write_val(hnsw->max_level);
    buf.write_val(hnsw->entry_point_id);
    buf.write_val(hnsw->seed);

    // vectors
    buf.write_val((uint64_t)hnsw->vectors.size());
    LOG_DEBUG_STREAM("Serializing HNSW: " << hnsw->vectors.size() << " vectors, " << hnsw->neighbors.size() << " levels.");
    for (const auto &kv : hnsw->vectors)
    {
        buf.write_val(kv.first);  // id
        buf.write_vec(kv.second); // vector
    }

    // neighbors
    buf.write_val((uint64_t)hnsw->neighbors.size()); // num levels
    for (const auto &level_kv : hnsw->neighbors)
    {
        buf.write_val(level_kv.first); // level
        const auto &node_map = level_kv.second;
        buf.write_val((uint64_t)node_map.size()); // num nodes in this level
        for (const auto &node_kv : node_map)
        {
            buf.write_val(node_kv.first); // node id
            const auto &nbrs = node_kv.second;
            std::vector<int> nbr_vec(nbrs.begin(), nbrs.end());
            buf.write_vec(nbr_vec);
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

    HNSWGraph<T> *hnsw = new HNSWGraph<T>(dim, m, ef_c, ef_s, lim, seed);
    hnsw->max_level = max_l;
    hnsw->entry_point_id = ep;

    uint64_t vec_size = buf.read_val<uint64_t>();
    for (uint64_t i = 0; i < vec_size; ++i)
    {
        int id = buf.read_val<int>();
        hnsw->vectors[id] = buf.read_vec<T>();
    }

    uint64_t levels = buf.read_val<uint64_t>();
    for (uint64_t i = 0; i < levels; ++i)
    {
        int level = buf.read_val<int>();
        uint64_t nodes = buf.read_val<uint64_t>();
        for (uint64_t j = 0; j < nodes; ++j)
        {
            int nid = buf.read_val<int>();
            std::vector<int> nbrs = buf.read_vec<int>();
            hnsw->neighbors[level][nid] = std::unordered_set<int>(nbrs.begin(), nbrs.end());
        }
    }
    return hnsw;
}

void serialize_routing(Buffer &buf, const std::unordered_map<int, int> &map)
{
    buf.write_val((uint64_t)map.size());
    for (const auto &kv : map)
    {
        buf.write_val(kv.first);
        buf.write_val(kv.second);
    }
}

void deserialize_routing(Buffer &buf, std::unordered_map<int, int> &map)
{
    uint64_t size = buf.read_val<uint64_t>();
    map.clear();
    for (uint64_t i = 0; i < size; ++i)
    {
        int k = buf.read_val<int>();
        int v = buf.read_val<int>();
        map[k] = v;
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

    unsigned int num_worker = 0;
    std::atomic<int> active_handlers{0};  // 追踪活跃的 handle_client 线程数

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

    Node(int node_id, Cluster<T> *cluster, HNSWGraph<T> *local_hnsw = nullptr, bool is_remote_node = false, unsigned int num_worker = 1)
        : node_id(node_id), cluster(cluster), local_hnsw(local_hnsw), is_remote_node(is_remote_node), num_worker(num_worker)
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
          port(other.port), server_thread(std::move(other.server_thread)), stop_server(other.stop_server.load()),
          num_worker(other.num_worker)
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

    // 向量 -> 主节点路由
    std::unordered_map<int, int> vector_node_map;

    // 模拟网络带宽 Mb/s
    double bandwidth = 1000.0;

    // 分区模式
    std::string partition_mode;
    int outer_ring_max_iterations = 0;

    int redundancy_level = 0;
    uint64_t redundancy_num = 0;

    bool is_distributed = false;
    bool is_reduce = false;

    int add_round_robin_counter = 0;

    Cluster(HNSWGraph<T> *hnsw, int node_num = 4)
        : hnsw(hnsw), node_num(node_num)
    {
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
            if (nodes[i].sockfd != -1)
            {
                shutdown(nodes[i].sockfd, SHUT_RDWR);
                close(nodes[i].sockfd);
                nodes[i].sockfd = -1;
            }
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
    static std::vector<int> kmeans(const std::unordered_map<int, vector<T>> &data,
                                   int k, int max_iters = 100, int seed = 123)
    {
        int n = (int)data.size();
        std::vector<int> labels(n, 0);
        if (n == 0 || k <= 0)
            return labels;

        int dim = (int)data.at(0).size();
        std::mt19937 rng(seed);

        // 初始化：随机选 k 个质心
        std::vector<vector<T>> centers;
        centers.reserve(k);
        std::unordered_set<int> chosen;
        std::uniform_int_distribution<int> uid(0, n);
        while ((int)centers.size() < k)
        {
            int idx = uid(rng);
            if (chosen.insert(idx).second)
                centers.push_back(data.at(idx));
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
                    T dist = distance(data.at(i), centers[j]);

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
                    sums[current_id][j] += data.at(i).at(j);
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
                    centers[j] = data.at(uid(rng));
                }
            }
        }
        return labels;
    }

    int outer_ring(vector<unordered_map<int, vector<T>>> &local_vectors)
    {
        // 1. 找出所有有跨节点邻居的候选 boundary vectors（及其跨节点邻居数）
        unordered_map<int, vector<pair<int, int>>> boundary_candidates; // node -> [(vid, cross_cnt)]
        for (int node_id = 0; node_id < node_num; node_id++)
        {
            for (const auto &kv : local_vectors[node_id])
            {
                int vid = kv.first;
                int cross_cnt = 0;
                const auto &neighbor_set = hnsw->neighbors[0][vid];
                for (int nid : neighbor_set)
                {
                    if (vector_node_map[nid] != node_id)
                    {
                        cross_cnt++;
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
            const auto &neighbor_set = hnsw->neighbors[0][vid];
            for (int nid : neighbor_set)
            {
                int neighbor_node = vector_node_map[nid];
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
            local_vectors[best_node][vid] = hnsw->vectors[vid];
            vector_node_map[vid] = best_node;
        }

        return (int)removed.size();
    }

    void partition(const string &mode, int redundancy = 1)
    {
        partition_mode = mode;
        redundancy_level = redundancy;
        redundancy_num = 0;

        int max_level = hnsw->max_level;
        auto &vectors = hnsw->vectors;
        auto &neighbors = hnsw->neighbors;

        // local_vectors: node_id -> map<vid, vector<T>>
        vector<unordered_map<int, vector<T>>> local_vectors(node_num);

        if (mode == "round_robin")
        {
            for (const auto &kv : vectors)
            {
                int vid = kv.first;
                int target = vid % node_num;
                vector_node_map[vid] = target;
                local_vectors[target][vid] = kv.second;
            }
        }
        else if (mode == "k-means")
        {
            auto labels = kmeans(hnsw->vectors, node_num, 100, hnsw->seed);
            for (size_t vid = 0; vid < hnsw->vectors.size(); vid++)
            {
                int target = labels[vid];
                vector_node_map[vid] = target;
                local_vectors[target][vid] = vectors[vid];
            }
        }
        else if (mode == "outer_ring")
        {
            // 先用 k-means 分配
            auto labels = kmeans(hnsw->vectors, node_num, 100, hnsw->seed);
            for (size_t vid = 0; vid < hnsw->vectors.size(); vid++)
            {
                int target = labels[vid];
                vector_node_map[vid] = target;
                local_vectors[target][vid] = vectors[vid];
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

        // redundancy: 将邻居复制到本地
        if (redundancy > 0)
        {
            for (int r = 0; r < redundancy; r++)
            {
                // 收集更新，避免遍历时修改
                vector<unordered_map<int, vector<T>>> updates(node_num);
                for (const auto &level_kv : neighbors)
                {
                    // level_kv: level -> map<vid, vector<int>>
                    for (const auto &neighbor_kv : level_kv.second)
                    {
                        int vid = neighbor_kv.first;
                        const auto &neighbor_set = neighbor_kv.second;
                        for (int node_id = 0; node_id < node_num; node_id++)
                        {
                            if (local_vectors[node_id].count(vid))
                            {
                                for (int neighbor_id : neighbor_set)
                                {
                                    if (!local_vectors[node_id].count(neighbor_id))
                                    {
                                        if (!updates[node_id].count(neighbor_id))
                                        {
                                            updates[node_id][neighbor_id] = vectors[neighbor_id];
                                            redundancy_num++;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                for (int node_id = 0; node_id < node_num; node_id++)
                {
                    for (const auto &kv : updates[node_id])
                        local_vectors[node_id][kv.first] = kv.second;
                }
            }
        }

        // 为每个 node 创建 local_hnsw（浅拷贝/新实例），并设置 vectors/neighbors/entrypoint/max_level
        for (int i = 0; i < node_num; i++)
        {
            HNSWGraph<T> *local_hnsw = new HNSWGraph<T>(hnsw->dim, hnsw->m, hnsw->ef_construction, hnsw->ef_search, hnsw->limited_level, hnsw->seed);
            local_hnsw->max_level = max_level;
            local_hnsw->vectors = local_vectors[i];
            local_hnsw->neighbors = hnsw->neighbors;
            local_hnsw->entry_point_id = hnsw->entry_point_id;
            nodes[i].local_hnsw = local_hnsw;
        }

        // Distribute partitions to remote nodes
        for (int i = 0; i < node_num; i++)
        {
            Node<T> &target_node = nodes[i];
            if (target_node.is_remote_node)
            {

                int sock = socket(AF_INET, SOCK_STREAM, 0);
                if (sock < 0)
                {
                    perror("socket");
                    continue;
                }

                struct sockaddr_in serv_addr;
                memset(&serv_addr, 0, sizeof(serv_addr));
                serv_addr.sin_family = AF_INET;
                serv_addr.sin_port = htons(target_node.port);
                if (inet_pton(AF_INET, target_node.ip_addr.c_str(), &serv_addr.sin_addr) <= 0)
                {
                    LOG_DEBUG_STREAM("Invalid address: " << target_node.ip_addr);
                    close(sock);
                    continue;
                }

                if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0)
                {
                    perror("connect");
                    close(sock);
                    continue;
                }

                Buffer buf;
                buf.write_val((int)MSG_INIT_GRAPH);
                serialize_routing(buf, vector_node_map);
                serialize_hnsw(buf, target_node.local_hnsw);

                uint64_t msg_size = buf.data.size();
                send_all(sock, &msg_size, sizeof(uint64_t));
                send_all(sock, buf.data.data(), msg_size);

                // Wait for ACK from worker node
                char ack = 0;
                try
                {
                    recv_all(sock, &ack, 1);
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
                close(sock);
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
                        int sock = socket(AF_INET, SOCK_STREAM, 0);
                        if (sock < 0)
                        {
                            out.payload = std::move(node_payload);
                            return out;
                        }
                        struct sockaddr_in serv_addr;
                        memset(&serv_addr, 0, sizeof(serv_addr));
                        serv_addr.sin_family = AF_INET;
                        serv_addr.sin_port = htons(node.port);
                        if (inet_pton(AF_INET, node.ip_addr.c_str(), &serv_addr.sin_addr) <= 0)
                        {
                            close(sock);
                            out.payload = std::move(node_payload);
                            return out;
                        }
                        if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0)
                        {
                            close(sock);
                            out.payload = std::move(node_payload);
                            return out;
                        }

                        LOG_DEBUG_STREAM("[Cluster] Sending query " << query_id << " to remote node " << i);

                        Buffer buf;
                        buf.write_val((int)MSG_QUERY);
                        buf.write_val(query_id);
                        buf.write_val(k);
                        buf.write_vec(query_vector);
                        serialize_payload(buf, node_payload);

                        uint64_t msg_size = buf.data.size();
                        try
                        {
                            send_all(sock, &msg_size, sizeof(uint64_t));
                            send_all(sock, buf.data.data(), msg_size);

                            uint64_t resp_size = 0;
                            recv_all(sock, &resp_size, sizeof(uint64_t));

                            std::vector<char> resp_data(resp_size);
                            recv_all(sock, resp_data.data(), resp_size);

                            LOG_DEBUG_STREAM("[Cluster] Received response for query " << query_id << " from remote node " << i);

                            close(sock);

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
                            close(sock);
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
        int target_node_id = vector_node_map[require_id];

        Node<T> &target_node = nodes[target_node_id];

        if (is_distributed && target_node.is_remote_node)
        {
            // 记录发送时间
            auto now = std::chrono::system_clock::now();
            payload.send_timestamp = std::chrono::duration<double, std::milli>(now.time_since_epoch()).count();

            // 创建新连接，发送后关闭（不使用连接池）
            int sock = socket(AF_INET, SOCK_STREAM, 0);
            if (sock < 0)
            {
                LOG_DEBUG_STREAM("[Cluster] ERROR: Failed to create socket for query " << query_id << " to node " << target_node_id << ": " << strerror(errno));
                return {};
            }

            // 设置SO_REUSEADDR允许重用TIME_WAIT状态的端口
            int reuse = 1;
            setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

            struct sockaddr_in serv_addr;
            memset(&serv_addr, 0, sizeof(serv_addr));
            serv_addr.sin_family = AF_INET;
            serv_addr.sin_port = htons(target_node.port);
            if (inet_pton(AF_INET, target_node.ip_addr.c_str(), &serv_addr.sin_addr) <= 0)
            {
                LOG_DEBUG_STREAM("[Cluster] ERROR: Invalid IP address " << target_node.ip_addr << " for node " << target_node_id << " query " << query_id);
                close(sock);
                return {};
            }

            if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0)
            {
                LOG_DEBUG_STREAM("[Cluster] ERROR: Failed to connect to node " << target_node_id << " (" << target_node.ip_addr << ":" << target_node.port << ") for query " << query_id << ": " << strerror(errno));
                close(sock);
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
                send_all(sock, &msg_size, sizeof(uint64_t));
                send_all(sock, buf.data.data(), msg_size);

                // 异步模式：发送后立即关闭连接，不等待响应
                close(sock);
                
                LOG_DEBUG_STREAM("[Cluster] MSG_SEARCH successfully sent to node " << target_node_id << " for query " << query_id);
                
                return {};
            }
            catch (...)
            {
                close(sock);
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
            j["add_round_robin_counter"] = add_round_robin_counter;

            // 保存 vector_node_map
            json jmap = json::object();
            for (const auto &kv : vector_node_map)
            {
                jmap[std::to_string(kv.first)] = kv.second;
            }
            j["vector_node_map"] = std::move(jmap);

            // 保存主 HNSW 图
            string hnsw_path = filepath + ".hnsw.json";
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
                    string local_hnsw_path = filepath + ".node" + std::to_string(i) + ".json";
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

        // 加载 vector_node_map
        if (j.contains("vector_node_map"))
        {
            for (auto const &kv : j["vector_node_map"].items())
            {
                int vid = std::stoi(kv.key());
                int node_id = kv.value().get<int>();
                cluster->vector_node_map[vid] = node_id;
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
    
    // 设置 SO_LINGER 为 0，发送 RST 而不是 FIN，立即关闭连接
    struct linger so_linger;
    so_linger.l_onoff = 1;   // 启用 linger
    so_linger.l_linger = 0;  // 超时时间为 0，立即发送 RST
    setsockopt(fd, SOL_SOCKET, SO_LINGER, &so_linger, sizeof(so_linger));
    
    close(fd);
}

template <typename T>
void Node<T>::start_server()
{
    cluster->is_distributed = true;
    if (is_remote_node)
        return;

    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0)
    {
        perror("socket");
        return;
    }

    int opt = 1;
    setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    memset(&sock_addr, 0, sizeof(sock_addr));
    sock_addr.sin_family = AF_INET;
    sock_addr.sin_addr.s_addr = INADDR_ANY;
    sock_addr.sin_port = htons(port);

    if (bind(sockfd, (struct sockaddr *)&sock_addr, sizeof(sock_addr)) < 0)
    {
        perror("bind");
        close(sockfd);
        sockfd = -1;
        return;
    }

    if (listen(sockfd, 1024) < 0)
    {
        perror("listen");
        close(sockfd);
        sockfd = -1;
        return;
    }

    try
    {
        server_thread = std::thread(&Node<T>::tcp_server_loop, this);
    }
    catch (const std::exception &e)
    {
        LOG_DEBUG_STREAM("Failed to start server thread: " << e.what());
        close(sockfd);
        sockfd = -1;
    }
}

template <typename T>
void Node<T>::tcp_server_loop()
{
    LOG_DEBUG_STREAM("Node " << node_id << " tcp_server_loop started on port " << port);

    while (!stop_server.load())
    {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(sockfd, (struct sockaddr *)&client_addr, &client_len);
        if (client_fd < 0)
        {
            LOG_DEBUG_STREAM("[Node " << node_id << "] Failed to accept");
            if (stop_server.load())
                break;
            continue;
        }

        // 防止高并发时 recv_all/send_all 永久阻塞：为每个连接设置读写超时
        {
            struct timeval tv;
            tv.tv_sec = 2;
            tv.tv_usec = 0;
            setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
            setsockopt(client_fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
        }

        // int buf = 4 * 1024 * 1024;
        // setsockopt(client_fd, SOL_SOCKET, SO_SNDBUF, &buf, sizeof(buf));
        // setsockopt(client_fd, SOL_SOCKET, SO_RCVBUF, &buf, sizeof(buf));

        // 限制并发线程数,避免线程爆炸
        if (num_worker > 0)
        {
            while (active_handlers.load() >= (int)num_worker)
            {
                LOG_DEBUG_STREAM("[Node " << node_id << "] waiting for idle threads");
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        
        std::thread(&Node<T>::handle_client, this, client_fd).detach();
    }
}

template <typename T>
void Node<T>::handle_client(int client_fd)
{
    active_handlers++;
    int current_active = active_handlers.load();
    if (current_active % 100 == 0 || current_active > 500)
    {
        LOG_DEBUG_STREAM("[Node " << node_id << "] Active handlers: " << current_active);
    }
    
    try
    {
        uint64_t msg_size = 0;
        recv_all(client_fd, &msg_size, sizeof(uint64_t));

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

            close_connection_hard(client_fd);
        }
        else if (msg_type == MSG_INIT_GRAPH)
        {
            LOG_DEBUG_STREAM("Node " << node_id << " receiving graph partition...");
            deserialize_routing(buf, cluster->vector_node_map);

            if (local_hnsw)
                delete local_hnsw;
            local_hnsw = deserialize_hnsw<T>(buf);

            LOG_DEBUG_STREAM("Node " << node_id << " initialized with " << local_hnsw->vectors.size() << " vectors.");

            if (buf.read_pos != buf.data.size())
            {
                LOG_DEBUG_STREAM("Warning: Buffer not fully consumed! read_pos=" << buf.read_pos << " size=" << buf.data.size());
            }
            else
            {
                LOG_DEBUG_STREAM("Buffer fully consumed.");
            }

            // Send ACK back to master
            char ack = 1;
            send_all(client_fd, &ack, 1);
            
            // MSG_INIT_GRAPH 是一次性操作，关闭连接
            close_connection_hard(client_fd);
        }
        else if (msg_type == MSG_ADD_VECTOR)
        {
            int vid = buf.read_val<int>();
            std::vector<T> vec = buf.read_vec<T>();

            if (local_hnsw == nullptr)
            {
                local_hnsw = new HNSWGraph<T>(
                    cluster->hnsw->dim, cluster->hnsw->m, cluster->hnsw->ef_construction,
                    cluster->hnsw->ef_search, cluster->hnsw->limited_level, cluster->hnsw->seed);
            }

            local_hnsw->add_vector(vid, vec);

            // Send ACK back
            char ack = 1;
            send_all(client_fd, &ack, 1);
            
            // MSG_ADD_VECTOR 是一次性操作，关闭连接
            close_connection_hard(client_fd);
        }
        else if (msg_type == MSG_QUERY)
        {
            int query_id = buf.read_val<int>();
            int k = buf.read_val<int>();
            std::vector<T> query_vector = buf.read_vec<T>();
            
            Payload<T> payload;
            deserialize_payload(buf, payload);

            // 本地查询，不跨节点

            std::vector<pair<T, int>> results;
            if (local_hnsw)
            {
                payload.continue_search = false;
                local_hnsw->query(query_vector, payload, k);

                // 从payload.results中获取带距离的结果
                auto pq = payload.results;
                while (!pq.empty())
                {
                    results.push_back(pq.top());
                    pq.pop();
                }  
            }
            LOG_DEBUG_STREAM("[Node " << node_id << "] MSG_QUERY: results.size=" << results.size());

            // 发送结果回主节点
            Buffer resp_buf;
            resp_buf.write_val((int)results.size());
            for (const auto &p : results)
            {
                resp_buf.write_val(p.first);  // distance
                resp_buf.write_val(p.second); // id
            }
            serialize_payload(resp_buf, payload);

            uint64_t resp_size = resp_buf.data.size();
            send_all(client_fd, &resp_size, sizeof(uint64_t));
            send_all(client_fd, resp_buf.data.data(), resp_size);
            // MSG_QUERY 需要可靠回包，使用正常 FIN 关闭，避免 RST 导致对端 recv_all 卡死
            close_connection_soft(client_fd);
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
            
            // MSG_SEARCH_RESULT 不需要回复，直接关闭连接
            close_connection_hard(client_fd);
        }
    }
    catch (const std::exception &e)
    {
        LOG_DEBUG_STREAM("Error handling client: " << e.what());
        close_connection_hard(client_fd);
    }
    catch (...)
    {
        close_connection_hard(client_fd);
    }
    
    active_handlers--;
    return;
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
    
    // 创建新连接发送结果
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0)
    {
        LOG_DEBUG_STREAM("[Node " << node_id << "] Failed to create socket to origin node");
        return;
    }

    // 设置SO_REUSEADDR允许重用TIME_WAIT状态的端口
    int reuse = 1;
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

    struct sockaddr_in serv_addr;
    memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(origin_node.port);
    if (inet_pton(AF_INET, origin_node.ip_addr.c_str(), &serv_addr.sin_addr) <= 0)
    {
        LOG_DEBUG_STREAM("[Node " << node_id << "] Invalid address for origin node");
        close(sock);
        return;
    }

    // 重试连接,防止高并发时连接失败
    int max_retries = 5;
    int retry_delay_ms = 10;
    bool connected = false;
    
    for (int retry = 0; retry < max_retries; retry++)
    {
        if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) == 0)
        {
            connected = true;
            break;
        }
        
        if (retry < max_retries - 1)
        {
            LOG_DEBUG_STREAM("[Node " << node_id << "] Failed to connect to origin node (attempt " << (retry + 1) << "/" << max_retries << "), retrying...");
            std::this_thread::sleep_for(std::chrono::milliseconds(retry_delay_ms));
            retry_delay_ms *= 2;  // 指数退避
        }
    }
    
    if (!connected)
    {
        LOG_DEBUG_STREAM("[Node " << node_id << "] ERROR: Failed to connect to origin node after " << max_retries << " attempts for query " << query_id);
        close(sock);
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
        send_all(sock, &msg_size, sizeof(uint64_t));
        send_all(sock, buf.data.data(), msg_size);
        
        close(sock);
        LOG_DEBUG_STREAM("[Node " << node_id << "] Result sent to origin node " << payload.origin_node_id);
    }
    catch (...)
    {
        LOG_DEBUG_STREAM("[Node " << node_id << "] Exception sending result to origin node");
        close(sock);
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
            return cluster->transfer_to_remote(query_id, query_vector, payload, k);;
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
        // 通过网络发送到远程节点
        int sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock < 0)
        {
            LOG_DEBUG_STREAM("Failed to create socket for add_vector");
            return;
        }

        struct sockaddr_in serv_addr;
        memset(&serv_addr, 0, sizeof(serv_addr));
        serv_addr.sin_family = AF_INET;
        serv_addr.sin_port = htons(port);
        if (inet_pton(AF_INET, ip_addr.c_str(), &serv_addr.sin_addr) <= 0)
        {
            LOG_DEBUG_STREAM("Invalid address for add_vector: " << ip_addr);
            close(sock);
            return;
        }

        if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0)
        {
            LOG_DEBUG_STREAM("Failed to connect for add_vector");
            close(sock);
            return;
        }

        Buffer buf;
        buf.write_val((int)MSG_ADD_VECTOR);
        buf.write_val(vid);
        buf.write_vec(vec);

        uint64_t msg_size = buf.data.size();
        try
        {
            send_all(sock, &msg_size, sizeof(uint64_t));
            send_all(sock, buf.data.data(), msg_size);

            // Wait for ACK
            char ack = 0;
            recv_all(sock, &ack, 1);
            close(sock);
        }
        catch (const std::exception &e)
        {
            LOG_DEBUG_STREAM("Error sending add_vector: " << e.what());
            close(sock);
        }
    }
    else
    {
        // 确保 local_hnsw 已初始化（reduce模式下按需初始化）
        if (local_hnsw == nullptr && cluster->is_reduce)
        {
            local_hnsw = new HNSWGraph<T>(
                cluster->hnsw->dim, cluster->hnsw->m, cluster->hnsw->ef_construction,
                cluster->hnsw->ef_search, cluster->hnsw->limited_level, cluster->hnsw->seed);
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
