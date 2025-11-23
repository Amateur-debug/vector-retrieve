
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
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "hnsw.h"
#include "payload.h"
#include "distance.h"

template <typename T>
class Node;

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

    // 统计远程传输
    std::unordered_map<int, int> remote_transfer_cnt;
    std::unordered_map<int, double> remote_transfer_latency;

    Cluster(HNSWGraph<T> *hnsw, int node_num = 4)
        : hnsw(hnsw), node_num(node_num)
    {
        for (int i = 0; i < node_num; i++)
        {
            nodes.emplace_back(Node<T>(i, this, nullptr));
        }
    }

    void reset()
    {
        remote_transfer_cnt.clear();
        remote_transfer_latency.clear();
        for (int i = 0; i < node_num; i++)
        {
            nodes[i].reset();
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
    }

    // 模拟跨节点传输：统计并将查询转发到目标节点
    std::vector<int> transfer_to_remote(int query_id, const vector<T> &query_vector, Payload<T> &payload, int k)
    {
        remote_transfer_cnt[query_id] += 1;
        double latency_ms = double(payload.size_bits()) / (bandwidth * 1e3); // ms
        remote_transfer_latency[query_id] += latency_ms;

        int require_id = payload.require_id;
        int target_node_id = vector_node_map[require_id];

        Node<T> &target_node = nodes[target_node_id];

        return target_node.query(query_id, query_vector, payload, k);
    }
};

template <typename T>
class Node
{
public:
    int node_id;
    Cluster<T> *cluster;
    HNSWGraph<T> *local_hnsw; // not owned
    uint64_t num_queries = 0;

    Node(int node_id, Cluster<T> *cluster, HNSWGraph<T> *local_hnsw = nullptr)
        : node_id(node_id), cluster(cluster), local_hnsw(local_hnsw)
    {
    }

    void reset()
    {
        num_queries = 0;
    }

    // 执行本地查询；payload 通过引用传入
    std::vector<int> query(int query_id, const vector<T> &query_vector, Payload<T> &payload, int k = 10)
    {
        std::vector<int> results;

        results = local_hnsw->query(query_vector, payload, k);

        if (payload.continue_search)
        {
            results = cluster->transfer_to_remote(query_id, query_vector, payload, k);
        }
        else
        {
            num_queries++;
        }

        return payload.topk_ids(k);
    }
};
