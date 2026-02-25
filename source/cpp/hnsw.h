#pragma once

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <mutex>
#include <queue>
#include <random>
#include <shared_mutex>
#include <string>
#include <sys/file.h>
#include <thread>
#include <type_traits>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "json.hpp"
using json = nlohmann::json;

#include "payload.h"
#include "distance.h"

template <typename T>
class HNSWGraph
{
public:
    int dim;
    int m;
    int ef_construction;
    int ef_search;
    int limited_level;
    int max_level;
    int entry_point_id;
    unsigned int seed;
    // 存向量: id -> std::vector<T>
    std::unordered_map<int, std::vector<T>> vectors;

    // neighbors[level][vid] = set(neighbor_ids)
    std::unordered_map<int, std::unordered_map<int, std::unordered_set<int>>> neighbors;

    // rng
    std::mt19937 rng;
    std::uniform_real_distribution<double> uniform_;

    // 线程安全保护
    mutable std::shared_mutex vectors_mutex;  // 保护 vectors
    mutable std::shared_mutex neighbors_mutex; // 保护 neighbors
    mutable std::mutex entry_mutex;          // 保护 entry_point_id 和 max_level
    mutable std::mutex rng_mutex;            // 保护随机数生成器
    
    // 节点级别的细粒度锁（使用shared_mutex减少竞争）
    mutable std::unordered_map<int, std::unique_ptr<std::mutex>> node_mutexes;
    mutable std::shared_mutex node_mutexes_mutex;  // 改用shared_mutex提高并发

    HNSWGraph(int dim, int m = 16, int ef_construction = 128, int ef_search = 64,
              int limited_level = 5, unsigned int seed = 123)
        : dim(dim),
          m(m),
          ef_construction(ef_construction),
          ef_search(ef_search),
          limited_level(limited_level),
          max_level(0),
          entry_point_id(-1),
          seed(seed),
          rng(seed),
          uniform_(0.0, 1.0)
    {
    }

    // 获取节点级别的锁（使用读写锁减少热点竞争）
    std::mutex* get_node_mutex(int node_id)
    {
        // 先用读锁尝试查找（允许多线程并发读取）
        {
            std::shared_lock<std::shared_mutex> read_lock(node_mutexes_mutex);
            auto it = node_mutexes.find(node_id);
            if (it != node_mutexes.end())
            {
                return it->second.get();
            }
        }
        
        // 需要创建新锁时才用写锁（双重检查锁定模式）
        std::unique_lock<std::shared_mutex> write_lock(node_mutexes_mutex);
        auto it = node_mutexes.find(node_id);
        if (it == node_mutexes.end())
        {
            node_mutexes[node_id] = std::make_unique<std::mutex>();
            return node_mutexes[node_id].get();
        }
        return it->second.get();
    }

    // 添加向量（线程安全，使用细粒度锁）
    void add_vector(int vid, const std::vector<T> &vec)
    {
        int level;
        {
            std::lock_guard<std::mutex> lock(rng_mutex);
            level = random_level();
        }

        // 检查是否是第一个节点
        bool is_first = false;
        {
            std::lock_guard<std::mutex> entry_lock(entry_mutex);
            is_first = (entry_point_id == -1);
        }

        // 添加向量和初始化邻居表（使用细粒度写锁）
        {
            std::unique_lock<std::shared_mutex> lock(vectors_mutex);
            vectors[vid] = vec;
        }
        {
            std::unique_lock<std::shared_mutex> lock(neighbors_mutex);
            for (int i = 0; i <= level; i++)
            {
                neighbors[i][vid];
            }
        }

        if (is_first)
        {
            std::lock_guard<std::mutex> entry_lock(entry_mutex);
            if (entry_point_id == -1)  // 双重检查
            {
                entry_point_id = vid;
                max_level = level;
            }
            return;
        }

        int ep, current_max_level;
        {
            std::lock_guard<std::mutex> entry_lock(entry_mutex);
            ep = entry_point_id;
            current_max_level = max_level;
        }

        // 从最高层向下搜索
        for (int i = current_max_level; i >= 0; i--)
        {
            Payload<T> payload;
            payload.entry_point_id = ep;

            // 高于目标层时仅做 greedy 导航（ef=1），不建边
            int ef = (i > level) ? 1 : ef_construction;
            search_layer_ef(vec, i, ef, payload);

            const auto &topk = payload.topk_ids(1);
            ep = topk[0];

            // 关键修复：节点只应出现在 [0..level]，更高层不插入
            if (i > level)
            {
                continue;
            }

            // 使用启发式算法从候选者中挑选邻居
            std::unordered_set<int> selected_neighbors = select_neighbors_heuristic(payload.results);

            // 使用节点级别的锁更新邻居关系（减少锁竞争）
            {
                std::unique_lock<std::shared_mutex> lock(neighbors_mutex);
                neighbors[i][vid] = selected_neighbors;
            }

            // 为被选中的邻居建立反向连接（批量优化：一次性更新所有邻居）
            // 步骤1: 批量读取所有需要的向量数据（一次加锁）
            std::unordered_map<int, std::vector<T>> all_vecs;
            {
                std::shared_lock<std::shared_mutex> vec_lock(vectors_mutex);
                all_vecs[vid] = vec; // 新插入的向量
                for (int neighbor_id : selected_neighbors)
                {
                    auto it = vectors.find(neighbor_id);
                    if (it != vectors.end())
                    {
                        all_vecs[neighbor_id] = it->second;
                    }
                }
                // 预读所有邻居的邻居向量（用于距离计算）
                for (int neighbor_id : selected_neighbors)
                {
                    auto level_it = neighbors.find(i);
                    if (level_it == neighbors.end())
                    {
                        continue;
                    }
                    auto conn_it = level_it->second.find(neighbor_id);
                    if (conn_it != level_it->second.end())
                    {
                        for (int conn_id : conn_it->second)
                        {
                            if (all_vecs.find(conn_id) == all_vecs.end())
                            {
                                auto it = vectors.find(conn_id);
                                if (it != vectors.end())
                                {
                                    all_vecs[conn_id] = it->second;
                                }
                            }
                        }
                    }
                }
            }

            // 步骤2: 批量读取所有邻居的连接状态（一次加锁）
            std::unordered_map<int, std::unordered_set<int>> all_connections;
            {
                std::shared_lock<std::shared_mutex> read_lock(neighbors_mutex);
                for (int neighbor_id : selected_neighbors)
                {
                    auto level_it = neighbors.find(i);
                    if (level_it == neighbors.end())
                    {
                        continue;
                    }
                    auto it = level_it->second.find(neighbor_id);
                    if (it != level_it->second.end())
                    {
                        all_connections[neighbor_id] = it->second;
                    }
                }
            }

            // 步骤3: 完全无锁计算所有更新决策
            std::unordered_map<int, std::pair<bool, int>> update_decisions;
            for (int neighbor_id : selected_neighbors)
            {
                const auto &neighbor_connections = all_connections[neighbor_id];
                bool should_add = false;
                int to_remove = -1;

                if ((int)neighbor_connections.size() < m)
                {
                    should_add = true;
                }
                else if (all_vecs.find(neighbor_id) != all_vecs.end())
                {
                    const auto &neighbor_vec = all_vecs[neighbor_id];
                    T max_dist = -1;
                    int furthest_id = -1;

                    for (int conn_id : neighbor_connections)
                    {
                        if (all_vecs.find(conn_id) != all_vecs.end())
                        {
                            T dist = distance(neighbor_vec, all_vecs[conn_id]);
                            if (dist > max_dist)
                            {
                                max_dist = dist;
                                furthest_id = conn_id;
                            }
                        }
                    }

                    T new_dist = distance(neighbor_vec, all_vecs[vid]);
                    if (furthest_id != -1 && new_dist < max_dist)
                    {
                        should_add = true;
                        to_remove = furthest_id;
                    }
                }

                update_decisions[neighbor_id] = {should_add, to_remove};
            }

            // 步骤4: 批量更新所有邻居（一次加锁，快速完成）
            {
                std::unique_lock<std::shared_mutex> write_lock(neighbors_mutex);
                for (const auto &kv : update_decisions)
                {
                    int neighbor_id = kv.first;
                    bool should_add = kv.second.first;
                    int to_remove = kv.second.second;

                    if (should_add)
                    {
                        auto &conns = neighbors[i][neighbor_id];
                        if (to_remove != -1)
                        {
                            conns.erase(to_remove);
                        }
                        conns.insert(vid);
                    }
                }
            }
        }

        // 更新入口点和最大层级
        if (level > current_max_level)
        {
            std::lock_guard<std::mutex> entry_lock(entry_mutex);
            if (level > max_level)  // 双重检查
            {
                max_level = level;
                entry_point_id = vid;
            }
        }
    }

    // query 接口：返回 k 个最接近点的 id 集合（线程安全）
    std::vector<int> query(const std::vector<T> &query_vector, Payload<T> &payload, int k)
    {
        // 如果不是继续搜索，初始化起始层与入口点
        if (!payload.continue_search)
        {
            std::lock_guard<std::mutex> entry_lock(entry_mutex);
            payload.start_level = max_level;
            payload.entry_point_id = entry_point_id;
        }

        // 从 start_level 向下搜索到 0
        for (int l = payload.start_level; l >= 0; l--)
        {
            payload.start_level = l;

            // 在单层上用 ef_search 搜索并把结果放到 payload.results
            search_layer_ef(query_vector, l, ef_search, payload);

            if (payload.continue_search)
            {
                return payload.topk_ids(k);
            }

            // 使用当前层的最佳结果作为下一层的入口点
            const auto topk = payload.topk_ids(1);
            payload.entry_point_id = topk[0];

            // 如果不是最低层，清空结果以便下一层重新构建
            if (l != 0)
            {
                payload.clear();
            }
        }

        std::vector<int> final_results;
        final_results = payload.topk_ids(k);

        return final_results;
    }

    // 搜索单层（使用细粒度读锁）
    void search_layer_ef(const std::vector<T> &query_vector, int level, int ef, Payload<T> &payload)
    {
        int entry_point_id = payload.entry_point_id;
        payload.entry_point_id = -1;

        if (!payload.continue_search)
        {
            std::vector<T> entry_vec;
            {
                std::shared_lock<std::shared_mutex> lock(vectors_mutex);
                auto it = vectors.find(entry_point_id);
                if (it == vectors.end())
                {
                    payload.continue_search = true;
                    payload.require_id = entry_point_id;
                    payload.start_level = level;
                    payload.entry_point_id = entry_point_id;
                    return;
                }
                entry_vec = it->second;
            }

            T dist = distance(query_vector, entry_vec);
            payload.distance_cnt++;  // 累加距离计算次数
            payload.visited.insert(entry_point_id);
            payload.candidates.push({dist, entry_point_id});
            payload.results.push({dist, entry_point_id});
        }
        else
        {
            if (entry_point_id != -1)
            {
                std::vector<T> entry_vec;
                {
                    std::shared_lock<std::shared_mutex> lock(vectors_mutex);
                    auto it = vectors.find(entry_point_id);
                    if (it != vectors.end())
                    {
                        entry_vec = it->second;
                    }
                }
                
                if (!entry_vec.empty())
                {
                    T dist = distance(query_vector, entry_vec);
                    payload.distance_cnt++;  // 累加距离计算次数
                    payload.visited.insert(entry_point_id);
                    payload.candidates.push({dist, entry_point_id});
                    payload.results.push({dist, entry_point_id});
                }
            }
        }

        // 主循环：从 candidates 中取最小距离的候选，扩展其邻居
        while (!payload.candidates.empty())
        {
            auto cand = payload.candidates.top();
            payload.candidates.pop();
            T candidate_dist = cand.first;
            int candidate_id = cand.second;

            auto worst_dist = payload.results.top().first;

            // 如果当前候选者比结果中最差的还差，就不用继续了
            if (candidate_dist > worst_dist)
            {
                break;
            }

            // 获取候选节点的邻居列表（快速读取，不长时间持锁）
            std::unordered_set<int> neighbor_ids;
            {
                std::shared_lock<std::shared_mutex> lock(neighbors_mutex);
                auto level_it = neighbors.find(level);
                if (level_it != neighbors.end())
                {
                    auto it = level_it->second.find(candidate_id);
                    if (it != level_it->second.end())
                    {
                        neighbor_ids = it->second;
                    }
                }
            }

            // 遍历候选节点的邻居
            for (int neighbor_id : neighbor_ids)
            {
                worst_dist = payload.results.top().first;

                if (payload.visited.find(neighbor_id))
                    continue;

                std::vector<T> neighbor_vec;
                {
                    std::shared_lock<std::shared_mutex> lock(vectors_mutex);
                    auto it = vectors.find(neighbor_id);
                    if (it == vectors.end())
                    {
                        payload.continue_search = true;
                        payload.require_id = neighbor_id;
                        payload.start_level = level;
                        payload.candidates.push({candidate_dist, candidate_id});
                        return;
                    }
                    neighbor_vec = it->second;
                }

                payload.visited.insert(neighbor_id);
                T neighbor_dist = distance(query_vector, neighbor_vec);
                payload.distance_cnt++;  // 累加距离计算次数

                // 决定是否把该邻居加入 results
                if ((int)payload.results.size() < ef)
                {
                    payload.results.push({neighbor_dist, neighbor_id});
                    payload.candidates.push({neighbor_dist, neighbor_id});
                }
                else if (neighbor_dist < worst_dist)
                {
                    payload.results.pop();
                    payload.results.push({neighbor_dist, neighbor_id});
                    payload.candidates.push({neighbor_dist, neighbor_id});
                }
            }
        }

        payload.continue_search = false;

        return;
    }

    // 计算层级（需要外部加锁调用）
    // 使用对数公式：level = floor(-ln(r) / ln(m))
    // 各层累积数严格以 1/m 指数递减：P(level >= L) = (1/m)^L
    // 例如 m=8, N=1e6: neighbors[0]=1e6, [1]=125000, [2]=15625, [3]=1953, [4]=244, [5]=30
    int random_level()
    {
        // 注意：调用者需要持有 rng_mutex
        double r = uniform_(rng);
        if (r == 0.0)
            r = std::numeric_limits<double>::min();

        int level = static_cast<int>(-log(r) / log(static_cast<double>(m)));

        if (limited_level >= 0)
            level = std::min(level, limited_level);

        return level;
    }

    // 启发式选择邻居，candidates: 按距离降序的 (dist, id)
    std::unordered_set<int> select_neighbors_heuristic(const priority_queue<pair<T, int>> &candidates)
    {
        // 先把堆拷贝出来，再转成 vector 并按距离升序排序
        std::vector<std::pair<T, int>> items;
        priority_queue<pair<T, int>> pq = candidates;
        while (!pq.empty())
        {
            items.push_back(pq.top());
            pq.pop();
        }
        std::sort(items.begin(), items.end(), [](const std::pair<T, int> &a, const std::pair<T, int> &b)
                  {
            if (a.first != b.first) return a.first < b.first;
            return a.second < b.second; });

        std::unordered_set<int> results;
        // 按小到大遍历，按启发式规则添加邻居，最多 m 个
        for (const auto &p : items)
        {
            if (results.size() >= static_cast<size_t>(m))
                break;

            T e_dist = p.first;
            int e_id = p.second;

            bool is_good_neighbor = true;
            for (int r_id : results)
            {
                T er_dist = distance_id(e_id, r_id);
                if (er_dist < e_dist)
                {
                    is_good_neighbor = false;
                    break;
                }
            }

            if (is_good_neighbor)
                results.insert(e_id);
        }

        return results;
    }

    // 计算距离
    T distance_id(int id1, int id2) const
    {
        std::vector<T> vec1, vec2;
        {
            std::shared_lock<std::shared_mutex> lock(vectors_mutex);
            auto it1 = vectors.find(id1);
            auto it2 = vectors.find(id2);
            if (it1 != vectors.end() && it2 != vectors.end())
            {
                vec1 = it1->second;
                vec2 = it2->second;
            }
        }

        if (vec1.empty() || vec2.empty())
            return std::numeric_limits<T>::max();

        return distance(vec1, vec2);
    }

    bool save(const string &filepath)
    {
        try
        {
            json j;
            j["dim"] = dim;
            j["m"] = m;
            j["ef_construction"] = ef_construction;
            j["ef_search"] = ef_search;
            j["limited_level"] = limited_level;
            j["max_level"] = max_level;
            j["entry_point_id"] = entry_point_id;

            // neighbors[level][vid] = set(neighbor_ids)
            json jneighbors = json::object();
            for (const auto &level_kv : neighbors)
            {
                json jlevel = json::object();
                for (const auto &neighbor_kv : level_kv.second)
                {
                    json nid_array = json::array();
                    for (int nid : neighbor_kv.second) {
                        nid_array.push_back(nid);
                    }
                    jlevel[std::to_string(neighbor_kv.first)] = std::move(nid_array);
                }
                jneighbors[std::to_string(level_kv.first)] = std::move(jlevel);
            }
            j["neighbors"] = std::move(jneighbors);

            // vectors: id -> std::vector<T>
            json jvector = json::object();
            for (const auto &kv : vectors)
            {
                json vector_array = json::array();
                for (const T &v : kv.second)
                {
                    vector_array.push_back(v);
                }
                jvector[std::to_string(kv.first)] = std::move(vector_array);
            }
            j["vectors"] = std::move(jvector);

            // 另外把 seed 一并写入
            j["seed"] = seed;

            std::string out = j.dump(4);
            int fd = ::open(filepath.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
            if (fd == -1)
                return false;
            // 非阻塞独占锁，若失败说明被其他进程锁定，跳过写入
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

            ::fsync(fd);        // 确保数据刷盘
            flock(fd, LOCK_UN); // 释放锁
            ::close(fd);
            return true;
        }
        catch (...)
        {
            return false;
        }
    }

    // 从 JSON 加载全部内容
    static HNSWGraph* load(const string &filepath)
    {
        std::ifstream ifs(filepath, std::ios::binary);
        if (!ifs)
            throw std::runtime_error("failed to open json: " + filepath);
        json j;
        ifs >> j;

        int dim = j.value("dim", 0);
        int m = j.value("m", 16);
        int efc = j.value("ef_construction", 128);
        int efs = j.value("ef_search", 64);
        int lim = j.value("limited_level", 5);

        HNSWGraph<T>* g = new HNSWGraph<T>(dim, m, efc, efs, lim);

        g->max_level = j.value("max_level", 0);
        g->entry_point_id = j.value("entry_point_id", -1);

        // neighbors
        if (j.contains("neighbors"))
        {
            for (auto const &level_pair : j["neighbors"].items())
            {
                const std::string &level_str = level_pair.key();
                const json &jlevel = level_pair.value();
                int level = std::stoi(level_str);

                for (auto const &neighbor_kv : jlevel.items())
                {
                    const std::string &nid_str = neighbor_kv.key();
                    const json &nid_array = neighbor_kv.value();
                    int nid = std::stoi(nid_str);

                    std::unordered_set<int> s;
                    s.reserve(nid_array.size());
                    for (const auto &v : nid_array)
                        s.insert(v.get<int>());

                    g->neighbors[level][nid] = std::move(s);
                }
            }
        }

        // vectors
        if (j.contains("vectors"))
        {
            for (auto const &vec_pair : j["vectors"].items())
            {
                const std::string &vid_str = vec_pair.key();
                const json &vec_arr = vec_pair.value();
                int vid = std::stoi(vid_str);

                std::vector<T> vec;
                vec.reserve(vec_arr.size());
                for (const auto &v : vec_arr)
                    vec.push_back(v.get<T>());

                g->vectors[vid] = std::move(vec);
            }
        }

        return g;
    }
};