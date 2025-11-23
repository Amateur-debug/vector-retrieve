#pragma once

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <queue>
#include <random>
#include <string>
#include <sys/file.h>
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

    // 添加向量
    void add_vector(int vid, const std::vector<T> &vec)
    {
        int level = random_level();
        vectors[vid] = vec;
        for (int i = 0; i <= level; i++)
        {
            neighbors[i][vid];
        }

        if (entry_point_id == -1)
        {
            entry_point_id = vid;
            max_level = level;
            return;
        }

        int ep = entry_point_id;
        // 从最高层向下搜索
        for (int i = max_level; i >= 0; i--)
        {
            Payload<T> payload;
            payload.entry_point_id = ep;

            // 使用 ef_construction 搜索候选
            search_layer_ef(vec, i, ef_construction, payload);

            // payload.results 是 priority_queue<pair<T,int>>
            const auto &topk = payload.topk_ids(1);
            ep = topk[0];

            // 使用启发式算法从候选者中挑选邻居
            std::unordered_set<int> selected_neighbors = select_neighbors_heuristic(payload.results);
            neighbors[i][vid] = selected_neighbors;

            // 为被选中的邻居建立反向连接
            for (int neighbor_id : selected_neighbors)
            {
                auto &neighbor_connections = neighbors[i][neighbor_id];
                // 如果邻居的连接数超限
                if ((int)neighbor_connections.size() >= m)
                {
                    // 将超出的邻居的现有连接和新节点一起放入候选池
                    std::priority_queue<std::pair<T, int>> new_candidates_for_neighbor;
                    for (int nid : neighbor_connections)
                    {
                        T dist = distance_id(neighbor_id, nid);
                        std::pair<T, int> item = {dist, nid};
                        new_candidates_for_neighbor.push(item);
                    }
                    std::pair<T, int> item = {distance_id(neighbor_id, vid), vid};
                    new_candidates_for_neighbor.push(item);
                    // 重新选择
                    std::unordered_set<int> new_neighbors = select_neighbors_heuristic(new_candidates_for_neighbor);
                    neighbors[i][neighbor_id] = move(new_neighbors);
                }
                else
                {
                    neighbors[i][neighbor_id].insert(vid);
                }
            }
        }

        // 更新入口点和最大层级
        if (level > max_level)
        {
            max_level = level;
            entry_point_id = vid;
        }
    }

    // query 接口：返回 k 个最接近点的 id 集合
    std::vector<int> query(const std::vector<T> &query_vector, Payload<T> &payload, int k)
    {
        // 如果不是继续搜索，初始化起始层与入口点
        if (!payload.continue_search)
        {
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

    // 搜索单层
    void search_layer_ef(const std::vector<T> &query_vector, int level, int ef, Payload<T> &payload)
    {
        int entry_point_id = payload.entry_point_id;
        payload.entry_point_id = -1;

        if (!payload.continue_search)
        {
            if (vectors.find(entry_point_id) == vectors.end())
            {
                payload.continue_search = true;
                payload.require_id = entry_point_id;
                payload.start_level = level;
                payload.entry_point_id = entry_point_id;
                return;
            }

            T dist = distance(query_vector, vectors[entry_point_id]);
            payload.visited.insert(entry_point_id);
            payload.candidates.push({dist, entry_point_id});
            payload.results.push({dist, entry_point_id});
        }
        else
        {
            if (entry_point_id != -1)
            {
                T dist = distance(query_vector, vectors[entry_point_id]);
                payload.visited.insert(entry_point_id);
                payload.candidates.push({dist, entry_point_id});
                payload.results.push({dist, entry_point_id});
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

            // 遍历候选节点的邻居
            for (int neighbor_id : neighbors[level][candidate_id])
            {
                worst_dist = payload.results.top().first;

                if (payload.visited.find(neighbor_id))
                    continue;

                if (vectors.find(neighbor_id) == vectors.end())
                {
                    payload.continue_search = true;
                    payload.require_id = neighbor_id;
                    payload.start_level = level;
                    payload.candidates.push({candidate_dist, candidate_id});
                    return;
                }

                payload.visited.insert(neighbor_id);
                T neighbor_dist = distance(query_vector, vectors[neighbor_id]);

                // 决定是否把该邻居加入 results：如果 results 小于 ef，直接加入；
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

    // 计算层级
    int random_level()
    {
        double r = uniform_(rng);
        if (r == 0.0)
            r = std::numeric_limits<double>::min();

        int level = static_cast<int>(-log(r) * (1.0 / log(static_cast<double>(m))));

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
        auto it1 = vectors.find(id1);
        auto it2 = vectors.find(id2);

        return distance(it1->second, it2->second);
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
    static HNSWGraph load(const string &filepath)
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

        HNSWGraph<T> g(dim, m, efc, efs, lim);

        g.max_level = j.value("max_level", 0);
        g.entry_point_id = j.value("entry_point_id", -1);

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

                    g.neighbors[level][nid] = std::move(s);
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

                g.vectors[vid] = std::move(vec);
            }
        }

        return g;
    }
};