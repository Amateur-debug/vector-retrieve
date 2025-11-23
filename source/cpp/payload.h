#pragma once

#include <unordered_set>
#include <queue>
#include <vector>
#include <utility>
#include <functional>
#include <cstdint>

#include "visited.h"

using namespace std;

template <typename T>
struct Payload {
    bool continue_search = false;   // 是否是跳转来的继续搜索
    int require_id = -1;            // 下一步需要的节点/向量ID 
    int start_level = -1;           // 继续跳转时的起始层级
    int entry_point_id = -1;        // 继续跳转时的入口点 

    Visited visited;

    // candidates as min-heap: (distance, id) with smallest distance on top
    priority_queue<pair<T,int>, vector<pair<T,int>>, greater<pair<T,int>>> candidates;
    
    // results as max-heap: (distance, id) with largest distance on top
    priority_queue<pair<T,int>> results;

    void clear() {
        // clear 
        visited.clear();
        candidates = decltype(candidates)();
        results = decltype(results)();
    }

    // 估算当前 Payload 的大小 (bits)，用于计算网络延迟
    uint64_t size_bits() const {

        uint64_t visited_size = visited.size() * sizeof(int) * 8;

        uint64_t candidates_size = candidates.size() * (sizeof(T) + sizeof(int)) * 8;;

        uint64_t results_size = results.size() * (sizeof(T) + sizeof(int)) * 8;

        uint64_t overhead = 256;

        return visited_size + candidates_size + results_size + overhead;
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