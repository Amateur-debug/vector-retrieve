#pragma once

#include <vector>
#include <cstdint>
#include <algorithm>

class Visited {
public:
    Visited() : cur_epoch(1), visited_count(0) {}

    explicit Visited(std::size_t n) : cur_epoch(1), visited_count(0) {
        ensure_capacity(n);
    }

    // 确保容量
    void ensure_capacity(std::size_t n) {
        if (visited_epoch.size() < n) visited_epoch.resize(n, 0);
    }

    // 插入（标记），返回是否为首次插入（true 表示此前未被标记）
    bool insert(int id) {
        if (id < 0) return false;
        std::size_t uid = static_cast<std::size_t>(id);
        if (uid >= visited_epoch.size()) visited_epoch.resize(uid + 1, 0);
        if (visited_epoch[uid] == cur_epoch) return false;
        visited_epoch[uid] = cur_epoch;
        ++visited_count;
        return true;
    }

    // 查询是否已被标记
    bool find(int id) const {
        if (id < 0) return false;
        std::size_t uid = static_cast<std::size_t>(id);
        return uid < visited_epoch.size() && visited_epoch[uid] == cur_epoch;
    }

    // 快速清空（通过 epoch++），避免逐项清零
    void clear() {
        ++cur_epoch;
        visited_count = 0;
        if (cur_epoch == 0) { // 防止溢出后恢复
            std::fill(visited_epoch.begin(), visited_epoch.end(), 0);
            cur_epoch = 1;
        }
    }

    std::vector<int> to_vector() const {
        std::vector<int> vec;
        vec.reserve(visited_count);
        for (size_t i = 0; i < visited_epoch.size(); ++i) {
            if (visited_epoch[i] == cur_epoch) {
                vec.push_back(static_cast<int>(i));
            }
        }
        return vec;
    }

    std::size_t size() const { return visited_count; }

    std::uint64_t capacity_bytes() const {
        return static_cast<std::uint64_t>(visited_epoch.capacity()) * sizeof(std::uint32_t);
    }

private:
    std::vector<std::uint32_t> visited_epoch;
    std::uint32_t cur_epoch;
    std::size_t visited_count;
};