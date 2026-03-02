/**
 * @file visited.h
 * @brief 已访问节点集合实现
 * 
 * 使用epoch机制实现O(1)清空，避免逐项清零
 */

#pragma once

#include <vector>
#include <cstdint>
#include <algorithm>

/**
 * @brief 已访问节点集合
 * 
 * 通过epoch计数器实现快速清空，适合多次搜索复用
 */
class Visited {
public:
    Visited() : cur_epoch_(1), visited_count_(0) {}

    explicit Visited(std::size_t n) : cur_epoch_(1), visited_count_(0) {
        ensure_capacity(n);
    }

    // 确保容量足够
    void ensure_capacity(std::size_t n) {
        if (visited_epoch_.size() < n) visited_epoch_.resize(n, 0);
    }

    // 标记节点为已访问，返回是否首次标记
    bool insert(int id) {
        if (id < 0) return false;
        std::size_t uid = static_cast<std::size_t>(id);
        if (uid >= visited_epoch_.size()) visited_epoch_.resize(uid + 1, 0);
        if (visited_epoch_[uid] == cur_epoch_) return false;
        visited_epoch_[uid] = cur_epoch_;
        ++visited_count_;
        return true;
    }

    // 检查节点是否已访问
    bool find(int id) const {
        if (id < 0) return false;
        std::size_t uid = static_cast<std::size_t>(id);
        return uid < visited_epoch_.size() && visited_epoch_[uid] == cur_epoch_;
    }

    // 快速清空(epoch++)，无需逐项清零
    void clear() {
        ++cur_epoch_;
        visited_count_ = 0;
        // 防止epoch溢出后回绕
        if (cur_epoch_ == 0) {
            std::fill(visited_epoch_.begin(), visited_epoch_.end(), 0);
            cur_epoch_ = 1;
        }
    }

    // 转换为vector(用于序列化)
    std::vector<int> to_vector() const {
        std::vector<int> vec;
        vec.reserve(visited_count_);
        for (size_t i = 0; i < visited_epoch_.size(); ++i) {
            if (visited_epoch_[i] == cur_epoch_) vec.push_back(static_cast<int>(i));
        }
        return vec;
    }

    std::size_t size() const { return visited_count_; }

    std::uint64_t capacity_bytes() const {
        return static_cast<std::uint64_t>(visited_epoch_.capacity()) * sizeof(std::uint32_t);
    }

private:
    std::vector<std::uint32_t> visited_epoch_;  // 每个节点的epoch标记
    std::uint32_t cur_epoch_;                   // 当前epoch
    std::size_t visited_count_;                 // 已访问节点数
};
