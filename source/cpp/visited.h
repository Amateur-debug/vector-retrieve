/**
 * @file visited.h
 * @brief VisitedListPool 实现（参考 hnswlib visited_list_pool.h）
 *
 * 池化管理的 visited 数组，O(1) 检查/标记/重置，多线程友好。
 * 替代旧的基于 epoch 的 Visited 类。
 */

#pragma once

#include <vector>
#include <cstring>
#include <mutex>

/**
 * @brief 单个 visited 列表
 *
 * 使用 unsigned short 数组 + tag 机制实现 O(1) 检查和标记。
 * 重置时只需递增 cur_tag，溢出时 memset 清零。
 */
class VisitedList {
public:
    unsigned short cur_tag;
    unsigned short* visited_array;
    int numelements;

    VisitedList(int numelements)
        : cur_tag(1), numelements(numelements) {
        visited_array = new unsigned short[numelements]();
    }

    // 标记节点为已访问
    void mark(int id) {
        visited_array[id] = cur_tag;
    }

    // 检查节点是否已访问
    bool is_visited(int id) const {
        return visited_array[id] == cur_tag;
    }

    // O(1) 重置：递增 tag，溢出时 memset
    void reset() {
        cur_tag++;
        if (cur_tag == 0) {
            memset(visited_array, 0, sizeof(unsigned short) * numelements);
            cur_tag = 1;
        }
    }

    ~VisitedList() {
        delete[] visited_array;
    }
};

/**
 * @brief VisitedList 对象池
 *
 * 线程安全的 VisitedList 池，每个搜索线程从池中获取一个 VisitedList，
 * 搜索结束后归还。避免多线程竞争和频繁内存分配。
 */
class VisitedListPool {
public:
    /**
     * @param initial_pool_size 初始池大小（预创建的 VisitedList 数量）
     * @param numelements 每个 VisitedList 支持的最大元素数量
     */
    VisitedListPool(int initial_pool_size, int numelements)
        : numelements_(numelements) {
        for (int i = 0; i < initial_pool_size; i++) {
            pool_.push_back(new VisitedList(numelements));
        }
    }

    /**
     * @brief 从池中获取一个空闲的 VisitedList
     *
     * 如果池为空则动态创建新的 VisitedList。
     * 获取后自动 reset（O(1)），调用方独占使用。
     */
    VisitedList* get_free_list() {
        VisitedList* vl;
        {
            std::unique_lock<std::mutex> lock(pool_mutex_);
            if (pool_.empty()) {
                vl = new VisitedList(numelements_);
            } else {
                vl = pool_.back();
                pool_.pop_back();
            }
        }
        vl->reset();
        return vl;
    }

    /**
     * @brief 将 VisitedList 归还到池中
     */
    void release_list(VisitedList* vl) {
        std::unique_lock<std::mutex> lock(pool_mutex_);
        pool_.push_back(vl);
    }

    /**
     * @brief 调整池中所有 VisitedList 的容量
     *
     * 当 max_elements 扩容时调用，重建所有池内的 VisitedList。
     * 注意：调用时确保没有 VisitedList 正在被使用。
     */
    void resize(int new_numelements) {
        std::unique_lock<std::mutex> lock(pool_mutex_);
        numelements_ = new_numelements;
        for (auto& vl : pool_) {
            delete vl;
            vl = new VisitedList(new_numelements);
        }
    }

    ~VisitedListPool() {
        for (auto* vl : pool_) {
            delete vl;
        }
        pool_.clear();
    }

private:
    std::vector<VisitedList*> pool_;
    std::mutex pool_mutex_;
    int numelements_;
};
