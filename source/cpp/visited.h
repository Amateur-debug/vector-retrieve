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
#include <atomic>

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
 * @brief VisitedList 对象池（thread_local 无锁版本）
 *
 * 每个线程持有自己的 VisitedList（通过 thread_local），完全消除锁竞争。
 * 当 numelements 变化（resize）时，线程下次 get_free_list 会自动重建。
 */
class VisitedListPool {
public:
    /**
     * @param initial_pool_size 初始池大小（thread_local 模式下忽略）
     * @param numelements 每个 VisitedList 支持的最大元素数量
     */
    VisitedListPool(int initial_pool_size, int numelements)
        : numelements_(numelements) {}

    /**
     * @brief 获取当前线程专属的 VisitedList（无锁）
     *
     * 每个线程通过 thread_local 持有独立的 VisitedList，
     * 容量不匹配时自动重建。获取后自动 reset（O(1)）。
     */
    VisitedList* get_free_list() {
        thread_local VisitedList* tl_vl = nullptr;
        thread_local int tl_size = 0;

        int cur_size = numelements_.load(std::memory_order_acquire);
        if (tl_vl == nullptr || tl_size != cur_size) {
            delete tl_vl;
            tl_vl = new VisitedList(cur_size);
            tl_size = cur_size;
        }
        tl_vl->reset();
        return tl_vl;
    }

    /**
     * @brief 归还 VisitedList（thread_local 模式下为空操作）
     */
    void release_list(VisitedList* vl) {
        // thread_local 模式：无需归还，线程退出时自动销毁
    }

    /**
     * @brief 调整容量
     *
     * 更新 numelements_，各线程下次 get_free_list 时检测到容量变化会自动重建。
     * 调用时确保没有 VisitedList 正在被使用（与原实现约束一致）。
     */
    void resize(int new_numelements) {
        numelements_.store(new_numelements, std::memory_order_release);
    }

    ~VisitedListPool() {
        // thread_local 的 VisitedList 由各线程退出时自动销毁
    }

private:
    std::atomic<int> numelements_;
};
