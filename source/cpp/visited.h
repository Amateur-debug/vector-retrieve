/**
 * @file visited.h
 * @brief 访问标记列表及其对象池实现（参考 hnswlib visited_list_pool.h）
 *
 * 提供高效的节点访问状态管理机制：
 *   - 使用 unsigned short 数组 + tag 递增实现 O(1) 的标记、查询和重置操作；
 *   - 采用 thread_local 无锁对象池模式，消除多线程竞争开销；
 *   - 当容量变化（如 resize）时，各线程自动重建本地实例。
 */

#pragma once

#include <vector>
#include <cstring>
#include <atomic>

/**
 * @brief 单个访问标记列表
 *
 * 核心机制：维护一个 unsigned short 数组和一个递增的 tag 值。
 * 标记节点时将对应位置设为当前 tag，查询时比较是否等于当前 tag。
 * 重置操作仅需递增 tag（O(1)），无需遍历数组清零。
 * 当 tag 溢出（回绕到 0）时，执行一次 memset 全量清零并重置 tag 为 1。
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

    // 将指定节点标记为已访问（将数组对应位置设为当前 tag 值）
    void mark(int id) {
        visited_array[id] = cur_tag;
    }

    // 检查指定节点是否在当前轮次中已被访问
    bool is_visited(int id) const {
        return visited_array[id] == cur_tag;
    }

    // 重置所有访问标记（O(1) 摊还复杂度）。
    // 通过递增 tag 值使所有旧标记失效；当 tag 溢出回绕到 0 时，
    // 执行一次 memset 全量清零以保证正确性。
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
 * @brief 访问标记列表对象池（基于 thread_local 的无锁实现）
 *
 * 每个线程通过 thread_local 持有独立的 VisitedList 实例，完全消除锁竞争。
 * 当全局容量发生变化（如 HNSW 索引 resize 扩容）时，线程在下次调用
 * get_free_list() 时检测到容量不匹配，会自动销毁旧实例并重建。
 */
class VisitedListPool {
public:
    /**
     * @param initial_pool_size 初始池大小（thread_local 模式下该参数被忽略，保留仅为接口兼容）
     * @param numelements 每个 VisitedList 支持的最大元素数量（对应 HNSW 索引的 max_elements）
     */
    VisitedListPool(int initial_pool_size, int numelements)
        : numelements_(numelements) {}

    /**
     * @brief 获取当前线程专属的 VisitedList 实例（完全无锁）
     *
     * 每个线程通过 thread_local 持有独立的 VisitedList 实例。
     * 若容量与当前全局 numelements_ 不匹配（说明索引已扩容），则自动销毁旧实例并重建。
     * 返回前自动执行 reset()，确保调用方拿到的是干净的访问标记列表。
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
     * @brief 归还 VisitedList 实例
     *
     * thread_local 模式下为空操作，保留接口仅为兼容旧的池化模式。
     * 线程退出时 thread_local 变量会自动销毁，无需手动管理生命周期。
     */
    void release_list(VisitedList* vl) {
        // thread_local 模式：无需归还，线程退出时自动销毁
    }

    /**
     * @brief 调整支持的最大元素数量
     *
     * 原子更新 numelements_ 的值。各线程在下次调用 get_free_list() 时，
     * 会检测到容量变化并自动重建 VisitedList 实例。
     * 调用约束：调用时需确保当前没有 VisitedList 正在被使用（与 hnswlib 原实现一致）。
     */
    void resize(int new_numelements) {
        numelements_.store(new_numelements, std::memory_order_release);
    }

    ~VisitedListPool() {
        // thread_local 持有的 VisitedList 实例由各线程退出时自动销毁，此处无需额外清理
    }

private:
    std::atomic<int> numelements_;
};
