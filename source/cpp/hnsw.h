/**
 * @file hnsw.h
 * @brief HNSW图实现 - 高性能近似最近邻搜索（hnswlib 风格连续内存布局）
 *
 * 特性:
 *   - 连续内存布局：data_level0_ 存储 level 0 数据，link_lists_ 存储高层链接
 *   - 无锁 query：search_layer_query 完全无 mutex
 *   - 节点级锁：建图时使用 link_locks_ 细粒度锁
 *   - 定长邻居数组：[count | id0 | id1 | ... | id_{max-1}]
 *   - 距离函数指针化：通过 SpaceInterface 运行时选择最优 SIMD 实现
 *   - VisitedListPool：池化 visited 数组，O(1) reset
 *   - 二进制 save/load：直接读写连续内存
 *   - 预取优化：搜索循环中使用 _mm_prefetch
 */

#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <queue>
#include <random>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <xmmintrin.h>
#endif

#include "payload.h"
#include "distance.h"
#include "visited.h"

template <typename T>
class HNSWGraph
{
public:
    // === 基本参数 ===
    int dim;
    int m;                  // 高层最大邻居数
    int ef_construction;
    int ef_search;
    int limited_level;
    int max_level;
    int entry_point_id;     // internal_id of entry point (-1 if empty)
    unsigned int seed;

    // === 连续内存布局 ===
    char *data_level0_;                     // level 0 连续内存块
    char **link_lists_;                     // 高层邻居指针数组
    std::vector<std::mutex> link_locks_;    // 节点级锁数组
    std::mutex global_mutex_;               // 全局锁（保护 entry_point_id, max_level）

    // === 内存布局参数 ===
    size_t max_elements_;           // 最大元素容量
    std::atomic<size_t> cur_element_count_;  // 当前元素数量
    size_t size_per_element_;       // 每个元素占用字节数
    size_t links_level0_size_;      // level 0 链接区域大小
    size_t data_offset_;            // 向量数据在元素内的偏移
    size_t label_offset_;           // label 在元素内的偏移
    size_t size_links_per_element_; // 高层每层链接占用字节数
    int max_m_;                     // 高层最大邻居数 (= m)
    int max_m0_;                    // level 0 最大邻居数 (= 2*m)
    double level_mult_;             // 层级生成乘数

    // === ID 映射 ===
    std::unordered_map<int, int> label_lookup_;  // vid -> internal_id
    std::mutex label_lookup_mutex_;

    // === 距离函数 ===
    DISTFUNC<T> dist_func_;
    void *dist_func_param_;
    std::unique_ptr<SpaceInterface<T>> space_;

    // === VisitedListPool ===
    std::unique_ptr<VisitedListPool> visited_pool_;

    // === 层级记录 ===
    std::vector<int> element_levels_;  // 记录每个 internal_id 的层级

    // === 随机数生成器 ===
    std::mt19937 rng;
    std::uniform_real_distribution<double> uniform_;

    // ========================================================
    // 构造函数
    // ========================================================
    HNSWGraph(int dim, int m = 16, int ef_construction = 128, int ef_search = 64,
              int limited_level = 5, size_t max_elements = 100000, unsigned int seed = 123)
        : dim(dim),
          m(m),
          ef_construction(ef_construction),
          ef_search(ef_search),
          limited_level(limited_level),
          max_level(0),
          entry_point_id(-1),
          seed(seed),
          data_level0_(nullptr),
          link_lists_(nullptr),
          max_elements_(max_elements),
          cur_element_count_(0),
          max_m_(m),
          max_m0_(2 * m),
          level_mult_(1.0 / log(static_cast<double>(m))),
          rng(seed),
          uniform_(0.0, 1.0)
    {
        // 计算内存布局参数
        links_level0_size_ = (max_m0_ + 1) * sizeof(int);  // count + max_m0_ 个邻居 ID
        data_offset_ = links_level0_size_;
        label_offset_ = links_level0_size_ + dim * sizeof(T);
        size_per_element_ = links_level0_size_ + dim * sizeof(T) + sizeof(int);
        size_links_per_element_ = (max_m_ + 1) * sizeof(int);  // count + max_m_ 个邻居 ID

        // 分配 level 0 连续内存
        data_level0_ = (char *)malloc(max_elements_ * size_per_element_);
        if (!data_level0_)
            throw std::runtime_error("HNSWGraph: failed to allocate data_level0_");
        memset(data_level0_, 0, max_elements_ * size_per_element_);

        // 分配高层指针数组
        link_lists_ = (char **)malloc(sizeof(char *) * max_elements_);
        if (!link_lists_)
            throw std::runtime_error("HNSWGraph: failed to allocate link_lists_");
        memset(link_lists_, 0, sizeof(char *) * max_elements_);

        // 预分配节点级锁
        link_locks_ = std::vector<std::mutex>(max_elements_);

        // 预分配层级记录
        element_levels_.resize(max_elements_, 0);

        // 初始化距离函数
        space_ = create_l2_space<T>(dim);
        dist_func_ = space_->get_dist_func();
        dist_func_param_ = space_->get_dist_func_param();

        // 初始化 VisitedListPool
        visited_pool_ = std::make_unique<VisitedListPool>(1, max_elements_);
    }

    // ========================================================
    // 析构函数
    // ========================================================
    ~HNSWGraph()
    {
        if (data_level0_)
        {
            free(data_level0_);
            data_level0_ = nullptr;
        }
        if (link_lists_)
        {
            for (size_t i = 0; i < cur_element_count_.load(); i++)
            {
                if (link_lists_[i])
                    free(link_lists_[i]);
            }
            free(link_lists_);
            link_lists_ = nullptr;
        }
    }

    // 禁止拷贝
    HNSWGraph(const HNSWGraph &) = delete;
    HNSWGraph &operator=(const HNSWGraph &) = delete;

    // ========================================================
    // 内存访问辅助函数
    // ========================================================

    // 获取 internal_id 对应的向量数据指针
    inline char *get_data_by_id(int internal_id) const
    {
        return data_level0_ + (size_t)internal_id * size_per_element_ + data_offset_;
    }

    // 获取 internal_id 对应的外部 label (vid)
    inline int get_external_label(int internal_id) const
    {
        return *(int *)(data_level0_ + (size_t)internal_id * size_per_element_ + label_offset_);
    }

    // 设置 internal_id 对应的外部 label (vid)
    inline void set_external_label(int internal_id, int label)
    {
        *(int *)(data_level0_ + (size_t)internal_id * size_per_element_ + label_offset_) = label;
    }

    // 获取 level 0 的邻居列表指针（包含 count 和 neighbor ids）
    inline int *get_linklist0(int internal_id) const
    {
        return (int *)(data_level0_ + (size_t)internal_id * size_per_element_);
    }

    // 获取高层的邻居列表指针
    inline int *get_linklist(int internal_id, int level) const
    {
        return (int *)(link_lists_[internal_id] + (size_t)(level - 1) * size_links_per_element_);
    }

    // 获取邻居列表的邻居数量
    inline int get_list_count(const int *data) const
    {
        return data[0];
    }

    // 设置邻居列表的邻居数量
    inline void set_list_count(int *data, int count)
    {
        data[0] = count;
    }

    // ========================================================
    // 动态扩容
    // ========================================================
    void resize_index(size_t new_max_elements)
    {
        if (new_max_elements <= max_elements_)
            return;

        // 扩容 data_level0_
        char *new_data = (char *)realloc(data_level0_, new_max_elements * size_per_element_);
        if (!new_data)
            throw std::runtime_error("HNSWGraph: failed to realloc data_level0_");
        // 新区域清零
        memset(new_data + max_elements_ * size_per_element_, 0,
               (new_max_elements - max_elements_) * size_per_element_);
        data_level0_ = new_data;

        // 扩容 link_lists_
        char **new_links = (char **)realloc(link_lists_, sizeof(char *) * new_max_elements);
        if (!new_links)
            throw std::runtime_error("HNSWGraph: failed to realloc link_lists_");
        memset(new_links + max_elements_, 0, sizeof(char *) * (new_max_elements - max_elements_));
        link_lists_ = new_links;

        // 扩容锁数组（需要创建新的）
        link_locks_ = std::vector<std::mutex>(new_max_elements);

        // 扩容层级记录
        element_levels_.resize(new_max_elements, 0);

        // 扩容 visited pool
        visited_pool_->resize(new_max_elements);

        max_elements_ = new_max_elements;
    }

    // ========================================================
    // 随机层级生成
    // ========================================================
    int random_level()
    {
        double r = uniform_(rng);
        if (r == 0.0)
            r = std::numeric_limits<double>::min();

        int level = static_cast<int>(-log(r) * level_mult_);

        if (limited_level >= 0)
            level = std::min(level, limited_level);

        return level;
    }

    // ========================================================
    // 启发式邻居选择（对齐 hnswlib getNeighborsByHeuristic2）
    // ========================================================
    void select_neighbors_heuristic(
        std::priority_queue<std::pair<T, int>, std::vector<std::pair<T, int>>, std::greater<std::pair<T, int>>> &top_candidates,
        int M)
    {
        if ((int)top_candidates.size() <= M)
            return;  // 早退优化

        // 将 min-heap 中的元素按距离从近到远提取出来
        std::priority_queue<std::pair<T, int>> queue_closest;  // max-heap
        std::vector<std::pair<T, int>> return_list;

        while (!top_candidates.empty())
        {
            queue_closest.push({-top_candidates.top().first, top_candidates.top().second});
            top_candidates.pop();
        }

        while (!queue_closest.empty())
        {
            if ((int)return_list.size() >= M)
                break;

            auto cur = queue_closest.top();
            queue_closest.pop();
            T dist_to_query = -cur.first;

            bool good = true;
            for (auto &ret : return_list)
            {
                T cur_dist = dist_func_(get_data_by_id(cur.second),
                                        get_data_by_id(ret.second),
                                        dist_func_param_);
                if (cur_dist < dist_to_query)
                {
                    good = false;
                    break;
                }
            }
            if (good)
            {
                return_list.push_back({dist_to_query, cur.second});
            }
        }

        // 写回 top_candidates
        for (auto &p : return_list)
        {
            top_candidates.push(p);
        }
    }

    // ========================================================
    // connect_new_element - 建立双向连接
    // 参考 hnswlib mutuallyConnectNewElement
    // ========================================================
    void connect_new_element(
        int cur_c,  // 新节点的 internal_id
        std::priority_queue<std::pair<T, int>, std::vector<std::pair<T, int>>, std::greater<std::pair<T, int>>> &top_candidates,
        int level)
    {
        int Mcurmax = (level == 0) ? max_m0_ : max_m_;

        select_neighbors_heuristic(top_candidates, Mcurmax);

        // 提取选中的邻居
        std::vector<int> selected_neighbors;
        selected_neighbors.reserve(top_candidates.size());
        while (!top_candidates.empty())
        {
            selected_neighbors.push_back(top_candidates.top().second);
            top_candidates.pop();
        }

        // 写入新节点自身的邻居列表（新节点不需要加锁，因为还没被其他线程发现）
        {
            int *ll_cur;
            if (level == 0)
                ll_cur = get_linklist0(cur_c);
            else
                ll_cur = get_linklist(cur_c, level);

            set_list_count(ll_cur, (int)selected_neighbors.size());
            int *data = ll_cur + 1;
            for (int i = 0; i < (int)selected_neighbors.size(); i++)
            {
                data[i] = selected_neighbors[i];
            }
        }

        // 为每个被选中的邻居建立反向连接
        for (int selected_id : selected_neighbors)
        {
            std::unique_lock<std::mutex> lock(link_locks_[selected_id]);

            int *ll_other;
            if (level == 0)
                ll_other = get_linklist0(selected_id);
            else
                ll_other = get_linklist(selected_id, level);

            int sz_other = get_list_count(ll_other);

            if (sz_other < Mcurmax)
            {
                // 直接追加
                int *data = ll_other + 1;
                data[sz_other] = cur_c;
                set_list_count(ll_other, sz_other + 1);
            }
            else
            {
                // 邻居已满，需要用启发式选择缩减
                T d_max = dist_func_(get_data_by_id(cur_c),
                                     get_data_by_id(selected_id),
                                     dist_func_param_);

                // 构建候选集 = 现有邻居 + 新节点
                std::priority_queue<std::pair<T, int>, std::vector<std::pair<T, int>>,
                                    std::greater<std::pair<T, int>>>
                    candidates;
                candidates.push({d_max, cur_c});

                int *data = ll_other + 1;
                for (int j = 0; j < sz_other; j++)
                {
                    T dist_j = dist_func_(get_data_by_id(data[j]),
                                          get_data_by_id(selected_id),
                                          dist_func_param_);
                    candidates.push({dist_j, data[j]});
                }

                select_neighbors_heuristic(candidates, Mcurmax);

                // 写回
                int count = 0;
                while (!candidates.empty())
                {
                    data[count] = candidates.top().second;
                    candidates.pop();
                    count++;
                }
                set_list_count(ll_other, count);
            }
        }
    }

    // ========================================================
    // search_layer_build - 建图搜索（加节点锁）
    // 参考 hnswlib searchBaseLayer
    // ========================================================
    std::priority_queue<std::pair<T, int>, std::vector<std::pair<T, int>>, std::greater<std::pair<T, int>>>
    search_layer_build(const void *query_data, int ep_id, int ef, int level)
    {
        VisitedList *vl = visited_pool_->get_free_list();
        unsigned short *visited_array = vl->visited_array;
        unsigned short visited_tag = vl->cur_tag;

        // 结果集 max-heap（距离最大的在堆顶，方便弹出最远的）
        std::priority_queue<std::pair<T, int>> top_candidates;
        // 候选集 min-heap（距离最小的在堆顶）
        std::priority_queue<std::pair<T, int>, std::vector<std::pair<T, int>>,
                            std::greater<std::pair<T, int>>>
            candidate_set;

        T dist = dist_func_(query_data, get_data_by_id(ep_id), dist_func_param_);
        top_candidates.push({dist, ep_id});
        candidate_set.push({dist, ep_id});
        visited_array[ep_id] = visited_tag;

        T lower_bound = dist;

        while (!candidate_set.empty())
        {
            auto cand = candidate_set.top();

            if (cand.first > lower_bound)
                break;

            candidate_set.pop();
            int cur_node_id = cand.second;

            // 建图搜索：加节点锁读取邻居
            int *ll_cur;
            {
                std::unique_lock<std::mutex> lock(link_locks_[cur_node_id]);
                if (level == 0)
                    ll_cur = get_linklist0(cur_node_id);
                else
                    ll_cur = get_linklist(cur_node_id, level);
            }

            int size = get_list_count(ll_cur);
            int *data = ll_cur + 1;

            for (int j = 0; j < size; j++)
            {
                int candidate_id = data[j];

                if (visited_array[candidate_id] == visited_tag)
                    continue;
                visited_array[candidate_id] = visited_tag;

                T dist_cand = dist_func_(query_data, get_data_by_id(candidate_id), dist_func_param_);

                if ((int)top_candidates.size() < ef || dist_cand < lower_bound)
                {
                    candidate_set.push({dist_cand, candidate_id});
                    top_candidates.push({dist_cand, candidate_id});

                    if ((int)top_candidates.size() > ef)
                        top_candidates.pop();

                    if (!top_candidates.empty())
                        lower_bound = top_candidates.top().first;
                }
            }
        }

        visited_pool_->release_list(vl);

        // 转换为 min-heap 返回
        std::priority_queue<std::pair<T, int>, std::vector<std::pair<T, int>>,
                            std::greater<std::pair<T, int>>>
            result;
        while (!top_candidates.empty())
        {
            result.push(top_candidates.top());
            top_candidates.pop();
        }
        return result;
    }

    // ========================================================
    // search_layer_query - query 搜索（完全无锁 + 预取优化）
    // 参考 hnswlib searchBaseLayerST
    // ========================================================
    std::priority_queue<std::pair<T, int>>
    search_layer_query(const void *query_data, int ep_id, int ef)
    {
        VisitedList *vl = visited_pool_->get_free_list();
        unsigned short *visited_array = vl->visited_array;
        unsigned short visited_tag = vl->cur_tag;

        // 结果集 max-heap
        std::priority_queue<std::pair<T, int>> top_candidates;
        // 候选集 min-heap
        std::priority_queue<std::pair<T, int>, std::vector<std::pair<T, int>>,
                            std::greater<std::pair<T, int>>>
            candidate_set;

        T dist = dist_func_(query_data, get_data_by_id(ep_id), dist_func_param_);
        top_candidates.push({dist, ep_id});
        candidate_set.push({dist, ep_id});
        visited_array[ep_id] = visited_tag;

        T lower_bound = dist;

        while (!candidate_set.empty())
        {
            auto cand = candidate_set.top();

            if (cand.first > lower_bound)
                break;

            candidate_set.pop();
            int cur_node_id = cand.second;

            // 完全无锁：直接读取 level 0 邻居列表
            int *ll_cur = get_linklist0(cur_node_id);
            int size = get_list_count(ll_cur);
            int *data = ll_cur + 1;

#if defined(__x86_64__) || defined(_M_X64)
            // 预取第一个邻居的 visited 标记和向量数据
            if (size > 0)
            {
                _mm_prefetch((char *)(visited_array + data[0]), _MM_HINT_T0);
                _mm_prefetch(get_data_by_id(data[0]), _MM_HINT_T0);
            }
#endif

            for (int j = 0; j < size; j++)
            {
#if defined(__x86_64__) || defined(_M_X64)
                // 预取下一个邻居
                if (j + 1 < size)
                {
                    _mm_prefetch((char *)(visited_array + data[j + 1]), _MM_HINT_T0);
                    _mm_prefetch(get_data_by_id(data[j + 1]), _MM_HINT_T0);
                }
#endif

                int candidate_id = data[j];

                if (visited_array[candidate_id] == visited_tag)
                    continue;
                visited_array[candidate_id] = visited_tag;

                T dist_cand = dist_func_(query_data, get_data_by_id(candidate_id), dist_func_param_);

                if ((int)top_candidates.size() < ef || dist_cand < lower_bound)
                {
                    candidate_set.push({dist_cand, candidate_id});
                    top_candidates.push({dist_cand, candidate_id});

                    if ((int)top_candidates.size() > ef)
                        top_candidates.pop();

                    if (!top_candidates.empty())
                        lower_bound = top_candidates.top().first;
                }
            }
        }

        visited_pool_->release_list(vl);

        return top_candidates;
    }

    // ========================================================
    // add_vector - 添加向量（参考 hnswlib addPoint）
    // ========================================================
    void add_vector(int vid, const std::vector<T> &vec)
    {
        int cur_c;  // 新节点的 internal_id
        int curlevel;

        {
            std::unique_lock<std::mutex> lock(global_mutex_);

            // 检查是否已存在
            auto it = label_lookup_.find(vid);
            if (it != label_lookup_.end())
            {
                // 已存在则更新向量数据
                int existing_id = it->second;
                memcpy(get_data_by_id(existing_id), vec.data(), dim * sizeof(T));
                return;
            }

            // 检查容量，必要时扩容
            if (cur_element_count_.load() >= max_elements_)
            {
                resize_index(max_elements_ * 2);
            }

            cur_c = (int)cur_element_count_.load();
            cur_element_count_++;
            curlevel = random_level();

            element_levels_[cur_c] = curlevel;
            label_lookup_[vid] = cur_c;
        }

        // 将向量数据写入 data_level0_
        memcpy(get_data_by_id(cur_c), vec.data(), dim * sizeof(T));
        // 设置外部 label
        set_external_label(cur_c, vid);

        // 初始化 level 0 邻居列表为空
        memset(data_level0_ + (size_t)cur_c * size_per_element_, 0, links_level0_size_);

        // 如果有高层，分配高层内存
        if (curlevel > 0)
        {
            size_t link_size = size_links_per_element_ * curlevel;
            link_lists_[cur_c] = (char *)malloc(link_size);
            if (!link_lists_[cur_c])
                throw std::runtime_error("HNSWGraph: failed to allocate link_lists_ for node");
            memset(link_lists_[cur_c], 0, link_size);
        }

        // 如果是第一个节点，设为入口点
        {
            std::unique_lock<std::mutex> lock(global_mutex_);
            if (entry_point_id == -1)
            {
                entry_point_id = cur_c;
                max_level = curlevel;
                return;
            }
        }

        int ep;
        int maxlevelcopy;
        {
            std::unique_lock<std::mutex> lock(global_mutex_);
            ep = entry_point_id;
            maxlevelcopy = max_level;
        }

        // 高层贪心导航（level maxlevelcopy ~ curlevel+1）
        if (curlevel < maxlevelcopy)
        {
            const void *query_data = vec.data();
            T cur_dist = dist_func_(query_data, get_data_by_id(ep), dist_func_param_);

            for (int level = maxlevelcopy; level > curlevel; level--)
            {
                bool changed = true;
                while (changed)
                {
                    changed = false;
                    int *ll = get_linklist(ep, level);
                    int size = get_list_count(ll);
                    int *data = ll + 1;

                    for (int i = 0; i < size; i++)
                    {
                        int cand = data[i];
                        T d = dist_func_(query_data, get_data_by_id(cand), dist_func_param_);
                        if (d < cur_dist)
                        {
                            cur_dist = d;
                            ep = cand;
                            changed = true;
                        }
                    }
                }
            }
        }

        // 对 min(curlevel, maxlevelcopy) ~ 0 层建立连接
        for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--)
        {
            auto top_candidates = search_layer_build(vec.data(), ep, ef_construction, level);

            // 从搜索结果中取最近的作为下一层入口
            if (!top_candidates.empty())
                ep = top_candidates.top().second;

            connect_new_element(cur_c, top_candidates, level);
        }

        // 如果新节点层级 > 当前最大层级，更新入口点
        if (curlevel > maxlevelcopy)
        {
            std::unique_lock<std::mutex> lock(global_mutex_);
            if (curlevel > max_level)
            {
                max_level = curlevel;
                entry_point_id = cur_c;
            }
        }
    }

    // ========================================================
    // search_layer_ef - 兼容旧接口的搜索层（用于 query 中的分布式搜索）
    // 使用 Payload 进行跨节点续搜
    // ========================================================
    void search_layer_ef(const std::vector<T> &query_vector, int level, int ef, Payload<T> &payload)
    {
        int ep_internal_id = -1;

        // 将 payload.entry_point_id (vid) 转为 internal_id
        if (payload.entry_point_id != -1)
        {
            std::unique_lock<std::mutex> lock(label_lookup_mutex_);
            auto it = label_lookup_.find(payload.entry_point_id);
            if (it != label_lookup_.end())
            {
                ep_internal_id = it->second;
            }
            else
            {
                // 该 vid 不在本地，触发跨节点搜索
                payload.continue_search = true;
                payload.require_id = payload.entry_point_id;
                payload.start_level = level;
                return;
            }
        }

        payload.entry_point_id = -1;

        // 获取 VisitedList
        VisitedList *vl = visited_pool_->get_free_list();
        unsigned short *visited_array = vl->visited_array;
        unsigned short visited_tag = vl->cur_tag;

        // 如果是续搜，恢复 visited 状态
        if (payload.continue_search)
        {
            for (int vid : payload.visited_vids)
            {
                std::unique_lock<std::mutex> lock(label_lookup_mutex_);
                auto it = label_lookup_.find(vid);
                if (it != label_lookup_.end())
                {
                    visited_array[it->second] = visited_tag;
                }
            }

            // 如果续搜有入口点，加入搜索
            if (ep_internal_id != -1 && visited_array[ep_internal_id] != visited_tag)
            {
                T dist = dist_func_(query_vector.data(), get_data_by_id(ep_internal_id), dist_func_param_);
                payload.distance_cnt++;
                visited_array[ep_internal_id] = visited_tag;

                // 将 candidates 和 results 中的 vid 转为 internal_id
                // 注意：续搜时 payload 中的 ID 是 vid
            }

            // 恢复 candidates（vid -> internal_id）
            std::priority_queue<std::pair<T, int>, std::vector<std::pair<T, int>>,
                                std::greater<std::pair<T, int>>>
                local_candidates;
            while (!payload.candidates.empty())
            {
                auto top = payload.candidates.top();
                payload.candidates.pop();
                std::unique_lock<std::mutex> lock(label_lookup_mutex_);
                auto it = label_lookup_.find(top.second);
                if (it != label_lookup_.end())
                {
                    local_candidates.push({top.first, it->second});
                }
            }

            // 恢复 results（vid -> internal_id）
            std::priority_queue<std::pair<T, int>> local_results;
            while (!payload.results.empty())
            {
                auto top = payload.results.top();
                payload.results.pop();
                std::unique_lock<std::mutex> lock(label_lookup_mutex_);
                auto it = label_lookup_.find(top.second);
                if (it != label_lookup_.end())
                {
                    local_results.push({top.first, it->second});
                }
            }

            // 执行搜索主循环（使用 internal_id）
            T lower_bound = local_results.empty() ? std::numeric_limits<T>::max() : local_results.top().first;

            while (!local_candidates.empty())
            {
                auto cand = local_candidates.top();
                if (cand.first > lower_bound)
                    break;
                local_candidates.pop();
                int cur_node = cand.second;

                int *ll_cur = (level == 0) ? get_linklist0(cur_node) : get_linklist(cur_node, level);
                int size = get_list_count(ll_cur);
                int *data = ll_cur + 1;

                for (int j = 0; j < size; j++)
                {
                    int neighbor_id = data[j];

                    if (visited_array[neighbor_id] == visited_tag)
                        continue;

                    // 检查邻居是否需要跨节点查找（通过检查是否有高层连接来简单判断不合适，
                    // 这里依赖于连续内存布局：如果 neighbor_id < cur_element_count_ 则在本地）
                    visited_array[neighbor_id] = visited_tag;
                    T neighbor_dist = dist_func_(query_vector.data(), get_data_by_id(neighbor_id), dist_func_param_);
                    payload.distance_cnt++;

                    if ((int)local_results.size() < ef || neighbor_dist < lower_bound)
                    {
                        local_candidates.push({neighbor_dist, neighbor_id});
                        local_results.push({neighbor_dist, neighbor_id});

                        if ((int)local_results.size() > ef)
                            local_results.pop();

                        if (!local_results.empty())
                            lower_bound = local_results.top().first;
                    }
                }
            }

            // 将 internal_id 转回 vid 写入 payload
            while (!local_results.empty())
            {
                auto top = local_results.top();
                local_results.pop();
                payload.results.push({top.first, get_external_label(top.second)});
            }

            // 提取 visited 列表（internal_id -> vid）
            payload.visited_vids.clear();
            size_t cur_count = cur_element_count_.load();
            for (size_t i = 0; i < cur_count; i++)
            {
                if (visited_array[i] == visited_tag)
                {
                    payload.visited_vids.push_back(get_external_label((int)i));
                }
            }

            payload.continue_search = false;
            visited_pool_->release_list(vl);
            return;
        }

        // 非续搜的正常搜索流程
        if (ep_internal_id == -1)
        {
            visited_pool_->release_list(vl);
            return;
        }

        T dist = dist_func_(query_vector.data(), get_data_by_id(ep_internal_id), dist_func_param_);
        payload.distance_cnt++;
        visited_array[ep_internal_id] = visited_tag;

        // 使用内部 candidates 和 results（internal_id）
        std::priority_queue<std::pair<T, int>, std::vector<std::pair<T, int>>,
                            std::greater<std::pair<T, int>>>
            local_candidates;
        std::priority_queue<std::pair<T, int>> local_results;

        local_candidates.push({dist, ep_internal_id});
        local_results.push({dist, ep_internal_id});

        T lower_bound = dist;

        while (!local_candidates.empty())
        {
            auto cand = local_candidates.top();
            if (cand.first > lower_bound)
                break;
            local_candidates.pop();
            int cur_node = cand.second;

            int *ll_cur;
            if (level == 0)
                ll_cur = get_linklist0(cur_node);
            else
            {
                if (element_levels_[cur_node] < level)
                {
                    continue;
                }
                ll_cur = get_linklist(cur_node, level);
            }

            int size = get_list_count(ll_cur);
            int *data = ll_cur + 1;

            for (int j = 0; j < size; j++)
            {
                int neighbor_id = data[j];

                if (visited_array[neighbor_id] == visited_tag)
                    continue;

                visited_array[neighbor_id] = visited_tag;
                T neighbor_dist = dist_func_(query_vector.data(), get_data_by_id(neighbor_id), dist_func_param_);
                payload.distance_cnt++;

                if ((int)local_results.size() < ef || neighbor_dist < lower_bound)
                {
                    local_candidates.push({neighbor_dist, neighbor_id});
                    local_results.push({neighbor_dist, neighbor_id});

                    if ((int)local_results.size() > ef)
                        local_results.pop();

                    if (!local_results.empty())
                        lower_bound = local_results.top().first;
                }
            }
        }

        // 将 internal_id 转回 vid 写入 payload
        while (!local_results.empty())
        {
            auto top = local_results.top();
            local_results.pop();
            payload.results.push({top.first, get_external_label(top.second)});
        }

        // 提取 visited 列表（internal_id -> vid）
        payload.visited_vids.clear();
        size_t cur_count = cur_element_count_.load();
        for (size_t i = 0; i < cur_count; i++)
        {
            if (visited_array[i] == visited_tag)
            {
                payload.visited_vids.push_back(get_external_label((int)i));
            }
        }

        payload.continue_search = false;
        visited_pool_->release_list(vl);
    }

    // ========================================================
    // query - K近邻搜索
    // 参考 hnswlib searchKnn
    // ========================================================
    std::vector<int> query(const std::vector<T> &query_vector, Payload<T> &payload, int k)
    {
        if (cur_element_count_.load() == 0)
            return {};

        // 如果是续搜，直接在当前层搜索
        if (payload.continue_search)
        {
            search_layer_ef(query_vector, payload.start_level, std::max(ef_search, k), payload);
            if (payload.continue_search)
                return payload.topk_ids(k);

            // 如果续搜后还有更低层要搜索
            if (payload.start_level > 0)
            {
                auto topk = payload.topk_ids(1);
                for (int l = payload.start_level - 1; l >= 0; l--)
                {
                    payload.clear();
                    payload.entry_point_id = topk[0];
                    payload.start_level = l;
                    search_layer_ef(query_vector, l, std::max(ef_search, k), payload);
                    if (payload.continue_search)
                        return payload.topk_ids(k);
                    topk = payload.topk_ids(1);
                }
            }
            return payload.topk_ids(k);
        }

        // 非续搜：从最高层开始搜索
        int ep;
        int curlevel;
        {
            ep = entry_point_id;
            curlevel = max_level;
        }

        if (ep == -1)
            return {};

        // 高层贪心导航（使用 internal_id）
        const void *query_data = query_vector.data();
        T cur_dist = dist_func_(query_data, get_data_by_id(ep), dist_func_param_);
        payload.distance_cnt++;

        for (int level = curlevel; level > 0; level--)
        {
            bool changed = true;
            while (changed)
            {
                changed = false;
                if (element_levels_[ep] < level)
                    break;
                int *ll = get_linklist(ep, level);
                int size = get_list_count(ll);
                int *data = ll + 1;

                for (int i = 0; i < size; i++)
                {
                    int cand = data[i];
                    T d = dist_func_(query_data, get_data_by_id(cand), dist_func_param_);
                    payload.distance_cnt++;
                    if (d < cur_dist)
                    {
                        cur_dist = d;
                        ep = cand;
                        changed = true;
                    }
                }
            }
        }

        // level 0 搜索（使用 search_layer_query，完全无锁）
        int actual_ef = std::max(ef_search, k);
        auto top_candidates = search_layer_query(query_data, ep, actual_ef);

        // 将 internal_id 转为 vid 写入 payload.results
        payload.results = decltype(payload.results)();
        while (!top_candidates.empty())
        {
            auto top = top_candidates.top();
            top_candidates.pop();
            payload.results.push({top.first, get_external_label(top.second)});
        }

        return payload.topk_ids(k);
    }

    // ========================================================
    // 辅助方法：获取向量（供 Python 绑定使用）
    // ========================================================
    std::vector<T> get_vector(int vid) const
    {
        auto it = label_lookup_.find(vid);
        if (it == label_lookup_.end())
            return {};
        int internal_id = it->second;
        const T *data = (const T *)get_data_by_id(internal_id);
        return std::vector<T>(data, data + dim);
    }

    // 获取邻居列表（供 Python 绑定使用）
    std::vector<int> get_neighbors(int vid, int level) const
    {
        auto it = label_lookup_.find(vid);
        if (it == label_lookup_.end())
            return {};
        int internal_id = it->second;

        int *ll;
        if (level == 0)
            ll = get_linklist0(internal_id);
        else
        {
            if (element_levels_[internal_id] < level)
                return {};
            ll = get_linklist(internal_id, level);
        }

        int size = get_list_count(ll);
        int *data = ll + 1;

        std::vector<int> result;
        result.reserve(size);
        for (int i = 0; i < size; i++)
        {
            result.push_back(get_external_label(data[i]));
        }
        return result;
    }

    // 获取当前元素数量
    size_t get_cur_element_count() const
    {
        return cur_element_count_.load();
    }

    // ========================================================
    // 兼容方法：获取所有向量（供 Cluster::partition 使用）
    // 返回 vid -> vector<T> 的映射
    // ========================================================
    std::unordered_map<int, std::vector<T>> get_all_vectors() const
    {
        std::unordered_map<int, std::vector<T>> result;
        size_t cur_count = cur_element_count_.load();
        result.reserve(cur_count);
        for (size_t i = 0; i < cur_count; i++)
        {
            int vid = get_external_label((int)i);
            const T *data = (const T *)get_data_by_id((int)i);
            result[vid] = std::vector<T>(data, data + dim);
        }
        return result;
    }

    // 兼容方法：获取所有邻居关系（供 Cluster::partition 使用）
    // 返回 level -> vid -> set(neighbor_vids)
    std::unordered_map<int, std::unordered_map<int, std::unordered_set<int>>> get_all_neighbors() const
    {
        std::unordered_map<int, std::unordered_map<int, std::unordered_set<int>>> result;
        size_t cur_count = cur_element_count_.load();
        for (size_t i = 0; i < cur_count; i++)
        {
            int vid = get_external_label((int)i);
            int node_level = element_levels_[i];

            // level 0
            int *ll0 = get_linklist0((int)i);
            int size0 = get_list_count(ll0);
            int *data0 = ll0 + 1;
            std::unordered_set<int> neighbors0;
            for (int j = 0; j < size0; j++)
            {
                neighbors0.insert(get_external_label(data0[j]));
            }
            result[0][vid] = std::move(neighbors0);

            // 高层
            for (int level = 1; level <= node_level; level++)
            {
                int *ll = get_linklist((int)i, level);
                int sz = get_list_count(ll);
                int *data = ll + 1;
                std::unordered_set<int> neighbors_l;
                for (int j = 0; j < sz; j++)
                {
                    neighbors_l.insert(get_external_label(data[j]));
                }
                result[level][vid] = std::move(neighbors_l);
            }
        }
        return result;
    }

    // ========================================================
    // save - 二进制保存
    // ========================================================
    bool save(const std::string &filepath)
    {
        try
        {
            std::ofstream ofs(filepath, std::ios::binary);
            if (!ofs)
                return false;

            size_t cur_count = cur_element_count_.load();

            // 写入参数头
            ofs.write((char *)&links_level0_size_, sizeof(size_t));
            ofs.write((char *)&max_elements_, sizeof(size_t));
            ofs.write((char *)&cur_count, sizeof(size_t));
            ofs.write((char *)&size_per_element_, sizeof(size_t));
            ofs.write((char *)&label_offset_, sizeof(size_t));
            ofs.write((char *)&data_offset_, sizeof(size_t));
            ofs.write((char *)&max_level, sizeof(int));
            ofs.write((char *)&entry_point_id, sizeof(int));
            ofs.write((char *)&max_m_, sizeof(int));
            ofs.write((char *)&max_m0_, sizeof(int));
            ofs.write((char *)&m, sizeof(int));
            ofs.write((char *)&level_mult_, sizeof(double));
            ofs.write((char *)&ef_construction, sizeof(int));
            ofs.write((char *)&dim, sizeof(int));
            ofs.write((char *)&ef_search, sizeof(int));
            ofs.write((char *)&limited_level, sizeof(int));
            ofs.write((char *)&seed, sizeof(unsigned int));

            // 写入 level 0 数据块
            ofs.write(data_level0_, cur_count * size_per_element_);

            // 写入高层链接数据
            for (size_t i = 0; i < cur_count; i++)
            {
                int link_level = element_levels_[i];
                ofs.write((char *)&link_level, sizeof(int));
                if (link_level > 0)
                {
                    size_t link_size = size_links_per_element_ * link_level;
                    ofs.write(link_lists_[i], link_size);
                }
            }

            return true;
        }
        catch (...)
        {
            return false;
        }
    }

    // ========================================================
    // load - 二进制加载（静态方法）
    // ========================================================
    static HNSWGraph *load(const std::string &filepath)
    {
        std::ifstream ifs(filepath, std::ios::binary);
        if (!ifs)
            throw std::runtime_error("failed to open binary file: " + filepath);

        size_t links_level0_size, max_elements, cur_count, size_per_element;
        size_t label_offset, data_offset;
        int max_level_val, entry_point_val, max_m, max_m0, m_val;
        double level_mult_val;
        int ef_c, dim_val, ef_s, lim;
        unsigned int seed_val;

        ifs.read((char *)&links_level0_size, sizeof(size_t));
        ifs.read((char *)&max_elements, sizeof(size_t));
        ifs.read((char *)&cur_count, sizeof(size_t));
        ifs.read((char *)&size_per_element, sizeof(size_t));
        ifs.read((char *)&label_offset, sizeof(size_t));
        ifs.read((char *)&data_offset, sizeof(size_t));
        ifs.read((char *)&max_level_val, sizeof(int));
        ifs.read((char *)&entry_point_val, sizeof(int));
        ifs.read((char *)&max_m, sizeof(int));
        ifs.read((char *)&max_m0, sizeof(int));
        ifs.read((char *)&m_val, sizeof(int));
        ifs.read((char *)&level_mult_val, sizeof(double));
        ifs.read((char *)&ef_c, sizeof(int));
        ifs.read((char *)&dim_val, sizeof(int));
        ifs.read((char *)&ef_s, sizeof(int));
        ifs.read((char *)&lim, sizeof(int));
        ifs.read((char *)&seed_val, sizeof(unsigned int));

        HNSWGraph<T> *g = new HNSWGraph<T>(dim_val, m_val, ef_c, ef_s, lim, max_elements, seed_val);
        g->max_level = max_level_val;
        g->entry_point_id = entry_point_val;
        g->cur_element_count_ = cur_count;

        // 读取 level 0 数据块
        ifs.read(g->data_level0_, cur_count * g->size_per_element_);

        // 读取高层链接数据并重建 label_lookup_
        for (size_t i = 0; i < cur_count; i++)
        {
            int link_level;
            ifs.read((char *)&link_level, sizeof(int));
            g->element_levels_[i] = link_level;

            if (link_level > 0)
            {
                size_t link_size = g->size_links_per_element_ * link_level;
                g->link_lists_[i] = (char *)malloc(link_size);
                if (!g->link_lists_[i])
                    throw std::runtime_error("HNSWGraph::load: failed to allocate link_lists_");
                ifs.read(g->link_lists_[i], link_size);
            }

            // 重建 label_lookup_
            int vid = g->get_external_label((int)i);
            g->label_lookup_[vid] = (int)i;
        }

        return g;
    }
};
