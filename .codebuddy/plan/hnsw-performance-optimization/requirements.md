# 需求文档：HNSW 性能优化（参考 hnswlib）

## 引言

当前 `vector-retrieve` 项目中的 `HNSWGraph` 实现（`source/cpp/hnsw.h`）在功能上完整，但与 hnswlib（`CoTra/third_party/hnswlib/hnswlib/hnswalg.h`）在性能上存在显著差距。核心瓶颈包括：

1. **内存布局**：使用 `std::unordered_map<int, std::vector<T>>` 存储向量，`unordered_map<int, unordered_map<int, unordered_set<int>>>` 存储邻居关系，导致大量 cache miss 和随机内存访问
2. **锁开销**：搜索路径上每访问一个邻居都需要 `shared_lock`（约 1700 次/查询），即使搜索是只读操作
3. **数据拷贝**：搜索和建图时频繁深拷贝整个向量（`neighbor_vec = it->second`）和邻居集合（`neighbor_ids = it->second`）
4. **邻居列表**：使用 `unordered_set<int>` 存储邻居，遍历有额外哈希开销，且邻居数量无严格上限控制
5. **距离函数**：编译期静态选择 SIMD 实现，不支持运行时根据 CPU 能力动态选择最优实现
6. **Visited 集合**：使用 Payload 内嵌的 `unordered_set`，无法复用，不适合高并发场景
7. **序列化**：使用 JSON 格式，序列化/反序列化极慢，文件体积大

**参考实现**：hnswlib (`CoTra/third_party/hnswlib/hnswlib/`) 的核心设计要点：
- `data_level0_memory_`: 单块连续内存存储 level 0 数据，每个元素的内存布局为 `[links(maxM0*4+4 bytes) | vector_data(dim*4 bytes) | label(8 bytes)]`
- `linkLists_`: 指针数组，每个高层节点单独 malloc 其 level 1+ 的邻居列表
- `link_list_locks_`: 预分配 `max_elements` 个 `std::mutex`，写邻居时只锁目标节点
- `searchBaseLayerST()`: query 搜索函数，**完全不加任何 mutex 锁**（仅使用 VisitedListPool）
- `searchBaseLayer()`: 建图搜索函数，读邻居时加 `link_list_locks_[curNodeNum]`（保证看到完整的邻居列表更新）
- `VisitedListPool`: 池化管理的 visited 数组，O(1) reset，多线程通过池分配避免竞争
- `DISTFUNC<dist_t>`: 函数指针类型，运行时根据 CPU 能力（AVX512/AVX/SSE）动态绑定最优距离函数
- `_mm_prefetch`: 在搜索循环中对下一个邻居的 visited 数组、向量数据进行 CPU 预取
- 二进制 save/load：直接 write/read `data_level0_memory_` 整块内存

本次优化目标是**参考 hnswlib 的做法**，将 `HNSWGraph` 改造为上述架构，同时保持与现有 `Cluster`、`Payload`、`pybind` 接口的兼容性。

---

## 命名规范

本次优化 **不改变现有函数、类、变量的名字**。新增的函数和变量须遵循现有代码的命名风格：

| 类别 | 风格 | 现有示例 |
|------|------|---------|
| 类名 | 大驼峰 | `HNSWGraph`, `Payload`, `VisitedListPool` |
| 成员变量 | 全小写 + 下划线 | `dim`, `m`, `ef_construction`, `ef_search`, `max_level`, `entry_point_id` |
| 锁变量 | `_mutex` 后缀 | `vectors_mutex`, `neighbors_mutex`, `entry_mutex` |
| 函数名 | 全小写 + 下划线 | `add_vector`, `search_layer_ef`, `random_level`, `select_neighbors_heuristic` |
| 模板参数 | 单字母大写 | `T` |

**新增命名映射**（hnswlib 名称 → 本项目名称）：

| hnswlib 命名 | 本项目新增命名 | 说明 |
|-------------|--------------|------|
| `data_level0_memory_` | `data_level0_` | level 0 连续内存块 |
| `linkLists_` | `link_lists_` | 高层邻居指针数组 |
| `link_list_locks_` | `link_locks_` | 节点级锁数组 |
| `size_data_per_element_` | `size_per_element_` | 每个元素占用字节数 |
| `size_links_per_element_` | `size_links_per_element_` | 每层链接占用字节数 |
| `label_offset_` | `label_offset_` | label 在元素内的偏移 |
| `offsetData_` | `data_offset_` | 向量数据在元素内的偏移 |
| `offsetLevel0_` | `links_level0_size_` | level 0 链接区域大小 |
| `maxM_` | `max_m_` | 高层最大邻居数（= M） |
| `maxM0_` | `max_m0_` | level 0 最大邻居数（= 2*M） |
| `mult_` | `level_mult_` | 层级生成乘数 |
| `ef_construction_` | `ef_construction`（已有） | 保持不变 |
| `enterpoint_node_` | `entry_point_id`（已有） | 保持不变 |
| `maxlevel_` | `max_level`（已有） | 保持不变 |
| `cur_element_count` | `cur_element_count_` | 当前元素数量（atomic） |
| `max_elements_` | `max_elements_` | 最大元素容量 |
| `label_lookup_` | `label_lookup_` | vid → internal_id 映射 |
| `visited_list_pool_` | `visited_pool_` | VisitedListPool 实例 |
| `getDataByInternalId()` | `get_data_by_id()` | 根据 internal_id 获取向量指针 |
| `getExternalLabel()` | `get_external_label()` | 根据 internal_id 获取 vid |
| `get_linklist0()` | `get_linklist0()` | 获取 level 0 邻居列表指针 |
| `get_linklist()` | `get_linklist()` | 获取高层邻居列表指针 |
| `getListCount()` | `get_list_count()` | 获取邻居列表长度 |
| `setListCount()` | `set_list_count()` | 设置邻居列表长度 |
| `searchBaseLayer()` | `search_layer_build()` | 建图搜索（加节点锁） |
| `searchBaseLayerST()` | `search_layer_query()` | query 搜索（完全无锁） |
| `mutuallyConnectNewElement()` | `connect_new_element()` | 建立双向连接 |
| `getNeighborsByHeuristic2()` | `select_neighbors_heuristic()`（已有，重构） | 启发式邻居选择 |
| `addPoint()` | `add_vector()`（已有，重构） | 添加向量 |
| `searchKnn()` | `query()`（已有，重构） | K 近邻搜索 |
| `saveIndex()` | `save()`（已有，重构） | 保存索引 |
| `loadIndex()` | `load()`（已有，重构） | 加载索引 |
| `resizeIndex()` | `resize_index()` | 动态扩容 |
| `getRandomLevel()` | `random_level()`（已有） | 保持不变 |
| `fstdistfunc_` | `dist_func_` | 距离函数指针 |
| `dist_func_param_` | `dist_func_param_` | 距离函数参数 |
| `SpaceInterface` | `SpaceInterface` | 距离空间抽象基类（沿用 hnswlib 命名，因为是独立接口类） |
| `DISTFUNC` | `DISTFUNC` | 距离函数指针类型定义（沿用 hnswlib 命名，因为是 typedef） |
| `VisitedList` | `VisitedList` | visited 列表（沿用，类名大驼峰符合规范） |
| `VisitedListPool` | `VisitedListPool` | visited 列表池（沿用，类名大驼峰符合规范） |

---

## 需求

### 需求 1：连续内存布局重构

**用户故事：** 作为一名开发者，我希望向量数据和邻居列表存储在预分配的连续内存中（参考 hnswlib 的 `data_level0_memory_` 设计），以便 CPU 缓存预取能高效工作，大幅减少随机内存访问带来的 cache miss。

#### 验收标准

1. WHEN `HNSWGraph` 初始化时指定 `max_elements_` 参数 THEN 系统 SHALL 一次性 `malloc` 分配 level 0 的连续内存块 `data_level0_`，大小为 `max_elements_ * size_per_element_`，其中每个元素的内存布局为 `[links_level0 | vector_data | external_label]`，即：
   - `links_level0`: `(max_m0_ * sizeof(int) + sizeof(int))` 字节，存储 level 0 的邻居列表（第一个 int 为邻居数量 count，后续为邻居的 internal_id）
   - `vector_data`: `dim * sizeof(T)` 字节，存储向量数据
   - `external_label`: `sizeof(int)` 字节，存储外部 ID (vid)
2. WHEN 通过 `add_vector(vid, vec)` 添加向量 THEN 系统 SHALL 递增 `cur_element_count_` 获取 `internal_id`，并将向量数据通过 `memcpy` 拷贝到 `data_level0_ + internal_id * size_per_element_ + data_offset_` 的位置
3. WHEN 系统需要访问某个节点的向量数据 THEN 系统 SHALL 通过 `get_data_by_id(internal_id)` 返回 `char*` 指针（即 `data_level0_ + internal_id * size_per_element_ + data_offset_`），直接偏移到连续内存中的对应位置，不进行任何数据拷贝
4. WHEN 系统存储高层（level > 0）邻居列表 THEN 系统 SHALL 维护一个指针数组 `link_lists_`（`char**`），每个有高层的节点在 `add_vector` 时单独 `malloc` 分配 `size_links_per_element_ * level` 字节，其中 `size_links_per_element_ = max_m_ * sizeof(int) + sizeof(int)`
5. WHEN `vid`（外部 ID）和 `internal_id`（内部 ID）之间需要映射 THEN 系统 SHALL 维护 `label_lookup_`（`unordered_map<int, int>`）用于外部 ID → 内部 ID 查找，外部 label 同时存储在 `data_level0_` 中用于内部 ID → 外部 ID 查找
6. WHEN 当前元素数量达到 `max_elements_` THEN 系统 SHALL 支持 `resize_index(new_max_elements)` 动态扩容（`realloc` data_level0_ 和 link_lists_，重新分配锁数组等）

---

### 需求 2：多线程建图读取无锁 + 写入节点级锁

**用户故事：** 作为一名开发者，我希望多线程并行建图时，读取向量数据完全无锁，读取邻居列表在建图搜索时加节点级细粒度锁（参考 hnswlib 的 `searchBaseLayer` 策略），以便减少锁争抢开销，提升建图吞吐。

#### 验收标准

1. WHEN 多线程并行执行 `add_vector()` THEN 读取已有节点的向量数据（`get_data_by_id`）SHALL 不需要任何锁，因为连续内存布局保证了地址稳定性
2. WHEN 建图中调用 `search_layer_build()` 遍历邻居 THEN 系统 SHALL 对当前候选节点加 `link_locks_[cur_node_num]` 锁后再读取其邻居列表（参考 hnswlib `searchBaseLayer` 实现），保证看到完整的邻居列表更新
3. WHEN 建图中 `connect_new_element()` 需要写入某个节点的邻居列表 THEN 系统 SHALL 使用 `link_locks_[node_id]`（预分配 `max_elements_` 个 `std::mutex`），只锁住目标节点
4. WHEN 建图线程写入新节点自身的邻居列表 THEN 系统 SHALL 不需要加锁（因为新节点尚未被其他线程发现）
5. WHEN 建图中需要更新反向连接（bidirectional link）的邻居节点 THEN 系统 SHALL 逐个加 `link_locks_[neighbor_id]` 写入，释放后再处理下一个
6. WHEN 多线程建图时需要修改 `entry_point_id` 和 `max_level` THEN 系统 SHALL 使用全局 `std::mutex global_mutex_` 保护
7. WHEN 分配新的 `internal_id` THEN 系统 SHALL 使用 `std::atomic<size_t> cur_element_count_` 在全局锁保护区域内递增（参考 hnswlib `addPoint` 中的实现：在 `label_lookup_lock` 内递增后解锁）

---

### 需求 3：并行 Query 完全无锁

**用户故事：** 作为一名用户，我希望多个查询线程能完全并行执行而不需要任何 mutex 锁（参考 hnswlib 的 `searchBaseLayerST` 和 `searchKnn`），以便在高并发搜索场景下获得线性扩展的吞吐。

#### 验收标准

1. WHEN 多线程并行执行 `query()` THEN 整个搜索路径 SHALL 不包含任何 `std::mutex` 或 `shared_lock`
2. WHEN query 的 `search_layer_query()` 读取邻居列表 THEN 系统 SHALL 直接读取 `get_linklist0(current_node_id)` 返回的定长数组，不加任何锁（参考 hnswlib `searchBaseLayerST` 实现，搜索只读不加 `link_locks_`）
3. WHEN query 读取向量数据 THEN 系统 SHALL 通过 `get_data_by_id` 指针直接访问连续内存，不加锁、不拷贝
4. WHEN query 需要 visited 标记 THEN 系统 SHALL 从 `visited_pool_` 中获取一个 `VisitedList` 对象（每个查询独享），查询结束后归还池中
5. WHEN query 需要读取 `entry_point_id` 和 `max_level` THEN 系统 SHALL 直接读取（这些值在 query-only 阶段不会被修改）

---

### 需求 4：消除搜索路径上的数据拷贝

**用户故事：** 作为一名开发者，我希望搜索路径上不进行任何向量或邻居列表的深拷贝，以便减少堆内存分配和数据复制的开销。

#### 验收标准

1. WHEN `search_layer_build()` 或 `search_layer_query()` 需要计算距离 THEN 系统 SHALL 直接使用 `char* data_ptr = get_data_by_id(candidate_id)` 获取指针，传给距离函数 `dist_func_(data_point, data_ptr, dist_func_param_)`，不进行 `std::vector<T>` 的拷贝
2. WHEN 搜索函数需要遍历邻居 THEN 系统 SHALL 通过 `int* data = (int*)get_linklist0(cur_node)` 直接获取邻居列表指针，`get_list_count(data)` 获取数量，`(int*)(data+1)` 获取邻居 ID 数组，不进行 `unordered_set` 的拷贝
3. WHEN `select_neighbors_heuristic()` 需要计算候选者间距离 THEN 系统 SHALL 直接使用 `get_data_by_id()` 返回指针，零拷贝零加锁
4. WHEN `connect_new_element()` 中检查反向连接是否已满并需要运行启发式选择 THEN 系统 SHALL 直接使用指针访问向量数据，不进行批量预读拷贝到 map

---

### 需求 5：定长邻居数组

**用户故事：** 作为一名开发者，我希望邻居列表使用固定大小的 int 数组代替 `unordered_set<int>`（参考 hnswlib 的 `linklistsizeint` + `tableint[]` 布局），以便邻居遍历变成顺序内存访问，提升 cache 效率。

#### 验收标准

1. WHEN 存储 level 0 的邻居列表 THEN 系统 SHALL 使用 `[int count | int neighbor_0 | ... | int neighbor_{max_m0_-1}]` 格式，内嵌在 `data_level0_` 中，最大邻居数为 `max_m0_ = 2 * m`
2. WHEN 存储 level > 0 的邻居列表 THEN 系统 SHALL 使用 `[int count | int neighbor_0 | ... | int neighbor_{max_m_-1}]` 格式，存储在 `link_lists_[internal_id]` 指向的独立内存中，最大邻居数为 `max_m_ = m`
3. WHEN `connect_new_element()` 需要为邻居节点添加反向连接 AND 该邻居的连接数 < max_cur THEN 系统 SHALL 直接在数组末尾追加新 ID，并更新 count
4. WHEN 邻居列表已满（连接数 >= max_cur）且需要插入新邻居 THEN 系统 SHALL 将 {现有邻居 ∪ 新节点} 作为候选集运行 `select_neighbors_heuristic()`，将结果直接写回定长数组
5. WHEN 遍历邻居 THEN 系统 SHALL 通过 `int size = get_list_count(data); int* datal = data + 1; for (j=0; j<size; j++) candidate = datal[j];` 顺序遍历

---

### 需求 6：优化 select_neighbors_heuristic（对齐 hnswlib getNeighborsByHeuristic2）

**用户故事：** 作为一名开发者，我希望启发式邻居选择算法对齐 hnswlib 的 `getNeighborsByHeuristic2` 实现，减少不必要的数据结构拷贝。

#### 验收标准

1. WHEN `select_neighbors_heuristic()` 被调用 THEN 系统 SHALL 接收一个 `priority_queue<pair<T, int>>` 引用（max-heap，距离为正），内部将其转为 min-heap（距离取负）按从近到远顺序处理
2. WHEN 执行启发式选择 THEN 系统 SHALL 将结果写回同一个 `priority_queue` 引用（清空后重新填入选中的邻居），便于调用方直接使用
3. WHEN 计算候选者之间的距离 THEN 系统 SHALL 直接通过 `dist_func_(get_data_by_id(id1), get_data_by_id(id2), dist_func_param_)` 计算，零拷贝零加锁
4. WHEN 候选者数量 ≤ m THEN 系统 SHALL 直接返回，不运行启发式筛选（与 hnswlib 一致的早退优化）

---

### 需求 7：VisitedListPool 替代 Payload.visited

**用户故事：** 作为一名开发者，我希望用 `VisitedListPool`（池化管理的 visited 数组）替代当前 Payload 中的 `unordered_set<int>` visited 集合（参考 hnswlib 的 `visited_list_pool.h`），以便 visited 标记的检查和重置都是 O(1) 操作，且多线程友好。

#### 验收标准

1. WHEN `HNSWGraph` 初始化 THEN 系统 SHALL 创建 `visited_pool_`（`VisitedListPool(num_initial_pools, max_elements_)`），其中每个 `VisitedList` 是一个大小为 `max_elements_` 的 `unsigned short` 数组
2. WHEN 搜索函数（`search_layer_build` / `search_layer_query`）开始 THEN 系统 SHALL 调用 `visited_pool_->get_free_list()` 获取一个 VisitedList，搜索结束后调用 `release_list()` 归还
3. WHEN 检查节点是否已访问 THEN 系统 SHALL 通过 `visited_array[candidate_id] == visited_tag` 判断（O(1) 数组随机访问），替代 `unordered_set::find()`
4. WHEN 标记节点为已访问 THEN 系统 SHALL 通过 `visited_array[candidate_id] = visited_tag` 赋值（O(1)）
5. WHEN VisitedList 被归还池中并再次取出时 THEN 系统 SHALL 通过递增 `cur_tag` 标签实现 O(1) 逻辑清空，当 `cur_tag` 溢出时 `memset` 物理清零
6. WHEN 当前 `Payload` 结构中的 `visited` 字段 THEN 系统 SHALL 将其类型从 `Visited`（基于 epoch 的数组）改为 `std::vector<int> visited_vids`，仅存储 vid 列表，专用于跨节点序列化传输。搜索内部的 visited 逻辑完全由 `VisitedListPool` 管理（基于 `internal_id`），不再使用 `Payload.visited`
7. WHEN 搜索完成并需要序列化 Payload 用于跨节点传输 THEN 系统 SHALL 将 `VisitedList` 中已标记的 `internal_id` 通过 `get_external_label()` 转换为 vid，存入 `payload.visited_vids`
8. WHEN 接收方反序列化 Payload 后续搜 THEN 系统 SHALL 从 `payload.visited_vids` 中取出 vid 列表，通过 `label_lookup_` 转为本地 `internal_id`，标记到从 `VisitedListPool` 获取的新 `VisitedList` 中
9. WHEN 旧的 `Visited` 类（visited.h） THEN 系统 SHALL 移除该类，用 `VisitedListPool` 完全替代

---

### 需求 8：距离函数指针化 + 运行时 SIMD 动态选择

**用户故事：** 作为一名开发者，我希望距离函数使用函数指针（参考 hnswlib 的 `DISTFUNC<dist_t>` 和 `SpaceInterface`），并在运行时根据 CPU 能力动态选择最优 SIMD 实现，以便在不同硬件上都能获得最佳性能。

#### 验收标准

1. WHEN 系统初始化距离计算 THEN 系统 SHALL 定义 `DISTFUNC` 函数指针类型 `T(*)(const void*, const void*, const void*)`，并在 `HNSWGraph` 中存储 `dist_func_` 成员和 `dist_func_param_`（指向 dim 的指针）
2. WHEN 构造 L2Space 或 InnerProductSpace THEN 系统 SHALL 在运行时检测 CPU 是否支持 AVX-512 / AVX / SSE（通过 `cpuid` 指令），动态绑定最优的 SIMD 距离函数（参考 hnswlib `L2Space::L2Space()` 中的 `AVX512Capable()` / `AVXCapable()` 检测逻辑）
3. WHEN 搜索和建图中计算距离 THEN 系统 SHALL 通过 `dist_func_(data_point, get_data_by_id(id), dist_func_param_)` 调用，单次函数指针间接调用（无 if-else 分支）
4. WHEN 维度 `dim % 16 == 0` THEN 系统 SHALL 选择 16-float 对齐版本; WHEN `dim % 4 == 0` THEN 选择 4-float 版本; 否则使用带 residual 处理的版本

---

### 需求 9：原子操作替代全局锁 + thread_local RNG

**用户故事：** 作为一名开发者，我希望用原子操作替代部分全局互斥锁，并使用 thread_local RNG 避免随机数生成器的锁争抢。

#### 验收标准

1. WHEN 分配新元素的 `internal_id` THEN 系统 SHALL 使用 `std::atomic<size_t> cur_element_count_` 在全局锁保护区域内递增（参考 hnswlib `addPoint` 中的实现：在 `label_lookup_lock` 内递增后解锁）
2. WHEN 建图过程中需要更新 `max_level` 和 `entry_point_id` THEN 系统 SHALL 使用 `std::mutex global_mutex_` 保护（参考 hnswlib 实现：只在 `curlevel > maxlevelcopy` 时持有全局锁直到建边完成）
3. WHEN 随机数生成器生成层级 THEN 系统 SHALL 在全局锁保护区域内调用 `random_level()`，或使用 `thread_local` RNG 避免争抢

---

### 需求 10：save/load 二进制序列化

**用户故事：** 作为一名用户，我希望优化后的 HNSWGraph 支持高效的二进制 save/load 功能（参考 hnswlib 的 `saveIndex`/`loadIndex`），以便快速保存和加载索引。

#### 验收标准

1. WHEN 调用 `save(filepath)` THEN 系统 SHALL 按以下顺序写入二进制文件：
   - 参数头：`links_level0_size_, max_elements_, cur_element_count_, size_per_element_, label_offset_, data_offset_, max_level, entry_point_id, max_m_, max_m0_, m, level_mult_, ef_construction`
   - level 0 数据块：直接 `write(data_level0_, cur_element_count_ * size_per_element_)`
   - 高层链接：逐节点写入 `link_list_size` + `link_lists_[i]` 数据
2. WHEN 调用 `load(filepath)` THEN 系统 SHALL 从二进制文件还原：读取参数头 → malloc 分配 `data_level0_` → 读取 level 0 数据 → 逐节点恢复 `link_lists_` → 重建 `label_lookup_` 映射 → 初始化锁和 `visited_pool_`
3. WHEN 保存二进制格式 THEN 文件体积 SHALL 比 JSON 格式减少 90% 以上，load 速度 SHALL 提升 10 倍以上
4. WHEN 旧的 JSON 格式 save/load 代码 THEN 系统 SHALL 直接移除，不需要向后兼容

---

### 需求 11：搜索循环预取优化

**用户故事：** 作为一名开发者，我希望在搜索循环中加入 CPU 预取指令（参考 hnswlib `searchBaseLayerST` 中的 `_mm_prefetch` 使用），以便在处理当前邻居时提前将下一个邻居的数据加载到缓存中，减少内存延迟。

#### 验收标准

1. WHEN `search_layer_query()` 遍历邻居列表 THEN 系统 SHALL 在处理第 j 个邻居之前，对第 j+1 个邻居执行以下预取（参考 hnswlib 实现）：
   - `_mm_prefetch((char*)(visited_array + *(data + j + 1)), _MM_HINT_T0)` — 预取 visited 标记
   - `_mm_prefetch(data_level0_ + (*(data + j + 1)) * size_per_element_ + data_offset_, _MM_HINT_T0)` — 预取向量数据
2. WHEN 搜索进入邻居循环前 THEN 系统 SHALL 预取第一个和第二个邻居的 visited 标记和向量数据
3. WHEN 候选者被加入 candidate_set THEN 系统 SHALL 预取 candidate_set 堆顶元素对应的 level 0 数据（下一轮循环将从堆顶开始处理）

---

### 需求 12：Payload 跨节点序列化的 ID 一致性

**用户故事：** 作为一名开发者，我希望 Payload 在跨节点序列化传输时，所有 ID 字段都使用全局唯一的 `vid`（外部 ID），以便不同节点之间能正确理解和使用这些 ID（每个节点的 `internal_id` 从 0 开始递增，跨节点无意义）。

#### 背景

引入 `internal_id` / `vid` 双 ID 体系后，Payload 中有 5 个位置存储 ID：
- `candidates`：`priority_queue<pair<T, int>>`，搜索候选集
- `results`：`priority_queue<pair<T, int>>`，搜索结果集
- `visited`：已访问节点集合
- `require_id`：跨节点转发时需要的目标向量 ID
- `entry_point_id`：续搜时的入口点 ID

在当前实现中，这些 ID 都是 `vid`（全局唯一），跨节点传输没有问题。改造为连续内存布局后，`HNSWGraph` 内部搜索使用 `internal_id`（节点本地、从 0 开始），但 Payload 跨节点传输时必须使用 `vid`，否则节点 A 的 `internal_id=3` 在节点 B 上指向完全不同的向量。

#### 验收标准

1. WHEN `HNSWGraph` 内部搜索（`search_layer_build` / `search_layer_query`）操作 `candidates` 和 `results` THEN 系统内部 SHALL 使用 `internal_id` 作为 pair 的 second（用于通过 `get_data_by_id` 直接访问连续内存）
2. WHEN `query()` 函数返回最终结果前 THEN 系统 SHALL 将 `payload.results` 和 `payload.candidates` 中的 `internal_id` 全部转换为 `vid`（通过 `get_external_label(internal_id)`），保证 Payload 对外暴露的 ID 始终是 `vid`
3. WHEN `search_layer_ef()` 检测到需要跨节点跳转（`payload.continue_search = true`）THEN `payload.require_id` SHALL 设置为 `vid`（通过 `get_external_label(neighbor_internal_id)` 转换），因为 `transfer_to_remote()` 使用 `require_id` 在 `vector_node_map`（key 为 vid）中查路由
4. WHEN 设置 `payload.entry_point_id` 用于跨节点续搜 THEN 该值 SHALL 为 `vid`（外部 ID），接收方节点通过 `label_lookup_` 将其转换为本地 `internal_id` 后再使用
5. WHEN `serialize_payload()` 序列化 Payload 中的 `candidates`、`results`、`visited` THEN 这些集合中的 ID SHALL 已经是 `vid`（在序列化前由 `query()` 完成转换），序列化逻辑本身无需修改
6. WHEN `deserialize_payload()` 反序列化 Payload 后续搜 THEN 接收方节点在将 `candidates`/`results` 中的 `vid` 传入 `search_layer_ef()` 之前 SHALL 将其转换为本地 `internal_id`（通过 `label_lookup_`），如果某个 `vid` 在本地不存在则保持 `continue_search` 状态
7. WHEN `Payload` 需要跨节点传输已访问信息 THEN `payload.visited_vids`（`std::vector<int>`）SHALL 存储 vid 列表（外部 ID），由发送方在序列化前从 `VisitedList`（基于 `internal_id`）转换而来，接收方反序列化后再转回本地 `internal_id` 标记到新的 `VisitedList` 中。不再保留旧的 `Visited` 类

---

### 需求 13：构造函数与接口兼容

**用户故事：** 作为一名用户，我希望优化后的 HNSWGraph 保持与现有 Python 接口（pybind11）和 Cluster 系统的兼容性，以便无缝切换。

#### 验收标准

1. WHEN 创建 `HNSWGraph` THEN 构造函数 SHALL 新增 `max_elements` 参数，签名变为 `HNSWGraph(dim, m, ef_construction, ef_search, limited_level, max_elements, seed)`
2. WHEN Python 侧调用 `add_vector(vid, vec)` THEN 接口签名 SHALL 保持不变，内部自动完成 `vid` → `internal_id` 映射（通过 `label_lookup_`）
3. WHEN Python 侧调用 `query(query_vector, payload, k)` THEN 接口签名 SHALL 保持不变，返回的 ID 列表 SHALL 使用外部 `vid`（通过 `get_external_label(internal_id)` 转换）
4. WHEN Python 侧访问 `vectors` 和 `neighbors` 属性 THEN pybind 接口 SHALL 提供兼容的访问方式（例如返回字典视图），或调整属性并更新上层 Python 调用
5. WHEN `Cluster` 系统调用 `HNSWGraph` THEN 所有现有接口（`add_vector`、`query`、`save`、`load`）SHALL 保持功能兼容

---

## 附录：hnswlib 关键源码参考路径

| 文件 | 内容 |
|------|------|
| `CoTra/third_party/hnswlib/hnswlib/hnswalg.h` | 核心算法：内存布局、searchBaseLayer、searchBaseLayerST、mutuallyConnectNewElement、addPoint、saveIndex/loadIndex |
| `CoTra/third_party/hnswlib/hnswlib/hnswlib.h` | 基础定义：DISTFUNC 类型、SpaceInterface 抽象类、AlgorithmInterface |
| `CoTra/third_party/hnswlib/hnswlib/space_l2.h` | L2 距离计算：运行时 SIMD 动态选择（AVX512Capable / AVXCapable 检测） |
| `CoTra/third_party/hnswlib/hnswlib/space_ip.h` | 内积距离计算：运行时 SIMD 动态选择 |
| `CoTra/third_party/hnswlib/hnswlib/visited_list_pool.h` | VisitedListPool：池化管理的 visited 数组，O(1) reset |
