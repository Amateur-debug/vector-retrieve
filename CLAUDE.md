# CLAUDE.md — vector-retrieve

## 项目概述

向量检索研究项目，包含三个独立的检索系统用于对比实验：

1. **vector-retrieve（自研系统）** — `source/cpp/` + `source/python/`：基于 HNSW 的分布式近似最近邻搜索系统，支持 TCP/RDMA 通信，C++17 实现 + pybind11 Python 绑定
2. **CoTra** — `CoTra/`（git submodule）：RDMA 优化的分布式 ANN 系统，使用 memcached 协调，包含 DiskANN/hnswlib 第三方库
3. **Weaviate** — `weaviate/`：Weaviate 向量数据库的集成与基准测试

## 技术栈（自研系统）

- **核心语言**: C++17（算法）+ Python 3.10（编排调度）
- **构建系统**: CMake 3.12+，pybind11 提供 Python 绑定
- **通信协议**: TCP（标准）、RDMA（可选，需要 libibverbs/librdmacm）
- **SIMD 优化**: AVX-512、AVX2+FMA、AVX、SSE4.2（运行时自动检测）
- **Python 依赖**: numpy, scikit-learn, tqdm, pyyaml, matplotlib
- **数据格式**: BigANN 基准测试格式（.fvecs/.ivecs）

## 项目结构

```
source/cpp/          # 自研系统 C++ 核心：HNSW 图、集群管理、传输层、距离计算、SDK
source/python/       # 自研系统 Pure Python 回退实现
CoTra/               # 独立检索系统（git submodule）：RDMA 优化的分布式 ANN
weaviate/            # 独立检索系统：Weaviate 向量数据库集成
dataset/             # 共享测试数据集（siftsmall, sift, gist, deep）
build/               # CMake 构建输出（编译后的 Python 扩展模块）
plot/                # 可视化脚本
results/             # 实验结果
save/                # 保存的图索引
```

### 关键文件（自研系统）

- `main.py` — 主入口，实验运行脚本
- `batch.py` — 批量实验运行器，支持参数网格搜索
- `source/cpp/hnsw.h` — HNSW 图实现（~1400 行）
- `source/cpp/cluster.h` — 分布式集群/节点管理（~2700 行）
- `source/cpp/distance.h` — 距离度量，含 SIMD 优化
- `source/cpp/pybind.cpp` — Python 绑定（导出 HNSWGraphFloat/Double、ClusterFloat 等）
- `source/cpp/tcp_transport.h` / `rdma_transport.h` — 传输层实现
- `source/cpp/sdk.h` — 客户端 SDK，远程操作接口
- `CMakeLists.txt` — 主构建配置
- `environment.yaml` — Conda 环境配置

### 其他系统

- `run_cotra.py` — CoTra 分布式运行脚本
- `weaviate/` — Weaviate 基准测试脚本

## 构建命令

```bash
# 构建自研系统 C++ 扩展
mkdir build && cd build && cmake .. && make -j$(nproc)

# 构建 CoTra
cd CoTra && mkdir build && cd build && cmake .. && make -j$(nproc)

# 创建 Conda 环境
conda env create -f environment.yaml
```

## 运行命令

```bash
# 使用 C++ 后端运行
python main.py --use_cpp True

# 使用纯 Python 回退实现运行
python main.py --use_cpp False

# 批量实验
python batch.py

# CoTra 分布式运行
python run_cotra.py --node_ips <IP1,IP2,...> --index_save_dir <path>
```

## 核心类（自研系统 source/cpp/）

| 类名 | 文件 | 说明 |
|------|------|------|
| `HNSWGraph<T>` | `source/cpp/hnsw.h` | HNSW 图，连续内存布局 |
| `Cluster<T>` | `source/cpp/cluster.h` | 分布式集群，支持 K-means/随机分区 |
| `Payload<T>` | `source/cpp/payload.h` | 查询负载序列化，用于分布式搜索 |
| `TcpTransport` | `source/cpp/tcp_transport.h` | TCP 传输层，含连接池 |
| `RdmaTransport` | `source/cpp/rdma_transport.h` | RDMA 传输层，含 QP 握手 |
| `ClusterSDKClient<T>` | `source/cpp/sdk.h` | 客户端 SDK，远程集群操作 |
| `SpaceInterface<T>` | `source/cpp/distance.h` | 抽象距离度量接口（L2、内积） |

## 检索流程（自研系统）

### 数据分区

建图完成后，通过 `Cluster::partition()` 将完整 HNSW 图拆分到多个节点。支持以下分区模式：

- **round_robin**：按 vid 取模均匀分配到各节点
- **k-means**：对向量做 K-means 聚类，同一簇的向量分配到同一节点（空间局部性更好）
- **outer_ring**：先 K-means 分区，再迭代识别边界向量做冗余分配，减少跨节点传输
- **layer**：按 HNSW 层级分区，不同层可分配到不同节点集合，支持层级内 replicate（全复制）或 split（K-means/round_robin 子分区）

分区后每个节点持有局部 HNSW 图，远程邻居在邻居列表中编码为负数（`-(vid + 1)`），通过 `compact()` 完成。

### 单节点 HNSW 查询（`HNSWGraph::query`）

标准 HNSW 逐层搜索，从最高层到第 0 层：

1. **高层导航**（level > 0）：ef=1，贪心搜索找到每层最近点，作为下一层入口
2. **第 0 层精搜**：ef=max(ef_search, k)，候选集（最小堆）展开邻居，结果集（最大堆，容量 ef）维护当前最优
3. **搜索主循环**：从候选集弹出最近点 → 展开其邻居列表 → 计算距离 → 更新候选集和结果集 → 当候选集堆顶距离 > 结果集堆顶距离时终止
4. **预取优化**：搜索循环中使用 `_mm_prefetch` 对下一个邻居的 visited 数组和向量数据做预取

### 分布式跨节点搜索（Forward 模式）

当搜索过程中遇到远程向量时，通过 Payload 携带搜索状态在节点间转发：

1. **本地搜索阶段**（`search_layer_query`）：
   - 展开邻居时，本地向量直接计算距离
   - 远程向量（负数编码）收集到 `payload.pending_vids`
   - 搜索主循环中遇到无法展开的远程候选也触发转发

2. **跨节点转发**（`Cluster::transfer_to_remote`）：
   - 通过 `vector_node_map` 路由表查找目标 vid 所在节点
   - 将查询向量 + Payload（含 candidates、results、visited_vids、pending_vids）序列化发送到目标节点
   - 支持冗余分区下的随机负载均衡选择

3. **远程续搜**（`payload.continue_search = true`）：
   - 目标节点收到后恢复 visited 状态（vid → internal_id 映射）
   - 先处理 `pending_vids`：在本地的计算距离并入堆，不在本地的继续转发
   - 继续搜索主循环，可能再次触发转发到其他节点

4. **结果回传**（`send_result_to_origin`）：
   - 搜索完成（`continue_search = false`）后，将 top-k 结果回传给起始节点
   - 起始节点通过 `pending_queries` + 条件变量异步等待结果
   - 中间节点只转发不等待，形成链式异步转发

### Reduce 模式查询（`Cluster::query`，`is_reduce=true`）

并行查询所有节点并合并结果：

1. 对每个节点并行发起独立查询（`std::async`），每个节点使用独立 Payload
2. 远程节点发送 `MSG_QUERY_LOCAL` 消息，节点只执行本地 HNSW 搜索（避免递归路由）
3. 收集所有节点返回的 (distance, vid) 结果，全局排序取 top-k
4. 汇总各节点的 distance_cnt、remote_transfer_cnt 等统计信息

### SDK 客户端查询（`ClusterSDKClient::query`）

外部客户端通过 SDK 发起查询：

1. 从 `query_endpoints` 中随机选择一个端点（负载均衡）
2. 发送 `MSG_QUERY` 消息（含 query_id、k、query_vector、payload）
3. 同步等待响应，反序列化得到结果 ID 列表和更新后的 payload 统计信息

## 代码规范

### C++
- **风格**: K&R 大括号风格，2–4 空格缩进
- **命名**: 变量/函数用 `snake_case`，类用 `PascalCase`，枚举/宏用 `UPPER_SNAKE_CASE`
- **私有成员**: 尾部下划线（如 `data_level0_`、`link_locks_`）
- **模板参数**: `T` 为类型参数，特化为 float/double
- **注释**: 中英双语，特别是内存布局和同步机制部分
- **头文件**: 使用 `#pragma once`，引入顺序为 系统头文件 → 项目库 → 本地头文件

### Python
- **风格**: PEP-8，4 空格缩进
- **命名**: 函数/变量用 `snake_case`，类用 `PascalCase`，常量用 `UPPER_SNAKE_CASE`
- **私有方法**: 前导下划线（`_private_method`）
- **引入顺序**: 标准库 → 第三方库 → 本地模块

## 架构要点

- **连续内存布局**：HNSW 图采用连续内存块存储，优化缓存命中率（参考 hnswlib 设计）
- **SIMD 自动检测**：运行时按优先级选择 AVX-512 → AVX2 → AVX → SSE → 标量内核
- **连接池**：TCP 和 RDMA 传输层均使用连接池管理
- **访问列表池**：基于计数器的 O(1) 重置，避免查询热路径中的内存分配
- **GIL 释放**：pybind11 绑定中使用 `py::call_guard<py::gil_scoped_release>()`，实现 C++ 层真正的并行执行
- **主从分布式模式**：主节点（ID=0）协调，工作节点服务本地 HNSW 分片
- **读写锁**：`shared_mutex` 保护图扩容操作；每节点独立互斥锁保护邻居列表更新
