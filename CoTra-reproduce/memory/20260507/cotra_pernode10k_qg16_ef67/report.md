# CoTra per-node 10K query test with query group size 16

## 结论

本轮已按“每个节点 10K query”修改查询规模并完成测试。由于当前 `ScalaANN_v3` 的查询划分是 `query_load = qsize / MACHINE_NUM`，8 节点时需要设置全局 `-Q 80000`，每个 rank 实际注入 `10000` 条 query。

结果：`QUERY_GROUP_SIZE=16`、8 nodes x 8 threads、SIFT100M、`ef=67` 下，leader 聚合结果为 recall `0.90090`，system QPS `45830.277`。

## 固定配置

- 代码目录：`/hpc2hdd/home/rmeng603/workspace/CoTra`
- 测试目录：`/hpc2hdd/home/rmeng603/workspace/CoTra/memory/20260507/cotra_pernode10k_qg16_ef67`
- Slurm job：`9722173`
- 节点：`cpu1-90 cpu1-91 cpu1-92 cpu1-93 cpu1-94 cpu1-95 cpu1-96 cpu1-97`
- `MACHINE_NUM=8`
- `QUERY_GROUP_SIZE=16`
- `COTRA_QUERY_GROUP_SIZE=16`
- `QUERY_COUNT=80000`
- `PER_NODE_QUERY_COUNT=10000`
- `SEARCH_THREADS=8` per rank
- `COTRA_EF_LIST=66,67`，其中 `66` 用作 warmup，`67` 作为测量点
- `COTRA_SKIP_RECALL` 未启用，recall 正常计算

## 缓存与输入

- 复用已有 SIFT100M 8 节点 CoTra 索引：`memory/20260424/sift100m_8node_experiment/results/index/merged_index*`
- 未重新建图
- 复用已有 640K amplified query/GT：`memory/20260507/cotra_integrated_640k_query_count_ef67/data/queries_640k.u8bin` 与 `gt_100_640k_ids_only.bin`
- 本轮只读取前 `80000` 条 query/GT

## 代码调整

- `include/anns/anns_config.h`：将 `QUERY_GROUP_SIZE` 固定为 `16`
- `src/rdma/rdma_comm.cc`：将 `wr_id` 生成中的 `query_id << 16` 改为先转成 `uint64_t` 再左移，避免 `qid >= 65536` 时 32-bit 溢出导致大 batch 卡住

## 结果

Leader 聚合指标：

| query_count | per_node_query_count | query_group_size | ef | recall | system_qps |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 80000 | 10000 | 16 | 67 | 0.90090 | 45830.277 |

Rank 日志摘要：

| rank | host | ef | recall | qps | memory_mb |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | cpu1-90 | 67 | 0.90090 | 45830.277 | 17660 |
| 1 | cpu1-91 | 67 | 0.90054 | 5492.800 | 18086 |
| 2 | cpu1-92 | 67 | 0.90071 | 13428.700 | 16906 |
| 3 | cpu1-93 | 67 | 0.90063 | 6049.940 | 18068 |
| 4 | cpu1-94 | 67 | 0.90041 | 6578.470 | 17470 |
| 5 | cpu1-95 | 67 | 0.90134 | 6041.760 | 17845 |
| 6 | cpu1-96 | 67 | 0.90114 | 5603.880 | 17807 |
| 7 | cpu1-97 | 67 | 0.90098 | 5942.040 | 18207 |

注意：rank 0 的输出是 leader 聚合 QPS；rank 1-7 的输出是各自本地视角，不应直接与 leader 聚合值等价相加。

## 与修复前现象的关系

此前同类 sweep 中，`Q=65536` 可完成，而 `Q=66000/70000/80000` 会超时或卡住。代码检查发现 RDMA read completion 的 `wr_id` 用 `(query_id << 16) | buffer_id` 编码，但 `query_id` 是 `uint32_t`，当 `query_id >= 65536` 时左移 16 位会在 32-bit 中溢出。本轮修复后，`Q=80000` 可以正常完成，说明这个溢出是大 batch 卡住的关键原因之一。

## 输出文件

- 聚合结果：`results/pernode10k_qg16_summary.tsv`
- rank 级结果：`results/per_rank_metrics.tsv`
- 派生指标：`results/derived_metrics.tsv`
- Slurm 设置：`results/run_9722173_settings.txt`
- leader 日志：`logs/anns_q80000_qg16_job9722173_cpu1-90_0.log`
- 完整命令记录：`commands_used.txt`
- 错误记录：`error_log.md`
