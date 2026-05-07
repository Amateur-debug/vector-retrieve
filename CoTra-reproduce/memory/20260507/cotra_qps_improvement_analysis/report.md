# CoTra QPS improvement analysis

Date: 2026-05-07 Asia/Shanghai
Repo: `/hpc2hdd/home/rmeng603/workspace/CoTra`

## Goal

Review the current CoTra implementation and identify the highest-leverage ways to improve QPS, using existing cached-index and prior test records instead of rebuilding the graph.

## Method

Read current source locations:

- `include/anns/anns_config.h`
- `include/anns/exec_query.h`
- `include/anns/scala_search.h`
- `include/anns/query_msg.h`
- `include/rdma/rdma_comm.h`
- `src/rdma/rdma_comm.cc`

Read prior memory results:

- `memory/20260430/perf_breakdown/report.md`
- `memory/20260430/perf_breakdown/results/summary.tsv`
- `memory/20260430/perf_breakdown/results/ann_profile_summary.tsv`
- `memory/20260505/cotra_revert_to_integrated_qps83k/report.md`
- `memory/20260505/cotra_qps_drop_top_index_analysis/report.md`
- `memory/20260507/cotra_inflight1024_qg16_ef67/report.md`
- `memory/20260507/cotra_integrated_640k_query_count_ef67/report.md`
- `memory/20260507/cotra_pernode10k_qg16_ef67/report.md`

No new performance job was submitted for this analysis.

## Key observations

- Current source default is `QUERY_GROUP_SIZE (4)` in `include/anns/anns_config.h`; `COTRA_QUERY_GROUP_SIZE` can override it at runtime.
- Prior QG16 run with 10K total queries reached median 94.2K QPS and best 95.3K QPS at recall around 0.90, but with high variance.
- The per-node-10K run uses global `Q=80000`, so each of 8 ranks injects 10K queries. With QG16 and ef=67 it completed at recall 0.90090 and leader QPS 45.8K.
- Larger true `qsize` increases internal query-state allocation and validation cost; it is not just a longer stream.
- The previously fixed RDMA `wr_id` cast in `src/rdma/rdma_comm.cc` is required for `qid >= 65536`; otherwise large batches can hang.
- Prior breakdown showed average `f_wait_time` 37.1%, `g_subquery_Q` 46.8%, `post_stage` 28.6%, and `g_steal` 10.2%, while network bandwidth was not saturated.

## Priority recommendations

1. Treat scheduler/post-stage/wait as the first bottleneck, not raw RDMA bandwidth or SIMD distance compute.
2. Keep enough in-flight work via QG tuning, but avoid using huge `qsize` as a service-QPS proxy. Prefer repeated microbatches or a streaming query source.
3. Reduce per-query allocation and initialization in `scala_search_init` and `QueryMsg`.
4. Separate search-only QPS from recall-validation QPS with `COTRA_SKIP_RECALL=1`.
5. Sweep `CAND_GROUP_SIZE`, `SYNC_BAR`, `POST_RES_MERGE`, and `POST_NODESYNC_MERGE` to reduce small-message and synchronization frequency.
6. Add profiler runs for the best QG setting to verify whether wait/post-stage fractions drop.

## Errors

No new command error occurred during this analysis.
