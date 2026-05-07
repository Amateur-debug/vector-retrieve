# CoTra QueryMsg early heavy-state release

Date: 2026-05-07 Asia/Shanghai
Repo: `/hpc2hdd/home/rmeng603/workspace/CoTra`
Memory root: `memory/20260507/cotra_heavy_release_qg16_smoke`

## Goal

Restore the original intent of releasing completed query state promptly, while avoiding dangling pointers in `global_query` and per-thread `TaskManager::global_query`.

## Implementation

Instead of restoring the original `delete res` inside `proc_res()`, the implementation keeps the `QueryMsg` shell alive and releases only its heavy owned state after the result is processed.

Changed files:

- `include/anns/query_msg.h`
  - Added `heavy_state_released`.
  - Added `release_owned_storage()` and destructor cleanup for the malloc-owned query vector.
  - Added `release_heavy_state_after_result()` to free/clear query vector, visit hash storage, candidate/result queues, sync buffers, local task/result vectors, async buffers, and queued chunks.
  - Added guards in `update_sub_query()` and `update_sub_query_from_noncore()` so late sync messages do not repopulate already released query state.
- `include/anns/hash_table.h`
  - Added `release_storage()` for `OptHashPosVector`.
  - Made `clear()` safe after release.
  - Made `CheckAndSet()` return already-visited for released storage, so late duplicate work is ignored rather than touching freed query state.
- `include/anns/exec_query.h`
  - After recall processing, call `res->release_heavy_state_after_result()`.
  - In `COTRA_SKIP_RECALL=1` mode, call the same release before incrementing completed query count.

Safety choice:

- The object is still deleted at the end of the ef round from `appr_alg.global_query[q]`, so existing global pointers are not made dangling during standby/polling.
- Only internal heavy buffers are freed as soon as the query result has been consumed.

## Build validation

Commands completed successfully:

```bash
cmake --build build --target scala_anns -j 8
cmake --build build --target test_querymsg -j 8
./build/tests/anns/test_querymsg
```

## Distributed validation

All runs used cached SIFT100M 8-node index from `memory/20260424/sift100m_8node_experiment/results/index/merged_index*`, QG16 via `COTRA_QUERY_GROUP_SIZE=16`, ef=66 warmup then ef=67 measurement, 8 nodes x 8 CPUs, no rebuild of graph.

| job | query_count | per_node_query_count | ef | recall | leader QPS | status |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 9723418 | 10000 | 1250 | 67 | 0.89977 | 95250.797 | OK |
| 9723428 | 80000 | 10000 | 67 | 0.90083 | 95672.148 | OK |
| 9723552 | 80000 | 10000 | 67 | 0.90084 | 74638.078 | OK, after late-sync guard |

For Q=80000, the two post-change runs had leader QPS min/mean/max `74638.078 / 85155.113 / 95672.148`. The prior pre-change per-node-10K QG16 test in `memory/20260507/cotra_pernode10k_qg16_ef67` reported leader QPS `45830.277`, so this change removes the large Q=80K regression in these tests, though run-to-run variance remains high.

## Output files

- `results/summary.tsv`
- `results/per_rank_metrics.tsv`
- `results/derived_metrics.tsv`
- `logs/anns_q10000_qg16_job9723418_*.log`
- `logs/anns_q80000_qg16_job9723428_*.log`
- `logs/anns_q80000_qg16_job9723552_*.log`
- `code_diff_at_job_9723552.patch`

## Notes

- End-of-run RSS is still similar to the old run because freed malloc arenas are not necessarily returned to the OS. The important difference is that completed queries no longer retain live vectors/queues/hash state until the end of the ef round.
- `src/rdma/rdma_comm.cc` `wr_id` 64-bit fix remains necessary for Q>=65536.
- The Q=80000 result still has high variance; use repeated runs or same-job repeated ef=67 measurements for final numbers.

## Errors

No build, unit, or Slurm validation error occurred after this implementation.
