# CoTra SIFT100M QPS Scaling with Per-node Memory

Date: 2026-05-07 Asia/Shanghai
Repo: `/hpc2hdd/home/rmeng603/workspace/CoTra`
Memory root: `memory/20260507/cotra_qscale_mem_sift100m`

## Goal

Increase total query count beyond the previous 80K run and measure both QPS and per-node memory usage for current CoTra on SIFT100M.

## Statistics Convention

- Each reported QPS mean uses the agreed 5-run trimmed mean: sort the five `ef=67` QPS values, drop the highest and lowest, then average the remaining three.
- `QPS raw mean` is retained only for auditability and is not the headline mean.

## Configuration

- Dataset: SIFT100M, uint8, L2
- Index: cached 8-node CoTra index from `memory/20260424/sift100m_8node_experiment/results/index/merged_index*`; no graph rebuild
- Query/GT: amplified 640K query cache from `memory/20260507/cotra_integrated_640k_query_count_ef67/data`
- Nodes: `cpu1-90` through `cpu1-97`
- Search threads: 8 per rank, 64 total
- Query group size: `COTRA_QUERY_GROUP_SIZE=16`
- EF list: `66,67,67,67,67,67`; `66` warmup, five `67` measurements
- Recall enabled; QPS includes result processing and recall calculation
- One Slurm job per total query count to avoid memcached counter reuse/hang
- Memory sampling: `/proc/$pid/status` and `/proc/meminfo` sampled once per second inside each node/rank while `scala_anns` was running

## QPS Results

Leader/rank-0 QPS is used as system-level CoTra QPS. Each new row summarizes five `ef=67` measurements after warmup.

| query_count | per_node_query_count | recall_mean | QPS mean(trimmed) | QPS raw mean | QPS median | QPS min | QPS max | job | runtime |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 80,000 | 10,000 | 0.90063 | 89,592 | 87,770 | 89,691 | 69,223 | 100,852 | 9724337 | 24s |
| 160,000 | 20,000 | 0.90071 | 80,309 | 81,827 | 74,183 | 58,668 | 109,538 | 9724367 | 42s |
| 320,000 | 40,000 | 0.90068 | 92,144 | 93,907 | 85,111 | 80,135 | 112,968 | 9724368 | 76s |
| 640,000 | 80,000 | 0.90071 | 102,206 | 101,149 | 105,667 | 83,536 | 115,592 | 9724369 | 144s |

## Memory Summary

Primary memory metric is max process RSS sampled from `/proc/$pid/status`. `Actual` is the program-reported `Actual memory usage` line. System-used memory includes OS/page cache and is recorded in TSV files but is not the primary CoTra process metric.

| query_count | per_node_query_count | proc RSS mean GiB | proc RSS max GiB | Actual mean GiB | Actual max GiB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 80,000 | 10,000 | n/a | n/a | 17.28 | 17.71 |
| 160,000 | 20,000 | 26.61 | 27.04 | 27.90 | 28.35 |
| 320,000 | 40,000 | 46.88 | 47.33 | 49.16 | 49.63 |
| 640,000 | 80,000 | 87.44 | 87.93 | 91.67 | 92.19 |

## Per-node Memory

| Q | node | rank | max proc RSS GiB | Actual GiB | samples |
| ---: | --- | ---: | ---: | ---: | ---: |
| 160K | cpu1-90 | 0 | 26.52 | 27.81 | 39 |
| 160K | cpu1-91 | 1 | 26.92 | 28.23 | 39 |
| 160K | cpu1-92 | 2 | 25.81 | 27.06 | 40 |
| 160K | cpu1-93 | 3 | 26.90 | 28.21 | 40 |
| 160K | cpu1-94 | 4 | 26.34 | 27.62 | 40 |
| 160K | cpu1-95 | 5 | 26.70 | 27.99 | 39 |
| 160K | cpu1-96 | 6 | 26.66 | 27.95 | 39 |
| 160K | cpu1-97 | 7 | 27.04 | 28.35 | 39 |
| 320K | cpu1-90 | 0 | 46.78 | 49.06 | 71 |
| 320K | cpu1-91 | 1 | 47.20 | 49.50 | 71 |
| 320K | cpu1-92 | 2 | 46.06 | 48.28 | 72 |
| 320K | cpu1-93 | 3 | 47.18 | 49.46 | 72 |
| 320K | cpu1-94 | 4 | 46.60 | 48.87 | 72 |
| 320K | cpu1-95 | 5 | 46.95 | 49.24 | 71 |
| 320K | cpu1-96 | 6 | 46.94 | 49.22 | 70 |
| 320K | cpu1-97 | 7 | 47.33 | 49.63 | 70 |
| 640K | cpu1-90 | 0 | 87.34 | 91.56 | 135 |
| 640K | cpu1-91 | 1 | 87.81 | 92.06 | 134 |
| 640K | cpu1-92 | 2 | 86.54 | 90.73 | 138 |
| 640K | cpu1-93 | 3 | 87.74 | 91.99 | 138 |
| 640K | cpu1-94 | 4 | 87.15 | 91.37 | 138 |
| 640K | cpu1-95 | 5 | 87.50 | 91.74 | 135 |
| 640K | cpu1-96 | 6 | 87.52 | 91.75 | 135 |
| 640K | cpu1-97 | 7 | 87.93 | 92.18 | 135 |

## Interpretation

- Current CoTra now completes true one-shot `-Q 640000` on SIFT100M; the older pre-fix 640K attempt in `memory/20260507/cotra_integrated_640k_query_count_ef67` timed out without an EF line.
- Larger total query count improves amortization under the trimmed-mean convention: QPS mean is `89.6K` at 80K and `102.2K` at 640K, although per-run variance remains visible.
- Per-node memory scales roughly linearly with `qsize`: Actual memory is about `17.3 GiB` at 80K, `27.9 GiB` at 160K, `49.2 GiB` at 320K, and `91.7 GiB` at 640K.
- The primary remaining scaling cost is still query-count-proportional state: `ScalaSearch::scala_search_init()` constructs `QueryMsg` for all `qsize` queries, and each `QueryMsg` owns a large visit hash table before being released after result processing.
- With 512 GiB nodes, 640K is safe in this setup, but extending beyond 640K would require generating larger query/GT caches and would likely increase memory linearly unless query state allocation becomes lazy/windowed.

## Files

- Method: `method.md`
- Submit script: `scripts/run_sift100m_qscale_mem.sbatch`
- Parser: `scripts/parse_qscale_mem.py`
- QPS summary: `results/qps_summary.tsv`
- Leader EF=67 rows: `results/leader_ef67_runs.tsv`
- Per-rank EF rows: `results/per_rank_all_ef_runs.tsv`
- Memory summary: `results/memory_summary.tsv`
- Per-node memory: `results/per_node_memory.tsv`
- Slurm status: `results/sacct_status.tsv`
- Raw memory samples: `mem_samples/q*_job*/*.tsv`
- Logs: `logs/anns_q*_qg16_job*_*.log`

## Error Log

- No runtime errors in jobs `9724367`, `9724368`, and `9724369`.
