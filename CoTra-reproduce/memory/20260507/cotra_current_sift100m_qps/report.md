# Current CoTra QPS on SIFT100M

Date: 2026-05-07 Asia/Shanghai
Repo: `/hpc2hdd/home/rmeng603/workspace/CoTra`
Memory root: `memory/20260507/cotra_current_sift100m_qps`

## Goal

Measure the current CoTra implementation's QPS on SIFT100M after the QueryMsg early heavy-state release change.

## Configuration

- Dataset: SIFT100M, uint8, L2
- Index: cached 8-node CoTra index from `memory/20260424/sift100m_8node_experiment/results/index/merged_index*`
- No graph rebuild
- Nodes: 8 x CPU nodes (`cpu1-90` through `cpu1-97` in the completed official runs)
- Search threads: 8 per rank, 64 total
- Query group size: `COTRA_QUERY_GROUP_SIZE=16`
- EF sequence: `66,67,67,67,67,67` (`66` warmup, then five `67` measurements)
- Recall: enabled; QPS includes result/recall processing
- Query/GT source: amplified 640K query cache under this memory directory, reading only the first `Q` queries

## Official Results

Statistics convention: QPS mean is the 5-run trimmed mean, i.e. drop the highest and lowest `ef=67` QPS values and average the remaining three. `QPS raw mean` is retained only for auditability.

Leader/rank-0 QPS is used as the system-level CoTra QPS. Each row summarizes five repeated `ef=67` measurements after one warmup.

| query_count | per_node_query_count | ef | recall_mean | QPS mean(trimmed) | QPS raw mean | QPS median | QPS min | QPS max | stdev | job |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 10,000 | 1,250 | 67 | 0.90049 | 81,691 | 76,876 | 81,577 | 40,089 | 99,218 | 22,102 | 9724339 |
| 80,000 | 10,000 | 67 | 0.90063 | 89,592 | 87,770 | 89,691 | 69,223 | 100,852 | 12,527 | 9724337 |

Interpretation:

- Under the trimmed-mean convention, current CoTra on SIFT100M at recall around 0.90 reaches about `81.7K` QPS for total `Q=10K` and about `89.6K` QPS for total `Q=80K` / per-node `10K` queries.
- The Q=80K result remains much better than the earlier pre-fix per-node-10K QG16 run (`45.8K QPS` in `memory/20260507/cotra_pernode10k_qg16_ef67`), consistent with the lifecycle fix reducing large-batch pressure.
- QPS variance is still high, especially for Q=10K; the trimmed mean reduces the influence of single-run outliers.


## Files

- Official summary: `results/qps_summary_official.tsv`
- Official per-run leader results: `results/leader_ef67_runs_official.tsv`
- Official per-rank/ef results: `results/per_rank_all_ef_runs_official.tsv`
- Derived metrics: `results/derived_metrics_official.tsv`
- Slurm status: `results/sacct_status.tsv`
- Logs: `logs/anns_q10000_qg16_job9724339_*.log`, `logs/anns_q80000_qg16_job9724337_*.log`
- Code snapshot/diff: `results/scala_anns_9724337.snapshot`, `code_diff_at_job_9724337.patch`

## Error / Caution

- Job `9724333` first completed Q=10K, then its second sequential `srun` for Q=80K hung during RDMA connection setup. Root cause: the script ran two independent `scala_anns` invocations under one memcached instance without resetting memcached counters between `srun`s. I canceled that allocation and reran Q=80K as a separate job (`9724337`).
- The Q=10K official result uses the clean single-query-count job `9724339`, not the canceled allocation.
