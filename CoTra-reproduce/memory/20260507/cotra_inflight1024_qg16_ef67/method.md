# Method

Goal: measure CoTra with the same effective in-flight query count as the current XY 8-node setup: 1024.

First-principles mapping:
- XY effective in-flight = 64 query threads * pipeline depth 16 = 1024.
- CoTra ScalaANN_v3 injects grouped queries through `QUERY_GROUP_SIZE`.
- With 8 nodes and 8 search/query-injection threads per node, CoTra effective in-flight is `8 * 8 * QUERY_GROUP_SIZE`.
- Therefore, `QUERY_GROUP_SIZE=16` gives `8 * 8 * 16 = 1024`.

Implementation steps:
- Reused the existing CoTra build directory: `/hpc2hdd/home/rmeng603/workspace/CoTra/build`.
- Temporarily changed `include/anns/anns_config.h` from `QUERY_GROUP_SIZE (4)` to `(16)`.
- Built only target `scala_anns` using the existing build tree; no new build directory was created.
- Saved the QG16 binary snapshot before restoration at `results/scala_anns_QUERY_GROUP_SIZE16_snapshot`.
- Reused the cached SIFT100M 8-node index prefix: `/hpc2hdd/home/rmeng603/workspace/CoTra/memory/20260424/sift100m_8node_experiment/results/index/merged_index`.
- Ran one EF=66 warmup followed by five EF=67 measurements, Q=10000 total queries, 8 nodes, 8 threads per node.
- After collecting results, restored `QUERY_GROUP_SIZE` to 4 and rebuilt `scala_anns` in the existing build directory to avoid contaminating future CoTra runs.

Primary artifacts:
- Run script: `scripts/run_cotra_inflight1024_qg16_ef67.sbatch`
- Slurm stdout: `logs/slurm-9722160.out`
- Rank-0 leader log: `logs/anns_job9722160_cpu1-90_0.log`
- Summary JSON: `summary.json`
- Parsed QPS table: `results/ef67_leader_qps.tsv`
- Parsed stats: `results/ef67_stats.tsv`
- QG16 build log: `build_logs/build_qg16.log`
- Restore-to-QG4 build log: `build_logs/rebuild_restore_qg4.log`
