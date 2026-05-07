# Method: SIFT100M larger-query QPS with per-node memory sampling

- Dataset/index: SIFT100M, uint8/L2, cached 8-node index from `memory/20260424/sift100m_8node_experiment/results/index/merged_index*`; no graph rebuild.
- Query/GT: amplified 640K query cache reused via symlinks from `memory/20260507/cotra_integrated_640k_query_count_ef67/data`.
- Query group size: `COTRA_QUERY_GROUP_SIZE=16`.
- Search resources: 8 Slurm nodes, 1 MPI/Slurm task per node, 8 search threads per rank.
- EF list: default `66,67,67,67,67,67`; `66` is warmup, five `67` rows are measured.
- QPS mean convention: for five measured `ef=67` QPS values, drop the highest and lowest and average the remaining three; arithmetic mean is kept only for audit.
- To avoid the previous memcached-counter hang, each total query count is submitted as a separate Slurm job with its own memcached lifecycle.
- Per-node memory sampling: inside each `srun` rank, the script starts `scala_anns` in the background and samples `/proc/$pid/status` plus `/proc/meminfo` every second into `mem_samples/q${QUERY_COUNT}_job${SLURM_JOB_ID}/${node}_rank${rank}.tsv`.
- Memory metrics: process `VmRSS`, process `VmHWM`, process `VmSize`, node `MemTotal`, node `MemAvailable`, and derived node-used memory.
