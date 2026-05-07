# Query-count QPS comparison method

Date: 2026-05-07 (Asia/Shanghai)

## Goal

Test the hypothesis that XY appears faster partly because its current `query.py --num_query_threads 64` sends 64 full copies of the 10K query set (640K physical requests per round), while integrated CoTra normally reports one 10K query sweep.

## XY setup

- Job: `9722149`, status `COMPLETED`, nodes `cpu1-[90-98]`.
- Query/client node: `cpu1-90`.
- Search worker nodes: `cpu1-[91-98]`.
- Worker CPU restriction: `--num_threads 8 --num_cpu 8`, `taskset -c 0-7` in `main.py` worker launch.
- Communication: TCP.
- Dataset/cache: SIFT100M uint8, graph and partition cache reused from `/hpc2hdd/home/rmeng603/workspace/vector-retrieve`.
- Profiling env vars were unset to avoid instrumentation overhead.

XY phases:

1. `query_q10k_p1`: one `query.py` rank, one worker process, full 10K query set; 5 runs.
2. `query_q10k_p64_sharded`: 64 `query.py` ranks on the query node, each with one worker process; `query_world_size=64` partitions the 10K query set, so total physical requests per round remain 10K while sender process count stays 64; 5 synchronized runs.
3. `query_q640k_p64`: one `query.py` rank with 64 worker processes; each worker sends the full 10K query set, so total physical requests per round are 640K; 5 runs.

## CoTra setup

- Baseline reference: job `9718741`, integrated `scala_anns`, EF list `66,67`, 8 nodes x 8 CPUs, rank0 EF=67 QPS `83479.422`, recall `0.89996`.
- True one-shot qsize=640K attempt: job `9722148`, nodes `cpu1-[30-37]`, `-Q 640000`, amplified query/GT files in `/hpc2hdd/home/rmeng603/workspace/CoTra/memory/20260507/cotra_integrated_640k_query_count_ef67/data`, EF list `66,67`. This timed out at 1h before printing any EF result.
- Fallback sequential 640K approximation: job `9722153`, nodes `cpu1-[90-97]`, `-Q 10000`, EF list `66` plus `67` repeated 64 times. Aggregate QPS is computed as `640000 / sum(10000 / qps_i)` over the 64 rank0 EF=67 rows.

## Important caveats

- The CoTra repeated-10K fallback is not the same as 64 concurrent external clients; it is integrated CoTra executing 64 sequential 10K rounds.
- CoTra's true `qsize=640000` path is pathological because `scala_search_init()` allocates one `QueryMsg` per query, and each `QueryMsg` owns a default `OptHashPosVector` hash table. This makes qsize itself change memory footprint and behavior, not just the number of requests.
- The XY `query_q10k_p1` phase intentionally reduces sender process count; the `query_q10k_p64_sharded` phase is the cleaner query-count-only control because it keeps 64 sender worker processes while sending only 10K total requests.
