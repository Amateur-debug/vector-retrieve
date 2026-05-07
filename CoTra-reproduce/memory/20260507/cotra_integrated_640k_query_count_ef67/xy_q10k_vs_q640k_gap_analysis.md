# XY 10K total vs 640K total QPS gap analysis

Date: 2026-05-07 (Asia/Shanghai)

## Question

Why does XY show much lower QPS when the query client sends only 10K total queries than when it sends 640K total queries?

## Relevant measured results

All measurements below use the same search-side setup: SIFT100M uint8, TCP, 8 worker nodes, each worker restricted to 8 CPUs with `taskset -c 0-7`, graph/partition cache reused.

| Phase | Physical queries per round | Sender shape | QPS | Per-run times |
|---|---:|---|---:|---|
| `query_q10k_p1` | 10,000 | 1 process sends full 10K | 12,196.68 | 2.061s, 0.834s, 0.853s, 0.776s, 0.753s |
| `query_q10k_p64_sharded` | 10,000 | 64 processes shard the 10K query set | 89,828.59 | 0.135s, 0.120s, 0.105s, 0.110s, 0.100s |
| `query_q640k_p64` | 640,000 | 64 processes each send full 10K | 126,351.18 | 5.069s, 5.049s, 5.424s, 5.078s, 5.022s |

Ratios:

- `query_q640k_p64 / query_q10k_p64_sharded = 1.41x`.
- `query_q10k_p64_sharded / query_q10k_p1 = 7.37x`.
- `query_q640k_p64 / query_q10k_p1 = 10.36x`.

## First-principles explanation

XY `query.py` computes per-run QPS as:

```text
run_qps = total_queries_this_round / max(query_time_per_sender_process)
```

The code sends queries asynchronously with a bounded pipeline:

- each sender process sends until `sent - received >= PIPELINE_DEPTH`, then waits for receives;
- current script uses `VR_QUERY_PIPELINE_DEPTH=16`;
- therefore maximum outstanding requests are approximately `num_sender_processes * 16`.

This means QPS depends strongly on both **how many requests are measured** and **how many sender processes keep the workers busy**.

## Main causes of the observed gap

### 1. If “10K total” is implemented by `num_query_threads=1`, the system is under-driven

`query_q10k_p1` is not a query-count-only control. It changes two things at once:

- physical query count drops from 640K to 10K;
- sender process count drops from 64 to 1.

With one sender process, outstanding requests are only about `1 * 16 = 16`. With 64 sender processes, outstanding requests are about `64 * 16 = 1024`. This is a 64x difference in request injection concurrency.

Also, `query.py` intentionally uses one process per SDK client so different PIDs can be sticky-balanced across entry endpoints. With only one PID, request injection is much more concentrated than with 64 PIDs. Therefore `query_q10k_p1` mainly measures client-side load-generation bottleneck, not the search engine's peak throughput.

This explains why `query_q10k_p1` is only 12.2K QPS while the 64-process 10K control reaches 89.8K QPS.

### 2. Even with 64 sender processes, 10K total queries are too short to measure steady state cleanly

The cleaner control is `query_q10k_p64_sharded`: it keeps 64 sender processes but shards the 10K query set, so each process sends only about `10000 / 64 ~= 156` queries.

At pipeline depth 16, each sender has only about 10 pipeline waves. The measured window is only around 0.10-0.13 seconds. In such a short run, fixed costs and tail effects are a large fraction of the measured time:

- ramp-up to fill the pipeline;
- final drain after the last sends;
- Python multiprocessing scheduling jitter;
- barrier start drift and the slowest process being used as the denominator;
- TCP/socket and SDK queue scheduling effects at the beginning/end of a burst.

Using the 640K steady-state result as a reference:

- 640K/64 process QPS = 126,351 QPS;
- if 10K could run at the same steady-state rate, expected time would be `10000 / 126351 ~= 0.079s`;
- actual 10K/64-process aggregate time is about `10000 / 89829 ~= 0.111s`;
- the extra ~32 ms is fixed/tail/ramp overhead, which is about 29% of the observed 10K run time.

This overhead is mostly amortized in the 640K case because each sender sends 10K queries and the measured round lasts about 5 seconds.

### 3. 640K does not make each individual query faster; it measures a saturated longer stream

`query_q640k_p64` is 64 copies of the same 10K query set, not 640K unique queries. The recall remains the same (`0.90201`), which confirms that the workload semantics per query are unchanged.

The higher QPS comes from better utilization and amortization:

- 64 sender processes continuously keep the worker queues full;
- the measurement interval is long enough to avoid startup/drain dominating;
- `max(query_time_per_sender_process)` becomes a steady-state throughput measure rather than a short-burst tail-latency measure.

## Conclusion

The large apparent gap has two layers:

1. The huge `10.36x` gap between `query_q10k_p1` and `query_q640k_p64` is mostly a sender-concurrency/load-generation artifact. It is not evidence that search computation becomes 10x faster with 640K queries.
2. The remaining `1.41x` gap between `query_q10k_p64_sharded` and `query_q640k_p64` is mainly due to short-run fixed overhead, pipeline ramp/drain, and slowest-sender tail effects in the 10K total run.

For fair comparisons, avoid using `num_query_threads=1` as “10K total” unless the goal is explicitly to measure a one-client scenario. The better 10K control is `query_q10k_p64_sharded`, while the better peak-throughput control is a sufficiently long stream with the same sender process count.

## Recommended next measurements

To quantify the amortization curve, run the same 64 sender processes with total query counts between 10K and 640K, e.g. 20K, 40K, 80K, 160K, 320K, 640K. This can be done by sharding/duplicating per-rank query assignments while keeping the sender process count fixed at 64. Expected result: QPS should rise from the short-burst value toward the 640K steady-state value and then plateau.
