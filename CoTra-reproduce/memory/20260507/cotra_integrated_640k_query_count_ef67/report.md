# Query-count QPS comparison: XY vs CoTra

## Results

| System/config | Physical queries per measured round | Query sender shape | Search resources | Recall@10 | QPS |
|---|---:|---|---|---:|---:|
| XY `query_q10k_p1` | 10,000 | 1 process sends full 10K | 8 worker nodes x 8 CPUs | 0.90201 | 12196.68 |
| XY `query_q10k_p64_sharded` | 10,000 | 64 processes collectively shard 10K | 8 worker nodes x 8 CPUs | 0.90201 | 89828.59 |
| XY `query_q640k_p64` | 640,000 | 64 processes each send full 10K | 8 worker nodes x 8 CPUs | 0.90201 | 126351.18 |
| CoTra integrated baseline (`ef=67`) | 10,000 | integrated in 8 ranks | 8 nodes x 8 CPUs | 0.89996 | 83479.42 |
| CoTra repeated 10K x64 (`ef=67`) | 640,000 sequential total | integrated in 8 ranks, 64 sequential rounds | 8 nodes x 8 CPUs | 0.90100 | 61938.05 |
| CoTra true `-Q 640000` | 640,000 | integrated in 8 ranks, one qsize=640K invocation | 8 nodes x 8 CPUs | N/A | timeout/no valid QPS |

## Key comparisons

- XY with 64 sender processes but only 10K total requests reaches `89828.59` QPS, close to CoTra's original `83479.42` QPS. This means the original 10K-vs-640K query-count mismatch is a real confound.
- XY 640K/64-process QPS is `1.41x` its 10K/64-process QPS. Larger per-round query count improves amortization/saturation, but it does not explain a 10x gap by itself.
- XY 10K with only one sender process is `12196.68` QPS, `7.37x` lower than the 64-sender 10K control. Therefore changing XY to “10K total” by simply setting `num_query_threads=1` confounds query count with client-side load generation.
- CoTra repeated 10K x64 aggregate is `61938.05` QPS, lower than its first EF=67 repeated row `82053.30` and the baseline `83479.42`. This fallback likely includes repeated-EF memory-pressure artifacts, so it should not be treated as a clean concurrent 640K client result.
- CoTra true `-Q 640000` timed out after 1h without any EF result line. This is strong evidence that CoTra's qsize parameter changes internal state allocation substantially; it is not just a longer query stream.

## Conclusion

The query-count mismatch does affect the comparison: XY's 640K/64-process mode is higher than its 10K/64-process mode, and one-process 10K under-drives the system badly. However, after controlling sender process count, XY 10K is about `1.08x` CoTra's 10K integrated baseline, not dramatically higher. Thus the previously observed large gap is largely a load-generation/query-count artifact, not solely a search-engine compute advantage.

For fair headline comparison, use either:

1. XY `query_q10k_p64_sharded` vs CoTra integrated 10K baseline, both 10K logical/physical queries and 8 search nodes x 8 CPUs; or
2. implement a CoTra external client that can stream 640K requests without setting `qsize=640000`, then compare against XY `query_q640k_p64`.
