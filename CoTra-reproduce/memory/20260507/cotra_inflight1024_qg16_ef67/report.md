# CoTra In-flight 1024 QPS Report

## Configuration

- Job: `9722160`, Slurm state `COMPLETED`, exit code `0:0`, elapsed `00:00:07`.
- Nodes: `cpu1-90` through `cpu1-97`.
- Threads: 8 per node, 64 total.
- Test setting: `QUERY_GROUP_SIZE=16`, effective in-flight `8 nodes * 8 threads/node * 16 = 1024`.
- Dataset/index: cached SIFT100M merged index from `memory/20260424/sift100m_8node_experiment/results/index/merged_index`.
- Query count: 10000 total, 1250 per rank.
- EF sequence: one EF=66 warmup, then five EF=67 measurements.

## Results

Rank-0 leader log is used for cluster-level QPS.

| Run | EF | Recall | Cluster QPS |
| --- | ---: | ---: | ---: |
| warmup | 66 | 0.89832 | 36362.844 |
| 1 | 67 | 0.89997 | 94203.648 |
| 2 | 67 | 0.90066 | 71683.047 |
| 3 | 67 | 0.90091 | 95301.633 |
| 4 | 67 | 0.90101 | 94696.969 |
| 5 | 67 | 0.89986 | 67434.062 |

EF=67 statistics:

| Metric | Value |
| --- | ---: |
| count | 5 |
| mean QPS | 84663.872 |
| median QPS | 94203.648 |
| min QPS | 67434.062 |
| max QPS | 95301.633 |
| sample stdev QPS | 13876.243 |
| mean recall | 0.900482 |
| median recall | 0.900660 |

## Baseline Comparison

Baseline QG4 effective in-flight was 256 (`8 nodes * 8 threads/node * QUERY_GROUP_SIZE 4`).

| Baseline | QPS | QG16 mean delta | QG16 median delta | QG16 best delta |
| --- | ---: | ---: | ---: | ---: |
| `memory/20260507/cotra_batch_size_qsweep_ef67`, Q=10000, EF=67 | 82116.641 | +3.10% | +14.72% | +16.06% |
| `memory/20260505/cotra_revert_to_integrated_qps83k`, EF66->EF67 | 83479.422 | +1.42% | +12.85% | +14.16% |

Interpretation: raising CoTra's effective in-flight from 256 to 1024 can reach about 94K-95K QPS in the stable/high runs, but the five EF=67 repeats have high variance. The median improves clearly over QG4, while the mean only improves mildly because two repeats dropped to 71.7K and 67.4K QPS.

## Post-test State

The temporary source change was restored after the measurement:

- Current `include/anns/anns_config.h`: `QUERY_GROUP_SIZE (4)`.
- Current `/hpc2hdd/home/rmeng603/workspace/CoTra/build/tests/scala_anns` was rebuilt back to QG4.
- The QG16 binary used for this test remains snapshotted at `results/scala_anns_QUERY_GROUP_SIZE16_snapshot`.
