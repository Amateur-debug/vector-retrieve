# CoTra Reproduce Snapshot

This directory contains the runnable CoTra source tree used for the May 7, 2026 SIFT100M QPS experiments, plus the relevant `memory/20260507` experiment records.

Included:

- Current CoTra source code from `/hpc2hdd/home/rmeng603/workspace/CoTra`.
- `memory/20260507` reports, scripts, Slurm logs, parsed result TSV/JSON files, code-diff snapshots, and per-node memory samples.
- The latest QPS reports use the agreed 5-run trimmed mean: drop the highest and lowest `ef=67` QPS values, then average the remaining three.

Intentionally omitted:

- Local build outputs under `build/`.
- Git metadata from the source checkout.
- Large generated query/GT binary caches under `memory/20260507/*/data/`.
- Binary `scala_anns` executable snapshots under `memory/20260507/*/results/`.

The omitted query/GT caches are documented in the reports and were generated/reused locally on the HPC filesystem; they are too large for normal GitHub upload.
