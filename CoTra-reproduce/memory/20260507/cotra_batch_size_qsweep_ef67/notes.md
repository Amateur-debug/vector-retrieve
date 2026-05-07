# Notes

- Reuse cached SIFT100M 8-node CoTra graph/index from `memory/20260424/sift100m_8node_experiment/results/index/merged_index*`; no graph rebuild.
- Use existing amplified 640K query/GT files from `memory/20260507/cotra_integrated_640k_query_count_ef67/data/` via symlinks.
- Keep retrieval workers fixed at 8 nodes x 8 CPU threads, `COTRA_MACHINE_NUM=8`, `COTRA_ENABLE_BREAKDOWN=OFF`.
- Sweep `-Q` to test integrated batch size effect. In `ScalaANN_v3`, each rank handles `Q/8` origin queries and all ranks allocate query state for `Q`.
