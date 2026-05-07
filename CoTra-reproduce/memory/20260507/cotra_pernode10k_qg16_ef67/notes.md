# CoTra per-node 10K query run, QUERY_GROUP_SIZE=16

- Goal: keep query group size at 16 and run current CoTra integrated ScalaANN_v3 with 10K origin queries per node.
- First-principles mapping: in `ScalaANN_v3`, `query_load = qsize / MACHINE_NUM`; current build has `MACHINE_NUM=8`, so per-node 10K means global `-Q 80000`.
- Dataset/index: reuse cached SIFT100M 8-node CoTra graph/index under `memory/20260424/sift100m_8node_experiment/results/index/merged_index*`; no graph rebuild.
- Query/GT: reuse amplified 640K query/GT files from `memory/20260507/cotra_integrated_640k_query_count_ef67/data/` and read the first 80K records.
- Build: `COTRA_MACHINE_NUM=8`, `COTRA_ENABLE_BREAKDOWN=OFF`, `QUERY_GROUP_SIZE=16`; `src/rdma/rdma_comm.cc` contains the wr_id cast fix needed for qid >= 65536.
