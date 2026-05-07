link_libraries(coromem rdma_anns rdma_index)

add_executable(profiling profiling.cc)
add_executable(kmeans kmeans_reorder.cc)
add_executable(cluster kmeans_cluster.cc)
add_executable(top_index top_index.cc)
add_executable(amplq ampl_query.cc)
add_executable(trans_hnsw trans_hnsw_top.cc)
add_executable(create_hnsw_top create_hnsw_top.cc)