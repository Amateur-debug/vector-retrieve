
link_libraries(coromem rdma_anns rdma_index)

add_subdirectory(rdma)
add_subdirectory(anns)
add_subdirectory(index)

add_executable(scala_anns scala_anns.cc)

add_executable(scala_index scala_index.cc)
target_link_libraries(scala_index diskann)
