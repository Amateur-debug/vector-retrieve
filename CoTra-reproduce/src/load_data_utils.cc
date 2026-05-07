#include <malloc.h>
#include <memory.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// Moved graph.resize(npts); inside the load_impl function

void load_graph(
    const std::string &filename, std::vector<std::vector<uint32_t>> &graph,
    uint32_t npts) {
  size_t expected_file_size;
  size_t file_frozen_pts;
  uint32_t start;
  size_t file_offset = 0;  // will need this for single file format support
  uint32_t max_observed_degree = 0;
  std::ifstream in;
  in.exceptions(std::ios::badbit | std::ios::failbit);
  in.open(filename, std::ios::binary);
  in.seekg(file_offset, in.beg);
  in.read((char *)&expected_file_size, sizeof(size_t));
  in.read((char *)&max_observed_degree, sizeof(uint32_t));
  in.read((char *)&start, sizeof(uint32_t));
  in.read((char *)&file_frozen_pts, sizeof(size_t));

  // Resize the graph to accommodate the number of points
  graph.resize(npts);
  size_t vamana_metadata_size =
      sizeof(size_t) + sizeof(uint32_t) + sizeof(uint32_t) + sizeof(size_t);

  std::cout << "From graph header, expected_file_size: " << expected_file_size
            << ", _max_observed_degree: " << max_observed_degree
            << ", _start: " << start;

  std::cout << "Loading vamana graph " << filename << "..." << std::flush;
  std::cout << std::endl;

  size_t bytes_read = vamana_metadata_size;
  size_t cc = 0;
  uint32_t nodes_read = 0;
  while (bytes_read != expected_file_size) {
    uint32_t k;
    in.read((char *)&k, sizeof(uint32_t));
    cc += k;
    ++nodes_read;
    std::vector<uint32_t> tmp(k);
    tmp.reserve(k);
    in.read((char *)tmp.data(), k * sizeof(uint32_t));
    graph[nodes_read - 1].swap(tmp);
    bytes_read += sizeof(uint32_t) * ((size_t)k + 1);
  }
}

template <typename T>
void load_fbin(
    const std::string &filename, T *&data, uint32_t &npts, uint32_t &ndims) {
  std::ifstream in;
  in.exceptions(std::ios::badbit | std::ios::failbit);
  in.open(filename, std::ios::binary);
  // file format: num vectors, num dimension, vector data
  in.seekg(0, in.beg);
  in.read((char *)&npts, sizeof(uint32_t));
  in.read((char *)&ndims, sizeof(uint32_t));
  data = new T[npts * ndims];
  in.read((char *)data, npts * ndims * sizeof(T));

  std::cout << "Loaded " << npts << " points with " << ndims
            << " dimensions from " << filename << std::endl;

  in.close();
}

void load_gt_file(
    const std::string &bin_file, uint32_t *&ids, float *&dists, size_t &npts,
    size_t &dim) {
  std::ifstream reader(bin_file, std::ios::binary);

  int npts_i32, dim_i32;
  reader.read((char *)&npts_i32, sizeof(int));
  reader.read((char *)&dim_i32, sizeof(int));
  npts = (unsigned)npts_i32;
  dim = (unsigned)dim_i32;

  std::cout << "Metadata: #pts = " << npts << ", #dims = " << dim << "... "
            << std::endl;

  size_t expected_file_size_with_dists =
      2 * npts * dim * sizeof(uint32_t) + 2 * sizeof(uint32_t);

  ids = new uint32_t[npts * dim];
  reader.read((char *)ids, npts * dim * sizeof(uint32_t));

  dists = new float[npts * dim];
  reader.read((char *)dists, npts * dim * sizeof(float));
}

template <typename T>
void align_data(T *&data, size_t npts, size_t ndims, size_t alignment) {
  size_t padded_ndims = (ndims + alignment - 1) / alignment * alignment;
  if (padded_ndims == ndims) {
    return;
  }
  T *aligned_data =
      (T *)memalign(alignment * sizeof(T), npts * padded_ndims * sizeof(T));
  memset(aligned_data, 0, npts * padded_ndims * sizeof(T));

  for (size_t i = 0; i < npts; i++) {
    memcpy(
        aligned_data + i * padded_ndims, data + i * ndims, ndims * sizeof(T));
  }
  delete[] data;
  data = aligned_data;
}

template <typename InType, typename OutType>
void convert_types(
    const InType *srcmat, OutType *destmat, size_t npts, size_t dim) {
#pragma omp parallel for schedule(static, 65536)
  for (int64_t i = 0; i < (int64_t)npts; i++) {
    for (uint64_t j = 0; j < dim; j++) {
      destmat[i * dim + j] = (OutType)srcmat[i * dim + j];
    }
  }
}

int main(int argc, char **argv) {
  if (argc != 6) {
    std::cerr << "Usage: " << argv[0]
              << " <base_data_file> <query_data_file> <gt_file> "
                 "<graph_index_file> <data_type>"
              << std::endl;
    return 1;
  }
  std::string base_data_file(argv[1]);
  std::string query_data_file(argv[2]);
  std::string gt_file(argv[3]);
  std::string graph_index_file(argv[4]);
  std::string data_type(argv[5]);

  std::vector<std::vector<uint32_t>> graph;
  uint32_t base_npts;
  uint32_t base_ndims;
  uint32_t q_npts;
  uint32_t q_ndims;
  if (data_type == "float") {
    float *base_data;
    float *query_data;
    load_fbin<float>(base_data_file, base_data, base_npts, base_ndims);
    load_fbin<float>(query_data_file, query_data, q_npts, q_ndims);
    load_graph(graph_index_file, graph, base_npts);
    // do something with the data
  } else if (data_type == "uint8") {
    uint8_t *base_data;
    uint8_t *query_data;
    load_fbin<uint8_t>(base_data_file, base_data, base_npts, base_ndims);

    std::cout << std::endl;
    load_fbin<uint8_t>(query_data_file, query_data, q_npts, q_ndims);
    std::cout << "Loaded " << base_npts << " points with " << base_ndims
              << " dimensions from " << base_data_file << std::endl;
    std::cout << "Loaded " << q_npts << " points with " << q_ndims
              << " dimensions from " << query_data_file << std::endl;
    load_graph(graph_index_file, graph, base_npts);
    // do something with the data
    // check
    // std::cout << graph[0][0] << ", " << graph[0][1] << std::endl;
    // std::cout << (uint32_t)base_data[4] << ", " << (uint32_t)query_data[123]
    // << std::endl;
  }

  std::cout << "finished loading data\n" << std::endl;
  return 0;
}
