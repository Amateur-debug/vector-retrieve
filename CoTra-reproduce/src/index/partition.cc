// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <omp.h>

#include <cmath>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>

#include "diskann/include/tsl/robin_map.h"
#include "diskann/include/tsl/robin_set.h"

#if defined(DISKANN_RELEASE_UNUSED_TCMALLOC_MEMORY_AT_CHECKPOINTS) && \
    defined(DISKANN_BUILD)
#include "diskann/include/gperftools/malloc_extension.h"
#endif

#include "diskann/include/index.h"
#include "diskann/include/memory_mapper.h"
#include "diskann/include/parameters.h"
#include "diskann/include/utils.h"
// #include "index/kmeans.h"
#include "index/math_utils.h"
#include "index/partition.h"
// #include "abstract_data_store.h"
// #include "in_mem_data_store.h"
#ifdef _WINDOWS
#include <xmmintrin.h>
#endif

#include <faiss/Index.h>
#include <faiss/IndexFlat.h>
#include <filesystem>

#include "coromem/include/utils.h"

// block size for reading/ processing large files and matrices in blocks
#define BLOCK_SIZE 500000

// #define SAVE_INFLATED_PQ true

template <typename T>
void gen_random_slice(
    const std::string base_file, const std::string output_prefix,
    double sampling_rate) {
  size_t read_blk_size = 64 * 1024 * 1024;
  cached_ifstream base_reader(base_file.c_str(), read_blk_size);
  std::ofstream sample_writer(
      std::string(output_prefix + "_data.bin").c_str(), std::ios::binary);
  std::ofstream sample_id_writer(
      std::string(output_prefix + "_ids.bin").c_str(), std::ios::binary);

  std::random_device
      rd;  // Will be used to obtain a seed for the random number engine
  auto x = rd();
  std::mt19937 generator(
      x);  // Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<float> distribution(0, 1);

  size_t npts, nd;
  uint32_t npts_u32, nd_u32;
  uint32_t num_sampled_pts_u32 = 0;
  uint32_t one_const = 1;

  base_reader.read((char *)&npts_u32, sizeof(uint32_t));
  base_reader.read((char *)&nd_u32, sizeof(uint32_t));
  diskann::cout << "Loading base " << base_file << ". #points: " << npts_u32
                << ". #dim: " << nd_u32 << "." << std::endl;
  sample_writer.write((char *)&num_sampled_pts_u32, sizeof(uint32_t));
  sample_writer.write((char *)&nd_u32, sizeof(uint32_t));
  sample_id_writer.write((char *)&num_sampled_pts_u32, sizeof(uint32_t));
  sample_id_writer.write((char *)&one_const, sizeof(uint32_t));

  npts = npts_u32;
  nd = nd_u32;
  std::unique_ptr<T[]> cur_row = std::make_unique<T[]>(nd);

  for (size_t i = 0; i < npts; i++) {
    base_reader.read((char *)cur_row.get(), sizeof(T) * nd);
    float sample = distribution(generator);
    if (sample < sampling_rate) {
      sample_writer.write((char *)cur_row.get(), sizeof(T) * nd);
      uint32_t cur_i_u32 = (uint32_t)i;
      sample_id_writer.write((char *)&cur_i_u32, sizeof(uint32_t));
      num_sampled_pts_u32++;
    }
  }
  sample_writer.seekp(0, std::ios::beg);
  sample_writer.write((char *)&num_sampled_pts_u32, sizeof(uint32_t));
  sample_id_writer.seekp(0, std::ios::beg);
  sample_id_writer.write((char *)&num_sampled_pts_u32, sizeof(uint32_t));
  sample_writer.close();
  sample_id_writer.close();
  diskann::cout << "Wrote " << num_sampled_pts_u32
                << " points to sample file: " << output_prefix + "_data.bin"
                << std::endl;
}

// streams data from the file, and samples each vector with probability p_val
// and returns a matrix of size slice_size* ndims as floating point type.
// the slice_size and ndims are set inside the function.

/***********************************
 * Reimplement using gen_random_slice(const T* inputdata,...)
 ************************************/

template <typename T>
void gen_random_slice(
    const std::string data_file, double p_val, float *&sampled_data,
    size_t &slice_size, size_t &ndims) {
  size_t npts;
  uint32_t npts32, ndims32;
  std::vector<std::vector<float>> sampled_vectors;

  // amount to read in one shot
  size_t read_blk_size = 64 * 1024 * 1024;
  // create cached reader + writer
  // cached_ifstream base_reader(data_file.c_str(), read_blk_size);

  std::ifstream base_reader(data_file.c_str(), std::ios::binary);

  // metadata: npts, ndims
  base_reader.read((char *)&npts32, sizeof(uint32_t));
  base_reader.read((char *)&ndims32, sizeof(uint32_t));
  npts = npts32;
  ndims = ndims32;

  std::unique_ptr<T[]> cur_vector_T = std::make_unique<T[]>(ndims);
  p_val = p_val < 1 ? p_val : 1;
  std::cout << "sample p_val: " << p_val << std::endl;

  std::random_device rd;  // Will be used to obtain a seed for the random number
  size_t x = rd();
  std::mt19937 generator((uint32_t)x);
  std::uniform_real_distribution<float> distribution(0, 1);

  std::vector<size_t> sample_ids;
  for (size_t i = 0; i < npts; i++) {
    if (i % 10000000 == 0) {
      std::cout << "Sampling " << (double)i * 100.0 / npts << "%pts\r"
                << std::flush;
    }
    float rnd_val = distribution(generator);
    if (rnd_val < p_val) {
      sample_ids.push_back(i);
    }
  }
  std::cout << "Sampled " << sample_ids.size() << "pts\n";
  std::sort(sample_ids.begin(), sample_ids.end());

  for (auto i : sample_ids) {
    if (i % 10000 == 0) {
      std::cout << "Reading " << (double)i * 100.0 / npts << "%pts\r"
                << std::flush;
    }
    base_reader.seekg(2 * sizeof(uint32_t) + i * ndims * sizeof(T));
    base_reader.read((char *)cur_vector_T.get(), ndims * sizeof(T));
    std::vector<float> cur_vector_float;
    for (size_t d = 0; d < ndims; d++)
      cur_vector_float.push_back(cur_vector_T[d]);
    sampled_vectors.push_back(cur_vector_float);
  }

  // for (size_t i = 0; i < npts; i++) {
  //   if (i % 10000000 == 0) {
  //     std::cout << "Sampling " << (double)i * 100.0 / npts << "%pts\r"
  //               << std::flush;
  //   }

  //   base_reader.read((char *)cur_vector_T.get(), ndims * sizeof(T));
  //   float rnd_val = distribution(generator);
  //   if (rnd_val < p_val) {
  //     std::vector<float> cur_vector_float;
  //     for (size_t d = 0; d < ndims; d++)
  //       cur_vector_float.push_back(cur_vector_T[d]);
  //     sampled_vectors.push_back(cur_vector_float);
  //   }
  // }
  slice_size = sampled_vectors.size();
  sampled_data = new float[slice_size * ndims];
  for (size_t i = 0; i < slice_size; i++) {
    for (size_t j = 0; j < ndims; j++) {
      sampled_data[i * ndims + j] = sampled_vectors[i][j];
    }
  }
}

// same as above, but samples from the matrix inputdata instead of a file of
// npts*ndims to return sampled_data of size slice_size*ndims.
template <typename T>
void gen_random_slice(
    const T *inputdata, size_t npts, size_t ndims, double p_val,
    float *&sampled_data, size_t &slice_size) {
  std::vector<std::vector<float>> sampled_vectors;
  const T *cur_vector_T;

  p_val = p_val < 1 ? p_val : 1;

  std::random_device
      rd;  // Will be used to obtain a seed for the random number engine
  size_t x = rd();
  std::mt19937 generator(
      (uint32_t)x);  // Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<float> distribution(0, 1);

  for (size_t i = 0; i < npts; i++) {
    cur_vector_T = inputdata + ndims * i;
    float rnd_val = distribution(generator);
    if (rnd_val < p_val) {
      std::vector<float> cur_vector_float;
      for (size_t d = 0; d < ndims; d++)
        cur_vector_float.push_back(cur_vector_T[d]);
      sampled_vectors.push_back(cur_vector_float);
    }
  }
  slice_size = sampled_vectors.size();
  sampled_data = new float[slice_size * ndims];
  for (size_t i = 0; i < slice_size; i++) {
    for (size_t j = 0; j < ndims; j++) {
      sampled_data[i * ndims + j] = sampled_vectors[i][j];
    }
  }
}

int estimate_cluster_sizes(
    float *test_data_float, size_t num_test, float *pivots,
    const size_t num_centers, const size_t test_dim, const size_t k_base,
    std::vector<size_t> &cluster_sizes) {
  cluster_sizes.clear();

  size_t *shard_counts = new size_t[num_centers];

  for (size_t i = 0; i < num_centers; i++) {
    shard_counts[i] = 0;
  }

  size_t block_size = num_test <= BLOCK_SIZE ? num_test : BLOCK_SIZE;
  uint32_t *block_closest_centers = new uint32_t[block_size * k_base];
  float *block_data_float;

  size_t num_blocks = DIV_ROUND_UP(num_test, block_size);

  for (size_t block = 0; block < num_blocks; block++) {
    size_t start_id = block * block_size;
    size_t end_id = (std::min)((block + 1) * block_size, num_test);
    size_t cur_blk_size = end_id - start_id;

    block_data_float = test_data_float + start_id * test_dim;

    math_utils::compute_closest_centers(
        block_data_float, cur_blk_size, test_dim, pivots, num_centers, k_base,
        block_closest_centers);

    for (size_t p = 0; p < cur_blk_size; p++) {
      for (size_t p1 = 0; p1 < k_base; p1++) {
        size_t shard_id = block_closest_centers[p * k_base + p1];
        shard_counts[shard_id]++;
      }
    }
  }

  diskann::cout << "Estimated cluster sizes: ";
  for (size_t i = 0; i < num_centers; i++) {
    uint32_t cur_shard_count = (uint32_t)shard_counts[i];
    cluster_sizes.push_back((size_t)cur_shard_count);
    diskann::cout << cur_shard_count << " ";
  }
  diskann::cout << std::endl;
  delete[] shard_counts;
  delete[] block_closest_centers;
  return 0;
}

template <typename T>
int shard_data_into_clusters(
    const std::string data_file, float *pivots, const size_t num_centers,
    const size_t dim, const size_t k_base, std::string prefix_path) {
  size_t read_blk_size = 64 * 1024 * 1024;
  //  uint64_t write_blk_size = 64 * 1024 * 1024;
  // create cached reader + writer
  cached_ifstream base_reader(data_file, read_blk_size);
  uint32_t npts32;
  uint32_t basedim32;
  base_reader.read((char *)&npts32, sizeof(uint32_t));
  base_reader.read((char *)&basedim32, sizeof(uint32_t));
  size_t num_points = npts32;
  if (basedim32 != dim) {
    diskann::cout << "Error. dimensions dont match for train set and base set"
                  << std::endl;
    return -1;
  }

  std::unique_ptr<size_t[]> shard_counts =
      std::make_unique<size_t[]>(num_centers);
  std::vector<std::ofstream> shard_data_writer(num_centers);
  std::vector<std::ofstream> shard_idmap_writer(num_centers);
  uint32_t dummy_size = 0;
  uint32_t const_one = 1;

  for (size_t i = 0; i < num_centers; i++) {
    std::string data_filename =
        prefix_path + "_subshard-" + std::to_string(i) + ".bin";
    std::string idmap_filename =
        prefix_path + "_subshard-" + std::to_string(i) + "_ids_uint32.bin";
    shard_data_writer[i] =
        std::ofstream(data_filename.c_str(), std::ios::binary);
    shard_idmap_writer[i] =
        std::ofstream(idmap_filename.c_str(), std::ios::binary);
    shard_data_writer[i].write((char *)&dummy_size, sizeof(uint32_t));
    shard_data_writer[i].write((char *)&basedim32, sizeof(uint32_t));
    shard_idmap_writer[i].write((char *)&dummy_size, sizeof(uint32_t));
    shard_idmap_writer[i].write((char *)&const_one, sizeof(uint32_t));
    shard_counts[i] = 0;
  }

  size_t block_size = num_points <= BLOCK_SIZE ? num_points : BLOCK_SIZE;
  std::unique_ptr<uint32_t[]> block_closest_centers =
      std::make_unique<uint32_t[]>(block_size * k_base);
  std::unique_ptr<T[]> block_data_T = std::make_unique<T[]>(block_size * dim);
  std::unique_ptr<float[]> block_data_float =
      std::make_unique<float[]>(block_size * dim);

  size_t num_blocks = DIV_ROUND_UP(num_points, block_size);

  for (size_t block = 0; block < num_blocks; block++) {
    size_t start_id = block * block_size;
    size_t end_id = (std::min)((block + 1) * block_size, num_points);
    size_t cur_blk_size = end_id - start_id;

    base_reader.read(
        (char *)block_data_T.get(), sizeof(T) * (cur_blk_size * dim));
    diskann::convert_types<T, float>(
        block_data_T.get(), block_data_float.get(), cur_blk_size, dim);

    math_utils::compute_closest_centers(
        block_data_float.get(), cur_blk_size, dim, pivots, num_centers, k_base,
        block_closest_centers.get());

    for (size_t p = 0; p < cur_blk_size; p++) {
      for (size_t p1 = 0; p1 < k_base; p1++) {
        size_t shard_id = block_closest_centers[p * k_base + p1];
        uint32_t original_point_map_id = (uint32_t)(start_id + p);
        shard_data_writer[shard_id].write(
            (char *)(block_data_T.get() + p * dim), sizeof(T) * dim);
        shard_idmap_writer[shard_id].write(
            (char *)&original_point_map_id, sizeof(uint32_t));
        shard_counts[shard_id]++;
      }
    }
  }

  size_t total_count = 0;
  diskann::cout << "Actual shard sizes: " << std::flush;
  for (size_t i = 0; i < num_centers; i++) {
    uint32_t cur_shard_count = (uint32_t)shard_counts[i];
    total_count += cur_shard_count;
    diskann::cout << cur_shard_count << " ";
    shard_data_writer[i].seekp(0);
    shard_data_writer[i].write((char *)&cur_shard_count, sizeof(uint32_t));
    shard_data_writer[i].close();
    shard_idmap_writer[i].seekp(0);
    shard_idmap_writer[i].write((char *)&cur_shard_count, sizeof(uint32_t));
    shard_idmap_writer[i].close();
  }

  diskann::cout << "\n Partitioned " << num_points
                << " with replication factor " << k_base << " to get "
                << total_count << " points across " << num_centers << " shards "
                << std::endl;
  return 0;
}

// useful for partitioning large dataset. we first generate only the IDS for
// each shard, and retrieve the actual vectors on demand.
template <typename T>
int shard_data_into_clusters_only_ids(
    const std::string data_file, float *pivots, const size_t num_centers,
    const size_t dim, const size_t k_base, std::string prefix_path) {
  size_t read_blk_size = 64 * 1024 * 1024;
  //  uint64_t write_blk_size = 64 * 1024 * 1024;
  // create cached reader + writer
  cached_ifstream base_reader(data_file, read_blk_size);
  uint32_t npts32;
  uint32_t basedim32;
  base_reader.read((char *)&npts32, sizeof(uint32_t));
  base_reader.read((char *)&basedim32, sizeof(uint32_t));
  size_t num_points = npts32;
  if (basedim32 != dim) {
    diskann::cout << "Error. dimensions dont match for train set and base set"
                  << std::endl;
    return -1;
  }

  std::unique_ptr<size_t[]> shard_counts =
      std::make_unique<size_t[]>(num_centers);

  std::vector<std::ofstream> shard_idmap_writer(num_centers);
  uint32_t dummy_size = 0;
  uint32_t const_one = 1;

  for (size_t i = 0; i < num_centers; i++) {
    std::string idmap_filename =
        prefix_path + "_subshard-" + std::to_string(i) + "_ids_uint32.bin";
    shard_idmap_writer[i] =
        std::ofstream(idmap_filename.c_str(), std::ios::binary);
    shard_idmap_writer[i].write((char *)&dummy_size, sizeof(uint32_t));
    shard_idmap_writer[i].write((char *)&const_one, sizeof(uint32_t));
    shard_counts[i] = 0;
  }

  size_t block_size = num_points <= BLOCK_SIZE ? num_points : BLOCK_SIZE;
  std::unique_ptr<uint32_t[]> block_closest_centers =
      std::make_unique<uint32_t[]>(block_size * k_base);
  std::unique_ptr<T[]> block_data_T = std::make_unique<T[]>(block_size * dim);
  std::unique_ptr<float[]> block_data_float =
      std::make_unique<float[]>(block_size * dim);

  size_t num_blocks = DIV_ROUND_UP(num_points, block_size);

  for (size_t block = 0; block < num_blocks; block++) {
    size_t start_id = block * block_size;
    size_t end_id = (std::min)((block + 1) * block_size, num_points);
    size_t cur_blk_size = end_id - start_id;

    base_reader.read(
        (char *)block_data_T.get(), sizeof(T) * (cur_blk_size * dim));
    diskann::convert_types<T, float>(
        block_data_T.get(), block_data_float.get(), cur_blk_size, dim);

    math_utils::compute_closest_centers(
        block_data_float.get(), cur_blk_size, dim, pivots, num_centers, k_base,
        block_closest_centers.get());

    for (size_t p = 0; p < cur_blk_size; p++) {
      for (size_t p1 = 0; p1 < k_base; p1++) {
        size_t shard_id = block_closest_centers[p * k_base + p1];
        uint32_t original_point_map_id = (uint32_t)(start_id + p);
        shard_idmap_writer[shard_id].write(
            (char *)&original_point_map_id, sizeof(uint32_t));
        shard_counts[shard_id]++;
      }
    }
  }

  size_t total_count = 0;
  diskann::cout << "Actual shard sizes: " << std::flush;
  for (size_t i = 0; i < num_centers; i++) {
    uint32_t cur_shard_count = (uint32_t)shard_counts[i];
    total_count += cur_shard_count;
    diskann::cout << cur_shard_count << " ";
    shard_idmap_writer[i].seekp(0);
    shard_idmap_writer[i].write((char *)&cur_shard_count, sizeof(uint32_t));
    shard_idmap_writer[i].close();
  }

  diskann::cout << "\n Partitioned " << num_points
                << " with replication factor " << k_base << " to get "
                << total_count << " points across " << num_centers << " shards "
                << std::endl;
  return 0;
}

/**
 * Use IndexParameter.
 */
template <typename T>
int shard_data_into_clusters_with_and_without_replica(
    IndexParameter &index_param, float *pivots, const size_t num_centers,
    const size_t dim, const size_t k_base) {
  size_t read_blk_size = 64 * 1024 * 1024;
  //  uint64_t write_blk_size = 64 * 1024 * 1024;
  // create cached reader + writer
  cached_ifstream base_reader(index_param.data_path, read_blk_size);
  uint32_t npts32;
  uint32_t basedim32;
  base_reader.read((char *)&npts32, sizeof(uint32_t));
  base_reader.read((char *)&basedim32, sizeof(uint32_t));
  size_t num_points = npts32;
  if (basedim32 != dim) {
    diskann::cout << "Error. dimensions dont match for train set and base set"
                  << std::endl;
    return -1;
  }

  std::unique_ptr<size_t[]> replica_shard_counts =
      std::make_unique<size_t[]>(num_centers);
  std::unique_ptr<size_t[]> noreplica_shard_counts =
      std::make_unique<size_t[]>(num_centers);

  std::vector<std::ofstream> noreplica_idmap_writer(num_centers);
  std::vector<std::ofstream> replica_idmap_writer(num_centers);
  uint32_t dummy_size = 0;
  uint32_t const_one = 1;

  for (size_t i = 0; i < num_centers; i++) {
    replica_idmap_writer[i] = std::ofstream(
        index_param.part_replica_idmap_filename[i].c_str(), std::ios::binary);
    replica_idmap_writer[i].write((char *)&dummy_size, sizeof(uint32_t));
    replica_idmap_writer[i].write((char *)&const_one, sizeof(uint32_t));

    noreplica_idmap_writer[i] = std::ofstream(
        index_param.part_noreplica_idmap_filename[i].c_str(), std::ios::binary);
    noreplica_idmap_writer[i].write((char *)&dummy_size, sizeof(uint32_t));
    noreplica_idmap_writer[i].write((char *)&const_one, sizeof(uint32_t));

    replica_shard_counts[i] = noreplica_shard_counts[i] = 0;
  }

  size_t block_size = num_points <= BLOCK_SIZE ? num_points : BLOCK_SIZE;
  std::unique_ptr<uint32_t[]> block_closest_centers =
      std::make_unique<uint32_t[]>(block_size * k_base);
  std::unique_ptr<T[]> block_data_T = std::make_unique<T[]>(block_size * dim);
  std::unique_ptr<float[]> block_data_float =
      std::make_unique<float[]>(block_size * dim);

  size_t num_blocks = DIV_ROUND_UP(num_points, block_size);


  math_utils::ClusterBalancer balancer(0.1, 0.95, 0.00, 1.0);
  // no replica / replica cluster stats.
  std::vector<math_utils::ClusterInfo> norp_cluster_stats(num_centers);
  std::vector<math_utils::ClusterInfo> rp_cluster_stats(num_centers);
  for (size_t j = 0; j < num_centers; ++j) {
    norp_cluster_stats[j] = {
      .count = 0,
      .totalDist = 0,  
      .maxDist = 0 
    };
    rp_cluster_stats[j] = {
      .count = 0,
      .totalDist = 0,  
      .maxDist = 0 
    };
  }

  for (size_t block = 0; block < num_blocks; block++) {
    size_t start_id = block * block_size;
    size_t end_id = (std::min)((block + 1) * block_size, num_points);
    size_t cur_blk_size = end_id - start_id;

    base_reader.read(
        (char *)block_data_T.get(), sizeof(T) * (cur_blk_size * dim));
    diskann::convert_types<T, float>(
        block_data_T.get(), block_data_float.get(), cur_blk_size, dim);

    math_utils::compute_balance_norp_rp_closest_centers(
      block_data_float.get(), cur_blk_size, dim, pivots, num_centers, k_base,
      block_closest_centers.get(), balancer, norp_cluster_stats, 
      rp_cluster_stats, index_param.data_type);

    for (size_t p = 0; p < cur_blk_size; p++) {
      for (size_t p1 = 0; p1 < k_base; p1++) {
        size_t shard_id = block_closest_centers[p * k_base + p1];
        uint32_t original_point_map_id = (uint32_t)(start_id + p);
        if (p1 == 0) {
          noreplica_idmap_writer[shard_id].write(
              (char *)&original_point_map_id, sizeof(uint32_t));
          noreplica_shard_counts[shard_id]++;
        }
        replica_idmap_writer[shard_id].write(
            (char *)&original_point_map_id, sizeof(uint32_t));
        replica_shard_counts[shard_id]++;
      }
    }
  }

  size_t total_count = 0;
  diskann::cout << "Actual no-replica shard sizes: " << std::flush;
  for (size_t i = 0; i < num_centers; i++) {
    uint32_t cur_shard_count = (uint32_t)noreplica_shard_counts[i];
    total_count += cur_shard_count;
    diskann::cout << cur_shard_count << " ";
    noreplica_idmap_writer[i].seekp(0);
    noreplica_idmap_writer[i].write((char *)&cur_shard_count, sizeof(uint32_t));
    noreplica_idmap_writer[i].close();
  }
  diskann::cout << "\n Partitioned " << num_points
                << " with no-replication to get " << total_count
                << " points across " << num_centers << " shards " << std::endl;

  total_count = 0;
  diskann::cout << "Actual replica shard sizes: " << std::flush;
  for (size_t i = 0; i < num_centers; i++) {
    uint32_t cur_shard_count = (uint32_t)replica_shard_counts[i];
    total_count += cur_shard_count;
    diskann::cout << cur_shard_count << " ";
    replica_idmap_writer[i].seekp(0);
    replica_idmap_writer[i].write((char *)&cur_shard_count, sizeof(uint32_t));
    replica_idmap_writer[i].close();
  }
  diskann::cout << "\n Partitioned " << num_points
                << " with replication factor " << k_base << " to get "
                << total_count << " points across " << num_centers << " shards "
                << std::endl;

  return 0;
}

/**
 * with replica balance cluster.
 */
template <typename T>
int shard_data_into_clusters_with_replica(
    IndexParameter &index_param, float *pivots, const size_t num_centers,
    const size_t dim, const size_t k_base) {
  size_t read_blk_size = 64 * 1024 * 1024;
  //  uint64_t write_blk_size = 64 * 1024 * 1024;
  // create cached reader + writer
  cached_ifstream base_reader(index_param.data_path, read_blk_size);
  uint32_t npts32;
  uint32_t basedim32;
  base_reader.read((char *)&npts32, sizeof(uint32_t));
  base_reader.read((char *)&basedim32, sizeof(uint32_t));
  size_t num_points = npts32;
  if (basedim32 != dim) {
    diskann::cout << "Error. dimensions dont match for train set and base set"
                  << std::endl;
    return -1;
  }

  std::unique_ptr<size_t[]> replica_shard_counts =
      std::make_unique<size_t[]>(num_centers);

  std::vector<std::ofstream> replica_idmap_writer(num_centers);
  uint32_t dummy_size = 0;
  uint32_t const_one = 1;

  for (size_t i = 0; i < num_centers; i++) {
    replica_idmap_writer[i] = std::ofstream(
        index_param.part_replica_idmap_filename[i].c_str(), std::ios::binary);
    replica_idmap_writer[i].write((char *)&dummy_size, sizeof(uint32_t));
    replica_idmap_writer[i].write((char *)&const_one, sizeof(uint32_t));

    replica_shard_counts[i] = 0;
  }

  size_t block_size = num_points <= BLOCK_SIZE ? num_points : BLOCK_SIZE;
  std::unique_ptr<uint32_t[]> block_closest_centers =
      std::make_unique<uint32_t[]>(block_size * k_base);
  std::unique_ptr<T[]> block_data_T = std::make_unique<T[]>(block_size * dim);
  std::unique_ptr<float[]> block_data_float =
      std::make_unique<float[]>(block_size * dim);

  size_t num_blocks = DIV_ROUND_UP(num_points, block_size);

  // atomic size counter.
  // std::vector<std::atomic<size_t>> cluster_counters(num_centers);
  // for (auto &counter : cluster_counters) counter.store(0);
  // try lambda refine.
  // math_utils::LambdaController lambda_ctl(0.5, 10.0, 0.0001, 0.95);
  math_utils::ClusterBalancer balancer(0.1, 0.95, 0.00, 1.0);
  std::vector<math_utils::ClusterInfo> cluster_stats(num_centers);
  for (size_t j = 0; j < num_centers; ++j) {
    cluster_stats[j] = {
        .count = 0,
        .totalDist = 0,  
        .maxDist = 0 
    };
  }

  for (size_t block = 0; block < num_blocks; block++) {
    // std::cout << "Retrieving cluster .. " << (float)block / num_blocks * 100
    //           << "% ,Current lambda: " << lambda_ctl.get_lambda() << " \r"
    //           << std::flush;
    size_t start_id = block * block_size;
    size_t end_id = (std::min)((block + 1) * block_size, num_points);
    size_t cur_blk_size = end_id - start_id;

    base_reader.read(
        (char *)block_data_T.get(), sizeof(T) * (cur_blk_size * dim));
    diskann::convert_types<T, float>(
        block_data_T.get(), block_data_float.get(), cur_blk_size, dim);

    math_utils::compute_balance_closest_centers(
        block_data_float.get(), cur_blk_size, dim, pivots, num_centers, k_base,
        block_closest_centers.get(), balancer, cluster_stats);
    
    // [without balancer]
    // math_utils::compute_closest_centers(
    //   block_data_float.get(), cur_blk_size, dim, pivots, num_centers, k_base,
    //   block_closest_centers.get());

    for (size_t p = 0; p < cur_blk_size; p++) {
      for (size_t p1 = 0; p1 < k_base; p1++) {
        size_t shard_id = block_closest_centers[p * k_base + p1];
        uint32_t original_point_map_id = (uint32_t)(start_id + p);
        replica_idmap_writer[shard_id].write(
            (char *)&original_point_map_id, sizeof(uint32_t));
        replica_shard_counts[shard_id]++;
      }
    }
  }

  size_t total_count = 0;
  diskann::cout << "Actual replica shard sizes: " << std::flush;
  for (size_t i = 0; i < num_centers; i++) {
    uint32_t cur_shard_count = (uint32_t)replica_shard_counts[i];
    total_count += cur_shard_count;
    diskann::cout << cur_shard_count << " ";
    replica_idmap_writer[i].seekp(0);
    replica_idmap_writer[i].write((char *)&cur_shard_count, sizeof(uint32_t));
    replica_idmap_writer[i].close();
  }
  diskann::cout << "\n Partitioned " << num_points
                << " with replication factor " << k_base << " to get "
                << total_count << " points across " << num_centers << " shards "
                << std::endl;

  return 0;
}

/**
 * with replica balance cluster.
 */
template <typename T>
int shard_data_into_clusters_without_replica(
    IndexParameter &index_param, float *pivots, const size_t num_centers,
    const size_t dim, const size_t k_base) {
  size_t read_blk_size = 64 * 1024 * 1024;
  //  uint64_t write_blk_size = 64 * 1024 * 1024;
  // create cached reader + writer
  cached_ifstream base_reader(index_param.data_path, read_blk_size);
  uint32_t npts32;
  uint32_t basedim32;
  base_reader.read((char *)&npts32, sizeof(uint32_t));
  base_reader.read((char *)&basedim32, sizeof(uint32_t));
  size_t num_points = npts32;
  if (basedim32 != dim) {
    diskann::cout << "Error. dimensions dont match for train set and base set"
                  << std::endl;
    return -1;
  }

  std::unique_ptr<size_t[]> noreplica_shard_counts =
      std::make_unique<size_t[]>(num_centers);

  std::vector<std::ofstream> noreplica_idmap_writer(num_centers);
  uint32_t dummy_size = 0;
  uint32_t const_one = 1;

  for (size_t i = 0; i < num_centers; i++) {
    noreplica_idmap_writer[i] = std::ofstream(
        index_param.part_noreplica_idmap_filename[i].c_str(), std::ios::binary);
    noreplica_idmap_writer[i].write((char *)&dummy_size, sizeof(uint32_t));
    noreplica_idmap_writer[i].write((char *)&const_one, sizeof(uint32_t));

    noreplica_shard_counts[i] = 0;
  }

  size_t block_size = num_points <= BLOCK_SIZE ? num_points : BLOCK_SIZE;
  std::unique_ptr<uint32_t[]> block_closest_centers =
      std::make_unique<uint32_t[]>(block_size * k_base);
  std::unique_ptr<T[]> block_data_T = std::make_unique<T[]>(block_size * dim);
  std::unique_ptr<float[]> block_data_float =
      std::make_unique<float[]>(block_size * dim);

  size_t num_blocks = DIV_ROUND_UP(num_points, block_size);

  // atomic size counter.
  // std::vector<std::atomic<size_t>> cluster_counters(num_centers);
  // for (auto &counter : cluster_counters) counter.store(0);
  // // try lambda refine.
  // math_utils::LambdaController lambda_ctl(0.5, 10.0, 0.0001, 0.95);

  math_utils::ClusterBalancer balancer(0.1, 0.95, 0.00, 1.0);
  std::vector<math_utils::ClusterInfo> cluster_stats(num_centers);
  for (size_t j = 0; j < num_centers; ++j) {
    cluster_stats[j] = {
        .count = 0,
        .totalDist = 0,  
        .maxDist = 0 
    };
  }

  for (size_t block = 0; block < num_blocks; block++) {
    size_t start_id = block * block_size;
    size_t end_id = (std::min)((block + 1) * block_size, num_points);
    size_t cur_blk_size = end_id - start_id;

    base_reader.read(
        (char *)block_data_T.get(), sizeof(T) * (cur_blk_size * dim));
    diskann::convert_types<T, float>(
        block_data_T.get(), block_data_float.get(), cur_blk_size, dim);

    math_utils::compute_balance_closest_centers(
        block_data_float.get(), cur_blk_size, dim, pivots, num_centers, k_base,
        block_closest_centers.get(), balancer, cluster_stats);

    // [without balancer]
    // math_utils::compute_closest_centers(
    //   block_data_float.get(), cur_blk_size, dim, pivots, num_centers, k_base,
    //   block_closest_centers.get());

    for (size_t p = 0; p < cur_blk_size; p++) {
      for (size_t p1 = 0; p1 < k_base; p1++) {
        size_t shard_id = block_closest_centers[p * k_base + p1];
        uint32_t original_point_map_id = (uint32_t)(start_id + p);
        if (p1 == 0) {
          noreplica_idmap_writer[shard_id].write(
              (char *)&original_point_map_id, sizeof(uint32_t));
          noreplica_shard_counts[shard_id]++;
        }
      }
    }
  }

  size_t total_count = 0;
  diskann::cout << "Actual no-replica shard sizes: " << std::flush;
  for (size_t i = 0; i < num_centers; i++) {
    uint32_t cur_shard_count = (uint32_t)noreplica_shard_counts[i];
    total_count += cur_shard_count;
    diskann::cout << cur_shard_count << " ";
    noreplica_idmap_writer[i].seekp(0);
    noreplica_idmap_writer[i].write((char *)&cur_shard_count, sizeof(uint32_t));
    noreplica_idmap_writer[i].close();
  }
  diskann::cout << "\n Partitioned " << num_points
                << " with no-replication to get " << total_count
                << " points across " << num_centers << " shards " << std::endl;

  return 0;
}

/**
 * Random partition.
 */
template <typename T>
int shard_data_into_random(IndexParameter &index_param, const size_t num_centers,
    const size_t dim) {
  size_t read_blk_size = 64 * 1024 * 1024;
  //  uint64_t write_blk_size = 64 * 1024 * 1024;
  // create cached reader + writer
  cached_ifstream base_reader(index_param.data_path, read_blk_size);
  uint32_t npts32;
  uint32_t basedim32;
  base_reader.read((char *)&npts32, sizeof(uint32_t));
  base_reader.read((char *)&basedim32, sizeof(uint32_t));
  size_t num_points = npts32;
  if (basedim32 != dim) {
    diskann::cout << "Error. dimensions dont match for train set and base set"
                  << std::endl;
    return -1;
  }

  std::unique_ptr<size_t[]> shard_counts =
      std::make_unique<size_t[]>(num_centers);

  std::vector<std::ofstream> shard_idmap_writer(num_centers);
  uint32_t dummy_size = 0;
  uint32_t const_one = 1;

  for (size_t i = 0; i < num_centers; i++) {
    std::string idmap_filename = index_param.part_noreplica_idmap_filename[i];
        // prefix_path + "_subshard-" + std::to_string(i) + "_ids_uint32.bin";
    shard_idmap_writer[i] =
        std::ofstream(idmap_filename.c_str(), std::ios::binary);
    shard_idmap_writer[i].write((char *)&dummy_size, sizeof(uint32_t));
    shard_idmap_writer[i].write((char *)&const_one, sizeof(uint32_t));
    shard_counts[i] = 0;
  }

  size_t block_size = num_points <= BLOCK_SIZE ? num_points : BLOCK_SIZE;
  std::unique_ptr<T[]> block_data_T = std::make_unique<T[]>(block_size * dim);
  std::unique_ptr<float[]> block_data_float =
      std::make_unique<float[]>(block_size * dim);

  size_t num_blocks = DIV_ROUND_UP(num_points, block_size);

  for (size_t block = 0; block < num_blocks; block++) {
    size_t start_id = block * block_size;
    size_t end_id = (std::min)((block + 1) * block_size, num_points);
    size_t cur_blk_size = end_id - start_id;

    base_reader.read(
        (char *)block_data_T.get(), sizeof(T) * (cur_blk_size * dim));
    diskann::convert_types<T, float>(
        block_data_T.get(), block_data_float.get(), cur_blk_size, dim);

    for (size_t p = 0; p < cur_blk_size; p++) {
      size_t shard_id = (start_id + p) / (npts32 / num_centers);
      uint32_t original_point_map_id = (uint32_t)(start_id + p);
      shard_idmap_writer[shard_id].write(
          (char *)&original_point_map_id, sizeof(uint32_t));
      shard_counts[shard_id]++;
    }

  }

  for (uint32_t m = 0; m < num_centers; m++) {
    index_param.partition_size[m] = index_param.vec_num/num_centers;
  }

  size_t total_count = 0;
  diskann::cout << "Actual shard sizes: " << std::flush;
  for (size_t i = 0; i < num_centers; i++) {
    uint32_t cur_shard_count = (uint32_t)shard_counts[i];
    total_count += cur_shard_count;
    diskann::cout << cur_shard_count << " ";
    shard_idmap_writer[i].seekp(0);
    shard_idmap_writer[i].write((char *)&cur_shard_count, sizeof(uint32_t));
    shard_idmap_writer[i].close();
  }

  diskann::cout << "\n Random Partitioned " << num_points
                << " to get "
                << total_count << " points across " << num_centers << " shards "
                << std::endl;
  return 0;
}

template <typename T>
int retrieve_shard_data_from_ids(
    const std::string data_file, std::string idmap_filename,
    std::string data_filename) {
  size_t read_blk_size = 64 * 1024 * 1024;
  //  uint64_t write_blk_size = 64 * 1024 * 1024;
  // create cached reader + writer
  if(!std::filesystem::exists(data_file) || !std::filesystem::exists(idmap_filename)){
    std::cerr<<"Error: not find the data_file or idmap_filename\n";
    abort();
  }
  cached_ifstream base_reader(data_file, read_blk_size);
  uint32_t npts32;
  uint32_t basedim32;
  base_reader.read((char *)&npts32, sizeof(uint32_t));
  base_reader.read((char *)&basedim32, sizeof(uint32_t));
  size_t num_points = npts32;
  size_t dim = basedim32;

  uint32_t dummy_size = 0;

  std::ofstream shard_data_writer(data_filename.c_str(), std::ios::binary);
  shard_data_writer.write((char *)&dummy_size, sizeof(uint32_t));
  shard_data_writer.write((char *)&basedim32, sizeof(uint32_t));

  uint32_t *shard_ids;
  uint64_t shard_size, tmp;
  diskann::load_bin<uint32_t>(idmap_filename, shard_ids, shard_size, tmp);

  uint32_t cur_pos = 0;
  uint32_t num_written = 0;
  std::cout << "Shard has " << shard_size << " points" << std::endl;

  size_t block_size = num_points <= BLOCK_SIZE ? num_points : BLOCK_SIZE;
  std::unique_ptr<T[]> block_data_T = std::make_unique<T[]>(block_size * dim);

  size_t num_blocks = DIV_ROUND_UP(num_points, block_size);

  for (size_t block = 0; block < num_blocks; block++) {
    size_t start_id = block * block_size;
    size_t end_id = (std::min)((block + 1) * block_size, num_points);
    size_t cur_blk_size = end_id - start_id;

    base_reader.read(
        (char *)block_data_T.get(), sizeof(T) * (cur_blk_size * dim));

    for (size_t p = 0; p < cur_blk_size; p++) {
      uint32_t original_point_map_id = (uint32_t)(start_id + p);
      if (cur_pos == shard_size) break;
      if (original_point_map_id == shard_ids[cur_pos]) {
        cur_pos++;
        shard_data_writer.write(
            (char *)(block_data_T.get() + p * dim), sizeof(T) * dim);
        num_written++;
      }
    }
    if (cur_pos == shard_size) break;
  }

  diskann::cout << "Written file with " << num_written << " points"
                << std::endl;

  shard_data_writer.seekp(0);
  shard_data_writer.write((char *)&num_written, sizeof(uint32_t));
  shard_data_writer.close();
  delete[] shard_ids;
  return 0;
}

/**
 * [Current method]
 * New function for retrieving all shard within one traversal.
 * write_noreplica: whether write noreplica shards, default is true.
 */
template <typename T>
int retrieve_shard_data_from_ids(
    IndexParameter &index_param, uint32_t k_base, bool write_noreplica) {
  size_t read_blk_size = 64 * 1024 * 1024;
  //  uint64_t write_blk_size = 64 * 1024 * 1024;
  // create cached reader + writer
  // cached_ifstream base_reader(data_file, read_blk_size);
  std::ifstream base_reader(index_param.data_path.c_str(), std::ios::binary);
  uint32_t npts32;
  uint32_t basedim32;
  base_reader.read((char *)&npts32, sizeof(uint32_t));
  base_reader.read((char *)&basedim32, sizeof(uint32_t));
  size_t num_points = npts32;
  size_t dim = basedim32;

  uint32_t dummy_size = 0;


  std::vector<std::ofstream> shard_replica_data_writer;
  if(k_base > 1){
    for (uint32_t p = 0; p < index_param.num_parts; p++) {
      shard_replica_data_writer.emplace_back(
          index_param.part_replica_base_filename[p]);
      if (!shard_replica_data_writer.back().is_open()) {
        std::cerr << "Failed to open file: "
                  << index_param.part_replica_base_filename[p] << std::endl;
        return 1;
      }
    }
  }
  

  std::vector<std::ofstream> shard_noreplica_data_writer;
  if (write_noreplica) {
    for (uint32_t p = 0; p < index_param.num_parts; p++) {
      shard_noreplica_data_writer.emplace_back(
          index_param.part_noreplica_base_filename[p]);
      if (!shard_noreplica_data_writer.back().is_open()) {
        std::cerr << "Failed to open file: "
                  << index_param.part_noreplica_base_filename[p] << std::endl;
        return 1;
      }
    }
  }

  // std::ofstream shard_data_writer(data_filename.c_str(), std::ios::binary);
  // write meta
  for (uint32_t p = 0; p < index_param.num_parts; p++) {
    if(k_base > 1){
      shard_replica_data_writer[p].write((char *)&dummy_size, sizeof(uint32_t));
      shard_replica_data_writer[p].write((char *)&basedim32, sizeof(uint32_t));
    }
    if (write_noreplica) {
      shard_noreplica_data_writer[p].write(
          (char *)&dummy_size, sizeof(uint32_t));
      shard_noreplica_data_writer[p].write(
          (char *)&basedim32, sizeof(uint32_t));
    }
  }

  uint32_t *shard_replica_ids[index_param.num_parts];
  uint32_t *shard_noreplica_ids[index_param.num_parts];
  uint64_t shard_replica_size[index_param.num_parts];
  uint64_t shard_noreplica_size[index_param.num_parts];

  for (uint32_t p = 0; p < index_param.num_parts; p++) {
    uint64_t tmp;
    if(k_base > 1){
      diskann::load_bin<uint32_t>(
          index_param.part_replica_idmap_filename[p], shard_replica_ids[p],
          shard_replica_size[p], tmp);
      std::cout << "Shard " << p << " replica has " << shard_replica_size[p]
                << " points" << std::endl;
    }
    if (write_noreplica) {
      diskann::load_bin<uint32_t>(
          index_param.part_noreplica_idmap_filename[p], shard_noreplica_ids[p],
          shard_noreplica_size[p], tmp);
      std::cout << "Shard " << p << " noreplica has " << shard_noreplica_size[p]
                << " points" << std::endl;
    }
  }

  // uint32_t *shard_ids;
  // uint64_t shard_size, tmp;
  // diskann::load_bin<uint32_t>(idmap_filename, shard_ids, shard_size, tmp);

  uint32_t cur_rp_pos[index_param.num_parts];
  uint32_t num_rp_written[index_param.num_parts];
  memset(cur_rp_pos, 0, sizeof(cur_rp_pos));
  memset(num_rp_written, 0, sizeof(num_rp_written));

  uint32_t cur_norp_pos[index_param.num_parts];
  uint32_t num_norp_written[index_param.num_parts];
  memset(cur_norp_pos, 0, sizeof(cur_norp_pos));
  memset(num_norp_written, 0, sizeof(num_norp_written));
  // std::cout << "Shard has " << shard_size << " points" << std::endl;

  size_t block_size = num_points <= BLOCK_SIZE ? num_points : BLOCK_SIZE;
  std::unique_ptr<T[]> block_data_T = std::make_unique<T[]>(block_size * dim);
  if(dim != index_param.dim){
    std::cerr<<"Error: mismatch dim \n";
    abort();
  }

  size_t num_blocks = DIV_ROUND_UP(num_points, block_size);

  printf("<<>> sizeof read granularity: %u\n", sizeof(T)  * dim);

  for (size_t block = 0; block < num_blocks; block++) {
    size_t start_id = block * block_size;
    size_t end_id = (std::min)((block + 1) * block_size, num_points);
    size_t cur_blk_size = end_id - start_id;

    base_reader.read(
        (char *)block_data_T.get(), sizeof(T) * (cur_blk_size * dim));

    for (size_t v = 0; v < cur_blk_size; v++) {
      uint32_t original_point_map_id = (uint32_t)(start_id + v);
      // if (cur_pos == shard_size) break;
      for (uint32_t p = 0; p < index_param.num_parts; p++) {
        if(k_base > 1){
          if (original_point_map_id == shard_replica_ids[p][cur_rp_pos[p]]) {
            cur_rp_pos[p]++;
            shard_replica_data_writer[p].write(
                (char *)(block_data_T.get() + v * dim), sizeof(T) * dim);
            num_rp_written[p]++;
          }
        }
        if (write_noreplica) {
          if (original_point_map_id ==
              shard_noreplica_ids[p][cur_norp_pos[p]]) {
            cur_norp_pos[p]++;
            shard_noreplica_data_writer[p].write(
                (char *)(block_data_T.get() + v * dim), sizeof(T) * dim);
            num_norp_written[p]++;
          }
        }
      }
    }

    std::cout<<"Data retrieval "<< (float)end_id*100.0/num_points 
      <<"% finished\r"<<std::flush;
    // here we sleep for avoiding our shared servers meet problem.
    sleep(3);
  }

  for (uint32_t p = 0; p < index_param.num_parts; p++) {
    if(k_base > 1){
      std::cout << "Shard " << p << " replica written " << num_rp_written[p]
                << " points" << std::endl;
      shard_replica_data_writer[p].seekp(0);
      shard_replica_data_writer[p].write(
          (char *)&num_rp_written[p], sizeof(uint32_t));
      shard_replica_data_writer[p].close();
    }
    if (write_noreplica) {
      std::cout << "Shard " << p << " noreplica written " << num_norp_written[p]
                << " points" << std::endl;
      shard_noreplica_data_writer[p].seekp(0);
      shard_noreplica_data_writer[p].write(
          (char *)&num_norp_written[p], sizeof(uint32_t));
      shard_noreplica_data_writer[p].close();
    }
  }

  
  for(uint32_t p = 0; p<index_param.num_parts; p++){
    if(k_base > 1){
      if(shard_replica_size[p] != num_rp_written[p] || 
        shard_noreplica_size[p] != num_norp_written[p]){
          std::cerr<<"Error: write shard num not equal to expected.\n";
          abort();
      }
    }
    else {
      if(shard_noreplica_size[p] != num_norp_written[p]){
          std::cerr<<"Error: write shard num not equal to expected.\n";
          abort();
      }
    }
  }
  // diskann::cout << "Written file with " << num_written << " points"
  //               << std::endl;

  // shard_data_writer.seekp(0);
  // shard_data_writer.write((char *)&num_written, sizeof(uint32_t));
  // shard_data_writer.close();
  for (int i = 0; i < index_param.num_parts; i++) {
    if(k_base > 1){
      if (shard_replica_ids[i] != nullptr) {
        delete[] shard_replica_ids[i];
        shard_replica_ids[i] = nullptr;
      }
    }
    
    if (shard_noreplica_ids[i] != nullptr) {
      delete[] shard_noreplica_ids[i];
      shard_noreplica_ids[i] = nullptr;
    }
  }

  std::cout << "Data retriveal over. \n";
  // delete[] shard_replica_ids;
  // delete[] shard_noreplica_ids;
  return 0;
}

// partitions a large base file into many shards using k-means hueristic
// on a random sample generated using sampling_rate probability. After this, it
// assignes each base point to the closest k_base nearest centers and creates
// the shards.
// The total number of points across all shards will be k_base * num_points.

template <typename T>
int partition(
    const std::string data_file, const float sampling_rate, size_t num_parts,
    size_t max_k_means_reps, const std::string prefix_path, size_t k_base) {
  size_t train_dim;
  size_t num_train;
  float *train_data_float;

  gen_random_slice<T>(
      data_file, sampling_rate, train_data_float, num_train, train_dim);

  float *pivot_data;

  std::string cur_file = std::string(prefix_path);
  std::string output_file;

  // kmeans_partitioning on training data

  //  cur_file = cur_file + "_kmeans_partitioning-" +
  //  std::to_string(num_parts);
  output_file = cur_file + "_centroids.bin";

  pivot_data = new float[num_parts * train_dim];

  // Process Global k-means for kmeans_partitioning Step
  diskann::cout << "Processing global k-means (kmeans_partitioning Step)"
                << std::endl;
  kmeans::kmeanspp_selecting_pivots(
      train_data_float, num_train, train_dim, pivot_data, num_parts);

  kmeans::run_lloyds(
      train_data_float, num_train, train_dim, pivot_data, num_parts,
      max_k_means_reps, NULL, NULL);

  diskann::cout << "Saving global k-center pivots" << std::endl;
  diskann::save_bin<float>(
      output_file.c_str(), pivot_data, (size_t)num_parts, train_dim);

  // now pivots are ready. need to stream base points and assign them to
  // closest clusters.

  shard_data_into_clusters<T>(
      data_file, pivot_data, num_parts, train_dim, k_base, prefix_path);
  delete[] pivot_data;
  delete[] train_data_float;
  return 0;
}

void read_idmap(const std::string &fname, std::vector<uint32_t> &ivecs) {
  uint32_t npts32, dim;
  size_t actual_file_size = get_file_size(fname);
  std::ifstream reader(fname.c_str(), std::ios::binary);
  reader.read((char *)&npts32, sizeof(uint32_t));
  reader.read((char *)&dim, sizeof(uint32_t));
  if (dim != 1 || actual_file_size != ((size_t)npts32) * sizeof(uint32_t) +
                                          2 * sizeof(uint32_t)) {
    std::stringstream stream;
    stream << "Error reading idmap file. Check if the file is bin file with "
              "1 dimensional data. Actual: "
           << actual_file_size
           << ", expected: " << (size_t)npts32 + 2 * sizeof(uint32_t)
           << std::endl;

    throw diskann::ANNException(
        stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
  }
  ivecs.resize(npts32);
  reader.read((char *)ivecs.data(), ((size_t)npts32) * sizeof(uint32_t));
  reader.close();
}

template <typename T>
int random_slice_parts(IndexParameter &index_param, 
    uint32_t target_parts) {
  // size_t train_dim;
  // size_t num_train;
  // float *train_data_float;
  // size_t max_k_means_reps = 15;

  // int num_parts = 3;
  // int num_parts = k_base + 1;
  int num_parts = target_parts;
  // bool fit_in_parts = false;
  // std::string shard_ids_file = prefix_path + "_ids_uint32.bin";
  // double pval = save_ids ? 1 : sampling_rate;
  // gen_random_slice<T>(data_file, pval, train_data_float, num_train, train_dim);

  // size_t test_dim;
  // size_t num_test;
  // float *test_data_float;
  // gen_random_slice<T>(data_file, sampling_rate, test_data_float, num_test,
  // test_dim);

  // float *pivot_data = nullptr;

  // std::string cur_file = std::string(prefix_path);
  // std::string output_file;

  // kmeans_partitioning on training data
  // output_file = cur_file + "_centroids.bin";

  // diskann::cout << "Saving global k-center pivots" << std::endl;

  // substite pivots with medoids that really represent the cluster
  shard_data_into_random<T>(
    index_param, num_parts, index_param.dim);

  // delete[] pivot_data;
  // delete[] train_data_float;
  // delete[] test_data_float;
  return num_parts;
}

template <typename T>
int partition_with_target_parts(
    const std::string data_file, const double sampling_rate, double ram_budget,
    const std::string prefix_path, size_t k_base,
    uint32_t target_parts, bool save_ids) {
  size_t train_dim;
  size_t num_train;
  float *train_data_float;
  size_t max_k_means_reps = 15;

  // int num_parts = 3;
  // int num_parts = k_base + 1;
  int num_parts = target_parts;
  bool fit_in_parts = false;
  std::string shard_ids_file = prefix_path + "_ids_uint32.bin";
  double pval = save_ids ? 1 : sampling_rate;
  gen_random_slice<T>(data_file, pval, train_data_float, num_train, train_dim);

  size_t test_dim;
  size_t num_test;
  // float *test_data_float;
  // gen_random_slice<T>(data_file, sampling_rate, test_data_float, num_test,
  // test_dim);

  float *pivot_data = nullptr;

  std::string cur_file = std::string(prefix_path);
  std::string output_file;

  // kmeans_partitioning on training data

  //  cur_file = cur_file + "_kmeans_partitioning-" +
  //  std::to_string(num_parts);
  output_file = cur_file + "_centroids.bin";

  while (!fit_in_parts) {
    // fit_in_ram = true;

    double max_ram_usage = 0;
    if (pivot_data != nullptr) delete[] pivot_data;

    pivot_data = new float[num_parts * train_dim];
    // Process Global k-means for kmeans_partitioning Step
    diskann::cout << "Processing global k-means (kmeans_partitioning Step)"
                  << std::endl;
    kmeans::kmeanspp_selecting_pivots(
        train_data_float, num_train, train_dim, pivot_data, num_parts);

    kmeans::run_lloyds(
        train_data_float, num_train, train_dim, pivot_data, num_parts,
        max_k_means_reps, NULL, NULL);

    // now pivots are ready. need to stream base points and assign them to
    // closest clusters.

    // std::vector<size_t> cluster_sizes;
    // estimate_cluster_sizes(test_data_float, num_test, pivot_data, num_parts,
    // train_dim, k_base, cluster_sizes); for (auto &p : cluster_sizes)
    // {
    //     // to account for the fact that p is the size of the shard over the
    //     // testing sample.
    //     p = (uint64_t)(p / sampling_rate);
    //     double cur_shard_ram_estimate =
    //         diskann::estimate_ram_usage(p, (uint32_t)train_dim, sizeof(T),
    //         (uint32_t)graph_degree);

    //     if (cur_shard_ram_estimate > max_ram_usage)
    //         max_ram_usage = cur_shard_ram_estimate;
    // }
    // diskann::cout << "With " << num_parts
    //               << " parts, max estimated RAM usage: " << max_ram_usage /
    //               (1024 * 1024 * 1024)
    //               << "GB, budget given is " << ram_budget << std::endl;
    // if (max_ram_usage > 1024 * 1024 * 1024 * ram_budget)
    // {
    //     fit_in_ram = false;
    //     num_parts += 2;
    // }
    if (num_parts >= target_parts) {
      fit_in_parts = true;
    } else {
      num_parts += 1;
    }
  }

  diskann::cout << "Saving global k-center pivots" << std::endl;

  // substite pivots with medoids that really represent the cluster
  if (save_ids) {
    // TODO: can merge these two functions into one.
    shard_data_into_clusters_only_ids<T>(
        data_file, pivot_data, num_parts, train_dim, k_base, prefix_path);
    shard_data_into_clusters_only_ids<T>(
        data_file, pivot_data, num_parts, train_dim, 1,
        prefix_path + "_meta_index");
  }

  // if (!save_ids) {
  //     // clustering get n centroids, we search the cloest points to them as
  //     medoids. float* all_shard_data = nullptr; size_t get_all_num,
  //     get_all_dim;
  //     // gen_random_slice<T>(data_file, 1.5, all_shard_data, get_all_num,
  //     get_all_dim); diskann::load_bin_to_float<T>(data_file, all_shard_data,
  //     get_all_num, get_all_dim);

  //     faiss::IndexFlatL2 flat_index(get_all_dim);
  //     flat_index.train(get_all_num, all_shard_data);
  //     flat_index.add(get_all_num, all_shard_data);
  //     std::cout << "Current data points count: " << get_all_num << "dim: " <<
  //     get_all_dim << std::endl; std::cout << "shard_data last dim: " <<
  //     all_shard_data[get_all_num * get_all_dim - 1] << std::endl;
  //     faiss::idx_t *res_ids = new faiss::idx_t[num_parts];
  //     float* dists = new float[num_parts];

  //     flat_index.search(num_parts, pivot_data, 1, dists, res_ids);
  //     std::vector<uint32_t> res_vec;
  //     for (size_t i = 0; i < num_parts; i++) {
  //         res_vec.push_back(res_ids[i]);
  //     }

  //     //deduplicate res_vec
  //     std::sort(res_vec.begin(), res_vec.end());
  //     std::vector<uint32_t>::iterator it = std::unique(res_vec.begin(),
  //     res_vec.end()); res_vec.resize(std::distance(res_vec.begin(), it));
  //     num_parts = res_vec.size();
  //     if (pivot_data != nullptr) {
  //         delete[] pivot_data;
  //     }

  //     pivot_data = new float[num_parts * train_dim];

  //     for (size_t i = 0; i < num_parts; i++) {
  //         memcpy(pivot_data + i * train_dim, all_shard_data + res_vec[i] *
  //         train_dim, sizeof(float) * train_dim);
  //     }

  //     std::vector<uint32_t> shard_ids;
  //     read_idmap(shard_ids_file, shard_ids);

  //     for (size_t i = 0; i < num_parts; i++) {
  //         auto a = shard_ids[res_vec[i]];
  //         res_vec[i] = a;
  //     }

  //     // save centroids data
  //     std::vector<T> pivot_data_T;
  //     pivot_data_T.resize(num_parts * train_dim);

  // #pragma omp parallel for
  //     for (size_t i = 0; i < num_parts; i++) {
  //         for (size_t j = 0; j < train_dim; j++) {
  //             pivot_data_T[i * train_dim + j] = static_cast<T>(pivot_data[i *
  //             train_dim + j]);
  //         }
  //     }
  //     diskann::save_bin<T>(output_file.c_str(), pivot_data_T.data(),
  //     (size_t)num_parts, train_dim);

  //     // save centroids ids
  //     std::string idmap_filename = prefix_path + "_centroids_ids_uint32.bin";
  //     std::cout << "Saving centroids ids to " << idmap_filename << std::endl;
  //     std::ofstream ids_writer(idmap_filename.c_str(), std::ios::binary);
  //     uint32_t dummy_size = num_parts;
  //     uint32_t const_one = 1;
  //     ids_writer.write((char *)&dummy_size, sizeof(uint32_t));
  //     ids_writer.write((char *)&const_one, sizeof(uint32_t));
  //     ids_writer.write((char *)res_vec.data(), sizeof(uint32_t) * num_parts);
  //     ids_writer.close();

  //     delete[] res_ids;
  //     delete[] dists;
  // }

  delete[] pivot_data;
  delete[] train_data_float;
  // delete[] test_data_float;
  return num_parts;
}

template <typename T>
int partition_with_target_parts(
    IndexParameter &index_param, const double sampling_rate, double ram_budget,
    uint32_t target_parts, bool save_ids) {
  size_t train_dim;
  size_t num_train;
  float *train_data_float;
  size_t max_k_means_reps = 15;
  size_t k_base = index_param.replica_num;
  std::cout<< "Start partition " << target_parts << " parts with " << k_base << " replica.\n";

  int num_parts = target_parts;
  // double pval = save_ids ? 1 : sampling_rate;
  double pval = sampling_rate;

  std::cout << "start gen random slice" << std::endl;
  gen_random_slice<T>(
      index_param.data_path, pval, train_data_float, num_train, train_dim);
  if(train_dim != index_param.dim) {
    std::cerr << "Error: mismatch file with param dim.\n";
    abort();
  }

  size_t test_dim;
  size_t num_test;
  // float *test_data_float;
  // gen_random_slice<T>(data_file, sampling_rate, test_data_float, num_test,
  // test_dim);

  float *pivot_data = nullptr;

  std::string cur_file = std::string(index_param.merged_index_prefix);
  std::string output_file;

  // kmeans_partitioning on training data

  //  cur_file = cur_file + "_kmeans_partitioning-" +
  //  std::to_string(num_parts);
  output_file = cur_file + "_centroids.bin";

  if (pivot_data != nullptr) delete[] pivot_data;

  pivot_data = new float[num_parts * train_dim];
  auto closest_docs = new std::vector<size_t>[num_parts];
  // Process Global k-means for kmeans_partitioning Step
  diskann::cout << "Processing global k-means (kmeans_partitioning Step)"
                << std::endl;
  kmeans::kmeanspp_selecting_pivots(
      train_data_float, num_train, train_dim, pivot_data, num_parts);

  kmeans::run_lloyds(
      train_data_float, num_train, train_dim, pivot_data, num_parts,
      max_k_means_reps, closest_docs, NULL);
  // now pivots are ready. need to stream base points and assign them to
  // closest clusters.

  diskann::cout << "Saving global k-center pivots" << std::endl;

  // save ids (w/replica) and partition ids (w/o replica).
  // [current solution] cluster together
  if(k_base == 1){ // no-replica 
    shard_data_into_clusters_without_replica<T>(
      index_param, pivot_data, num_parts, train_dim, 1);
  }else {
    shard_data_into_clusters_with_and_without_replica<T>(
      index_param, pivot_data, num_parts, train_dim, k_base);
  }

  delete[] pivot_data;
  delete[] train_data_float;
  // delete[] test_data_float;
  return num_parts;
}

template <typename T>
int partition_with_ram_budget(
    const std::string data_file, const double sampling_rate, double ram_budget,
    size_t graph_degree, const std::string prefix_path, size_t k_base) {
  size_t train_dim;
  size_t num_train;
  float *train_data_float;
  size_t max_k_means_reps = 10;

  // int num_parts = 3;
  int num_parts = k_base + 1;
  bool fit_in_ram = false;

  gen_random_slice<T>(
      data_file, sampling_rate, train_data_float, num_train, train_dim);

  size_t test_dim;
  size_t num_test;
  float *test_data_float;
  gen_random_slice<T>(
      data_file, sampling_rate, test_data_float, num_test, test_dim);

  float *pivot_data = nullptr;

  std::string cur_file = std::string(prefix_path);
  std::string output_file;

  // kmeans_partitioning on training data

  //  cur_file = cur_file + "_kmeans_partitioning-" +
  //  std::to_string(num_parts);
  output_file = cur_file + "_centroids.bin";

  while (!fit_in_ram) {
    fit_in_ram = true;

    double max_ram_usage = 0;
    if (pivot_data != nullptr) delete[] pivot_data;

    pivot_data = new float[num_parts * train_dim];
    // Process Global k-means for kmeans_partitioning Step
    diskann::cout << "Processing global k-means (kmeans_partitioning Step)"
                  << std::endl;
    kmeans::kmeanspp_selecting_pivots(
        train_data_float, num_train, train_dim, pivot_data, num_parts);

    kmeans::run_lloyds(
        train_data_float, num_train, train_dim, pivot_data, num_parts,
        max_k_means_reps, NULL, NULL);

    // now pivots are ready. need to stream base points and assign them to
    // closest clusters.

    std::vector<size_t> cluster_sizes;
    estimate_cluster_sizes(
        test_data_float, num_test, pivot_data, num_parts, train_dim, k_base,
        cluster_sizes);

    for (auto &p : cluster_sizes) {
      // to account for the fact that p is the size of the shard over the
      // testing sample.
      p = (uint64_t)(p / sampling_rate);
      double cur_shard_ram_estimate = diskann::estimate_ram_usage(
          p, (uint32_t)train_dim, sizeof(T), (uint32_t)graph_degree);

      if (cur_shard_ram_estimate > max_ram_usage)
        max_ram_usage = cur_shard_ram_estimate;
    }
    diskann::cout << "With " << num_parts << " parts, max estimated RAM usage: "
                  << max_ram_usage / (1024 * 1024 * 1024)
                  << "GB, budget given is " << ram_budget << std::endl;
    if (max_ram_usage > 1024 * 1024 * 1024 * ram_budget) {
      fit_in_ram = false;
      num_parts += 2;
    }
  }

  diskann::cout << "Saving global k-center pivots" << std::endl;
  diskann::save_bin<float>(
      output_file.c_str(), pivot_data, (size_t)num_parts, train_dim);

  shard_data_into_clusters_only_ids<T>(
      data_file, pivot_data, num_parts, train_dim, k_base, prefix_path);
  delete[] pivot_data;
  delete[] train_data_float;
  delete[] test_data_float;
  return num_parts;
}

// Instantations of supported templates

template void DISKANN_DLLEXPORT gen_random_slice<int8_t>(
    const std::string base_file, const std::string output_prefix,
    double sampling_rate);
template void DISKANN_DLLEXPORT gen_random_slice<uint8_t>(
    const std::string base_file, const std::string output_prefix,
    double sampling_rate);
template void DISKANN_DLLEXPORT gen_random_slice<float>(
    const std::string base_file, const std::string output_prefix,
    double sampling_rate);

template void DISKANN_DLLEXPORT gen_random_slice<float>(
    const float *inputdata, size_t npts, size_t ndims, double p_val,
    float *&sampled_data, size_t &slice_size);
template void DISKANN_DLLEXPORT gen_random_slice<uint8_t>(
    const uint8_t *inputdata, size_t npts, size_t ndims, double p_val,
    float *&sampled_data, size_t &slice_size);
template void DISKANN_DLLEXPORT gen_random_slice<int8_t>(
    const int8_t *inputdata, size_t npts, size_t ndims, double p_val,
    float *&sampled_data, size_t &slice_size);

template void DISKANN_DLLEXPORT gen_random_slice<float>(
    const std::string data_file, double p_val, float *&sampled_data,
    size_t &slice_size, size_t &ndims);
template void DISKANN_DLLEXPORT gen_random_slice<uint8_t>(
    const std::string data_file, double p_val, float *&sampled_data,
    size_t &slice_size, size_t &ndims);
template void DISKANN_DLLEXPORT gen_random_slice<int8_t>(
    const std::string data_file, double p_val, float *&sampled_data,
    size_t &slice_size, size_t &ndims);

template DISKANN_DLLEXPORT int partition<int8_t>(
    const std::string data_file, const float sampling_rate, size_t num_centers,
    size_t max_k_means_reps, const std::string prefix_path, size_t k_base);
template DISKANN_DLLEXPORT int partition<uint8_t>(
    const std::string data_file, const float sampling_rate, size_t num_centers,
    size_t max_k_means_reps, const std::string prefix_path, size_t k_base);
template DISKANN_DLLEXPORT int partition<float>(
    const std::string data_file, const float sampling_rate, size_t num_centers,
    size_t max_k_means_reps, const std::string prefix_path, size_t k_base);

template DISKANN_DLLEXPORT int partition_with_target_parts<int8_t>(
    const std::string data_file, const double sampling_rate, double ram_budget,
    const std::string prefix_path, size_t k_base,
    uint32_t target_parts, bool save_ids);
template DISKANN_DLLEXPORT int partition_with_target_parts<uint8_t>(
    const std::string data_file, const double sampling_rate, double ram_budget,
    const std::string prefix_path, size_t k_base,
    uint32_t target_parts, bool save_ids);
template DISKANN_DLLEXPORT int partition_with_target_parts<float>(
    const std::string data_file, const double sampling_rate, double ram_budget,
    const std::string prefix_path, size_t k_base,
    uint32_t target_parts, bool save_ids);

template DISKANN_DLLEXPORT int partition_with_target_parts<int8_t>(
    IndexParameter &index_param, const double sampling_rate, double ram_budget,
    uint32_t target_parts, bool save_ids);
template DISKANN_DLLEXPORT int partition_with_target_parts<uint8_t>(
    IndexParameter &index_param, const double sampling_rate, double ram_budget,
    uint32_t target_parts, bool save_ids);
template DISKANN_DLLEXPORT int partition_with_target_parts<float>(
    IndexParameter &index_param, const double sampling_rate, double ram_budget,
    uint32_t target_parts, bool save_ids);

template DISKANN_DLLEXPORT int random_slice_parts<int8_t>(
  IndexParameter &index_param, 
  uint32_t target_parts);
template DISKANN_DLLEXPORT int random_slice_parts<uint8_t>(
  IndexParameter &index_param, 
  uint32_t target_parts);
template DISKANN_DLLEXPORT int random_slice_parts<float>(
  IndexParameter &index_param, 
  uint32_t target_parts);

template DISKANN_DLLEXPORT int partition_with_ram_budget<int8_t>(
    const std::string data_file, const double sampling_rate, double ram_budget,
    size_t graph_degree, const std::string prefix_path, size_t k_base);
template DISKANN_DLLEXPORT int partition_with_ram_budget<uint8_t>(
    const std::string data_file, const double sampling_rate, double ram_budget,
    size_t graph_degree, const std::string prefix_path, size_t k_base);
template DISKANN_DLLEXPORT int partition_with_ram_budget<float>(
    const std::string data_file, const double sampling_rate, double ram_budget,
    size_t graph_degree, const std::string prefix_path, size_t k_base);

template DISKANN_DLLEXPORT int retrieve_shard_data_from_ids<float>(
    const std::string data_file, std::string idmap_filename,
    std::string data_filename);
template DISKANN_DLLEXPORT int retrieve_shard_data_from_ids<uint8_t>(
    const std::string data_file, std::string idmap_filename,
    std::string data_filename);
template DISKANN_DLLEXPORT int retrieve_shard_data_from_ids<int8_t>(
    const std::string data_file, std::string idmap_filename,
    std::string data_filename);

template DISKANN_DLLEXPORT int retrieve_shard_data_from_ids<float>(
    IndexParameter &index_param, uint32_t k_base, bool write_noreplica);
template DISKANN_DLLEXPORT int retrieve_shard_data_from_ids<uint8_t>(
    IndexParameter &index_param, uint32_t k_base, bool write_noreplica);
template DISKANN_DLLEXPORT int retrieve_shard_data_from_ids<int8_t>(
    IndexParameter &index_param, uint32_t k_base, bool write_noreplica);
