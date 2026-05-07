#pragma once

#include <atomic>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <filesystem>

#include "anns/hash_table.h"
#include "coromem/include/utils.h"
#include "index/index_param.h"

class TopIndex {
 public:
  std::string path;
  uint32_t vec_num;
  uint32_t hop_num;
  uint32_t *hoplevel_cnt;
  uint32_t last_level_bar;
  char *vec_ptr;
  size_t size_data_per_element_;
  size_t ngh_ofs;
  size_t label_ofs;

  TopIndex() {}

  TopIndex(uint32_t part_num) {
    char top_path[1024];
    snprintf(
        top_path, sizeof(top_path),
        // "/data/share/users/xyzhi/data/bigann/vamana/balance_kmeans_%u_part/"
        // "hnsw_top_index_mem.index",
        // "/data/share/users/xyzhi/data/bigann/hnsw_top_index/base_100M_hnsw_0.001_top_idnex.index",
        "/data/share/users/xyzhi/data/bigann/vamana/balance_kmeans_%u_part/"
        "hnsw_top_index_mem.index",
        part_num);
    path = std::string(top_path);
    show_top_index();
  }

  TopIndex(std::string _path) : path(_path) { show_top_index(); }

  void show_top_index() {
    printf("Loading Top index from %s\n", path.c_str());
    std::ifstream input(path.c_str(), std::ios::binary);

    input.read((char *)&vec_num, sizeof(vec_num));
    input.read((char *)&hop_num, sizeof(hop_num));

    hoplevel_cnt = new uint32_t[hop_num];
    for (uint32_t i = 0; i < hop_num; i++) {
      input.read((char *)&hoplevel_cnt[i], sizeof(hoplevel_cnt[i]));
    }

    vec_ptr = (char *)malloc(vec_num * size_data_per_element_);
    input.read(vec_ptr, vec_num * size_data_per_element_);

    printf("Top index vec_num %u hop_num %u\n", vec_num, hop_num);
    last_level_bar = 0;
    for (uint32_t i = 0; i < hop_num; i++) {
      printf("Hop %u cnt %u\n", i, hoplevel_cnt[i]);
      last_level_bar += hoplevel_cnt[i];
    }
    last_level_bar -= hoplevel_cnt[hop_num - 1];
    printf("Last level bar %u\n", last_level_bar);

    input.close();
  }

  uint32_t *get_ngh_arr(uint32_t vid) {
    return (uint32_t *)(vec_ptr + vid * size_data_per_element_);
  }

  char *get_vector(uint32_t vid) {
    return vec_ptr + vid * size_data_per_element_ + ngh_ofs;
  }

  uint64_t get_global_id(uint32_t vid) {
    uint64_t *label_ptr =
        (uint64_t *)(vec_ptr + vid * size_data_per_element_ + label_ofs);
    return label_ptr[0];
  }

  bool is_last_level(uint32_t vid) { return vid > last_level_bar; }

  // void build_hnsw_top_index(commandLine &cmd);

  // void build_hnsw_top_index(IndexParameter &index_param, commandLine &cmd);
};

class HNSWTopIndex {
 public:
  std::string path;
  uint32_t vec_num;
  uint32_t hop_num;
  uint32_t *hoplevel_cnt;
  uint32_t last_level_bar;
  char *vec_ptr;
  size_t size_data_per_element_;
  size_t ngh_ofs;
  size_t label_ofs;

  // TopIndex() {
  //   // default path.
  //   // path = std::string(
  //   // "/data/share/users/xyzhi/data/bigann/vamana/balance_kmeans_4_part/"
  //   //     "hnsw_top_index_mem.index");
  //   path = std::string(
  //       "/data/share/users/xyzhi/data/bigann/hnsw_top_index/base_100M_hnsw_0.001_top_idnex.index");
  //   show_top_index();
  // }
  HNSWTopIndex() {
    // default path.
    // path = std::string(
    //     "/data/share/users/xyzhi/data/bigann/vamana/balance_kmeans_4_part/"
    //     "hnsw_top_index_mem.index");
    // show_top_index();
  }

  HNSWTopIndex(IndexParameter &index_param) {
    // char top_path[1024];
    // snprintf(
    //     top_path, sizeof(top_path),
    //     // "/data/share/users/xyzhi/data/bigann/vamana/balance_kmeans_%u_part/"
    //     // "hnsw_top_index_mem.index",
    //     // "/data/share/users/xyzhi/data/bigann/hnsw_top_index/base_100M_hnsw_0.001_top_idnex.index",
    //     // "/data/share/users/xyzhi/data/bigann/hnsw_top_index/base_100M_hnsw_0.0001_top_idnex.index",
    //     // "/data/share/users/xyzhi/data/bigann/hnsw_top_index/base_100M_hnsw_0.00001_top_idnex.index",
    //     // "/data/share/users/xyzhi/data/bigann/hnsw_top_index/"
    //     // "base_100M_hnsw_0.01_top_idnex.index",

    //     "/data/share/users/xyzhi/data/bigann/vamana/"
    //     "scala_build_test_10M_4P/merged_index_top.index",

    //     // "/data/share/users/xyzhi/data/bigann/hnsw_top_index/base_100M_hnsw_0.1_top_idnex.index",
    //     // "/data/share/users/xyzhi/data/bigann/vamana/balance_kmeans_%u_part/"
    //     // "hnsw_top_index_mem.index",
    //     part_num);

    path = std::string(index_param.top_index_path);
    load_top_index();
  }

  HNSWTopIndex(std::string _path) : path(_path) { load_top_index(); }

  void load_top_index() {
    printf("Loading Top index from %s\n", path.c_str());
    if(!std::filesystem::exists(path)){
      std::cerr<<"Top index file not exist. \n";
      abort();
    }
    std::ifstream input(path.c_str(), std::ios::binary);

    input.read((char *)&vec_num, sizeof(uint32_t));
    input.read((char *)&hop_num, sizeof(uint32_t));

    printf("Read vec_num %u hop_num %u\n", vec_num, hop_num);

    hoplevel_cnt = new uint32_t[hop_num];
    for (uint32_t i = 0; i < hop_num; i++) {
      input.read((char *)&hoplevel_cnt[i], sizeof(uint32_t));
    }
    input.read((char *)&size_data_per_element_, sizeof(size_t));
    size_t deg;
    input.read((char *)&deg, sizeof(size_t));
    ngh_ofs = (deg + 1) * sizeof(uint32_t);
    input.read((char *)&label_ofs, sizeof(size_t));

    vec_ptr = (char *)malloc(vec_num * size_data_per_element_);
    input.read(vec_ptr, vec_num * size_data_per_element_);

    printf("Top index vec_num %u hop_num %u\n", vec_num, hop_num);
    last_level_bar = 0;
    for (uint32_t i = 0; i < hop_num; i++) {
      printf("Hop %u cnt %u\n", i, hoplevel_cnt[i]);
      last_level_bar += hoplevel_cnt[i];
    }
    last_level_bar -= hoplevel_cnt[hop_num - 1];
    printf("Last level bar %u\n", last_level_bar);

    input.close();
  }

  void build_hnsw_top_index(commandLine &cmd);


  /**
   * The top index is build on base vector data and use the **tranformed** K-means ids to align 
   * with base index.
   */
  void build_hnsw_top_index(IndexParameter &index_param, commandLine &cmd);

  uint32_t *get_ngh_arr(size_t vid) {
    return (uint32_t *)(vec_ptr + vid * size_data_per_element_);
  }

  char *get_vector(size_t vid) {
    return vec_ptr + vid * size_data_per_element_ + ngh_ofs;
  }

  uint64_t get_global_id(size_t vid) {
    uint64_t *label_ptr =
        (uint64_t *)(vec_ptr + vid * size_data_per_element_ + label_ofs);
    return label_ptr[0];
  }

  bool is_last_level(size_t vid) { return vid > last_level_bar; }
};

/**
 * Generate Top cache index based on kmeans vamana.
 */
// void generate_top_index(
//     std::string index_path, uint64_t node_num, std::string top_index_path) {
//   std::ifstream input(index_path.c_str(), std::ios::binary);

//   size_t size_data_per_element_ = 332;
//   size_t data_ofs = (48 + 1) * 4;
//   size_t label_ofs = (48 + 1) * 4 + 128;

//   size_t max_elements_ = 0;
//   input.read((char *)&max_elements_, sizeof(size_t));
//   uint32_t enterpoint_node_ = 0;
//   input.read((char *)&enterpoint_node_, sizeof(uint32_t));

//   printf(
//       "kmeans vamana read max %u enter point %u\n", max_elements_,
//       enterpoint_node_);
//   char *buff = new char[size_data_per_element_ * max_elements_];
//   input.read(buff, size_data_per_element_ * max_elements_);
//   input.close();
//   printf("kmeans vamana read over\n");

//   std::unordered_map<uint32_t, uint32_t> hash;

//   const uint32_t hnum = 6;
//   std::vector<uint32_t> ans;
//   std::vector<uint32_t> hop_nodes[hnum];
//   uint32_t cache_cnt = 0;
//   hop_nodes[0].emplace_back(enterpoint_node_);
//   hash[enterpoint_node_] = ans.size();
//   ans.emplace_back(enterpoint_node_);

//   uint32_t hoplevel_num[hnum];
//   hoplevel_num[0] = hop_nodes[0].size();

//   for (uint32_t hop = 1; hop < hnum; hop++) {
//     for (uint32_t node : hop_nodes[hop - 1]) {
//       uint32_t *ptr = (uint32_t *)(buff + size_data_per_element_ * node);
//       uint32_t deg = ptr[0];
//       for (uint32_t i = 0; i < deg; i++) {
//         uint32_t next_node = ptr[i + 1];
//         if (!hash[next_node]) {
//           hop_nodes[hop].emplace_back(next_node);
//           hash[next_node] = ans.size();
//           ans.emplace_back(next_node);
//         }
//       }
//     }
//     hoplevel_num[hop] = hop_nodes[hop].size();

//     // Compute expect cache size.
//     size_t bytes = hoplevel_num[hop] * size_data_per_element_;
//     size_t MB = bytes >> 20;
//     printf("%u-hop size %lu %llu MB\n", hop, hoplevel_num[hop], MB);
//   }

//   uint32_t vec_num = ans.size();
//   size_t bytes = sizeof(vec_num) + sizeof(hnum) + hnum * sizeof(uint32_t) +
//                  vec_num * size_data_per_element_;
//   size_t MB = bytes >> 20;
//   printf("Final cache vector size %lu %llu MB\n", vec_num, MB);

//   /**
//    * Top index format:
//    * <vector_num: uint32_t> <hop_num: uint32_t>
//    * <level_hopcnt[0-hop_num]: uint32_t>
//    * <ngh_vector[0-vector_num]: 332 bytes>
//    */
//   char *top_index_ptr = (char *)malloc(bytes);
//   uint32_t *uint_ptr = (uint32_t *)top_index_ptr;
//   uint_ptr[0] = vec_num;
//   uint_ptr[1] = hnum;
//   for (uint32_t i = 0; i < hnum; i++) {
//     uint_ptr[2 + i] = hoplevel_num[i];
//   }
//   size_t ofs = 2 * sizeof(uint32_t) + hnum * sizeof(uint32_t);

//   for (uint32_t hop = 0; hop < hnum - 1; hop++) {
//     printf("Generate Hop %u\n", hop);
//     for (uint32_t node : hop_nodes[hop]) {
//       uint32_t *ptr = (uint32_t *)(buff + size_data_per_element_ * node);
//       memcpy(top_index_ptr + ofs, ptr, size_data_per_element_);
//       uint32_t *top_ptr = (uint32_t *)(top_index_ptr + ofs);
//       uint32_t deg = top_ptr[0];
//       for (uint32_t i = 0; i < deg; i++) {
//         uint32_t ngh = top_ptr[i + 1];
//         top_ptr[i + 1] = hash[ngh];
//         if (!hash[ngh]) {
//           printf("Error: should be visited.\n");
//           abort();
//         }
//         if (ngh != ans[hash[ngh]]) {
//           printf("Error: should be equal.\n");
//           abort();
//         }
//       }
//       uint64_t *label_ptr = (uint64_t *)(top_index_ptr + ofs + label_ofs);
//       label_ptr[0] = node;
//       ofs += size_data_per_element_;
//     }
//   }
//   printf("Generate Hop 5\n");
//   for (uint32_t node : hop_nodes[hnum - 1]) {
//     uint32_t *ptr = (uint32_t *)(buff + size_data_per_element_ * node);
//     memcpy(top_index_ptr + ofs, ptr, size_data_per_element_);
//     uint32_t *top_ptr = (uint32_t *)(top_index_ptr + ofs);
//     uint32_t deg = top_ptr[0];
//     std::vector<uint32_t> true_ngh;
//     for (uint32_t i = 0; i < deg; i++) {
//       uint32_t ngh = top_ptr[i + 1];
//       if (!hash[ngh]) {
//         continue;
//       } else {
//         true_ngh.emplace_back(hash[ngh]);
//       }
//     }
//     top_ptr[0] = true_ngh.size();
//     for (uint32_t i = 0; i < top_ptr[0]; i++) {
//       top_ptr[i + 1] = true_ngh[i];
//     }
//     uint64_t *label_ptr = (uint64_t *)(top_index_ptr + ofs + label_ofs);
//     label_ptr[0] = node;
//     ofs += size_data_per_element_;
//   }
//   printf("Top index generate over\n");

//   // Write top index to file.
//   std::ofstream output(top_index_path, std::ios::binary);
//   size_t write_size = 1e8;
//   for (size_t i = 0; i < bytes; i += write_size) {
//     if (i + write_size > bytes) {
//       write_size = (long)(bytes - i);
//     }
//     output.write(top_index_ptr + i, write_size);
//     sleep(3);
//     printf("%.2f%%\n", (float)(i + write_size) * 100.0 / bytes);
//   }
//   output.close();
//   printf("Top index write over\n");
// }
