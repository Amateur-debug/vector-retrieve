#pragma once

#include <atomic>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "anns/anns_param.h"
#include "anns/task_manager.h"
#include "index/index_param.h"
#include "vamana/vamana.h"
#include "worklists/balance_queue.h"

/**
 * ScalaGraphIndex: Uniform index storage for ScalaSearch
 * disk file format:
 * each partiton:
 * index file:
 * <partition_id: uint32_t>, <partition_num: uint32_t>,
 * <all_vec_num: uint64_t>, <deg_ngh_size: uint64_t>,
 * <NeighborOffset: uint64_t> * all_vec_num,
 * <DegreeNeighbor: deg_ngh_size>.
 *
 * data file:
 * <partition_id: uint32_t>, <partition_num: uint32_t>,
 * <partition_vec_num: uint32_t>, <all_vec_num: uint64_t>,
 * <vector: dim * type + label> * partition_vec_num.
 */

class ScalaGraphIndex {
  // struct PartitionMap {
  //   uint32_t pid_map, offset;
  // };

  // struct DegreeOffset {
  //   uint32_t deg, offset;
  // };

  // PartitionMap* partition_map;
  // AnnsParameter anns_param;
  IndexParameter &index_param;

  uint64_t *ngh_offset;

  // DegreeOffset* deg_offset;
  char *deg_ngh;

  char *v3_data_ptr;

 public:
  char *vector_ptr;
  char *enter_p_ptr;  // enter point pointer

  uint64_t all_vec_num;   // all vector num
  uint64_t deg_ngh_size;  // bytes of deg_ngh array.
  uint32_t partition_num;
  uint32_t partition_id;
  uint32_t *partition_vec_num;
  uint32_t *cumu_partition_num;
  uint32_t local_part_vec_num;
  uint64_t partition_ofs;
  uint32_t subset_size_milllions;
  uint64_t enter_p;  // enter point

  uint32_t local_vec_num;
  size_t local_ngh_ele_size;
  uint64_t remote_ngh_num;

  int *kmeans_label_pos_map, *kmeans_pos_label_map;

  ScalaGraphIndex(IndexParameter &_index_param) : index_param(_index_param) {
    partition_num = index_param.num_parts;
    partition_id = index_param.machine_id;
    partition_vec_num = (uint32_t *)malloc(sizeof(uint32_t) * partition_num);
    // Inclusive prefix sums write indices [0, partition_num], so allocate
    // partition_num + 1 entries rather than appending a single byte.
    cumu_partition_num =
        (uint32_t *)malloc(sizeof(uint32_t) * (partition_num + 1));
    subset_size_milllions = index_param.subset_size_milllions;
  }

  ~ScalaGraphIndex() {}

  void load_index() {
    char pos_label_path[1024];
    char scala_index_path[partition_num][1024];
    // scalagraph v0
    if (index_param.graph_type == SCALA_V3) {
      printf("ScalaGraph V3 loading index ...\n");

      int *kmeans_part_size;
      int kmeans_vec_size, kmeans_part_num;
      get_pos_label_map(
          index_param.temp_pos2label_map_file.c_str(), kmeans_vec_size,
          kmeans_part_num, kmeans_part_size, kmeans_pos_label_map);

      printf(
          "Loading partition index %d from %s \n", partition_id,
          index_param.final_scala_index_path.c_str());

      if (!std::filesystem::exists(index_param.final_scala_index_path)) {
        std::cout << "Error: file does not exist" << std::endl;
        throw std::runtime_error("file does not exist");
      }
      std::ifstream idx_input(
          index_param.final_scala_index_path.c_str(), std::ios::binary);

      /*
        [ScalaGraph V3 index format]:
        <partition_id: uint32_t>, <partition_num: uint32_t>,
        <part_vec_num: uint32_t>, <local_ngh_ele_size: uint32_t>,
        <enter_point: uint64_t>,
        <remote_ngh_num: uint64_t>
        (header 32 bytes)
        <local_ngh_array: local_ngh_ele_size> * part_vec_num,
        <remote_ngh_array: uint32_t> * remote_ngh_num.
      */
      uint32_t read_pid, read_pnum;
      idx_input.read((char *)&read_pid, sizeof(uint32_t));
      idx_input.read((char *)&read_pnum, sizeof(uint32_t));
      idx_input.read((char *)&local_vec_num, sizeof(uint32_t));
      uint32_t read_local_ngh_ele_size;
      idx_input.read((char *)&read_local_ngh_ele_size, sizeof(uint32_t));
      local_ngh_ele_size = read_local_ngh_ele_size;
      idx_input.read((char *)&enter_p, sizeof(uint64_t));
      idx_input.read((char *)&remote_ngh_num, sizeof(uint64_t));
      printf(
          "partid %u partnum %u local_vec_num %llu "
          "local_ngh_ele_size %llu remote_ngh_num %llu enter_point %llu\n",
          read_pid, read_pnum, local_vec_num, local_ngh_ele_size,
          remote_ngh_num, enter_p);
      assert(read_pid == partition_id && read_pnum == partition_num);

      size_t index_size = local_ngh_ele_size * local_vec_num +
                          remote_ngh_num * sizeof(uint32_t);
      deg_ngh_size = index_size;
      printf("read v3 index size %llu\n", index_size);
      deg_ngh = (char *)malloc(index_size);
      idx_input.read((char *)deg_ngh, index_size);
      idx_input.close();

      /*
        Data format:
        <partition_id: uint32_t>, <partition_num: uint32_t>,
        <partition_vec_num: uint32_t>, <all_vec_num: uint64_t>,
        (header 20 bytes)
        <enterpoint vector: dim * type + label>
        <vector: dim * type + label> * partition_vec_num.
      */
      for (uint32_t p = 0; p < partition_num; p++) {
        printf(
            "Loading partition %u meta data form %s\n", p,
            index_param.final_part_scala_data_path[p].c_str());
        if (!std::filesystem::exists(
                index_param.final_part_scala_data_path[p])) {
          std::cout << "Error: file does not exist" << std::endl;
          throw std::runtime_error("file does not exist");
        }
        std::ifstream data_input(
            index_param.final_part_scala_data_path[p].c_str(),
            std::ios::binary);
        data_input.read((char *)&read_pid, 4);
        data_input.read((char *)&read_pnum, 4);
        data_input.read((char *)&partition_vec_num[p], 4);
        data_input.read((char *)&all_vec_num, 8);
        printf(
            "partid %u partnum %u part_vec_num %llu all_vec_num %llu\n",
            read_pid, read_pnum, partition_vec_num[p], all_vec_num);
        if (p == partition_id) {
          enter_p_ptr = (char *)malloc(index_param.vec_size + sizeof(uint64_t));
          data_input.read(
              (char *)enter_p_ptr, index_param.vec_size + sizeof(uint64_t));

          vector_ptr = (char *)malloc(
              (index_param.vec_size + sizeof(uint64_t)) *
              partition_vec_num[partition_id]);
          // TODO: polish const data.
          data_input.read(
              (char *)vector_ptr, (index_param.vec_size + sizeof(uint64_t)) *
                                      partition_vec_num[partition_id]);
        }
        data_input.close();
      }
      cumu_partition_num[0] = 0;
      for (uint32_t p = 1; p <= partition_num; p++) {
        cumu_partition_num[p] =
            cumu_partition_num[p - 1] + partition_vec_num[p - 1];
      }
      partition_ofs = cumu_partition_num[partition_id];
      printf("partition_ofs: %u\n", partition_ofs);
      local_part_vec_num = partition_vec_num[partition_id];

      printf("ScalaGraph V3 Load Over\n");

    } else {
      printf("Error: graph type not supported\n");
      abort();
    }
  }

  uint32_t get_machine_id(uint32_t cand_id) {
    // TODO: may optimize.
    for (uint32_t p = 1; p <= partition_num; p++) {
      if (cumu_partition_num[p] > cand_id) {
        return p - 1;
      }
    }
  }

  // for scala V2 format, may remove
  void verify_scala() {
    char default_sift100M_kmeans_path[1024];
    // snprintf(
    //     default_sift100M_kmeans_path, sizeof(default_sift100M_kmeans_path),
    //     "/data/share/users/xyzhi/data/bigann/vamana/kmeans/"
    //     "kmeans_index_mem.index");
    snprintf(
        default_sift100M_kmeans_path, sizeof(default_sift100M_kmeans_path),
        "/data/share/users/xyzhi/data/bigann/vamana/balance_kmeans/"
        "kmeans_index_mem.index");

    printf("loading gt index from %s ...\n", default_sift100M_kmeans_path);
    std::ifstream input(default_sift100M_kmeans_path, std::ios::binary);

    if (!input.is_open()) throw std::runtime_error("Cannot open file");

    size_t max_elements_;
    uint32_t enterpoint_node_;
    input.read((char *)&max_elements_, 8);
    input.read((char *)&enterpoint_node_, 4);

    size_t cur_element_count = max_elements_;
    // offsetLevel0_ = 0;

    size_t size_data_per_element_ = 332;
    size_t label_offset_ = 324;
    size_t offsetData_ = 4 + 48 * 4;
    size_t maxM0_ = 48;

    char *data_level0_memory_ =
        (char *)malloc(max_elements_ * size_data_per_element_);
    if (data_level0_memory_ == nullptr)
      throw std::runtime_error(
          "Not enough memory: loadIndex failed to allocate level0");
    input.read(data_level0_memory_, cur_element_count * size_data_per_element_);

    auto verify_data = [&](uint32_t cand_id) {
      char *pdata = scala_graph_get_data(cand_id);
      char *gtdata = (char *)(data_level0_memory_ +
                              cand_id * size_data_per_element_ + offsetData_);
      if (memcmp(pdata, gtdata, 128)) {
        printf("%d data verified failed\n", cand_id);
      }
    };

    printf("Ready to input >> vid\n");
    uint32_t qid = 0;
    while (std::cin >> qid) {
      uint32_t *gt_ptr =
          (uint32_t *)(data_level0_memory_ + qid * size_data_per_element_);
      printf("gt ngh:\n");
      for (uint32_t d = 1; d < gt_ptr[0] + 1; d++) {
        printf("%u ", gt_ptr[d]);
      }
      printf("\n");

      printf("Read from local partition %u\n", partition_id);
      uint64_t ofs = ngh_offset[qid];
      while (true) {
        uint32_t *ptr = (uint32_t *)(deg_ngh + ofs);
        uint32_t end_flag = ptr[0] >> 31;
        uint32_t pid = (ptr[0] >> 16) & 0x00000fff;
        uint32_t deg = ptr[0] & 0x0000ffff;
        if (pid == partition_id) {
          printf("local ngh deg %u:\n", deg);
          for (uint32_t d = 0; d < deg; d++) {
            uint32_t cand_id = scala_graph_get_id(ofs, d);
            printf("%u ", cand_id);
            verify_data(cand_id);
          }
          printf("\n");

          ofs += (deg + 1) * 4LL;
        } else {
          printf("remote p%d has neighbor\n", pid);
          if (deg <= INLINE_DEG_BAR) {
            // case1
            ofs += (deg + 1) * 4LL;
          } else {
            // case2
            ofs += 12;
          }
        }
        if (end_flag) break;
      }
    }
  }

  char *get_partition_ptr() { return vector_ptr; }

  size_t get_partition_size() {
    return (index_param.vec_size + sizeof(uint64_t)) *
           partition_vec_num[partition_id];
  }

  inline uint32_t *scala_graph_get_ngh_ptr(uint64_t offset) {
    if (offset + sizeof(uint32_t) > deg_ngh_size) {
      printf(
          "Error: offset %llu exceed deg_ngh_size: %llu\n", offset,
          deg_ngh_size);
      abort();
    }
    return (uint32_t *)(deg_ngh + offset + sizeof(uint32_t));
  }

  inline uint32_t *scala_graph_v3_get_ngh_ptr(uint64_t offset) {
    if (offset >= deg_ngh_size) {
      printf(
          "Error: offset %llu exceed deg_ngh_size: %llu\n", offset,
          deg_ngh_size);
      abort();
    }
    return (uint32_t *)(deg_ngh + offset);
  }

  uint32_t scala_graph_get_id(uint64_t offset, uint32_t d) {
    uint32_t *remote_ptr = (uint32_t *)(deg_ngh + offset);
    return remote_ptr[d + 1];
  }

  uint32_t scala_graph_get_internal_id(uint32_t machine_id, uint32_t vid) {
    return vid - cumu_partition_num[machine_id];
  }

  template <typename dist_t>
  uint32_t scala_graph_get_task(
      uint32_t current_node_id, uint32_t query_id, TaskManager<dist_t> &tman) {
    uint64_t ofs = ngh_offset[current_node_id];
    _mm_prefetch(deg_ngh + ofs, _MM_HINT_T0);
    uint32_t post_cnt = 0;
    dist_t lb = tman.global_query[query_id]->lowerBound;
    uint8_t ef_fill =
        (tman.global_query[query_id]->top_candidates.size() <
         tman.global_query[query_id]->ef);

    while (true) {
      uint32_t *ptr = (uint32_t *)(deg_ngh + ofs);
      uint32_t end_flag = *(ptr) >> 31;
      uint32_t pid = (*(ptr) >> 16) & 0x00000fff;
      uint32_t deg = *(ptr) & 0x0000ffff;

      if (pid == partition_id) {
        // printf("local has %u neighbors, offset %llu\n", deg, ofs);
        _mm_prefetch(deg_ngh + ofs + (deg + 1) * 4LL, _MM_HINT_T0);
        tman.next_task_queue.emplace_back(partition_id, query_id, deg, ofs);
        ofs += (deg + 1) * 4LL;
        post_cnt++;
      } else {
        // printf("remote p%d has neighbor\n", pid);
        if (deg <= INLINE_DEG_BAR) {
          // case1: micro degree.
          _mm_prefetch(deg_ngh + ofs + (deg + 1) * 4LL, _MM_HINT_T0);
          for (uint32_t d = 0; d < deg; d++) {
            uint32_t vid = *(ptr + d + 1);
            uint32_t internal_id = scala_graph_get_internal_id(pid, vid);
            // BufferCache *bc =
            //     new BufferCache(pid, vid, internal_id, -1, nullptr);
            tman.global_query[query_id]->async_read_queue.emplace_back(
                pid, vid, internal_id, -1, nullptr);
          }
          ofs += (deg + 1) * 4LL;
          // No need to post_cnt++ for case1.
        } else {
          // case2: large degree.
          _mm_prefetch(deg_ngh + ofs + 12, _MM_HINT_T0);
          uint64_t *ofs_ptr = (uint64_t *)(deg_ngh + ofs + 4);
          tman.task_queue[pid].emplace_back(
              partition_id, query_id, deg, ofs_ptr[0], lb, ef_fill);
          ofs += 12;
          post_cnt++;
        }
      }

      if (end_flag) {
        break;
      }
    }
    return post_cnt;
  }

  /**
   * Identify local core tasks, remote core tasks, remote non-core tasks.
   */
  template <typename dist_t>
  uint32_t scala_graph_v3_get_subquery_task_lb(
      uint32_t current_node_id, uint32_t query_id, TaskManager<dist_t> &tman,
      BalanceQueue<ComputeTask<dist_t>> &ib) {
    if (current_node_id < partition_ofs ||
        current_node_id >= partition_ofs + local_part_vec_num) {
      printf(
          "Error: current_node_id %u less or exceed partition_ofs %u\n",
          current_node_id, partition_ofs);
      abort();
    }
    size_t ofs = local_ngh_ele_size * (current_node_id - partition_ofs);
    _mm_prefetch(deg_ngh + ofs, _MM_HINT_T0);
    uint32_t post_cnt = 0;
    dist_t lb = tman.global_query[query_id]->lowerBound;
    uint8_t ef_fill =
        (tman.global_query[query_id]->top_candidates.size() <
         tman.global_query[query_id]->ef);

    while (true) {
      uint8_t *ptr = (uint8_t *)(deg_ngh + ofs);
      uint8_t end_flag = ptr[1] & 0x80;
      uint8_t local_flag = ptr[1] & 0x40;
      uint32_t pid = ptr[1] & 0x3f;
      uint32_t deg = ptr[0];

      ofs += 2;
      if (local_flag) {
#ifdef DEBUG
        // printf("put %u neighbors to local task bag\n", deg);
#endif
        // after previous stage, will put into local task queue.
        ib.emplace_back(partition_id, query_id, deg, ofs);
        post_cnt++;
        // the local ngh must be end.
        break;
      } else {
        if (deg <= INLINE_DEG_BAR) {
          // micro task to other machine
          _mm_prefetch(deg_ngh + ofs + deg * 4, _MM_HINT_T0);
          uint32_t *ngh_ptr = (uint32_t *)(deg_ngh + ofs);
          if (tman.global_query[query_id]->is_core_machine[pid]) {
            // core ngh send to core machine
            for (uint32_t d = 0; d < deg; d++) {
              tman.global_query[query_id]
                  ->sync_msg.core_micro_ngh[pid]
                  .emplace_back(ngh_ptr[d]);
            }
          } else {  // async read to non-core machine
            for (uint32_t d = 0; d < deg; d++) {
              uint32_t internal_id =
                  scala_graph_get_internal_id(pid, ngh_ptr[d]);
              // TODO: change to bufcache_ptr
              // BufferCache *bc =
              //     new BufferCache(pid, vid, internal_id, -1, nullptr);
              tman.global_query[query_id]->async_read_queue.push_back(
                  BufferCache(pid, ngh_ptr[d], internal_id, -1, nullptr));
              // Need to post_cnt++ for micro async read.
              post_cnt++;
            }
          }
          ofs += deg * 4;
        } else {
          // high degree task to other machine
          // TODO: here prefetch may not need.
          _mm_prefetch(deg_ngh + ofs + 8, _MM_HINT_T0);
          uint64_t *ofs_ptr = (uint64_t *)(deg_ngh + ofs);
          if (tman.global_query[query_id]->is_core_machine[pid]) {
            // no need for post_cnt.
            tman.global_query[query_id]
                ->sync_msg.core_large_ngh[pid]
                .emplace_back(ofs_ptr[0], deg);
          } else {
            tman.task_queue[pid].emplace_back(
                partition_id, query_id, deg, ofs_ptr[0], lb, ef_fill);
            post_cnt++;
          }
          ofs += 8;
        }
      }
      if (end_flag) {
        break;
      }
    }
    // post_cnt = number of async read vectors + number of remote tasks.
    return post_cnt;
  }

  /**
   * Identify local core tasks, remote core tasks, remote non-core tasks.
   * Scala V2 format
   */
  template <typename dist_t>
  void scala_graph_get_nodetask_msg(
      TaskManager<dist_t> &tman, NodeTask &task, NodeResult<dist_t> &res_msg) {
#ifdef DEBUG
    std::cout << "get node task msg for v " << task.node_id << "\n";
#endif
    uint64_t ofs = ngh_offset[task.node_id];
    _mm_prefetch(deg_ngh + ofs, _MM_HINT_T0);
    res_msg.post_cnt = 0;

    while (true) {
      uint32_t *ptr = (uint32_t *)(deg_ngh + ofs);
      uint32_t end_flag = *(ptr) >> 31;
      uint32_t pid = (*(ptr) >> 16) & 0x00000fff;
      uint32_t deg = *(ptr) & 0x0000ffff;
#ifdef DEBUG
      // std::cout << "q" << res_msg.qid << " ngh partition: " << pid << "\n";
#endif
      if (pid == partition_id) {
        // std::cout << "local partition\n";
        _mm_prefetch(deg_ngh + ofs + (deg + 1) * 4LL, _MM_HINT_T0);
        // after previous stage, will put into local task queue.
        // tman.global_query[query_id]->local_task.emplace_back(
        //     partition_id, query_id, deg, ofs);
        // ib.emplace_back(partition_id, task.query_id, deg, ofs);

        // set local compute offset, will proc later.
        task.deg = deg;
        task.offset = ofs;
        // post_cnt++;
        ofs += (deg + 1) * 4LL;
      } else {
        if (deg <= INLINE_DEG_BAR) {
          // micro task to other machine
          _mm_prefetch(deg_ngh + ofs + (deg + 1) * 4LL, _MM_HINT_T0);

          if (task.is_core_machine[pid]) {
            // std::cout << "low deg core machine\n";
            // core ngh send to core machine
            tman.noncore_sync_msg[pid].emplace_back(task.qid);
            NonCoreSyncMsg<dist_t> &msg = tman.noncore_sync_msg[pid].back();
            for (uint32_t d = 0; d < deg; d++) {
              uint32_t vid = *(ptr + d + 1);
              msg.core_micro_ngh.emplace_back(vid);
              // tman.global_query[task.query_id]
              //     ->sync_msg.core_micro_ngh[pid]
              //     .emplace_back(vid);
            }
          } else {  // send task to non-core machine
            // std::cout << "low deg non-core machine\n";
            uint64_t tmp_ofs_for_micro_combine = 0;
            // uint32_t *tmp_ptr = (uint32_t *)(&tmp_ofs_for_micro_combine);
            for (uint32_t d = 0; d < deg; d++) {
              uint32_t vid = *(ptr + d + 1);
              // tmp_ptr[d] = vid;
              tmp_ofs_for_micro_combine |= ((uint64_t)(vid) << (d * 32LL));
              // uint32_t internal_id = scala_graph_get_internal_id(pid, vid);

              // TODO: change to bufcache_ptr
              // tman.global_query[task.query_id]->async_read_queue.push_back(
              //     BufferCache(pid, vid, internal_id, -1, nullptr));
              // Need to post_cnt++ for micro async read.
              // post_cnt++;
              // tman.post_async++;
            }
            // NOTE: send back to origin machine (task.machine_id, not partition
            // id)
            tman.task_queue[pid].emplace_back(
                task.machine_id, task.qid, deg, tmp_ofs_for_micro_combine, 0,
                0);
            res_msg.post_cnt++;
          }
          ofs += (deg + 1) * 4LL;
        } else {
          // high degree task to other machine
          _mm_prefetch(deg_ngh + ofs + 12, _MM_HINT_T0);
          uint64_t *ofs_ptr = (uint64_t *)(deg_ngh + ofs + 4);
          if (task.is_core_machine[pid]) {
            // std::cout << "high deg core machine\n";
            // no need for post_cnt.
            // tman.global_query[task.qid]
            //     ->sync_msg.core_large_ngh[pid]
            //     .emplace_back(ofs_ptr[0], deg);
            tman.noncore_sync_msg[pid].emplace_back(task.qid);
            NonCoreSyncMsg<dist_t> &msg = tman.noncore_sync_msg[pid].back();
            msg.core_large_ngh.emplace_back(ofs_ptr[0], deg);
          } else {
            // std::cout << "high deg non-core machine\n";
            // NOTE: send back to origin machine (task.machine_id, not partition
            // id)
            tman.task_queue[pid].emplace_back(
                task.machine_id, task.qid, deg, ofs_ptr[0], 0, 0);
            res_msg.post_cnt++;
            // tman.post_task++;
          }
          ofs += 12;
        }
      }
      if (end_flag) {
        break;
      }
    }
    // post_cnt = number of remote tasks.
    return;
  }

  /**
   * Identify local core tasks, remote core tasks, remote non-core tasks.
   */
  template <typename dist_t>
  void scala_graph_v3_get_nodetask_msg(
      TaskManager<dist_t> &tman, NodeTask &task, NodeResult<dist_t> &res_msg) {
#ifdef DEBUG
    // std::cout << "get node task msg for v " << task.node_id << "\n";
#endif
    size_t ofs = local_ngh_ele_size * (task.node_id - partition_ofs);
    _mm_prefetch(deg_ngh + ofs, _MM_HINT_T0);
    res_msg.post_cnt = 0;

    while (true) {
      uint8_t *ptr = (uint8_t *)(deg_ngh + ofs);
      uint8_t end_flag = ptr[1] & 0x80;
      uint8_t local_flag = ptr[1] & 0x40;
      uint32_t pid = ptr[1] & 0x3f;
      uint32_t deg = ptr[0];
      ofs += 2;
#ifdef DEBUG
      // std::cout << "q" << res_msg.qid << " ngh partition: " << pid << "\n";
#endif
      if (local_flag) {
        // std::cout << "local partition\n";
        _mm_prefetch(deg_ngh + ofs + deg * 4, _MM_HINT_T0);
        // after previous stage, will put into local task queue.
        // tman.global_query[query_id]->local_task.emplace_back(
        //     partition_id, query_id, deg, ofs);
        // ib.emplace_back(partition_id, task.query_id, deg, ofs);

        // set local compute offset, will proc later.
        task.deg = deg;
        task.offset = ofs;
        // post_cnt++;
        // ofs += deg * 4;
        break;
      } else {
        if (deg <= INLINE_DEG_BAR) {
          // micro task to other machine
          _mm_prefetch(deg_ngh + ofs + deg * 4, _MM_HINT_T0);
          uint32_t *ngh_ptr = (uint32_t *)(deg_ngh + ofs);
          if (task.is_core_machine[pid]) {
            // std::cout << "low deg core machine\n";
            // core ngh send to core machine
            tman.noncore_sync_msg[pid].emplace_back(task.qid);
            NonCoreSyncMsg<dist_t> &msg = tman.noncore_sync_msg[pid].back();
            for (uint32_t d = 0; d < deg; d++) {
              msg.core_micro_ngh.emplace_back(ngh_ptr[d]);
              // tman.global_query[task.query_id]
              //     ->sync_msg.core_micro_ngh[pid]
              //     .emplace_back(vid);
            }
          } else {  // send task to non-core machine
            // std::cout << "low deg non-core machine\n";
            uint64_t tmp_ofs_for_micro_combine = 0;
            // uint32_t *tmp_ptr = (uint32_t *)(&tmp_ofs_for_micro_combine);
            for (uint32_t d = 0; d < deg; d++) {
              // tmp_ptr[d] = vid;
              tmp_ofs_for_micro_combine |=
                  ((uint64_t)(ngh_ptr[d]) << (d * 32LL));
              // uint32_t internal_id = scala_graph_get_internal_id(pid, vid);

              // TODO: change to bufcache_ptr
              // tman.global_query[task.query_id]->async_read_queue.push_back(
              //     BufferCache(pid, vid, internal_id, -1, nullptr));
              // Need to post_cnt++ for micro async read.
              // post_cnt++;
              // tman.post_async++;
            }
            // NOTE: send back to origin machine (task.machine_id, not partition
            // id)
            tman.task_queue[pid].emplace_back(
                task.machine_id, task.qid, deg, tmp_ofs_for_micro_combine, 0,
                0, 0, 1); // core count +0  nocore count +1
            res_msg.post_cnt++;
          }
          ofs += deg * 4;
        } else {
          // high degree task to other machine
          _mm_prefetch(deg_ngh + ofs + 8, _MM_HINT_T0);
          uint64_t *ofs_ptr = (uint64_t *)(deg_ngh + ofs);
          if (task.is_core_machine[pid]) {
            // std::cout << "high deg core machine\n";
            // no need for post_cnt.
            // tman.global_query[task.qid]
            //     ->sync_msg.core_large_ngh[pid]
            //     .emplace_back(ofs_ptr[0], deg);
            tman.noncore_sync_msg[pid].emplace_back(task.qid);
            NonCoreSyncMsg<dist_t> &msg = tman.noncore_sync_msg[pid].back();
            // printf(
            //     "]]] high deg core machine q%u >> %llu %u\n", task.qid,
            //     ofs_ptr[0], deg);
            msg.core_large_ngh.emplace_back(ofs_ptr[0], deg);
          } else {
            // std::cout << "high deg non-core machine\n";
            // NOTE: send back to origin machine (task.machine_id, not partition
            // id)
            tman.task_queue[pid].emplace_back(
                task.machine_id, task.qid, deg, ofs_ptr[0], 0, 0, 0, 1); 
            // core count +0  nocore count +1
            res_msg.post_cnt++;
            // tman.post_task++;
          }
          ofs += 8;
        }
      }
      if (end_flag) {
        break;
      }
    }
    // post_cnt = number of remote tasks.
    return;
  }

  inline uint32_t scala_graph_get_partition_ofs() { return partition_ofs; }

  char *scala_graph_get_data(uint64_t candidate_id) {
    if (candidate_id < partition_ofs ||
        partition_ofs + local_part_vec_num <= candidate_id) {
      printf(
          "IndexError: candidate_id out of range. vid:%u ofs:%u\n",
          candidate_id, partition_ofs);
      abort();
    }
    uint64_t data_ofs = candidate_id - partition_ofs;
    return (char *)(vector_ptr +
                    (index_param.vec_size + sizeof(uint64_t)) * data_ofs);
  }

  /**
   * Transfer kmeans vamana index to scalagraph format.
   */
  void transfer_kmeans_vamana(
      std::string vamana_path = "", std::string scg_path = "") {
    // std::string label_pos_path =
    //     "/data/share/users/xyzhi/data/bigann/vamana/kmeans/"
    //     "100M_4_labelpos_map.bin";
    // for new balanced kmeans partition
    std::string label_pos_path =
        "/data/share/users/xyzhi/data/bigann/vamana/balance_kmeans/"
        "kmeans_label_pos_map.bin";

    int *kmeans_part_size;
    int kmeans_vec_size, kmeans_part_num;
    get_label_pos_map(
        label_pos_path.c_str(), kmeans_vec_size, kmeans_part_num,
        kmeans_part_size, kmeans_label_pos_map);

    auto judge_partition = [&](uint64_t vid) {
      size_t cumunum = 0;
      for (int p = 0; p < kmeans_part_num; p++) {
        cumunum += kmeans_part_size[p];
        if (cumunum > vid) return p;
      }
      return -1;
    };

    char default_sift100M_kmeans_path[1024];
    // snprintf(
    //     default_sift100M_kmeans_path, sizeof(default_sift100M_kmeans_path),
    //     "/data/share/users/xyzhi/data/bigann/vamana/kmeans/"
    //     "kmeans_index_mem.index");
    // balance kmeans
    snprintf(
        default_sift100M_kmeans_path, sizeof(default_sift100M_kmeans_path),
        "/data/share/users/xyzhi/data/bigann/vamana/balance_kmeans/"
        "kmeans_index_mem.index");

    printf("loading index from %s ...\n", default_sift100M_kmeans_path);
    std::ifstream input(default_sift100M_kmeans_path, std::ios::binary);

    if (!input.is_open()) throw std::runtime_error("Cannot open file");

    size_t max_elements_;
    uint64_t enterpoint_node_;
    input.read((char *)&max_elements_, 8);
    input.read((char *)&enterpoint_node_, 4);

    size_t cur_element_count = max_elements_;
    // offsetLevel0_ = 0;

    size_t size_data_per_element_ = 332;
    size_t label_offset_ = 324;
    size_t maxM0_ = 48;
    size_t offsetData_ = 4 + maxM0_ * 4;

    char *data_level0_memory_ =
        (char *)malloc(max_elements_ * size_data_per_element_);
    if (data_level0_memory_ == nullptr)
      throw std::runtime_error(
          "Not enough memory: loadIndex failed to allocate level0");
    input.read(data_level0_memory_, cur_element_count * size_data_per_element_);

    printf(
        "Kmeans index file read over, enter node %llu, part num %d\n",
        enterpoint_node_, kmeans_part_num);

    // Each index partition path.
    char scala_index_path[kmeans_part_num][1024];
    uint32_t subset_size_milllions = 100;
    for (int i = 0; i < kmeans_part_num; i++) {
      snprintf(
          scala_index_path[i], sizeof(scala_index_path[i]),
          "/data/share/users/xyzhi/data/bigann/scalagraph/balance_v1/"
          "%dM_index_%d-%d.bin",
          subset_size_milllions, i, kmeans_part_num);
    }
    char scala_data_path[kmeans_part_num][1024];
    for (int i = 0; i < kmeans_part_num; i++) {
      snprintf(
          scala_data_path[i], sizeof(scala_data_path[i]),
          "/data/share/users/xyzhi/data/bigann/scalagraph/balance_v1/"
          "%dM_data_%d-%d.bin",
          subset_size_milllions, i, kmeans_part_num);
    }

    printf("Start compute scala partition size ...\n");
    // Step1: Compute each partition size.
    uint64_t index_meta_len = 32LL;
    uint64_t data_meta_len = 20LL;
    uint64_t scala_index_size[kmeans_part_num],
        scala_data_size[kmeans_part_num];
    for (int p = 0; p < kmeans_part_num; p++) {
      scala_index_size[p] =
          (uint64_t)(index_meta_len + cur_element_count * 8LL);
      scala_data_size[p] =
          (uint64_t)(data_meta_len + (index_param.vec_size + sizeof(uint64_t)) *
                                         (kmeans_part_size[p] + 1LL));
    }
    for (uint64_t vid = 0; vid < cur_element_count; vid++) {
      int vid_pid = judge_partition(vid);
      int *ptr = (int *)(data_level0_memory_ + vid * size_data_per_element_);
      int cnt[kmeans_part_num];  // ngh distribute count.
      for (int i = 0; i < kmeans_part_num; i++) cnt[i] = 0;
      for (int i = 1; i < ptr[0] + 1; i++) {
        cnt[judge_partition(ptr[i])]++;
      }
      for (int p = 0; p < kmeans_part_num; p++) {
        if (cnt[p] > 0) {
          for (int ngh_p = 0; ngh_p < kmeans_part_num; ngh_p++) {
            if (ngh_p == p) {  // local ngh
              scala_index_size[ngh_p] +=
                  (uint64_t)((cnt[p] + 1LL) * 4LL);              // deg + nghs
            } else {                                             // remote ngh
              scala_index_size[ngh_p] += (uint64_t)(4LL + 8LL);  // deg + ptr
            }
          }
        }
      }
    }

    for (int p = 0; p < kmeans_part_num; p++) {
      printf(
          "scala_index_size: %llu, scala_data_size: %llu\n",
          scala_index_size[p], scala_data_size[p]);
    }
    printf("Start write index and data... \n");
    // scala_index_size: 8359571036, scala_data_size: 5208751332
    // scala_index_size: 5567326640, scala_data_size: 2900225452
    // scala_index_size: 6153537612, scala_data_size: 3239405780
    // scala_index_size: 4952724792, scala_data_size: 2251617516

    /*
      <partition_id: uint32_t>, <partition_num: uint32_t>,
      <all_vec_num: uint64_t>, <deg_ngh_size: uint64_t>,
      <enter_point: uint64_t>
      (header 32 bytes)
      <NeighborOffset: uint64_t> * all_vec_num,
      <DegreeNeighbor: uint32_t> * deg_ngh_num.
    */
    printf("Build index ...\n");
    char *out_index[kmeans_part_num];
    uint64_t index_ofs[kmeans_part_num];
    for (uint32_t i = 0; i < kmeans_part_num; i++) index_ofs[i] = 0;
    for (uint32_t i = 0; i < kmeans_part_num; i++) {
      out_index[i] = (char *)malloc(scala_index_size[i]);
      memcpy(out_index[i], &i, 4);
      memcpy(out_index[i] + 4, &kmeans_part_num, 4);
      memcpy(out_index[i] + 8, &cur_element_count, 8);
      memcpy(out_index[i] + 24, &enterpoint_node_, 8);
    }

    uint64_t ofs_len = cur_element_count * 8L;

    for (uint64_t vid = 0; vid < cur_element_count; vid++) {
      for (uint32_t p = 0; p < kmeans_part_num; p++) {
        memcpy(out_index[p] + index_meta_len + vid * 8LL, &index_ofs[p], 8);
      }
      int vid_pid = judge_partition(vid);
      uint32_t *ptr =
          (uint32_t *)(data_level0_memory_ + vid * size_data_per_element_);
      std::vector<uint32_t> vv[kmeans_part_num];
      for (int i = 1; i < ptr[0] + 1; i++) {
        int part = judge_partition(ptr[i]);
        vv[part].emplace_back(ptr[i]);
      }
      std::vector<uint32_t> part_cnt;
      for (int p = 0; p < kmeans_part_num; p++) {
        if (vv[p].size() > 0) {
          part_cnt.push_back(p);
        }
      }

      for (uint32_t j = 0; j < part_cnt.size(); j++) {
        uint32_t end_flag = 0;
        if (j == part_cnt.size() - 1) {
          end_flag = 1 << 31;
        }

        uint32_t p = part_cnt[j];

        uint32_t msg = end_flag | (p << 16) | vv[p].size();
        if (vid == 0) {
          printf(
              "write vid0 part %u deg %u end %llu\n", p, vv[p].size(),
              end_flag);
          printf(
              "expect read vid0 part %u deg %u end %d\n",
              (msg >> 16) & 0x00000fff, msg & 0x0000ffff, msg >> 31);
        }
        for (int remote_p = 0; remote_p < kmeans_part_num; remote_p++) {
          if (remote_p != p) {  // remote ngh
            if (vid == 0) {
              printf(
                  "write vid0 as remote part %u deg %u end %llu ofs %llu\n",
                  remote_p, vv[p].size(), end_flag, index_ofs[remote_p]);
              printf(
                  "expect read vid0 part %u deg %u end %d\n",
                  (msg >> 16) & 0x00000fff, msg & 0x0000ffff, msg >> 31);
            }
            memcpy(
                out_index[remote_p] + index_meta_len + ofs_len +
                    index_ofs[remote_p],
                &msg, sizeof(msg));
            index_ofs[remote_p] += sizeof(msg);
            memcpy(
                out_index[remote_p] + index_meta_len + ofs_len +
                    index_ofs[remote_p],
                &index_ofs[p], sizeof(index_ofs[p]));
            index_ofs[remote_p] += sizeof(index_ofs[p]);
            if (vid == 0) {
              printf(
                  "remote_p%d index ofs : %u\n", remote_p, index_ofs[remote_p]);
            }
          }
        }
        // local ngh
        memcpy(
            out_index[p] + index_meta_len + ofs_len + index_ofs[p], &msg,
            sizeof(msg));
        index_ofs[p] += sizeof(msg);
        for (auto &v : vv[p]) {
          memcpy(
              out_index[p] + index_meta_len + ofs_len + index_ofs[p], &v,
              sizeof(v));
          index_ofs[p] += sizeof(v);
        }
        if (vid == 0) {
          printf("p%d index ofs : %u\n", p, index_ofs[p]);
        }
      }
    }

    for (uint32_t i = 0; i < kmeans_part_num; i++) {
      memcpy(out_index[i] + 16, &index_ofs[i], 8);
      printf(
          "part %d index size %llu expect %llu\n", i,
          index_meta_len + ofs_len + index_ofs[i], scala_index_size[i]);
    }

    for (uint32_t p = 0; p < kmeans_part_num; p++) {
      printf("Write part %d index size %llu...\n", p, scala_index_size[p]);
      uint64_t write_size = 1LL << 28;

      std::ofstream output(scala_index_path[p], std::ios::binary);
      for (uint64_t i = 0; i < scala_index_size[p]; i += write_size) {
        if (i + write_size > scala_index_size[p]) {
          write_size = (uint64_t)(scala_index_size[p] - i);
        }
        output.write((char *)(out_index[p] + i), write_size);
        sleep(4);
        printf(
            "%.2f%%\n",
            (float)(i + write_size) * 100.0 / (scala_index_size[p]));
      }

      printf(
          "partition index file %d write over, expect file size %llu\n", p,
          scala_index_size[p]);
      output.close();
    }

    // Step2: Write to each partition data.
    /*
      <partition_id: uint32_t>, <partition_num: uint32_t>,
      <partition_vec_num: uint32_t>, <all_vec_num: uint64_t>,
      (header 20 bytes)
      <enterpoint vector: dim * type + label>
      <vector: dim * type + label> * partition_vec_num.
    */
    printf("Build data ...\n");
    char *out_data[kmeans_part_num];
    uint64_t data_ofs[kmeans_part_num];
    for (uint32_t i = 0; i < kmeans_part_num; i++) data_ofs[i] = data_meta_len;
    for (uint32_t i = 0; i < kmeans_part_num; i++) {
      out_data[i] = (char *)malloc(scala_data_size[i]);
      memcpy(out_data[i], &i, 4);
      memcpy(out_data[i] + 4, &kmeans_part_num, 4);
      memcpy(out_data[i] + 8, &kmeans_part_size[i], 4);
      memcpy(out_data[i] + 12, &cur_element_count, 8);
    }
    // copy enterpoint
    char *enter_ptr =
        (char *)(data_level0_memory_ +
                 enterpoint_node_ * size_data_per_element_ + offsetData_);
    for (uint32_t p = 0; p < kmeans_part_num; p++) {
      memcpy(
          out_data[p] + data_ofs[p], enter_ptr,
          index_param.vec_size + sizeof(uint64_t));
      data_ofs[p] += index_param.vec_size + sizeof(uint64_t);
    }

    for (uint64_t vid = 0; vid < cur_element_count; vid++) {
      int pid = judge_partition(vid);
      char *ptr = (char *)(data_level0_memory_ + vid * size_data_per_element_ +
                           offsetData_);
      memcpy(
          out_data[pid] + data_ofs[pid], ptr,
          index_param.vec_size + sizeof(uint64_t));
      data_ofs[pid] += index_param.vec_size + sizeof(uint64_t);
    }

    for (uint32_t p = 0; p < kmeans_part_num; p++) {
      assert(data_ofs[p] == scala_data_size[p]);
      printf("Write part %d size %llu ...\n", p, scala_data_size[p]);
      uint64_t write_size = 1LL << 28;

      std::ofstream output(scala_data_path[p], std::ios::binary);
      for (uint64_t i = 0; i < scala_data_size[p]; i += write_size) {
        if (i + write_size > scala_data_size[p]) {
          write_size = (uint64_t)(scala_data_size[p] - i);
        }
        output.write((char *)(out_data[p] + i), write_size);
        sleep(4);
        printf(
            "%.2f%%\n", (float)(i + write_size) * 100.0 / (scala_data_size[p]));
      }

      printf(
          "partition data file %d write over, expect file size %llu\n", p,
          scala_data_size[p]);
      output.close();
    }
  }

  /**
   * Transfer b1 kmeans index to scalagraph v2 format.
   * v2 format involves some small degs to inline storage.
   */
  void transfer_b1_to_scala_v2(int part_num, int million_num) {
    // for new balanced kmeans partition
    // std::string label_pos_path =
    //     "/data/share/users/xyzhi/data/bigann/vamana/balance_kmeans/"
    //     "kmeans_label_pos_map.bin";
    char label_pos_path[1024];
    snprintf(
        label_pos_path, sizeof(label_pos_path),
        "/data/share/users/xyzhi/data/bigann/vamana/balance_kmeans_%d_part/"
        "100M_%d_labelpos_map.bin",
        part_num, part_num);

    int *kmeans_part_size;
    int kmeans_vec_size, kmeans_part_num;
    get_label_pos_map(
        label_pos_path, kmeans_vec_size, kmeans_part_num, kmeans_part_size,
        kmeans_label_pos_map);

    auto judge_partition = [&](uint64_t vid) {
      size_t cumunum = 0;
      for (int p = 0; p < kmeans_part_num; p++) {
        cumunum += kmeans_part_size[p];
        if (cumunum > vid) return p;
      }
      return -1;
    };

    char kmeans_b1_path[1024];
    snprintf(
        kmeans_b1_path, sizeof(kmeans_b1_path),
        "/data/share/users/xyzhi/data/bigann/vamana/balance_kmeans_%d_part/"
        "kmeans_b1_mem.index",
        part_num);

    printf("loading kmeans b1graph index from %s ...\n", kmeans_b1_path);
    std::ifstream input(kmeans_b1_path, std::ios::binary);

    if (!input.is_open()) throw std::runtime_error("Cannot open file");

    size_t max_elements_;
    uint64_t enterpoint_node_;
    input.read((char *)&max_elements_, 8);
    input.read((char *)&enterpoint_node_, 4);

    size_t cur_element_count = max_elements_;
    // offsetLevel0_ = 0;

    size_t size_data_per_element_ = 332;
    size_t label_offset_ = 324;
    size_t maxM0_ = 48;
    size_t offsetData_ = 4 + maxM0_ * 4;

    char *data_level0_memory_ =
        (char *)malloc(max_elements_ * size_data_per_element_);
    if (data_level0_memory_ == nullptr)
      throw std::runtime_error(
          "Not enough memory: loadIndex failed to allocate level0");
    input.read(data_level0_memory_, cur_element_count * size_data_per_element_);

    printf(
        "Kmeans index file read over, enter node %llu, part num %d\n",
        enterpoint_node_, kmeans_part_num);

    // Each index partition path.
    char scala_index_path[kmeans_part_num][1024];
    uint32_t subset_size_milllions = 100;
    for (int i = 0; i < kmeans_part_num; i++) {
      snprintf(
          scala_index_path[i], sizeof(scala_index_path[i]),
          // bar=4
          // "/data/share/users/xyzhi/data/bigann/scalagraph/balance_v2/"
          // bar=2
          "/data/share/users/xyzhi/data/bigann/scalagraph/balance_v3/"
          "%dM_index_%d-%d.bin",
          subset_size_milllions, i, kmeans_part_num);
    }
    char scala_data_path[kmeans_part_num][1024];
    for (int i = 0; i < kmeans_part_num; i++) {
      snprintf(
          scala_data_path[i], sizeof(scala_data_path[i]),
          "/data/share/users/xyzhi/data/bigann/scalagraph/balance_v1/"
          "%dM_data_%d-%d.bin",
          subset_size_milllions, i, kmeans_part_num);
    }

    printf("Start compute scala partition size ...\n");
    // Step1: Compute each partition size.
    uint64_t index_meta_len = 32LL;
    uint64_t data_meta_len = 20LL;
    uint64_t scala_index_size[kmeans_part_num],
        scala_data_size[kmeans_part_num];
    for (int p = 0; p < kmeans_part_num; p++) {
      scala_index_size[p] =
          (uint64_t)(index_meta_len + cur_element_count * 8LL);
      scala_data_size[p] =
          (uint64_t)(data_meta_len + (index_param.vec_size + sizeof(uint64_t)) *
                                         (kmeans_part_size[p] + 1LL));
    }
    for (uint64_t vid = 0; vid < cur_element_count; vid++) {
      int vid_pid = judge_partition(vid);
      int *ptr = (int *)(data_level0_memory_ + vid * size_data_per_element_);
      int cnt[kmeans_part_num];  // ngh distribute count.
      for (int i = 0; i < kmeans_part_num; i++) cnt[i] = 0;
      for (int i = 1; i < ptr[0] + 1; i++) {
        cnt[judge_partition(ptr[i])]++;
      }
      for (int p = 0; p < kmeans_part_num; p++) {
        if (cnt[p] > 0) {
          for (int ngh_p = 0; ngh_p < kmeans_part_num; ngh_p++) {
            if (ngh_p == p) {  // local ngh
              scala_index_size[ngh_p] +=
                  (uint64_t)((cnt[p] + 1LL) * 4LL);  // deg + nghs
            } else {                                 // remote ngh
              // case1: <= INLINE_DEG_BAR
              if (cnt[p] <= INLINE_DEG_BAR) {
                scala_index_size[ngh_p] +=
                    (uint64_t)((cnt[p] + 1LL) * 4LL);  // deg + nghs
              } else {
                // case2: > INLINE_DEG_BAR
                scala_index_size[ngh_p] += (uint64_t)(4LL + 8LL);  // deg + ptr
              }
            }
          }
        }
      }
    }

    for (int p = 0; p < kmeans_part_num; p++) {
      printf(
          "scala_index_size: %llu, scala_data_size: %llu\n",
          scala_index_size[p], scala_data_size[p]);
    }
    printf("Start write index and data... \n");
    // scala_index_size: 8359571036, scala_data_size: 5208751332
    // scala_index_size: 5567326640, scala_data_size: 2900225452
    // scala_index_size: 6153537612, scala_data_size: 3239405780
    // scala_index_size: 4952724792, scala_data_size: 2251617516

    /*
      <partition_id: uint32_t>, <partition_num: uint32_t>,
      <all_vec_num: uint64_t>, <deg_ngh_size: uint64_t>,
      <enter_point: uint64_t>
      (header 32 bytes)
      <NeighborOffset: uint64_t> * all_vec_num,
      <DegreeNeighbor: uint32_t> * deg_ngh_num.
    */
    printf("Build index ...\n");

    uint32_t prange = 8;
    for (uint32_t LP = 0, RP = prange; RP <= kmeans_part_num;
         LP += prange, RP += prange) {
      char *out_index[kmeans_part_num];
      uint64_t index_ofs[kmeans_part_num];
      for (uint32_t i = 0; i < kmeans_part_num; i++) index_ofs[i] = 0;
      for (uint32_t i = 0; i < kmeans_part_num; i++) {
        if (LP <= i && i < RP) {
          out_index[i] = (char *)malloc(scala_index_size[i]);
          memcpy(out_index[i], &i, 4);
          memcpy(out_index[i] + 4, &kmeans_part_num, 4);
          memcpy(out_index[i] + 8, &cur_element_count, 8);
          memcpy(out_index[i] + 24, &enterpoint_node_, 8);
        }
      }

      uint64_t ofs_len = cur_element_count * 8L;

      for (uint64_t vid = 0; vid < cur_element_count; vid++) {
        for (uint32_t p = 0; p < kmeans_part_num; p++) {
          if (LP <= p && p < RP) {
            memcpy(out_index[p] + index_meta_len + vid * 8LL, &index_ofs[p], 8);
          }
        }
        int vid_pid = judge_partition(vid);
        uint32_t *ptr =
            (uint32_t *)(data_level0_memory_ + vid * size_data_per_element_);
        std::vector<uint32_t> vv[kmeans_part_num];
        for (int i = 1; i < ptr[0] + 1; i++) {
          int part = judge_partition(ptr[i]);
          vv[part].emplace_back(ptr[i]);
        }
        std::vector<uint32_t> part_cnt;
        for (int p = 0; p < kmeans_part_num; p++) {
          if (vv[p].size() > 0) {
            part_cnt.push_back(p);
          }
        }

        for (uint32_t j = 0; j < part_cnt.size(); j++) {
          uint32_t end_flag = 0;
          if (j == part_cnt.size() - 1) {
            end_flag = 1 << 31;
          }

          // p is which partition this vector located.
          uint32_t p = part_cnt[j];

          uint32_t msg = end_flag | (p << 16) | vv[p].size();
          for (int remote_p = 0; remote_p < kmeans_part_num; remote_p++) {
            if (remote_p != p) {  // remote ngh
              if (LP <= remote_p && remote_p < RP) {
                memcpy(
                    out_index[remote_p] + index_meta_len + ofs_len +
                        index_ofs[remote_p],
                    &msg, sizeof(msg));
              }
              index_ofs[remote_p] += sizeof(msg);
              // TODO: change to internal offset.
              // case 1: deg <= INLINE_DEG_BAR
              if (vv[p].size() <= INLINE_DEG_BAR) {
                for (auto &v : vv[p]) {
                  if (LP <= remote_p && remote_p < RP) {
                    memcpy(
                        out_index[remote_p] + index_meta_len + ofs_len +
                            index_ofs[remote_p],
                        &v, sizeof(v));
                  }
                  index_ofs[remote_p] += sizeof(v);
                }
              } else {
                // case 2: deg > INLINE_DEG_BAR
                if (LP <= remote_p && remote_p < RP) {
                  memcpy(
                      out_index[remote_p] + index_meta_len + ofs_len +
                          index_ofs[remote_p],
                      &index_ofs[p], sizeof(index_ofs[p]));
                }
                index_ofs[remote_p] += sizeof(index_ofs[p]);
              }
            }
          }
          // local ngh
          if (LP <= p && p < RP) {
            memcpy(
                out_index[p] + index_meta_len + ofs_len + index_ofs[p], &msg,
                sizeof(msg));
          }
          index_ofs[p] += sizeof(msg);
          for (auto &v : vv[p]) {
            if (LP <= p && p < RP) {
              memcpy(
                  out_index[p] + index_meta_len + ofs_len + index_ofs[p], &v,
                  sizeof(v));
            }
            index_ofs[p] += sizeof(v);
          }
        }
      }

      for (uint32_t i = 0; i < kmeans_part_num; i++) {
        if (LP <= i && i < RP) {
          memcpy(out_index[i] + 16, &index_ofs[i], 8);
          printf(
              "part %d index size %llu expect %llu\n", i,
              index_meta_len + ofs_len + index_ofs[i], scala_index_size[i]);
        }
      }

      // **Write index**.
      for (uint32_t p = 0; p < kmeans_part_num; p++) {
        if (LP <= p && p < RP) {
          printf("Write part %d index size %llu...\n", p, scala_index_size[p]);
          uint64_t write_size = 1LL << 28;

          std::ofstream output(scala_index_path[p], std::ios::binary);
          for (uint64_t i = 0; i < scala_index_size[p]; i += write_size) {
            if (i + write_size > scala_index_size[p]) {
              write_size = (uint64_t)(scala_index_size[p] - i);
            }
            output.write((char *)(out_index[p] + i), write_size);
            sleep(4);
            printf(
                "%.2f%%\n",
                (float)(i + write_size) * 100.0 / (scala_index_size[p]));
          }

          printf(
              "partition index file %d write over, expect file size %llu\n", p,
              scala_index_size[p]);
          output.close();
          free(out_index[p]);
        }
      }
    }

    // char *out_index[kmeans_part_num];
    // uint64_t index_ofs[kmeans_part_num];
    // for (uint32_t i = 0; i < kmeans_part_num; i++) index_ofs[i] = 0;
    // for (uint32_t i = 0; i < kmeans_part_num; i++) {
    //   out_index[i] = (char *)malloc(scala_index_size[i]);
    //   memcpy(out_index[i], &i, 4);
    //   memcpy(out_index[i] + 4, &kmeans_part_num, 4);
    //   memcpy(out_index[i] + 8, &cur_element_count, 8);
    //   memcpy(out_index[i] + 24, &enterpoint_node_, 8);
    // }

    // uint64_t ofs_len = cur_element_count * 8L;

    // for (uint64_t vid = 0; vid < cur_element_count; vid++) {
    //   for (uint32_t p = 0; p < kmeans_part_num; p++) {
    //     memcpy(out_index[p] + index_meta_len + vid * 8LL, &index_ofs[p], 8);
    //   }
    //   int vid_pid = judge_partition(vid);
    //   uint32_t *ptr =
    //       (uint32_t *)(data_level0_memory_ + vid * size_data_per_element_);
    //   std::vector<uint32_t> vv[kmeans_part_num];
    //   for (int i = 1; i < ptr[0] + 1; i++) {
    //     int part = judge_partition(ptr[i]);
    //     vv[part].emplace_back(ptr[i]);
    //   }
    //   std::vector<uint32_t> part_cnt;
    //   for (int p = 0; p < kmeans_part_num; p++) {
    //     if (vv[p].size() > 0) {
    //       part_cnt.push_back(p);
    //     }
    //   }

    //   for (uint32_t j = 0; j < part_cnt.size(); j++) {
    //     uint32_t end_flag = 0;
    //     if (j == part_cnt.size() - 1) {
    //       end_flag = 1 << 31;
    //     }

    //     // p is which partition this vector located.
    //     uint32_t p = part_cnt[j];

    //     uint32_t msg = end_flag | (p << 16) | vv[p].size();
    //     for (int remote_p = 0; remote_p < kmeans_part_num; remote_p++) {
    //       if (remote_p != p) {  // remote ngh
    //         memcpy(
    //             out_index[remote_p] + index_meta_len + ofs_len +
    //                 index_ofs[remote_p],
    //             &msg, sizeof(msg));
    //         index_ofs[remote_p] += sizeof(msg);
    //         // TODO: change to internal offset.
    //         // case 1: deg <= INLINE_DEG_BAR
    //         if (vv[p].size() <= INLINE_DEG_BAR) {
    //           for (auto &v : vv[p]) {
    //             memcpy(
    //                 out_index[remote_p] + index_meta_len + ofs_len +
    //                     index_ofs[remote_p],
    //                 &v, sizeof(v));
    //             index_ofs[remote_p] += sizeof(v);
    //           }
    //         } else {
    //           // case 2: deg > INLINE_DEG_BAR
    //           memcpy(
    //               out_index[remote_p] + index_meta_len + ofs_len +
    //                   index_ofs[remote_p],
    //               &index_ofs[p], sizeof(index_ofs[p]));
    //           index_ofs[remote_p] += sizeof(index_ofs[p]);
    //         }
    //       }
    //     }
    //     // local ngh
    //     memcpy(
    //         out_index[p] + index_meta_len + ofs_len + index_ofs[p], &msg,
    //         sizeof(msg));
    //     index_ofs[p] += sizeof(msg);
    //     for (auto &v : vv[p]) {
    //       memcpy(
    //           out_index[p] + index_meta_len + ofs_len + index_ofs[p], &v,
    //           sizeof(v));
    //       index_ofs[p] += sizeof(v);
    //     }
    //   }
    // }

    // for (uint32_t i = 0; i < kmeans_part_num; i++) {
    //   memcpy(out_index[i] + 16, &index_ofs[i], 8);
    //   printf(
    //       "part %d index size %llu expect %llu\n", i,
    //       index_meta_len + ofs_len + index_ofs[i], scala_index_size[i]);
    // }

    // // **Write index**.
    // for (uint32_t p = 0; p < kmeans_part_num; p++) {
    //   printf("Write part %d index size %llu...\n", p, scala_index_size[p]);
    //   uint64_t write_size = 1LL << 28;

    //   std::ofstream output(scala_index_path[p], std::ios::binary);
    //   for (uint64_t i = 0; i < scala_index_size[p]; i += write_size) {
    //     if (i + write_size > scala_index_size[p]) {
    //       write_size = (uint64_t)(scala_index_size[p] - i);
    //     }
    //     output.write((char *)(out_index[p] + i), write_size);
    //     sleep(4);
    //     printf(
    //         "%.2f%%\n",
    //         (float)(i + write_size) * 100.0 / (scala_index_size[p]));
    //   }

    //   printf(
    //       "partition index file %d write over, expect file size %llu\n", p,
    //       scala_index_size[p]);
    //   output.close();
    // }

    // Step2: Write to each partition data.
    /*
      <partition_id: uint32_t>, <partition_num: uint32_t>,
      <partition_vec_num: uint32_t>, <all_vec_num: uint64_t>,
      (header 20 bytes)
      <enterpoint vector: dim * type + label>
      <vector: dim * type + label> * partition_vec_num.
    */
    printf("Build data ...\n");
    char *out_data[kmeans_part_num];
    uint64_t data_ofs[kmeans_part_num];
    for (uint32_t i = 0; i < kmeans_part_num; i++) data_ofs[i] = data_meta_len;
    for (uint32_t i = 0; i < kmeans_part_num; i++) {
      out_data[i] = (char *)malloc(scala_data_size[i]);
      memcpy(out_data[i], &i, 4);
      memcpy(out_data[i] + 4, &kmeans_part_num, 4);
      memcpy(out_data[i] + 8, &kmeans_part_size[i], 4);
      memcpy(out_data[i] + 12, &cur_element_count, 8);
    }
    // copy enterpoint
    char *enter_ptr =
        (char *)(data_level0_memory_ +
                 enterpoint_node_ * size_data_per_element_ + offsetData_);
    for (uint32_t p = 0; p < kmeans_part_num; p++) {
      memcpy(
          out_data[p] + data_ofs[p], enter_ptr,
          index_param.vec_size + sizeof(uint64_t));
      data_ofs[p] += index_param.vec_size + sizeof(uint64_t);
    }

    for (uint64_t vid = 0; vid < cur_element_count; vid++) {
      int pid = judge_partition(vid);
      char *ptr = (char *)(data_level0_memory_ + vid * size_data_per_element_ +
                           offsetData_);
      memcpy(
          out_data[pid] + data_ofs[pid], ptr,
          index_param.vec_size + sizeof(uint64_t));
      data_ofs[pid] += index_param.vec_size + sizeof(uint64_t);
    }

    for (uint32_t p = 0; p < kmeans_part_num; p++) {
      assert(data_ofs[p] == scala_data_size[p]);
      printf("Write part %d size %llu ...\n", p, scala_data_size[p]);
      uint64_t write_size = 1LL << 28;

      std::ofstream output(scala_data_path[p], std::ios::binary);
      for (uint64_t i = 0; i < scala_data_size[p]; i += write_size) {
        if (i + write_size > scala_data_size[p]) {
          write_size = (uint64_t)(scala_data_size[p] - i);
        }
        output.write((char *)(out_data[p] + i), write_size);
        sleep(5);
        printf(
            "%.2f%%\n", (float)(i + write_size) * 100.0 / (scala_data_size[p]));
      }

      printf(
          "partition data file %d write over, expect file size %llu\n", p,
          scala_data_size[p]);
      output.close();
    }
  }

  /**
   * Transfer b1 kmeans index to scalagraph v3 format.
   * This is used for verify v3 performance.
   * v3 format only store the local_ngh array and remote_ngh array
   * local_ngh array: fixed size for the vertex only in this partition, storing
   * the local ngh and pointer to other machine ngh. remote_ngh array: for the
   * edge that indicated to this partition vertex.
   */
  void transfer_b1_to_scala_v3(int part_num, int million_num) {
    char label_pos_path[1024];
    snprintf(
        label_pos_path, sizeof(label_pos_path),
        "/data/share/users/xyzhi/data/bigann/vamana/balance_kmeans_%d_part/"
        "100M_%d_labelpos_map.bin",
        part_num, part_num);

    int *kmeans_part_size;
    int kmeans_vec_size, kmeans_part_num;
    get_label_pos_map(
        label_pos_path, kmeans_vec_size, kmeans_part_num, kmeans_part_size,
        kmeans_label_pos_map);

    uint32_t cumupsize[kmeans_part_num + 1];
    memset(cumupsize, 0, sizeof(cumupsize));
    for (int p = 1; p < kmeans_part_num + 1; p++) {
      cumupsize[p] = cumupsize[p - 1] + kmeans_part_size[p - 1];
    }

    auto judge_partition = [&](uint64_t vid) {
      for (int p = 0; p < kmeans_part_num; p++) {
        if (cumupsize[p + 1] > vid) return p;
      }
      return -1;
    };

    char kmeans_b1_path[1024];
    snprintf(
        kmeans_b1_path, sizeof(kmeans_b1_path),
        "/data/share/users/xyzhi/data/bigann/vamana/balance_kmeans_%d_part/"
        "kmeans_b1_mem.index",
        part_num);

    printf("loading kmeans b1graph index from %s ...\n", kmeans_b1_path);
    std::ifstream input(kmeans_b1_path, std::ios::binary);

    if (!input.is_open()) throw std::runtime_error("Cannot open file");

    size_t max_elements_;
    uint64_t enterpoint_node_;
    input.read((char *)&max_elements_, 8);
    input.read((char *)&enterpoint_node_, 4);

    size_t cur_element_count = max_elements_;
    size_t size_data_per_element_ = 332;
    size_t label_offset_ = 324;
    size_t maxM0_ = 48;
    size_t offsetData_ = 4 + maxM0_ * 4;

    char *data_level0_memory_ =
        (char *)malloc(max_elements_ * size_data_per_element_);
    if (data_level0_memory_ == nullptr)
      throw std::runtime_error(
          "Not enough memory: loadIndex failed to allocate level0");
    input.read(data_level0_memory_, cur_element_count * size_data_per_element_);

    printf(
        "Kmeans index file read over, enter node %llu, part num %d\n",
        enterpoint_node_, kmeans_part_num);

    // Each index partition path.
    char scala_index_path[kmeans_part_num][1024];
    uint32_t subset_size_milllions = 100;
    for (int i = 0; i < kmeans_part_num; i++) {
      snprintf(
          scala_index_path[i], sizeof(scala_index_path[i]),
          "/data/share/users/xyzhi/data/bigann/scalagraph/scalagraph_v3/"
          "%dM_index_%d-%d.bin",
          subset_size_milllions, i, kmeans_part_num);
    }
    char scala_data_path[kmeans_part_num][1024];
    for (int i = 0; i < kmeans_part_num; i++) {
      snprintf(
          scala_data_path[i], sizeof(scala_data_path[i]),
          "/data/share/users/xyzhi/data/bigann/scalagraph/scalagraph_v3/"
          "%dM_data_%d-%d.bin",
          subset_size_milllions, i, kmeans_part_num);
    }

    /*
      [index format]:
      <partition_id: uint32_t>, <partition_num: uint32_t>,
      <part_vec_num: uint32_t>, <local_ngh_ele_size: uint32_t>,
      <enter_point: uint64_t>,
      <remote_ngh_num: uint64_t>
      (header 32 bytes)
      <local_ngh_array: local_ngh_ele_size> * part_vec_num,
      <remote_ngh_array: uint32_t> * remote_ngh_num.
    */
    printf("Start compute scala partition size ...\n");
    // Step1: Compute each partition size.
    size_t index_meta_len = 4 * sizeof(uint32_t) + 2 * sizeof(uint64_t);
    size_t data_meta_len = 20LL;
    /**
     * each ngh max store:
     * max_deg * ngh + partition_deg_msg * part_num
     */
    size_t max_deg = 48;
    size_t local_ngh_ele_size =
        max_deg * sizeof(uint32_t) + kmeans_part_num * sizeof(uint16_t);

    // local ngh array size and remote ngh array num.
    // final index size[p] = meta_size + local_ngh_size[p] +
    // remote_ngh_num[p]*4.
    size_t local_ngh_size[kmeans_part_num], remote_ngh_num[kmeans_part_num];
    size_t scala_data_size[kmeans_part_num];

    for (int p = 0; p < kmeans_part_num; p++) {
      local_ngh_size[p] = (size_t)(kmeans_part_size[p] * local_ngh_ele_size);
      remote_ngh_num[p] = 0;
      scala_data_size[p] =
          (size_t)(data_meta_len + (index_param.vec_size + sizeof(uint64_t)) *
                                       (kmeans_part_size[p] + 1));
    }
    for (uint64_t vid = 0; vid < cur_element_count; vid++) {
      int vid_pid = judge_partition(vid);
      int *ptr = (int *)(data_level0_memory_ + vid * size_data_per_element_);
      int cnt[kmeans_part_num];  // ngh distribute count.
      for (int i = 0; i < kmeans_part_num; i++) cnt[i] = 0;
      for (int i = 1; i < ptr[0] + 1; i++) {
        cnt[judge_partition(ptr[i])]++;
      }
      for (int p = 0; p < kmeans_part_num; p++) {
        // get all remote ngh.
        if (cnt[p] > INLINE_DEG_BAR && p != vid_pid) {
          // case: > INLINE_DEG_BAR, need to store to remote.
          remote_ngh_num[p] += cnt[p];
        }
      }
    }

    size_t expect_index_size[kmeans_part_num];
    size_t expect_data_size[kmeans_part_num];
    for (int p = 0; p < kmeans_part_num; p++) {
      expect_index_size[p] = index_meta_len + local_ngh_size[p] +
                             remote_ngh_num[p] * sizeof(uint32_t);
      expect_data_size[p] = scala_data_size[p];
      printf(
          "EXPECT scala_index_size: %llu, scala_data_size: %llu\n",
          expect_index_size[p], expect_data_size[p]);
    }
    printf("Start reformat index and data... \n");

    printf("Rewrite index ...\n");
    char *index_ptr[kmeans_part_num];
    size_t local_ngh_ofs[kmeans_part_num];
    size_t remote_ngh_ofs[kmeans_part_num];
    for (uint32_t p = 0; p < kmeans_part_num; p++) {
      index_ptr[p] = (char *)malloc(expect_index_size[p]);
      local_ngh_ofs[p] = 0;
      remote_ngh_ofs[p] = index_meta_len + local_ngh_size[p];
    }
    for (uint32_t p = 0; p < kmeans_part_num; p++) {
      printf(
          "Rewrite part %d expect index size %llu ...\n", p,
          expect_index_size[p]);
      /*
        [index format]:
        <partition_id: uint32_t>, <partition_num: uint32_t>,
        <part_vec_num: uint32_t>, <local_ngh_ele_size: uint32_t>,
        <enter_point: uint64_t>,
        <remote_ngh_num: uint64_t>
        (header 32 bytes)
        <local_ngh_array: local_ngh_ele_size> * part_vec_num,
        <remote_ngh_array: uint32_t> * remote_ngh_num.
      */
      memcpy(index_ptr[p] + local_ngh_ofs[p], &p, sizeof(uint32_t));
      local_ngh_ofs[p] += sizeof(uint32_t);
      memcpy(
          index_ptr[p] + local_ngh_ofs[p], &kmeans_part_num, sizeof(uint32_t));
      local_ngh_ofs[p] += sizeof(uint32_t);
      memcpy(
          index_ptr[p] + local_ngh_ofs[p], &kmeans_part_size[p],
          sizeof(uint32_t));
      local_ngh_ofs[p] += sizeof(uint32_t);
      memcpy(
          index_ptr[p] + local_ngh_ofs[p], &local_ngh_ele_size,
          sizeof(uint32_t));
      local_ngh_ofs[p] += sizeof(uint32_t);
      memcpy(
          index_ptr[p] + local_ngh_ofs[p], &enterpoint_node_, sizeof(uint64_t));
      local_ngh_ofs[p] += sizeof(uint64_t);
      memcpy(
          index_ptr[p] + local_ngh_ofs[p], &remote_ngh_num[p],
          sizeof(uint64_t));
      local_ngh_ofs[p] += sizeof(uint64_t);
      printf(
          "meta part num %u ngh_num %llu\n", kmeans_part_size[p],
          remote_ngh_num[p]);

      for (uint32_t vid = cumupsize[p]; vid < cumupsize[p + 1];
           vid++, local_ngh_ofs[p] += local_ngh_ele_size) {
        if (vid % 1000000 == 0) {
          std::cout << (double)(vid - cumupsize[p]) * 100.0 /
                           (cumupsize[p + 1] - cumupsize[p])
                    << "%\r" << std::flush;
        }
        // p is source node partition id, q is target node partition id.
        size_t local_ofs = local_ngh_ofs[p];
        uint32_t *ptr =
            (uint32_t *)(data_level0_memory_ + vid * size_data_per_element_);
        std::vector<uint32_t> nghs[kmeans_part_num];
        for (int i = 1; i < ptr[0] + 1; i++) {
          int q = judge_partition(ptr[i]);
          nghs[q].emplace_back(ptr[i]);
        }
        std::vector<uint32_t> part_ids;
        for (int q = 0; q < kmeans_part_num; q++) {
          if (nghs[q].size() > 0 && (q != p)) {
            part_ids.push_back(q);
          }
        }
        // put the local ngh to the last
        if (nghs[p].size() > 0) {
          part_ids.push_back(p);
        }

        for (uint32_t i = 0; i < part_ids.size(); i++) {
          uint32_t end_flag = 0;
          // q is which partition this vector ngh located.
          uint32_t q = part_ids[i];
          if (i == part_ids.size() - 1) {
            end_flag = (1 << 15);
            if (q == p) {
              end_flag |= (1 << 14);
            }
          }

          uint16_t msg = end_flag | (q << 8) | nghs[q].size();

          // write local ngh
          memcpy(index_ptr[p] + local_ofs, &msg, sizeof(msg));
          local_ofs += sizeof(msg);
          if (q != p) {
            // remote nghs
            if (nghs[q].size() <= INLINE_DEG_BAR) {
              for (auto &v : nghs[q]) {
                memcpy(index_ptr[p] + local_ofs, &v, sizeof(v));
                local_ofs += sizeof(v);
              }
            } else {
              size_t relative_ofs = remote_ngh_ofs[q] - index_meta_len;
              memcpy(
                  index_ptr[p] + local_ofs, &relative_ofs,
                  sizeof(relative_ofs));
              local_ofs += sizeof(relative_ofs);
              for (auto &v : nghs[q]) {
                memcpy(index_ptr[q] + remote_ngh_ofs[q], &v, sizeof(v));
                remote_ngh_ofs[q] += sizeof(v);
              }
            }
          } else {
            // local nghs
            for (auto &v : nghs[q]) {
              memcpy(index_ptr[p] + local_ofs, &v, sizeof(v));
              local_ofs += sizeof(v);
            }
          }
          if (local_ofs > local_ngh_ofs[p] + local_ngh_ele_size) {
            printf(
                "Error: index exceed.. vid %d local_ofs %llu local_ngh_ofs "
                "%llu local_ngh_ele_size: %llu\n",
                vid, local_ofs, local_ngh_ofs[p], local_ngh_ele_size);
            abort();
          }
        }
      }
    }

    for (uint32_t p = 0; p < kmeans_part_num; p++) {
      size_t actual_size = remote_ngh_ofs[p];
      size_t expect_size = expect_index_size[p];
      printf(
          "part %d index size %llu expect %llu\n", p, actual_size, expect_size);
      if (actual_size != expect_size) {
        printf(
            "Error: part %d index size %llu expect %llu\n", p, actual_size,
            expect_size);
        abort();
      }
    }

    // **Write index**.
    // for (uint32_t p = 0; p < kmeans_part_num; p++) {
    //   printf("Write part %d index size %llu...\n", p, expect_index_size[p]);
    //   uint64_t write_size = 1LL << 28;

    //   std::ofstream output(scala_index_path[p], std::ios::binary);
    //   for (uint64_t i = 0; i < expect_index_size[p]; i += write_size) {
    //     if (i + write_size > expect_index_size[p]) {
    //       write_size = (uint64_t)(expect_index_size[p] - i);
    //     }
    //     output.write((char *)(index_ptr[p] + i), write_size);
    //     sleep(4);
    //     std::cout << "write "
    //               << (float)(i + write_size) * 100.0 / (expect_index_size[p])
    //               << " %\r" << std::flush;
    //   }
    //   printf(
    //       "partition %d index file %s write over, expect file size %llu\n",
    //       p, scala_index_path[p], expect_index_size[p]);
    // }
    // // clear index memory
    // for (uint32_t p = 0; p < kmeans_part_num; p++) {
    //   free(index_ptr[p]);
    // }

    // // Step2: Write to each partition data.
    // /*
    //   <partition_id: uint32_t>, <partition_num: uint32_t>,
    //   <partition_vec_num: uint32_t>, <all_vec_num: uint64_t>,
    //   (header 20 bytes)
    //   <enterpoint vector: dim * type + label>
    //   <vector: dim * type + label> * partition_vec_num.
    // */
    // printf("Rewrite data ...\n");
    // char *out_data[kmeans_part_num];
    // uint64_t data_ofs[kmeans_part_num];
    // for (uint32_t i = 0; i < kmeans_part_num; i++) data_ofs[i] =
    // data_meta_len; for (uint32_t i = 0; i < kmeans_part_num; i++) {
    //   out_data[i] = (char *)malloc(scala_data_size[i]);
    //   memcpy(out_data[i], &i, 4);
    //   memcpy(out_data[i] + 4, &kmeans_part_num, 4);
    //   memcpy(out_data[i] + 8, &kmeans_part_size[i], 4);
    //   memcpy(out_data[i] + 12, &cur_element_count, 8);
    // }
    // // copy enterpoint
    // char *enter_ptr =
    //     (char *)(data_level0_memory_ +
    //              enterpoint_node_ * size_data_per_element_ + offsetData_);
    // for (uint32_t p = 0; p < kmeans_part_num; p++) {
    //   memcpy(out_data[p] + data_ofs[p], enter_ptr, index_param.vec_size +
    //   sizeof(uint64_t)); data_ofs[p] += index_param.vec_size +
    //   sizeof(uint64_t);
    // }

    // for (uint64_t vid = 0; vid < cur_element_count; vid++) {
    //   int pid = judge_partition(vid);
    //   char *ptr = (char *)(data_level0_memory_ + vid * size_data_per_element_
    //   +
    //                        offsetData_);
    //   memcpy(out_data[pid] + data_ofs[pid], ptr, index_param.vec_size +
    //   sizeof(uint64_t)); data_ofs[pid] += index_param.vec_size +
    //   sizeof(uint64_t);
    // }

    // for (uint32_t p = 0; p < kmeans_part_num; p++) {
    //   assert(data_ofs[p] == scala_data_size[p]);
    //   printf("Write part %d data size %llu ...\n", p, scala_data_size[p]);
    //   uint64_t write_size = 1LL << 28;

    //   std::ofstream output(scala_data_path[p], std::ios::binary);
    //   for (uint64_t i = 0; i < scala_data_size[p]; i += write_size) {
    //     if (i + write_size > scala_data_size[p]) {
    //       write_size = (uint64_t)(scala_data_size[p] - i);
    //     }
    //     output.write((char *)(out_data[p] + i), write_size);
    //     sleep(4);
    //     std::cout << "write "
    //               << (float)(i + write_size) * 100.0 / (scala_data_size[p])
    //               << " %\r" << std::flush;
    //   }

    //   printf(
    //       "partition %d data file %s write over, expect file size %llu\n", p,
    //       scala_data_path[p], scala_data_size[p]);
    //   output.close();
    // }
    // // clear data memory
    // for (uint32_t p = 0; p < kmeans_part_num; p++) {
    //   free(out_data[p]);
    // }
    // free(data_level0_memory_);
  }

  /**
   * Transfer kmeans vamana index to **force balance** scalagraph format.
   */
  void transfer_kmeans_vamana_balance(
      std::string vamana_path = "", std::string scg_path = "") {
    // TODO: use cmd
    // for forced balance kmeans
    std::string label_pos_path =
        "/data/share/users/xyzhi/data/bigann/vamana/kmeans/"
        "100M_4_labelpos_map.bin";

    int *kmeans_part_size;
    int kmeans_vec_size, kmeans_part_num;
    get_label_pos_map(
        label_pos_path.c_str(), kmeans_vec_size, kmeans_part_num,
        kmeans_part_size, kmeans_label_pos_map);

    uint32_t sum_part_num = 0;
    for (uint32_t p = 0; p < kmeans_part_num; p++) {
      sum_part_num += kmeans_part_size[p];
    }
    uint32_t accumu_part_num = 0;
    for (uint32_t p = 0; p < kmeans_part_num; p++) {
      kmeans_part_size[p] = std::min(
          sum_part_num / kmeans_part_num + 1, sum_part_num - accumu_part_num);
      accumu_part_num += kmeans_part_size[p];
    }
    assert(accumu_part_num == sum_part_num);

    std::string new_label_pos_path =
        "/data/share/users/xyzhi/data/bigann/scalagraph/balance_v0/"
        "balance_v0_100M_4_labelpos_map.bin";
    std::ofstream new_label_pos_input(new_label_pos_path, std::ios::binary);
    new_label_pos_input.write((char *)&kmeans_vec_size, 4);
    printf("new label_pos vec_size %d\n", kmeans_vec_size);
    new_label_pos_input.write((char *)&kmeans_part_num, 4);
    printf("new label_pos part_num %d\n", kmeans_part_num);

    new_label_pos_input.write((char *)kmeans_part_size, 4 * kmeans_part_num);
    printf("write new lobel_pos part num: ");
    for (int i = 0; i < kmeans_part_num; i++) {
      printf("%d ", kmeans_part_size[i]);
    }
    printf("to %s\n", new_label_pos_path.c_str());
    new_label_pos_input.write(
        (char *)kmeans_label_pos_map, 4 * kmeans_vec_size);

    new_label_pos_input.close();

    auto judge_partition = [&](uint64_t vid) {
      size_t cumunum = 0;
      for (int p = 0; p < kmeans_part_num; p++) {
        cumunum += kmeans_part_size[p];
        if (cumunum > vid) return p;
      }
      return -1;
    };

    char default_sift100M_kmeans_path[1024];
    snprintf(
        default_sift100M_kmeans_path, sizeof(default_sift100M_kmeans_path),
        "/data/share/users/xyzhi/data/bigann/vamana/kmeans/"
        "kmeans_index_mem.index");

    printf("loading index from %s ...\n", default_sift100M_kmeans_path);
    std::ifstream input(default_sift100M_kmeans_path, std::ios::binary);

    if (!input.is_open()) throw std::runtime_error("Cannot open file");

    size_t max_elements_;
    uint64_t enterpoint_node_;
    input.read((char *)&max_elements_, 8);
    input.read((char *)&enterpoint_node_, 4);

    size_t cur_element_count = max_elements_;
    // offsetLevel0_ = 0;

    size_t size_data_per_element_ = 332;
    size_t label_offset_ = 324;
    size_t maxM0_ = 48;
    size_t offsetData_ = 4 + maxM0_ * 4;

    char *data_level0_memory_ =
        (char *)malloc(max_elements_ * size_data_per_element_);
    if (data_level0_memory_ == nullptr)
      throw std::runtime_error(
          "Not enough memory: loadIndex failed to allocate level0");
    input.read(data_level0_memory_, cur_element_count * size_data_per_element_);

    printf(
        "Kmeans index file read over, enter node %llu, part num %d\n",
        enterpoint_node_, kmeans_part_num);

    // Each index partition path.
    char scala_index_path[kmeans_part_num][1024];
    uint32_t subset_size_milllions = 100;
    for (int i = 0; i < kmeans_part_num; i++) {
      snprintf(
          scala_index_path[i], sizeof(scala_index_path[i]),
          "/data/share/users/xyzhi/data/bigann/scalagraph/balance_v0/"
          "%dM_index_%d-%d.bin",
          subset_size_milllions, i, kmeans_part_num);
    }
    char scala_data_path[kmeans_part_num][1024];
    for (int i = 0; i < kmeans_part_num; i++) {
      snprintf(
          scala_data_path[i], sizeof(scala_data_path[i]),
          "/data/share/users/xyzhi/data/bigann/scalagraph/balance_v0/"
          "%dM_data_%d-%d.bin",
          subset_size_milllions, i, kmeans_part_num);
    }

    printf("Start compute scala partition size ...\n");
    // Step1: Compute each partition size.
    uint64_t index_meta_len = 32LL;
    uint64_t data_meta_len = 20LL;
    uint64_t scala_index_size[kmeans_part_num],
        scala_data_size[kmeans_part_num];
    for (int p = 0; p < kmeans_part_num; p++) {
      scala_index_size[p] =
          (uint64_t)(index_meta_len + cur_element_count * 8LL);
      scala_data_size[p] =
          (uint64_t)(data_meta_len + (index_param.vec_size + sizeof(uint64_t)) *
                                         (kmeans_part_size[p] + 1LL));
    }
    for (uint64_t vid = 0; vid < cur_element_count; vid++) {
      int vid_pid = judge_partition(vid);
      int *ptr = (int *)(data_level0_memory_ + vid * size_data_per_element_);
      int cnt[kmeans_part_num];  // ngh distribute count.
      for (int i = 0; i < kmeans_part_num; i++) cnt[i] = 0;
      for (int i = 1; i < ptr[0] + 1; i++) {
        cnt[judge_partition(ptr[i])]++;
      }
      for (int p = 0; p < kmeans_part_num; p++) {
        if (cnt[p] > 0) {
          for (int ngh_p = 0; ngh_p < kmeans_part_num; ngh_p++) {
            if (ngh_p == p) {  // local ngh
              scala_index_size[ngh_p] +=
                  (uint64_t)((cnt[p] + 1LL) * 4LL);              // deg + nghs
            } else {                                             // remote ngh
              scala_index_size[ngh_p] += (uint64_t)(4LL + 8LL);  // deg + ptr
            }
          }
        }
      }
    }

    for (int p = 0; p < kmeans_part_num; p++) {
      printf(
          "scala_index_size: %llu, scala_data_size: %llu\n",
          scala_index_size[p], scala_data_size[p]);
    }
    printf("Start write index and data... \n");
    // scala_index_size: 8359571036, scala_data_size: 5208751332
    // scala_index_size: 5567326640, scala_data_size: 2900225452
    // scala_index_size: 6153537612, scala_data_size: 3239405780
    // scala_index_size: 4952724792, scala_data_size: 2251617516

    /*
      <partition_id: uint32_t>, <partition_num: uint32_t>,
      <all_vec_num: uint64_t>, <deg_ngh_size: uint64_t>,
      <enter_point: uint64_t>
      (header 32 bytes)
      <NeighborOffset: uint64_t> * all_vec_num,
      <DegreeNeighbor: uint32_t> * deg_ngh_num.
    */
    printf("Build index ...\n");
    char *out_index[kmeans_part_num];
    uint64_t index_ofs[kmeans_part_num];
    for (uint32_t i = 0; i < kmeans_part_num; i++) index_ofs[i] = 0;
    for (uint32_t i = 0; i < kmeans_part_num; i++) {
      out_index[i] = (char *)malloc(scala_index_size[i]);
      memcpy(out_index[i], &i, 4);
      memcpy(out_index[i] + 4, &kmeans_part_num, 4);
      memcpy(out_index[i] + 8, &cur_element_count, 8);
      memcpy(out_index[i] + 24, &enterpoint_node_, 8);
    }

    uint64_t ofs_len = cur_element_count * 8L;

    for (uint64_t vid = 0; vid < cur_element_count; vid++) {
      for (uint32_t p = 0; p < kmeans_part_num; p++) {
        memcpy(out_index[p] + index_meta_len + vid * 8LL, &index_ofs[p], 8);
      }
      int vid_pid = judge_partition(vid);
      uint32_t *ptr =
          (uint32_t *)(data_level0_memory_ + vid * size_data_per_element_);
      std::vector<uint32_t> vv[kmeans_part_num];
      for (int i = 1; i < ptr[0] + 1; i++) {
        int part = judge_partition(ptr[i]);
        vv[part].emplace_back(ptr[i]);
      }
      std::vector<uint32_t> part_cnt;
      for (int p = 0; p < kmeans_part_num; p++) {
        if (vv[p].size() > 0) {
          part_cnt.push_back(p);
        }
      }

      for (uint32_t j = 0; j < part_cnt.size(); j++) {
        uint32_t end_flag = 0;
        if (j == part_cnt.size() - 1) {
          end_flag = 1 << 31;
        }

        uint32_t p = part_cnt[j];

        uint32_t msg = end_flag | (p << 16) | vv[p].size();
        if (vid == 0) {
          printf(
              "write vid0 part %u deg %u end %llu\n", p, vv[p].size(),
              end_flag);
          printf(
              "expect read vid0 part %u deg %u end %d\n",
              (msg >> 16) & 0x00000fff, msg & 0x0000ffff, msg >> 31);
        }
        for (int remote_p = 0; remote_p < kmeans_part_num; remote_p++) {
          if (remote_p != p) {  // remote ngh
            if (vid == 0) {
              printf(
                  "write vid0 as remote part %u deg %u end %llu ofs %llu\n",
                  remote_p, vv[p].size(), end_flag, index_ofs[remote_p]);
              printf(
                  "expect read vid0 part %u deg %u end %d\n",
                  (msg >> 16) & 0x00000fff, msg & 0x0000ffff, msg >> 31);
            }
            memcpy(
                out_index[remote_p] + index_meta_len + ofs_len +
                    index_ofs[remote_p],
                &msg, sizeof(msg));
            index_ofs[remote_p] += sizeof(msg);
            memcpy(
                out_index[remote_p] + index_meta_len + ofs_len +
                    index_ofs[remote_p],
                &index_ofs[p], sizeof(index_ofs[p]));
            index_ofs[remote_p] += sizeof(index_ofs[p]);
            if (vid == 0) {
              printf(
                  "remote_p%d index ofs : %u\n", remote_p, index_ofs[remote_p]);
            }
          }
        }
        // local ngh
        memcpy(
            out_index[p] + index_meta_len + ofs_len + index_ofs[p], &msg,
            sizeof(msg));
        index_ofs[p] += sizeof(msg);
        for (auto &v : vv[p]) {
          memcpy(
              out_index[p] + index_meta_len + ofs_len + index_ofs[p], &v,
              sizeof(v));
          index_ofs[p] += sizeof(v);
        }
        if (vid == 0) {
          printf("p%d index ofs : %u\n", p, index_ofs[p]);
        }
      }
    }

    for (uint32_t i = 0; i < kmeans_part_num; i++) {
      memcpy(out_index[i] + 16, &index_ofs[i], 8);
      printf(
          "part %d index size %llu expect %llu\n", i,
          index_meta_len + ofs_len + index_ofs[i], scala_index_size[i]);
    }

    for (uint32_t p = 0; p < kmeans_part_num; p++) {
      printf("Write part %d index size %llu...\n", p, scala_index_size[p]);
      uint64_t write_size = 1LL << 28;

      std::ofstream output(scala_index_path[p], std::ios::binary);
      for (uint64_t i = 0; i < scala_index_size[p]; i += write_size) {
        if (i + write_size > scala_index_size[p]) {
          write_size = (uint64_t)(scala_index_size[p] - i);
        }
        output.write((char *)(out_index[p] + i), write_size);
        sleep(5);
        printf(
            "%.2f%%\n",
            (float)(i + write_size) * 100.0 / (scala_index_size[p]));
      }

      printf(
          "partition index file %d write over, expect file size %llu\n", p,
          scala_index_size[p]);
      output.close();
    }

    // Step2: Write to each partition data.
    /*
      <partition_id: uint32_t>, <partition_num: uint32_t>,
      <partition_vec_num: uint32_t>, <all_vec_num: uint64_t>,
      (header 20 bytes)
      <enterpoint vector: dim * type + label>
      <vector: dim * type + label> * partition_vec_num.
    */
    printf("Build data ...\n");
    char *out_data[kmeans_part_num];
    uint64_t data_ofs[kmeans_part_num];
    for (uint32_t i = 0; i < kmeans_part_num; i++) data_ofs[i] = data_meta_len;
    for (uint32_t i = 0; i < kmeans_part_num; i++) {
      out_data[i] = (char *)malloc(scala_data_size[i]);
      memcpy(out_data[i], &i, 4);
      memcpy(out_data[i] + 4, &kmeans_part_num, 4);
      memcpy(out_data[i] + 8, &kmeans_part_size[i], 4);
      memcpy(out_data[i] + 12, &cur_element_count, 8);
    }
    // copy enterpoint
    char *enter_ptr =
        (char *)(data_level0_memory_ +
                 enterpoint_node_ * size_data_per_element_ + offsetData_);
    for (uint32_t p = 0; p < kmeans_part_num; p++) {
      memcpy(
          out_data[p] + data_ofs[p], enter_ptr,
          index_param.vec_size + sizeof(uint64_t));
      data_ofs[p] += index_param.vec_size + sizeof(uint64_t);
    }

    for (uint64_t vid = 0; vid < cur_element_count; vid++) {
      int pid = judge_partition(vid);
      char *ptr = (char *)(data_level0_memory_ + vid * size_data_per_element_ +
                           offsetData_);
      memcpy(
          out_data[pid] + data_ofs[pid], ptr,
          index_param.vec_size + sizeof(uint64_t));
      data_ofs[pid] += index_param.vec_size + sizeof(uint64_t);
    }

    for (uint32_t p = 0; p < kmeans_part_num; p++) {
      assert(data_ofs[p] == scala_data_size[p]);
      printf("Write part %d size %llu ...\n", p, scala_data_size[p]);
      uint64_t write_size = 1LL << 28;

      std::ofstream output(scala_data_path[p], std::ios::binary);
      for (uint64_t i = 0; i < scala_data_size[p]; i += write_size) {
        if (i + write_size > scala_data_size[p]) {
          write_size = (uint64_t)(scala_data_size[p] - i);
        }
        output.write((char *)(out_data[p] + i), write_size);
        sleep(5);
        printf(
            "%.2f%%\n", (float)(i + write_size) * 100.0 / (scala_data_size[p]));
      }

      printf(
          "partition data file %d write over, expect file size %llu\n", p,
          scala_data_size[p]);
      output.close();
    }
  }
};
