#pragma once

#include <boost/program_options.hpp>
#include <iostream>

#include "coromem/include/utils.h"
#include "diskann/include/cached_io.h"
#include "diskann/include/common_includes.h"
#include "diskann/include/utils.h"
// #include "kmeans.h"
#include "rdma/rdma_config.h"

#define TARGET_PARTS (MACHINE_NUM)
// #define TARGET_PARTS 2
// #define TARGET_PARTS 8
// #define TARGET_PARTS 16
#define NUM_REPLICA 2
#define KMEANS_SAMPLE_SIZE 1000000
#define META_DATA_RATIO (0.01)

#define INLINE_DEG_BAR (2)

namespace po = boost::program_options;

typedef enum {
  HNSW_RANDOM,
  HNSW_KMEANS,
  VAMANA,
  KmeansSharedNothing,
  SharedNothing,
  SCALA_V3
} GraphType;

class IndexParameter {
 public:
  size_t vec_num;       // vector num
  uint32_t max_vid;     // = vector num
  size_t dim;           // vector dim
  size_t vec_size;      // vector size
  uint32_t machine_id;  // local machine id.
  uint32_t num_parts;

  uint32_t subset_size_milllions; // vec_num = subset_size_milllions * 1e6
  uint32_t thread_num;

  size_t local_max_vid;  // local partitioon max node id
  uint32_t max_degree;   // final max degree

  size_t b1_ele_size;

  size_t replica_num;  // partition replica number.

  std::string config_file; 

  std::string graph_type_str, data_type, dist_fn, data_path, index_path_prefix,
      codebook_prefix, label_file, universal_label, label_type;

  GraphType graph_type;

  // TODO: merge this
  std::string index_prefix_path;

  std::string local_replica_base_file;
  std::string local_noreplica_base_file;
  std::string local_replica_idmap_filename;
  std::string local_noreplica_idmap_filename;
  std::string local_replica_labels_file;
  std::string local_noreplica_labels_file;
  std::string local_index_file;  // not final index, before merge
  std::string temp_b1_index;     // for test, **all** merged b1 index.
  std::string temp_label2pos_map_file;
  std::string temp_pos2label_map_file;
  std::string temp_id2part_map_file;

  std::string b2_index_file[MACHINE_NUM];
  std::string local_b2_index_file;

  // only for b1graph transfered from vamana
  std::string temp_kmeans_data_file; // only store the kmeans vector data.
  // std::string temp_vamana_b1graph_file; // store the b1graph transfer from vamana, now set equal to final

  size_t disk_pq_dims = 0;
  bool use_disk_pq = false;
  size_t build_pq_bytes = 0;

  std::string data_file_to_use;
  std::string labels_file_original;
  std::string labels_file_to_use;
  std::string pq_pivots_path_base;
  std::string pq_pivots_path;
  std::string pq_compressed_vectors_path;
  std::string mem_index_path;       // temp index name, use for single machine.
  std::string final_index_path;     // final **local** merged index path (final_part_index_path[local id])
  std::string final_b1_index_path;  // final **local** merged b1 index path (final_part_b1_index_path[local id])
  std::string
      final_scala_index_path;         // final **local** merged scala index path (final_part_scala_index_path[local id])
  std::string final_scala_data_path;  // final **local** merged scala data path (final_part_scala_data_path[local id])
  // final **all** merged scala index and data path.
  std::string final_part_scala_index_path[MACHINE_NUM];
  std::string final_part_scala_data_path[MACHINE_NUM];
  // final **all** merged index path, for test merge
  std::string final_part_index_path[MACHINE_NUM];
  std::string final_part_b1_index_path[MACHINE_NUM];
  std::string disk_index_path;
  std::string medoids_path;
  std::string centroids_path;

  uint32_t topindex_deg;
  std::string top_sample_percent; // top index sampling rate.
  std::string top_index_path;  // store the path of top index.

  std::string labels_to_medoids_path = disk_index_path;
  std::string mem_labels_file = mem_index_path;
  std::string disk_labels_file = disk_index_path;
  std::string mem_univ_label_file = mem_index_path;
  std::string disk_univ_label_file = disk_index_path;
  std::string disk_labels_int_map_file = disk_index_path;
  std::string dummy_remap_file;  // remap will be used if we break-up points of
                                 // high label-density to create copies

  std::string sample_base_prefix;
  // optional, used if disk index file must store pq data
  std::string disk_pq_pivots_path;
  // optional, used if disk index must store pq data
  std::string disk_pq_compressed_vectors_path;
  std::string prepped_base;  // temp file for storing pre-processed base file
                             // for cosine/ mips metrics
  bool created_temp_file_for_processed_data = false;

  std::string final_index_universal_label_file;

  std::string merged_index_prefix;
  std::string merged_meta_index_prefix;

  // for partition ids without replica.
  std::string part_noreplica_idmap_filename[MACHINE_NUM];
  std::string part_noreplica_base_filename[MACHINE_NUM];
  std::string part_noreplica_labels_filename[MACHINE_NUM];

  // for partition ids with replica.
  std::string part_replica_idmap_filename[MACHINE_NUM];
  std::string part_replica_base_filename[MACHINE_NUM];
  std::string part_replica_labels_filename[MACHINE_NUM];
  std::string part_replica_index_filename[MACHINE_NUM];

  uint32_t partition_size[MACHINE_NUM];
  // local partition vector number without replica.
  uint32_t local_partition_size;

  uint32_t medoid;  // local partition medoid

  // store the sorted ids with replica in this partition.
  std::vector<uint32_t> local_replica_ids;
  // store the sorted ids (only belong to this part) in this partition.
  std::vector<uint32_t> local_noreplica_ids;
  // store the sorted ids of each partition (with replica).
  std::vector<uint32_t> part_replica_ids[MACHINE_NUM];
  // store the sorted ids of each partition (without replica).
  std::vector<uint32_t> part_noreplica_ids[MACHINE_NUM];

  // kmeans_idmap[vid]: parition id stored the vid (here vid is label).
  uint32_t *kmeans_id2part_map;

  // whether recv all meta for one partition.
  bool recv_meta[MACHINE_NUM];

  /* dispatch & collect ids */
  size_t dispatch_id_num[MACHINE_NUM];  // for send meta
  char *dispatch_ids[MACHINE_NUM];        // for send ids.
  size_t collect_id_num[MACHINE_NUM];   // for recv meta
  size_t collect_id_cnt[MACHINE_NUM];   // for recv count.
  char *collected_ids[MACHINE_NUM];       // recved id to merge.

  /* dispatch & collect nghs */
  // store the size of dispatch nghs. (local partitions)
  size_t dispatch_ngh_num[MACHINE_NUM];  // for send meta
  char *dispatch_nghs[MACHINE_NUM];        // for send nghs.
  size_t collect_ngh_num[MACHINE_NUM];   // for recv meta
  size_t collect_ngh_cnt[MACHINE_NUM];   // for recv count.
  char *collected_nghs[MACHINE_NUM];       // recved ngh to merge.

  // map
  uint32_t *label2pos_map;
  uint32_t *pos2label_map;

  uint32_t num_threads, L, disk_PQ, build_PQ, QD, Lf, filter_threshold;
  float B, M;
  bool append_reorder_data = false;
  bool use_opq = false;
  bool use_filters = false;
  diskann::Metric metric;
  float _meta_data_ratio;

  IndexParameter() {}
  IndexParameter(int argc, char **argv);
  ~IndexParameter() {}

  void show() { std::cout << "IndexParameter" << std::endl; }

  size_t serialize_info(char *ptr, uint32_t machine_id) {
    size_t offset = 0;
    memcpy(ptr + offset, &max_vid, sizeof(max_vid));
    offset += sizeof(max_vid);
    memcpy(ptr + offset, &dim, sizeof(dim));
    offset += sizeof(dim);
    return offset;
  }

  void deserialize_info(char *ptr) {

  }

};