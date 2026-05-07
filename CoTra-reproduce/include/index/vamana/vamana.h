#pragma once

#include <cassert>
#include <fstream>

#include "anns/hash_table.h"
#include "index/kmeans.h"
#include "index/index_param.h"

/**
 * Read Kmeans VAMANA format index.
 * Degree aligned.
 */

void show_kmeans_vamana(std::string index_path, uint64_t node_num) {
  std::ifstream input(index_path.c_str(), std::ios::binary);

  size_t max_elements_ = 0;
  readBinaryPOD(input, max_elements_);
  uint32_t enterpoint_node_ = 0;
  readBinaryPOD(input, enterpoint_node_);

  printf(
      "kmeans vamana read max %u enter point %u\n", max_elements_,
      enterpoint_node_);
  size_t size_data_per_element_ = 332;
  char *buff = new char[size_data_per_element_ * max_elements_];
  input.read(buff, size_data_per_element_ * max_elements_);
  input.close();
  printf("kmeans vamana read over\n");

  OptHashPosVector hash;

  std::vector<uint32_t> ans;
  std::vector<uint32_t> hop_nodes;
  hop_nodes.emplace_back(enterpoint_node_);
  ans.emplace_back(enterpoint_node_);
  hash.CheckAndSet(enterpoint_node_);
  std::vector<uint32_t> nxt_hop_nodes;

  const uint32_t hnum = 5;
  for (uint32_t hop = 1; hop <= hnum; hop++) {
    for (uint32_t node : hop_nodes) {
      uint32_t *ptr = (uint32_t *)(buff + size_data_per_element_ * node);
      uint32_t deg = ptr[0];
      // printf("node %u deg %u :\n", node, deg);
      for (uint32_t i = 0; i < deg; i++) {
        uint32_t next_node = ptr[i + 1];
        // printf("%u", next_node);
        if (hash.CheckAndSet(next_node)) continue;
        nxt_hop_nodes.emplace_back(next_node);
        ans.emplace_back(next_node);
      }
      // printf("\n");
    }
    hop_nodes.clear();
    swap(hop_nodes, nxt_hop_nodes);

    // compute expect cache size
    size_t bytes = ans.size() * size_data_per_element_;
    size_t MB = bytes >> 20;
    printf("%u-hop size %lu %llu MB\n", hop, ans.size(), MB);
  }
}

/**
 * Read vamana format index.
 * Degree is not aligned.
 *
 */
void show_vamana(std::string index_path, uint64_t node_num) {
  std::ifstream input(index_path.c_str(), std::ios::binary);

  uint64_t index_size = 0;
  uint32_t max_degree = 0;
  input.read((char *)&index_size, sizeof(uint64_t));
  input.read((char *)&max_degree, sizeof(uint32_t));
  uint32_t start;
  input.read((char *)&start, sizeof(uint32_t));
  uint64_t tmp_num = 0;  // useless for now.
  input.read((char *)&tmp_num, sizeof(uint64_t));

  uint64_t read_sz = 24;
  uint64_t neq_maxdeg_cnt = 0;
  char *buf = (char *)malloc(max_degree * sizeof(uint32_t));
  for (uint64_t i = 0; i < node_num; i++) {
    uint32_t deg = 0;
    input.read((char *)&deg, sizeof(uint32_t));
    // printf("deg %llu\n", deg);
    input.read(buf, deg * sizeof(uint32_t));
    if (deg != max_degree) {
      neq_maxdeg_cnt++;
    }
    read_sz += (uint64_t)(sizeof(uint32_t) * (deg + 1));
    // int *ptr = (int *)buf;
    // for (int j = 0; j < deg; j++) {
    //   printf("%llu ", ptr[j]);
    // }
    // printf("\n");
  }
  printf("read_sz %llu\n", read_sz);
  printf("index_size %llu\n", index_size);
  printf("max_degree %llu\n", max_degree);
  printf("start %llu\n", start);
  printf("node_num %llu\n", node_num);
  printf("not equal to max deg vertex cnt: %llu\n", neq_maxdeg_cnt);
  input.close();
}

/*
 Transfer vamana data storage to uniform format.
*/
void transfer_kmeans(int part_num, int million_num) {
  int vec_size = million_num * 1e6;  // 10M/100M/1B
  int *part = (int *)malloc(vec_size * 2 * 4);
  memset(part, -1, vec_size * 2 * 4);
  /*
  part: <fisrt_part_id, second_part_id(if have)> [vec_size]
  part[2*x]: the first part id of vector x.

  label_pos_map[x]: the vector x(label) new position.
  pos_label_map[x]: the label vector in postition x has.
  */
  int *label_pos_map = (int *)malloc(vec_size * 4);
  memset(label_pos_map, -1, vec_size * 4);
  int *pos_label_map = (int *)malloc(vec_size * 4);
  memset(pos_label_map, -1, vec_size * 4);
  int *part_offset = (int *)malloc(vec_size * 4);
  memset(part_offset, -1, vec_size * 4);
  int *part_size = (int *)malloc(part_num * 4);
  memset(part_size, -1, part_num * 4);

  char partition_path[part_num][1024];
  for (int i = 0; i < part_num; i++) {
    snprintf(
        partition_path[i], sizeof(partition_path[i]),
        "/data/share/users/xyzhi/data/bigann/vamana/balance_kmeans_%d_part/"
        "merged_index_mem.index_tempFiles_meta_index_subshard-%d_ids_uint32."
        "bin",
        // "merged_index_mem.index_tempFiles_subshard-%d_ids_uint32.bin",
        part_num, i);
  }

  std::string path_data = "/data/share/users/xyzhi/data/bigann/base.100M.u8bin";

  char pos_label_path[1024];
  snprintf(
      pos_label_path, sizeof(pos_label_path),
      "/data/share/users/xyzhi/data/bigann/vamana/balance_kmeans_%d_part/"
      "100M_%d_poslabel_map.bin",
      part_num, part_num);
  // std::string pos_label_path =
  //     "/data/share/users/xyzhi/data/bigann/vamana/balance_kmeans_%d_part/"
  //     "100M_4_poslabel_map.bin";

  char label_pos_path[1024];
  snprintf(
      label_pos_path, sizeof(label_pos_path),
      "/data/share/users/xyzhi/data/bigann/vamana/balance_kmeans_%d_part/"
      "100M_%d_labelpos_map.bin",
      part_num, part_num);

  char output_path[1024];
  snprintf(
      output_path, sizeof(output_path),
      "/data/share/users/xyzhi/data/bigann/vamana/balance_kmeans_%d_part/"
      "kmeans_%d_100M.bin",
      part_num, part_num);

  // read kmeans partition id
  read_partition(
      partition_path, part, label_pos_map, pos_label_map, part_offset,
      part_size, million_num, vec_size, part_num);

  // reorganize index by kmeans ids.
  write_map(
      pos_label_map, label_pos_map, part_size, vec_size, part_num,
      pos_label_path, label_pos_path);

  kmeans_reorder(
      part, label_pos_map, pos_label_map, part_offset, part_size, million_num,
      vec_size, part_num, path_data.c_str(), output_path);

  int *tmp_part_size, *tmp_pos_label;
  int tmp_vec_size, tmp_part_num;
  get_pos_label_map(
      pos_label_path, tmp_vec_size, tmp_part_num, tmp_part_size, tmp_pos_label);

  verify_reorder_data(path_data.c_str(), output_path, tmp_pos_label, vec_size);

  return;
}

/**
 * Based on kmeans, transform to b1_graph (baseline1) format.
 */
void transfer_vamana_to_b1graph(int part_num, int million_num) {
  // TODO: concise to cmd
  char pos_label_path[1024];
  snprintf(
      pos_label_path, sizeof(pos_label_path),
      "/data/share/users/xyzhi/data/bigann/vamana/balance_kmeans_%d_part/"
      "100M_%d_poslabel_map.bin",
      part_num, part_num);

  char label_pos_path[1024];
  snprintf(
      label_pos_path, sizeof(label_pos_path),
      "/data/share/users/xyzhi/data/bigann/vamana/balance_kmeans_%d_part/"
      "100M_%d_labelpos_map.bin",
      part_num, part_num);
  // std::string label_pos_path =
  //     "/data/share/users/xyzhi/data/bigann/vamana/balance_kmeans/"
  //     "kmeans_label_pos_map.bin";
  // std::string pos_label_path =
  //     "/data/share/users/xyzhi/data/bigann/vamana/balance_kmeans/"
  //     "kmeans_pos_label_map.bin";

  // Reordered data based on kmeans.
  char kmeans_reorder_data_path[1024];
  snprintf(
      kmeans_reorder_data_path, sizeof(kmeans_reorder_data_path),
      "/data/share/users/xyzhi/data/bigann/vamana/balance_kmeans_%d_part/"
      "kmeans_%d_100M.bin",
      part_num, part_num);

  // Generated index by vamana.
  char vamana_index_path[1024];
  snprintf(
      vamana_index_path, sizeof(vamana_index_path),
      "/data/share/users/xyzhi/data/bigann/vamana/balance_kmeans_%d_part/"
      "merged_index_mem.index",
      part_num);
  // std::string index_path =
  //     "/data/share/users/xyzhi/data/bigann/"
  //     "vamana/balance_kmeans/merged_index_mem.index";

  // Target output: B1Index based on vamana and reordered data.
  char output_b1_path[1024];
  snprintf(
      output_b1_path, sizeof(output_b1_path),
      "/data/share/users/xyzhi/data/bigann/vamana/balance_kmeans_%d_part/"
      "kmeans_b1_mem.index",
      part_num);
  // std::string kmeans_index_path =
  //     "/data/share/users/xyzhi/data/bigann/"
  //     "vamana/balance_kmeans/kmeans_index_mem.index";

  uint64_t node_num = million_num * 1e6;

  std::ifstream data_input(kmeans_reorder_data_path, std::ios::binary);
  std::ofstream output(output_b1_path, std::ios::binary);
  std::ifstream index_input(vamana_index_path, std::ios::binary);

  uint64_t index_size = 0;
  index_input.read((char *)&index_size, sizeof(uint64_t));
  uint32_t max_degree = 0;
  index_input.read((char *)&max_degree, sizeof(uint32_t));
  uint32_t start;
  index_input.read((char *)&start, sizeof(uint32_t));
  uint64_t tmp_num = 0;  // this has bug, useless.
  index_input.read((char *)&tmp_num, sizeof(uint64_t));

  int *kmeans_part_size, *kmeans_label_pos_map, *kmeans_pos_label_map;
  int kmeans_vec_size, kmeans_part_num;
  get_label_pos_map(
      label_pos_path, kmeans_vec_size, kmeans_part_num, kmeans_part_size,
      kmeans_label_pos_map);
  get_pos_label_map(
      pos_label_path, kmeans_vec_size, kmeans_part_num, kmeans_part_size,
      kmeans_pos_label_map);
  assert(node_num == kmeans_vec_size);

  /*
    b1graph element size:
    deg + deg*ngh + vector + label
    4 + 48*4 + 128 + 8 = 332

    b1_graph format:
    <vec_num> <start>
    vec_num * <ele_size>
  */
  output.write((char *)&node_num, sizeof(node_num));
  output.write((char *)&start, sizeof(start));

  uint64_t b1_ele_size = 332;
  uint64_t kmeans_graph_size = b1_ele_size * node_num;
  printf("b1graph size %llu\n", kmeans_graph_size + 12);

  char *kmeans_graph = (char *)malloc(kmeans_graph_size);

  uint64_t neq_maxdeg_cnt = 0;
  uint64_t vec_data_len = 128;
  char *ngh_buf = (char *)malloc(max_degree * sizeof(uint32_t));
  for (uint64_t i = 0; i < node_num; i++) {
    uint32_t deg = 0;
    index_input.read((char *)&deg, sizeof(uint32_t));
    // printf("deg %llu\n", deg);
    index_input.read(ngh_buf, deg * sizeof(uint32_t));
    if (deg != max_degree) {
      neq_maxdeg_cnt++;
    }
    char *ele_buf =
        kmeans_graph + (uint64_t)kmeans_label_pos_map[i] * b1_ele_size;
    uint32_t *ele_buf_ptr = (uint32_t *)(ele_buf);
    uint32_t *ngh_buf_ptr = (uint32_t *)(ngh_buf);
    ele_buf_ptr[0] = deg;
    for (uint32_t d = 0; d < deg; d++) {
      ele_buf_ptr[d + 1] = kmeans_label_pos_map[ngh_buf_ptr[d]];
    }
    memcpy(
        ele_buf + sizeof(uint32_t) * (max_degree + 1) + vec_data_len, &i,
        sizeof(i));

    data_input.read(
        kmeans_graph + (uint64_t)i * b1_ele_size +
            sizeof(uint32_t) * (max_degree + 1),
        vec_data_len);
    if (i % (1000000) == 0) {
      printf("%.2f%%\n", (float)(i) * 100.0 / (node_num));
    }
  }
  index_input.close();
  data_input.close();

  printf("Write b1graph ...\n");
  uint64_t write_size = 1LL << 30;

  for (uint64_t i = 0; i < kmeans_graph_size; i += write_size) {
    if (i + write_size > kmeans_graph_size) {
      write_size = (uint64_t)(kmeans_graph_size - i);
    }
    output.write((char *)(kmeans_graph + i), write_size);
    printf("%.2f%%\n", (float)(i + write_size) * 100.0 / (kmeans_graph_size));
    sleep(5);
  }
  printf(
      "b1graph file write over, expect file size %llu\n",
      kmeans_graph_size + 12);
  output.close();
}



/*
 Transfer vamana data storage to uniform format.
*/
void transfer_kmeans(IndexParameter &index_param) {
  int vec_size = index_param.subset_size_milllions * 1e6;  // 10M/100M/1B
  uint32_t part_num = index_param.num_parts;
  uint32_t million_num = index_param.subset_size_milllions;
  int *part = (int *)malloc(vec_size * 2 * 4);
  memset(part, -1, vec_size * 2 * 4);
  /*
  part: <fisrt_part_id, second_part_id(if have)> [vec_size]
  part[2*x]: the first part id of vector x.

  label_pos_map[x]: the vector x(label) new position.
  pos_label_map[x]: the label vector in postition x has.
  */
  int *label_pos_map = (int *)malloc(vec_size * 4);
  memset(label_pos_map, -1, vec_size * 4);
  int *pos_label_map = (int *)malloc(vec_size * 4);
  memset(pos_label_map, -1, vec_size * 4);
  int *part_offset = (int *)malloc(vec_size * 4);
  memset(part_offset, -1, vec_size * 4);
  int *part_size = (int *)malloc(part_num * 4);
  memset(part_size, -1, part_num * 4);

  char partition_path[part_num][1024];
  for (int i = 0; i < part_num; i++) {
    snprintf(
        partition_path[i], sizeof(partition_path[i]),
        "%s", index_param.part_noreplica_idmap_filename[i].c_str());
  }

  std::string path_data = index_param.data_path;

  char pos_label_path[1024];
  snprintf(
      pos_label_path, sizeof(pos_label_path),
      "%s", index_param.temp_pos2label_map_file.c_str());

  char label_pos_path[1024];
  snprintf(
      label_pos_path, sizeof(label_pos_path),
      "%s", index_param.temp_label2pos_map_file.c_str());
  
  std::cout<<"p2l l2p file: "<< index_param.temp_pos2label_map_file.c_str() <<"\n"
  << index_param.temp_label2pos_map_file.c_str() <<"\n";

  char output_path[1024];
  snprintf(
      output_path, sizeof(output_path),
      "%s", index_param.temp_kmeans_data_file.c_str());

  // read kmeans partition id
  read_partition(
      partition_path, part, label_pos_map, pos_label_map, part_offset,
      part_size, million_num, vec_size, part_num);

  // reorganize index by kmeans ids.
  write_map(
      pos_label_map, label_pos_map, part_size, vec_size, part_num,
      pos_label_path, label_pos_path);

  kmeans_reorder(
      part, label_pos_map, pos_label_map, part_offset, part_size, million_num,
      vec_size, part_num, path_data.c_str(), output_path, index_param.dim, index_param.vec_size);

  int *tmp_part_size, *tmp_pos_label;
  int tmp_vec_size, tmp_part_num;
  get_pos_label_map(
      pos_label_path, tmp_vec_size, tmp_part_num, tmp_part_size, tmp_pos_label);

  verify_reorder_data(path_data.c_str(), output_path, tmp_pos_label, vec_size, index_param.vec_size);

  return;
}

/**
 * Based on kmeans, transform to b1_graph (baseline1) format.
 */
 void transfer_vamana_to_b1graph(IndexParameter &index_param) {
  uint32_t part_num = index_param.num_parts;
  uint32_t million_num = index_param.subset_size_milllions;
  char pos_label_path[1024];
  snprintf(
      pos_label_path, sizeof(pos_label_path),
      "%s", index_param.temp_pos2label_map_file.c_str());

  char label_pos_path[1024];
  snprintf(
      label_pos_path, sizeof(label_pos_path),
      "%s", index_param.temp_label2pos_map_file.c_str());

  // Reordered data based on kmeans.
  char kmeans_reorder_data_path[1024];
  snprintf(
      kmeans_reorder_data_path, sizeof(kmeans_reorder_data_path),
      "%s", index_param.temp_kmeans_data_file.c_str());

  // Generated index by vamana.
  char vamana_index_path[1024];
  snprintf(
      vamana_index_path, sizeof(vamana_index_path),
      "%s", index_param.mem_index_path.c_str());

  // Target output: B1Index based on vamana and reordered data.
  char output_b1_path[1024];
  snprintf(
      output_b1_path, sizeof(output_b1_path),
      "%s", index_param.temp_b1_index.c_str());

  uint64_t node_num = million_num * 1e6;

  std::ifstream data_input(kmeans_reorder_data_path, std::ios::binary);
  std::ofstream output(output_b1_path, std::ios::binary);
  std::ifstream index_input(vamana_index_path, std::ios::binary);

  uint64_t index_size = 0;
  index_input.read((char *)&index_size, sizeof(uint64_t));
  uint32_t max_degree = 0;
  index_input.read((char *)&max_degree, sizeof(uint32_t));
  uint32_t start;
  index_input.read((char *)&start, sizeof(uint32_t));
  uint64_t tmp_num = 0;  // this has bug, useless.
  index_input.read((char *)&tmp_num, sizeof(uint64_t));

  int *kmeans_part_size, *kmeans_label_pos_map, *kmeans_pos_label_map;
  int kmeans_vec_size, kmeans_part_num;
  get_label_pos_map(
      label_pos_path, kmeans_vec_size, kmeans_part_num, kmeans_part_size,
      kmeans_label_pos_map);
  get_pos_label_map(
      pos_label_path, kmeans_vec_size, kmeans_part_num, kmeans_part_size,
      kmeans_pos_label_map);
  assert(node_num == kmeans_vec_size);

  /*
    b1graph element size:
    deg + deg*ngh + vector + label
    4 + 48*4 + 128 + 8 = 332

    b1_graph format:
    <vec_num> <start>
    vec_num * <ele_size>
  */
  output.write((char *)&node_num, sizeof(node_num));
  output.write((char *)&start, sizeof(start));

  uint64_t b1_ele_size = index_param.b1_ele_size;
  uint64_t kmeans_graph_size = b1_ele_size * node_num;
  printf("b1graph size %llu\n", kmeans_graph_size + 12);

  char *kmeans_graph = (char *)malloc(kmeans_graph_size);

  uint64_t neq_maxdeg_cnt = 0;
  uint64_t vec_data_len = index_param.vec_size;

  if(index_param.max_degree != max_degree){
    printf("Error: max degree mismatch ..\n");
    printf("param: %u read %u\n", index_param.max_degree, max_degree);
    abort();
  }
  char *ngh_buf = (char *)malloc(max_degree * sizeof(uint32_t));
  for (uint64_t i = 0; i < node_num; i++) {
    uint32_t deg = 0;
    index_input.read((char *)&deg, sizeof(uint32_t));
    // printf("deg %llu\n", deg);
    index_input.read(ngh_buf, deg * sizeof(uint32_t));
    if (deg != max_degree) {
      neq_maxdeg_cnt++;
    }
    char *ele_buf =
        kmeans_graph + (uint64_t)kmeans_label_pos_map[i] * b1_ele_size;
    uint32_t *ele_buf_ptr = (uint32_t *)(ele_buf);
    uint32_t *ngh_buf_ptr = (uint32_t *)(ngh_buf);
    ele_buf_ptr[0] = deg;
    for (uint32_t d = 0; d < deg; d++) {
      ele_buf_ptr[d + 1] = kmeans_label_pos_map[ngh_buf_ptr[d]];
    }
    memcpy(
        ele_buf + sizeof(uint32_t) * (max_degree + 1) + vec_data_len, &i,
        sizeof(i));

    data_input.read(
        kmeans_graph + (uint64_t)i * b1_ele_size +
            sizeof(uint32_t) * (max_degree + 1),
        vec_data_len);
    if (i % (1000000) == 0) {
      printf("transfer %.2f%%\r", (float)(i) * 100.0 / (node_num));
    }
  }
  printf("\n");
  index_input.close();
  data_input.close();

  printf("Write b1graph ...\n");
  uint64_t write_size = 1LL << 30;

  for (uint64_t i = 0; i < kmeans_graph_size; i += write_size) {
    if (i + write_size > kmeans_graph_size) {
      write_size = (uint64_t)(kmeans_graph_size - i);
    }
    output.write((char *)(kmeans_graph + i), write_size);
    printf("%.2f%%\r", (float)(i + write_size) * 100.0 / (kmeans_graph_size));
    sleep(4);
  }
  printf(
      "\nb1graph file write over, expect file size %llu\n",
      kmeans_graph_size + 12);
  output.close();
}


/**
 * Based on dist-vamana index, transform to b2_graph (baseline2) format.
 */
void transfer_vamana_to_b2graph(int part_num, int million_num) {
  // origin random data.
  char kmeans_reorder_data_path[1024];
  snprintf(
      kmeans_reorder_data_path, sizeof(kmeans_reorder_data_path),
      "/data/share/users/xyzhi/data/bigann/"
      "base.%dM.u8bin",
      million_num);

  // Generated index by vamana.
  char vamana_index_path[1024];
  snprintf(
      vamana_index_path, sizeof(vamana_index_path),
      "/data/share/users/xyzhi/data/bigann/vamana/b2_d48_%d_part/"
      "merged_index_mem.index",
      part_num);
  // snprintf(
  //     vamana_index_path, sizeof(vamana_index_path),
  //     "/data/share/users/xyzhi/data/bigann/vamana/%dM_%d_part/"
  //     "merged_index_mem.index",
  //     million_num, part_num);

  // Target output: B1Index based on vamana and reordered data.
  char output_b2_path[part_num][1024];
  for (uint32_t p = 0; p < part_num; p++) {
    snprintf(
        output_b2_path[p], sizeof(output_b2_path[p]),
        "/data/share/users/xyzhi/data/bigann/vamana/b2_d48_%d_part/"
        "%dM_index_%d-%d.bin",
        part_num, million_num, p, part_num);
  }

  // std::string kmeans_index_path =
  //     "/data/share/users/xyzhi/data/bigann/"
  //     "vamana/balance_kmeans/kmeans_index_mem.index";

  uint64_t node_num = million_num * 1e6;

  // TODO: all base data and kmeans data should be the same format.
  std::ifstream data_input(kmeans_reorder_data_path, std::ios::binary);
  uint32_t data_size = 0;
  data_input.read((char *)&data_size, sizeof(uint32_t));
  uint32_t data_dim = 0;
  data_input.read((char *)&data_dim, sizeof(uint32_t));
  std::cout << "base data_size " << data_size << " base data_dim " << data_dim
            << "\n";

  std::ifstream index_input(vamana_index_path, std::ios::binary);
  uint64_t index_size = 0;
  index_input.read((char *)&index_size, sizeof(uint64_t));
  uint32_t max_degree = 0;
  index_input.read((char *)&max_degree, sizeof(uint32_t));
  uint32_t start;
  index_input.read((char *)&start, sizeof(uint32_t));
  uint64_t tmp_num = 0;  // this has bug, useless.
  index_input.read((char *)&tmp_num, sizeof(uint64_t));

  /*
    b2graph element size:
    deg + deg*ngh + vector + label
    4 + 48*4 + 128 + 8 = 332

    b1_graph format:
    <vec_num> <start>
    vec_num * <ele_size>
  */
  uint64_t part_sz = node_num / part_num;
  for (uint32_t p = 0; p < part_num; p++) {
    std::ofstream output(output_b2_path[p], std::ios::binary);
    output.write((char *)&part_sz, sizeof(part_sz));
    output.write((char *)&start, sizeof(start));

    uint64_t b2_ele_size = 332;
    uint64_t kmeans_graph_size = b2_ele_size * part_sz;
    printf("b2graph size %llu\n", kmeans_graph_size + 12);

    char *kmeans_graph = (char *)malloc(kmeans_graph_size);

    uint64_t neq_maxdeg_cnt = 0;
    uint64_t vec_data_len = 128;
    char *ngh_buf = (char *)malloc(max_degree * sizeof(uint32_t));

    uint64_t L = part_sz * p;
    uint64_t R = part_sz * (p + 1);
    for (uint64_t i = L; i < R; i++) {
      uint32_t deg = 0;
      index_input.read((char *)&deg, sizeof(uint32_t));
      // printf("deg %llu\n", deg);
      index_input.read(ngh_buf, deg * sizeof(uint32_t));
      if (deg != max_degree) {
        neq_maxdeg_cnt++;
      }
      char *ele_buf = kmeans_graph + (uint64_t)(i - L) * b2_ele_size;
      uint32_t *ele_buf_ptr = (uint32_t *)(ele_buf);
      uint32_t *ngh_buf_ptr = (uint32_t *)(ngh_buf);
      ele_buf_ptr[0] = deg;
      for (uint32_t d = 0; d < deg; d++) {
        ele_buf_ptr[d + 1] = ngh_buf_ptr[d] - L;
      }

      // copy label
      memcpy(
          ele_buf + sizeof(uint32_t) * (max_degree + 1) + vec_data_len, &i,
          sizeof(i));

      data_input.read(
          kmeans_graph + (uint64_t)(i - L) * b2_ele_size +
              sizeof(uint32_t) * (max_degree + 1),
          vec_data_len);
      if (i % (1000000) == 0) {
        printf("%.2f%%\n", (float)(i - L) * 100.0 / (part_sz));
      }
    }

    printf("Write b2graph part %u...\n", p);
    uint64_t write_size = 1LL << 30;

    for (uint64_t i = 0; i < kmeans_graph_size; i += write_size) {
      if (i + write_size > kmeans_graph_size) {
        write_size = (uint64_t)(kmeans_graph_size - i);
      }
      output.write((char *)(kmeans_graph + i), write_size);
      printf("%.2f%%\n", (float)(i + write_size) * 100.0 / (kmeans_graph_size));
      sleep(5);
    }
    printf(
        "b2graph part %u write over, expect file size %llu\n", p,
        kmeans_graph_size + 12);
    output.close();
  }
  index_input.close();
  data_input.close();
}

/**
 * Profiling the neighbor.
 */
void test_vamana_index() {
  // std::string label_pos_path =
  //     "/data/share/users/xyzhi/data/bigann/vamana/kmeans/"
  //     "100M_4_labelpos_map.bin";
  // std::string label_pos_path =
  //     "/data/share/users/xyzhi/data/bigann/scalagraph/balance_v0/"
  //     "balance_v0_100M_4_labelpos_map.bin";
  std::string label_pos_path =
      "/data/share/users/xyzhi/data/bigann/vamana/balance_kmeans/"
      "kmeans_label_pos_map.bin";

  int *kmeans_part_size, *kmeans_label_pos_map, *kmeans_pos_label_map;
  int kmeans_vec_size, kmeans_part_num;
  get_label_pos_map(
      label_pos_path.c_str(), kmeans_vec_size, kmeans_part_num,
      kmeans_part_size, kmeans_label_pos_map);
  std::cout << "kmeans_part_num " << kmeans_part_num << "\n";
  auto judge_partition = [&](uint32_t vid) {
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
  snprintf(
      default_sift100M_kmeans_path, sizeof(default_sift100M_kmeans_path),
      "/data/share/users/xyzhi/data/bigann/vamana/balance_kmeans/"
      "kmeans_index_mem.index");

  printf("loading index from %s ...\n", default_sift100M_kmeans_path);
  std::ifstream input(default_sift100M_kmeans_path, std::ios::binary);

  if (!input.is_open()) throw std::runtime_error("Cannot open file");

  size_t max_elements_;
  uint32_t enterpoint_node_;
  input.read((char *)&max_elements_, 8);
  input.read((char *)&enterpoint_node_, 4);
  printf("max ele size: %llu\n", max_elements_);

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

  // size_links_per_element_ = maxM_ * sizeof(tableint) +
  // sizeof(linklistsizeint);
  printf("enter node %llu\n", enterpoint_node_);

  // profiling ngh distribution.
  int ratio[11];
  for (int i = 0; i < 11; i++) ratio[i] = 0;
  for (uint32_t vid = 0; vid < cur_element_count; vid++) {
    int *ptr = (int *)(data_level0_memory_ + vid * size_data_per_element_);
    int cnt[kmeans_part_num];
    for (int i = 0; i < kmeans_part_num; i++) cnt[i] = 0;
    for (int i = 1; i < ptr[0] + 1; i++) {
      cnt[judge_partition(ptr[i])]++;
    }
    int mx = 0;
    for (int i = 0; i < kmeans_part_num; i++) mx = std::max(mx, cnt[i]);
    ratio[mx * 10 / ptr[0]]++;
  }

  for (int i = 0; i < 11; i++) {
    printf(
        "%d0-%d0%%: %d %.2f\n", i, i + 1, ratio[i],
        (float)ratio[i] * 100.0 / cur_element_count);
  }

  // size_t q;
  // while (std::cin >> q) {
  //   std::cout << "try get " << q << " ngh\n";
  //   int *ptr = (int *)(data_level0_memory_ + q * size_data_per_element_);
  //   std::cout << "deg=" << ptr[0] << "\nngh: ";
  //   for (int i = 1; i < ptr[0] + 1; i++) {
  //     std::cout << ptr[i] << " ";
  //   }
  //   std::cout << "\n";
  //   for (int i = 1; i < ptr[0] + 1; i++) {
  //     std::cout << judge_partition(ptr[i]) << " ";
  //   }
  //   std::cout << "\n";
  // }
}
