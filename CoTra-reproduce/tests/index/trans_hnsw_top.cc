#include "anns/exec_query.h"
#include "index/graph_index.h"

void trans_hnsw_top(uint32_t part_num) {
  int *kmeans_label_pos_map, *kmeans_pos_label_map;

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

  RandomVectorIndex<int> hnsw(
      std::string("/data/share/users/xyzhi/data/bigann/1B_100M_index.bin"));
  hnsw.loadIndex();

  auto getDataByInternalId = [&](tableint internal_id) {
    return (
        hnsw.data_level0_memory_ + internal_id * hnsw.size_data_per_element_ +
        hnsw.offsetData_);
  };

  auto get_linklist = [&](tableint internal_id, int level) {
    return (linklistsizeint *)(hnsw.linkLists_[internal_id] +
                               (level - 1) * hnsw.size_links_per_element_);
  };

  auto getListCount = [&](linklistsizeint *ptr) {
    return *((unsigned short int *)ptr);
  };

  auto getExternalLabel = [&](tableint internal_id) {
    labeltype return_label;
    memcpy(
        &return_label,
        (hnsw.data_level0_memory_ + internal_id * hnsw.size_data_per_element_ +
         hnsw.label_offset_),
        sizeof(labeltype));
    return return_label;
  };

  tableint currObj = hnsw.enterpoint_node_;
  //   dist_t curdist = fstdistfunc_(
  //       query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

  std::vector<tableint> cand_set[hnsw.maxlevel_ + 1];
  cand_set[hnsw.maxlevel_].push_back(currObj);
  //   std::vector<tableint> nxt_cand_set;
  for (int level = hnsw.maxlevel_; level > 0; level--) {
    for (auto cand : cand_set[level]) {
      unsigned int *data;
      data = (unsigned int *)get_linklist(cand, level);
      int size = getListCount(data);
      tableint *datal = (tableint *)(data + 1);
      for (int i = 0; i < size; i++) {
        tableint cand = datal[i];
        // printf("cand %u\n", cand);
        if (cand < 0 || cand > hnsw.max_elements_)
          throw std::runtime_error("cand error");
        char *vec_ptr = getDataByInternalId(cand);

        auto label = getExternalLabel(cand);
        // printf("label %u\n", label);
        auto pos = kmeans_label_pos_map[label];
        // printf("pos %u\n", pos);
        char *vamana_data = data_level0_memory_ + pos * size_data_per_element_;
        if (memcmp(vec_ptr, vamana_data + offsetData_, 128)) {
          printf("Not Match: cand %d, label %d, pos %d\n", cand, label, pos);
        }
        cand_set[level - 1].push_back(cand);
      }
    }
  }

  printf("Verify over\n");
  uint32_t total = 0;
  for (int i = 0; i <= hnsw.maxlevel_; i++) {
    printf("level %d: %lu\n", i, cand_set[i].size());
    total += cand_set[i].size();
  }
  printf("total num: %u\n", total);
  uint32_t level_num = hnsw.maxlevel_ + 1;
  uint32_t meta_len = 4 + 4 + level_num * 4;
  uint32_t index_size = meta_len + (total * size_data_per_element_);
  uint64_t MB_sz = index_size >> 20;
  printf("total %lu MB\n", MB_sz);

  char *index_ptr = (char *)malloc(index_size);
  size_t ofs = 0;
  memcpy(index_ptr, &total, 4);
  ofs += 4;
  memcpy(index_ptr + ofs, &level_num, 4);
  ofs += 4;
  for (int l = hnsw.maxlevel_; l >= 0; l--) {
    uint32_t sz = cand_set[l].size();
    memcpy(index_ptr + ofs, &sz, 4);
    ofs += 4;
  }
  uint32_t cand_cnt = 0;
  uint32_t ngh_cnt = 1;
  for (int l = hnsw.maxlevel_; l > 0; l--) {
    for (auto cand : cand_set[l]) {
      unsigned int *data;
      data = (unsigned int *)get_linklist(cand, l);
      int size = getListCount(data);
      char *ptr = index_ptr + meta_len + cand_cnt * size_data_per_element_;
      memcpy(ptr, &size, 4);
      tableint *datal = (tableint *)(ptr + 4);
      for (int i = 0; i < size; i++) {
        datal[i] = ngh_cnt++;
      }
      memcpy(ptr + offsetData_, getDataByInternalId(cand), 128);
      auto label = kmeans_label_pos_map[getExternalLabel(cand)];
      memcpy(ptr + label_offset_, &label, 8);
      cand_cnt++;
    }
  }
  for (auto cand : cand_set[0]) {
    char *ptr = index_ptr + meta_len + cand_cnt * size_data_per_element_;
    memcpy(ptr + offsetData_, getDataByInternalId(cand), 128);
    auto label = kmeans_label_pos_map[getExternalLabel(cand)];
    memcpy(ptr + label_offset_, &label, 8);
    cand_cnt++;
  }

  char output_path[1024];
  snprintf(
      output_path, sizeof(output_path),
      "/data/share/users/xyzhi/data/bigann/vamana/balance_kmeans_%u_part/"
      "hnsw_top_index_mem.index",
      part_num);

  // write hnsw top index
  std::ofstream output(output_path, std::ios::binary);
  uint32_t write_sz = 1 << 28;
  for (uint64_t i = 0; i < index_size; i += write_sz) {
    if (i + write_sz > index_size) {
      write_sz = (uint64_t)(index_size - i);
    }
    output.write((char *)(index_ptr + i), write_sz);
    printf("%.2f%%\n", (float)(i + write_sz) * 100.0 / index_size);
  }
  output.close();
  printf(
      "hnsw top index for %u part write over to %s, expect file size %u\n",
      part_num, output_path, index_size);
}
// ./tests/index/trans_hnsw
int main() {
  // trans_hnsw_top(4);
  trans_hnsw_top(8);
  return 0;
}