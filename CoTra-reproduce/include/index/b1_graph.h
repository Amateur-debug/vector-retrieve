#ifndef INDEX_H
#define INDEX_H

#include <atomic>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "kmeans.h"
#include "rdma/rdma_param.h"

/**
 * B1Graph (Baseline1 graph index format):
 * element =
 * [<linklist: neighbor_num * uint64_t>, <vector: dim * data_type>,
 * <label: uint64_t>]
 *
 */
class B1GraphInterface {
 public:
  // general param
  std::string location;
  size_t max_elements_{0};
  size_t cur_element_count{0};
  size_t size_data_per_element_{0};
  size_t size_links_per_element_{0};
  size_t M_{0};
  size_t maxM_{0};
  size_t maxM0_{0};
  size_t ef_construction_{0};
  size_t ef_{0};

  double mult_{0.0};
  int maxlevel_{0};

  typedef unsigned int tableint;
  typedef unsigned int linklistsizeint;

  tableint enterpoint_node_{0};

  size_t size_links_level0_{0};
  size_t offsetData_{0}, offsetLevel0_{0}, label_offset_{0};

  char *data_level0_memory_{nullptr};

  char *partition_data{nullptr};

  char **linkLists_{nullptr};
  std::vector<int> element_levels_;  // keeps level of each element

  size_t data_size_{0};

  virtual void print() = 0;
  virtual std::string get_location() = 0;

  virtual void loadIndex() = 0;

  virtual uint32_t get_vector_machine(size_t vector_id) = 0;
  virtual size_t get_internal_id(size_t vector_id) = 0;
  virtual void init_ptr(
      uint32_t machine_id_, uint32_t machine_num_, size_t max_elements,
      char *data_level0_memory, size_t size_data_per_element) = 0;
  virtual char *get_partition_ptr() = 0;

  virtual size_t get_partition_size() = 0;

  virtual uint32_t get_machine_id(size_t node_id) = 0;
};

/*
    Used to manage graph index data.

*/
template <typename DistType>
class RandomVectorIndex : public B1GraphInterface {
 public:
  // rdma param
  std::string location;
  char **partition_ptr;
  size_t *partition_vec_num;
  size_t *partition_size;

  size_t avg_size;
  uint32_t machine_id;
  uint32_t machine_num;

  RandomVectorIndex() {}

  RandomVectorIndex(const std::string &_location) : location(_location) {
    std::cout << location << std::endl;
  }

  void print() override {
    std::cout << "random index: " << location << std::endl;
  }

  std::string get_location() override { return location; }

  // void build_index() {}

  void loadIndex() {
    printf("loading index from %s ...\n", location.c_str());
    std::ifstream input(location, std::ios::binary);

    if (!input.is_open()) throw std::runtime_error("Cannot open file");

    free(data_level0_memory_);
    data_level0_memory_ = nullptr;
    for (tableint i = 0; i < cur_element_count; i++) {
      if (element_levels_[i] > 0) free(linkLists_[i]);
    }
    free(linkLists_);
    linkLists_ = nullptr;
    cur_element_count = 0;
    // get file size:
    input.seekg(0, input.end);
    std::streampos total_filesize = input.tellg();
    input.seekg(0, input.beg);

    readBinaryPOD(input, offsetLevel0_);
    readBinaryPOD(input, max_elements_);
    readBinaryPOD(input, cur_element_count);

    size_t max_elements = 0;
    if (max_elements < cur_element_count) max_elements = max_elements_;
    max_elements_ = max_elements;
    readBinaryPOD(input, size_data_per_element_);
    readBinaryPOD(input, label_offset_);
    readBinaryPOD(input, offsetData_);
    readBinaryPOD(input, maxlevel_);
    readBinaryPOD(input, enterpoint_node_);

    readBinaryPOD(input, maxM_);
    readBinaryPOD(input, maxM0_);
    readBinaryPOD(input, M_);
    readBinaryPOD(input, mult_);
    readBinaryPOD(input, ef_construction_);

    auto pos = input.tellg();

    uint64_t index_size = 0;
    /// Optional - check if index is ok:
    input.seekg(cur_element_count * size_data_per_element_, input.cur);
    for (size_t i = 0; i < cur_element_count; i++) {
      if (input.tellg() < 0 || input.tellg() >= total_filesize) {
        throw std::runtime_error("Index seems to be corrupted or unsupported");
      }

      unsigned int linkListSize;
      readBinaryPOD(input, linkListSize);
      if (linkListSize != 0) {
        input.seekg(linkListSize, input.cur);
      }
      index_size += sizeof(linklistsizeint);
      index_size += linkListSize;
    }

    // throw exception if it either corrupted or old index
    if (input.tellg() != total_filesize)
      throw std::runtime_error("Index seems to be corrupted or unsupported");

    input.clear();
    /// Optional check end

    input.seekg(pos, input.beg);

    data_level0_memory_ = (char *)malloc(max_elements * size_data_per_element_);
    if (data_level0_memory_ == nullptr)
      throw std::runtime_error(
          "Not enough memory: loadIndex failed to allocate level0");
    input.read(data_level0_memory_, cur_element_count * size_data_per_element_);

    size_links_per_element_ =
        maxM_ * sizeof(tableint) + sizeof(linklistsizeint);

    size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);

    linkLists_ = (char **)malloc(sizeof(void *) * max_elements);
    if (linkLists_ == nullptr)
      throw std::runtime_error(
          "Not enough memory: loadIndex failed to allocate linklists");
    element_levels_ = std::vector<int>(max_elements);
    ef_ = 10;
    for (size_t i = 0; i < cur_element_count; i++) {
      unsigned int linkListSize;
      readBinaryPOD(input, linkListSize);
      if (linkListSize == 0) {
        element_levels_[i] = 0;
        linkLists_[i] = nullptr;
      } else {
        element_levels_[i] = linkListSize / size_links_per_element_;
        linkLists_[i] = (char *)malloc(linkListSize);
        if (linkLists_[i] == nullptr)
          throw std::runtime_error(
              "Not enough memory: loadIndex failed to allocate linklist");
        input.read(linkLists_[i], linkListSize);
      }
    }

    input.close();

    printf("Show parameters:\n");
    printf("size of disk file: %llu\n", total_filesize);
    printf("cur_element_count: %llu\n", cur_element_count);
    printf("max_elements_: %llu\n", max_elements_);
    printf("enterpoint_node_: %llu\n", enterpoint_node_);
    printf("size_links_level0_: %llu\n", size_links_level0_);
    printf("offsetData_: %llu\n", offsetData_);
    printf("offsetLevel0_: %llu\n", offsetLevel0_);
    printf("label_offset_: %llu\n", label_offset_);
    printf("data_size_: %llu\n", data_size_);
    printf("size_data_per_element_: %llu\n", size_data_per_element_);
    printf("size_links_per_element_:%llu\n", size_links_per_element_);
    printf("size of disk file: %llu\n", total_filesize);
    printf(
        "size of data: %llu\n",
        (uint64_t)cur_element_count * size_data_per_element_);
    printf(
        "size of data_level0_memory_: %llu expect equal to above\n",
        (uint64_t)max_elements_ * size_data_per_element_);
    printf("size of top index: %llu\n", index_size);
    printf(
        "ratio of top index: %.2f%\n",
        (float)index_size * 100.0 /
            (index_size + max_elements_ * size_data_per_element_));

    return;
  }

  uint32_t get_vector_machine(size_t vector_id) override {
    uint32_t vector_machine = vector_id / avg_size;
    return vector_machine;
  }

  size_t get_internal_id(size_t vector_id) override {
    return vector_id - get_vector_machine(vector_id) * avg_size;
  }

  void init_ptr(
      uint32_t machine_id_, uint32_t machine_num_, size_t max_elements,
      char *data_level0_memory, size_t size_data_per_element) override {
    machine_id = machine_id_;
    machine_num = machine_num_;
    printf(
        "machine_id %d machinenum %d max_elements %d\n", machine_id,
        machine_num, max_elements);
    partition_ptr = (char **)malloc(sizeof(void *) * MACHINE_NUM);
    partition_vec_num = (size_t *)malloc(sizeof(size_t) * MACHINE_NUM);
    partition_size = (size_t *)malloc(sizeof(size_t) * MACHINE_NUM);

    // different partition ratio
    avg_size = max_elements / MACHINE_NUM;
    printf("Partition vector num:\n");
    size_t cumu_num = 0;
    for (int i = 0; i < MACHINE_NUM - 1; i++) {
      partition_vec_num[i] = avg_size;
      partition_size[i] = partition_vec_num[i] * size_data_per_element;
      printf("%llu ", partition_vec_num[i]);
      partition_ptr[i] =
          (data_level0_memory + cumu_num * size_data_per_element);
      cumu_num += partition_vec_num[i];
    }
    partition_vec_num[MACHINE_NUM - 1] = (max_elements - cumu_num);
    partition_size[MACHINE_NUM - 1] =
        partition_vec_num[MACHINE_NUM - 1] * size_data_per_element;
    partition_ptr[MACHINE_NUM - 1] =
        (data_level0_memory + cumu_num * size_data_per_element);
    printf("%llu\n", (max_elements - cumu_num));

    // assert(cumu_num > partition_size - MACHINE_NUM);

    printf("Partition size:\n");
    for (int i = 0; i < MACHINE_NUM; i++) {
      printf("%llu ", partition_size[i]);
    }
    printf("\n");

    return;
  }
  char *get_partition_ptr() override {
    return partition_ptr[machine_id];
    // return partition_data;
  }

  size_t get_partition_size() override { return partition_size[machine_id]; }

  uint32_t get_machine_id(size_t node_id) {return 0;}; // not use.
};

template <typename DistType>
class KmeansVectorIndex : public B1GraphInterface {
 public:
  // // general param
  std::string location;
  std::string map_location;
  // rdma param
  char **partition_ptr;
  size_t *partition_vec_num;
  size_t *partition_size;

  uint32_t machine_id;
  int machine_num;
  int kmeans_pnum;

  KmeansVectorIndex() {}
  KmeansVectorIndex(
      const std::string &_location, const std::string &_map_location,
      int _kmeans_pnum)
      : location(_location),
        map_location(_map_location),
        kmeans_pnum(_kmeans_pnum) {}

  void print() override {
    std::cout << "kmeans index: " << location << std::endl;
    std::cout << "kmeans map: " << map_location << std::endl;
  }

  void loadIndex() {}

  std::string get_location() override { return location; }

  uint32_t get_vector_machine(size_t vector_id) override {
    size_t cumunum = 0;
    for (int p = 0; p < kmeans_pnum; p++) {
      cumunum += partition_vec_num[p];
      if (cumunum > vector_id) return p;
    }
    assert(0);  // error
    return 0;
  }

  size_t get_internal_id(size_t vector_id) override {
    size_t cumunum = 0;
    for (int p = 0; p < kmeans_pnum; p++) {
      if (cumunum + partition_vec_num[p] > vector_id) {
        return vector_id - cumunum;
      } else {
        cumunum += partition_vec_num[p];
      }
    }
    assert(0);  // error
    return 0;
  }

  void init_ptr(
      uint32_t machine_id_, uint32_t machine_num_, size_t max_elements,
      char *data_level0_memory, size_t size_data_per_element) override {
    machine_id = machine_id_;
    machine_num = machine_num_;
    printf(
        "machine_id %d machinenum %d max_elements %d\n", machine_id,
        machine_num, max_elements);
    partition_ptr = (char **)malloc(sizeof(void *) * MACHINE_NUM);
    partition_vec_num = (size_t *)malloc(sizeof(size_t) * MACHINE_NUM);
    partition_size = (size_t *)malloc(sizeof(size_t) * MACHINE_NUM);

    int *kmeans_part_size, *kmeans_pos_label_map;
    int kmeans_vec_size, kmeans_part_num;
    get_pos_label_map(
        map_location.c_str(), kmeans_vec_size, kmeans_part_num,
        kmeans_part_size, kmeans_pos_label_map);
    // assert(kmeans_pnum == kmeans_part_num);

    /*
      For now, we only use the kpart0,1 as one machine, and 2,3 as another.
    */
    printf("Partition vector num:\n");
    size_t cumu_size = 0;
    for (int i = 0; i < MACHINE_NUM; i++) {
      // partition_vec_num[i] =
      // (kmeans_part_size[2 * i] + kmeans_part_size[2 * i + 1]);
      partition_vec_num[i] = kmeans_part_size[i];
      partition_size[i] = partition_vec_num[i] * size_data_per_element;
      printf("%llu ", partition_vec_num[i]);
      partition_ptr[i] = (data_level0_memory + cumu_size);
      cumu_size += partition_size[i];
    }
    assert(cumu_size == kmeans_vec_size);

    printf("\nPartition size:\n");
    for (int i = 0; i < MACHINE_NUM; i++) {
      printf("%llu ", partition_size[i]);
    }
    printf("\n");

    return;
  }

  char *get_partition_ptr() override {
    return partition_ptr[machine_id];
    // return partition_data;
  }

  size_t get_partition_size() override { return partition_size[machine_id]; }

  uint32_t get_machine_id(size_t node_id) {return 0;}; // not use.
};

// TODO: DistType useless
template <typename DistType>
class VamanaIndex : public B1GraphInterface {
 public:

  IndexParameter &index_param;
  // // general param
  std::string location;
  // std::string map_location;
  // rdma param
  char **partition_ptr;
  size_t *partition_vec_num;
  size_t *partition_size;
  uint32_t *cumu_partition_num;

  uint32_t machine_id;
  int machine_num;
  int kmeans_pnum;

  std::string kmeans_map;

  // VamanaIndex() {}

  VamanaIndex(IndexParameter &_index_param): index_param(_index_param) {
    if(index_param.graph_type == VAMANA){
      location = index_param.temp_b1_index;
    }
    else{
      location = index_param.local_b2_index_file;
    }
    
    kmeans_pnum = index_param.num_parts;
    machine_id = index_param.machine_id;
    size_t vector_size = index_param.vec_size;
    size_t deg = index_param.max_degree;
    kmeans_map = index_param.temp_pos2label_map_file;
    std::cout<<"Index param vector_size: "<<vector_size <<" max_deg: "<< deg <<"\n";
    offsetLevel0_ = 0;
    size_data_per_element_ = (deg+1)*sizeof(uint32_t) + vector_size +sizeof(uint64_t);
    label_offset_ = (deg+1)*sizeof(uint32_t) + vector_size;
    offsetData_ = (deg+1)*sizeof(uint32_t);
    maxlevel_ = 0;
    maxM_ = 0;  // useless
    maxM0_ = deg;
    M_ = deg;               // useless
    ef_construction_ = 0;  // useless

    cumu_partition_num =
        (uint32_t *)malloc(sizeof(uint32_t) * (MACHINE_NUM + 1));
  }

  void loadIndex() {
    // for kmeans index
    if(index_param.graph_type == KmeansSharedNothing){
      int *kmeans_part_size;
      int kmeans_vec_size, kmeans_part_num;
      int *kmeans_pos_label_map;
      get_pos_label_map(
          index_param.temp_pos2label_map_file.c_str(), kmeans_vec_size,
          kmeans_part_num, kmeans_part_size, kmeans_pos_label_map);
      std::cout<< "Partition size: \n";
      for(uint32_t m = 0; m < MACHINE_NUM; m++){
        index_param.partition_size[m] = kmeans_part_size[m];
        std::cout << index_param.partition_size[m] <<" ";
      }
      std::cout << "\n";
      cumu_partition_num[0] = 0;

      std::cout<< "cumu Partition size: \n";
      for (uint32_t m = 1; m <= MACHINE_NUM; m++) {
        cumu_partition_num[m] = cumu_partition_num[m - 1] + index_param.partition_size[m - 1];
        std::cout << cumu_partition_num[m] <<" ";
      }
      std::cout<< "\n";
    }

    printf("loading index from %s ...\n", location.c_str());
    std::ifstream input(location, std::ios::binary);

    if (!input.is_open()) throw std::runtime_error("Cannot open file");

    free(data_level0_memory_);
    data_level0_memory_ = nullptr;
    for (tableint i = 0; i < cur_element_count; i++) {
      if (element_levels_[i] > 0) free(linkLists_[i]);
    }
    free(linkLists_);
    linkLists_ = nullptr;

    readBinaryPOD(input, max_elements_);
    readBinaryPOD(input, enterpoint_node_);

    // TODO: WARN: just for now, no enternode.
    // enterpoint_node_ = 0;
    cur_element_count = max_elements_;
    std::cout << " enterpoint_node: " << enterpoint_node_
              << " cur_element_count: " << cur_element_count
              << " max_elements_: " << max_elements_ << std::endl;

    data_level0_memory_ =
        (char *)malloc((size_t)max_elements_ * size_data_per_element_);
    if (data_level0_memory_ == nullptr)
      throw std::runtime_error(
          "Not enough memory: loadIndex failed to allocate level0");
    input.read(data_level0_memory_, cur_element_count * size_data_per_element_);

    size_links_per_element_ =
        maxM_ * sizeof(tableint) + sizeof(linklistsizeint);

    size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);

    element_levels_ = std::vector<int>(max_elements_);
    linkLists_ = (char **)malloc(sizeof(void *) * max_elements_);
    if (linkLists_ == nullptr)
      throw std::runtime_error(
          "Not enough memory: loadIndex failed to allocate linklists");
    ef_ = 10;

    input.close();

    printf("Show parameters:\n");
    printf("cur_element_count: %llu\n", cur_element_count);
    printf("max_elements_: %llu\n", max_elements_);
    printf("enterpoint_node_: %llu\n", enterpoint_node_);
    printf("size_links_level0_: %llu\n", size_links_level0_);
    printf("offsetData_: %llu\n", offsetData_);
    printf("offsetLevel0_: %llu\n", offsetLevel0_);
    printf("label_offset_: %llu\n", label_offset_);
    printf("data_size_: %llu\n", data_size_);
    printf("size_data_per_element_: %llu\n", size_data_per_element_);
    printf("size_links_per_element_:%llu\n", size_links_per_element_);
    printf(
        "size of data: %llu\n",
        (uint64_t)cur_element_count * size_data_per_element_);
    printf(
        "size of data_level0_memory_: %llu expect equal to above\n",
        (uint64_t)max_elements_ * size_data_per_element_);

#if defined(PROF_Q_DISTRI) || defined(PROF_ALL_Q_DISTRI) || defined(PROF_LAZY)
    partition_vec_num = (size_t *)malloc(sizeof(size_t) * MACHINE_NUM);
    printf("Loading kmeans meta data..\n");
    int *kmeans_part_size;
    int *kmeans_pos_label_map;
    int kmeans_vec_size, kmeans_part_num;
    get_pos_label_map(
        kmeans_map.c_str(), kmeans_vec_size,
        kmeans_part_num, kmeans_part_size, kmeans_pos_label_map);
    
    for (uint32_t p = 0; p < kmeans_part_num; p++) {
      partition_vec_num[p] = kmeans_part_size[p];
      printf("Part %u size %llu\n", p, partition_vec_num[p]);
    }
#endif

    return;
  }

  void print() override {
    std::cout << "vamana index path: " << location << std::endl;
  }

  std::string get_location() override { return location; }

  uint32_t get_vector_machine(size_t vector_id) override {
    size_t cumunum = 0;
    for (int p = 0; p < kmeans_pnum; p++) {
      cumunum += partition_vec_num[p];
      if (cumunum > vector_id) return p;
    }
    assert(0);  // error
    return 0;
  }

  size_t get_internal_id(size_t vector_id) override {
    size_t cumunum = 0;
    for (int p = 0; p < kmeans_pnum; p++) {
      if (cumunum + partition_vec_num[p] > vector_id) {
        return vector_id - cumunum;
      } else {
        cumunum += partition_vec_num[p];
      }
    }
    assert(0);  // error
    return 0;
  }

  void init_ptr(
      uint32_t machine_id_, uint32_t machine_num_, size_t max_elements,
      char *data_level0_memory, size_t size_data_per_element) override {
    machine_id = machine_id_;
    machine_num = machine_num_;
    printf(
        "machine_id %d machinenum %d max_elements %d\n", machine_id,
        machine_num, max_elements);
    partition_ptr = (char **)malloc(sizeof(void *) * MACHINE_NUM);
    partition_vec_num = (size_t *)malloc(sizeof(size_t) * MACHINE_NUM);
    partition_size = (size_t *)malloc(sizeof(size_t) * MACHINE_NUM);

    // [TEMP] no use for b2
    for (int i = 0; i < MACHINE_NUM; i++) {
      // partition_vec_num[i] = max_elements_;
      // partition_size[i] = max_elements_ * size_data_per_element_;
      partition_ptr[i] = data_level0_memory_;
    }
    partition_size[machine_id_] = max_elements_ * size_data_per_element_;

    std::cout << "Partition size: " << partition_size[machine_id_] << "\n";

    return;
  }

  char *get_partition_ptr() override {
    return partition_ptr[machine_id];
    // return partition_data;
  }

  size_t get_partition_size() override { return partition_size[machine_id]; }


  // will only be called on Kmeans shard nothing baseline.
  uint32_t get_machine_id(size_t node_id) {
    for (uint32_t p = 1; p <= MACHINE_NUM; p++) {
      if (cumu_partition_num[p] > node_id) {
        return p - 1;
      }
    }
  }
};

#endif
