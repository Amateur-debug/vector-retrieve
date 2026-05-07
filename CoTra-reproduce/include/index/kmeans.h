#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include "../anns/space_l2.h"
#include "anns/vec_buffer.h"
#include "coromem/include/utils.h"

/**
 * Standard Kmeans cluster.
 */
void kmeans_cluster() {
  // Vector data input.
  hnswlib::L2SpaceI spc(128);

  auto data_size_ = spc.get_data_size();
  auto fstdistfunc_ = spc.get_dist_func();
  auto dist_func_param_ = spc.get_dist_func_param();

  auto sift_split_1M =
      std::string("/data/share/users/xyzhi/data/bigann/base.1M.u8bin");

  std::ifstream data_input(sift_split_1M.c_str(), std::ios::binary);

  uint32_t vecnum = 0;
  data_input.read((char *)&vecnum, 4);
  printf("vecnum %d\n", vecnum);
  uint32_t dim = 0;
  data_input.read((char *)&dim, 4);
  printf("dim : %u\n", dim);
  char *buf = (char *)malloc(vecnum * dim);
  data_input.read(buf, vecnum * dim);

  printf("test data read over\n");

  // srand48(1);
  // // test random dist compute
  // for (int _ = 0; _ < 100; _++) {
  //   size_t v1 = rand() % vecnum;
  //   size_t v2 = rand() % vecnum;

  //   auto dist = fstdistfunc_(buf + v1 * dim, buf + v2 * dim,
  //   dist_func_param_);
  // }
  // printf("random compute over\n");

  uint32_t k;
  uint32_t cid[100];
  char centroid[100][dim];
  printf("input kmeans k >> \n");
  // Repeat till converge.
  while (std::cin >> k) {
    printf("perform %d means ...\n");
    assert(k < 100);
    // Init random center point.
    std::vector<uint32_t> kmeans_part[100];

    for (uint32_t c = 0; c < k; c++) {
      cid[c] = rand() % vecnum;
      printf("init cid%u: %u\n", c, cid[c]);
      kmeans_part[c].clear();
      char *vptr = buf + cid[c] * dim;
      for (uint32_t d = 0; d < dim; d++) {
        centroid[c][d] = vptr[d];
      }
    }
    for (uint32_t iter = 0; iter < 10; iter++) {
      float means_vec[100][dim];
      for (uint32_t c = 0; c < k; c++) {
        kmeans_part[c].clear();
        memset(means_vec[c], 0, sizeof(means_vec[c]));
      }
      // Assign the label for each data point
      for (uint32_t i = 0; i < vecnum; i++) {
        int mn = -1;
        int mn_cid = -1;
        char *vptr = buf + i * dim;
        for (uint32_t c = 0; c < k; c++) {
          auto dist = fstdistfunc_(vptr, centroid[c], dist_func_param_);
          if (dist < mn || mn == -1) {
            mn = dist;
            mn_cid = c;
          }
        }
        assert(mn_cid != -1);
        kmeans_part[mn_cid].emplace_back(i);
        for (uint32_t d = 0; d < dim; d++) {
          means_vec[mn_cid][d] += vptr[d];
        }
      }

      // Update cid
      for (uint32_t c = 0; c < k; c++) {
        for (uint32_t d = 0; d < dim; d++) {
          centroid[c][d] = uint8_t(means_vec[c][d] / kmeans_part[c].size());
          // printf("%d ", centroid[c][d]);
        }
        // printf("\n");
      }

      printf("iter %u: ", iter);
      for (uint32_t c = 0; c < k; c++) {
        printf("%u ", kmeans_part[c].size());
      }
      printf("\n");
    }
  }

  // Output data.

  return;
}

/**
 * Kmeans cluster with balance constraint.
 */
void kmeans_cluster_balance() {
  // Vector data input.
  hnswlib::L2SpaceI spc(128);

  auto data_size_ = spc.get_data_size();
  auto fstdistfunc_ = spc.get_dist_func();
  auto dist_func_param_ = spc.get_dist_func_param();

  // auto sift_split_1M =
  //     std::string("/data/share/users/xyzhi/data/bigann/base.1M.u8bin");
  // std::ifstream data_input(sift_split_1M.c_str(), std::ios::binary);

  auto sift_split_100M =
      std::string("/data/share/users/xyzhi/data/bigann/base.100M.u8bin");
  std::ifstream data_input(sift_split_100M.c_str(), std::ios::binary);

  uint64_t vecnum = 0;
  data_input.read((char *)&vecnum, 4);
  printf("vecnum %llu\n", vecnum);
  uint32_t dim = 0;
  data_input.read((char *)&dim, 4);
  printf("dim : %u\n", dim);
  char *buf = (char *)malloc(vecnum * dim);
  data_input.read(buf, vecnum * dim);

  printf("test data read over\n");

  // srand48(1);
  // // test random dist compute
  // for (int _ = 0; _ < 100; _++) {
  //   size_t v1 = rand() % vecnum;
  //   size_t v2 = rand() % vecnum;

  //   auto dist = fstdistfunc_(buf + v1 * dim, buf + v2 * dim,
  //   dist_func_param_);
  // }
  // printf("random compute over\n");

  uint32_t k;
  uint32_t cid[100];
  char centroid[100][dim];
  printf("input kmeans k >> \n");
  // Repeat till converge.
  while (std::cin >> k) {
    printf("perform %d means ...\n");
    assert(k < 100);
    // Init random center point.
    int kmeans_part_size[100];
    std::vector<uint32_t> kmeans_part[100];

    for (uint32_t c = 0; c < k; c++) {
      cid[c] = rand() % vecnum;
      printf("init cid%u: %u\n", c, cid[c]);
      kmeans_part_size[c] = 0;
      kmeans_part[c].clear();
    }

    std::vector<double> lam = {0, 0.01, 0.1, 1.0, 10.0, 100.0};
    for (double lambda : lam) {
      printf("lambda: %.2f\n", lambda);
      for (uint32_t c = 0; c < k; c++) {
        kmeans_part[c].clear();
        char *vptr = buf + cid[c] * dim;
        for (uint32_t d = 0; d < dim; d++) {
          centroid[c][d] = vptr[d];
        }
      }
      for (uint32_t iter = 0; iter < 30; iter++) {
        double means_vec[100][dim];
        for (uint32_t c = 0; c < k; c++) {
          kmeans_part_size[c] = kmeans_part[c].size();
          // printf("kmeans_part_size[%d] = %d\n", c, kmeans_part_size[c]);
          kmeans_part[c].clear();
          memset(means_vec[c], 0, sizeof(means_vec[c]));
        }
        // Assign the label for each data point by Eq. (8);
        double sum_object = 0.0;
        double sum_dist = 0.0;
        for (uint64_t i = 0; i < vecnum; i++) {
          double mn = -1.0;
          double mn_dis = -1.0;
          int mn_cid = -1;
          char *vptr = buf + i * dim;
          for (uint32_t c = 0; c < k; c++) {
            auto dist = fstdistfunc_(vptr, centroid[c], dist_func_param_);
            double weight_value = (double)dist + lambda * kmeans_part[c].size();
            if (weight_value < mn || mn < 0.0) {
              mn = weight_value;
              mn_dis = dist;
              mn_cid = c;
            }
          }
          assert(mn_cid != -1);
          sum_object += mn;
          sum_dist += mn_dis;
          kmeans_part[mn_cid].emplace_back(i);
          for (uint32_t d = 0; d < dim; d++) {
            means_vec[mn_cid][d] += vptr[d];
          }
        }

        // Update cid
        for (uint32_t c = 0; c < k; c++) {
          for (uint32_t d = 0; d < dim; d++) {
            centroid[c][d] = uint8_t(means_vec[c][d] / kmeans_part[c].size());
            // printf("%d ", centroid[c][d]);
          }
          // printf("\n");
        }

        printf(
            "iter %u sum_object %.2f sum_dist %.2f : ", iter,
            sum_object / vecnum, sum_dist / vecnum);
        for (uint32_t c = 0; c < k; c++) {
          printf("%u ", kmeans_part[c].size());
        }
        printf("\n");
      }
    }
  }

  // Output data.

  return;
}

/**
 * Rebalnace existing Kmeans cluster with balance constraint.
 * And reorder the vector data.
 */
void rebalance_kmeans_cluster() {
  // Vector data input.

  // Kmeans map input.

  // Repeat till converge.

  // Update the centroid matrix C by Eq. (7) and the cluster size si;

  // Assign the label for each data point by Eq. (8);

  // Get balanced kmeans map.

  // Reorder data.

  // Output data.
}

bool exists_test(const std::string &name) {
  std::ifstream f(name.c_str());
  return f.good();
}

/**
 * Read kmeans partition information.
 */
void read_partition(
    char path_data[][1024], int *pid, int *label_pos_map, int *pos_label_map,
    int *part_offset, int *part_size, int million_num, int vec_size,
    int part_num) {
  // char path_data[part_num][1024];
  // for (int i = 0; i < part_num; i++) {
  //   snprintf(
  //       path_data[i], sizeof(path_data[i]),
  //       "/data/share/users/xyzhi/data/bigann/%dM_kmeans/merged_%d_uint32/"
  //       "merged_%d-%d_uint32.bin",
  //       million_num, part_num, i, part_num);
  // }

  int cnt = 0;
  for (int i = 0; i < part_num; i++) {
    printf("%s\n", path_data[i]);
    if (exists_test(path_data[i])) {
      std::ifstream input(path_data[i], std::ios::binary);
      input.seekg(0, input.end);
      std::streampos total_filesize = input.tellg();
      input.seekg(0, input.beg);

      uint32_t in;
      input.read((char *)&in, 4);
      printf("%d ", in);
      input.read((char *)&in, 4);
      printf("%d\n", in);

      int in_part_offset = 0;
      while (input.tellg() < total_filesize) {
        input.read((char *)&in, 4);

        // Assign the reorder id.
        label_pos_map[in] = cnt;
        pos_label_map[cnt] = in;
        // Assign the part-offset
        part_offset[in] = (i << 28) | in_part_offset;
        in_part_offset++;

        if (pid[2 * in] != -1) {
          pid[2 * in + 1] = i + 1;  // first part
        } else {
          pid[2 * in] = i + 1;
        }
        cnt++;
      }
      part_size[i] = in_part_offset;
      printf("part_size: %d \n", part_size[i]);
      input.close();
    } else {
      printf("error\n");
    }
  }
  printf("%d\n", cnt);

  // check pid
  for (int i = 0; i < 2 * vec_size; i += 2) {
    if (!pid[i]) {
      printf("error with zero element.\n");
      printf("%d \n", i);
      break;
    }
  }
}

/**
 * write label position(id) map data.
 * pos_label_map[i]: i-th original vector.
 * label_pos_map[i]: original i-th vector postition.
 * map_data_format:
 * <vec_size> <part_num>
 * <part_size[0], part_size[1], ..., part_size[part_num-1]>
 * <pos_label_map[0]>, <pos_label_map[0]>, ..., <pos_label_map[vec_size-1]>
 */

void write_map(
    int *pos_label_map, int *label_pos_map, int *part_size, int vec_size,
    int part_num, const char *pos_label_path, const char *label_pos_path) {
  if (exists_test(pos_label_path)) {
    printf("Pos Label Map exist, skip...\n");
  } else {
    std::ofstream map_out(pos_label_path, std::ios::binary);
    std::streampos position;
    map_out.write((char *)&vec_size, 4);
    map_out.write((char *)&part_num, 4);
    map_out.write((char *)part_size, 4 * part_num);
    map_out.write((char *)pos_label_map, 4 * vec_size);
    map_out.close();
    printf("Pos Label Map file write over.\n");
  }

  if (exists_test(label_pos_path)) {
    printf("Label Pos Map exist, skip...\n");
  } else {
    std::ofstream map_out(label_pos_path, std::ios::binary);
    std::streampos position;
    map_out.write((char *)&vec_size, 4);
    map_out.write((char *)&part_num, 4);
    map_out.write((char *)part_size, 4 * part_num);
    map_out.write((char *)label_pos_map, 4 * vec_size);
    map_out.close();
    printf("Label Pos Map file write over.\n");
  }
}

void get_pos_label_map(
    const char *pos_label_path, int &vec_size, int &part_num, int *&part_size,
    int *&pos_label_map) {
  if (exists_test(pos_label_path)) {
    std::ifstream map_in(pos_label_path, std::ios::binary);
    std::streampos position;

    map_in.read((char *)&vec_size, 4);
    printf("vec_size %d\n", vec_size);
    map_in.read((char *)&part_num, 4);
    printf("part_num %d\n", part_num);
    // assert(vec_size == 1e8 && part_num == 4);

    part_size = (int *)malloc(4 * part_num);
    map_in.read((char *)part_size, 4 * part_num);
    for (int i = 0; i < part_num; i++) {
      printf("%d ", part_size[i]);
    }
    printf("\n");
    pos_label_map = (int *)malloc(4 * vec_size);
    map_in.read((char *)pos_label_map, 4 * vec_size);

    map_in.close();
    return;
  } else {
    printf("Pos Label Map file %s does not exist !!!\n", pos_label_path);
    exit(0);
    return;
  }
}

void get_label_pos_map(
    const char *label_pos_path, int &vec_size, int &part_num, int *&part_size,
    int *&label_pos_map) {
  if (exists_test(label_pos_path)) {
    std::ifstream map_in(label_pos_path, std::ios::binary);
    std::streampos position;

    map_in.read((char *)&vec_size, 4);
    printf("vec_size %d\n", vec_size);
    map_in.read((char *)&part_num, 4);
    printf("part_num %d\n", part_num);
    // assert(vec_size == 1e8 && part_num == 4);

    part_size = (int *)malloc(4 * part_num);
    map_in.read((char *)part_size, 4 * part_num);
    for (int i = 0; i < part_num; i++) {
      printf("%d ", part_size[i]);
    }
    printf("\n");
    label_pos_map = (int *)malloc(4 * vec_size);
    map_in.read((char *)label_pos_map, 4 * vec_size);

    map_in.close();
    return;
  } else {
    printf("Label Pos Map file does not exist !!!\n");
    exit(0);
    return;
  }
}

void kmeans_reorder(
    int *pid, int *label_pos_map, int *pos_label_map, int *part_offset,
    int *part_size, int million_num, size_t vec_size, int part_num,
    const char *path_data, const char *output_path, size_t vecdim = 128, size_t vec_ele_len = 128) {
  // reorder vector
  // size_t vec_ele_len = vecdim;
  char *reorder_data = (char *)malloc(vec_ele_len * vec_size);
  unsigned char *massb = new unsigned char[vec_ele_len];
  std::ifstream input(path_data, std::ios::binary);

  uint32_t vecnum_1b = 0;
  input.read((char *)&vecnum_1b, 4);
  assert(vecnum_1b == 1e9);

  uint32_t read_dim = 0;
  input.read((char *)&read_dim, 4);
  printf("dim : %u\n", read_dim);
  assert(read_dim == vecdim);

  input.read((char *)massb, vec_ele_len);
  // assign vector to new position.
  memcpy(reorder_data + (size_t)(vec_ele_len * label_pos_map[0]), massb, vec_ele_len);
  int j1 = 0;
  size_t report_every = 100000;
  for (int i = 1; i < vec_size; i++) {
    int j2 = 0;
    {
      input.read((char *)massb, vec_ele_len);
      // assign vector to new position.
      j1++;
      j2 = j1;
      memcpy(
          reorder_data + (size_t)(vec_ele_len * label_pos_map[j2]), massb, vec_ele_len);
      if (j1 % report_every == 0) {
        std::cout << j1 / (0.01 * vec_size) << " % \r" << std::flush;
      }
    }
  }
  std::cout<<"\n";
  input.close();

  // verify
  std::ifstream origin_in(path_data, std::ios::binary);
  char origin_vec[vec_ele_len];
  for (int i = 0; i < 1000; i++) {
    int vec_id = rand() % vec_size;
    origin_in.seekg(pos_label_map[vec_id] * vec_ele_len + 8, origin_in.beg);
    origin_in.read(origin_vec, vec_ele_len);

    if (memcmp(reorder_data + vec_ele_len * vec_id, origin_vec, vec_ele_len)) {
      printf("mismatch vec_id %d\n", vec_id);
      float *tmp = (float *)(reorder_data + vec_ele_len * vec_id);
      for (int i = 0; i < vecdim; i++) {
        printf("%.3f ", tmp[i]);
      }
      printf("\n ------- \n");
      tmp = (float *)origin_vec;
      for (int i = 0; i < vecdim; i++) {
        printf("%.3f ", tmp[i]);
      }
      printf("\n");
      abort();
    }
    assert(
        memcmp(reorder_data + vec_ele_len * vec_id, origin_vec, vec_ele_len) == 0);
  }
  printf("Generated data verified. \n");
  origin_in.close();

  // write reorder data.
  printf("Write reorder data ...\n");
  long write_size = 1e8;

  std::ofstream output(output_path, std::ios::binary);
  std::streampos position;
  for (size_t i = 0; i < vec_ele_len * vec_size; i += write_size) {
    if (i + write_size > vec_ele_len * vec_size) {
      write_size = (long)(vec_ele_len * vec_size - i);
    }
    output.write((char *)(reorder_data + i), write_size);
    printf(
        "%.2f%%\r",
        (float)(i + write_size) * 100.0 / (vec_ele_len * vec_size));
    sleep(2);
  }
  printf("\n");
  output.close();
  printf("K-means reorder file write over.\n");
}

void verify_reorder_data(
    const char *path_data, const char *output_path, int *tmp_pos_label,
    size_t vec_size, size_t ele_size = 128) {
  srand(1);
  std::ifstream origin_in(path_data, std::ios::binary);
  std::ifstream reorder_in(output_path, std::ios::binary);
  char origin_vec[ele_size], reorder_vec[ele_size];
  for (int i = 0; i < 1000; i++) {
    size_t vec_id = rand() % vec_size;
    reorder_in.seekg(vec_id * ele_size, reorder_in.beg);
    reorder_in.read(reorder_vec, ele_size);

    origin_in.seekg(tmp_pos_label[vec_id] * ele_size + 8, origin_in.beg);
    origin_in.read(origin_vec, ele_size);

    if (memcmp(reorder_vec, origin_vec, ele_size)) {
      printf("Wrong match vec_id %llu\n", vec_id);
      float *tmp = (float *)reorder_vec;
      printf("Reordered:");
      for (int j = 0; j < 10; j++) {
        printf("%.3f ", tmp[j]);
      }
      printf("\noriginal:");
      tmp = (float *)origin_vec;
      for (int j = 0; j < 10; j++) {
        printf("%.3f ", tmp[j]);
      }
      printf("\n");
      abort();
    }
    assert(memcmp(reorder_vec, origin_vec, ele_size) == 0);
  }
  reorder_in.close();
  origin_in.close();
  printf("K-means reorder file verified.\n");
}

void transfer_to_scalagraph() {}