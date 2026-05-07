#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include "coromem/include/utils.h"
#include "index/kmeans.h"

using namespace std;

// #define DEBUG
// #define NO_KMEANS
#ifdef DEBUG
// const uint32_t debug_q_num = 6666;
const uint32_t debug_q_num = 4673;
// const uint32_t debug_q_num = 7492;
#endif

void read_partition(int *pid, int million_num, int vec_size, int part_num) {
  char path_data[part_num][1024];
  for (int i = 0; i < part_num; i++) {
    snprintf(
        path_data[i], sizeof(path_data[i]),
        "/data/share/users/xyzhi/data/bigann/%dM_kmeans/merged_%d_uint32/"
        "merged_%d-%d_uint32.bin",
        million_num, part_num, i, part_num);
  }

  int cnt = 0;
  for (int i = 0; i < part_num; i++) {
    printf("%s\n", path_data[i]);
    if (exists_test(path_data[i])) {
      ifstream input(path_data[i], ios::binary);
      input.seekg(0, input.end);
      std::streampos total_filesize = input.tellg();
      input.seekg(0, input.beg);

      uint32_t in;
      input.read((char *)&in, 4);
      printf("%d ", in);
      input.read((char *)&in, 4);
      printf("%d\n", in);

      while (input.tellg() < total_filesize) {
        input.read((char *)&in, 4);
        if (pid[2 * in] != -1) {
          pid[2 * in + 1] = i + 1;  // first part
        } else {
          pid[2 * in] = i + 1;
        }
        cnt++;
      }
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

// ./rdma-hnsw/profiling -m 100 -p 4
int main(int argc, char **argv) {
  commandLine cmd(argc, argv, "usage: -m <million_num> -p <part_num>");

  auto million_num = cmd.getOptionIntValue("-m", 10);
  int vec_size = million_num * 1e6;  // 10M/100M/1B
  int part_num = cmd.getOptionIntValue("-p", 2);

  int *part = (int *)malloc(vec_size * 2 * 4);
  memset(part, -1, vec_size * 2 * 4);

  read_partition(part, million_num, vec_size, part_num);

  // path profiling
  // vector<string> files{"output_path.bin"};
  // vector<string> files{"output_path.bin", "output_distribution.bin"};

  // for merged index
  // vector<string> files{"output_distribution_merge.bin"};

  // for single machine only
  // vector<string> files{"output_distribution_single.bin"};

  //   for (auto f : files) {
  //     ifstream input = ifstream(f, ios::binary);
  //     int qnum, efnum, id;
  //     vector<int> efs;
  //     input.read((char *)&qnum, 4);  // query num
  //     printf("query num: %d\n", qnum);

  //     // Record the accessed partition number of each query
  //     uint32_t *part_cnt = (uint32_t *)malloc(qnum * part_num * 4);
  //     // Record the final id that top index reached.
  //     uint32_t *base_start_id = (uint32_t *)malloc(qnum * 4);
  //     // Count subpath length num.
  //     uint32_t subplen_cnt[20000];

  //     input.read((char *)&efnum, 4);  // efs num
  //     printf("efs num: %d\n", efnum);
  //     for (int i = 0; i < efnum; i++) {
  //       input.read((char *)&id, 4);  // ef
  //       efs.emplace_back(id);
  //       printf("%d ", id);
  //     }
  //     printf("\n");
  //     for (auto ef : efs) {
  //       memset(part_cnt, 0, qnum * part_num * 4);
  //       memset(subplen_cnt, 0, 20000 * 4);
  //       memset(base_start_id, 0, qnum * 4);
  //       printf("ef: %d\n", ef);
  //       uint32_t all_path_len = 0;
  //       uint32_t all_path_num = 0;

  //       // record the sub neighbor access of a candidate.
  //       vector<uint32_t> sub_ngh;  // Only for distribution currently.
  //       uint64_t sum_ngh_num_cnt = 0;
  //       uint64_t sum_ngh_part_cnt = 0;

  //       for (int q = 0; q < qnum; q++) {
  //         int plen;
  //         input.read((char *)&plen, 4);
  //         // printf("path length: %d\n", plen);

  //         int cur_p = 0;
  //         int prev_p = -1;
  //         all_path_len += plen;

  //         sub_ngh.clear();
  //         for (int p = 0; p < plen; p++) {
  //           input.read((char *)&id, 4);
  //           if (id >> 31) {        // Sub neighbor end or top path end.
  //             id &= 0x7fffffff;    // Remove the MSB
  //             if (id >> 30) {      // Top index path end
  //               id &= 0x3fffffff;  // Remove the SSB
  //               base_start_id[q] = id;
  //               continue;
  //             }
  //             sub_ngh.emplace_back(id);
  //             // profiling sub neigbor path sub_ngh.
  //             int tmp_ngh_pid[part_num];
  //             for (int i = 0; i < part_num; i++) tmp_ngh_pid[i] = 0;
  //             for (auto ngh : sub_ngh) {
  //               tmp_ngh_pid[part[ngh * 2]]++;
  //             }
  //             int32_t cnt = 0;  // Count each hop carry how many partitions.
  //             for (int i = 0; i < part_num; i++) {
  //               if (tmp_ngh_pid[i]) {
  //                 cnt++;
  //               }
  //             }
  //             sum_ngh_num_cnt++;
  //             sum_ngh_part_cnt += cnt;
  //             sub_ngh.clear();
  //           } else {
  //             sub_ngh.emplace_back(id);
  //           }

  // // without k-means (uniform partition)
  // #ifdef NO_KMEANS
  //           int new_id_part = 1 + id / part_size;
  //           if (new_id_part != cur_p) {
  //             cur_p = new_id_part;
  //             part_cnt[part_num * q + cur_p]++;
  //             if (prev_p != -1) {
  //               // We only record new path at the end of path.
  //               subplen_cnt[p - prev_p]++;
  //               all_path_num++;
  //             }
  //             prev_p = p;
  //             // subplen_cnt[p - prev_p]++;
  //             // prev_p = p;
  //             // all_path_num++;
  //           }
  // #else
  //           // with k-means
  //           if (part[2 * id] != cur_p && part[2 * id + 1] != cur_p) {
  //             cur_p = part[2 * id];
  //             part_cnt[part_num * q + cur_p]++;
  //             // p indicate the begin of new path
  //             // [1,1,1,2     ,2,2,2,3,3,3]
  //             //  .,.,.,prev_p,.,.,.,p
  //             if (prev_p != -1) {
  //               // We only record new path at the end of path.
  //               subplen_cnt[p - prev_p]++;
  //               all_path_num++;
  //             }
  //             prev_p = p;
  //           }
  // #endif
  //           // Remember to process the last path.
  //           if (p == plen - 1) {
  //             subplen_cnt[p - prev_p + 1]++;
  //             all_path_num++;
  //           }

  // // show for debug
  // #ifdef DEBUG
  //           if (q == debug_q_num) {
  //             printf("%d ", cur_p);
  //           }
  // #endif
  //         }
  //       }

  //       // Compute the ratio of the correct start id.
  //       uint32_t start_correct = 0, start_sec_correct = 0;

  //       // compute the part ratio
  //       float ratio_cnt[part_num];
  //       for (int i = 0; i < part_num; i++) {
  //         ratio_cnt[i] = 0.0;
  //       }
  //       for (int q = 0; q < qnum; q++) {
  //         uint32_t mx = 0, mx2 = 0, mxid = 0, mxid2 = 0;
  //         for (int p = 0; p < part_num; p++) {
  //           if (part_cnt[part_num * q + p] > mx) {
  //             mx2 = mx;
  //             mxid2 = mxid;
  //             mx = part_cnt[part_num * q + p];
  //             mxid = p;
  //           } else if (part_cnt[part_num * q + p] > mx2) {
  //             mx2 = part_cnt[part_num * q + p];
  //             mxid2 = p;
  //           }
  //         }
  //         // If the top-1 partition id equal to the base start id.
  //         if (mxid == part[2 * base_start_id[q]]) {
  //           start_correct++;
  //         } else if (mxid2 == part[2 * base_start_id[q]]) {
  //           start_sec_correct++;
  //         }

  //         sort(
  //             part_cnt + part_num * q, part_cnt + part_num * q + part_num,
  //             greater<int>());
  //         assert(
  //             part_cnt[part_num * q] >= part_cnt[part_num * q + 1] &&
  //             part_cnt[part_num * q + 1] >= part_cnt[part_num * q + 2]);

  //         uint32_t sum_part = 0;
  //         for (int p = 0; p < part_num; p++) {
  //           sum_part += part_cnt[part_num * q + p];
  //         }
  //         for (int p = 0; p < part_num; p++) {
  //           float pr = part_cnt[part_num * q + p] * 100.0 / sum_part;
  //           ratio_cnt[p] += pr;
  // #ifdef DEBUG
  //           if (q == debug_q_num) {
  //             printf("+ratio %.2f\n ", pr);
  //           }
  // #endif
  //         }
  //         // ratio_cnt[mx * 10 / sum]++;
  //       }

  //       printf(
  //           "start_correct ratio %.2f%% start_sec_correct ratio %.2f%%\n ",
  //           (float)start_correct * 100.0 / qnum,
  //           (float)(start_sec_correct + start_correct) * 100.0 / qnum);

  //       printf(
  //           "avg sub_ngh_part_num %.2f\n",
  //           sum_ngh_part_cnt * 1.0 / sum_ngh_num_cnt);

  //       // uint32_t part_sum = 0;
  //       // for (int i = 0; i < part_num; i++) {
  //       //   part_sum += ratio_cnt[i];
  //       // }
  //       // for (int i = 0; i < part_num; i++) {
  //       //   printf("top-%d: %.2f\n", i + 1, ratio_cnt[i] * 100.0 /
  //       part_sum);
  //       // }
  //       for (int i = 0; i < part_num; i++) {
  //         printf("top-%d: %.2f %%\n", i + 1, ratio_cnt[i] / qnum);
  //       }

  //       printf("average acc num: %.2f\n", all_path_len * 1.0 / qnum);
  //       // Compute the average subpath
  //       printf("average sub path num: %.2f\n", all_path_num * 1.0 / qnum);

  //       uint32_t cumu_num = 0;
  //       uint32_t weight_cumu_num = 0;

  //       uint32_t interval[] = {1, 2, 3, 4, 5, 10, 100, 500, 1000, 10000,
  //       20000};

  //       for (int l = 0; l < sizeof(interval) / sizeof(uint32_t) - 1; l++) {
  //         uint32_t range_sublen_cnt = 0, range_sublen_cumu = 0;
  //         for (int i = interval[l]; i < interval[l + 1]; i++) {
  //           cumu_num += subplen_cnt[i];
  //           weight_cumu_num += subplen_cnt[i] * i;
  //           range_sublen_cnt += subplen_cnt[i];
  //           range_sublen_cumu += subplen_cnt[i] * i;
  //         }
  //         printf(
  //             "[%d, %.2f, %.2f, %.2f, %.2f],\n", interval[l],
  //             range_sublen_cnt * 100.0 / all_path_num,
  //             cumu_num * 100.0 / all_path_num,
  //             range_sublen_cumu * 100.0 / all_path_len,
  //             weight_cumu_num * 100.0 / all_path_len);
  //       }
  //       printf(
  //           "cumu_num: %u all_path_num: %u "
  //           "weight_cumu_num: %u all_path_len: %u\n",
  //           cumu_num, all_path_num, weight_cumu_num, all_path_len);
  //       assert(cumu_num == all_path_num && weight_cumu_num == all_path_len);
  //     }
  //     input.close();
  //   }
}
