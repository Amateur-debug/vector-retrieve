#include <fstream>
#include <iostream>

#include "index/scala_graph.h"
#include "index/vamana/vamana.h"


void load_query() {
  char default_query_path[1024];  // Default query path
  std::string query_path;         // Query path
  char default_gt_path[1024];     // default gt path
  std::string gt_path;            // gt path

  char default_o_query_path[1024];  // Default query path
  std::string o_query_path;         // Query path
  char default_o_gt_path[1024];     // default gt path
  std::string o_gt_path;            // gt path

  uint32_t qsize = 10000;
  uint32_t vecdim = 128;
  snprintf(
      default_query_path, sizeof(default_query_path),
      "/data/share/users/xyzhi/data/bigann/query.public.10K.u8bin");
  query_path = std::string(default_query_path);

  snprintf(
      default_gt_path, sizeof(default_gt_path),
      "/data/share/users/xyzhi/data/bigann/GT_%dM/bigann-%dM", 100, 100);
  gt_path = std::string(default_gt_path);

  snprintf(
      default_o_query_path, sizeof(default_o_query_path),
      "/data/share/users/xyzhi/data/bigann/query.public.100K.u8bin");
  o_query_path = std::string(default_o_query_path);

  snprintf(
      default_o_gt_path, sizeof(default_o_gt_path),
      "/data/share/users/xyzhi/data/bigann/GT_%dM/bigann-%dM_100K", 100, 100);
  o_gt_path = std::string(default_o_gt_path);

  unsigned char *massb = new unsigned char[vecdim];
  unsigned char *massQ = new unsigned char[qsize * vecdim];

  printf("Loading GT from: %s\n", gt_path.c_str());
  ifstream inputGT(gt_path.c_str(), ios::binary);
  unsigned int *massQA = new unsigned int[qsize * 1000];

  uint32_t num_queries, K_NN;
  inputGT.read((char *)(&num_queries), 4);
  inputGT.read((char *)(&K_NN), 4);
  printf("Query number: %d K-NN: %d\n", num_queries, K_NN);

  for (int i = 0; i < qsize; i++) {
    inputGT.read((char *)(massQA + K_NN * i), K_NN * 4);
  }
  inputGT.close();

  uint32_t o_num_queries = 10 * num_queries;
  uint32_t o_K_NN = K_NN;
  ofstream outputGT(o_gt_path.c_str(), ios::binary);
  outputGT.write((char *)(&o_num_queries), 4);
  outputGT.write((char *)(&o_K_NN), 4);

  for (int r = 0; r < 10; r++) {
    outputGT.write((char *)(massQA), qsize * K_NN * 4);
  }
  outputGT.close();

  printf("Loading queries from %s:\n", query_path.c_str());
  ifstream inputQ(query_path.c_str(), ios::binary);

  uint32_t num_q, dim;
  inputQ.read((char *)(&num_q), 4);
  inputQ.read((char *)(&dim), 4);
  printf("Query number: %d Dim: %d\n", num_q, dim);

  for (int i = 0; i < qsize; i++) {
    inputQ.read((char *)massb, dim);
    for (int j = 0; j < dim; j++) {
      massQ[i * dim + j] = massb[j];
    }
  }
  inputQ.close();

  uint32_t o_num_q = 10 * num_q;
  uint32_t o_dim = dim;
  ofstream outputQ(o_query_path.c_str(), ios::binary);
  outputQ.write((char *)(&o_num_q), 4);
  outputQ.write((char *)(&o_dim), 4);
  for (uint32_t r = 0; r < 10; r++) {
    outputQ.write((char *)(massQ), qsize * dim);
  }
  outputQ.close();
}
int main() {
  load_query();
  return 0;
}