#include "index/top_index.h"


// ./tests/index/top_index
int main(int argc, char **argv) {
  auto vamana = std::string(
      "/data/share/users/xyzhi/data/bigann/vamana/balance_kmeans_4_part/"
      "kmeans_index_mem.index");
  auto vamana_top_index = std::string(
      "/data/share/users/xyzhi/data/bigann/vamana/balance_kmeans_4_part/"
      "kmeans_top_index_mem.index");

  // generate_top_index(vamana, 100000000, vamana_top_index);
  TopIndex top_index;
  top_index.show_top_index();

  return 0;
}