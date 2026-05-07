#include "coromem/include/utils.h"
#include "index/kmeans.h"

// ./tests/index/cluster
int main(int argc, char **argv) {
  commandLine cmd(
      argc, argv,
      "usage: -m <million_num> -p <part_num> -d <path_data> -o <output_path>");

  //   kmeans_cluster();
  kmeans_cluster_balance();
}
