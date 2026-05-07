
#include <omp.h>

#include <boost/program_options.hpp>

#include "coromem/include/share_mem.h"
#include "coromem/include/utils.h"
#include "index/scala_build.h"
#include "index/top_index.h"
#include "index/vamana/vamana.h"

int main(int argc, char **argv) {
  SharedMem coromem;
  commandLine cmd(argc, argv, "Usage : \n");
  RdmaParameter rdma_param(cmd);
  auto &tp = getThreadPool();
  tp.poolPause();

  IndexParameter index_param(argc, argv);

  ScalaBuild scala_build(index_param, rdma_param);

  // Start rdma server.
  scala_build.init_rdma();

  // Start to do partition and dispatch.
  scala_build.partition();

  
  // Build local partition index.
  scala_build.swap_partition_info();

  scala_build.build();


  // [For Test] test in single machine 
  // if(rdma_param.machine_id == 0){
  //   scala_build.single_build();
  // }
  // else {
  //   return 0;
  // }
  
  tp.poolContiue();

  if (index_param.graph_type == GraphType::SCALA_V3){
    scala_build.write_partition_info();

    // Load partiiton.
    scala_build.partition_checkpoint();

    // [For Test] merge in single machine for test.
    // scala_build.single_merge();

    // Merge graph from other machine.
    scala_build.merge();

    // Now reformat b1 graph to scalagraph v3.
    scala_build.transfer_to_scala_v3();


    // [For Test] verify scala_v3.
    // if (index_param.machine_id == 0) {
    //   scala_build.verify_scala_v3();
    // }

    // [For Test] Vamana.
    if (index_param.machine_id == 0) {
      scala_build.merge_all_b1_index();
    }

    // build top index
    if (index_param.machine_id == 0) {
      scala_build.topindex_checkpoint();
      HNSWTopIndex top_index;
      top_index.build_hnsw_top_index(scala_build.index_param, cmd);
    }
  }
  else if (index_param.graph_type == GraphType::SharedNothing){
    scala_build.write_partition_info();
    // transfer to b2graph
    scala_build.shared_nothing_checkpoint();
    scala_build.trans_to_b2graph();

    // for K-means shard-nothing
    if (index_param.machine_id == 0) {
      scala_build.topindex_checkpoint();
      HNSWTopIndex top_index;
      top_index.build_hnsw_top_index(scala_build.index_param, cmd);
    }
  }
  
  // Delete temp files.
  // scala_build.delete_temp_files();
  return 0;
}