#include "anns/exec_query.h"
#include "index/graph_index.h"

#include <cstdlib>

static bool cotra_skip_recall_enabled() {
  const char *env = std::getenv("COTRA_SKIP_RECALL");
  return env != nullptr && env[0] != '\0' && std::string(env) != "0" &&
         std::string(env) != "false" && std::string(env) != "False";
}

// ../scripts/scala_anns.sh
int main(int argc, char **argv) {
  SharedMem coromem;
  // coromem.tp.hw_info.show();

  commandLine cmd(
      argc, argv,
      "Usage : -d <data_path> -q <query_path> -m <merge_index> -i <index> -g "
      "<gt> --<app_type = single, b1, b2, b1_migrate, b1_migrate_async, "
      "b1_async, scala_v0, scala_v1>\n");

  // AnnsParameter anns_param(cmd);
  
  RdmaParameter rdma_param(cmd);
  IndexParameter index_param(argc, argv);
  AnnsParameter anns_param(argc, argv, index_param);
  
  GraphIndex graph_index(index_param);

  if(anns_param.dist_fn == std::string("l2")){
    if(anns_param.data_type_str == std::string("uint8")){
      if (cotra_skip_recall_enabled()) {
        anns_param.load_query_only();
      } else {
        anns_param.load_query_gt(anns_param.int_answers);
      }
      ScalaSearch<int> appr_alg(anns_param, rdma_param, anns_param.l2_space_i, graph_index);
      appr_alg.set_evaluation_save_path(anns_param.evaluation_save_path);
      test_vs_recall<int>(anns_param, appr_alg, anns_param.int_answers);
    }
    else if(anns_param.data_type_str == std::string("int8")){
      if (cotra_skip_recall_enabled()) {
        anns_param.load_query_only();
      } else {
        anns_param.load_query_gt(anns_param.int_answers);
      }
      ScalaSearch<int> appr_alg(anns_param, rdma_param, anns_param.l2_space_ii, graph_index);
      appr_alg.set_evaluation_save_path(anns_param.evaluation_save_path);
      test_vs_recall<int>(anns_param, appr_alg, anns_param.int_answers);
    }
    else if(anns_param.data_type_str == std::string("float")){
      if (cotra_skip_recall_enabled()) {
        anns_param.load_query_only();
      } else {
        anns_param.load_query_gt(anns_param.float_answers);
      }
      ScalaSearch<float> appr_alg(anns_param, rdma_param, anns_param.l2_space, graph_index);
      appr_alg.set_evaluation_save_path(anns_param.evaluation_save_path);
      test_vs_recall<float>(anns_param, appr_alg, anns_param.float_answers);
    }
  }else if(anns_param.dist_fn == std::string("mips")){
    if (cotra_skip_recall_enabled()) {
      anns_param.load_query_only();
    } else {
      anns_param.load_query_gt(anns_param.float_answers);
    }
    ScalaSearch<float> appr_alg(anns_param, rdma_param, anns_param.ip_space, graph_index);
    appr_alg.set_evaluation_save_path(anns_param.evaluation_save_path);
    test_vs_recall<float>(anns_param, appr_alg, anns_param.float_answers);
  } else{
    std::cerr<<"Error unexpected dist function.\n";
    abort();
  }

  cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb \n";

  return 0;
}
