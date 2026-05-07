#pragma once
#include "anns/anns_param.h"
#include "b1_graph.h"
#include "index_param.h"
#include "scala_graph.h"
#include "top_index.h"

/**
 * General graph index interface.
 *
 */
class GraphIndex {
 public:
  B1GraphInterface* b1_graph = nullptr;
  ScalaGraphIndex* scala_graph = nullptr;
  HNSWTopIndex* top_index = nullptr;

  IndexParameter& index_param;

  GraphIndex(IndexParameter& _index_param) : index_param(_index_param) {
    if (index_param.graph_type == HNSW_KMEANS) {
      // b1_graph = new KmeansVectorIndex<int>(
      //     anns_param.kmeans_path, anns_param.map_path,
      //     index_param.num_parts);
    } else if (index_param.graph_type == HNSW_RANDOM) {
      // b1_graph = new RandomVectorIndex<int>(index_param.final_b1_index_path);
    } else if (index_param.graph_type == VAMANA) {
      printf("VAMANA type\n");
      b1_graph = new VamanaIndex<int>(index_param);
      // here for test
      // top_index = new HNSWTopIndex(index_param);
    } 
    else if(index_param.graph_type == SharedNothing){
      b1_graph = new VamanaIndex<int>(index_param);
    }
    else if(index_param.graph_type == KmeansSharedNothing){
      printf("Graph type: SharedNothing\n");
      b1_graph = new VamanaIndex<int>(index_param);
      // for kmeans
      top_index = new HNSWTopIndex(index_param);
    }
    else if (index_param.graph_type == SCALA_V3) {
      scala_graph = new ScalaGraphIndex(index_param);
      // top_index = new TopIndex(MACHINE_NUM);
      top_index = new HNSWTopIndex(index_param);
    } else {
      printf("Error: unrecogonized graph storrage type.\n");
    }
  }

  ~GraphIndex() {}

  char* get_partition_ptr() {
    if (index_param.graph_type == SCALA_V3) {
      return scala_graph->get_partition_ptr();
    } else {
      return b1_graph->get_partition_ptr();
    }
  }

  size_t get_partition_size() {
    if (index_param.graph_type == SCALA_V3) {
      return scala_graph->get_partition_size();
    } else {
      return b1_graph->get_partition_size();
    }
  }
};
