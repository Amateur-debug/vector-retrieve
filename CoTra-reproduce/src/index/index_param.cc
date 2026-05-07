#include "index/index_param.h"

#include "anns/anns_param.h"
#include "diskann/include/program_options_utils.hpp"

AnnsParameter::AnnsParameter(int argc, char **argv, IndexParameter &index_param) {

  po::options_description desc{program_options_utils::make_program_description(
      "build_disk_index", "Build a disk-based index.")};
  try {
    desc.add_options()("help,h", "Print information on arguments");

    // Required parameters
    po::options_description required_configs("Required");
    // required_configs.add_options()(
    //     "graph_type", po::value<std::string>(&graph_type_str)->required(),
    //     "...");  // TODO: add description
    required_configs.add_options()(
        "config_file", po::value<std::string>(&config_file)->required(),
        "...");  // TODO: add description

    required_configs.add_options()(
        "data_type", po::value<std::string>(&data_type_str)->required(),
        program_options_utils::DATA_TYPE_DESCRIPTION);
    required_configs.add_options()(
        "dist_fn", po::value<std::string>(&dist_fn)->required(),
        program_options_utils::DISTANCE_FUNCTION_DESCRIPTION);
    // These can merge.
    required_configs.add_options()(
        "thread_num,t", po::value<uint32_t>(&thd_num)->required(),
        "TEMP thread");
    required_configs.add_options()(
        "million_num,s",
        po::value<uint32_t>(&subset_size_milllions)->required(),
        "TEMP million");

    required_configs.add_options()(
      "query_path", po::value<std::string>(&query_path)->required(),
      program_options_utils::DISTANCE_FUNCTION_DESCRIPTION);
    
    required_configs.add_options()(
      "gt_path", po::value<std::string>(&gt_path)->required(),
      program_options_utils::DISTANCE_FUNCTION_DESCRIPTION);

    required_configs.add_options()(
      "app_type", po::value<std::string>(&app_type_str)->required(),
      program_options_utils::DISTANCE_FUNCTION_DESCRIPTION);
    
    // Optional parameters
    po::options_description optional_configs("Optional");

    optional_configs.add_options()(
      "res_knn", po::value<uint32_t>(&res_knn)->default_value(10),
      "default knn size is 10");


    optional_configs.add_options()(
        "max_degree,R", po::value<uint32_t>(&M)->default_value(64),
        program_options_utils::MAX_BUILD_DEGREE);

    optional_configs.add_options()(
        "query_size,Q", po::value<size_t>(&qsize)->default_value(10000),
        "default query size is 10000");
    
    optional_configs.add_options()(
        "scala_v3", po::bool_switch()->default_value(false), "Search version");
    // Merge required and optional parameters
    desc.add(required_configs).add(optional_configs);
    

    po::variables_map vm;
    auto parsed = po::command_line_parser(argc, argv)
          .options(desc)
          .allow_unregistered()
          .run();
    po::store(parsed, vm);
    
    // po::store(po::parse_command_line(argc, argv, desc, po::command_line_style::allow_unregistered), vm);
    if (vm.count("help")) {
      std::cout << desc;
      abort();
    }
    po::notify(vm);
  } catch (const std::exception &ex) {
    std::cerr << ex.what() << '\n';
    abort();
  }

  machine_id = get_machine_id(config_file);
  vecsize = index_param.vec_size;
  vecnum = subset_size_milllions * 1000000;
  if(vecnum != index_param.vec_num){
    std::cerr<<"Missmatch parameter and data file..\n";
    abort();
  }
  vecdim = index_param.dim;

  // Reuse the already-parsed index thread setting for ANN so search and index
  // runs stay consistent even when the ANN parser falls back to its default.
  thd_num = index_param.thread_num;

  omp_set_num_threads(thd_num);
  auto numThreads = setActiveThreads(thd_num);
  printf("Set thread num = %d\n", numThreads);
  // kmeans_pnum = cmd.getOptionIntValue("-k", MACHINE_NUM);
  // std::string config_file = cmd.getOptionValue("--config_file", "none");

  printf("Run RDMA-ANNS machine number: %d\n", MACHINE_NUM);
  std::vector<std::string> machine_name = get_machine_name(config_file);
  if(machine_name.size() != MACHINE_NUM){
    std::cerr<<"Error: machine number mismatch with ip list.\n";
    abort();
  }
  if (machine_id == 0) {
    printf(
        "This is leader machine %d, name %s\n", machine_id,
        machine_name[machine_id].c_str());
  } else {
    printf(
        "This is follower machine %d, name: %s\n", machine_id,
        machine_name[machine_id].c_str());
  }

  if (dist_fn == std::string("l2")){
    if(data_type_str == std::string("uint8"))
      l2_space_i = new L2SpaceI(vecdim);
    else if(data_type_str == std::string("int8")){
      l2_space_ii = new L2SpaceII(vecdim);
    }
    else 
      l2_space = new L2Space(vecdim);
  }
  else if (dist_fn == std::string("mips"))
    ip_space = new InnerProductSpace(vecdim);
  else {
    std::cout << "Error. Only l2 and mips distance functions are supported"
              << std::endl;
    abort();
  }

  
  if(app_type_str == std::string("single")){
    app_type = SINGLE_MACHINE;
  }else if(app_type_str == std::string("b1")){
    app_type = B1;
  }else if(app_type_str == std::string("b2")){
    app_type = B2;
  }else if(app_type_str == std::string("b2kmeans")){
    app_type = B2Kmeans;
  }else if(app_type_str == std::string("b2kmeansbatch")){
    app_type = B2KmeansBatch;
  }else if(app_type_str == std::string("scala_v3")){
    app_type = ScalaANN_v3;
  }else {
    std::cerr<<"Error: expect at least one app type.\n";
    abort();
  }
}


IndexParameter::IndexParameter(int argc, char **argv) {
  _meta_data_ratio = META_DATA_RATIO;
  num_parts = MACHINE_NUM;

  std::cout << "Using meta_data_ratio: " << _meta_data_ratio << std::endl;
  po::options_description desc{program_options_utils::make_program_description(
      "build_disk_index", "Build a disk-based index.")};
  try {
    desc.add_options()("help,h", "Print information on arguments");

    // Required parameters
    po::options_description required_configs("Required");
    required_configs.add_options()(
        "graph_type", po::value<std::string>(&graph_type_str)->required(),
        "...");  // TODO: add description

    required_configs.add_options()(
        "config_file", po::value<std::string>(&config_file)->required(),
        "...");  // TODO: add description

    required_configs.add_options()(
        "data_type", po::value<std::string>(&data_type)->required(),
        program_options_utils::DATA_TYPE_DESCRIPTION);
    required_configs.add_options()(
        "dist_fn", po::value<std::string>(&dist_fn)->required(),
        program_options_utils::DISTANCE_FUNCTION_DESCRIPTION);
    required_configs.add_options()(
        "index_path_prefix",
        po::value<std::string>(&index_path_prefix)->required(),
        program_options_utils::INDEX_PATH_PREFIX_DESCRIPTION);
    required_configs.add_options()(
        "data_path", po::value<std::string>(&data_path)->required(),
        program_options_utils::INPUT_DATA_PATH);
    required_configs.add_options()(
        "search_DRAM_budget,B", po::value<float>(&B)->required(),
        "DRAM budget in GB for searching the index to set the "
        "compressed level for data while search happens");
    required_configs.add_options()(
        "build_DRAM_budget,M", po::value<float>(&M)->required(),
        "DRAM budget in GB for building the index");
    // These can merge.
    required_configs.add_options()(
        "thread_num,t", po::value<uint32_t>(&thread_num)->required(),
        "TEMP thread");
    required_configs.add_options()(
        "million_num,s",
        po::value<uint32_t>(&subset_size_milllions)->required(),
        "TEMP million");

    // Optional parameters
    po::options_description optional_configs("Optional");
    optional_configs.add_options()(
        "num_threads,T",
        po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
        program_options_utils::NUMBER_THREADS_DESCRIPTION);
    optional_configs.add_options()(
        "max_degree,R", po::value<uint32_t>(&max_degree)->default_value(64),
        program_options_utils::MAX_BUILD_DEGREE);
    optional_configs.add_options()(
        "Lbuild,L", po::value<uint32_t>(&L)->default_value(100),
        program_options_utils::GRAPH_BUILD_COMPLEXITY);
    optional_configs.add_options()(
        "QD", po::value<uint32_t>(&QD)->default_value(0),
        " Quantized Dimension for compression");
    optional_configs.add_options()(
        "codebook_prefix",
        po::value<std::string>(&codebook_prefix)->default_value(""),
        "Path prefix for pre-trained codebook");
    optional_configs.add_options()(
        "PQ_disk_bytes", po::value<uint32_t>(&disk_PQ)->default_value(0),
        "Number of bytes to which vectors should be compressed "
        "on SSD; 0 for no compression");
    optional_configs.add_options()(
        "append_reorder_data", po::bool_switch()->default_value(false),
        "Include full precision data in the index. Use only in "
        "conjuction with compressed data on SSD.");
    optional_configs.add_options()(
        "build_PQ_bytes", po::value<uint32_t>(&build_PQ)->default_value(0),
        program_options_utils::BUIlD_GRAPH_PQ_BYTES);
    optional_configs.add_options()(
        "use_opq", po::bool_switch()->default_value(false),
        program_options_utils::USE_OPQ);
    optional_configs.add_options()(
        "label_file", po::value<std::string>(&label_file)->default_value(""),
        program_options_utils::LABEL_FILE);
    optional_configs.add_options()(
        "universal_label",
        po::value<std::string>(&universal_label)->default_value(""),
        program_options_utils::UNIVERSAL_LABEL);
    optional_configs.add_options()(
        "FilteredLbuild", po::value<uint32_t>(&Lf)->default_value(0),
        program_options_utils::FILTERED_LBUILD);
    optional_configs.add_options()(
        "filter_threshold,F",
        po::value<uint32_t>(&filter_threshold)->default_value(0),
        "Threshold to break up the existing nodes to generate new graph "
        "internally where each node has a maximum F labels.");
    optional_configs.add_options()(
        "label_type",
        po::value<std::string>(&label_type)->default_value("uint"),
        program_options_utils::LABEL_TYPE_DESCRIPTION);
    // TODO: this rmv?
    optional_configs.add_options()(
        "meta_data_ratio",
        po::value<float>(&_meta_data_ratio)->default_value(0.01),
        "Ratio of meta data to be stored in meta_index");
    optional_configs.add_options()(
        "scalagraph_v2", po::bool_switch()->default_value(false),
        "ScalaGraph format");
    optional_configs.add_options()(
        "scalagraph_v3", po::bool_switch()->default_value(false),
        "ScalaGraph format");
    optional_configs.add_options()(
        "scala_v3", po::bool_switch()->default_value(false), "Search version");
    
        
    optional_configs.add_options()(
      "topindex_deg", po::value<uint32_t>(&topindex_deg)->default_value(16),
      "Top HNSW index degree");
    optional_configs.add_options()(
      "sample_percent", po::value<std::string>(&top_sample_percent)->default_value("0.01"), 
      "Top index sampling rate");
    
            
    // Merge required and optional parameters
    desc.add(required_configs).add(optional_configs);


    po::variables_map vm;
    auto parsed = po::command_line_parser(argc, argv)
          .options(desc)
          .allow_unregistered() 
          .run();
    po::store(parsed, vm);

    // po::variables_map vm;
    // po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
      std::cout << desc;
      abort();
    }
    po::notify(vm);
    if (vm["append_reorder_data"].as<bool>()) append_reorder_data = true;
    if (vm["use_opq"].as<bool>()) use_opq = true;
  } catch (const std::exception &ex) {
    std::cerr << ex.what() << '\n';
    abort();
  }

  machine_id = get_machine_id(config_file);

  bool use_filters = (label_file != "") ? true : false;
  if (dist_fn == std::string("l2"))
    metric = diskann::Metric::L2;
  else if (dist_fn == std::string("mips"))
    metric = diskann::Metric::INNER_PRODUCT;
  else if (dist_fn == std::string("cosine"))
    metric = diskann::Metric::COSINE;
  else {
    std::cout << "Error. Only l2 and mips distance functions are supported"
              << std::endl;
    abort();
  }

  if (graph_type_str == std::string("hnsw_random")) {
    graph_type = HNSW_RANDOM;
  } else if (graph_type_str == std::string("hnsw_kmeans")) {
    graph_type = HNSW_KMEANS;
  } else if (graph_type_str == std::string("shared_nothing")) {
    graph_type = SharedNothing;
  } else if (graph_type_str == std::string("Kshared_nothing")) {
    graph_type = KmeansSharedNothing;
  } else if (graph_type_str == std::string("vamana")) {
    graph_type = VAMANA;
  } else if (graph_type_str == std::string("scalagraph_v3")) {
    graph_type = SCALA_V3;
  } else {
    std::cerr << "Error. Unsupported graph type" << std::endl;
    abort();
  }

  disk_pq_dims = 0;
  use_disk_pq = false;
  build_pq_bytes = 0;

  data_file_to_use = data_path;
  labels_file_original = label_file;
  index_prefix_path = index_path_prefix;
  labels_file_to_use = index_prefix_path + "_label_formatted.txt";
  pq_pivots_path_base = codebook_prefix;
  pq_pivots_path = file_exists(pq_pivots_path_base)
                       ? pq_pivots_path_base + "_pq_pivots.bin"
                       : index_prefix_path + "_pq_pivots.bin";
  pq_compressed_vectors_path = index_prefix_path + "_pq_compressed.bin";
  mem_index_path = index_prefix_path + "_mem.index";
  final_index_path = index_prefix_path + to_string(machine_id) + "_final.index";
  final_b1_index_path =
      index_prefix_path + to_string(machine_id) + "_final_b1.index";
  final_scala_index_path =
      index_prefix_path + to_string(machine_id) + "_final_scala.index";
  final_scala_data_path =
      index_prefix_path + to_string(machine_id) + "_final_scala.data";
  disk_index_path = index_prefix_path + "_disk.index";
  medoids_path = disk_index_path + "_medoids.bin";
  centroids_path = disk_index_path + "_centroids.bin";

  labels_to_medoids_path = disk_index_path + "_labels_to_medoids.txt";
  mem_labels_file = mem_index_path + "_labels.txt";
  disk_labels_file = disk_index_path + "_labels.txt";
  mem_univ_label_file = mem_index_path + "_universal_label.txt";
  disk_univ_label_file = disk_index_path + "_universal_label.txt";
  disk_labels_int_map_file = disk_index_path + "_labels_map.txt";
  dummy_remap_file =
      disk_index_path +
      "_dummy_remap.txt";  // remap will be used if we break-up points of
                           // high label-density to create copies

  sample_base_prefix = index_prefix_path + "_sample";
  // optional, used if disk index file must store pq data
  disk_pq_pivots_path = index_prefix_path + "_disk.index_pq_pivots.bin";
  // optional, used if disk index must store pq data
  disk_pq_compressed_vectors_path =
      index_prefix_path + "_disk.index_pq_compressed.bin";
  prepped_base = index_prefix_path +
                 "_prepped_base.bin";  // temp file for storing pre-processed
                                       // base file for cosine/ mips metrics
  created_temp_file_for_processed_data = false;

  // where the universal label is to be saved in the final graph
  final_index_universal_label_file = mem_index_path + "_universal_label.txt";

  merged_index_prefix = mem_index_path + "_tempFiles";
  // save the partition size
  merged_meta_index_prefix = merged_index_prefix + "_meta_index";

  // get vector num and dim.
  diskann::get_bin_metadata(data_path, vec_num, dim);
  max_vid = vec_num;

  for (size_t i = 0; i < num_parts; i++) {
    final_part_scala_index_path[i] =
        index_prefix_path + to_string(i) + "_final_scala.index";
    final_part_scala_data_path[i] =
        index_prefix_path + to_string(i) + "_final_scala.data";
    final_part_index_path[i] =
        index_prefix_path + to_string(i) + "_final.index";
    final_part_b1_index_path[i] =
        index_prefix_path + to_string(i) + "_final_b1.index";
    part_replica_idmap_filename[i] = merged_index_prefix + "_subshard-" +
                                     std::to_string(i) + "_ids_uint32.bin";
    part_noreplica_idmap_filename[i] = merged_meta_index_prefix + "_subshard-" +
                                       std::to_string(i) + "_ids_uint32.bin";
    part_replica_base_filename[i] =
        merged_index_prefix + "_subshard-" + std::to_string(i) + ".bin";
    part_noreplica_base_filename[i] =
        merged_meta_index_prefix + "_subshard-" + std::to_string(i) + ".bin";

    part_replica_labels_filename[i] =
        merged_index_prefix + "_subshard-" + std::to_string(i) + "_labels.txt";
    part_noreplica_labels_filename[i] = merged_meta_index_prefix +
                                        "_subshard-" + std::to_string(i) +
                                        "_labels.txt";

    part_replica_index_filename[i] =
        merged_index_prefix + "_subshard-" + std::to_string(i) + "_mem.index";

    b2_index_file[i] =  merged_index_prefix + "_subshard-" + std::to_string(i) + "_b2.index";
  }
  local_replica_base_file = part_replica_base_filename[machine_id];
  local_noreplica_base_file = part_noreplica_base_filename[machine_id];
  local_replica_idmap_filename = part_replica_idmap_filename[machine_id];
  local_noreplica_idmap_filename = part_noreplica_idmap_filename[machine_id];
  local_replica_labels_file = part_replica_labels_filename[machine_id];
  local_noreplica_labels_file = part_noreplica_labels_filename[machine_id];
  local_index_file = part_replica_index_filename[machine_id];
  temp_b1_index = index_prefix_path + "_temp_b1_final.index";
  temp_label2pos_map_file = index_prefix_path + "_temp_label2pos.map";
  temp_pos2label_map_file = index_prefix_path + "_temp_pos2label.map";
  temp_id2part_map_file = index_prefix_path + "_temp_id2part.map";
  top_index_path = index_prefix_path + "_top.index";

  local_b2_index_file = b2_index_file[machine_id];

  temp_kmeans_data_file = index_prefix_path + "_temp_kmeans.data";
  // temp_vamana_b1graph_file = index_prefix_path + "_temp_vamana2b1.index";

  // Compute vector size based on dim and data type.
  if (dim == 0) {
    diskann::cerr << "Error. Dimension is 0" << std::endl;
    abort();
  }
  if (data_type == std::string("int8"))
    vec_size = dim * sizeof(int8_t);
  else if (data_type == std::string("uint8"))
    vec_size = dim * sizeof(uint8_t);
  else if (data_type == std::string("float"))
    vec_size = dim * sizeof(float);
  else {
    diskann::cerr << "Error. Unsupported data type" << std::endl;
    abort();
  }

  b1_ele_size = (max_degree + 1)*sizeof(uint32_t) + vec_size + sizeof(uint64_t);
}
