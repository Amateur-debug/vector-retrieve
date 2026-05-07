#pragma once

#include "diskann/include/index.h"
#include "diskann/include/utils.h"
#include "index/disk_utils.h"
#include "index/math_utils.h"
#include "rdma/rdma_comm.h"

/**
 * ScalaBuild: Build ScalaGraph based on vector data and index algorithm.
 * Input: vector data, index algorithm.
 * Output: Shared ScalaGraph.
 */
class ScalaBuild {
 public:
  char *for_read_ptr;
  uint32_t for_read_sz;

  IndexParameter index_param;

  // RDMA communication info.
  RdmaParameter rdma_param;
  RdmaCommunication rdma_comm;

  ScalaBuild(IndexParameter &index_param_, RdmaParameter &rdma_param_)
      : index_param(index_param_), rdma_param(rdma_param_) {
    index_param.show();
  }

  ~ScalaBuild() {}

  /*Start rdma server*/
  void init_rdma() {
    // TODO: temp buffer here.
    for_read_sz = 1024 * 1024 * 1024;
    for_read_ptr = (char *)malloc(for_read_sz);
    rdma_comm.com_init(for_read_ptr, for_read_sz, rdma_param);
    rdma_comm.com_read_init();
    rdma_comm.com_write_init();
  }

  void partition() {
    if (index_param.graph_type == GraphType::SCALA_V3) {
      index_param.replica_num = 2; // set to two for global index
      kmeans_partition();
     } else if (index_param.graph_type == GraphType::SharedNothing) {
      index_param.replica_num = 1; // set to one
      random_partition();
    } else if (index_param.graph_type == GraphType::KmeansSharedNothing) {
      index_param.replica_num = 1; // set to one
      kmeans_partition();
    } else {
      std::cerr << "Error: Graph type not supported.\n";
      abort();
    }
    return;
  }

  void write_partition_info(){
    // ONLY ONE MACHINE WRITE THIS FILE IN OUR CLUSTER
    if (rdma_param.machine_id == 0) {
      std::cout << "Write Partition Map Info ...\n";
      std::cout << "max vid: " << index_param.max_vid
              << "dim: " << index_param.dim << "\n";
      // std::cout << "Start to send partition info ...\n";
      index_param.kmeans_id2part_map =
          (uint32_t *)malloc(sizeof(uint32_t) * (index_param.max_vid));
      memset(
          index_param.kmeans_id2part_map, 0,
          sizeof(index_param.kmeans_id2part_map));

      // for (uint64_t m = 0; m < index_param.num_parts; m++) {
      //   diskann::read_idmap(
      //       index_param.part_replica_idmap_filename[m],
      //       index_param.part_replica_ids[m]);
      //   std::cout << " replica partition " << m
      //             << " size: " << index_param.part_replica_ids[m].size()
      //             << std::endl;
      // }
      for (uint64_t m = 0; m < index_param.num_parts; m++) {
        diskann::read_idmap(
            index_param.part_noreplica_idmap_filename[m],
            index_param.part_noreplica_ids[m]);
        index_param.partition_size[m] = index_param.part_noreplica_ids[m].size();
        std::cout << " no-replica partition " << m
                  << " size: " << index_param.partition_size[m] << std::endl;
      }
      // reorganize id for each partition.
      for (uint64_t m = 0; m < index_param.num_parts; m++) {
        size_t nelems = index_param.part_noreplica_ids[m].size();
        printf("noreplica id size: %d\n", nelems);
        for (uint32_t i = 0; i < nelems; i++) {
          uint32_t id = index_param.part_noreplica_ids[m][i];
          index_param.kmeans_id2part_map[id] = m;
        }
      }
      std::ofstream id2part_map_out(
          index_param.temp_id2part_map_file, std::ios::binary);
      id2part_map_out.write((char *)&index_param.max_vid, sizeof(uint32_t));
      id2part_map_out.write((char *)&index_param.num_parts, sizeof(uint32_t));
      id2part_map_out.write(
          (char *)index_param.kmeans_id2part_map,
          sizeof(uint32_t) * (index_param.max_vid));
      id2part_map_out.close();
      std::cout << "Id to Part Map file write over.\n";
      free(index_param.kmeans_id2part_map);

      
      // create label2pos map and pos2label map
      index_param.label2pos_map =
          (uint32_t *)malloc(index_param.max_vid * sizeof(uint32_t));
      memset(index_param.label2pos_map, 0, sizeof(index_param.label2pos_map));
      index_param.pos2label_map =
          (uint32_t *)malloc(index_param.max_vid * sizeof(uint32_t));
      memset(index_param.pos2label_map, 0, sizeof(index_param.pos2label_map));

      uint32_t pos = 0;
      for (uint32_t m = 0; m < MACHINE_NUM; m++) {
        for (uint32_t i = 0; i < index_param.part_noreplica_ids[m].size(); i++) {
          uint32_t v = index_param.part_noreplica_ids[m][i];
          index_param.label2pos_map[v] = pos;
          index_param.pos2label_map[pos] = v;
          pos++;
        }
      }
      /**
      * write label position(id) map data.
      * pos_label_map[i]: i-th original vector.
      * label_pos_map[i]: original i-th vector postition.
      * map_data_format:
      * <vec_num> <part_num>
      * <part_size[0], part_size[1], ..., part_size[part_num-1]>
      * <pos_label_map[0]>, <pos_label_map[0]>, ..., <pos_label_map[vec_num-1]>
      */
      std::ofstream p2l_map_out(
          index_param.temp_pos2label_map_file, std::ios::binary);
      p2l_map_out.write((char *)&index_param.max_vid, sizeof(uint32_t));
      p2l_map_out.write((char *)&index_param.num_parts, sizeof(uint32_t));
      for (uint32_t i = 0; i < index_param.num_parts; i++) {
        p2l_map_out.write(
            (char *)&index_param.partition_size[i], sizeof(uint32_t));
      }
      p2l_map_out.write(
          (char *)index_param.pos2label_map,
          sizeof(uint32_t) * index_param.max_vid);
      p2l_map_out.close();
      printf("Pos Label Map file write over.\n");

      std::ofstream l2p_map_out(
          index_param.temp_label2pos_map_file, std::ios::binary);
      l2p_map_out.write((char *)&index_param.max_vid, sizeof(uint32_t));
      l2p_map_out.write((char *)&index_param.num_parts, sizeof(uint32_t));
      for (uint32_t i = 0; i < index_param.num_parts; i++) {
        l2p_map_out.write(
            (char *)&index_param.partition_size[i], sizeof(uint32_t));
      }
      l2p_map_out.write(
          (char *)index_param.label2pos_map,
          sizeof(uint32_t) * index_param.max_vid);
      l2p_map_out.close();
      std::cout << "Label Pos Map file write over.\n";
      free(index_param.label2pos_map);
      free(index_param.pos2label_map);
      for (uint32_t m = 0; m < MACHINE_NUM; m++) {
        index_param.part_noreplica_ids[m].clear();
        index_param.part_replica_ids[m].clear();
      }
      rdma_comm.build_sync();
    }
    else {
      rdma_comm.build_sync();
    }
  }

  /**
   * As a sync before build.
   */
  void swap_partition_info() {
    if (rdma_param.machine_id == 0) {
      /**
       * Send meta data and partition to remote machine.
       */
      for (uint32_t m = 0; m < MACHINE_NUM; m++) {
        if (m == index_param.machine_id) {
          // local partition.
          continue;
        }
        rdma_comm.post_partition_info(index_param, m);
      }
      std::cout << "Partition info send over.\n";
    } else {
      // Other machine wait for partition.
      std::cout << "Waiting for partition info ...\n";
      rdma_comm.poll_partition_info(index_param);
      std::cout << "Partition info all recved.\n";
    }
    return;
  }

  void partition_checkpoint() {
    // update index param about partition meta info.
    // show replica ids info.
    std::cout << "======== Partition Checkpoint BEGIN ========\n";
    // std::ifstream base_reader(index_param.data_path, std::ios::binary);
    // base_reader.read((char *)&index_param.max_vid, sizeof(uint32_t));
    // base_reader.read((char *)&index_param.dim, sizeof(uint32_t));
    // base_reader.close();
    // index_param.max_degree = index_param.max_degree;
    
    // show no-replica ids info and update partition size.
    // std::vector<std::vector<uint32_t>>
    // noreplica_idmaps(index_param.num_parts);
    
    diskann::read_idmap(
        index_param.local_noreplica_idmap_filename,
        index_param.local_noreplica_ids);
    std::cout << " noreplica partition " << index_param.machine_id
        << " size: " << index_param.local_noreplica_ids.size()
        << std::endl;
    // for (uint64_t m = 0; m < index_param.num_parts; m++) {
    //   index_param.local_part2id_map[m].clear();
    // }
    size_t nelems = 0;
    if(index_param.replica_num > 1){
      diskann::read_idmap(
        index_param.local_replica_idmap_filename,
        index_param.local_replica_ids);
      std::cout << " replica partition " << index_param.machine_id
          << " size: " << index_param.local_replica_ids.size()
          << std::endl;
      
      // find max local node id 
      nelems = index_param.local_replica_ids.size();
      index_param.local_max_vid = 0;
      for (uint32_t i = 0; i < nelems; i++) {
        uint32_t id = index_param.local_replica_ids[i];
        index_param.local_max_vid =
            std::max(index_param.local_max_vid, (size_t)id);
        // uint32_t p = index_param.kmeans_id2part_map[id];
        // index_param.local_part2id_map[p].emplace_back(id);
      }
    }
    else {
      nelems = index_param.local_noreplica_ids.size();
    }

  
    // read kmeans_id2part_map
    index_param.kmeans_id2part_map =
          (uint32_t *)malloc(sizeof(uint32_t) * (index_param.max_vid));
    memset(
        index_param.kmeans_id2part_map, 0,
        sizeof(index_param.kmeans_id2part_map));

    std::ifstream id2part_map_in(
        index_param.temp_id2part_map_file, std::ios::binary);
    uint32_t read_max_vid, read_num_parts;
    id2part_map_in.read((char *)&read_max_vid, sizeof(uint32_t));
    id2part_map_in.read((char *)&read_num_parts, sizeof(uint32_t));
    // if(read_max_vid != index_param.max_vid || read_num_parts != index_param.num_parts){
    if(read_num_parts != index_param.num_parts){
      std::cerr << "Error: Id2part map Mismactch of numparts.\n";
      std::cerr << "read_num_parts " << read_num_parts 
        << " index_param.num_parts" << index_param.num_parts <<"\n";
      abort();
    }
    id2part_map_in.read(
        (char *)index_param.kmeans_id2part_map,
        sizeof(uint32_t) * (index_param.max_vid));
    id2part_map_in.close();

    // read label2pos_map & pos2label_map
    index_param.label2pos_map =
        (uint32_t *)malloc(index_param.max_vid * sizeof(uint32_t));
    memset(index_param.label2pos_map, 0, sizeof(index_param.label2pos_map));
    index_param.pos2label_map =
        (uint32_t *)malloc(index_param.max_vid * sizeof(uint32_t));
    memset(index_param.pos2label_map, 0, sizeof(index_param.pos2label_map));
    
    std::ifstream p2l_map_in(
        index_param.temp_pos2label_map_file, std::ios::binary);
    p2l_map_in.read((char *)&read_max_vid, sizeof(uint32_t));
    p2l_map_in.read((char *)&read_num_parts, sizeof(uint32_t));
    if(read_max_vid != index_param.max_vid || read_num_parts != index_param.num_parts){
      std::cerr << "Error: label2pos map Mismactch of max vid or numparts.\n";
      abort();
    }
    for (uint32_t i = 0; i < index_param.num_parts; i++) {
      p2l_map_in.read(
          (char *)&index_param.partition_size[i], sizeof(uint32_t));
    }
    p2l_map_in.read(
        (char *)index_param.pos2label_map,
        sizeof(uint32_t) * index_param.max_vid);
    p2l_map_in.close();
    printf("Pos Label Map file read over.\n");

    std::ifstream l2p_map_in(
        index_param.temp_label2pos_map_file, std::ios::binary);
    l2p_map_in.read((char *)&read_max_vid, sizeof(uint32_t));
    l2p_map_in.read((char *)&read_num_parts, sizeof(uint32_t));
    if(read_max_vid != index_param.max_vid || read_num_parts != index_param.num_parts){
      std::cerr << "Error: label2pos map Mismactch of max vid or numparts.\n";
      abort();
    }
    for (uint32_t i = 0; i < index_param.num_parts; i++) {
      l2p_map_in.read(
          (char *)&index_param.partition_size[i], sizeof(uint32_t));
    }
    l2p_map_in.read(
        (char *)index_param.label2pos_map,
        sizeof(uint32_t) * index_param.max_vid);
    l2p_map_in.close();
    printf("Label Pos Map file read over.\n");

    if(index_param.replica_num > 1){
      diskann::cout << "# max_vid: " << index_param.max_vid
                  << ", max_degree: " << index_param.max_degree
                  << ", # nodeselems(include replica): " << nelems << std::endl;
    }
    else {
      diskann::cout << "# max_vid: " << index_param.max_vid
                  << ", max_degree: " << index_param.max_degree
                  << ", # nodeselems(without replica): " << nelems << std::endl;
    }
    std::cout << "======== Partition Checkpoint END ========\n";
  }

  /* Balanced kmeans & replica & dispatch partition vector data. */
  int kmeans_partition() {
    if (rdma_param.machine_id == 0) {
      try {
        if (index_param.label_file != "" &&
            index_param.label_type == "ushort") {
          if (index_param.data_type == std::string("int8"))
            diskann::kmeans_partition<int8_t>(index_param);
          else if (index_param.data_type == std::string("uint8"))
            diskann::kmeans_partition<uint8_t, uint16_t>(index_param);
          else if (index_param.data_type == std::string("float"))
            diskann::kmeans_partition<float, uint16_t>(index_param);
          else {
            diskann::cerr << "Error. Unsupported data type" << std::endl;
            abort();
          }
        } else {
          if (index_param.data_type == std::string("int8"))
            diskann::kmeans_partition<int8_t>(index_param);
          else if (index_param.data_type == std::string("uint8"))
            diskann::kmeans_partition<uint8_t>(index_param);
          else if (index_param.data_type == std::string("float"))
            diskann::kmeans_partition<float>(index_param);
          else {
            diskann::cerr << "Error. Unsupported data type" << std::endl;
            abort();
          }
        }
      } catch (const std::exception &e) {
        std::cout << std::string(e.what()) << std::endl;
        diskann::cerr << "Kmeans Partition failed." << std::endl;
        abort();
      }
    }
    return 0;
  }

  /* Balanced kmeans & replica & dispatch partition vector data. */
  int random_partition() {
    if (rdma_param.machine_id == 0) {
      try {
        if (index_param.label_file != "" &&
            index_param.label_type == "ushort") {
          if (index_param.data_type == std::string("int8"))
            diskann::random_partition<int8_t>(index_param);
          else if (index_param.data_type == std::string("uint8"))
            diskann::random_partition<uint8_t, uint16_t>(index_param);
          else if (index_param.data_type == std::string("float"))
            diskann::random_partition<float, uint16_t>(index_param);
          else {
            diskann::cerr << "Error. Unsupported data type" << std::endl;
            abort();
          }
        } else {
          if (index_param.data_type == std::string("int8"))
            diskann::random_partition<int8_t>(index_param);
          else if (index_param.data_type == std::string("uint8"))
            diskann::random_partition<uint8_t>(index_param);
          else if (index_param.data_type == std::string("float"))
            diskann::random_partition<float>(index_param);
          else {
            diskann::cerr << "Error. Unsupported data type" << std::endl;
            abort();
          }
        }
      } catch (const std::exception &e) {
        std::cout << std::string(e.what()) << std::endl;
        diskann::cerr << "Random Partition failed." << std::endl;
        abort();
      }
    }
    return 0;
  }

  /**
  * build all partition in local machine. For test.
  */
  void single_build(){
    // build partition graph index.
    printf("<<>> index_param.data_type: %s", index_param.data_type.c_str());
    try {
      if (index_param.data_type == std::string("int8"))
        diskann::build_all_partition<int8_t>(index_param);
      else if (index_param.data_type == std::string("uint8"))
        diskann::build_all_partition<uint8_t>(index_param);
      else if (index_param.data_type == std::string("float"))
        diskann::build_all_partition<float>(index_param);
      else {
        diskann::cerr << "Error. Unsupported data type" << std::endl;
        abort();
      }
    } catch (const std::exception &e) {
      std::cout << std::string(e.what()) << std::endl;
      diskann::cerr << "Build Partition Index Failed." << std::endl;
      abort();
    }
    return;
  }


  /*Build index based on given algorithms. */
  void build() {
    // build partition graph index.
    try {
      if (index_param.label_file != "" && index_param.label_type == "ushort") {
        if (index_param.data_type == std::string("int8"))
          diskann::build_local_partition<int8_t>(index_param);
        else if (index_param.data_type == std::string("uint8"))
          diskann::build_local_partition<uint8_t, uint16_t>(index_param);
        else if (index_param.data_type == std::string("float"))
          diskann::build_local_partition<float, uint16_t>(index_param);
        else {
          diskann::cerr << "Error. Unsupported data type" << std::endl;
          abort();
        }
      } else {
        if (index_param.data_type == std::string("int8"))
          diskann::build_local_partition<int8_t>(index_param);
        else if (index_param.data_type == std::string("uint8"))
          diskann::build_local_partition<uint8_t>(index_param);
        else if (index_param.data_type == std::string("float"))
          diskann::build_local_partition<float>(index_param);
        else {
          diskann::cerr << "Error. Unsupported data type" << std::endl;
          abort();
        }
      }
    } catch (const std::exception &e) {
      std::cout << std::string(e.what()) << std::endl;
      diskann::cerr << "Build Partition Index Failed." << std::endl;
      abort();
    }
    // sync for all build.
    rdma_comm.build_sync();
    std::cout << "All partition index build over...\n";
  }

  /**
   * Scatter & Gather merge graph index.
   * Final index format is b1graph.
   * */
  void merge() {
    if (index_param.graph_type == GraphType::SCALA_V3) {
      std::cout << "Start to merge ...\n";
      // Step 1: partition the replica part for each machine.
      std::cout << "Reorganize replica partitions ...\n";
      diskann::collect_replica_partition(index_param);
      // Step 2: Send the replica part to each machine.
      std::cout << "Dispatch replica partitions ...\n";
      send_replica_partition();
      // Step 3: Merge the replica parts from each machine.
      diskann::merge_replica_partition(index_param);
      // trasfer to b1graph format
      transfer_to_b1graph_format();
      rdma_comm.build_sync();
    } else if (index_param.graph_type == GraphType::SharedNothing) {
      // single machine merge.
      trans_to_b2graph();
    } else {
      // TODO: need modify here.
      // single machine merge.
      diskann::merge_shards(
          index_param.merged_index_prefix + "_subshard-", "_mem.index",
          index_param.merged_index_prefix + "_subshard-", "_ids_uint32.bin",
          index_param.num_parts, index_param.max_degree, index_param.mem_index_path,
          index_param.medoids_path, index_param.use_filters,
          index_param.labels_to_medoids_path);
    }
  }

  /**
   * TEMP: single machine merge function for verify.
   * store to index_param.mem_index_path.
   */
  void single_merge(){
    diskann::merge_shards(
      index_param.merged_index_prefix + "_subshard-", "_mem.index",
      index_param.merged_index_prefix + "_subshard-", "_ids_uint32.bin",
      index_param.num_parts, index_param.max_degree, index_param.mem_index_path,
      index_param.medoids_path, index_param.use_filters,
      index_param.labels_to_medoids_path);
  }

  void send_replica_partition() {
    for (uint32_t m = 0; m < MACHINE_NUM; m++) {
      if (m == index_param.machine_id) {
        // clear finish flag
        clear_finish();
        // collect local ids & nghs (local ngh need not to send)
        collect_local_ids_nghs();

        // wait for all shard recved.
        diskann::cout << "Collect partition " << m << "from other machine ..."
                      << std::endl;
        // wait for all meta data.
        while (!collect_meta_finish()) {
          poll_build();
        }
        rdma_comm.build_sync();
        diskann::cout << "Partition " << m << " meta Collected\n";

        while (!collect_id_finish()) {
          poll_build();
        }
        rdma_comm.build_sync();
        diskann::cout << "Partition " << m << " id Collected\n";

        while (!collect_ngh_finish()) {
          poll_build();
        }
        rdma_comm.build_sync();
        diskann::cout << "Partition " << m << " ngh Collected\n";

      } else {
        diskann::cout << "Dispatch partition " << m << " to machine" << m
                      << " ..." << std::endl;

        // send corresponding part to remote machine.
        auto *buf = rdma_comm.send_write_buf.getLocal();
        // send meta data first
        uint32_t buf_id = rdma_comm.get_write_send_buf(m);
        size_t len = serialize_dispatch_meta(buf->buffer[m][buf_id], m);
        rdma_comm.post_write_send(m, buf_id, len, DISPATCH_META);
        rdma_comm.build_sync();
        diskann::cout << "Partition " << m << " meta Dispatched\n";

        // send idmap
        size_t id_num = 0;
        while (id_num < index_param.dispatch_id_num[m]) {
          uint32_t buf_id = rdma_comm.get_write_send_buf(m);
          size_t len =
              serialize_dispatch_ids(buf->buffer[m][buf_id], id_num, m);
          rdma_comm.post_write_send(m, buf_id, len, DISPATCH_ID);
        }
        rdma_comm.build_sync();
        diskann::cout << "Partition " << m << " ids Dispatched\n";
        // send ngh
        size_t ngh_num = 0;
        while (ngh_num < index_param.dispatch_ngh_num[m]) {
          uint32_t buf_id = rdma_comm.get_write_send_buf(m);
          size_t len =
              serialize_dispatch_nghs(buf->buffer[m][buf_id], ngh_num, m);
          rdma_comm.post_write_send(m, buf_id, len, DISPATCH_NGH);
        }
        rdma_comm.build_sync();
        diskann::cout << "Partition " << m << " nghs Dispatched\n";
      }
      // Sync for this parition.
      // rdma_comm.build_sync();
    }
    return;
  }

  void poll_build() {
    auto *rbuf = rdma_comm.recv_write_buf.getLocal();  // recv buffer

    rdma_comm.poll_send();
    rdma_comm.poll_recv();

    // Collect recved meta.
    while (rbuf->dispatch_meta_queue.size() > 0) {
      char *meta_ptr = rbuf->dispatch_meta_queue.front();
      rbuf->dispatch_meta_queue.pop_front();
      proc_dispatch_meta(meta_ptr);
      free(meta_ptr);
    }

    // Collect recved ids.
    while (rbuf->dispatch_id_queue.size() > 0) {
      char *id_ptr = rbuf->dispatch_id_queue.front();
      rbuf->dispatch_id_queue.pop_front();
      proc_dispatch_id(id_ptr);
      free(id_ptr);
    }

    // Collect recved nghs.
    while (rbuf->dispatch_ngh_queue.size() > 0) {
      char *ngh_ptr = rbuf->dispatch_ngh_queue.front();
      rbuf->dispatch_ngh_queue.pop_front();
      proc_dispatch_ngh(ngh_ptr);
      free(ngh_ptr);
    }

    return;
  }

  void collect_local_ids_nghs() {
    // copy local ids.
    // index_param.collect_id_num[index_param.machine_id] =
    //     index_param.local_replica_ids.size();
    // uint32_t id_num = index_param.dispatch_id_num[index_param.machine_id];
    // index_param.collected_ids[index_param.machine_id] =
    //     (char *)malloc(sizeof(uint32_t) * id_num);

    // NOTE: dont double free here.
    index_param.collect_id_num[index_param.machine_id] =
        index_param.dispatch_id_num[index_param.machine_id];
    index_param.collect_ngh_num[index_param.machine_id] =
        index_param.dispatch_ngh_num[index_param.machine_id];

    index_param.collected_ids[index_param.machine_id] =
        index_param.dispatch_ids[index_param.machine_id];
    index_param.collected_nghs[index_param.machine_id] =
        index_param.dispatch_nghs[index_param.machine_id];

    return;
  }

  /**
   * Inform other machine the meta data of local corresponding partition.
   * machine_id: remote machine id (send partition id).
   */
  size_t serialize_dispatch_meta(char *ptr, uint32_t machine_id) {
    size_t offset = 0;
    // local machine id
    memcpy(ptr + offset, &index_param.machine_id, sizeof(uint32_t));
    offset += sizeof(uint32_t);
    // id num
    size_t id_num = index_param.dispatch_id_num[machine_id];
    memcpy(ptr + offset, &id_num, sizeof(size_t));
    offset += sizeof(size_t);
    // ngh num
    size_t ngh_num = index_param.dispatch_ngh_num[machine_id];
    memcpy(ptr + offset, &ngh_num, sizeof(size_t));
    offset += sizeof(size_t);

    printf(
        "send dispatch info to m%u idnum %u nghnum %u\n", machine_id, id_num,
        ngh_num);

    return offset;
  }

  void proc_dispatch_meta(char *ptr) {
    size_t offset = 0;
    uint32_t machine_id;
    memcpy(&machine_id, ptr + offset, sizeof(uint32_t));
    offset += sizeof(uint32_t);
    memcpy(
        &index_param.collect_id_num[machine_id], ptr + offset,
        sizeof(size_t));
    offset += sizeof(size_t);
    memcpy(
        &index_param.collect_ngh_num[machine_id], ptr + offset,
        sizeof(size_t));
    offset += sizeof(size_t);

    printf(
        "recv collect info from m%u idnum %llu nghnum %llu\n", machine_id,
        index_param.collect_id_num[machine_id],
        index_param.collect_ngh_num[machine_id]);
    index_param.recv_meta[machine_id] = true;
    index_param.collect_id_cnt[machine_id] = 0;  // reset count
    index_param.collect_ngh_cnt[machine_id] = 0;
    // alloc memory for part id and ngh.
    index_param.collected_ids[machine_id] = (char *)malloc(
        sizeof(uint32_t) * index_param.collect_id_num[machine_id]);
    index_param.collected_nghs[machine_id] = (char *)malloc(
        sizeof(uint32_t) * index_param.collect_ngh_num[machine_id]);
    return;
  }

  bool collect_meta_finish() {
    // local has collected
    index_param.recv_meta[index_param.machine_id] = true;
    for (uint32_t m = 0; m < MACHINE_NUM; m++) {
      if (!index_param.recv_meta[m]) {
        return false;
      }
    }
    std::cout << "expect partition id size: \n";
    for (uint32_t m = 0; m < MACHINE_NUM; m++) {
      std::cout << "m " << m << ":" << index_param.collect_id_num[m] << "\n";
    }
    std::cout << "expect partition ngh size: \n";
    for (uint32_t m = 0; m < MACHINE_NUM; m++) {
      std::cout << "m " << m << ":" << index_param.collect_ngh_num[m] << "\n";
    }
    return true;
  }

  size_t serialize_dispatch_ids(
      char *ptr, size_t &id_num, uint32_t machine_id) {
    size_t offset = 0;
    memcpy(ptr + offset, &index_param.machine_id, sizeof(uint32_t));
    offset += sizeof(uint32_t);
    size_t max_trans_num = std::min(
        (size_t)((MAX_QUERYBUFFER_SIZE - offset - sizeof(size_t)) /
                   sizeof(uint32_t)),
        index_param.dispatch_id_num[machine_id] - id_num);
    memcpy(ptr + offset, &max_trans_num, sizeof(size_t));
    offset += sizeof(size_t);

    memcpy(
        ptr + offset,
        index_param.dispatch_ids[machine_id] +
            (size_t)id_num * sizeof(uint32_t),
        max_trans_num * sizeof(uint32_t));
    id_num += max_trans_num;
    // printf(
    //     "]] dispatch m%d id num+ %d curcnt: %d expectcnt: %d\n", machine_id,
    //     max_trans_num, id_num, index_param.dispatch_id_num[machine_id]);
    offset += max_trans_num * sizeof(uint32_t);
    return offset;
  }

  void proc_dispatch_id(char *ptr) {
    size_t offset = 0;
    uint32_t machine_id;
    memcpy(&machine_id, ptr + offset, sizeof(uint32_t));
    offset += sizeof(uint32_t);
    size_t trans_num;
    memcpy(&trans_num, ptr + offset, sizeof(size_t));
    offset += sizeof(size_t);
    memcpy(
        index_param.collected_ids[machine_id] +
            (size_t)index_param.collect_id_cnt[machine_id] * sizeof(uint32_t),
        ptr + offset, trans_num * sizeof(uint32_t));
    index_param.collect_id_cnt[machine_id] += trans_num;
    // printf(
    //     "]] collected m%d id num+ %d curcnt: %d expectcnt: %d\n", machine_id,
    //     trans_num, index_param.collect_id_cnt[machine_id],
    //     index_param.collect_id_num[machine_id]);
    offset += trans_num * sizeof(uint32_t);
    return;
  }

  size_t serialize_dispatch_nghs(
      char *ptr, size_t &ngh_num, uint32_t machine_id) {
    size_t offset = 0;
    memcpy(ptr + offset, &index_param.machine_id, sizeof(uint32_t));
    offset += sizeof(uint32_t);
    size_t max_trans_num = std::min(
        (size_t)((MAX_QUERYBUFFER_SIZE - offset - sizeof(size_t)) /
                   sizeof(uint32_t)),
        index_param.dispatch_ngh_num[machine_id] - ngh_num);
    memcpy(ptr + offset, &max_trans_num, sizeof(size_t));
    offset += sizeof(size_t);
    memcpy(
        ptr + offset,
        index_param.dispatch_nghs[machine_id] +
            (size_t)ngh_num * sizeof(uint32_t),
        max_trans_num * sizeof(uint32_t));
    ngh_num += max_trans_num;
    offset += max_trans_num * sizeof(uint32_t);
    return offset;
  }

  void proc_dispatch_ngh(char *ptr) {
    size_t offset = 0;
    uint32_t machine_id;
    memcpy(&machine_id, ptr + offset, sizeof(uint32_t));
    offset += sizeof(uint32_t);
    size_t trans_num;
    memcpy(&trans_num, ptr + offset, sizeof(size_t));
    offset += sizeof(size_t);
    memcpy(
        index_param.collected_nghs[machine_id] +
            index_param.collect_ngh_cnt[machine_id] * sizeof(uint32_t),
        ptr + offset, trans_num * sizeof(uint32_t));
    index_param.collect_ngh_cnt[machine_id] += trans_num;
    offset += trans_num * sizeof(uint32_t);
    return;
  }

  bool collect_id_finish() {
    for (uint32_t m = 0; m < MACHINE_NUM; m++) {
      if (m == index_param.machine_id) continue;
      if (index_param.collect_id_cnt[m] < index_param.collect_id_num[m]) {
        return false;
      }
    }
    return true;
  }

  bool collect_ngh_finish() {
    for (uint32_t m = 0; m < MACHINE_NUM; m++) {
      if (m == index_param.machine_id) continue;
      if (index_param.collect_ngh_cnt[m] < index_param.collect_ngh_num[m]) {
        return false;
      }
    }
    return true;
  }

  void clear_finish() {
    for (uint32_t m = 0; m < MACHINE_NUM; m++) {
      index_param.collect_id_cnt[m] = index_param.collect_ngh_cnt[m] = 0;
      index_param.recv_meta[m] = false;
    }
    return;
  }

  void transfer_to_b1graph_format() {
    std::cout<<"Start transfer index to b1graph.\n";
    size_t read_blk_size = 64 * 1024 * 1024;
    if(!std::filesystem::exists(index_param.final_index_path)){
      std::cerr<< "Error: final_index_path does not exist.\n";
      abort();
    }

    cached_ifstream reader(index_param.final_index_path, read_blk_size);
    size_t fsize = reader.get_file_size();
    if (fsize == 0) {
      std::stringstream stream;
      stream << "Error, new file specified is empty. Not appending. "
             << std::endl;
      throw diskann::ANNException(stream.str(), -1);
    }

    char *final_idx_ptr = (char *)malloc(fsize);
    size_t num_blocks = DIV_ROUND_UP(fsize, read_blk_size);
    for (uint64_t i = 0; i < num_blocks; i++) {
      size_t cur_block_size = read_blk_size > fsize - (i * read_blk_size)
                                  ? fsize - (i * read_blk_size)
                                  : read_blk_size;
      reader.read(final_idx_ptr + i * read_blk_size, cur_block_size);
    }

    // merge index with remap id
    /*
      b1graph element size:
      deg + deg*ngh + vector + label

      b1_graph format:
      <vec_num> <start>
      vec_num * <ele_size>
    */
    uint64_t vec_num = index_param.partition_size[index_param.machine_id];
    // deg + deg*ngh + vector + label
    uint64_t b1_ele_size = sizeof(uint32_t) +
                           index_param.max_degree * sizeof(uint32_t) +
                           index_param.vec_size + sizeof(uint64_t);
    uint64_t merge_graph_size =
        sizeof(uint64_t) + sizeof(index_param.medoid) + b1_ele_size * vec_num;

    size_t merge_meta_offset = 0;
    char *merge_graph = (char *)malloc(merge_graph_size);
    if(merge_graph == nullptr){
      std::cerr << "Error: failed to malloc b1graph mem.\n";
      abort();
    }

    memcpy(merge_graph + merge_meta_offset, &vec_num, sizeof(vec_num));
    merge_meta_offset += sizeof(vec_num);

    memcpy(
        merge_graph + merge_meta_offset, &index_param.medoid,
        sizeof(index_param.medoid));
    merge_meta_offset += sizeof(index_param.medoid);
    printf(
        "b1graph vnum: %u medoid %u size %llu\n", vec_num, index_param.medoid,
        merge_graph_size);

    std::ifstream base_reader(
        index_param.local_noreplica_base_file.c_str(), std::ios::binary);
    std::cout << "read base data from: "
              << index_param.local_noreplica_base_file << std::endl;
    uint32_t dummy_size, basedim32;
    base_reader.read((char *)&dummy_size, sizeof(uint32_t));
    base_reader.read((char *)&basedim32, sizeof(uint32_t));

    size_t offset = 0;
    size_t index_size;
    memcpy(&index_size, final_idx_ptr + offset, sizeof(index_size));
    offset += sizeof(index_size);
    uint32_t max_deg;
    memcpy(&max_deg, final_idx_ptr + offset, sizeof(max_deg));
    offset += sizeof(max_deg);
    if(max_deg != index_param.max_degree){
      std::cerr<<"Error: Mismatch of max degree.\n";
      abort();
    }
    uint32_t medoid;
    memcpy(&medoid, final_idx_ptr + offset, sizeof(medoid));
    offset += sizeof(medoid);

    // set interpoint to partition interpoint.
    index_param.medoid = medoid;

    uint64_t index_frozen;
    memcpy(&index_frozen, final_idx_ptr + offset, sizeof(index_frozen));
    offset += sizeof(index_frozen);

    uint32_t *ngh_ptr = (uint32_t *)(final_idx_ptr + offset);
    size_t ngh_offset = 0;
    size_t ele_num = 0;
    size_t partition_ofs = 0;
    for (uint32_t m = 0; m < index_param.machine_id; m++) {
      partition_ofs += index_param.partition_size[m];
    }
    std::cout << "partition_ofs: " << partition_ofs << std::endl;
    while (offset + ngh_offset * sizeof(uint32_t) < index_size) {
      char *ele_ptr =
          (char *)(merge_graph + merge_meta_offset + ele_num * b1_ele_size);
      size_t ele_ofs = 0;
      // write degree
      uint32_t degree = ngh_ptr[ngh_offset];
      uint32_t remap_ngh[degree + 1];
      remap_ngh[0] = degree;
      for (uint32_t i = 1; i < degree + 1; i++) {
        remap_ngh[i] = index_param.label2pos_map[ngh_ptr[ngh_offset + i]];
      }
      memcpy(
          ele_ptr + ele_ofs, (char *)remap_ngh,
          (degree + 1) * sizeof(uint32_t));
      // NOTE: fixed deg in b1.
      ele_ofs += (index_param.max_degree + 1) * sizeof(uint32_t);
      // write vector
      base_reader.read(ele_ptr + ele_ofs, index_param.vec_size);
      ele_ofs += index_param.vec_size;
      // write label
      size_t pos = partition_ofs + ele_num;
      size_t label = index_param.pos2label_map[pos];
      memcpy(ele_ptr + ele_ofs, (char *)&label, sizeof(size_t));
      ngh_offset += degree + 1;
      ele_num++;
    }
    std::cout << "max deg: " << max_deg << " medoid: " << medoid
              << " index frozen: " << index_frozen << std::endl;
    std::cout << ele_num << " nodes read.\n";

    std::cout << "start write index\n";
    // write index
    cached_ofstream output(
        index_param.final_b1_index_path, BUFFER_SIZE_FOR_CACHED_IO);
    // write merge graph in block
    size_t write_blk_size = 64 * 1024 * 1024;
    num_blocks = DIV_ROUND_UP(merge_graph_size, write_blk_size);
    for (uint64_t i = 0; i < num_blocks; i++) {
      size_t cur_block_size =
          write_blk_size > merge_graph_size - (i * write_blk_size)
              ? merge_graph_size - (i * write_blk_size)
              : write_blk_size;
      output.write(merge_graph + i * write_blk_size, cur_block_size);
      sleep(2);
    }
    std::cout << " b1 graph vnum num:" << ele_num << "\n";
    std::cout << "Transfer index to b1graph over: "
              << index_param.final_b1_index_path << "\n";
  }

  /**
   * Merge distributed b1 index to get vamana graph to verify the
   * correctness.
   */
  void merge_all_b1_index() {
    cached_ofstream output(
        index_param.temp_b1_index, BUFFER_SIZE_FOR_CACHED_IO);
    uint64_t vec_num = index_param.max_vid;
    output.write((char *)&vec_num, sizeof(vec_num));
    output.write((char *)&index_param.medoid, sizeof(index_param.medoid));
    uint64_t b1_ele_size = sizeof(uint32_t) +
                           index_param.max_degree * sizeof(uint32_t) +
                           index_param.vec_size + sizeof(uint64_t);
    size_t ele_num = 0;
    for (uint32_t m = 0; m < MACHINE_NUM; m++) {
      size_t read_blk_size = 64 * 1024 * 1024;
      cached_ifstream reader(
          index_param.final_part_b1_index_path[m], read_blk_size);
      size_t fsize = reader.get_file_size();
      if (fsize == 0) {
        std::stringstream stream;
        stream << "Error, new file specified is empty. Not appending. "
               << std::endl;
        throw diskann::ANNException(stream.str(), -1);
      }
      size_t part_vec_num;
      reader.read((char *)&part_vec_num, sizeof(part_vec_num));
      ele_num += part_vec_num;
      uint32_t part_medoid;
      reader.read((char *)&part_medoid, sizeof(part_medoid));
      size_t part_vec_size = part_vec_num * b1_ele_size;
      printf(
          "part b1graph vnum: %u medoid %u size %llu\n", part_vec_num,
          part_medoid, part_vec_size);
      size_t delta = fsize - part_vec_size;
      printf("delta: %lu, expect equal to 12..\n", delta);

      char *b1_idx_ptr = (char *)malloc(read_blk_size);
      size_t num_blocks = DIV_ROUND_UP(part_vec_size, read_blk_size);
      for (uint64_t i = 0; i < num_blocks; i++) {
        size_t cur_block_size =
            read_blk_size > part_vec_size - (i * read_blk_size)
                ? part_vec_size - (i * read_blk_size)
                : read_blk_size;
        reader.read(b1_idx_ptr, cur_block_size);
        output.write(b1_idx_ptr, cur_block_size);
        sleep(2);
      }
    }
    std::cout << " b1 graph vnum num:" << ele_num << "\n";
    std::cout << "Merge index to b1graph over: " << index_param.temp_b1_index
              << "\n";

    return;
  }

  void transfer_to_scala_v3();
  void verify_scala_v3();

  void topindex_checkpoint() {
    // update index param about partition meta info.
    // show replica ids info.
    std::cout << "======== TopIndex Checkpoint BEGIN ========\n";
    // std::ifstream base_reader(index_param.data_path, std::ios::binary);
    // base_reader.read((char *)&index_param.max_vid, sizeof(uint32_t));
    // base_reader.read((char *)&index_param.dim, sizeof(uint32_t));
    // base_reader.close();
    std::cout << "max vid: " << index_param.max_vid
              << "dim: " << index_param.dim << "\n";
    std::cout << "pos label map file: " << index_param.temp_pos2label_map_file
              << std::endl;
    std::cout << "label pos map file: " << index_param.temp_label2pos_map_file
              << std::endl;
    std::cout << "======== TopIndex Checkpoint END ========\n";
  }

  /*Delete temp index file. */
  void delete_temp_file() {
    // delete tempFiles
    std::remove(index_param.local_replica_base_file.c_str());
    std::remove(index_param.local_replica_idmap_filename.c_str());
    std::remove(index_param.local_noreplica_base_file.c_str());
    std::remove(index_param.local_noreplica_idmap_filename.c_str());
  }

  /**
  * for baseline2 checkpoint.
  */
  void shared_nothing_checkpoint(){
    index_param.num_parts = MACHINE_NUM;
    partition_checkpoint();
    std::cout<<"==== Checkpoint Partition Size ====\n";
    for (uint32_t m = 0; m < MACHINE_NUM; m++) { 
      std::cout<<index_param.partition_size[m]<<" ";
    }
    std::cout<<"\n";
  }

/**
 * Used to trans local partition vamana graph to b2graph. 
*/
int trans_to_b2graph() {
  size_t partition_ofs = 0;
  for (uint32_t m = 0; m < index_param.machine_id; m++) {
    partition_ofs += index_param.partition_size[m];
  }
  std::cout<<"Local partition ofs: "<< partition_ofs << "\n";

  // Read ID maps
  if(!std::filesystem::exists(index_param.local_noreplica_idmap_filename)){
    std::cout<<"Error: local_replica_idmap_filename not exist.\n";
    abort();
  }
  // diskann::read_idmap(index_param.local_noreplica_idmap_filename, index_param.local_noreplica_ids);

  // find max node id
  size_t nelems = index_param.local_noreplica_ids.size();
  for (auto &id : index_param.local_noreplica_ids) {
    index_param.local_max_vid = std::max(index_param.local_max_vid, (size_t)id);
  }
  diskann::cout << "# max id: " << index_param.local_max_vid 
                << ", max. degree: " << index_param.max_degree
                << ", # nodeselems(local b2 graph): " << nelems << std::endl;

  // create cached vamana readers
  // expected file size + max degree + medoid_id + frozen_point info
  if(!std::filesystem::exists(index_param.local_index_file)){
    std::cout<<"Error: local_index_file not exist.\n";
    abort();
  }

  std::ifstream vamana_reader(index_param.local_index_file.c_str(), std::ios::binary);
  size_t expected_file_size;
  vamana_reader.read((char *)&expected_file_size, sizeof(uint64_t));
  // read width from each vamana to advance buffer by sizeof(uint32_t) bytes
  uint32_t input_width;
  vamana_reader.read((char *)&input_width, sizeof(uint32_t));
  uint32_t medoid;
  // read medoid
  vamana_reader.read((char *)&medoid, sizeof(uint32_t));
  uint64_t vamana_index_frozen;
  vamana_reader.read((char *)&vamana_index_frozen, sizeof(uint64_t));
  // rename medoid
  // index_param.medoid = index_param.local_noreplica_ids[medoid];
  index_param.medoid = medoid;


  uint64_t vec_num = index_param.partition_size[index_param.machine_id];
  if(vec_num != nelems){
    std::cerr<<"Error: b2 graph vector num mismatch read nelems: " \
      << nelems << " expected num: "<< vec_num << "\n";
    abort();
  }
  // deg + deg*ngh + vector + label
  uint64_t b2_ele_size = sizeof(uint32_t) +
                          index_param.max_degree * sizeof(uint32_t) +
                          index_param.vec_size + sizeof(uint64_t);
  uint64_t b2_graph_size =
      sizeof(uint64_t) + sizeof(index_param.medoid) + b2_ele_size * vec_num;
  
  diskann::cout<< "b2_ele_size " << b2_ele_size << " b2_graph_size "<< b2_graph_size << "\n";

  size_t b2_meta_offset = 0;
  char *b2_graph = (char *)malloc(b2_graph_size);
  memcpy(b2_graph + b2_meta_offset, &vec_num, sizeof(vec_num));
  b2_meta_offset += sizeof(vec_num);
  // uint32_t local_medoid = index_param.medoid - partition_ofs;
  uint32_t local_medoid = index_param.medoid;
  
  memcpy(
      b2_graph + b2_meta_offset, &local_medoid,
      sizeof(local_medoid));
      b2_meta_offset += sizeof(local_medoid);
  printf(
      "b2graph vnum: %u medoid %u size %llu\n", vec_num, local_medoid,
      b2_graph_size);
  
  if(!std::filesystem::exists(index_param.local_noreplica_base_file)){
    std::cout<<"Error: local_noreplica_base_file not exist.\n";
    std::cout<<index_param.local_noreplica_base_file<<"\n";
    abort();
  }
  
  // should not original data path, should be reorder path.
  std::ifstream base_reader(
    index_param.local_noreplica_base_file.c_str(), std::ios::binary);
  
  std::cout << "read base data from: "
            << index_param.local_noreplica_base_file << std::endl;
  // size_t base_ofs = sizeof(uint32_t) + sizeof(uint32_t) + index_param.vec_size * partition_ofs;
  size_t base_ofs = sizeof(uint32_t) + sizeof(uint32_t);
  base_reader.seekg(base_ofs, std::ios::beg);

  diskann::cout << "Starting transform to b2graph format ..." << std::endl;
  size_t internal_id = 0;
  for (uint32_t &node_id : index_param.local_noreplica_ids) {
    // size_t internal_id = node_id - partition_ofs;
    size_t ele_ofs = 0;
    char *ele_ptr =
        (char *)(b2_graph + b2_meta_offset + internal_id * b2_ele_size);
    // read from shard_id ifstream
    uint32_t degree = 0;
    vamana_reader.read((char *)&degree, sizeof(uint32_t));

    if (degree == 0 || degree > index_param.max_degree) {
      diskann::cout << "WARNING: shard #" << index_param.machine_id << ", node_id " << node_id
                    << " has 0 nbrs" << std::endl;
      abort();
    }

    std::vector<uint32_t> shard_nhood(degree);
    if (degree > 0)
      vamana_reader.read(
          (char *)shard_nhood.data(), degree * sizeof(uint32_t));
    uint32_t remap_ngh[degree + 1];
    remap_ngh[0] = degree;
    for (uint32_t i = 0; i < degree; i++) {
      // here in each partition, the ngh id is the local id(offset) in partitin.
      if(shard_nhood[i] >= index_param.local_noreplica_ids.size()){
        std::cerr << "Error: ngh" << shard_nhood[i] << "exceed the local ids range :"
          << index_param.local_noreplica_ids.size()<< " \n";
        abort();
      }
      // remap_ngh[i + 1] = index_param.local_noreplica_ids[shard_nhood[i]] - partition_ofs;
      remap_ngh[i + 1] = shard_nhood[i];
    }
    memcpy(
        ele_ptr + ele_ofs, (char *)remap_ngh,
        (degree + 1) * sizeof(uint32_t));
    // NOTE: fixed deg in b1.
    ele_ofs += (index_param.max_degree + 1) * sizeof(uint32_t);
    
    // write vector
    base_reader.read(ele_ptr + ele_ofs, index_param.vec_size);
    ele_ofs += index_param.vec_size;
    // write label
    size_t label = node_id;
    memcpy(ele_ptr + ele_ofs, (char *)&label, sizeof(size_t));
    if (internal_id % 50000 == 0) {
      diskann::cout << (float)internal_id*100.0/vec_num 
        <<"%\n"<< std::flush;
    }
    internal_id ++;
  }

  diskann::cout << "\nExpected size: " << b2_graph_size << std::endl;
  diskann::cout << "Finished merge" << std::endl;

  std::cout << "start write index\n";
  // write index
  std::ofstream b2_writer(index_param.local_b2_index_file.c_str(), std::ios::binary);
  // write merge graph in block
  size_t write_blk_size = 64 * 1024 * 1024;
  size_t num_blocks = DIV_ROUND_UP(b2_graph_size, write_blk_size);
  for (uint64_t i = 0; i < num_blocks; i++) {
    size_t cur_block_size =
        write_blk_size > b2_graph_size - (i * write_blk_size)
            ? b2_graph_size - (i * write_blk_size)
            : write_blk_size;
    b2_writer.write(b2_graph + i * write_blk_size, cur_block_size);
    sleep(2);
  }
  std::cout << "b2 graph vnum num:" << vec_num << "\n";
  std::cout << "Transfer index to b2graph over: "
            << index_param.local_b2_index_file << "\n";
  b2_writer.close();  

  free(b2_graph);
  
  // remove temp file
  // std::remove(index_param.local_replica_base_file.c_str());
  // std::remove(index_param.local_replica_idmap_filename.c_str());
  // std::remove(index_param.local_index_file.c_str());

  return 0;  
}

};
