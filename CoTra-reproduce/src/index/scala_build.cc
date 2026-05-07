#include "index/scala_build.h"

#include <filesystem>

/**
 * Transfer b1graph to scala graph V3 format.
 * [ScalaGraph V3 index format]:
 * <partition_id: uint32_t>, <partition_num: uint32_t>,
 * <part_vec_num: uint32_t>, <local_ngh_ele_size: uint32_t>,
 * <enter_point: uint64_t>,
 * <remote_ngh_num: uint64_t>
 * (header 32 bytes)
 * <local_ngh_array: local_ngh_ele_size> * part_vec_num,
 * <remote_ngh_array: uint32_t> * remote_ngh_num.
 */
void ScalaBuild::transfer_to_scala_v3() {
  std::cout<< "Start transfer b1graph to scalagraph (V3)..\n";
  uint32_t cumupsize[index_param.num_parts + 1];
  memset(cumupsize, 0, sizeof(cumupsize));
  for (int p = 1; p < index_param.num_parts + 1; p++) {
    cumupsize[p] = cumupsize[p - 1] + index_param.partition_size[p - 1];
    std::cout << "cumupsize[" << p << "] = " << cumupsize[p] << std::endl;
  }
  index_param.local_partition_size =
      index_param.partition_size[index_param.machine_id];

  auto judge_partition = [&](uint64_t vid) {
    for (int p = 0; p < index_param.num_parts; p++) {
      if (cumupsize[p + 1] > vid) return p;
    }
    return -1;
  };

  // read local b1 index
  std::cout << "Loading kmeans b1graph index from "
            << index_param.final_b1_index_path << "...\n";
  std::ifstream input(
      index_param.final_b1_index_path.c_str(), std::ios::binary);
  if (!input.is_open()) throw std::runtime_error("Cannot open file");

  size_t read_max_elements_;
  uint64_t read_enterpoint_node_;
  input.read((char *)&read_max_elements_, sizeof(uint64_t));
  input.read((char *)&read_enterpoint_node_, sizeof(uint32_t));
  index_param.medoid = read_enterpoint_node_;
  if (read_max_elements_ != index_param.local_partition_size) {
    std::cout << "read_max_elements_ " << read_max_elements_
              << " partition_size " << index_param.local_partition_size
              << std::endl;
    std::cout << " medoid " << index_param.medoid << std::endl;
    std::cout << "Error: partition elements num not matched.\n";
    abort();
  }

  uint64_t size_data_per_element_ = sizeof(uint32_t) +
                                    index_param.max_degree * sizeof(uint32_t) +
                                    index_param.vec_size + sizeof(uint64_t);
  size_t label_offset_ = sizeof(uint32_t) +
                         index_param.max_degree * sizeof(uint32_t) +
                         index_param.vec_size;
  size_t offsetData_ =
      sizeof(uint32_t) + index_param.max_degree * sizeof(uint32_t);

  char *data_level0_memory_ =
      (char *)malloc(index_param.local_partition_size * size_data_per_element_);
  if (data_level0_memory_ == nullptr)
    throw std::runtime_error(
        "Not enough memory: loadIndex failed to allocate level0");
  input.read(
      data_level0_memory_,
      index_param.local_partition_size * size_data_per_element_);

  std::cout << "Kmeans & Index file read over, parition num:"
            << index_param.num_parts << " parition size:" << read_max_elements_
            << "size data per element:" << size_data_per_element_ << "\n";

  /*
    [index format]:
    <partition_id: uint32_t>, <partition_num: uint32_t>,
    <part_vec_num: uint32_t>, <local_ngh_ele_size: uint32_t>,
    <enter_point: uint64_t>,
    <remote_ngh_num: uint64_t>
    (header 32 bytes)
    <local_ngh_array: local_ngh_ele_size> * part_vec_num,
    <remote_ngh_array: uint32_t> * remote_ngh_num.
  */
  std::cout << ">>> Step 0. Start compute scala graph partition size ...\n";
  // Step1: Compute each partition size.
  size_t index_meta_len = 4 * sizeof(uint32_t) + 2 * sizeof(uint64_t);
  /**
   * each ngh max store:
   * max_deg * ngh + partition_deg_msg * part_num
   */
  size_t max_deg = index_param.max_degree;
  size_t local_ngh_ele_size =
      max_deg * sizeof(uint32_t) + index_param.num_parts * sizeof(uint16_t);

  // local ngh array size and remote ngh array num.
  // final index size[p] = meta_size + local_ngh_size[p] +
  // remote_ngh_num[p]*4.
  // here out_remote_ngh_num is different from the remote_ngh_num in single
  // machine version, this is use to record the number of neighbor of this
  // partition that will send to each machine. (the remote_ngh_num is to record
  // all remote neighbor that will send to this partition).
  size_t local_ngh_size;
  std::vector<size_t> out_remote_ngh_num(index_param.num_parts, 0);
  size_t scala_data_size;

  local_ngh_size =
      (size_t)(index_param.local_partition_size * local_ngh_ele_size);

  for (uint64_t vid = 0; vid < index_param.local_partition_size; vid++) {
    uint32_t vid_pid = index_param.machine_id;
    uint32_t *ptr =
        (uint32_t *)(data_level0_memory_ + vid * size_data_per_element_);
    uint32_t cnt[index_param.num_parts];  // ngh distribute count.
    memset(cnt, 0, sizeof(cnt));
    for (int i = 1; i < ptr[0] + 1; i++) {
      cnt[judge_partition(ptr[i])]++;
    }
    for (int p = 0; p < index_param.num_parts; p++) {
      // get all remote ngh.
      if (cnt[p] > INLINE_DEG_BAR && p != vid_pid) {
        // case: > INLINE_DEG_BAR, need to store to remote.
        out_remote_ngh_num[p] += cnt[p];
      }
    }
  }

  /*******
   * Step 1. Swap paritiona info.
   */
  // record the number of remote ngh will recv form each machine.
  // in_remote_ngh_num[p][q]: the number of neighbor of partition p
  // send to machine q.
  std::cout << ">>> Step 1. Start swap partition info ... \n";
  std::vector<std::vector<size_t>> in_remote_ngh_num(index_param.num_parts);
  in_remote_ngh_num[index_param.machine_id] =
      std::vector<size_t>(index_param.num_parts, 0);
  rdma_comm.swap_scala_index_meta_info(
      index_param.machine_id, out_remote_ngh_num, in_remote_ngh_num);
  size_t all_in_remote_ngh_num = 0;

  for (int i = 0; i < index_param.num_parts; i++) {
    if (i == index_param.machine_id) {
      if (in_remote_ngh_num[i][i] != 0) {
        std::cout << "Error: in_remote_ngh_num[" << index_param.machine_id
                  << "] is not 0.\n";
        abort();
      }
      continue;
    }
    all_in_remote_ngh_num += in_remote_ngh_num[i][index_param.machine_id];
    std::cout << "in_remote_ngh_num[" << i << "][" << index_param.machine_id
              << "] = " << in_remote_ngh_num[i][index_param.machine_id] << "\n";
  }
  std::cout << "all_in_remote_ngh_num: " << all_in_remote_ngh_num << std::endl;

  size_t expect_index_size;
  size_t expect_data_size;
  expect_index_size = index_meta_len + local_ngh_size +
                      all_in_remote_ngh_num * sizeof(uint32_t);
  expect_data_size = scala_data_size;
  printf(
      "EXPECT scala_index_size: %llu, scala_data_size: %llu\n",
      expect_index_size, expect_data_size);

  /*******
   * Step 2. Recompute ngh partition ofs.
   */
  std::cout << ">>> Step 2. Recompute ngh partition ofs ... \n";
  size_t out_remote_ngh_start_ofs[index_param.num_parts];
  for (uint32_t p = 0; p < index_param.num_parts; p++) {
    // out_remote_ngh_start_ofs[p] is the start offset of remote ngh of
    // partition p.
    out_remote_ngh_start_ofs[p] =
        index_param.partition_size[p] * local_ngh_ele_size;
    for (uint32_t q = 0; q < index_param.machine_id; q++) {
      out_remote_ngh_start_ofs[p] += in_remote_ngh_num[q][p] * sizeof(uint32_t);
    }
    std::cout << "out_remote_ngh_start_ofs[" << p
              << "] = " << out_remote_ngh_start_ofs[p] << "\n";
  }
  // NOTE: partition[p] out_remote_ngh_start_ofs[q] should be equal to
  // partition[q] in_remote_ngh_start_ofs[p] - index_meta_len.

  // in_remote_ngh_start_ofs is the actual pointer offset, so need the meta len.
  size_t in_remote_ngh_start_ofs[index_param.num_parts];
  in_remote_ngh_start_ofs[0] =
      index_meta_len + index_param.local_partition_size * local_ngh_ele_size;
  std::cout << "in_remote_ngh_start_ofs[0] = " << in_remote_ngh_start_ofs[0]
            << "\n";
  for (uint32_t q = 1; q < index_param.num_parts; q++) {
    in_remote_ngh_start_ofs[q] =
        in_remote_ngh_start_ofs[q - 1] +
        in_remote_ngh_num[q - 1][index_param.machine_id] * sizeof(uint32_t);
    std::cout << "in_remote_ngh_start_ofs[" << q
              << "] = " << in_remote_ngh_start_ofs[q] << "\n";
  }

  /*******
   * Step 3. Reformat index and data.
   */
  std::cout << ">>> Step 3. Start reformat index and data ... \n";
  std::cout << "Rewrite index ...\n";
  char *index_ptr;
  // should declare new out remote ngh ptr to send here.
  std::vector<char *> out_remote_ngh_ptr(index_param.num_parts);
  for (int p = 0; p < index_param.num_parts; p++) {
    out_remote_ngh_ptr[p] =
        (char *)malloc(out_remote_ngh_num[p] * sizeof(uint32_t));
  }

  size_t local_ngh_ofs;
  // this record the local send buffer offsets.
  size_t out_remote_ngh_ofs[index_param.num_parts];
  index_ptr = (char *)malloc(expect_index_size);
  local_ngh_ofs = 0;
  memset(out_remote_ngh_ofs, 0, sizeof(out_remote_ngh_ofs));
  printf("Rewrite expect index size %llu ...\n", expect_index_size);
  /*
    [index format]:
    <partition_id: uint32_t>, <partition_num: uint32_t>,
    <part_vec_num: uint32_t>, <local_ngh_ele_size: uint32_t>,
    <enter_point: uint64_t>,
    <remote_ngh_num: uint64_t>
    (header 32 bytes)
    <local_ngh_array: local_ngh_ele_size> * part_vec_num,
    <remote_ngh_array: uint32_t> * remote_ngh_num.
  */
  memcpy(index_ptr + local_ngh_ofs, &index_param.machine_id, sizeof(uint32_t));
  local_ngh_ofs += sizeof(uint32_t);
  memcpy(index_ptr + local_ngh_ofs, &index_param.num_parts, sizeof(uint32_t));
  local_ngh_ofs += sizeof(uint32_t);
  memcpy(
      index_ptr + local_ngh_ofs, &index_param.local_partition_size,
      sizeof(uint32_t));
  local_ngh_ofs += sizeof(uint32_t);
  memcpy(index_ptr + local_ngh_ofs, &local_ngh_ele_size, sizeof(uint32_t));
  local_ngh_ofs += sizeof(uint32_t);
  size_t enterpoint_node_ = index_param.medoid;
  memcpy(index_ptr + local_ngh_ofs, &enterpoint_node_, sizeof(uint64_t));
  local_ngh_ofs += sizeof(uint64_t);
  memcpy(index_ptr + local_ngh_ofs, &all_in_remote_ngh_num, sizeof(uint64_t));
  local_ngh_ofs += sizeof(uint64_t);
  std::cout << "INFO: meta part num " << index_param.local_partition_size
            << " ngh_num " << all_in_remote_ngh_num << "\n";

  for (uint32_t vid = 0; vid < index_param.local_partition_size;
       vid++, local_ngh_ofs += local_ngh_ele_size) {
    uint32_t p = index_param.machine_id;
    if (vid % 1000000 == 0) {
      std::cout << (double)vid * 100.0 / index_param.local_partition_size
                << "%\r" << std::flush;
    }
    // p is source node partition id, q is target node partition id.
    size_t local_ofs = local_ngh_ofs;
    uint32_t *ptr =
        (uint32_t *)(data_level0_memory_ + vid * size_data_per_element_);
    std::vector<uint32_t> nghs[index_param.num_parts];
    for (int i = 1; i < ptr[0] + 1; i++) {
      int q = judge_partition(ptr[i]);
      nghs[q].emplace_back(ptr[i]);
    }
    std::vector<uint32_t> part_ids;
    for (int q = 0; q < index_param.num_parts; q++) {
      if (nghs[q].size() > 0 && (q != p)) {
        part_ids.push_back(q);
      }
    }
    // put the local ngh to the last
    if (nghs[p].size() > 0) {
      part_ids.push_back(p);
    }

    for (uint32_t i = 0; i < part_ids.size(); i++) {
      uint32_t end_flag = 0;
      // q is which partition this vector ngh located.
      uint32_t q = part_ids[i];
      if (i == part_ids.size() - 1) {
        end_flag = (1 << 15);
        if (q == p) {
          end_flag |= (1 << 14);
        }
      }

      uint16_t msg = end_flag | (q << 8) | nghs[q].size();

      // write local ngh
      memcpy(index_ptr + local_ofs, &msg, sizeof(msg));
      local_ofs += sizeof(msg);
      if (q != p) {
        // remote nghs
        if (nghs[q].size() <= INLINE_DEG_BAR) {
          for (auto &v : nghs[q]) {
            memcpy(index_ptr + local_ofs, &v, sizeof(v));
            local_ofs += sizeof(v);
          }
        } else {
          // remember sender shoule sub index_meta_len in advance.
          // The actual offset in remote machine.
          size_t relative_ofs =
              out_remote_ngh_start_ofs[q] + out_remote_ngh_ofs[q];
          memcpy(index_ptr + local_ofs, &relative_ofs, sizeof(relative_ofs));
          local_ofs += sizeof(relative_ofs);
          for (auto &v : nghs[q]) {
            memcpy(
                out_remote_ngh_ptr[q] + out_remote_ngh_ofs[q], &v, sizeof(v));
            out_remote_ngh_ofs[q] += sizeof(v);
          }
        }
      } else {
        // local nghs
        for (auto &v : nghs[q]) {
          memcpy(index_ptr + local_ofs, &v, sizeof(v));
          local_ofs += sizeof(v);
        }
      }
      if (local_ofs > local_ngh_ofs + local_ngh_ele_size) {
        printf(
            "Error: index exceed.. vid %d local_ofs %llu local_ngh_ofs "
            "%llu local_ngh_ele_size: %llu\n",
            vid, local_ofs, local_ngh_ofs, local_ngh_ele_size);
        abort();
      }
    }
  }

  /*******
   * Step 4. Send & Recv remote ngh.
   */
  std::cout << "Step 4. Send & Recv remote ngh ... \n";
  rdma_comm.swap_scalagraph_remote_ngh(
      out_remote_ngh_ptr, out_remote_ngh_num, index_ptr,
      in_remote_ngh_start_ofs, in_remote_ngh_num);

  // add remote ngh size
  for (int p = 0; p < index_param.num_parts; p++) {
    local_ngh_ofs +=
        in_remote_ngh_num[p][index_param.machine_id] * sizeof(uint32_t);
  }

  size_t actual_size = local_ngh_ofs;
  size_t expect_size = expect_index_size;
  printf(
      "part %d index size %llu expect %llu\n", index_param.machine_id,
      actual_size, expect_size);
  if (actual_size != expect_size) {
    printf(
        "Error: part %d index size %llu expect %llu\n", index_param.machine_id,
        actual_size, expect_size);
    abort();
  }

  /*******
   * Step 5. Write Scala V3 index.
   */
  std::cout << "Step 5. Write Scala V3 index ... \n";
  printf(
      "Write part %u index size %llu...\n", index_param.machine_id,
      expect_index_size);
  uint64_t write_size = 1LL << 27;

  std::ofstream index_output(
      index_param.final_scala_index_path.c_str(), std::ios::binary);
  for (uint64_t i = 0; i < expect_index_size; i += write_size) {
    if (i + write_size > expect_index_size) {
      write_size = (uint64_t)(expect_index_size - i);
    }
    index_output.write((char *)(index_ptr + i), write_size);
    sleep(4);
    std::cout << "write "
              << (float)(i + write_size) * 100.0 / (expect_index_size) << "%\r "
              << std::flush;
  }
  std::cout << "partition" << index_param.machine_id << "index file "
            << index_param.final_scala_index_path
            << " write over, expect file size " << expect_index_size << "\n";
  index_output.close();
  // clear index memory
  free(index_ptr);

  /*******
   * Step 6. Write Scala V3 data.
   */
  std::cout << "Step 6. Write Scala V3 data ... \n";
  /*
    <partition_id: uint32_t>, <partition_num: uint32_t>,
    <partition_vec_num: uint32_t>, <all_vec_num: uint64_t>,
    (header 20 bytes)
    <enterpoint vector: dim * type + label>
    <vector: dim * type + label> * partition_vec_num.
  */
  char *out_data;
  uint64_t data_ofs = 0;
  size_t data_meta_len = 3 * sizeof(uint32_t) + sizeof(uint64_t);
  scala_data_size =
      (size_t)(data_meta_len + (index_param.vec_size + sizeof(uint64_t)) *
                                   (index_param.local_partition_size + 1));
  out_data = (char *)malloc(scala_data_size);

  memcpy(out_data + data_ofs, &index_param.machine_id, sizeof(uint32_t));
  data_ofs += sizeof(uint32_t);
  memcpy(out_data + data_ofs, &index_param.num_parts, sizeof(uint32_t));
  data_ofs += sizeof(uint32_t);
  memcpy(
      out_data + data_ofs, &index_param.local_partition_size, sizeof(uint32_t));
  data_ofs += sizeof(uint32_t);
  size_t max_vid = index_param.max_vid;
  memcpy(out_data + data_ofs, &max_vid, sizeof(uint64_t));
  data_ofs += sizeof(uint64_t);
  // copy enterpoint, for scala V3 is useless.
  char *enter_ptr =
      (char *)(data_level0_memory_ + 0 * size_data_per_element_ + offsetData_);
  memcpy(
      out_data + data_ofs, enter_ptr, index_param.vec_size + sizeof(uint64_t));
  data_ofs += index_param.vec_size + sizeof(uint64_t);
  printf(
      "Rewrite part %d data, size %llu ...\n", index_param.machine_id,
      scala_data_size);
  for (uint64_t vid = 0; vid < index_param.local_partition_size; vid++) {
    char *ptr = (char *)(data_level0_memory_ + vid * size_data_per_element_ +
                         offsetData_);
    memcpy(out_data + data_ofs, ptr, index_param.vec_size + sizeof(uint64_t));
    data_ofs += index_param.vec_size + sizeof(uint64_t);
  }

  if(data_ofs != scala_data_size){
    std::cerr<<"Error: data offset not equal to expected.\n";
    abort();
  }
  printf(
      "Write part %d data size %llu ...\n", index_param.machine_id,
      scala_data_size);
  std::ofstream data_output(
      index_param.final_scala_data_path.c_str(), std::ios::binary);
  for (uint64_t i = 0; i < scala_data_size; i += write_size) {
    if (i + write_size > scala_data_size) {
      write_size = (uint64_t)(scala_data_size - i);
    }
    data_output.write((char *)(out_data + i), write_size);
    // for our nfs cluster ..
    if(index_param.machine_id == 7 || index_param.machine_id == 6){
      sleep(1);
    }
    else {
      sleep(4);
    }
    
    std::cout << "write " << (float)(i + write_size) * 100.0 / (scala_data_size)
              << " %\r" << std::flush;
  }

  printf(
      "Partition %d data file %s write over, expect file size %llu\n",
      index_param.machine_id, index_param.final_scala_data_path.c_str(),
      scala_data_size);
  data_output.close();
  // clear data memory
  free(out_data);
  free(data_level0_memory_);

  return;
}

void ScalaBuild::verify_scala_v3() {
  uint32_t cumupsize[index_param.num_parts + 1];
  memset(cumupsize, 0, sizeof(cumupsize));
  for (int p = 1; p < index_param.num_parts + 1; p++) {
    cumupsize[p] = cumupsize[p - 1] + index_param.partition_size[p - 1];
    std::cout << "cumupsize[" << p << "] = " << cumupsize[p] << std::endl;
  }
  index_param.local_partition_size =
      index_param.partition_size[index_param.machine_id];

  auto judge_partition = [&](uint64_t vid) {
    for (int p = 0; p < index_param.num_parts; p++) {
      if (cumupsize[p + 1] > vid) return p;
    }
    return -1;
  };
  // load original index
  std::cout << "Loading kmeans b1graph index from " << index_param.temp_b1_index
            << "...\n";
  std::ifstream input(index_param.temp_b1_index.c_str(), std::ios::binary);
  if (!input.is_open()) throw std::runtime_error("Cannot open file");

  size_t read_max_elements_;
  uint64_t read_enterpoint_node_;
  input.read((char *)&read_max_elements_, sizeof(uint64_t));
  input.read((char *)&read_enterpoint_node_, sizeof(uint32_t));
  index_param.medoid = read_enterpoint_node_;

  uint64_t size_data_per_element_ = sizeof(uint32_t) +
                                    index_param.max_degree * sizeof(uint32_t) +
                                    index_param.vec_size + sizeof(uint64_t);
  size_t label_offset_ = sizeof(uint32_t) +
                         index_param.max_degree * sizeof(uint32_t) +
                         index_param.vec_size;
  size_t offsetData_ =
      sizeof(uint32_t) + index_param.max_degree * sizeof(uint32_t);

  char *data_level0_memory_ =
      (char *)malloc(read_max_elements_ * size_data_per_element_);
  if (data_level0_memory_ == nullptr)
    throw std::runtime_error(
        "Not enough memory: loadIndex failed to allocate level0");
  input.read(data_level0_memory_, read_max_elements_ * size_data_per_element_);

  std::cout << "Kmeans & Index file read over, parition num:"
            << index_param.num_parts << " parition size:" << read_max_elements_
            << "size data per element:" << size_data_per_element_ << "\n";

  // load scala index
  printf("ScalaGraph V3 loading index ...\n");

  uint32_t partition_num = index_param.num_parts;
  uint32_t partition_id = index_param.machine_id;

  /*
    [ScalaGraph V3 index format]:
    <partition_id: uint32_t>, <partition_num: uint32_t>,
    <part_vec_num: uint32_t>, <local_ngh_ele_size: uint32_t>,
    <enter_point: uint64_t>,
    <remote_ngh_num: uint64_t>
    (header 32 bytes)
    <local_ngh_array: local_ngh_ele_size> * part_vec_num,
    <remote_ngh_array: uint32_t> * remote_ngh_num.
  */
  char *deg_ngh[MACHINE_NUM];
  uint32_t local_ngh_ele_size;
  for (uint32_t m = 0; m < MACHINE_NUM; m++) {
    if (!std::filesystem::exists(index_param.final_part_scala_index_path[m])) {
      std::cout << "Error: file does not exist" << std::endl;
      throw std::runtime_error("file does not exist");
    }
    std::ifstream idx_input(
        index_param.final_part_scala_index_path[m], std::ios::binary);
    uint32_t read_pid, read_pnum, read_pvec_num;
    uint64_t enter_p, remote_ngh_num;
    idx_input.read((char *)&read_pid, sizeof(uint32_t));
    idx_input.read((char *)&read_pnum, sizeof(uint32_t));
    idx_input.read((char *)&read_pvec_num, sizeof(uint32_t));
    idx_input.read((char *)&local_ngh_ele_size, sizeof(uint32_t));
    idx_input.read((char *)&enter_p, sizeof(uint64_t));
    idx_input.read((char *)&remote_ngh_num, sizeof(uint64_t));
    printf(
        "partid %u partnum %u local_vec_num %llu "
        "local_ngh_ele_size %llu remote_ngh_num %llu enter_point %llu\n",
        read_pid, read_pnum, read_pvec_num, local_ngh_ele_size, remote_ngh_num,
        enter_p);
    assert(read_pid == partition_id && read_pnum == partition_num);

    size_t index_size =
        local_ngh_ele_size * read_pvec_num + remote_ngh_num * sizeof(uint32_t);
    printf("read v3 index size %llu\n", index_size);
    deg_ngh[m] = (char *)malloc(index_size);
    idx_input.read((char *)deg_ngh[m], index_size);
    idx_input.close();
  }

  // load scala data.
  char *vector_ptr[partition_num];
  uint32_t partition_vec_num[partition_num];
  for (uint32_t p = 0; p < partition_num; p++) {
    printf("Loading partition %u meta data\n", p);
    if (!std::filesystem::exists(index_param.final_part_scala_data_path[p])) {
      std::cout << "Error: file does not exist" << std::endl;
      throw std::runtime_error("file does not exist");
    }
    std::ifstream data_input(
        index_param.final_part_scala_data_path[p], std::ios::binary);
    uint32_t read_pid, read_pnum, all_vec_num;
    data_input.read((char *)&read_pid, 4);
    data_input.read((char *)&read_pnum, 4);
    data_input.read((char *)&partition_vec_num[p], 4);
    data_input.read((char *)&all_vec_num, 8);
    printf(
        "partid %u partnum %u part_vec_num %llu all_vec_num %llu\n", read_pid,
        read_pnum, partition_vec_num[p], all_vec_num);
    char *enter_p_ptr = (char *)malloc(index_param.vec_size + sizeof(uint64_t));
    data_input.read(
        (char *)enter_p_ptr, index_param.vec_size + sizeof(uint64_t));

    vector_ptr[p] = (char *)malloc(
        (index_param.vec_size + sizeof(uint64_t)) * partition_vec_num[p]);
    data_input.read(
        (char *)vector_ptr[p],
        (index_param.vec_size + sizeof(uint64_t)) * partition_vec_num[p]);
    data_input.close();
  }

  // verify index
  printf("Verify index ...\n");

  // traverse all graph first.
  for(size_t current_node_id = 0; current_node_id < read_max_elements_; current_node_id++) {
    // get gt ngh.
    uint32_t *ptr = (uint32_t *)(data_level0_memory_ +
                                 current_node_id * size_data_per_element_);
    std::vector<uint32_t> gt_vec;
    for (int i = 1; i < ptr[0] + 1; i++) {
      gt_vec.push_back(ptr[i]);
    }

    uint32_t vid_pid = judge_partition(current_node_id);

    // verify data.
    uint64_t data_ofs = current_node_id - cumupsize[vid_pid];
    char *scala_data =
        (char *)(vector_ptr[vid_pid] +
                 (index_param.vec_size + sizeof(uint64_t)) * data_ofs);
    if (memcmp(
            scala_data,
            data_level0_memory_ + current_node_id * size_data_per_element_ +
                offsetData_,
                index_param.vec_size + sizeof(uint64_t)) != 0) {
      printf("Error: data not matched.\n");
    }

    // verify ngh.
    size_t ofs = local_ngh_ele_size * (current_node_id - cumupsize[vid_pid]);
    std::vector<uint32_t> scala_vec;
    uint32_t post_cnt = 0;
    while (true) {
      uint8_t *ptr = (uint8_t *)(deg_ngh[vid_pid] + ofs);
      uint8_t end_flag = ptr[1] & 0x80;
      uint8_t local_flag = ptr[1] & 0x40;
      uint32_t pid = ptr[1] & 0x3f;
      uint32_t deg = ptr[0];

      ofs += 2;
      if (local_flag) {
        // after previous stage, will put into local task queue.
        uint32_t *ngh_ptr = (uint32_t *)(deg_ngh[vid_pid] + ofs);
        for (uint32_t d = 0; d < deg; d++) {
          scala_vec.push_back(ngh_ptr[d]);
        }
        // the local ngh must be end.
        break;
      } else {
        if (deg <= INLINE_DEG_BAR) {
          uint32_t *ngh_ptr = (uint32_t *)(deg_ngh[vid_pid] + ofs);
          for (uint32_t d = 0; d < deg; d++) {
            scala_vec.push_back(ngh_ptr[d]);
          }
          ofs += deg * 4;
        } else {
          uint64_t *ofs_ptr = (uint64_t *)(deg_ngh[vid_pid] + ofs);
          uint32_t *ngh_ptr = (uint32_t *)(deg_ngh[pid] + ofs_ptr[0]);
          for (uint32_t d = 0; d < deg; d++) {
            scala_vec.push_back(ngh_ptr[d]);
          }
          ofs += 8;
        }
      }
      if (end_flag) {
        break;
      }
    }

    if(gt_vec.size() != scala_vec.size()){
      printf("! Mismatch node ngh size: %llu, gt sz: %llu scala sz %llu\n",
         current_node_id, gt_vec.size(), scala_vec.size());
      printf("gt_vec: \n");
      for(auto gtv : gt_vec){
        printf("%llu ", gtv);
      }
      printf("\n scala_vec: \n");
      for(auto scv : scala_vec){
        printf("%llu ", scv);
      }
      printf("\n");
      abort();
    }
    bool mismatch = false;
    sort(gt_vec.begin(), gt_vec.end());
    sort(scala_vec.begin(), scala_vec.end());
    for(uint32_t d = 0;d<gt_vec.size();d++){
      if(gt_vec[d] != scala_vec[d]){
        mismatch = true;
      }
    }

    if(mismatch){
      printf("! Mismatch node ngh: %llu, gt sz: %llu scala sz %llu\n",
        current_node_id, gt_vec.size(), scala_vec.size());
      printf("gt_vec: \n");
      for(auto gtv : gt_vec){
        printf("%llu ", gtv);
      }
      printf("\n scala_vec: \n");
      for(auto scv : scala_vec){
        printf("%llu ", scv);
      }
      printf("\n");
      abort();
    }
    if(current_node_id % 10000 == 0){
      printf("%.2f%% \r", current_node_id * 100.0 / read_max_elements_);
    }
  }
  printf("\n");

  printf("Input vertex id to verify >>>\n");
  size_t current_node_id = 0;
  while (std::cin >> current_node_id) {
    // get gt ngh.
    uint32_t *ptr = (uint32_t *)(data_level0_memory_ +
                                 current_node_id * size_data_per_element_);
    printf("[gt] current_node_id %llu deg %u ngh: ", current_node_id, ptr[0]);
    for (int i = 1; i < ptr[0] + 1; i++) {
      printf("%u ", ptr[i]);
    }
    printf("\n");

    uint32_t vid_pid = judge_partition(current_node_id);

    // verify data.
    uint64_t data_ofs = current_node_id - cumupsize[vid_pid];
    char *scala_data =
        (char *)(vector_ptr[vid_pid] +
                 (index_param.vec_size + sizeof(uint64_t)) * data_ofs);
    if (memcmp(
            scala_data,
            data_level0_memory_ + current_node_id * size_data_per_element_ +
                offsetData_,
                index_param.vec_size + sizeof(uint64_t)) != 0) {
      printf("Error: data not matched.\n");
    }

    // verify ngh.
    size_t ofs = local_ngh_ele_size * (current_node_id - cumupsize[vid_pid]);
    printf(
        "current_node_id %llu vid_pid %u ofs %llu\n", current_node_id, vid_pid,
        ofs);
    uint32_t post_cnt = 0;
    while (true) {
      uint8_t *ptr = (uint8_t *)(deg_ngh[vid_pid] + ofs);
      uint8_t end_flag = ptr[1] & 0x80;
      uint8_t local_flag = ptr[1] & 0x40;
      uint32_t pid = ptr[1] & 0x3f;
      uint32_t deg = ptr[0];

      ofs += 2;
      if (local_flag) {
        // after previous stage, will put into local task queue.
        uint32_t *ngh_ptr = (uint32_t *)(deg_ngh[vid_pid] + ofs);
        printf("> pid %u deg %u ngh: ", pid, deg);
        for (uint32_t d = 0; d < deg; d++) {
          printf("%u ", ngh_ptr[d]);
        }
        printf("\n");
        // the local ngh must be end.
        break;
      } else {
        if (deg <= INLINE_DEG_BAR) {
          uint32_t *ngh_ptr = (uint32_t *)(deg_ngh[vid_pid] + ofs);
          printf("> pid %u deg %u ngh: ", pid, deg);
          for (uint32_t d = 0; d < deg; d++) {
            printf("%u ", ngh_ptr[d]);
          }
          printf("\n");
          ofs += deg * 4;
        } else {
          uint64_t *ofs_ptr = (uint64_t *)(deg_ngh[vid_pid] + ofs);
          uint32_t *ngh_ptr = (uint32_t *)(deg_ngh[pid] + ofs_ptr[0]);
          printf("> pid %u deg %u ngh: ", pid, deg);
          for (uint32_t d = 0; d < deg; d++) {
            printf("%u ", ngh_ptr[d]);
          }
          printf("\n");
          ofs += 8;
        }
      }
      if (end_flag) {
        break;
      }
    }
  }

  return;
}