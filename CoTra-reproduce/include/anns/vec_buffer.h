#pragma once
#include <stdlib.h>

#include <cassert>
#include <deque>

#include "../rdma/rdma_param.h"
#include "anns/anns_param.h"

template <typename dist_t>
struct SyncMsg {
  uint32_t qid;
  // ABOUT TERMINATION
  uint8_t update_term{0};  // whether update term state.
  uint8_t has_token{0}, token_is_black{0}, proc_is_black{1}, last_is_white{1};
  dist_t lower_bound;
  // local top_candidate set update
  std::vector<std::pair<dist_t, uint32_t>> local_add_cand;

  // core machine ngh update
  std::vector<uint32_t> core_micro_ngh[MACHINE_NUM];
  std::vector<std::pair<size_t, uint32_t>> core_large_ngh[MACHINE_NUM];

  SyncMsg() {}

  void clear() {
    local_add_cand.clear();
    for (uint32_t i = 0; i < MACHINE_NUM; i++) {
      core_micro_ngh[i].clear();
      core_large_ngh[i].clear();
    }
  }

  SyncMsg(uint32_t _qid) : qid(_qid) {
    local_add_cand.reserve(48);
    for (uint32_t i = 0; i < MACHINE_NUM; i++) {
      core_micro_ngh[i].reserve(48);
      core_large_ngh[i].reserve(48);
    }
  }

  SyncMsg(char* ptr) {
    size_t offset = 0;
    memcpy(&qid, static_cast<const char*>(ptr) + offset, sizeof(qid));
    offset += sizeof(qid);

    memcpy(
        &update_term, static_cast<const char*>(ptr) + offset,
        sizeof(update_term));
    offset += sizeof(update_term);

    memcpy(
        &has_token, static_cast<const char*>(ptr) + offset, sizeof(has_token));
    offset += sizeof(has_token);
    memcpy(
        &token_is_black, static_cast<const char*>(ptr) + offset,
        sizeof(token_is_black));
    offset += sizeof(token_is_black);

    // printf(
    //     "]] recv sync qid %u update_term %u token_is_black %u\n", qid,
    //     update_term, token_is_black);
    memcpy(
        &proc_is_black, static_cast<const char*>(ptr) + offset,
        sizeof(proc_is_black));
    offset += sizeof(proc_is_black);
    memcpy(
        &last_is_white, static_cast<const char*>(ptr) + offset,
        sizeof(last_is_white));
    offset += sizeof(last_is_white);
    memcpy(
        &lower_bound, static_cast<const char*>(ptr) + offset,
        sizeof(lower_bound));
    offset += sizeof(lower_bound);
    uint32_t size;
    memcpy(&size, static_cast<const char*>(ptr) + offset, sizeof(size));
    offset += sizeof(size);
    for (uint32_t i = 0; i < size; i++) {
      std::pair<dist_t, uint32_t> elem;
      memcpy(&elem, static_cast<const char*>(ptr) + offset, sizeof(elem));
      offset += sizeof(elem);
      local_add_cand.emplace_back(elem);
    }
    for (uint32_t i = 0; i < MACHINE_NUM; i++) {
      uint32_t size2;
      memcpy(&size2, static_cast<const char*>(ptr) + offset, sizeof(size2));
      offset += sizeof(size2);
      uint32_t size3;
      memcpy(&size3, static_cast<const char*>(ptr) + offset, sizeof(size3));
      offset += sizeof(size3);
      for (uint32_t j = 0; j < size2; j++) {
        uint32_t elem;
        memcpy(&elem, static_cast<const char*>(ptr) + offset, sizeof(elem));
        offset += sizeof(elem);
        core_micro_ngh[i].emplace_back(elem);
      }
      for (uint32_t j = 0; j < size3; j++) {
        std::pair<size_t, uint32_t> elem;
        memcpy(&elem, static_cast<const char*>(ptr) + offset, sizeof(elem));
        offset += sizeof(elem);
        core_large_ngh[i].emplace_back(elem);
      }
    }
  }

  size_t serialize(char* ptr) {
    size_t offset = 0;
    memcpy(ptr + offset, &qid, sizeof(qid));
    offset += sizeof(qid);
    memcpy(ptr + offset, &update_term, sizeof(update_term));
    offset += sizeof(update_term);
    memcpy(ptr + offset, &has_token, sizeof(has_token));
    offset += sizeof(has_token);
    memcpy(ptr + offset, &token_is_black, sizeof(token_is_black));
    offset += sizeof(token_is_black);
    memcpy(ptr + offset, &proc_is_black, sizeof(proc_is_black));
    offset += sizeof(proc_is_black);
    memcpy(ptr + offset, &last_is_white, sizeof(last_is_white));
    offset += sizeof(last_is_white);
    memcpy(ptr + offset, &lower_bound, sizeof(lower_bound));
    offset += sizeof(lower_bound);

    std::vector<std::pair<dist_t, uint32_t>> filterred_local_cand;
    for (std::pair<dist_t, uint32_t>& elem : local_add_cand) {
      // filter.
      if (elem.first > lower_bound) {
        continue;
      }
      filterred_local_cand.emplace_back(elem);
    }
    uint32_t size = filterred_local_cand.size();
    memcpy(ptr + offset, &size, sizeof(size));
    offset += sizeof(size);
    for (std::pair<dist_t, uint32_t>& elem : filterred_local_cand) {
      memcpy(ptr + offset, &elem, sizeof(elem));
      offset += sizeof(elem);
    }
    // TODO: serialize only target machine.
    for (uint32_t i = 0; i < MACHINE_NUM; i++) {
      uint32_t size2 = core_micro_ngh[i].size();
      memcpy(ptr + offset, &size2, sizeof(size2));
      offset += sizeof(size2);
      uint32_t size3 = core_large_ngh[i].size();
      memcpy(ptr + offset, &size3, sizeof(size3));
      offset += sizeof(size3);

      for (uint32_t& elem : core_micro_ngh[i]) {
        memcpy(ptr + offset, &elem, sizeof(elem));
        offset += sizeof(elem);
      }
      for (std::pair<size_t, uint32_t>& elem : core_large_ngh[i]) {
        memcpy(ptr + offset, &elem, sizeof(elem));
        offset += sizeof(elem);
        // printf("emplace-sync %llu %u\n", elem.first, elem.second);
      }
      // printf(
      //     "send to q%u sync task size %u, %u, ofs %llu\n", qid, size2, size3,
      //     offset);
    }
    if (offset >= MAX_QUERYBUFFER_SIZE) {
      printf("Error: sync msg exceed buffer size\n");
      abort();
    }
    return offset;
  }
};

template <typename dist_t>
struct ResultMsg {
  uint32_t qid;
  uint8_t core_cnt;
  uint8_t nocore_cnt;
  std::vector<std::pair<dist_t, uint32_t>> res;

  ResultMsg(uint32_t _qid, uint8_t _core_cnt = 1, uint8_t _nocore_cnt = 0) : 
    qid(_qid), core_cnt(_core_cnt), nocore_cnt(_nocore_cnt){
    // TODO: use macro.
    res.reserve(48);
  }

  ResultMsg(char* ptr, uint64_t& offset) {
    memcpy(&qid, ptr + offset, sizeof(qid));
    offset += sizeof(qid);

    memcpy(&core_cnt, ptr + offset, sizeof(core_cnt));
    offset += sizeof(core_cnt);
    memcpy(&nocore_cnt, ptr + offset, sizeof(nocore_cnt));
    offset += sizeof(nocore_cnt);

    // uint32_t* uint_ptr = (uint32_t*)(ptr + offset);
    // qid = uint_ptr[0];
    // offset += sizeof(uint32_t);
    // uint32_t res_size = uint_ptr[1];
    uint32_t res_size;
    memcpy(&res_size, ptr + offset, sizeof(res_size));
    offset += sizeof(res_size);
    std::pair<dist_t, uint32_t>* res_ptr =
        (std::pair<dist_t, uint32_t>*)(ptr + offset);
    // printf("dese res size %u\n", res_size);
    for (uint32_t i = 0; i < res_size; i++) {
      // printf("dist %llu, id %u\n", res_ptr[i].first, res_ptr[i].second);
      res.emplace_back(res_ptr[i]);
      offset += sizeof(std::pair<dist_t, uint32_t>);
    }
    // printf("dese res size over\n");
  }

  void serialize(char* ptr, uint64_t& offset = 0) {
    // Local id to inform remote machine.
    memcpy(ptr + offset, &qid, sizeof(qid));
    offset += sizeof(qid);

    memcpy(ptr + offset, &core_cnt, sizeof(core_cnt));
    offset += sizeof(core_cnt);
    memcpy(ptr + offset, &nocore_cnt, sizeof(nocore_cnt));
    offset += sizeof(nocore_cnt);

    uint32_t res_size = res.size();
    memcpy(ptr + offset, &res_size, sizeof(res_size));
    offset += sizeof(res_size);

    // printf("res.size() %u\n", res.size());
    for (std::pair<dist_t, uint32_t>& r : res) {
      memcpy(ptr + offset, &r, sizeof(r));
      offset += sizeof(r);
    }
    // printf("ofs: %llu\n", offset);
    if (offset > MAX_QUERYBUFFER_SIZE) {
      printf("Error: Result Msg exceed buffer size\n");
      abort();
    }
  }
};

struct ComputeMsg {
  uint32_t qid;
  uint8_t is_count;  // whether should be couted as recved task;
  uint32_t vec_size;
  std::vector<uint32_t> vec_id;  // vector id.
  std::vector<char*> vec_ptr;    // vector pointer.
  std::vector<uint32_t> sg_size;

  char* buf_ptr;  // for record the recved buffer pointer.

  ComputeMsg(uint32_t _qid) : qid(_qid) {
    vec_id.reserve(MAX_SGE_NUM);
    vec_ptr.reserve(MAX_SGE_NUM);
    sg_size.reserve(MAX_SGE_NUM);
  }

  ComputeMsg(uint32_t _qid, uint8_t _is_count, uint32_t _vec_size)
      : qid(_qid), is_count(_is_count), vec_size(_vec_size) {
    vec_id.reserve(MAX_SGE_NUM - 1);
    vec_ptr.reserve(MAX_SGE_NUM);
    sg_size.reserve(MAX_SGE_NUM);
    vec_ptr.emplace_back(nullptr);
    sg_size.emplace_back(0);
  }

  // deserialize
  ComputeMsg(char* ptr) : buf_ptr(ptr) {
    size_t offset = 0;
    memcpy(&qid, static_cast<const char*>(ptr) + offset, sizeof(qid));
    offset += sizeof(qid);
    memcpy(&is_count, static_cast<const char*>(ptr) + offset, sizeof(is_count));
    offset += sizeof(is_count);
    memcpy(&vec_size, static_cast<const char*>(ptr) + offset, sizeof(vec_size));
    offset += sizeof(vec_size);
    uint32_t vec_num;
    memcpy(&vec_num, static_cast<const char*>(ptr) + offset, sizeof(vec_num));
    offset += sizeof(vec_num);
    for (uint32_t i = 0; i < vec_num; i++) {
      uint32_t id;
      memcpy(&id, static_cast<const char*>(ptr) + offset, sizeof(id));
      vec_id.emplace_back(id);
      offset += sizeof(id);
    }
    // deserilized pointer doesnt has pos0 as the meta pointer, only contain the
    // vector pointer.
    for (uint32_t i = 0; i < vec_num; i++) {
      vec_ptr.emplace_back(ptr + offset);
      offset += vec_size;
    }
  }

  /**
   * Serialize format:
   * query_id,
   * is_count, (whether count for finished task)
   * vec_num, (upto MAX_SGE_NUM - 1, one for meta data ptr)
   * vec_id list: <id_1>, <id_2>, ...
   */
  void serialize(char* ptr, size_t& offset) {
    // insert meta ptr to the first position of vec_ptr
    // which indicates the meta ptr.
    vec_ptr[0] = ptr;
    memcpy(ptr + offset, &qid, sizeof(qid));
    offset += sizeof(qid);
    memcpy(ptr + offset, &is_count, sizeof(is_count));
    offset += sizeof(is_count);
    memcpy(ptr + offset, &vec_size, sizeof(vec_size));
    offset += sizeof(vec_size);
    uint32_t vec_num = vec_id.size();
    memcpy(ptr + offset, &vec_num, sizeof(vec_num));
    offset += sizeof(vec_num);
    for (auto& id : vec_id) {
      memcpy(ptr + offset, &id, sizeof(id));
      offset += sizeof(id);
    }
    sg_size[0] = offset;
    if (offset > MAX_QUERYBUFFER_SIZE) {
      printf("exceed buffer size\n");
      abort();
    }
    return;
  }
};

/**
 * Store the node info.
 * NodeTask is used for msg transfer from core to non-core machine.
 */
struct NodeTask {
 public:
  uint32_t machine_id;  // orginal machine id.
  uint32_t qid;
  uint32_t node_id;
  uint32_t is_core_machine[MACHINE_NUM];

  // Record the ngh offset and degree,
  // only for local compute, no need to transfer.
  uint32_t deg{0};
  uint64_t offset{0};

  NodeTask(uint32_t _mid, uint32_t _qid, uint32_t _nid, uint32_t* _is_core_m)
      : machine_id(_mid), qid(_qid), node_id(_nid) {
    memcpy(is_core_machine, _is_core_m, sizeof(is_core_machine));
    // printf("send task core of q%d : ", qid);
    // for (uint32_t _ = 0; _ < MACHINE_NUM; _++)
    //   printf("%d ", is_core_machine[_]);
    // printf("\n");
  }

  NodeTask(char* ptr, uint64_t& ofs) {
    memcpy(&machine_id, static_cast<char*>(ptr) + ofs, sizeof(machine_id));
    ofs += sizeof(machine_id);
    // printf("NodeTask dese machine id is %d\n", machine_id);
    memcpy(&qid, static_cast<char*>(ptr) + ofs, sizeof(qid));
    ofs += sizeof(qid);
    memcpy(&node_id, static_cast<char*>(ptr) + ofs, sizeof(node_id));
    ofs += sizeof(node_id);
    memcpy(
        is_core_machine, static_cast<char*>(ptr) + ofs,
        sizeof(is_core_machine));
    ofs += sizeof(is_core_machine);
    // printf("recv task core of q%d: ", qid);
    // for (uint32_t _ = 0; _ < MACHINE_NUM; _++)
    //   printf("%d ", is_core_machine[_]);
    // printf("\n");
  }

  void serialize(char* ptr, uint64_t& ofs) {
    memcpy(static_cast<char*>(ptr) + ofs, &machine_id, sizeof(machine_id));
    ofs += sizeof(machine_id);
    // printf("NodeTask se machine id is %d\n", machine_id);
    memcpy(static_cast<char*>(ptr) + ofs, &qid, sizeof(qid));
    ofs += sizeof(qid);
    memcpy(static_cast<char*>(ptr) + ofs, &node_id, sizeof(node_id));
    ofs += sizeof(node_id);
    memcpy(
        static_cast<char*>(ptr) + ofs, is_core_machine,
        sizeof(is_core_machine));
    ofs += sizeof(is_core_machine);
  }
};

/**
 * Store the returned node results.
 * NodeResult is used for msg transfer from non-core to core machine.
 */
template <typename dist_t>
struct NodeResult {
 public:
  uint32_t qid;
  // post_cnt record the number of new outside results to other non-core
  // machines.
  uint32_t post_cnt;
  std::vector<std::pair<dist_t, uint32_t>> res;

  NodeResult(uint32_t _qid) : qid(_qid), post_cnt(0) {
    // TODO: use macro.
    res.reserve(48);
  }

  NodeResult(char* ptr, uint64_t& offset) {
    uint32_t* uint_ptr = (uint32_t*)(ptr + offset);
    qid = uint_ptr[0];
    offset += sizeof(uint32_t);
    post_cnt = uint_ptr[1];
    offset += sizeof(uint32_t);
    uint32_t res_size = uint_ptr[2];
    offset += sizeof(uint32_t);
    std::pair<dist_t, uint32_t>* res_ptr =
        (std::pair<dist_t, uint32_t>*)(ptr + offset);
    // printf("dese res size %u\n", res_size);
    for (uint32_t i = 0; i < res_size; i++) {
      // printf("dist %llu, id %u\n", res_ptr[i].first, res_ptr[i].second);
      res.emplace_back(res_ptr[i]);
      offset += sizeof(std::pair<dist_t, uint32_t>);
    }
    // printf("dese res size over\n");
  }

  void serialize(char* ptr, uint64_t& offset = 0) {
    // Local id to inform remote machine.
    memcpy(ptr + offset, &qid, sizeof(qid));
    offset += sizeof(qid);

    memcpy(ptr + offset, &post_cnt, sizeof(post_cnt));
    offset += sizeof(post_cnt);

    uint32_t res_size = res.size();
    memcpy(ptr + offset, &res_size, sizeof(res_size));
    offset += sizeof(res_size);

    // printf("res.size() %u\n", res.size());
    for (std::pair<dist_t, uint32_t>& r : res) {
      memcpy(ptr + offset, &r, sizeof(r));
      offset += sizeof(r);
    }
    // printf("ofs: %llu\n", offset);
    if (offset > MAX_QUERYBUFFER_SIZE) {
      printf(
          "Error: NodeResult Msg exceed buffer size, res_size: %u\n", res_size);
    }
  }
};

/**
 * Sync msg from non-core machine to core machine.
 */
template <typename dist_t>
struct NonCoreSyncMsg {
  uint32_t qid;

  // core machine ngh update
  std::vector<uint32_t> core_micro_ngh;
  std::vector<std::pair<size_t, uint32_t>> core_large_ngh;

  NonCoreSyncMsg() {}

  void clear() {
    core_micro_ngh.clear();
    core_large_ngh.clear();
  }

  NonCoreSyncMsg(uint32_t _qid) : qid(_qid) {
    core_micro_ngh.reserve(48);
    core_large_ngh.reserve(48);
  }

  NonCoreSyncMsg(char* ptr, size_t& offset) {
    memcpy(&qid, static_cast<const char*>(ptr) + offset, sizeof(qid));
    offset += sizeof(qid);

    // printf(
    //     "]] recv sync qid %u update_term %u token_is_black %u\n", qid,
    //     update_term, token_is_black);
    uint32_t size;
    memcpy(&size, static_cast<const char*>(ptr) + offset, sizeof(size));
    offset += sizeof(size);
    for (uint32_t i = 0; i < size; i++) {
      uint32_t elem;
      memcpy(&elem, static_cast<const char*>(ptr) + offset, sizeof(elem));
      offset += sizeof(elem);
      core_micro_ngh.emplace_back(elem);
    }

    memcpy(&size, static_cast<const char*>(ptr) + offset, sizeof(size));
    offset += sizeof(size);
    for (uint32_t i = 0; i < size; i++) {
      std::pair<size_t, uint32_t> elem;
      memcpy(&elem, static_cast<const char*>(ptr) + offset, sizeof(elem));
      offset += sizeof(elem);
      core_large_ngh.emplace_back(elem);
    }
  }

  size_t serialize(char* ptr, size_t& offset) {
    memcpy(ptr + offset, &qid, sizeof(qid));
    offset += sizeof(qid);

    // TODO: serialize only target machine.
    uint32_t size = core_micro_ngh.size();
    memcpy(ptr + offset, &size, sizeof(size));
    offset += sizeof(size);
    for (uint32_t& elem : core_micro_ngh) {
      memcpy(ptr + offset, &elem, sizeof(elem));
      offset += sizeof(elem);
    }

    size = core_large_ngh.size();
    memcpy(ptr + offset, &size, sizeof(size));
    offset += sizeof(size);
    for (std::pair<size_t, uint32_t>& elem : core_large_ngh) {
      memcpy(ptr + offset, &elem, sizeof(elem));
      offset += sizeof(elem);
      // printf("emplace-sync %llu %u\n", elem.first, elem.second);
    }
    // printf(
    //     "send to q%u sync task size %u, %u, ofs %llu\n", qid, size2, size3,
    //     offset);
    if (offset >= MAX_QUERYBUFFER_SIZE) {
      printf("Error: Non-core node sync msg exceed buffer size\n");
      abort();
    }
    return offset;
  }
};

template <typename dist_t>
struct ComputeTask {
 public:
  uint32_t
      machine_id;  // indicates which machine the result should be returned.
  uint32_t qid;
  uint32_t deg;
  uint64_t offset;
  dist_t lower_bound;
  uint8_t ef_fill;

  // count for core or nocore
  uint8_t core_cnt{1};
  uint8_t nocore_cnt{0};

  // to avoid count for work steal
  bool is_count;  // whether should be couted as recved task;

  ComputeTask() {}
  // for local task construct
  // TODO: can concise
  ComputeTask(
      uint32_t _machine_id, uint32_t _query_id, uint32_t _deg, uint64_t _ofs)
      : machine_id(_machine_id),
        qid(_query_id),
        deg(_deg),
        offset(_ofs),
        is_count(true) {}

  ComputeTask(
      uint32_t _machine_id, uint32_t _query_id, uint32_t _deg, uint64_t _ofs,
      bool _is_count)
      : machine_id(_machine_id),
        qid(_query_id),
        deg(_deg),
        offset(_ofs),
        is_count(_is_count) {}

  // for remote task construct
  ComputeTask(
      uint32_t _machine_id, uint32_t _query_id, uint32_t _deg, uint64_t _ofs,
      dist_t _lower_bound, uint8_t _ef_fill, 
      uint8_t _core_cnt = 1, uint8_t _nocore_cnt = 0)
      : machine_id(_machine_id),
        qid(_query_id),
        deg(_deg),
        offset(_ofs),
        lower_bound(_lower_bound),
        ef_fill(_ef_fill),
        core_cnt(_core_cnt),
        nocore_cnt(_nocore_cnt) {}

  ComputeTask(char* ptr, uint64_t& ofs) {
    // TODO: ask gpt to optimize
    // machine_id = mid;
    memcpy(&machine_id, static_cast<char*>(ptr) + ofs, sizeof(machine_id));
    ofs += sizeof(machine_id);
    memcpy(&qid, static_cast<char*>(ptr) + ofs, sizeof(qid));
    ofs += sizeof(qid);
    memcpy(&deg, static_cast<char*>(ptr) + ofs, sizeof(deg));
    ofs += sizeof(deg);
    memcpy(&offset, static_cast<char*>(ptr) + ofs, sizeof(offset));
    ofs += sizeof(offset);
    memcpy(&lower_bound, static_cast<char*>(ptr) + ofs, sizeof(lower_bound));
    ofs += sizeof(lower_bound);
    memcpy(&ef_fill, static_cast<char*>(ptr) + ofs, sizeof(ef_fill));
    ofs += sizeof(ef_fill);
    memcpy(&core_cnt, static_cast<char*>(ptr) + ofs, sizeof(core_cnt));
    ofs += sizeof(core_cnt);
    memcpy(&nocore_cnt, static_cast<char*>(ptr) + ofs, sizeof(nocore_cnt));
    ofs += sizeof(nocore_cnt);
    // uint32_t* uint_ptr = (uint32_t*)(ptr + ofs);
    // qid = *(uint_ptr);
    // ofs += sizeof(uint32_t);
    // deg = *(uint_ptr + 1);
    // ofs += sizeof(uint32_t);
    // uint64_t* lint_ptr = (uint64_t*)(ptr + ofs);
    // offset = *(lint_ptr);
    // ofs += sizeof(uint64_t);
    // dist_t* lb_ptr = (dist_t*)(ptr + ofs);
    // lower_bound = *(lb_ptr);
    // ofs += sizeof(dist_t);
    // uint8_t* ef_ptr = (uint8_t*)(ptr + ofs);
    // ef_fill = *(ef_ptr);
    // ofs += sizeof(uint8_t);
    // printf("qid %u deg %u offset %llu\n", qid, deg, offset);
  }

  void serialize(char* ptr, uint64_t& ofs) {
    memcpy(static_cast<char*>(ptr) + ofs, &machine_id, sizeof(machine_id));
    ofs += sizeof(machine_id);
    memcpy(static_cast<char*>(ptr) + ofs, &qid, sizeof(qid));
    ofs += sizeof(qid);
    memcpy(static_cast<char*>(ptr) + ofs, &deg, sizeof(deg));
    ofs += sizeof(deg);
    memcpy(static_cast<char*>(ptr) + ofs, &offset, sizeof(offset));
    ofs += sizeof(offset);
    memcpy(static_cast<char*>(ptr) + ofs, &lower_bound, sizeof(lower_bound));
    ofs += sizeof(lower_bound);
    memcpy(static_cast<char*>(ptr) + ofs, &ef_fill, sizeof(ef_fill));
    ofs += sizeof(ef_fill);
    memcpy(static_cast<char*>(ptr) + ofs, &core_cnt, sizeof(core_cnt));
    ofs += sizeof(core_cnt);
    memcpy(static_cast<char*>(ptr) + ofs, &nocore_cnt, sizeof(nocore_cnt));
    ofs += sizeof(nocore_cnt);
  }
};

struct BufferCache {
  uint32_t machine_id;
  uint64_t vector_id;
  uint64_t internal_id;
  int buffer_id;
  char* buffer_ptr;
  BufferCache() {}

  BufferCache(
      uint32_t _machine_id, uint64_t _vector_id, uint64_t _internal_id,
      int _buffer_id, char* _buffer_ptr)
      : machine_id(_machine_id),
        vector_id(_vector_id),
        internal_id(_internal_id),
        buffer_id(_buffer_id),
        buffer_ptr(_buffer_ptr) {}
};

template <typename dist_t>
struct VectorCache {
  dist_t dist;
  uint64_t vector_id;
  int buffer_id;  // -1 means local buffer or not dispatch.
  char* ptr;
};

// TODO: ReadBuffer
struct VectorBuffer {
  size_t vec_len = MAX_VEC_LEN;
  std::deque<int> buff_id_list;
  char vector[MAX_CACHE_VEC][MAX_VEC_LEN];
  BufferCache bufcache_ptr[MAX_CACHE_VEC];
  size_t cur_buffer_id[MACHINE_NUM];
  std::deque<size_t> recv_list;
};

/**
 * Send Write Buffer
 */
struct SendWriteBuffer {
  size_t buffer_len = MAX_QUERYBUFFER_SIZE * MAX_WRITE_NUM;
  /*
    When used as send buffer, position_list[m] record the
    available id of buffer for machine m.
  */

  std::deque<uint32_t>* buff_id_list[MACHINE_NUM];

  char* buffer[MACHINE_NUM][MAX_WRITE_NUM];

  void init() {
    for (int i = 0; i < MACHINE_NUM; i++) {
      // TODO: may reserve for optimization.
      buff_id_list[i] = new std::deque<uint32_t>();
    }

    // TODO: may use numa alloc.
    char* buf_ptr = static_cast<char*>(
        malloc(MACHINE_NUM * MAX_WRITE_NUM * MAX_QUERYBUFFER_SIZE));

    std::fill(
        buf_ptr, buf_ptr + MACHINE_NUM * MAX_WRITE_NUM * MAX_QUERYBUFFER_SIZE,
        0);
    for (int i = 0; i < MACHINE_NUM; i++) {
      for (int j = 0; j < MAX_WRITE_NUM; j++) {
        buffer[i][j] = buf_ptr + (i * MAX_WRITE_NUM + j) * MAX_QUERYBUFFER_SIZE;
      }
    }
  }
};

/**
 * Recv Write Buffer
 */
struct RecvWriteBuffer {
  size_t buffer_len = MAX_QUERYBUFFER_SIZE * MAX_WRITE_NUM;
  /*
    When used as recv buffer, buff_id_list[m] record the
    meta data of received buffer for machine m.
  */
  std::deque<char*> b2_start_queue;  // for B2
  std::deque<char*> b2_end_queue;    // for B2

  std::deque<char*> part_info_queue;      // for build
  std::deque<char*> build_sync_queue;     // for build
  std::deque<char*> dispatch_meta_queue;  // for build
  std::deque<char*> dispatch_id_queue;    // for build
  std::deque<char*> dispatch_ngh_queue;   // for build

  std::deque<char*> scala_meta_queue;  // for scala meta
  std::deque<char*> scala_ngh_queue;  // for scala ngh

  std::deque<uint32_t>* release_id_list[MACHINE_NUM];
  // Collected migrate query pointer.
  // Transfer to general serialize and deserialize method in template class.
  std::deque<char*> migrate_queue;
  std::deque<char*> core_info_queue;
  std::deque<char*> sync_queue;
  std::deque<char*> recv_task;
  std::deque<char*> recv_result;
  std::deque<char*> recv_node_res;   // for node task
  std::deque<char*> recv_node_sync;  // for node task
  std::deque<char*> recv_compute;
  char* buffer[MACHINE_NUM][MAX_WRITE_NUM];

  void init() {
    for (int i = 0; i < MACHINE_NUM; i++) {
      release_id_list[i] = new std::deque<uint32_t>();
    }

    char* buf_ptr = static_cast<char*>(
        malloc(MACHINE_NUM * MAX_WRITE_NUM * MAX_QUERYBUFFER_SIZE));

    std::fill(
        buf_ptr, buf_ptr + MACHINE_NUM * MAX_WRITE_NUM * MAX_QUERYBUFFER_SIZE,
        0);
    for (int i = 0; i < MACHINE_NUM; i++) {
      for (int j = 0; j < MAX_WRITE_NUM; j++) {
        buffer[i][j] = buf_ptr + (i * MAX_WRITE_NUM + j) * MAX_QUERYBUFFER_SIZE;
      }
    }
  }
};
