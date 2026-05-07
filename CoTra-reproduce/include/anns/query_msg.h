#pragma once

#include <atomic>
#include <fstream>
#include <list>
#include <map>
#include <memory>
#include <queue>
#include <random>
#include <unordered_set>

#include "anns_param.h"
#include "coromem/include/runtime/corobj.h"
#include "coromem/include/worklists/balance_queue.h"
#include "coromem/include/worklists/chunk.h"
#include "coromem/include/worklists/lockfreeQ.h"
#include "hash_table.h"
#include "rdma/profiler.h"
#include "vec_buffer.h"

/**
 * START: query init state.
 * PRE_STAGE: first 10-hop.
 * POST_STAGE: after dispatch sub-query.
 * PAUSE: no local compute, wait for res/task or termination.
 * END: query end.
 */
typedef enum { START, PRE_STAGE, POST_STAGE, PAUSE, END } QueryState;

// TODO: merge Corobj & QueryMsg
template <typename dist_t>
class QueryMsg {
 public:
  size_t ef;
  size_t query_id;
  size_t origin_machine;
  size_t current_node_id;
  dist_t lowerBound;
  OptHashPosVector visit_hash;

  Profiler profiler;

  size_t vector_size{0};
  char* vector{nullptr};
  QueryState state;

  uint32_t b2_dispatched_num{0}, b2_recv_num{0};
  uint32_t b2_original_mid{0};

  // Path count, for migration ref.
  uint32_t path_cnt[MACHINE_NUM];

  // [TEMP] for profiling.
  uint32_t tmp_m_cnt[MACHINE_NUM];
  uint32_t tmp_path_cnt[MACHINE_NUM];

  uint32_t max_path_id{0}, max_path_cnt{0};
  uint32_t path_length{0};

  // for sub-query sync
  uint32_t leader_machine{0};
  uint8_t has_token{0}, token_is_black{0}, proc_is_black{1}, last_is_white{1};
  uint8_t global_term{0};

  std::vector<uint32_t> core_machine;
  uint32_t is_core_machine[MACHINE_NUM];

  // filterred candidates.
  std::vector<std::pair<dist_t, uint32_t>> filterred[MACHINE_NUM];
  uint32_t filter_num[MACHINE_NUM];

  uint32_t post_cnt{0}, recv_cnt{0};  // post/recv count.
  uint32_t nocore_post{0}, nocore_recv{0};  // no-core post/recv count.
  uint32_t sync_step{0};
  SyncMsg<dist_t> sync_msg;  // for sub-query sync.

  /**
   * After previous stage, local task will not put into global task queue
   * (tman).
   * local_task / local_micro_task: local genrated task & recved ngh sync from
   * other machines.
   */
  std::vector<ComputeTask<dist_t>> local_task;
  std::vector<uint32_t> local_micro_task;
  std::vector<ResultMsg<dist_t>> local_res;

  // indicate whether new (has coro) for this machine, forbid serialize.
  bool is_new{true};
  bool heavy_state_released{false};

  // group cand nghs
  //   std::pair<dist_t, uint32_t> group_nodes[CAND_GROUP_SIZE];
  // corobj
  Corobj<bool> coro;

  bool has_cur_res = false;
  ResultMsg<dist_t>* cur_res;

  bool has_cur_cmp = false;
  ComputeMsg* cur_cmp;

  bool has_cur_task = false;
  ComputeTask<dist_t>* cur_task;

  bool has_cur_async = false;
  uint32_t async_read_num, async_read_cnt;
  std::vector<BufferCache> async_read_queue;       // pre stage only
  std::vector<BufferCache> recv_async_read_queue;  // post stage only

  LockFreeQueue<worklists::Chunk<std::pair<dist_t, uint32_t>, MAX_GRAPH_DEG>>
      con_task_queue;

  struct CompareByFirst {
    constexpr bool operator()(
        std::pair<dist_t, uint32_t> const& a,
        std::pair<dist_t, uint32_t> const& b) const noexcept {
      return a.first < b.first;
    }
  };

  std::priority_queue<
      std::pair<dist_t, uint32_t>, std::vector<std::pair<dist_t, uint32_t>>,
      CompareByFirst>
      top_candidates;
  std::priority_queue<
      std::pair<dist_t, uint32_t>, std::vector<std::pair<dist_t, uint32_t>>,
      CompareByFirst>
      candidate_set;

  // Used in top index (pre-stage).
  std::priority_queue<
      std::pair<dist_t, uint32_t>, std::vector<std::pair<dist_t, uint32_t>>,
      CompareByFirst>
      pre_top_cand;
  std::priority_queue<
      std::pair<dist_t, uint32_t>, std::vector<std::pair<dist_t, uint32_t>>,
      CompareByFirst>
      pre_cand_set;

  std::priority_queue<std::pair<dist_t, size_t>> result;

  QueryMsg() {}
  ~QueryMsg() { release_owned_storage(); }

  QueryMsg(QueryState _state) : state(_state), origin_machine(0) {}

  QueryMsg(char* _vector, size_t _vector_size, size_t _query_id, size_t _ef, size_t _machine)
      : vector_size(_vector_size), query_id(_query_id), ef(_ef), origin_machine(_machine), state(START) {
    vector = (char* )malloc(vector_size);
    memcpy(vector, _vector, vector_size);
    memset(path_cnt, 0, sizeof(path_cnt));
    memset(is_core_machine, 0, sizeof(is_core_machine));
  }

  void release_owned_storage() {
    if (vector != nullptr) {
      free(vector);
      vector = nullptr;
    }
    vector_size = 0;
  }

  template <typename T>
  static void release_vector_storage(std::vector<T>& vec) {
    std::vector<T>().swap(vec);
  }

  template <typename QueueT>
  static void release_queue_storage(QueueT& queue) {
    QueueT empty;
    queue.swap(empty);
  }

  void release_sync_msg_storage() {
    release_vector_storage(sync_msg.local_add_cand);
    for (uint32_t i = 0; i < MACHINE_NUM; i++) {
      release_vector_storage(sync_msg.core_micro_ngh[i]);
      release_vector_storage(sync_msg.core_large_ngh[i]);
    }
  }

  void release_heavy_state_after_result() {
    if (heavy_state_released) {
      return;
    }

    release_owned_storage();
    visit_hash.release_storage();

    release_vector_storage(core_machine);
    for (uint32_t i = 0; i < MACHINE_NUM; i++) {
      release_vector_storage(filterred[i]);
    }
    release_sync_msg_storage();

    release_vector_storage(local_task);
    release_vector_storage(local_micro_task);
    release_vector_storage(local_res);
    release_vector_storage(async_read_queue);
    release_vector_storage(recv_async_read_queue);
    // Keep the QueryMsg shell alive in global_query, but drain queued chunks.
    while (con_task_queue.pop().has_value()) {
    }

    release_queue_storage(top_candidates);
    release_queue_storage(candidate_set);
    release_queue_storage(pre_top_cand);
    release_queue_storage(pre_cand_set);
    release_queue_storage(result);

    cur_res = nullptr;
    cur_cmp = nullptr;
    cur_task = nullptr;
    has_cur_res = has_cur_cmp = has_cur_task = has_cur_async = false;
    heavy_state_released = true;
  }

  void print_msg() {
    printf("ef %d\n", ef);
    printf("query_id %d\n", query_id);
    printf("current_node_id %d\n", current_node_id);
    printf("lowerBound %d\n", lowerBound);
    printf("top_candidates %d\n", top_candidates.size());
    printf("candidate_set %d\n", candidate_set.size());
    printf("result %d\n", result.size());
  }

  /**
   *  Update sub-query based on sync msg.
   */
  void update_sub_query(
      uint32_t machine_id, SyncMsg<dist_t>* sync_msg,
      BalanceQueue<ComputeTask<dist_t>>* ib) {
    if (heavy_state_released) {
      return;
    }
    if (sync_msg->update_term) {
      // printf("q%u PROP update\n", sync_msg->qid);
      has_token = sync_msg->has_token;
      token_is_black = sync_msg->token_is_black;
    }

    // Sync top_candidate set & lower bound.
    for (std::pair<dist_t, uint32_t>& cand : sync_msg->local_add_cand) {
      if (top_candidates.size() < ef || cand.first < lowerBound) {
        top_candidates.push(cand);
        // printf("--update d%llu v%u\n", cand.second, cand.first);
      }
    }
    while (top_candidates.size() > ef) {
      top_candidates.pop();
    }
    if (!top_candidates.empty()) lowerBound = top_candidates.top().first;

    for (uint32_t& elem : sync_msg->core_micro_ngh[machine_id]) {
      local_micro_task.emplace_back(elem);
    }

    // Put into local queue.
    for (std::pair<size_t, uint32_t>& elem :
         sync_msg->core_large_ngh[machine_id]) {
      local_task.emplace_back(machine_id, query_id, elem.second, elem.first);
    }

    return;
  }

  /**
   *  Update sub-query based on sync msg from non-core machine.
   */
  void update_sub_query_from_noncore(
      uint32_t machine_id, NonCoreSyncMsg<dist_t>* sync_msg) {
    if (heavy_state_released) {
      return;
    }
    // printf("update sub query from noncore\n");
    for (uint32_t& elem : sync_msg->core_micro_ngh) {
      local_micro_task.emplace_back(elem);
    }

    // Put into local queue.
    for (std::pair<size_t, uint32_t>& elem : sync_msg->core_large_ngh) {
      local_task.emplace_back(machine_id, query_id, elem.second, elem.first);
    }

    return;
  }

  // New Serialization function, without visit list.
  size_t serialize2(void* dest) {
    size_t offset = 0;
    memcpy(static_cast<char*>(dest) + offset, &ef, sizeof(ef));
    offset += sizeof(ef);
    memcpy(static_cast<char*>(dest) + offset, &query_id, sizeof(query_id));
    offset += sizeof(query_id);
    memcpy(
        static_cast<char*>(dest) + offset, &origin_machine,
        sizeof(origin_machine));
    offset += sizeof(origin_machine);

    memcpy(
        static_cast<char*>(dest) + offset, &current_node_id,
        sizeof(current_node_id));
    offset += sizeof(current_node_id);
    memcpy(static_cast<char*>(dest) + offset, &lowerBound, sizeof(lowerBound));
    offset += sizeof(lowerBound);
    memcpy(static_cast<char*>(dest) + offset, &vector_size, sizeof(vector_size));
    offset += sizeof(vector_size);
    memcpy(static_cast<char*>(dest) + offset, vector, vector_size);
    offset += vector_size;
    memcpy(static_cast<char*>(dest) + offset, &state, sizeof(state));
    offset += sizeof(state);

    // Path count, for migration ref.
    memcpy(static_cast<char*>(dest) + offset, path_cnt, sizeof(path_cnt));
    offset += sizeof(path_cnt);
    memcpy(
        static_cast<char*>(dest) + offset, &max_path_id, sizeof(max_path_id));
    offset += sizeof(max_path_id);
    memcpy(
        static_cast<char*>(dest) + offset, &max_path_cnt, sizeof(max_path_cnt));
    offset += sizeof(max_path_cnt);
    memcpy(
        static_cast<char*>(dest) + offset, &path_length, sizeof(path_length));
    offset += sizeof(path_length);

    memcpy(
        static_cast<char*>(dest) + offset, &leader_machine,
        sizeof(leader_machine));
    offset += sizeof(leader_machine);
    // core machines
    // TODO: use uint32_t to reduce amount.
    size_t core_machine_size = core_machine.size();
    memcpy(
        static_cast<char*>(dest) + offset, &core_machine_size,
        sizeof(core_machine_size));
    offset += sizeof(core_machine_size);
    for (auto m : core_machine) {
      memcpy(static_cast<char*>(dest) + offset, &m, sizeof(m));
      offset += sizeof(m);
    }

    // #ifdef PROFILER
    //     profiler.serialize(static_cast<char*>(dest), offset);
    // #endif

    // Serialize top_candidates
    // printf("Serialize top_candidates\n");
    size_t top_candidates_size = top_candidates.size();
    memcpy(
        static_cast<char*>(dest) + offset, &top_candidates_size,
        sizeof(top_candidates_size));
    offset += sizeof(top_candidates_size);

    while (!top_candidates.empty()) {
      auto& elem = top_candidates.top();
      memcpy(static_cast<char*>(dest) + offset, &elem, sizeof(elem));
      offset += sizeof(elem);
      top_candidates.pop();
    }

    // Serialize candidate_set
    // TODO: Filter candset
    std::vector<std::pair<dist_t, uint32_t>> filter_queue;
    while (!candidate_set.empty()) {
      auto& elem = candidate_set.top();
      if (-elem.first < lowerBound) {
        filter_queue.push_back(elem);
      }
      candidate_set.pop();
    }
    size_t filter_queue_size = filter_queue.size();
    memcpy(
        static_cast<char*>(dest) + offset, &filter_queue_size,
        sizeof(filter_queue_size));
    offset += sizeof(filter_queue_size);
    for (auto filter_ele : filter_queue) {
      memcpy(
          static_cast<char*>(dest) + offset, &filter_ele, sizeof(filter_ele));
      offset += sizeof(filter_ele);
    }

    // size_t candidate_set_size = candidate_set.size();
    // memcpy(
    //     static_cast<char*>(dest) + offset, &candidate_set_size,
    //     sizeof(candidate_set_size));
    // offset += sizeof(candidate_set_size);

    // while (!candidate_set.empty()) {
    //   auto& elem = candidate_set.top();
    //   memcpy(static_cast<char*>(dest) + offset, &elem, sizeof(elem));
    //   offset += sizeof(elem);
    //   candidate_set.pop();
    // }

    // TODO: if the final transmission (state = END), only need the result.
    // Serialize result
    size_t result_size = result.size();
    memcpy(
        static_cast<char*>(dest) + offset, &result_size, sizeof(result_size));
    offset += sizeof(result_size);

    while (!result.empty()) {
      auto& elem = result.top();
      memcpy(static_cast<char*>(dest) + offset, &elem, sizeof(elem));
      offset += sizeof(elem);
      result.pop();
    }

#ifdef DEBUG
    memcpy(static_cast<char*>(dest) + offset, tmp_m_cnt, sizeof(tmp_m_cnt));
    offset += sizeof(tmp_m_cnt);
#endif

    assert(offset <= MAX_QUERYBUFFER_SIZE);
    if (offset > MAX_QUERYBUFFER_SIZE) {
      printf("Error: exceed buffer size.\n");
      abort();
    }
    return offset;
  }

  /**
   * Fork QueryMsg in different core machines.
   */
  size_t fork(uint32_t machine_id, void* dest) {
    size_t offset = 0;
    memcpy(static_cast<char*>(dest) + offset, &ef, sizeof(ef));
    offset += sizeof(ef);
    memcpy(static_cast<char*>(dest) + offset, &query_id, sizeof(query_id));
    offset += sizeof(query_id);
    memcpy(
        static_cast<char*>(dest) + offset, &origin_machine,
        sizeof(origin_machine));
    offset += sizeof(origin_machine);

    memcpy(
        static_cast<char*>(dest) + offset, &current_node_id,
        sizeof(current_node_id));
    offset += sizeof(current_node_id);
    memcpy(static_cast<char*>(dest) + offset, &lowerBound, sizeof(lowerBound));
    offset += sizeof(lowerBound);
    memcpy(static_cast<char*>(dest) + offset, &vector_size, sizeof(vector_size));
    offset += sizeof(vector_size);
    memcpy(static_cast<char*>(dest) + offset, vector, vector_size);
    offset += vector_size;
    memcpy(static_cast<char*>(dest) + offset, &state, sizeof(state));
    offset += sizeof(state);

    // Path count, for migration ref.
    memcpy(static_cast<char*>(dest) + offset, path_cnt, sizeof(path_cnt));
    offset += sizeof(path_cnt);
    memcpy(
        static_cast<char*>(dest) + offset, &max_path_id, sizeof(max_path_id));
    offset += sizeof(max_path_id);
    memcpy(
        static_cast<char*>(dest) + offset, &max_path_cnt, sizeof(max_path_cnt));
    offset += sizeof(max_path_cnt);
    memcpy(
        static_cast<char*>(dest) + offset, &path_length, sizeof(path_length));
    offset += sizeof(path_length);

    memcpy(
        static_cast<char*>(dest) + offset, &leader_machine,
        sizeof(leader_machine));
    offset += sizeof(leader_machine);
    // core machines
    // TODO: use uint32_t to reduce amount.
    size_t core_machine_size = core_machine.size();
    memcpy(
        static_cast<char*>(dest) + offset, &core_machine_size,
        sizeof(core_machine_size));
    offset += sizeof(core_machine_size);
    for (auto m : core_machine) {
      memcpy(static_cast<char*>(dest) + offset, &m, sizeof(m));
      offset += sizeof(m);
    }

    // #ifdef PROFILER
    //     profiler.serialize(static_cast<char*>(dest), offset);
    // #endif

    // Serialize top_candidates
    // printf("Serialize top_candidates\n");
    size_t top_candidates_size = top_candidates.size();
    memcpy(
        static_cast<char*>(dest) + offset, &top_candidates_size,
        sizeof(top_candidates_size));
    offset += sizeof(top_candidates_size);

    auto top_candidates_copy = top_candidates;
    while (!top_candidates.empty()) {
      auto& elem = top_candidates.top();
      memcpy(static_cast<char*>(dest) + offset, &elem, sizeof(elem));
      offset += sizeof(elem);
      top_candidates.pop();
    }
    swap(top_candidates, top_candidates_copy);

    // Serialize candidate_set
    size_t filter_queue_size = filterred[machine_id].size();
    memcpy(
        static_cast<char*>(dest) + offset, &filter_queue_size,
        sizeof(filter_queue_size));
    offset += sizeof(filter_queue_size);
    for (auto filter_ele : filterred[machine_id]) {
      memcpy(
          static_cast<char*>(dest) + offset, &filter_ele, sizeof(filter_ele));
      offset += sizeof(filter_ele);
    }

    // TODO: if the final transmission (state = END), only need the result.
    // Serialize result
    size_t result_size = result.size();
    memcpy(
        static_cast<char*>(dest) + offset, &result_size, sizeof(result_size));
    offset += sizeof(result_size);

    while (!result.empty()) {
      auto& elem = result.top();
      memcpy(static_cast<char*>(dest) + offset, &elem, sizeof(elem));
      offset += sizeof(elem);
      result.pop();
    }

    assert(offset <= MAX_QUERYBUFFER_SIZE);
    if (offset > MAX_QUERYBUFFER_SIZE) {
      printf("Error: exceed buffer size.\n");
      abort();
    }
    return offset;
  }

  // Deserialization function
  /*
    Construct QueryMsg from given pointer.
  */
  // TODO: now only tranfer the END query, so can concise.
  void deserialize_and_merge(const void* src) {
    size_t offset = 0;
    memcpy(&ef, static_cast<const char*>(src) + offset, sizeof(ef));
    offset += sizeof(ef);
    memcpy(&query_id, static_cast<const char*>(src) + offset, sizeof(query_id));
    offset += sizeof(query_id);

    memcpy(
        &origin_machine, static_cast<const char*>(src) + offset,
        sizeof(origin_machine));
    offset += sizeof(origin_machine);
    memcpy(
        &current_node_id, static_cast<const char*>(src) + offset,
        sizeof(current_node_id));
    offset += sizeof(current_node_id);

    memcpy(
        &lowerBound, static_cast<const char*>(src) + offset,
        sizeof(lowerBound));
    offset += sizeof(lowerBound);
    memcpy(
        &vector_size, static_cast<const char*>(src) + offset,
        sizeof(vector_size));
    offset += sizeof(vector_size);
    if(vector == nullptr) vector = (char*)malloc(vector_size);
    memcpy(
        vector, static_cast<const char*>(src) + offset,
        vector_size);
    offset += vector_size;
    memcpy(&state, static_cast<const char*>(src) + offset, sizeof(state));
    offset += sizeof(state);

    // Path count, for migration ref.
    memcpy(path_cnt, static_cast<const char*>(src) + offset, sizeof(path_cnt));
    offset += sizeof(path_cnt);

    memcpy(
        &max_path_id, static_cast<const char*>(src) + offset,
        sizeof(max_path_id));
    offset += sizeof(max_path_id);

    memcpy(
        &max_path_cnt, static_cast<const char*>(src) + offset,
        sizeof(max_path_cnt));
    offset += sizeof(max_path_cnt);

    memcpy(
        &path_length, static_cast<const char*>(src) + offset,
        sizeof(path_length));
    offset += sizeof(path_length);

    // leader machine id
    memcpy(
        &leader_machine, static_cast<const char*>(src) + offset,
        sizeof(leader_machine));
    offset += sizeof(leader_machine);
    // core machines
    size_t core_machine_size;
    memcpy(
        &core_machine_size, static_cast<const char*>(src) + offset,
        sizeof(core_machine_size));
    offset += sizeof(core_machine_size);
    for (size_t i = 0; i < core_machine_size; ++i) {
      uint32_t m;
      memcpy(&m, static_cast<const char*>(src) + offset, sizeof(m));
      offset += sizeof(m);
      core_machine.emplace_back(m);
      is_core_machine[m] = 1;
    }

    // printf("start deserialize top\n");

    // #ifdef PROFILER
    //     profiler.deserialize(static_cast<const char*>(src), offset);
    // #endif

    // Deserialize top_candidates
    size_t top_candidates_size;
    memcpy(
        &top_candidates_size, static_cast<const char*>(src) + offset,
        sizeof(top_candidates_size));
    offset += sizeof(top_candidates_size);
    while (!top_candidates.empty()) {
      top_candidates.pop();
    }
    for (size_t i = 0; i < top_candidates_size; ++i) {
      std::pair<dist_t, uint32_t> elem;
      memcpy(&elem, static_cast<const char*>(src) + offset, sizeof(elem));
      offset += sizeof(elem);
      top_candidates.push(elem);
    }

    // printf("start candidate_set_size\n");
    // Deserialize candidate_set
    size_t candidate_set_size;
    memcpy(
        &candidate_set_size, static_cast<const char*>(src) + offset,
        sizeof(candidate_set_size));
    offset += sizeof(candidate_set_size);
    while (!candidate_set.empty()) {
      candidate_set.pop();
    }

    // printf("candidate_set_size %u\n", candidate_set_size);

    for (size_t i = 0; i < candidate_set_size; ++i) {
      std::pair<dist_t, uint32_t> elem;
      memcpy(&elem, static_cast<const char*>(src) + offset, sizeof(elem));
      offset += sizeof(elem);
      candidate_set.push(elem);
      visit_hash.CheckAndSet(elem.second);
    }

    // printf("start result_size\n");

    // TODO: this can concise maybe.
    // Deserialize result
    size_t result_size;
    memcpy(
        &result_size, static_cast<const char*>(src) + offset,
        sizeof(result_size));
    offset += sizeof(result_size);
    while (!result.empty()) {
      result.pop();
    }
    for (size_t i = 0; i < result_size; ++i) {
      std::pair<dist_t, size_t> elem;
      memcpy(&elem, static_cast<const char*>(src) + offset, sizeof(elem));
      offset += sizeof(elem);
      result.push(elem);
    }

#ifdef DEBUG
    memcpy(
        tmp_m_cnt, static_cast<const char*>(src) + offset, sizeof(tmp_m_cnt));
    offset += sizeof(tmp_m_cnt);
#endif

    // printf("offset: %d\n", offset);

    return;
  }

  // Serialization function
  void serialize(void* dest) const {
    size_t offset = 0;
    memcpy(static_cast<char*>(dest) + offset, &ef, sizeof(ef));
    offset += sizeof(ef);
    memcpy(static_cast<char*>(dest) + offset, &query_id, sizeof(query_id));
    offset += sizeof(query_id);
    memcpy(
        static_cast<char*>(dest) + offset, &origin_machine,
        sizeof(origin_machine));
    offset += sizeof(origin_machine);

    memcpy(
        static_cast<char*>(dest) + offset, &current_node_id,
        sizeof(current_node_id));
    offset += sizeof(current_node_id);
    memcpy(static_cast<char*>(dest) + offset, &lowerBound, sizeof(lowerBound));
    offset += sizeof(lowerBound);
    memcpy(static_cast<char*>(dest) + offset, &vector_size, sizeof(vector_size));
    offset += sizeof(vector_size);
    memcpy(static_cast<char*>(dest) + offset, vector, vector_size);
    offset += vector_size;
    memcpy(static_cast<char*>(dest) + offset, &state, sizeof(state));
    offset += sizeof(state);

    // Serialize visit_hash
// #ifdef MSGQ
//     memcpy(
//         static_cast<char*>(dest) + offset, &visit_hash.m_secondHash,
//         sizeof(visit_hash.m_secondHash));
//     offset += sizeof(visit_hash.m_secondHash);
//     memcpy(
//         static_cast<char*>(dest) + offset, &visit_hash.m_exp,
//         sizeof(visit_hash.m_exp));
//     offset += sizeof(visit_hash.m_exp);
//     memcpy(
//         static_cast<char*>(dest) + offset, &visit_hash.m_poolSize,
//         sizeof(visit_hash.m_poolSize));
//     offset += sizeof(visit_hash.m_poolSize);

//     size_t hash_table_size = (visit_hash.m_poolSize + 1) * 2;
//     // printf("hash_table_size %d offset %d\n", hash_table_size, offset);
//     memcpy(
//         static_cast<char*>(dest) + offset, visit_hash.m_hashTable.get(),
//         sizeof(OptHashPosVector::SizeType) * hash_table_size);
//     offset += sizeof(OptHashPosVector::SizeType) * hash_table_size;

#ifdef PROFILER
    profiler.serialize(static_cast<char*>(dest), offset);
#endif

    // Serialize top_candidates
    // printf("Serialize top_candidates\n");
    size_t top_candidates_size = top_candidates.size();
    memcpy(
        static_cast<char*>(dest) + offset, &top_candidates_size,
        sizeof(top_candidates_size));
    offset += sizeof(top_candidates_size);

    auto copy_queue = top_candidates;
    while (!copy_queue.empty()) {
      auto& elem = copy_queue.top();
      memcpy(static_cast<char*>(dest) + offset, &elem, sizeof(elem));
      offset += sizeof(elem);
      copy_queue.pop();
    }

    // Serialize candidate_set
    size_t candidate_set_size = candidate_set.size();
    memcpy(
        static_cast<char*>(dest) + offset, &candidate_set_size,
        sizeof(candidate_set_size));
    offset += sizeof(candidate_set_size);

    copy_queue = candidate_set;
    while (!copy_queue.empty()) {
      auto& elem = copy_queue.top();
      memcpy(static_cast<char*>(dest) + offset, &elem, sizeof(elem));
      offset += sizeof(elem);
      copy_queue.pop();
    }

    // Serialize result
    size_t result_size = result.size();
    memcpy(
        static_cast<char*>(dest) + offset, &result_size, sizeof(result_size));
    offset += sizeof(result_size);

    auto copy_result = result;
    while (!copy_result.empty()) {
      auto& elem = copy_result.top();
      memcpy(static_cast<char*>(dest) + offset, &elem, sizeof(elem));
      offset += sizeof(elem);
      copy_result.pop();
    }

    assert(offset <= MAX_QUERYBUFFER_SIZE);
  }

  // Deserialization function
  /*
    Construct QueryMsg from given pointer.
  */
  static QueryMsg* deserialize(const void* src) {
    QueryMsg* msg = new QueryMsg();
    size_t offset = 0;
    memcpy(&msg->ef, static_cast<const char*>(src) + offset, sizeof(msg->ef));
    offset += sizeof(msg->ef);
    memcpy(
        &msg->query_id, static_cast<const char*>(src) + offset,
        sizeof(msg->query_id));
    offset += sizeof(msg->query_id);

    memcpy(
        &msg->origin_machine, static_cast<const char*>(src) + offset,
        sizeof(msg->origin_machine));
    offset += sizeof(msg->origin_machine);
    memcpy(
        &msg->current_node_id, static_cast<const char*>(src) + offset,
        sizeof(msg->current_node_id));
    offset += sizeof(msg->current_node_id);

    memcpy(
        &msg->lowerBound, static_cast<const char*>(src) + offset,
        sizeof(msg->lowerBound));
    offset += sizeof(msg->lowerBound);
    memcpy(
        &msg->vector_size, static_cast<const char*>(src) + offset,
        sizeof(msg->vector_size));
    offset += sizeof(msg->vector_size);
    if(msg->vector == nullptr) msg->vector = (char*)malloc(msg->vector_size);
    memcpy(
        msg->vector, static_cast<const char*>(src) + offset,
        msg->vector_size);
    offset += msg->vector_size;
    memcpy(
        &msg->state, static_cast<const char*>(src) + offset,
        sizeof(msg->state));
    offset += sizeof(msg->state);

// #ifdef MSGQ
//     memcpy(
//         &msg->visit_hash.m_secondHash, static_cast<const char*>(src) + offset,
//         sizeof(msg->visit_hash.m_secondHash));
//     offset += sizeof(msg->visit_hash.m_secondHash);
//     memcpy(
//         &msg->visit_hash.m_exp, static_cast<const char*>(src) + offset,
//         sizeof(msg->visit_hash.m_exp));
//     offset += sizeof(msg->visit_hash.m_exp);
//     memcpy(
//         &msg->visit_hash.m_poolSize, static_cast<const char*>(src) + offset,
//         sizeof(msg->visit_hash.m_poolSize));
//     offset += sizeof(msg->visit_hash.m_poolSize);

//     size_t hash_table_size = (msg->visit_hash.m_poolSize + 1) * 2;
//     msg->visit_hash.m_hashTable.reset(
//         new OptHashPosVector::SizeType[hash_table_size]);
//     memcpy(
//         msg->visit_hash.m_hashTable.get(),
//         static_cast<const char*>(src) + offset,
//         sizeof(OptHashPosVector::SizeType) * hash_table_size);
//     offset += sizeof(OptHashPosVector::SizeType) * hash_table_size;

    // #ifdef PROFILER
    //     msg->profiler.deserialize(static_cast<const char*>(src), offset);
    // #endif

    // Deserialize top_candidates
    size_t top_candidates_size;
    memcpy(
        &top_candidates_size, static_cast<const char*>(src) + offset,
        sizeof(top_candidates_size));
    offset += sizeof(top_candidates_size);

    for (size_t i = 0; i < top_candidates_size; ++i) {
      std::pair<dist_t, uint32_t> elem;
      memcpy(&elem, static_cast<const char*>(src) + offset, sizeof(elem));
      offset += sizeof(elem);
      msg->top_candidates.push(elem);
    }

    // Deserialize candidate_set
    size_t candidate_set_size;
    memcpy(
        &candidate_set_size, static_cast<const char*>(src) + offset,
        sizeof(candidate_set_size));
    offset += sizeof(candidate_set_size);

    for (size_t i = 0; i < candidate_set_size; ++i) {
      std::pair<dist_t, uint32_t> elem;
      memcpy(&elem, static_cast<const char*>(src) + offset, sizeof(elem));
      offset += sizeof(elem);
      msg->candidate_set.push(elem);
    }

    // Deserialize result
    size_t result_size;
    memcpy(
        &result_size, static_cast<const char*>(src) + offset,
        sizeof(result_size));
    offset += sizeof(result_size);

    for (size_t i = 0; i < result_size; ++i) {
      std::pair<dist_t, size_t> elem;
      memcpy(&elem, static_cast<const char*>(src) + offset, sizeof(elem));
      offset += sizeof(elem);
      msg->result.push(elem);
    }

    // printf("offset: %d\n", offset);

    return msg;
  }
};

// TODO: polish this
template <typename dist_t>
class QueryMsg2 {
 public:
  size_t ef;
  size_t query_id;
  size_t origin_machine;
  size_t current_node_id;
  dist_t lowerBound;
  OptHashPosVector visit_hash;
  char vector[128];
  QueryState state;

  struct CompareByDist {
    constexpr bool operator()(
        VectorCache<dist_t> const& a,
        VectorCache<dist_t> const& b) const noexcept {
      return a.dist < b.dist;
    }
  };

  typedef QueryMsg<dist_t>::CompareByFirst CompareByFirst;

  std::priority_queue<
      VectorCache<dist_t>, std::vector<VectorCache<dist_t>>, CompareByDist>
      top_candidates;

  std::priority_queue<
      VectorCache<dist_t>, std::vector<VectorCache<dist_t>>, CompareByDist>
      candidate_set;
  std::priority_queue<std::pair<dist_t, size_t>> result;

  QueryMsg2() {}
  ~QueryMsg2() {}

  QueryMsg2(QueryState _state) : state(_state), origin_machine(0) {}

  QueryMsg2(char* _vector, size_t _query_id, size_t _ef, size_t _machine)
      : query_id(_query_id), ef(_ef), origin_machine(_machine) {
    memcpy(vector, _vector, 128);
    state = START;
  }

  void print_msg() {
    printf("ef %d\n", ef);
    printf("query_id %d\n", query_id);
    printf("current_node_id %d\n", current_node_id);
    printf("lowerBound %d\n", lowerBound);
    printf("top_candidates %d\n", top_candidates.size());
    printf("candidate_set %d\n", candidate_set.size());
    printf("result %d\n", result.size());
  }

  // Serialization function
  void serialize(void* dest) const {
    size_t offset = 0;
    memcpy(static_cast<char*>(dest) + offset, &ef, sizeof(ef));
    offset += sizeof(ef);
    memcpy(static_cast<char*>(dest) + offset, &query_id, sizeof(query_id));
    offset += sizeof(query_id);
    memcpy(
        static_cast<char*>(dest) + offset, &origin_machine,
        sizeof(origin_machine));
    offset += sizeof(origin_machine);

    memcpy(
        static_cast<char*>(dest) + offset, &current_node_id,
        sizeof(current_node_id));
    offset += sizeof(current_node_id);
    memcpy(static_cast<char*>(dest) + offset, &lowerBound, sizeof(lowerBound));
    offset += sizeof(lowerBound);
    memcpy(static_cast<char*>(dest) + offset, vector, sizeof(vector));
    offset += sizeof(vector);
    memcpy(static_cast<char*>(dest) + offset, &state, sizeof(state));
    offset += sizeof(state);

    // Serialize visit_hash
    // printf("Serialize visit_hash\n");
    memcpy(
        static_cast<char*>(dest) + offset, &visit_hash.m_secondHash,
        sizeof(visit_hash.m_secondHash));
    offset += sizeof(visit_hash.m_secondHash);
    memcpy(
        static_cast<char*>(dest) + offset, &visit_hash.m_exp,
        sizeof(visit_hash.m_exp));
    offset += sizeof(visit_hash.m_exp);
    memcpy(
        static_cast<char*>(dest) + offset, &visit_hash.m_poolSize,
        sizeof(visit_hash.m_poolSize));
    offset += sizeof(visit_hash.m_poolSize);

    size_t hash_table_size = (visit_hash.m_poolSize + 1) * 2;
    // printf("hash_table_size %d offset %d\n", hash_table_size, offset);
    memcpy(
        static_cast<char*>(dest) + offset, visit_hash.m_hashTable.get(),
        sizeof(OptHashPosVector::SizeType) * hash_table_size);
    offset += sizeof(OptHashPosVector::SizeType) * hash_table_size;

    // Serialize top_candidates
    // printf("Serialize top_candidates\n");
    size_t top_candidates_size = top_candidates.size();
    memcpy(
        static_cast<char*>(dest) + offset, &top_candidates_size,
        sizeof(top_candidates_size));
    offset += sizeof(top_candidates_size);

    auto copy_queue = top_candidates;
    while (!copy_queue.empty()) {
      auto& elem = copy_queue.top();
      memcpy(static_cast<char*>(dest) + offset, &elem, sizeof(elem));
      offset += sizeof(elem);
      copy_queue.pop();
    }

    // Serialize candidate_set
    size_t candidate_set_size = candidate_set.size();
    memcpy(
        static_cast<char*>(dest) + offset, &candidate_set_size,
        sizeof(candidate_set_size));
    offset += sizeof(candidate_set_size);

    copy_queue = candidate_set;
    while (!copy_queue.empty()) {
      auto& elem = copy_queue.top();
      memcpy(static_cast<char*>(dest) + offset, &elem, sizeof(elem));
      offset += sizeof(elem);
      copy_queue.pop();
    }

    // Serialize result
    size_t result_size = result.size();
    memcpy(
        static_cast<char*>(dest) + offset, &result_size, sizeof(result_size));
    offset += sizeof(result_size);

    auto copy_result = result;
    while (!copy_result.empty()) {
      auto& elem = copy_result.top();
      memcpy(static_cast<char*>(dest) + offset, &elem, sizeof(elem));
      offset += sizeof(elem);
      copy_result.pop();
    }
  }

  // Deserialization function
  /*
    Construct QueryMsg from given pointer.
  */
  static QueryMsg2* deserialize(const void* src) {
    QueryMsg2* msg = new QueryMsg2();
    size_t offset = 0;
    memcpy(&msg->ef, static_cast<const char*>(src) + offset, sizeof(msg->ef));
    offset += sizeof(msg->ef);
    memcpy(
        &msg->query_id, static_cast<const char*>(src) + offset,
        sizeof(msg->query_id));
    offset += sizeof(msg->query_id);

    memcpy(
        &msg->origin_machine, static_cast<const char*>(src) + offset,
        sizeof(msg->origin_machine));
    offset += sizeof(msg->origin_machine);
    memcpy(
        &msg->current_node_id, static_cast<const char*>(src) + offset,
        sizeof(msg->current_node_id));
    offset += sizeof(msg->current_node_id);

    memcpy(
        &msg->lowerBound, static_cast<const char*>(src) + offset,
        sizeof(msg->lowerBound));
    offset += sizeof(msg->lowerBound);
    memcpy(
        msg->vector, static_cast<const char*>(src) + offset,
        sizeof(msg->vector));
    offset += sizeof(msg->vector);
    memcpy(
        &msg->state, static_cast<const char*>(src) + offset,
        sizeof(msg->state));
    offset += sizeof(msg->state);

    // Deserialize visit_hash
    memcpy(
        &msg->visit_hash.m_secondHash, static_cast<const char*>(src) + offset,
        sizeof(msg->visit_hash.m_secondHash));
    offset += sizeof(msg->visit_hash.m_secondHash);
    memcpy(
        &msg->visit_hash.m_exp, static_cast<const char*>(src) + offset,
        sizeof(msg->visit_hash.m_exp));
    offset += sizeof(msg->visit_hash.m_exp);
    memcpy(
        &msg->visit_hash.m_poolSize, static_cast<const char*>(src) + offset,
        sizeof(msg->visit_hash.m_poolSize));
    offset += sizeof(msg->visit_hash.m_poolSize);

    size_t hash_table_size = (msg->visit_hash.m_poolSize + 1) * 2;
    msg->visit_hash.m_hashTable.reset(
        new OptHashPosVector::SizeType[hash_table_size]);
    memcpy(
        msg->visit_hash.m_hashTable.get(),
        static_cast<const char*>(src) + offset,
        sizeof(OptHashPosVector::SizeType) * hash_table_size);
    offset += sizeof(OptHashPosVector::SizeType) * hash_table_size;

    // Deserialize top_candidates
    size_t top_candidates_size;
    memcpy(
        &top_candidates_size, static_cast<const char*>(src) + offset,
        sizeof(top_candidates_size));
    offset += sizeof(top_candidates_size);

    for (size_t i = 0; i < top_candidates_size; ++i) {
      std::pair<dist_t, uint32_t> elem;
      memcpy(&elem, static_cast<const char*>(src) + offset, sizeof(elem));
      offset += sizeof(elem);
      msg->top_candidates.push(elem);
    }

    // Deserialize candidate_set
    size_t candidate_set_size;
    memcpy(
        &candidate_set_size, static_cast<const char*>(src) + offset,
        sizeof(candidate_set_size));
    offset += sizeof(candidate_set_size);

    for (size_t i = 0; i < candidate_set_size; ++i) {
      std::pair<dist_t, uint32_t> elem;
      memcpy(&elem, static_cast<const char*>(src) + offset, sizeof(elem));
      offset += sizeof(elem);
      msg->candidate_set.push(elem);
    }

    // Deserialize result
    size_t result_size;
    memcpy(
        &result_size, static_cast<const char*>(src) + offset,
        sizeof(result_size));
    offset += sizeof(result_size);

    for (size_t i = 0; i < result_size; ++i) {
      std::pair<dist_t, size_t> elem;
      memcpy(&elem, static_cast<const char*>(src) + offset, sizeof(elem));
      offset += sizeof(elem);
      msg->result.push(elem);
    }

    // printf("offset: %d\n", offset);

    return msg;
  }
};
