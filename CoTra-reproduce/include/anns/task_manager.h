#pragma once

#include <stdlib.h>

#include <cassert>
#include <deque>

#include "../rdma/rdma_param.h"
#include "anns/anns_param.h"
#include "anns/query_msg.h"
#include "anns/scala_scheduler.h"
#include "anns/vec_buffer.h"

/**
 * Manage tasks and results.
 */
template <typename dist_t>
class TaskManager {
 public:
  // temp
  // uint32_t post_async, post_task;
  // uint32_t rpost_async, rpost_task;

  uint32_t local_id;

  // my_machine_id: local task, others id: to send task.
  std::vector<ComputeTask<dist_t>> task_queue[MACHINE_NUM];
  std::vector<NodeTask> node_task_queue[MACHINE_NUM];
  std::vector<NodeTask> cur_node_queue;
  std::vector<NodeResult<dist_t>> node_result_queue[MACHINE_NUM];
  // send from non-core to core machine sync msg.
  std::vector<NonCoreSyncMsg<dist_t>> noncore_sync_msg[MACHINE_NUM];

  std::vector<ComputeTask<dist_t>> cur_task_queue;
  std::vector<ComputeTask<dist_t>> next_task_queue;

  // Store the query id in pre-stage.
  std::vector<size_t> async_task_queue;

  // my_machine_id: recved compute vectors, others id: to send compute.
  std::vector<ComputeMsg> compute_queue[MACHINE_NUM];

  // my_machine_id: recved results, others id: to send results.
  // Only for query in pre-stage.
  std::vector<ResultMsg<dist_t>> result_queue[MACHINE_NUM];

  // global balance queue pointer.
  BalanceQueue<ComputeTask<dist_t>>* ba_task_queue;

  // scala v1, multi-query
  // std::deque<uint32_t> avail_query_queue;
  // std::deque<uint32_t> cur_query_queue;
  uint32_t cur_query_num{0};

  // For migration.
  QueryMsg<dist_t>** global_query;
  std::vector<size_t> query_queue;

  std::mutex schequery_mutex;
  std::vector<ScheduleMsg> schedule_query_queue;

  std::vector<uint32_t> subquery_queue;  // store subquery ids.
  std::vector<uint32_t> next_subquery_queue;

  // for b2 kmeans
  std::map<uint32_t, QueryMsg<dist_t>*> ongoing_queries;
  // ongoing query dispatched from other machines.
  std::vector<uint32_t> b2_query_queue;
  // finished query queue.
  std::vector<uint32_t> b2_finished_queue;

  void swap_task_queue() {
    std::swap(cur_task_queue, next_task_queue);
    next_task_queue.clear();
    std::swap(subquery_queue, next_subquery_queue);
    next_subquery_queue.clear();
    return;
  }

  void serialize_task(uint32_t machine_id, char* ptr, uint64_t& offset = 0) {
    assert(machine_id != local_id);

    // Local id to inform remote machine.
    memcpy(ptr + offset, &local_id, sizeof(local_id));
    offset += sizeof(local_id);

    uint32_t t_size = task_queue[machine_id].size();
    memcpy(ptr + offset, &t_size, sizeof(t_size));
    offset += sizeof(t_size);

    for (ComputeTask<dist_t>& t : task_queue[machine_id]) {
      t.serialize(ptr, offset);
    }

    t_size = node_task_queue[machine_id].size();
    memcpy(ptr + offset, &t_size, sizeof(t_size));
    offset += sizeof(t_size);
    for (NodeTask& t : node_task_queue[machine_id]) {
      t.serialize(ptr, offset);
    }

    if (offset > MAX_QUERYBUFFER_SIZE) {
      printf("Error: task buffer size exceed limit.\n");
      abort();
    }
    return;
  }

  /**
   * Deserialize task from buffer.
   */
  void deserialize_task(char* ptr) {
    uint64_t offset = 0;
    uint32_t send_mid;
    memcpy(&send_mid, ptr + offset, sizeof(send_mid));
    offset += sizeof(send_mid);
    // printf("task msg: send mid: %u, offset: %u\n", send_mid, offset);
    uint32_t t_size;
    memcpy(&t_size, ptr + offset, sizeof(t_size));
    offset += sizeof(t_size);
    // printf("t_size: %u offset %u\n", t_size, offset);
    for (uint32_t t = 0; t < t_size; t++) {
      // NOTE: task_queue local_id for recv task, remote_id for send tasks.
      cur_task_queue.emplace_back(ptr, offset);  // should not !!!
    }

    uint32_t n_size;
    memcpy(&n_size, ptr + offset, sizeof(n_size));
    offset += sizeof(n_size);
    // printf("get %d task %d node_task from m%d\n", t_size, n_size, send_mid);
    for (uint32_t t = 0; t < n_size; t++) {
      cur_node_queue.emplace_back(ptr, offset);
    }

    return;
  }

  /**
   * Serialize result in buffer.
   */
  void serialize_result(uint32_t machine_id, char* ptr, uint64_t& offset = 0) {
    assert(machine_id != local_id);

    // Local id to inform remote machine.
    memcpy(ptr + offset, &local_id, sizeof(local_id));
    offset += sizeof(local_id);

    uint32_t r_size = result_queue[machine_id].size();
    memcpy(ptr + offset, &r_size, sizeof(r_size));
    offset += sizeof(r_size);

    for (ResultMsg<dist_t>& r : result_queue[machine_id]) {
      r.serialize(ptr, offset);
      if (offset >= MAX_QUERYBUFFER_SIZE) {
        printf("Error: BUFFER size large than limit.\n");
        printf(
            "mid: %llu size %llu offset: %llu\n", machine_id,
            result_queue[machine_id].size(), offset);
        abort();
      }
    }
    assert(offset <= MAX_QUERYBUFFER_SIZE);
    return;
  }

  /**
   * Deserialize result from buffer.
   */
  void deserialize_result(char* ptr) {
    uint64_t offset = 0;
    uint32_t* uint_ptr = (uint32_t*)(ptr + offset);
    uint32_t send_mid = uint_ptr[0];

    offset += sizeof(uint32_t);
    uint32_t r_size = uint_ptr[1];
    // printf("recv result %u from m%u\n", r_size, send_mid);
    offset += sizeof(uint32_t);

    for (uint32_t r = 0; r < r_size; r++) {
      // NOTE: result_queue local_id for recv results, remote_id for send
      // results.
      result_queue[local_id].emplace_back(ptr, offset);

      uint32_t sz = result_queue[local_id].size();
      auto& msg = result_queue[local_id][sz - 1];
      // if (global_query[msg.qid]->state == END) {
      //   printf("Error: q%u is END\n", msg.qid);
      //   abort();
      // }
    }
    // printf("res offset %u\n", offset);
    return;
  }

  /**
   * Serialize node result in buffer.
   */
  void serialize_node_result(
      uint32_t machine_id, char* ptr, uint64_t& offset = 0) {
    assert(machine_id != local_id);

    // Local id to inform remote machine.
    memcpy(ptr + offset, &local_id, sizeof(local_id));
    offset += sizeof(local_id);

    uint32_t r_size = node_result_queue[machine_id].size();
    memcpy(ptr + offset, &r_size, sizeof(r_size));
    offset += sizeof(r_size);

    for (NodeResult<dist_t>& r : node_result_queue[machine_id]) {
      r.serialize(ptr, offset);
      if (offset >= MAX_QUERYBUFFER_SIZE) {
        printf("Error: BUFFER size large than limit.\n");
        printf(
            "mid: %llu size %llu offset: %llu\n", machine_id,
            node_result_queue[machine_id].size(), offset);
        abort();
      }
    }
    assert(offset <= MAX_QUERYBUFFER_SIZE);
    return;
  }

  /**
   * Deserialize node result from buffer.
   */
  void deserialize_node_result(char* ptr) {
    uint64_t offset = 0;
    uint32_t* uint_ptr = (uint32_t*)(ptr + offset);
    uint32_t send_mid = uint_ptr[0];

    offset += sizeof(uint32_t);
    uint32_t r_size = uint_ptr[1];
    // printf("recv result %u from m%u\n", r_size, send_mid);
    offset += sizeof(uint32_t);

    for (uint32_t r = 0; r < r_size; r++) {
      // NOTE: result_queue local_id for recv results, remote_id for send
      // results.
      node_result_queue[local_id].emplace_back(ptr, offset);
      // auto& msg = node_result_queue[local_id].back();
      // if (global_query[msg.qid]->state == END) {
      //   printf("Error: q%u is END\n", msg.qid);
      //   abort();
      // }
    }
    // printf("res offset %u\n", offset);
    return;
  }

  /**
   * Serialize node result in buffer.
   */
  void serialize_node_sync(
      uint32_t machine_id, char* ptr, uint64_t& offset = 0) {
    assert(machine_id != local_id);

    // Local id to inform remote machine.
    memcpy(ptr + offset, &local_id, sizeof(local_id));
    offset += sizeof(local_id);

    uint32_t size = noncore_sync_msg[machine_id].size();
    memcpy(ptr + offset, &size, sizeof(size));
    offset += sizeof(size);

    for (NonCoreSyncMsg<dist_t>& msg : noncore_sync_msg[machine_id]) {
      msg.serialize(ptr, offset);
      if (offset >= MAX_QUERYBUFFER_SIZE) {
        printf("Error: BUFFER size large than limit.\n");
        printf(
            "mid: %llu size %llu offset: %llu\n", machine_id,
            noncore_sync_msg[machine_id].size(), offset);
        abort();
      }
    }
    assert(offset <= MAX_QUERYBUFFER_SIZE);
    return;
  }

  /**
   * Deserialize node result from buffer.
   */
  void deserialize_node_sync(char* ptr) {
    uint64_t offset = 0;
    uint32_t remote_id;
    memcpy(&remote_id, ptr + offset, sizeof(remote_id));
    offset += sizeof(remote_id);

    uint32_t size;
    memcpy(&size, ptr + offset, sizeof(size));
    offset += sizeof(size);

    for (uint32_t i = 0; i < size; i++) {
      NonCoreSyncMsg<dist_t> sync_msg(ptr, offset);
      global_query[sync_msg.qid]->update_sub_query_from_noncore(
          local_id, &sync_msg);
      // avoke the query.
      if (global_query[sync_msg.qid]->state == PAUSE) {
        // printf("sync voke q%u\n", sync_msg.qid);
        global_query[sync_msg.qid]->state = POST_STAGE;
        subquery_queue.emplace_back(sync_msg.qid);
      }
    }

    return;
  }

  /**
   * push new compute task to queue and ready to send to remote machines.
   * If the queue size exceed the limitation of queue.
   */
  void push_compute(
      uint32_t machine_id, uint32_t query_id,
      std::vector<std::pair<uint32_t, char*>>& id_ptr) {
    abort(); // TODO: this func never use for scala mode.
    compute_queue[machine_id].emplace_back(query_id, 0, 136);  // qid, count,
    size_t cmp_post_sz = compute_queue[machine_id].size();
    ComputeMsg* cmp_msg = &compute_queue[machine_id][cmp_post_sz - 1];

    for (auto& vec_id_ptr : id_ptr) {
      if (cmp_msg->vec_id.size() >= MAX_SGE_NUM - 1) {
        // printf("compute sg num %u..\n", cmp_msg->vec_id.size());
        // exceed the limitation, alloc new compute_msg
        compute_queue[machine_id].emplace_back(query_id, 0, 136);
        cmp_post_sz = compute_queue[machine_id].size();
        cmp_msg = &compute_queue[machine_id][cmp_post_sz - 1];
      }
      cmp_msg->vec_id.emplace_back(vec_id_ptr.first);
      cmp_msg->vec_ptr.emplace_back(vec_id_ptr.second);
      cmp_msg->sg_size.emplace_back(cmp_msg->vec_size);
    }
    // Set the last compute count to true.
    cmp_msg->is_count = 1;
    // printf("compute sg num %u..\n", cmp_msg->vec_id.size());
    return;
  }

  /**
   * Serialize meta data in buffer.
   * Including: qid, is_count, compute_num, vec_id list
   */
  void serialize_compute(ComputeMsg& cmp_msg, char* ptr, size_t& offset = 0) {
    cmp_msg.serialize(ptr, offset);
    return;
  }

  /**
   * Deserialize
   */
  void deserialize_compute(char* ptr) {
    compute_queue[local_id].emplace_back(ptr);
    return;
  }

  void deserialize_query(char* ptr) {
    size_t offset = 0;
    offset += sizeof(size_t);
    size_t qid = 0;
    memcpy(&qid, static_cast<const char*>(ptr) + offset, sizeof(qid));
    // printf("deser query %u\n", qid);
    global_query[qid]->deserialize_and_merge(ptr);
    // printf("deser query %u over\n", qid);
    query_queue.push_back(qid);
    return;
  }

  void deserialize_core_info(char* ptr) {
    // printf("deser query %u\n", qid);
    // printf("deser query %u over\n", qid);
    schequery_mutex.lock();
    schedule_query_queue.emplace_back(ptr);
    schequery_mutex.unlock();
    return;
  }

  /**
   * recv new sync : update sub-query.
   * TODO: may mv to scala search
   */
  void deserialize_sync(char* ptr) {
    SyncMsg<dist_t> sync_msg(ptr);
    global_query[sync_msg.qid]->update_sub_query(
        local_id, &sync_msg, ba_task_queue);
    // avoke the query.
    if (global_query[sync_msg.qid]->state == PAUSE) {
      // printf("sync voke q%u\n", sync_msg.qid);
      global_query[sync_msg.qid]->state = POST_STAGE;
      subquery_queue.emplace_back(sync_msg.qid);
    }
    return;
  }
  std::vector<uint32_t>& get_subquery_queue() { return subquery_queue; }

  std::vector<ComputeTask<dist_t>>& get_local_task_queue() {
    return cur_task_queue;
  }

  std::vector<NodeTask>& get_node_task_queue() { return cur_node_queue; }

  std::vector<NodeResult<dist_t>>& get_node_result_queue() {
    return node_result_queue[local_id];
  }

  std::vector<size_t>& get_async_task_queue() { return async_task_queue; }

  std::vector<ResultMsg<dist_t>>& get_result_queue() {
    return result_queue[local_id];
  }

  std::vector<size_t>& get_query_queue() { return query_queue; }

  std::vector<ComputeMsg>& get_compute_queue() {
    return compute_queue[local_id];
  }

  void clear() {
    for (uint32_t m = 0; m < MACHINE_NUM; m++) {
      task_queue[m].clear();
      result_queue[m].clear();
    }
    cur_task_queue.clear();
    next_task_queue.clear();
    // result_queue.clear();
    // while (cur_query_queue.size()) {
    //   cur_query_queue.pop_front();
    // }
    cur_query_num = 0;
    // for (uint32_t i = 0; i < QUERY_GROUP_SIZE; i++) {
    //   avail_query_queue.push_back(i);
    // }
    return;
  }

  TaskManager(uint32_t _local_id) : local_id(_local_id) {
    // for (uint32_t i = 0; i < QUERY_GROUP_SIZE; i++) {
    //   avail_query_queue.push_back(i);
    // }
    // TODO: use macro
    cur_task_queue.reserve(128);
    next_task_queue.reserve(128);
    for (uint32_t i = 0; i < MACHINE_NUM; i++) {
      task_queue[i].reserve(128);
      result_queue[i].reserve(128);
    }
    // result_queue.reserve(128);
  }
  ~TaskManager() {}

  void add_remote_task(
      uint32_t ngh_id, uint32_t machine_id, uint32_t query_id) {
    node_task_queue[machine_id].emplace_back(
        local_id, query_id, ngh_id, global_query[query_id]->is_core_machine);
    return;
  }
};
