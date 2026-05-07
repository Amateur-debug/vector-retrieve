#pragma once
#include "anns/scala_scheduler.h"
#include "anns/task_manager.h"
#include "anns/vec_buffer.h"
#include "coromem/include/backend.h"
#include "coromem/include/galois/loops.h"
#include "index/index_param.h"
#include "rdma_context.h"
#include "server_connect.h"

#define KEY_MSG_SIZE (59 + 24 + 2) /* Message size without gid. */
#define KEY_MSG_SIZE_GID (108)     /* Message size with gid (MGID as well). */
#define SYNC_SPEC_ID (5)
#define KEY_PRINT_FMT "%04x:%04x:%06x:%06x:%08x:%016llx:%08x:%016llx:%08x"
#define KEY_PRINT_FMT_GID                                                      \
  "%04x:%04x:%06x:%06x:%08x:%016llx:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%" \
  "02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%08x:"

class RdmaCommunication {
 public:
  ServerConnect *sc{nullptr};

  PerThreadStorage<RdmaContext> rdma_ctx;
  PerThreadStorage<VectorBuffer> read_buf;

  PerThreadStorage<SendWriteBuffer> send_write_buf;
  PerThreadStorage<RecvWriteBuffer> recv_write_buf;

  // PerThreadStorage<SendBuffer> send_buf;
  // PerThreadStorage<RecvBuffer> recv_buf;

  // TODO: move to anns param
  RdmaParameter rdma_param;

  // global query batch info.
  size_t vector_size;
  size_t correct{0}, total{0}, query_load{0};
  double query_lat_sum{0.0};
  // for comp eff profiling.
  size_t all_computation_cnt{0};

  RdmaCommunication() {}
  ~RdmaCommunication() {}

  // Indicator for termination.
  bool term[MACHINE_NUM];

  int ethernet_leader_connect();
  int ethernet_member_connect(int leader_id);
  int establish_connection();

  int set_up_connection(RdmaContext *ctx);
  int ethernet_read_keys(struct MessageContext *rem_dest, int sockdfd);
  int ethernet_write_keys(struct MessageContext *my_dest, int sockdfd);

  int ctx_hand_shake(
      struct MessageContext *my_dest, struct MessageContext *rem_dest,
      int sockdfd);
  int ctx_close_connection();

  void com_init(
      char *vec_part_ptr, size_t vec_part_len, RdmaParameter &rdma_param);

  /**
   * Poll cq. Include send cq and recv cq.
   */
  /* Try to collect the send cq, return the number of collected elements. */
  int poll_send();
  /* Try to collect the recv cq, return the number of collected elements. */
  int poll_recv();

  /**
   * Read operation. Include sync read and async read.
   */
  void com_read_init();
  /* Sync read */
  char *sync_read(int machine_id, size_t internal_id, uint32_t size);
  /* Async read */
  int post_read(
      std::vector<BufferCache> &vectors, size_t vec_size,
      uint32_t query_id = 0);
  /* Sync wait post read request */
  int poll_read();
  void release_cache(int buffer_id);
  char *fetch_vector(uint32_t internal_id, size_t machine_id, size_t vec_size);

  /**
   * Write operation. Include sync write and async write.
   */
  void com_write_init();
  /* Async send/recv write */
  int get_write_send_buf(
      int machine_id, bool send_release = false, bool pop = true);
  int post_write_send(
      int machine_id, uint32_t buf_id, uint32_t size, ControlType control);
  int post_write_recv(int machine_id, uint32_t buf_id);
  int post_sg_write_send(
      int machine_id, uint32_t buf_id, std::vector<char *> &sg_ptr,
      std::vector<uint32_t> &size, ControlType control);
  void send_write_release(int machine_id, uint32_t buf_id);
  void clear_write_release();

  /**
   * Send/Recv operation. Include sync send/recv and async send/recv.
   */
  void com_send_init();
  /* Sync send/recv */
  int sync_send(int machine_id, size_t offset, uint32_t size);
  int sync_recv(uint32_t recv_num = 1);
  /* Async send/recv */
  int post_send(int machine_id, size_t buff_id, uint32_t size);
  int post_recv(int machine_id, size_t buff_id, uint32_t size);

  template <typename dist_t>
  void collect_recv_query(std::deque<QueryMsg<dist_t> *> &ret) {
    auto *rbuf = recv_write_buf.getLocal();
    // Try to collect recv queue once.
    poll_recv();
    while (rbuf->migrate_queue.size() > 0) {
      char *query_ptr = rbuf->migrate_queue.front();
      rbuf->migrate_queue.pop_front();
      QueryMsg<dist_t> *new_query = QueryMsg<dist_t>::deserialize(query_ptr);
      // printf("[Recv migrate q%d]\n", new_query->query_id);
      ret.push_back(new_query);
      free(query_ptr);
    }
    return;
  }

  template <typename dist_t>
  void poll_task_result(
      TaskManager<dist_t> &tman, Profiler *profiler = nullptr) {
    auto *rbuf = recv_write_buf.getLocal();  // recv buffer
    auto *rd_buf = read_buf.getLocal();      // read buffer
    // TODO: general api, transfer the de/serialize data.

    // printf("poll;;\n");

    poll_send();
    poll_recv();

    // printf("collect queues\n");
    // Collect recved tasks.
    while (rbuf->recv_task.size() > 0) {
#ifdef PROFILER
      profiler->start("g_poll_proc");
#endif
      char *task_ptr = rbuf->recv_task.front();
      rbuf->recv_task.pop_front();
      tman.deserialize_task(task_ptr);  // impl
      // printf("[Recv migrate q%d]\n", new_query->query_id);
      free(task_ptr);
#ifdef PROFILER
      profiler->end("g_poll_proc");
#endif
    }

    // printf("collect results\n");
    // Collect recved results.
    while (rbuf->recv_result.size() > 0) {
#ifdef PROFILER
      profiler->start("g_poll_proc");
#endif
      char *res_ptr = rbuf->recv_result.front();
      rbuf->recv_result.pop_front();
      // printf("dese res\n");
      tman.deserialize_result(res_ptr);
      free(res_ptr);
// printf("deser res over\n");
#ifdef PROFILER
      profiler->end("g_poll_proc");
#endif
    }

    while (rbuf->recv_node_res.size() > 0) {
      char *res_ptr = rbuf->recv_node_res.front();
      rbuf->recv_node_res.pop_front();
      // printf("dese res\n");
      tman.deserialize_node_result(res_ptr);
      free(res_ptr);
    }

    while (rbuf->recv_node_sync.size() > 0) {
      char *sync_ptr = rbuf->recv_node_sync.front();
      rbuf->recv_node_sync.pop_front();
      // printf("dese res\n");
      tman.deserialize_node_sync(sync_ptr);
      free(sync_ptr);
    }

    // printf("collect compute\n");
    while (rbuf->recv_compute.size() > 0) {
#ifdef PROFILER
      profiler->start("g_poll_proc");
#endif
      char *cmp_ptr = rbuf->recv_compute.front();
      rbuf->recv_compute.pop_front();
      // printf("dese res\n");
      tman.deserialize_compute(cmp_ptr);  // impl
// NOTE: should not free this buffer untill the compute is over.
// free(cmp_ptr);
#ifdef PROFILER
      profiler->end("g_poll_proc");
#endif
    }

    // printf("collect migrate query\n");
    // Collect migrated queries.
    while (rbuf->migrate_queue.size() > 0) {
#ifdef PROFILER
      profiler->start("g_poll_proc");
#endif
      // printf("collect query_ptr\n");
      char *query_ptr = rbuf->migrate_queue.front();
      rbuf->migrate_queue.pop_front();
      tman.deserialize_query(query_ptr);  // impl
      // printf("[Recv migrate q%d]\n", new_query->query_id);
      free(query_ptr);
#ifdef PROFILER
      profiler->end("g_poll_proc");
#endif
    }

    while (rbuf->core_info_queue.size() > 0) {
#ifdef PROFILER
      profiler->start("g_poll_proc");
#endif
      char *core_ptr = rbuf->core_info_queue.front();
      rbuf->core_info_queue.pop_front();
      tman.deserialize_core_info(core_ptr);
      free(core_ptr);
#ifdef PROFILER
      profiler->end("g_poll_proc");
#endif
    }

    // printf("collect sync\n");
    // Collect sub-queries sync msg.
    while (rbuf->sync_queue.size() > 0) {
#ifdef PROFILER
      profiler->start("g_poll_proc");
#endif
      char *sync_ptr = rbuf->sync_queue.front();
      rbuf->sync_queue.pop_front();
      tman.deserialize_sync(sync_ptr);  // impl
      free(sync_ptr);
#ifdef PROFILER
      profiler->end("g_poll_proc");
#endif
    }

    // printf("collect async\n");
    // Async read list.
    while (rd_buf->recv_list.size() > 0) {
#ifdef PROFILER
      profiler->start("g_poll_proc");
#endif
      uint32_t qid = (rd_buf->recv_list.front() >> 16);
      uint32_t buf_id = (rd_buf->recv_list.front() & 0xffff);
      // printf("collect async q%u buf%u\n", qid, buf_id);
      tman.global_query[qid]->async_read_cnt++;
      rd_buf->recv_list.pop_front();
      tman.global_query[qid]->recv_async_read_queue.push_back(
          rd_buf->bufcache_ptr[buf_id]);
      // if query in pre-stage
      if (tman.global_query[qid]->state == PRE_STAGE) {
        if (tman.global_query[qid]->async_read_cnt ==
            tman.global_query[qid]->async_read_num) {
          // recved all async read.
          tman.async_task_queue.emplace_back(qid);
        }
      } else if (tman.global_query[qid]->state == PAUSE) {
        // printf("async voke q%u\n", qid);
        tman.subquery_queue.push_back(qid);
        tman.global_query[qid]->state = POST_STAGE;
      }
#ifdef PROFILER
      profiler->end("g_poll_proc");
#endif
    }

    // printf("collect over\n");

    return;
  }

  // For Baseline1 migration.
  template <typename dist_t>
  void migrate_query(size_t machine_id, QueryMsg<dist_t> *query) {
    auto *buf = send_write_buf.getLocal();
    // Collect the sent ack and release the position.
    uint32_t buf_id = get_write_send_buf(machine_id);
    // printf(
    //     "[migrate q%d] to m%d buf %d\n", query->query_id, machine_id,
    //     buf_id);
    // query->serialize(buf->buffer[machine_id][buf_id]);
    // New migration: no need for visit list.
    size_t len = query->serialize2(buf->buffer[machine_id][buf_id]);
    // post_write_send(machine_id, buf_id, len, QUERY);
    post_write_send(machine_id, buf_id, len, QUERY);
  }

  /**
   * Fork query to other machine.
   */
  template <typename dist_t>
  void fork_query(size_t machine_id, QueryMsg<dist_t> *query) {
    auto *buf = send_write_buf.getLocal();
    // Collect the sent ack and release the position.
    uint32_t buf_id = get_write_send_buf(machine_id);
    // New migration: no need for visit list.
    size_t len = query->fork(machine_id, buf->buffer[machine_id][buf_id]);
    if(len >= MAX_QUERYBUFFER_SIZE){
      printf("Error: query msg exceed the buffer size.\n");
      abort();
    }
    // post_write_send(machine_id, buf_id, len, QUERY);
    post_write_send(machine_id, buf_id, len, QUERY);
  }

  /**
   * Send query core machine info for scheduler.
   */
  template <typename dist_t>
  void send_core_machine_info(
      TaskManager<dist_t> &tman, QueryMsg<dist_t> *query) {
    uint32_t leader_m = 0;  // leader machine id.
    if (rdma_param.machine_id == leader_m) {
      tman.schedule_query_queue.emplace_back(
          query->query_id, query->core_machine);
    } else {
      auto *buf = send_write_buf.getLocal();
      uint32_t buf_id = get_write_send_buf(leader_m);
      ScheduleMsg msg(query->query_id, query->core_machine);
      size_t len = msg.serialize(buf->buffer[leader_m][buf_id]);
      post_write_send(leader_m, buf_id, len, CORE_INFO);
    }
    return;
  }

  /**
   * Send scheduler plan to other machine.
   */
  void send_schedule_plan(ScalaScheduler &scheduler) { return; }

  template <typename dist_t>
  void post_subquery_sync(
      QueryMsg<dist_t> *query, bool prop_token = false, bool taint = false) {
    auto *buf = send_write_buf.getLocal();
    uint32_t core_m_sz = query->core_machine.size();
    uint32_t nxt_prop_m;
    query->sync_msg.qid = query->query_id;

    // TODO: can put into init.
    for (uint32_t m_idx = 0; m_idx < core_m_sz; m_idx++) {
      if (query->core_machine[m_idx] == rdma_param.machine_id) {
        nxt_prop_m = query->core_machine[(m_idx + 1) % core_m_sz];
        break;
      }
    }
    for (uint32_t m : query->core_machine) {
      if (m == rdma_param.machine_id) {  // this machine.
        continue;
      } else {  // remote machine.
        uint32_t buf_id = get_write_send_buf(m);

        if (prop_token && (nxt_prop_m == m) && query->has_token) {
          // if prop_token, update the term.

          query->sync_msg.update_term = true;
          // give the token to nxt core machine.
          query->sync_msg.has_token = true;
          query->sync_msg.token_is_black = taint;
          query->has_token = 0;
          // printf("prop token to nxt core machine m%u\n", nxt_prop_m);
          // printf(
          //     "]]%u %u\n", query->sync_msg.update_term,
          //     query->sync_msg.has_token);
        } else {
          query->sync_msg.update_term = 0;
          query->sync_msg.has_token = 0;
        }
        size_t len = query->sync_msg.serialize(buf->buffer[m][buf_id]);
        post_write_send(m, buf_id, len, SYNC);
        // printf("q%u post sync to m%u\n", query->query_id, m);
      }
    }
    return;
  }

  // For ScalaGraph task push.
  template <typename dist_t>
  void post_remote_task(
      TaskManager<dist_t> &tman, uint32_t query_id, size_t read_size,
      Profiler *profiler = nullptr) {
    auto *buf = send_write_buf.getLocal();

    // post async read
    tman.global_query[query_id]->async_read_num =
        tman.global_query[query_id]->async_read_queue.size();

    // TODO: use macro
    if (tman.global_query[query_id]->async_read_num > 0) {
      tman.global_query[query_id]->async_read_cnt = 0;
      post_read(
          tman.global_query[query_id]->async_read_queue, read_size, query_id);
      tman.global_query[query_id]->async_read_queue.clear();
    }

    // post write task
    for (uint32_t m = 0; m < MACHINE_NUM; m++) {
      if (m == tman.local_id) continue;
      // Send remote task & node.
      if (tman.task_queue[m].size() > 0 || tman.node_task_queue[m].size() > 0) {
        uint32_t buf_id = get_write_send_buf(m);
        uint64_t offset = 0;
        tman.serialize_task(m, buf->buffer[m][buf_id], offset);
        // printf("q%u post task to m%d buf%u\n", query_id, m, buf_id);
        post_write_send(m, buf_id, offset, TASK);
        tman.task_queue[m].clear();
        tman.node_task_queue[m].clear();
      }
    }
  }

  // For task push from node task.
  template <typename dist_t>
  void post_node_remote_task(TaskManager<dist_t> &tman) {
    auto *buf = send_write_buf.getLocal();

    // post write task
    for (uint32_t m = 0; m < MACHINE_NUM; m++) {
      if (m == tman.local_id) continue;
      // Send remote task & node.
      if (tman.task_queue[m].size() > 0 || tman.node_task_queue[m].size() > 0) {
        uint32_t buf_id = get_write_send_buf(m);
        uint64_t offset = 0;
        tman.serialize_task(m, buf->buffer[m][buf_id], offset);
        // printf("q%u post task to m%d buf%u\n", query_id, m, buf_id);
        post_write_send(m, buf_id, offset, TASK);
        tman.task_queue[m].clear();
        tman.node_task_queue[m].clear();
      }
    }
  }

  // TODO: change to multi-merge post.
  // TODO: desperate.
  template <typename dist_t>
  void post_result(ResultMsg<dist_t> &res_msg, uint32_t machine_id) {
    auto *buf = send_write_buf.getLocal();
    uint32_t buf_id = get_write_send_buf(machine_id);
    uint64_t offset = 0;
    // printf("post result to m%u\n", machine_id);
    res_msg.serialize(buf->buffer[machine_id][buf_id], offset);
    post_write_send(machine_id, buf_id, offset, RESULT);
  }

  template <typename dist_t>
  void post_compute(TaskManager<dist_t> &tman) {
    auto *buf = send_write_buf.getLocal();
    for (uint32_t m = 0; m < MACHINE_NUM; m++) {
      if (m == tman.local_id) continue;
      if (tman.compute_queue[m].size() > 0) {
        // printf("post compute to %u\n", m);
        // The compute queue has been format to limit (<= max_sge_num = 16).
        for (ComputeMsg &cmp_msg : tman.compute_queue[m]) {
          uint64_t offset = 0;
          uint32_t buf_id = get_write_send_buf(m);
          tman.serialize_compute(cmp_msg, buf->buffer[m][buf_id], offset);
          // printf("post to m%d\n", m);
          post_sg_write_send(
              m, buf_id, cmp_msg.vec_ptr, cmp_msg.sg_size, COMPUTE);
        }
        // printf("post compute over\n");
        tman.compute_queue[m].clear();
      }
    }
  }

  template <typename dist_t>
  void post_remote_result(TaskManager<dist_t> &tman, uint32_t machine_id) {
    auto *buf = send_write_buf.getLocal();
    uint32_t buf_id = get_write_send_buf(machine_id);
    uint64_t offset = 0;
    tman.serialize_result(machine_id, buf->buffer[machine_id][buf_id], offset);
    // printf("post res to m%d buf%u\n", machine_id, buf_id);
    post_write_send(machine_id, buf_id, offset, RESULT);
    tman.result_queue[machine_id].clear();
  }

  template <typename dist_t>
  void post_remote_result(TaskManager<dist_t> &tman) {
    auto *buf = send_write_buf.getLocal();

    for (uint32_t m = 0; m < MACHINE_NUM; m++) {
      if (m == tman.local_id) continue;
      if (tman.result_queue[m].size() > 0) {
        uint32_t buf_id = get_write_send_buf(m);
        uint64_t offset = 0;
        tman.serialize_result(m, buf->buffer[m][buf_id], offset);
        // printf("post res to m%d buf%u\n", m, buf_id);
        post_write_send(m, buf_id, offset, RESULT);
        tman.result_queue[m].clear();
      }
    }
  }

  /**
   * Send node task result to origin machine. This machine must be non-core
   * machine. This func is called when the node result queue is full.
   */
  template <typename dist_t>
  void post_node_result(TaskManager<dist_t> &tman, uint32_t m) {
    // Send NodeResult to original machine (NodeResult queue).
    auto *buf = send_write_buf.getLocal();
    if (tman.node_result_queue[m].size() > 0) {
      // std::cout << "]] post node res to origin:" << m << "\n";
      uint32_t buf_id = get_write_send_buf(m);
      uint64_t offset = 0;
      tman.serialize_node_result(m, buf->buffer[m][buf_id], offset);
      // printf("post res to m%d buf%u\n", m, buf_id);
      post_write_send(m, buf_id, offset, NODE_RESULT);
      tman.node_result_queue[m].clear();
    }
  }

  /**
   * Send node sync msg to non-core machine. This machine must be non-core
   * machine.
   */
  template <typename dist_t>
  void post_node_sync(TaskManager<dist_t> &tman, uint32_t m) {
    // Send NodeResult to original machine (NodeResult queue).
    auto *buf = send_write_buf.getLocal();
    if (tman.noncore_sync_msg[m].size() > 0) {
      // std::cout << "]] post node sync to core:" << m << "\n";
      uint32_t buf_id = get_write_send_buf(m);
      uint64_t offset = 0;
      tman.serialize_node_sync(m, buf->buffer[m][buf_id], offset);
      // printf("post res to m%d buf%u\n", m, buf_id);
      post_write_send(m, buf_id, offset, NODE_SYNC);
      tman.noncore_sync_msg[m].clear();
    }
  }

  /**
   * Send node task result, task, sync msg to origin machine, coremachine,
   * non-core machine, respectively. This machine must be non-core machine.
   */
  template <typename dist_t>
  void post_remote_res_task_sync(TaskManager<dist_t> &tman) {
/*
   Send NodeResult to original machine (NodeResult queue).
   Send Sync to core machine (sync queue).
   Send Task to non-core machine (task queue).
 */
// 1. Send NodeResult to original machine (NodeResult queue).
#ifdef DEBUG
    // std::cout << "non core machine send node result sync\n";
#endif

    auto *buf = send_write_buf.getLocal();
    for (uint32_t m = 0; m < MACHINE_NUM; m++) {
      if (m == tman.local_id) continue;
      if (tman.node_result_queue[m].size() > 0) {
        // std::cout << "]] post node res to origin:" << m << "\n";
        uint32_t buf_id = get_write_send_buf(m);
        uint64_t offset = 0;
        tman.serialize_node_result(m, buf->buffer[m][buf_id], offset);
        // printf("post res to m%d buf%u\n", m, buf_id);
        post_write_send(m, buf_id, offset, NODE_RESULT);
        tman.node_result_queue[m].clear();
      }
    }

    // 2. Send Sync to core machine (sync queue).
    for (uint32_t m = 0; m < MACHINE_NUM; m++) {
      if (m == tman.local_id) continue;
      if (tman.noncore_sync_msg[m].size() > 0) {
        // std::cout << "]] post node sync to core:" << m << "\n";
        uint32_t buf_id = get_write_send_buf(m);
        uint64_t offset = 0;
        tman.serialize_node_sync(m, buf->buffer[m][buf_id], offset);
        // printf("post res to m%d buf%u\n", m, buf_id);
        post_write_send(m, buf_id, offset, NODE_SYNC);
        tman.noncore_sync_msg[m].clear();
      }
    }

    // 3. Send Task to non-core machine (task queue).
    post_node_remote_task(tman);

    return;
  }

  /**
   * Used for termination.
   */
  void send_term(
      int my_machine_id, size_t q_correct, size_t q_total, 
      size_t q_query_load, double cumu_lat_us = 0.0, size_t computation_cnt = 0);
  bool check_term();
  void reset_term();
  void reset_sendrecv();

  /* This part is for index building. */
  void post_partition_info(IndexParameter &index_param, uint32_t machine_id);
  void poll_partition_info(IndexParameter &index_param);
  void build_sync();

  void post_b2_start(uint32_t machine_id, uint32_t qid, char *query, size_t query_size, size_t ef);
  void poll_b2_start(char *query_data, size_t &ef, uint32_t &qid);


  template<typename dist_t>
  void poll_b2_start(std::vector<uint32_t> &b2_query_queue, TaskManager<dist_t> &tman) {
    auto *rbuf = recv_write_buf.getLocal();  // recv buffer
    poll_send();
    poll_recv();

    // Collect recved partition info.
    while (rbuf->b2_start_queue.size() > 0) {
      char *b2_start_ptr = rbuf->b2_start_queue.front();
      rbuf->b2_start_queue.pop_front();
      size_t ofs = 0;
      size_t query_size = 0;
      memcpy(&query_size, b2_start_ptr + ofs, sizeof(size_t));
      ofs += sizeof(size_t);
      // memcpy(query_data, b2_start_ptr + ofs, query_size);
      ofs += query_size;
      size_t tmp_ef;
      memcpy(&tmp_ef, b2_start_ptr + ofs, sizeof(size_t));
      ofs += sizeof(size_t);
      uint32_t qid;
      memcpy(&qid, b2_start_ptr + ofs, sizeof(uint32_t));
      tman.global_query[qid]->ef = tmp_ef;
      ofs += sizeof(uint32_t);
      free(b2_start_ptr);
      b2_query_queue.push_back(qid);
    }
  }


  template<typename dist_t>
  void post_b2_end(
      uint32_t machine_id, uint32_t qid, std::vector<std::pair<dist_t, size_t>> &result_buf, 
      size_t compute_cnt = 0) {
    auto *buf = send_write_buf.getLocal();
    uint32_t buf_id = get_write_send_buf(machine_id);
    size_t len = 0;
    memcpy(buf->buffer[machine_id][buf_id] + len, &qid, sizeof(uint32_t));
    len += sizeof(uint32_t);
    size_t buf_sz = result_buf.size();
    memcpy(buf->buffer[machine_id][buf_id] + len, &buf_sz, sizeof(size_t));
    len += sizeof(size_t);
    for (auto &res : result_buf) {
      memcpy(buf->buffer[machine_id][buf_id] + len, &res, sizeof(res));
      len += sizeof(res);
    }
    memcpy(buf->buffer[machine_id][buf_id] + len, &compute_cnt, sizeof(size_t));
    len += sizeof(size_t);
    post_write_send(machine_id, buf_id, len, B2_END);
  }

  template<typename dist_t>
  void poll_b2_end(size_t k, std::vector<uint32_t> &b2_finished_queue, 
    std::map<uint32_t, QueryMsg<dist_t>*> &ongoing_queries, TaskManager<dist_t> &tman) {
    auto *rbuf = recv_write_buf.getLocal();  // recv buffer
    poll_send();
    poll_recv();

    // Collect recved partition info.
    while (rbuf->b2_end_queue.size() > 0) {
      char *b2_end_ptr = rbuf->b2_end_queue.front();
      rbuf->b2_end_queue.pop_front();
      size_t ofs = 0;
      uint32_t qid;
      memcpy(&qid, b2_end_ptr + ofs, sizeof(uint32_t));
      ofs += sizeof(uint32_t);
      size_t buf_sz = 0;
      memcpy(&buf_sz, b2_end_ptr + ofs, sizeof(size_t));
      ofs += sizeof(size_t);
      auto &result = tman.global_query[qid]->result;
      for (size_t i = 0; i < buf_sz; i++) {
        std::pair<dist_t, size_t> res;
        memcpy(&res, b2_end_ptr + ofs, sizeof(std::pair<dist_t, size_t>));
        ofs += sizeof(std::pair<dist_t, size_t>);
        result.push(res);
      }

      size_t recv_comp_cnt = 0;
      memcpy(&recv_comp_cnt, b2_end_ptr + ofs, sizeof(size_t));
      ofs += sizeof(size_t);
      
      while (result.size() > k) {
        result.pop();
      }
      free(b2_end_ptr);
      tman.global_query[qid]->b2_recv_num ++;
      // printf("]]]q%u b2 recv num=%u\n", qid, tman.global_query[qid]->b2_recv_num);
      if(tman.global_query[qid]->b2_recv_num == tman.global_query[qid]->b2_dispatched_num){
        b2_finished_queue.push_back(qid);
        ongoing_queries.erase(qid);
      }
    }
  }


  template<typename dist_t>
  void poll_b2_end(
      std::priority_queue<std::pair<dist_t, size_t>> &result, size_t k, uint32_t &qid, size_t &compute_cnt, 
      size_t recv_cnt = 0, size_t end_cnt = MACHINE_NUM) {
    auto *rbuf = recv_write_buf.getLocal();  // recv buffer
    for (; recv_cnt < end_cnt;) {
      poll_send();
      poll_recv();

      // Collect recved partition info.
      if (rbuf->b2_end_queue.size() > 0) {
        char *b2_end_ptr = rbuf->b2_end_queue.front();
        rbuf->b2_end_queue.pop_front();
        size_t ofs = 0;
        memcpy(&qid, b2_end_ptr + ofs, sizeof(uint32_t));
        ofs += sizeof(uint32_t);
        size_t buf_sz = 0;
        memcpy(&buf_sz, b2_end_ptr + ofs, sizeof(size_t));
        ofs += sizeof(size_t);
        for (size_t i = 0; i < buf_sz; i++) {
          std::pair<dist_t, size_t> res;
          memcpy(&res, b2_end_ptr + ofs, sizeof(std::pair<dist_t, size_t>));
          ofs += sizeof(std::pair<dist_t, size_t>);

          result.push(res);
        }

        size_t recv_comp_cnt = 0;
        memcpy(&recv_comp_cnt, b2_end_ptr + ofs, sizeof(size_t));
        ofs += sizeof(size_t);
        compute_cnt += recv_comp_cnt;
        
        while (result.size() > k) {
          result.pop();
        }
        free(b2_end_ptr);
        recv_cnt++;
      }
    }
    // printf("all recved.\n");
  }


  // For ScalaGraph.
  void swap_scala_index_meta_info(
      uint32_t machine_id, std::vector<size_t> &out_remote_ngh_num,
      std::vector<std::vector<size_t>> &in_remote_ngh_num);

  void swap_scalagraph_remote_ngh(
      std::vector<char *> &out_remote_ngh_ptr,
      std::vector<size_t> &out_remote_ngh_num, char *recv_ptr,
      size_t *in_remote_ngh_start_ofs,
      std::vector<std::vector<size_t>> &in_remote_ngh_num);
};
