#pragma once

#include <assert.h>
#include <stdlib.h>

#include <atomic>
#include <fstream>
#include <list>
#include <map>
#include <memory>
#include <random>
#include <unordered_set>

#include "anns_param.h"
#include "coromem/include/backend.h"
#include "coromem/include/galois/loops.h"
#include "coromem/include/runtime/range.h"
#include "hash_table.h"
#include "index/graph_index.h"
#include "query_msg.h"
#include "rdma/rdma_comm.h"
#include "scala_scheduler.h"
#include "task_manager.h"
#include "visited_list_pool.h"
#include "worklists/balance_queue.h"
#include "worklists/chunk.h"

namespace hnswlib {
typedef unsigned int tableint;
typedef unsigned int linklistsizeint;

static inline uint32_t get_runtime_query_group_size() {
  static uint32_t value = []() {
    const char *env = getenv("COTRA_QUERY_GROUP_SIZE");
    if (env != nullptr && env[0] != '\0') {
      char *end = nullptr;
      unsigned long parsed = strtoul(env, &end, 10);
      if (end != env && *end == '\0' && parsed > 0) {
        return static_cast<uint32_t>(parsed);
      }
    }
    return static_cast<uint32_t>(QUERY_GROUP_SIZE);
  }();
  return value;
}

// TODO: change name to lowercase name method.
template <typename dist_t>
class ScalaSearch : public AlgorithmInterface<dist_t> {
 public:
  size_t max_elements_{0};
  size_t cur_element_count{0};
  size_t size_data_per_element_{0};
  size_t size_links_per_element_{0};
  size_t M_{0};
  size_t maxM_{0};
  size_t maxM0_{0};
  size_t ef_construction_{0};
  size_t ef_{0};

  double mult_{0.0};  // useless
  int maxlevel_{0};

  std::unique_ptr<VisitedListPool> visited_list_pool_{nullptr};

  tableint enterpoint_node_{0};

  size_t size_links_level0_{0};
  size_t offsetData_{0}, offsetLevel0_{0}, label_offset_{0};

  char *data_level0_memory_{nullptr};

  char *partition_data{nullptr};

  char **linkLists_{nullptr};
  std::vector<int> element_levels_;  // keeps level of each element

  size_t data_size_{0}; // vector data size
  size_t data_label_size{0}; //  vector data + label size

  std::string evaluation_save_path;  // evaluation save path
  std::ofstream evaluation_save_file;

  DISTFUNC<dist_t> fstdistfunc_;
  void *dist_func_param_{nullptr};

  std::default_random_engine level_generator_;
  std::default_random_engine update_probability_generator_;

  mutable std::atomic<long> metric_distance_computations{0};
  mutable std::atomic<long> metric_hops{0};

  AnnsParameter anns_param;
  GraphIndex &graph_index;

  // InsertBag<ComputeTask<dist_t>> global_task_queue;
  // InsertBag<ComputeTask<dist_t>> nxt_global_task_queue;

  BalanceQueue<ComputeTask<dist_t>> ba_task_queue;

  bool should_term{false};
  bool has_send_term{false};
  // Store the global query pointer.
  QueryMsg<dist_t> **global_query;
  PerThreadStorage<Profiler> all_profiler;

  ScalaScheduler scheduler;

  // RDMA
  /* These are RDMA connection related resources */
  // Time profiling.

  bool haveInit = false;  // Init state for post recv.
  std::atomic<cycles_t> basetime{0};

  // RDMA communication info.
  RdmaParameter rdma_param;
  RdmaCommunication rdma_comm;

  typedef QueryMsg<dist_t>::CompareByFirst CompareByFirst;

  // For rdma meta data.
  char **remote_ptr;
  char *partition_ptr;  // = remote_ptr[machine_id]
  size_t partition_size;

  size_t *remote_size;

  // RDMA profiling data
  size_t remote_access_num{0}, local_access_num{0};
  int hop;
  std::map<int, int> hop_cnt;
  std::map<int, int> ratio;
  int per_q_remote_acc{0}, per_q_local_acc{0};

  // global task queue
  PerThreadStorage<TaskManager<dist_t>> task_ma{anns_param.machine_id};

#ifdef PROF_DEGREE
  uint64_t local_distri[MAX_GRAPH_DEG];
  uint64_t remote_distri[MAX_GRAPH_DEG];

  uint64_t local_cnt{0}, remote_cnt{0};
  uint64_t local_deg{0}, remote_deg{0};

#endif

#ifdef PROF_COMPUTATION
  // NOTE: no parallel, should be counted in single thread.
  // profiling shard index computation efficiency. 
  size_t computation_cnt = 0;
#endif

#ifdef PROF_ALL_Q_DISTRI
  // NOTE: no parallel, should be counted in single thread.
  std::vector<size_t> all_q_distr_cnt = std::vector<size_t>(MACHINE_NUM, 0);
  std::vector<float> all_q_dist_cnt = std::vector<float>(MACHINE_NUM, 0.0);
#endif

  // set evaluation path
  void set_evaluation_save_path(std::string path) {
    evaluation_save_path = path;
    if (path != "") {
      evaluation_save_file.open(evaluation_save_path, std::ios::out);
      if (evaluation_save_file.is_open()) {
        printf("Open evaluation save file %s\n", evaluation_save_path.c_str());
      } else {
        printf(
            "Open evaluation save file %s failed\n",
            evaluation_save_path.c_str());
      }
    }
  }

  std::ofstream &get_evaluation_save_file() { return evaluation_save_file; }

  // for all app type
  ScalaSearch(
      AnnsParameter anns_param_, RdmaParameter rdma_param_,
      SpaceInterface<dist_t> *space_interface,
      GraphIndex &graph_index_)
      : anns_param(anns_param_),
        rdma_param(rdma_param_),
        graph_index(graph_index_) {
    data_size_ = space_interface->get_data_size();
    data_label_size = data_size_ + sizeof(size_t);
    fstdistfunc_ = space_interface->get_dist_func();
    dist_func_param_ = space_interface->get_dist_func_param();
    if(data_size_ != graph_index.index_param.vec_size){
      std::cerr<<"Error: Mismatch vector data size. \n";
      abort();
    }
    
    global_query =
        (QueryMsg<dist_t> **)new QueryMsg<dist_t> *[anns_param.qsize];
    if (anns_param.app_type == SINGLE_MACHINE) {
      printf("Single machine baseline ...\n");
      graph_index.b1_graph->loadIndex();

      offsetLevel0_ = graph_index.b1_graph->offsetLevel0_;
      max_elements_ = graph_index.b1_graph->max_elements_;
      cur_element_count = graph_index.b1_graph->cur_element_count;
      size_data_per_element_ = graph_index.b1_graph->size_data_per_element_;
      label_offset_ = graph_index.b1_graph->label_offset_;
      offsetData_ = graph_index.b1_graph->offsetData_;
      maxlevel_ = graph_index.b1_graph->maxlevel_;
      enterpoint_node_ = graph_index.b1_graph->enterpoint_node_;

      maxM_ = graph_index.b1_graph->maxM_;
      maxM0_ = graph_index.b1_graph->maxM0_;
      M_ = graph_index.b1_graph->M_;
      mult_ = graph_index.b1_graph->mult_;
      ef_construction_ = graph_index.b1_graph->ef_construction_;
      data_level0_memory_ = graph_index.b1_graph->data_level0_memory_;

      size_links_per_element_ = graph_index.b1_graph->size_links_per_element_;

      size_links_level0_ = graph_index.b1_graph->size_links_level0_;

      // Set visit list number equal to the thread number.
      int max_thd = omp_get_max_threads();
      printf("Set visit list number to %d\n", max_thd);
      visited_list_pool_.reset(new VisitedListPool(max_thd, max_elements_));

      linkLists_ = (char **)malloc(sizeof(void *) * max_elements_);
      if (linkLists_ == nullptr)
        throw std::runtime_error(
            "Not enough memory: loadIndex failed to allocate linklists");
      element_levels_ = std::vector<int>(max_elements_);
      ef_ = 10;
      for (size_t i = 0; i < cur_element_count; i++) {
        element_levels_[i] = graph_index.b1_graph->element_levels_[i];
        linkLists_[i] = graph_index.b1_graph->linkLists_[i];
      }
    } else if (anns_param.app_type == B2 || anns_param.app_type == B2Kmeans || anns_param.app_type == B2KmeansBatch) {
      printf("Baseline2 leader/member init ...\n");
      // graph_index.b1_graph->print();
      graph_index.b1_graph->loadIndex();

      offsetLevel0_ = graph_index.b1_graph->offsetLevel0_;
      max_elements_ = graph_index.b1_graph->max_elements_;
      cur_element_count = graph_index.b1_graph->cur_element_count;
      size_data_per_element_ = graph_index.b1_graph->size_data_per_element_;
      label_offset_ = graph_index.b1_graph->label_offset_;
      offsetData_ = graph_index.b1_graph->offsetData_;
      maxlevel_ = graph_index.b1_graph->maxlevel_;
      enterpoint_node_ = graph_index.b1_graph->enterpoint_node_;

      maxM_ = graph_index.b1_graph->maxM_;
      maxM0_ = graph_index.b1_graph->maxM0_;
      M_ = graph_index.b1_graph->M_;
      mult_ = graph_index.b1_graph->mult_;
      ef_construction_ = graph_index.b1_graph->ef_construction_;
      data_level0_memory_ = graph_index.b1_graph->data_level0_memory_;

      size_links_per_element_ = graph_index.b1_graph->size_links_per_element_;

      size_links_level0_ = graph_index.b1_graph->size_links_level0_;

      // Set visit list number equal to the thread number.
      int max_thd = omp_get_max_threads();
      printf("Set visit list number to %d\n", max_thd);
      visited_list_pool_.reset(new VisitedListPool(max_thd, max_elements_));

      linkLists_ = (char **)malloc(sizeof(void *) * max_elements_);
      if (linkLists_ == nullptr)
        throw std::runtime_error(
            "Not enough memory: loadIndex failed to allocate linklists");
      element_levels_ = std::vector<int>(max_elements_);
      ef_ = 10;
      for (size_t i = 0; i < cur_element_count; i++) {
        element_levels_[i] = graph_index.b1_graph->element_levels_[i];
        linkLists_[i] = graph_index.b1_graph->linkLists_[i];
      }
      graph_index.b1_graph->print();
      graph_index.b1_graph->init_ptr(
          rdma_param.machine_id, MACHINE_NUM, max_elements_,
          data_level0_memory_, size_data_per_element_);
      run_rdma();
    } else if (
        anns_param.app_type == B1 || anns_param.app_type == B1_MIGRATE ||
        anns_param.app_type == B1_ASYNC) {
      printf("Baseline1 leader/member init ...\n");
      // graph_index.b1_graph->print();
      graph_index.b1_graph->loadIndex();

      offsetLevel0_ = graph_index.b1_graph->offsetLevel0_;
      max_elements_ = graph_index.b1_graph->max_elements_;
      cur_element_count = graph_index.b1_graph->cur_element_count;
      size_data_per_element_ = graph_index.b1_graph->size_data_per_element_;
      label_offset_ = graph_index.b1_graph->label_offset_;
      offsetData_ = graph_index.b1_graph->offsetData_;
      maxlevel_ = graph_index.b1_graph->maxlevel_;
      enterpoint_node_ = graph_index.b1_graph->enterpoint_node_;

      maxM_ = graph_index.b1_graph->maxM_;
      maxM0_ = graph_index.b1_graph->maxM0_;
      M_ = graph_index.b1_graph->M_;
      mult_ = graph_index.b1_graph->mult_;
      ef_construction_ = graph_index.b1_graph->ef_construction_;
      data_level0_memory_ = graph_index.b1_graph->data_level0_memory_;

      size_links_per_element_ = graph_index.b1_graph->size_links_per_element_;

      size_links_level0_ = graph_index.b1_graph->size_links_level0_;

      // Set visit list number equal to the thread number.
      int max_thd = omp_get_max_threads();
      printf("Set visit list number to %d\n", max_thd);
      visited_list_pool_.reset(new VisitedListPool(max_thd, max_elements_));

      linkLists_ = (char **)malloc(sizeof(void *) * max_elements_);
      if (linkLists_ == nullptr)
        throw std::runtime_error(
            "Not enough memory: loadIndex failed to allocate linklists");
      element_levels_ = std::vector<int>(max_elements_);
      ef_ = 10;
      for (size_t i = 0; i < cur_element_count; i++) {
        element_levels_[i] = graph_index.b1_graph->element_levels_[i];
        linkLists_[i] = graph_index.b1_graph->linkLists_[i];
      }
      graph_index.b1_graph->print();
      graph_index.b1_graph->init_ptr(
          rdma_param.machine_id, MACHINE_NUM, max_elements_,
          data_level0_memory_, size_data_per_element_);
      run_rdma();
    } else if (
        anns_param.app_type == ScalaANN_v2 ||
        anns_param.app_type == ScalaANN_ASYNC ||
        anns_param.app_type == ScalaANN_v3) {
      printf("ScalaGraph init ...\n");
      graph_index.scala_graph->load_index();
      run_rdma();
    } else {
      printf("Unrecogenized App Type.\n");
      abort();
    }
    printf("Init over ...\n");
  }

  // For reorder test only [TEMP]
  ScalaSearch(std::ifstream &index) {
    printf("reorder test init ...\n");
    verify_reorder_index(index, 0);
  }

  ~ScalaSearch() { clear(); }

  void clear() {
    free(data_level0_memory_);
    data_level0_memory_ = nullptr;
    for (tableint i = 0; i < cur_element_count; i++) {
      if (element_levels_[i] > 0) free(linkLists_[i]);
    }
    free(linkLists_);
    linkLists_ = nullptr;
    cur_element_count = 0;
    // TODO: rmv visitlist
    visited_list_pool_.reset(nullptr);
  }

  uint32_t *checked_scala_v3_ngh_ptr(
      uint64_t offset, uint32_t deg, uint32_t qid, uint32_t result_mid,
      const char *stage, bool require_alignment = false) {
    auto *scala_graph = graph_index.scala_graph;
    const uint64_t deg_ngh_size = scala_graph->deg_ngh_size;
    const uint64_t span = static_cast<uint64_t>(deg) * sizeof(uint32_t);
    const bool bad_align =
        require_alignment && ((offset % sizeof(uint32_t)) != 0);
    const bool bad_offset = offset >= deg_ngh_size;
    const bool bad_span =
        !bad_offset && (span > (deg_ngh_size - offset));

    if (bad_align || bad_offset || bad_span) {
      printf(
          "Error: invalid ScalaGraph V3 span at %s on machine %u for q%u "
          "return_mid %u offset %llu deg %u span %llu deg_ngh_size %llu "
          "partition_ofs %llu local_vec_num %u remote_ngh_num %llu\n",
          stage, anns_param.machine_id, qid, result_mid,
          (unsigned long long)offset, deg, (unsigned long long)span,
          (unsigned long long)deg_ngh_size,
          (unsigned long long)scala_graph->partition_ofs,
          scala_graph->local_vec_num,
          (unsigned long long)scala_graph->remote_ngh_num);
      abort();
    }

    return scala_graph->scala_graph_v3_get_ngh_ptr(offset);
  }

  void read_test() {
    printf("Read operation test\n");
    srand(time(NULL));

    RangeIterator all(100);
    on_each(all);

    int summm = 0;

    printf("Verify rdma read by 100 random group reads ...\n");
    do_all(iterate(all), [&](const uint32 &i) {
      std::vector<BufferCache> remote_vec;

      // Random vector number.
      int vector_num = rand() % MAX_READ_NUM;
      for (int v = 0; v < vector_num; v++) {
        // Random vector id.
        int vector_id = rand() % max_elements_;
        uint32_t machine_id =
            graph_index.b1_graph->get_vector_machine(vector_id);
        if (machine_id != rdma_param.machine_id) {
          size_t internal_id = graph_index.b1_graph->get_internal_id(vector_id);
          remote_vec.push_back(
              BufferCache{machine_id, vector_id, internal_id, -1, NULL});
        }
      }
      summm += remote_vec.size();
      // printf("vec size: %d summm %d\n", remote_vec.size(), summm);

      // Step 2: send async read to remote vec.
      rdma_comm.post_read(remote_vec, size_data_per_element_);

      rdma_comm.poll_read();

      // printf("buff list << ");
      for (BufferCache &r : remote_vec) {
        uint32_t vector_id = r.vector_id;
        char *vector_ptr = r.buffer_ptr;
        size_t machine_id = r.machine_id;
        size_t internal_id = r.internal_id;

        int *read_data = (int *)(vector_ptr);
        int *truedata =
            (int *)(data_level0_memory_ + vector_id * size_data_per_element_);

        if (memcmp(read_data, truedata, size_data_per_element_)) {
          printf("Read operator verify failed.\n");
          printf(
              "internal_id %llu offset %llu size %llu\n", vector_id,
              internal_id * size_data_per_element_, size_data_per_element_);
          printf("read_data\n");
          for (int i = 0; i < 10; i++) {
            printf("%d ", read_data[i]);
          }
          printf("\n truedata\n");
          for (int i = 0; i < 10; i++) {
            printf("%d ", truedata[i]);
          }
          printf("\n");
          abort();
        }
        rdma_comm.release_cache(r.buffer_id);
        // printf(" %d", r.buffer_id);
      }
      // printf("\n");
    });
    printf("Verify SUCCESS ...\n");
  }

  void send_test() {
    printf("Send operation test\n");

    printf("Verify rdma send by send message...\n");
    // Each thread send there thread_id*100 + machine_id to the remote machine.
    // Each member machine verify this msg.
    on_each([&](uint64 tid, uint64 total) {
      auto *ctx = rdma_comm.rdma_ctx.getLocal();
      if (rdma_param.machine == MEMBER) {
        printf("Test member recv.\n");
        // recv from leader
        int *recv_vec = (int *)ctx->recv_ptr[0];
        rdma_comm.post_recv(0, 0, size_data_per_element_);
        rdma_comm.sync_recv();
        if (recv_vec[0] != tid * 100) {
          printf("Send verify failed\n");
          printf(
              "expect recv_vec = %d, but recved: %d\n", tid * 100, recv_vec[0]);
          abort();
        }

        printf("Test member send.\n");
        // send to leader
        int send_msg = tid * 100 + rdma_param.machine_id;
        int *send_vec = (int *)(ctx->send_ptr[0]);
        send_vec[0] = send_msg;
        rdma_comm.sync_send(0, 0, size_data_per_element_);

      } else {
        int send_msg = tid * 100 + rdma_param.machine_id;
        for (int m = 1; m < MACHINE_NUM; m++) {
          rdma_comm.post_recv(m, 0, size_data_per_element_);
        }

        printf("Test leader send.\n");
        for (int m = 1; m < MACHINE_NUM; m++) {
          int *send_vec = (int *)(ctx->send_ptr[m]);
          send_vec[0] = send_msg;
          rdma_comm.sync_send(m, 0, size_data_per_element_);
        }

        printf("Test leader recv.\n");
        rdma_comm.sync_recv(MACHINE_NUM - 1);
        for (int m = 1; m < MACHINE_NUM; m++) {
          int *recv_vec = (int *)ctx->recv_ptr[m];
          if (recv_vec[0] != tid * 100 + m) {
            printf("Send verify failed\n");
            printf(
                "expect %d machine recv_vec = %d, but recved: %d\n", m,
                tid * 100 + m, recv_vec[0]);
            abort();
          }
        }
      }
    });

    printf("Verify SUCCESS ...\n");
  }

  /**
   * Collect communication meta info, bandwidth, comm number.
   */
  void report_comm_info(float time) {
    uint64_t all_read_size = 0;
    uint64_t all_write_size = 0;
    uint64_t all_read_cnt = 0;
    uint64_t all_write_cnt = 0;

    for (uint32_t t = 0; t < getActiveThreads(); ++t) {
      auto *ctx = rdma_comm.rdma_ctx.getRemote(t);
      all_read_size += ctx->read_size;
      all_write_size += ctx->write_size;
      all_read_cnt += ctx->read_cnt;
      all_write_cnt += ctx->write_cnt;
    }
    float read_sz_ps = 1000000.0 * (all_read_size >> 20) / time;
    float read_num_ps = 1000000.0 * all_read_cnt / time;
    float write_sz_ps = 1000000.0 * (all_write_size >> 20) / time;
    float write_num_ps = 1000000.0 * all_write_cnt / time;
    printf("-------------- Overall COMM Info --------------\n");
    printf("Read: %.2f MB/s, %.2f num/s\n", read_sz_ps, read_num_ps);
    printf("Write: %.2f MB/s, %.2f num/s\n", write_sz_ps, write_num_ps);
    printf("--------------------  End  --------------------\n");
  }


  /**
   * For random shared nothing baseline, send to all machines.
   */
  void dispatch_query(void *query_data, uint32_t qid, size_t k) {
    if (rdma_param.machine == LEADER) {
      for (int m = 1; m < MACHINE_NUM; m++) {
        rdma_comm.post_b2_start(m, qid, (char *)query_data, data_size_, ef_);
      }
    } else {
      uint32_t qid;
      rdma_comm.poll_b2_start((char *)query_data, ef_, qid);
    }
  }


  void dispatch_kmeans_query_start(uint32_t qid, size_t k) {
    TaskManager<dist_t> &tman = *task_ma.getLocal();
    auto &ongoing_queries = tman.ongoing_queries;
    auto &b2_query_queue = tman.b2_query_queue;
    ongoing_queries[qid] = global_query[qid];
    char* query_data = global_query[qid]->vector;
     // Count
    uint32_t su_cnt = 0;
    uint32_t mx_cnt = 0, mx_mid;

    // Select sub machines.
    OptHashPosVector vis_hash;
    std::priority_queue<
        std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
        CompareByFirst>
        top_candidates;
    std::priority_queue<
        std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
        CompareByFirst>
        candidate_set;
    dist_t lowerBound;
    size_t ef = std::max(ef_, k);
    uint32_t filter_num[MACHINE_NUM];
    memset(filter_num, 0, sizeof(filter_num));

    auto proc_candidate = [&](dist_t dist, uint32_t candidate_id) {
      bool flag_consider_candidate = top_candidates.size() < ef || lowerBound > dist;

      if (flag_consider_candidate) {
        candidate_set.emplace(-dist, candidate_id);
        top_candidates.emplace(dist, candidate_id);

        while (top_candidates.size() > ef) {
          top_candidates.pop();
        }
        if (!top_candidates.empty())
          lowerBound = top_candidates.top().first;
      }
    };

    std::pair<dist_t, uint32_t> group_nodes[CAND_GROUP_SIZE];
    uint32_t enterpoint = 0;
    // assume enter point is cached.
    char *ep_data = graph_index.top_index->get_vector(enterpoint);

    // Traverse on base index. (Use one-side migrate)
    dist_t dist = fstdistfunc_(query_data, ep_data, dist_func_param_);
    lowerBound = dist;
    top_candidates.emplace(dist, enterpoint);
    candidate_set.emplace(-dist, enterpoint);

    uint32_t gt_id = graph_index.top_index->get_global_id(enterpoint);
    vis_hash.CheckAndSet(gt_id);
    
    ef = std::max((uint32_t)ef_, anns_param.res_knn);

    // Pre-stage.
    while (!candidate_set.empty()) {
      std::pair<dist_t, uint32_t> cand = candidate_set.top();
      dist_t candidate_dist = -cand.first;
      if (candidate_dist > lowerBound) {
        break;
      }
      candidate_set.pop();

      _mm_prefetch(query_data, _MM_HINT_T0);
      uint32_t *ngh_arr = graph_index.top_index->get_ngh_arr(cand.second);
      _mm_prefetch((char *)ngh_arr, _MM_HINT_T0);

      _mm_prefetch((char *)(ngh_arr + 1), _MM_HINT_T0);
      for (uint32_t d = 0; d < ngh_arr[0]; d++) {
        _mm_prefetch(
            graph_index.top_index->get_vector(ngh_arr[d + 1]), _MM_HINT_T0);

        // TODO: can prefetch here.
        uint32_t gt_id = graph_index.top_index->get_global_id(ngh_arr[d]);
        if (!vis_hash.CheckAndSet(gt_id)) {
          char *vec_data = graph_index.top_index->get_vector(ngh_arr[d]);
          dist_t dist = fstdistfunc_(
              query_data, vec_data, dist_func_param_);
          proc_candidate(dist, ngh_arr[d]);
        }
      }
    }
    /**
      * Previous stage over, start to dispatch sub-query.
      * First filterred the candidate set.
      */
    while (!top_candidates.empty()) {
      auto cand_top = top_candidates.top();
      uint32_t gt_id = graph_index.top_index->get_global_id(cand_top.second);
      int mid = graph_index.b1_graph->get_machine_id(gt_id);
      filter_num[mid]++;
      top_candidates.pop();
    }

   
#ifdef DEBUG
    printf("q%u filter num: \n", qid);
#endif
    for (uint32_t m = 0; m < MACHINE_NUM; m++) {
#ifdef DEBUG
      printf("%u ", filter_num[m]);
#endif
      su_cnt += filter_num[m];
      if (filter_num[m] > mx_cnt) {
        mx_cnt = filter_num[m];
        mx_mid = m;
      }
    }

    // Identify which partition should be selected.
    global_query[qid]->b2_dispatched_num = 0;
    global_query[qid]->b2_recv_num = 0;
    for (uint32_t m = 0; m < MACHINE_NUM; m++) {
      float ratio_m = (float)filter_num[m] / su_cnt;
      // if (ratio_m > 1.0 / (MACHINE_NUM)) {  // core machine ratio bar.
      // if (filter_num[m] > 0) {  // core machine ratio bar.
      if (true) {  // core machine ratio bar.
        if(m == anns_param.machine_id) {
#ifdef DEBUG
          printf("local is primary machine.\n");
#endif
          b2_query_queue.push_back(qid);
        }
        else {
          rdma_comm.post_b2_start(m, qid, (char *)query_data, data_size_, ef_);
        }
        global_query[qid]->b2_dispatched_num++;
      }
    }
    global_query[qid]->b2_original_mid = anns_param.machine_id;
    return ;
  }

  void collect_kmeans_query_start(void *query_data, uint32_t &qid) {
    rdma_comm.poll_b2_start((char *)query_data, ef_, qid);
    return ;
  }

  void collect_kmeans_query_start(std::vector<uint32_t> &b2_query_queue) {
    TaskManager<dist_t> &tman = *task_ma.getLocal();
    rdma_comm.poll_b2_start(b2_query_queue, tman);
    return ;
  }

  void collect_query(
      std::priority_queue<std::pair<dist_t, labeltype>> &result, size_t k,
      std::priority_queue<std::pair<dist_t, labeltype>> *sub_result = nullptr) {
    if (rdma_param.machine == LEADER) {
      // Leader machine recv the result from member.
      size_t compute_cnt = 0;
      uint32_t qid;
      size_t recv_cnt = 1, end_cnt = MACHINE_NUM;
      rdma_comm.poll_b2_end(result, k, qid, compute_cnt, recv_cnt, end_cnt);
#ifdef PROF_COMPUTATION
      computation_cnt += compute_cnt;
#endif

    } else {
      // Member machine send the result to leader.
      std::vector<std::pair<dist_t, size_t>> result_buf;
      int res_cnt = 0;
      while (result.size() > 0) {
        result_buf.push_back(result.top());
        result.pop();
      }
      assert(res_cnt == k);

      size_t compute_cnt = 0;
#ifdef PROF_COMPUTATION
      compute_cnt = computation_cnt;
      computation_cnt = 0;
#endif
      uint32_t qid;
      rdma_comm.post_b2_end(0, qid, result_buf, compute_cnt);

      // memcpy(ctx->send_ptr[0], result_buf, result_size);
      // int send_machine_id = 0;
      // rdma_comm.sync_send(send_machine_id, 0, size_data_per_element_);
    }
  }

  
  void dispatch_kmeans_query_end(
    std::priority_queue<std::pair<dist_t, labeltype>> &result, size_t k, uint32_t qid) {

    // Member machine send the result to leader.
    std::vector<std::pair<dist_t, size_t>> result_buf;
    int res_cnt = 0;
    while (result.size() > 0) {
      result_buf.push_back(result.top());
      result.pop();
    }
    assert(res_cnt == k);
    size_t compute_cnt = 0;
    rdma_comm.post_b2_end(0, qid, result_buf, compute_cnt);
    // printf("post q %u end\n", qid);
  }

  // for kmeans batch 
  void dispatch_kmeans_query_end(uint32_t qid, size_t k) {
    auto &result = global_query[qid]->result;
    // Member machine send the result to leader.
    std::vector<std::pair<dist_t, size_t>> result_buf;
    int res_cnt = 0;
    while (result.size() > 0) {
      result_buf.push_back(result.top());
      result.pop();
    }
    assert(res_cnt == k);
    size_t compute_cnt = 0;
    rdma_comm.post_b2_end(0, qid, result_buf, compute_cnt);
  }

  void collect_kmeans_query_end(
    std::priority_queue<std::pair<dist_t, labeltype>> &result, size_t k, size_t collect_num = MACHINE_NUM, 
    bool local_is_primary = false) {
    // int result_size = sizeof(std::pair<int, labeltype>) * k;
    // auto *ctx = rdma_comm.rdma_ctx.getLocal();
    size_t recv_cnt = (size_t)local_is_primary;
    size_t compute_cnt; // useless
    uint32_t qid;
    rdma_comm.poll_b2_end(result, k, qid, compute_cnt, recv_cnt, collect_num);
    // printf("collect qid %u collect num %u\n", qid, collect_num);
  }

  // kmeans batch
  void collect_kmeans_query_end(size_t k, std::vector<uint32_t> &b2_finished_queue, 
      std::map<uint32_t, QueryMsg<dist_t>*> &ongoing_queries) {
    TaskManager<dist_t> &tman = *task_ma.getLocal();
    rdma_comm.poll_b2_end(k, b2_finished_queue, ongoing_queries, tman);
  }

  std::vector<uint32_t> b2_deal_local(size_t k){
    TaskManager<dist_t> &tman = *task_ma.getLocal();
    for(uint32_t q : tman.b2_query_queue){
      // 2. local search from task queue.
      localSearchKnn(q, k);
      // if is local query
      if(tman.ongoing_queries[q]){
        tman.ongoing_queries[q]->b2_recv_num ++;
        // printf("]]] q%u recv num %u\n", q, tman.ongoing_queries[q]->b2_recv_num);
        if(tman.ongoing_queries[q]->b2_recv_num == tman.ongoing_queries[q]->b2_dispatched_num){
          tman.b2_finished_queue.push_back(q);
          tman.ongoing_queries.erase(q);
        }
      }
      else { // else remote
        // dispatch_kmeans_query_end(q, k);
        std::cerr<<"q" <<q <<" main machine b2_deal_local should not be here.\n";
        abort();
      } 
    }
    tman.b2_query_queue.clear();

    // 3. poll collect query & proc finished.
    for(;;){
      collect_kmeans_query_end(k, tman.b2_finished_queue, tman.ongoing_queries);
      if(tman.ongoing_queries.size() < B2_KMEANS_BATCH) break;
    }

    auto ret = tman.b2_finished_queue;
    tman.b2_finished_queue.clear();
    return ret;
  }

  std::vector<uint32_t> b2_deal_local_standby(size_t k){
    TaskManager<dist_t> &tman = *task_ma.getLocal();
    // 3. poll collect query & proc finished.
    collect_kmeans_query_end(k, tman.b2_finished_queue, tman.ongoing_queries);
    auto ret = tman.b2_finished_queue;
    tman.b2_finished_queue.clear();
    return ret;
  }
  
  void b2_member_deal_local_standby(size_t k){
    TaskManager<dist_t> &tman = *task_ma.getLocal();
    for(;;){
      collect_kmeans_query_start(tman.b2_query_queue); // need to clear new query.
      for(uint32_t q : tman.b2_query_queue){
        // printf("]]] recv q%u\n", q);
        localSearchKnn(q, k);
        dispatch_kmeans_query_end(q, k);
      }
      tman.b2_query_queue.clear();
    }
  }

  void run_rdma() {
    // For leader, we only need to register memory for read buffer now.
    partition_ptr = graph_index.get_partition_ptr();
    partition_size = graph_index.get_partition_size();
    rdma_comm.com_init(partition_ptr, partition_size, rdma_param);
    rdma_comm.com_read_init();
    rdma_comm.com_write_init();
  }

  void end_rdma() {
    if (rdma_comm.ctx_close_connection()) {
      fprintf(stderr, "Failed to close connection between server and client\n");
      return;
    }
  }

  int check_remote_local(char *remote, char *local, int len) {
    return memcmp((void *)remote, (void *)local, len);
  }

  void set_ef(size_t ef) { ef_ = ef; }

  inline labeltype getExternalLabel(tableint internal_id) const {
    labeltype return_label;
    memcpy(
        &return_label,
        (data_level0_memory_ + internal_id * size_data_per_element_ +
         label_offset_),
        sizeof(labeltype));
    return return_label;
  }

  inline labeltype getRemoteLabel(tableint vector_id) {
    labeltype return_label;
// machine_id equal to the machine id the vector locate.
    size_t machine_id = graph_index.b1_graph->get_vector_machine(vector_id);
    if (machine_id == rdma_param.machine_id) {
      // local memory
      memcpy(
          &return_label,
          (data_level0_memory_ + vector_id * size_data_per_element_ +
           label_offset_),
          sizeof(labeltype));
      return return_label;
    } else {
// remote memory
      size_t internal_id = graph_index.b1_graph->get_internal_id(vector_id);
      char *vector_ptr = rdma_comm.fetch_vector(
          internal_id, machine_id, size_data_per_element_);
      // int thd_id = omp_get_thread_num();
#ifdef VERIFY_BY_LOCAL
      if (check_remote_local(
              vector_ptr + label_offset_,
              data_level0_memory_ + vector_id * size_data_per_element_ +
                  label_offset_,
              sizeof(labeltype))) {
        printf("Remote label verify failed ..., vector_id: %llu \n", vector_id);
        printf(
            "%llu %llu\n", internal_id, internal_id * size_data_per_element_);
        exit(0);
      }
#endif
      memcpy(&return_label, (vector_ptr + label_offset_), sizeof(labeltype));

      return return_label;
    }
  }

  inline char *getDataByInternalId(tableint internal_id) const {
    return (
        data_level0_memory_ + internal_id * size_data_per_element_ +
        offsetData_);
  }

  bool isRemote(tableint vector_id) {
    return graph_index.b1_graph->get_vector_machine(vector_id) !=
           rdma_param.machine_id;
  }

  // TODO: uniform tableint type
  // TODO: reorder hnswalg func
  inline char *get_local_element(uint64_t vector_id) {
    return (data_level0_memory_ + vector_id * size_data_per_element_);
  }

  inline char *get_element_data(char *vector_ptr) {
    return (vector_ptr + offsetData_);
  }

  inline linklistsizeint *get_element_ngh(char *vector_ptr) {
    return (linklistsizeint *)(vector_ptr + offsetLevel0_);
  }

  inline labeltype get_element_label(char *vector_ptr) {
    labeltype return_label;
    memcpy(&return_label, (vector_ptr + label_offset_), sizeof(labeltype));
    return return_label;
  }

  inline char *getLocalData(tableint vector_id) {
    return (
        data_level0_memory_ + vector_id * size_data_per_element_ + offsetData_);
  }

  inline char *getRemoteData(tableint vector_id) {
    // remote memory
    size_t machine_id = graph_index.b1_graph->get_vector_machine(vector_id);
    size_t internal_id = graph_index.b1_graph->get_internal_id(vector_id);
    char *vector_ptr =
        rdma_comm.fetch_vector(internal_id, machine_id, size_data_per_element_);
#ifdef VERIFY_BY_LOCAL
    // verify the data
    if (check_remote_local(
            vector_ptr + offsetData_,
            data_level0_memory_ + vector_id * size_data_per_element_ +
                offsetData_,
            data_size_)) {
      printf("Remote data verify failed ..., vector_id: %llu \n", vector_id);
      printf("%llu %llu\n", internal_id, internal_id * size_data_per_element_);
      exit(0);
    }
#endif
    return vector_ptr + offsetData_;
  }

  inline char *getData(tableint vector_id) {
    if (isRemote(vector_id)) {
      return getRemoteData(vector_id);
    } else
      return getLocalData(vector_id);
  }

  void reportAccessNum(size_t qsize, float time_us_per_query) {
    size_t sum = local_access_num + remote_access_num;

    float avg_rmt_acc_num = (float)remote_access_num / qsize;
    printf(
        "base level local: %llu remote: %llu\n", local_access_num,
        remote_access_num);
    printf(
        "avg access num: %.2f\t avg rmt access num: %.2f\t "
        "avg rmt access lat: %.2f\n",
        (float)sum / qsize, avg_rmt_acc_num,
        time_us_per_query / avg_rmt_acc_num);
    printf(
        "local: %.2f%% remote: %.2f%%\n", (float)local_access_num * 100.0 / sum,
        (float)remote_access_num * 100.0 / sum);
    local_access_num = remote_access_num = 0;
    // report hop
    /*
      hop_cnt: <key, value> key: hop length, value: count number.
    */
    int sum_hop[] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    int cnt_range[] = {0, 50, 100, 150, 200, 300, 400, 500, 600, 1e9};
    int range_len = 9;
    for (auto &h : hop_cnt) {
      for (int i = 0; i < range_len; i++) {
        if (h.first < cnt_range[i + 1]) {
          sum_hop[i] += h.second;
          break;
        }
      }
    }
    int sum_sumhop = 0;
    for (int i = 0; i < range_len; i++) {
      sum_sumhop += sum_hop[i];
    }
    printf("sum_sumhop: %d expect equal to 10000\n", sum_sumhop);
    // report avg hop num
    uint64_t sum_hop_length = 0;
    for (auto &h : hop_cnt) {
      sum_hop_length += h.first * h.second;
    }
    printf("avg_hop: %.2f\n", (float)sum_hop_length / sum_sumhop);

    // reset gop_cnt
    for (auto &h : hop_cnt) {
      h.second = 0;
    }

    // report ratio
    int ratio_sum = 0;
    for (int i = 0; i <= 10; i++) {
      ratio_sum += ratio[i];
    }
    printf("ratio_sum: %d expect equal to 10000\n", ratio_sum);
    for (int i = 0; i <= 10; i++) {
      printf(
          "ratio range %d - %d: cnt: %d, ratio: %.2f\n", i * 10, (i + 1) * 10,
          ratio[i], (float)ratio[i] * 100.0 / ratio_sum);
    }
    // reset ratio
    for (auto &r : ratio) {
      r.second = 0;
    }
  }

  // bare_bone_search means there is no check for deletions and stop condition
  // is ignored in return of extra performance
  template <bool bare_bone_search = true, bool collect_metrics = false>
  std::priority_queue<
      std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
      CompareByFirst>
  searchBaseLayerST(
      tableint ep_id, const void *data_point, size_t ef, int query_id = -1,
      BaseFilterFunctor *isIdAllowed = nullptr,
      BaseSearchStopCondition<dist_t> *stop_condition = nullptr) {
    // VisitedList *vl = visited_list_pool_->getFreeVisitedList();
    // vl_type *visited_array = vl->mass;
    // vl_type visited_array_tag = vl->curV;
    OptHashPosVector vis_hash;

#ifdef PROFILER
    Profiler &m_profiler = *all_profiler.getLocal();
    m_profiler.start("traversal");
#endif

    std::priority_queue<
        std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
        CompareByFirst>
        top_candidates;
    std::priority_queue<
        std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
        CompareByFirst>
        candidate_set;

#ifdef PROF_LAZY
    std::vector<uint32_t> main_part{0,0,0,0,0,0,0,0}; 
    std::vector<size_t> local_q_distri = 
      std::vector<size_t>(MACHINE_NUM, 0);
    for(uint32_t attemp = 0; attemp < 2;attemp++) {
      // computation_cnt = 0;
      while(!top_candidates.empty()){
        top_candidates.pop(); 
      }
      while(!candidate_set.empty()){
        candidate_set.pop(); 
      }
      vis_hash.clear();
#endif

    dist_t lowerBound;
    char *ep_data = getDataByInternalId(ep_id);
    dist_t dist = fstdistfunc_(data_point, ep_data, dist_func_param_);
    lowerBound = dist;
    top_candidates.emplace(dist, ep_id);
    candidate_set.emplace(-dist, ep_id);
    // visited_array[ep_id] = visited_array_tag;

#ifdef DEBUG
    std::cout << "ep_id: " << ep_id << " dist: " << dist << std::endl;
#endif

    vis_hash.CheckAndSet(ep_id);

#ifdef PROF_Q_DISTRI
    std::vector<std::pair<dist_t, uint32_t>> q_distri;
#endif 

#ifdef PROF_ALL_Q_DISTRI
    std::vector<size_t> local_q_distri = 
      std::vector<size_t>(MACHINE_NUM, 0);
    std::vector<float> local_q_dist = std::vector<float>(MACHINE_NUM, 0.0);
#endif


#ifdef PROF_LAZY
    bool has_no_empty = true; // lazy for stimulate async.
    int async_step =16;
    std::vector<std::pair<dist_t, tableint>> async_q[async_step];
    while (!candidate_set.empty() || has_no_empty) {
      if(!candidate_set.empty()) {
#else 
    while (!candidate_set.empty()) {
#endif
      std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
      dist_t candidate_dist = -current_node_pair.first;

      bool flag_stop_search;
      flag_stop_search = candidate_dist > lowerBound;
      if (flag_stop_search) {
        break;
      }
      candidate_set.pop();

      tableint current_node_id = current_node_pair.second;

      int *data = (int *)get_linklist0(current_node_id);
      size_t size = getListCount((linklistsizeint *)data);

#ifdef USE_SSE
      // _mm_prefetch((char *)(visited_array + *(data + 1)), _MM_HINT_T0);
      // _mm_prefetch((char *)(visited_array + *(data + 1) + 64), _MM_HINT_T0);
      _mm_prefetch(
          data_level0_memory_ + (*(data + 1)) * size_data_per_element_ +
              offsetData_,
          _MM_HINT_T0);
      _mm_prefetch((char *)(data + 2), _MM_HINT_T0);
#endif

      for (size_t j = 1; j <= size; j++) {
        int candidate_id = *(data + j);
#ifdef USE_SSE
        // _mm_prefetch((char *)(visited_array + *(data + j + 1)), _MM_HINT_T0);
        _mm_prefetch(
            data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_
            +
                offsetData_,
            _MM_HINT_T0);  ////////////
#endif
        // if (!(visited_array[candidate_id] == visited_array_tag)) {
        if (!vis_hash.CheckAndSet(candidate_id)) {
#ifdef PROF_COMPUTATION
          computation_cnt ++;
#ifdef PROF_LAZY
          if(attemp == 0){ // only count second attempt.
            computation_cnt --;
          }
          uint32_t part_id = graph_index.b1_graph->get_vector_machine(candidate_id);
          local_q_distri[part_id] ++;
#endif
#endif
          // visited_array[candidate_id] = visited_array_tag;
          char *currObj1 = (getDataByInternalId(candidate_id));
#ifdef PROFILER
          // m_profiler.count("q_compute_num");
#endif
          dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);
#ifdef PROF_Q_DISTRI
          uint32_t part_id = graph_index.b1_graph->get_vector_machine(candidate_id);
          q_distri.emplace_back(dist, part_id);
#endif
#ifdef PROF_ALL_Q_DISTRI
          uint32_t part_id = graph_index.b1_graph->get_vector_machine(candidate_id);
          local_q_distri[part_id] ++;
          local_q_dist[part_id] += (float)dist;
#endif
          bool flag_consider_candidate;
          flag_consider_candidate =
              top_candidates.size() < ef || lowerBound > dist;

          if (flag_consider_candidate) {
#ifdef PROF_LAZY
            uint32_t part_id = graph_index.b1_graph->get_vector_machine(candidate_id);
            if(main_part[part_id] == 1){
              async_q[async_step - 1].emplace_back(dist, candidate_id);
            }
            else {
#endif 
            candidate_set.emplace(-dist, candidate_id);
#ifdef USE_SSE
            _mm_prefetch(
                data_level0_memory_ +
                    candidate_set.top().second * size_data_per_element_ +
                    offsetLevel0_,  ///////////
                _MM_HINT_T0);       ////////////////////////
#endif

            top_candidates.emplace(dist, candidate_id);
            bool flag_remove_extra = false;
            flag_remove_extra = top_candidates.size() > ef;
            while (flag_remove_extra) {
              tableint id = top_candidates.top().second;
              top_candidates.pop();
              flag_remove_extra = top_candidates.size() > ef;
            }

            if (!top_candidates.empty())
              lowerBound = top_candidates.top().first;
#ifdef PROF_LAZY
            }
#endif
          }
        }
      }

#ifdef PROF_LAZY
      }
      // pop async_q
      has_no_empty = false;
      for(auto &v : async_q[0]){
        dist_t dist = v.first;
        auto candidate_id = v.second;
        candidate_set.emplace(-dist, candidate_id);
        top_candidates.emplace(dist, candidate_id);
        bool flag_remove_extra = false;
        flag_remove_extra = top_candidates.size() > ef;
        while (flag_remove_extra) {
          tableint id = top_candidates.top().second;
          top_candidates.pop();
          flag_remove_extra = top_candidates.size() > ef;
        }
        if (!top_candidates.empty())
          lowerBound = top_candidates.top().first;
      }
      async_q[0].clear();
      for(uint32_t q = 0; q < async_step - 1; q++){
        std::swap(async_q[q], async_q[q + 1]);
        has_no_empty |= (async_q[q].size()>0);
      }
#endif

    }

#ifdef PROF_LAZY
      // reorganize main_part
      std::vector<std::pair<int, size_t>> indexed_vec;
      indexed_vec.reserve(local_q_distri.size());
      for (size_t i = 0; i < local_q_distri.size(); ++i) {
        indexed_vec.emplace_back(local_q_distri[i], i);
      }
      std::sort(indexed_vec.begin(), indexed_vec.end(),
          [](const auto& a, const auto& b) { return a.first > b.first; });
      uint32_t mainpart_bar = 3;
      // only top async
      // for(uint32_t i = 0; i<mainpart_bar; i++){
      //   main_part[indexed_vec[i].second] = 1;
      // }
      // only top sync
      for(uint32_t i = 0 ;i<MACHINE_NUM;i++){
        if(i < mainpart_bar) continue;
        main_part[indexed_vec[i].second] = 1;
      }
    }
#endif

#ifdef PROF_Q_DISTRI
    std::string filename = "distri_q_" + std::to_string(query_id) + ".log";
    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        abort();
    }
    // printf("Part Distri: \n");
    for(auto qd: q_distri){
      outFile << qd.first << " " << qd.second << "\n";
      // printf("(%u, %u),", qd.first, qd.second);
    }
    // printf("\n");
    outFile.close();
    std::cout << "Successfully wrote " << q_distri.size() 
      << " tuples to " << filename << std::endl;
#endif

#ifdef PROF_ALL_Q_DISTRI
    std::vector<std::pair<int, size_t>> indexed_vec;
    indexed_vec.reserve(local_q_distri.size());
    for (size_t i = 0; i < local_q_distri.size(); ++i) {
        indexed_vec.emplace_back(local_q_distri[i], i);
    }
    std::sort(indexed_vec.begin(), indexed_vec.end(),
        [](const auto& a, const auto& b) { return a.first < b.first; });
    for(uint32_t m = 0;m<MACHINE_NUM; m++){
      all_q_distr_cnt[m] += indexed_vec[m].first;
      all_q_dist_cnt[m] += local_q_dist[indexed_vec[m].second];
    }
#endif

#ifdef PROFILER
    m_profiler.end("traversal");
#endif
    // visited_list_pool_->releaseVisitedList(vl);
    return top_candidates;
  }

  // bare_bone_search means there is no check for deletions and stop condition
  // is ignored in return of extra performance
  template <bool bare_bone_search = true, bool collect_metrics = false>
  std::priority_queue<
      std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
      CompareByFirst>
  searchRemoteBaseLayerST(
      tableint ep_id, const void *data_point, size_t ef,
      BaseFilterFunctor *isIdAllowed = nullptr,
      BaseSearchStopCondition<dist_t> *stop_condition = nullptr) {
    QueryMsg<dist_t> query;

    char *ep_data = getData(ep_id);
    dist_t dist = fstdistfunc_(data_point, ep_data, dist_func_param_);
    query.lowerBound = dist;
    query.top_candidates.emplace(dist, ep_id);
    query.candidate_set.emplace(-dist, ep_id);

    query.visit_hash.CheckAndSet(ep_id);

    char *cur_node_buf = (char *)malloc(size_data_per_element_);

    // size_t max_cand_size = 0;
    while (!query.candidate_set.empty()) {
      std::pair<dist_t, tableint> current_node_pair = query.candidate_set.top();
      dist_t candidate_dist = -current_node_pair.first;

      if (candidate_dist > query.lowerBound) {
        break;
      }
      query.candidate_set.pop();
#ifdef PROFILING
      hop++;
#endif

      tableint current_node_id = current_node_pair.second;

#ifdef DEBUG
      printf("%d,", candidate_dist);
#endif

      int *data = (int *)get_Remotelinklist0(current_node_id);
      size_t size = getListCount((linklistsizeint *)data);

      memcpy(cur_node_buf, data, size_data_per_element_);
      data = (int *)cur_node_buf;

      for (size_t j = 1; j <= size; j++) {
        int candidate_id = *(data + j);
        if (!query.visit_hash.CheckAndSet(candidate_id)) {
          char *currObj1 = (getData(candidate_id));
          dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);

          bool flag_consider_candidate =
              query.top_candidates.size() < ef || query.lowerBound > dist;

          if (flag_consider_candidate) {
            query.candidate_set.emplace(-dist, candidate_id);
            query.top_candidates.emplace(dist, candidate_id);
            while (query.top_candidates.size() > ef) {
              tableint id = query.top_candidates.top().second;
              query.top_candidates.pop();
            }

            if (!query.top_candidates.empty())
              query.lowerBound = query.top_candidates.top().first;
          }
        }
      }
      // max_cand_size = std::max(max_cand_size, query.candidate_set.size());
    }
    // printf("max_cand_size %d\n", max_cand_size);
#ifdef PROFILING
    if (!hop_cnt[hop]) {
      hop_cnt[hop] = 1;
    } else {
      hop_cnt[hop] += 1;
    }
#endif

    free(cur_node_buf);
    return query.top_candidates;
  }

  std::priority_queue<std::pair<dist_t, labeltype>> remoteAsyncSearchKnn(
      const void *query_data, size_t k,
      BaseFilterFunctor *isIdAllowed = nullptr) {
    std::priority_queue<std::pair<dist_t, labeltype>> result;
    if (cur_element_count == 0) return result;

    tableint currObj = enterpoint_node_;

    dist_t curdist =
        fstdistfunc_(query_data, getData(enterpoint_node_), dist_func_param_);

    for (int level = maxlevel_; level > 0; level--) {
      bool changed = true;
      while (changed) {
        changed = false;
        unsigned int *data;

        data = (unsigned int *)get_linklist(currObj, level);
        int size = getListCount(data);
        // metric_hops++;
        // metric_distance_computations += size;

        tableint *datal = (tableint *)(data + 1);
        for (int i = 0; i < size; i++) {
          tableint cand = datal[i];
          if (cand < 0 || cand > max_elements_)
            throw std::runtime_error("cand error");
          dist_t d = fstdistfunc_(query_data, getData(cand), dist_func_param_);

          if (d < curdist) {
            curdist = d;
            currObj = cand;
            changed = true;
          }
        }
      }
    }

#ifdef PROFILING
    hop = maxlevel_;
#endif

    bool bare_bone_search = true;
    result = searchAsyncRemoteBaseLayerST<true>(
        currObj, query_data, std::max(ef_, k), k, isIdAllowed);

#ifdef PROFILING
    // compute ratio and reset per_q_remote_acc
    int per_q_sum = per_q_remote_acc + per_q_local_acc;
    int r = per_q_remote_acc * 10 / per_q_sum;
    if (!ratio[r]) {
      ratio[r] = 1;
    } else {
      ratio[r] += 1;
    }
    per_q_remote_acc = per_q_local_acc = 0;
#endif

    return result;
  }

  // Search remote use async read optimization.
  template <bool bare_bone_search = true, bool collect_metrics = false>
  std::priority_queue<std::pair<dist_t, size_t>> searchAsyncRemoteBaseLayerST(
      tableint ep_id, const void *data_point, size_t ef, int k,
      BaseFilterFunctor *isIdAllowed = nullptr,
      BaseSearchStopCondition<dist_t> *stop_condition = nullptr) {
#ifdef DEBUG
    auto *buf = rdma_comm.read_buf.getLocal();
    printf("start b1 async, buflist size: %d\n", buf->buff_id_list.size());
#endif
    QueryMsg2<dist_t> query;

    char *ep_ptr = get_local_element(ep_id);
    char *ep_data = get_element_data(ep_ptr);
    dist_t dist = fstdistfunc_(data_point, ep_data, dist_func_param_);
    query.lowerBound = dist;
    query.top_candidates.emplace(VectorCache<dist_t>(dist, ep_id, -1, ep_ptr));
    query.candidate_set.emplace(VectorCache<dist_t>(-dist, ep_id, -1, ep_ptr));

    query.visit_hash.CheckAndSet(ep_id);

    char *cur_node_buf = (char *)malloc(size_data_per_element_);

    VectorCache<dist_t> group_nodes[CAND_GROUP_SIZE];
    while (!query.candidate_set.empty()) {
      int gsz = 0;
      for (gsz = 0; gsz < CAND_GROUP_SIZE && !query.candidate_set.empty();
           gsz++) {
        group_nodes[gsz] = query.candidate_set.top();
        dist_t candidate_dist = -group_nodes[gsz].dist;
        if (candidate_dist > query.lowerBound) {
          break;
        }
        query.candidate_set.pop();
        // do not release here, since may in top candidates queue.
      }
      if (gsz == 0) {
        // should stop.
        break;
      }

      // Step 1: split remote and local node.
      std::vector<BufferCache> remote_vec;
      std::vector<uint64_t> local_vec;
      for (int g = 0; g < gsz; g++) {
        tableint current_node_id = group_nodes[g].vector_id;
        int *data = (int *)get_element_ngh(group_nodes[g].ptr);
        size_t size = getListCount((linklistsizeint *)data);
        memcpy(cur_node_buf, data, size_data_per_element_);

        data = (int *)cur_node_buf;
        for (size_t j = 1; j <= size; j++) {
          int candidate_id = *(data + j);
          if (!query.visit_hash.CheckAndSet(candidate_id)) {
            uint32_t machine_id =
                graph_index.b1_graph->get_vector_machine(candidate_id);
            if (machine_id != rdma_param.machine_id) {
              size_t internal_id =
                  graph_index.b1_graph->get_internal_id(candidate_id);
              remote_vec.push_back(
                  BufferCache{machine_id, candidate_id, internal_id, -1, NULL});
              // BufferCache *ptr = new BufferCache{
              //     machine_id, candidate_id, internal_id, -1, NULL};
              // remote_vec.push_back(ptr);
            } else {
              local_vec.push_back(candidate_id);
            }
          }
        }
      }

      // Step 2: send async read to remote vec.
      rdma_comm.post_read(remote_vec, size_data_per_element_);

      // Step 3: compute local vec.
      for (auto &l : local_vec) {
        uint32_t candidate_id = l;
        char *ele_ptr = get_local_element(l);
        char *vec_ptr = get_element_data(ele_ptr);

        dist_t dist = fstdistfunc_(data_point, vec_ptr, dist_func_param_);

        bool flag_consider_candidate =
            query.top_candidates.size() < ef || query.lowerBound > dist;

        if (flag_consider_candidate) {
          query.candidate_set.emplace(
              VectorCache<dist_t>(-dist, candidate_id, -1, ele_ptr));
          query.top_candidates.emplace(
              VectorCache<dist_t>(dist, candidate_id, -1, ele_ptr));
          while (query.top_candidates.size() > ef) {
            int buff_id = query.top_candidates.top().buffer_id;
            query.top_candidates.pop();
            /*
              Only release in 2 case:
              Case1 : pop from top queue.
              Case2 : not select to cand queue.
              This is Case 1.
            */
            if (buff_id != -1) {
              // if not local vector.
              rdma_comm.release_cache(buff_id);
            }
          }

          if (!query.top_candidates.empty())
            query.lowerBound = query.top_candidates.top().dist;
        }
      }

      // Step 4: compute remote vec.
      rdma_comm.poll_read();

      for (BufferCache &r : remote_vec) {
        uint32_t candidate_id = r.vector_id;
        char *ele_ptr = r.buffer_ptr;
        char *vec_ptr = get_element_data(ele_ptr);

        dist_t dist = fstdistfunc_(data_point, vec_ptr, dist_func_param_);

        bool flag_consider_candidate =
            query.top_candidates.size() < ef || query.lowerBound > dist;

        if (flag_consider_candidate) {
          query.candidate_set.emplace(
              VectorCache<dist_t>(-dist, candidate_id, r.buffer_id, ele_ptr));
          query.top_candidates.emplace(
              VectorCache<dist_t>(dist, candidate_id, r.buffer_id, ele_ptr));
          while (query.top_candidates.size() > ef) {
            int buff_id = query.top_candidates.top().buffer_id;
            query.top_candidates.pop();
            if (buff_id != -1) {
              // if not local vector.
              rdma_comm.release_cache(buff_id);
            }
          }

          if (!query.top_candidates.empty())
            query.lowerBound = query.top_candidates.top().dist;
        } else {
          // Case2 : release cache buffer.
          rdma_comm.release_cache(r.buffer_id);
        }
      }
    }

    while (query.top_candidates.size() > k) {
      VectorCache<dist_t> rez = query.top_candidates.top();
      query.top_candidates.pop();
      if (rez.buffer_id != -1) {
        rdma_comm.release_cache(rez.buffer_id);
      }
    }

    while (query.top_candidates.size() > 0) {
      VectorCache<dist_t> rez = query.top_candidates.top();
      query.result.push(
          std::pair<dist_t, labeltype>(rez.dist, get_element_label(rez.ptr)));
      query.top_candidates.pop();
      if (rez.buffer_id != -1) {
        rdma_comm.release_cache(rez.buffer_id);
      }
    }

#ifdef DEBUG
    printf("query over, buflist size: %d\n", buf->buff_id_list.size());
#endif

    free(cur_node_buf);
    return query.result;
  }


  void printQueryBatchInfo(float time) {
    float recall = rdma_comm.total ? 1.0f * rdma_comm.correct / rdma_comm.total : 0.0f;
    float time_us_per_query = rdma_comm.query_load ? time / rdma_comm.query_load : 0.0f;
    float avg_lat = rdma_comm.query_load ? rdma_comm.query_lat_sum / rdma_comm.query_load : 0.0f;
    size_t all_comp_cnt = rdma_comm.all_computation_cnt;
    float qps = time > 0 ? 1000000.0 * rdma_comm.query_load / time : 0.0f;
    // printf("-------------- Overall info --------------\n");
    // printf(
    //     "correct: %llu total: %llu query load: %llu\n", rdma_comm.correct,
    //     rdma_comm.total, rdma_comm.query_load);
#ifdef PROF_COMPUTATION
    printf(
        "%d \t %.5f \t %.2f \t %.3f \t %llu \n", ef_, recall, avg_lat,
        qps, all_comp_cnt);
#else   
    printf(
        "%d \t %.5f \t %.2f \t %.3f \t \n", ef_, recall, avg_lat,
        qps);
#endif
    
    // printf("------------------  End ------------------\n");
  }

  void printQueryBatchInfoSave(float time, std::ofstream &out) {
    float recall = rdma_comm.total ? 1.0f * rdma_comm.correct / rdma_comm.total : 0.0f;
    float time_us_per_query = rdma_comm.query_load ? time / rdma_comm.query_load : 0.0f;
    float avg_lat = rdma_comm.query_load ? rdma_comm.query_lat_sum / rdma_comm.query_load : 0.0f;
    float qps = time > 0 ? 1000000.0 * rdma_comm.query_load / time : 0.0f;
    printf("-------------- Overall info --------------\n");
    printf(
        "correct: %llu total: %llu query load: %llu\n", rdma_comm.correct,
        rdma_comm.total, rdma_comm.query_load);
    printf(
        "%d \t %.5f \t %.2f \t %.3f \t \n", ef_, recall, avg_lat,
        qps);
    printf("------------------  End ------------------\n");

    // save ef_, QPS, mean latency, recall to file
    out << ef_ << "," << recall << "," << avg_lat << "," << qps
        << std::endl;
  }

  linklistsizeint *get_linklist0(tableint internal_id) const {
    return (linklistsizeint *)(data_level0_memory_ +
                               internal_id * size_data_per_element_ +
                               offsetLevel0_);
  }

  linklistsizeint *get_Remotelinklist0(tableint vector_id) {
    size_t machine_id = graph_index.b1_graph->get_vector_machine(vector_id);
    if (machine_id == rdma_param.machine_id) {
      // local memory
      return (linklistsizeint *)(data_level0_memory_ +
                                 vector_id * size_data_per_element_ +
                                 offsetLevel0_);
    } else {
      // remote memory
      size_t internal_id = graph_index.b1_graph->get_internal_id(vector_id);
      char *vector_ptr = rdma_comm.fetch_vector(
          internal_id, machine_id, size_data_per_element_);
#ifdef VERIFY_BY_LOCAL
      if (check_remote_local(
              vector_ptr + offsetLevel0_,
              data_level0_memory_ + vector_id * size_data_per_element_ +
                  offsetLevel0_,
              size_data_per_element_)) {
        printf("Remote link verify failed ..., vector_id: %llu \n", vector_id);
        printf(
            "%llu %llu\n", internal_id, internal_id * size_data_per_element_);
        exit(0);
      }
#endif
      return (linklistsizeint *)(vector_ptr + offsetLevel0_);
    }
  }

  linklistsizeint *get_linklist0(
      tableint internal_id, char *data_level0_memory_) const {
    return (linklistsizeint *)(data_level0_memory_ +
                               internal_id * size_data_per_element_ +
                               offsetLevel0_);
  }

  linklistsizeint *get_linklist(tableint internal_id, int level) const {
    return (linklistsizeint *)(linkLists_[internal_id] +
                               (level - 1) * size_links_per_element_);
  }

  size_t indexFileSize() const {
    size_t size = 0;
    size += sizeof(offsetLevel0_);
    size += sizeof(max_elements_);
    size += sizeof(cur_element_count);
    size += sizeof(size_data_per_element_);
    size += sizeof(label_offset_);
    size += sizeof(offsetData_);
    size += sizeof(maxlevel_);
    size += sizeof(enterpoint_node_);
    size += sizeof(maxM_);

    size += sizeof(maxM0_);
    size += sizeof(M_);
    size += sizeof(mult_);
    size += sizeof(ef_construction_);

    size += cur_element_count * size_data_per_element_;

    for (size_t i = 0; i < cur_element_count; i++) {
      unsigned int linkListSize =
          element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i]
                                 : 0;
      size += sizeof(linkListSize);
      size += linkListSize;
    }
    return size;
  }

  void verify_reorder_index(std::ifstream &input, size_t max_elements_i = 0) {
    if (!input.is_open()) throw std::runtime_error("Cannot open file");

    clear();
    // get file size:
    input.seekg(0, input.end);
    std::streampos total_filesize = input.tellg();
    input.seekg(0, input.beg);

    readBinaryPOD(input, offsetLevel0_);
    readBinaryPOD(input, max_elements_);
    readBinaryPOD(input, cur_element_count);

    size_t max_elements = max_elements_i;
    if (max_elements < cur_element_count) max_elements = max_elements_;
    max_elements_ = max_elements;
    readBinaryPOD(input, size_data_per_element_);
    readBinaryPOD(input, label_offset_);
    readBinaryPOD(input, offsetData_);
    readBinaryPOD(input, maxlevel_);
    readBinaryPOD(input, enterpoint_node_);

    readBinaryPOD(input, maxM_);
    readBinaryPOD(input, maxM0_);
    readBinaryPOD(input, M_);
    readBinaryPOD(input, mult_);
    readBinaryPOD(input, ef_construction_);
    auto pos = input.tellg();
    input.seekg(pos, input.beg);
  }


  unsigned short int getListCount(linklistsizeint *ptr) const {
    return *((unsigned short int *)ptr);
  }

  void setListCount(linklistsizeint *ptr, unsigned short int size) const {
    *((unsigned short int *)(ptr)) = *((unsigned short int *)&size);
  }

  std::priority_queue<std::pair<dist_t, labeltype>> searchKnn(
      const void *query_data, size_t k,
      BaseFilterFunctor *isIdAllowed = nullptr) {};

  std::priority_queue<std::pair<dist_t, labeltype>> localSearchKnn(
      const void *query_data, size_t k, int query_id = -1, 
      BaseFilterFunctor *isIdAllowed = nullptr) {
    std::priority_queue<std::pair<dist_t, labeltype>> result;
    if (cur_element_count == 0) return result;

#ifdef DEBUG
    printf("get enterpoint data: %d\n", enterpoint_node_);
    printf(
        "size_data_per_element_ %d offsetData_ %d\n", size_data_per_element_,
        offsetData_);
#endif
    tableint currObj = enterpoint_node_;
    dist_t curdist = fstdistfunc_(
        query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

    for (int level = maxlevel_; level > 0; level--) {
      bool changed = true;
      while (changed) {
        changed = false;
        unsigned int *data;

        data = (unsigned int *)get_linklist(currObj, level);
        int size = getListCount(data);
        // metric_hops++;
        // metric_distance_computations += size;

        tableint *datal = (tableint *)(data + 1);
        for (int i = 0; i < size; i++) {
          tableint cand = datal[i];
          if (cand < 0 || cand > max_elements_)
            throw std::runtime_error("cand error");
          dist_t d = fstdistfunc_(
              query_data, getDataByInternalId(cand), dist_func_param_);

          if (d < curdist) {
            curdist = d;
            currObj = cand;
            changed = true;
          }
        }
      }
    }

    std::priority_queue<
        std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
        CompareByFirst>
        top_candidates;
    bool bare_bone_search = !isIdAllowed;
#ifdef DEBUG
    std::cout << "start bare_bone_searching: \n";
#endif
    if (bare_bone_search) {
      // base local
      top_candidates = searchBaseLayerST<true>(
          currObj, query_data, std::max(ef_, k), query_id, isIdAllowed);
      // currObj, query_data, std::max(*my_ef, k), isIdAllowed);
    } else {
      top_candidates = searchBaseLayerST<false>(
          currObj, query_data, std::max(ef_, k), query_id, isIdAllowed);
    }

    while (top_candidates.size() > k) {
      top_candidates.pop();
    }
#ifdef DEBUG
    // std::cout << "ANNS result: \n";
#endif
    while (top_candidates.size() > 0) {
      std::pair<dist_t, tableint> rez = top_candidates.top();
      result.push(std::pair<dist_t, labeltype>(
          rez.first, getExternalLabel(rez.second)));
      top_candidates.pop();
#ifdef DEBUG
      // std::cout << getExternalLabel(rez.second) << " ";
#endif
    }
#ifdef DEBUG
    // std::cout << " \n";
#endif

    return result;
  }


  // for b2 kmeans
  std::priority_queue<std::pair<dist_t, labeltype>> localSearchKnn(uint32_t query_id, size_t k) {
    auto &result = global_query[query_id]->result;
    char* query_data = global_query[query_id]->vector;
    if (cur_element_count == 0) return result;

#ifdef DEBUG
    printf("get enterpoint data: %d\n", enterpoint_node_);
    printf(
        "size_data_per_element_ %d offsetData_ %d\n", size_data_per_element_,
        offsetData_);
#endif
    tableint currObj = enterpoint_node_;
    dist_t curdist = fstdistfunc_(
        query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

    std::priority_queue<
        std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
        CompareByFirst>
        top_candidates;

    top_candidates = searchBaseLayerST<true>(
        currObj, query_data, std::max(global_query[query_id]->ef, k), query_id);

    while (top_candidates.size() > k) {
      top_candidates.pop();
    }

    while (top_candidates.size() > 0) {
      std::pair<dist_t, tableint> rez = top_candidates.top();
      result.push(std::pair<dist_t, labeltype>(
          rez.first, getExternalLabel(rez.second)));
      top_candidates.pop();
    }

    return result;
  }


  std::priority_queue<std::pair<dist_t, labeltype>> remoteSearchKnn(
      const void *query_data, size_t k,
      BaseFilterFunctor *isIdAllowed = nullptr) {
    std::priority_queue<std::pair<dist_t, labeltype>> result;
    if (cur_element_count == 0) return result;

    tableint currObj = enterpoint_node_;

    dist_t curdist =
        fstdistfunc_(query_data, getData(enterpoint_node_), dist_func_param_);

    for (int level = maxlevel_; level > 0; level--) {
      bool changed = true;
      while (changed) {
        changed = false;
        unsigned int *data;

        data = (unsigned int *)get_linklist(currObj, level);
        int size = getListCount(data);
        // metric_hops++;
        // metric_distance_computations += size;

        tableint *datal = (tableint *)(data + 1);
        for (int i = 0; i < size; i++) {
          tableint cand = datal[i];
          if (cand < 0 || cand > max_elements_)
            throw std::runtime_error("cand error");
          dist_t d = fstdistfunc_(query_data, getData(cand), dist_func_param_);

          if (d < curdist) {
            curdist = d;
            currObj = cand;
            changed = true;
          }
        }
      }
    }

#ifdef PROFILING
    hop = maxlevel_;
#endif

    std::priority_queue<
        std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
        CompareByFirst>
        top_candidates;
    bool bare_bone_search = true;
    top_candidates = searchRemoteBaseLayerST<true>(
        currObj, query_data, std::max(ef_, k), isIdAllowed);

    while (top_candidates.size() > k) {
      top_candidates.pop();
    }
    while (top_candidates.size() > 0) {
      std::pair<dist_t, tableint> rez = top_candidates.top();
      result.push(
          std::pair<dist_t, labeltype>(rez.first, getRemoteLabel(rez.second)));
      top_candidates.pop();
    }

#ifdef PROFILING
    // compute ratio and reset per_q_remote_acc
    int per_q_sum = per_q_remote_acc + per_q_local_acc;
    int r = per_q_remote_acc * 10 / per_q_sum;
    if (!ratio[r]) {
      ratio[r] = 1;
    } else {
      ratio[r] += 1;
    }
    per_q_remote_acc = per_q_local_acc = 0;
#endif

    return result;
  }

  void sendTerm(size_t correct, size_t total, size_t query_load, double cumu_lat_us = 0.0) {
#ifdef PROF_COMPUTATION
    rdma_comm.send_term(rdma_param.machine_id, correct, total, 
      query_load, cumu_lat_us, computation_cnt);
    computation_cnt = 0;
#else 
    rdma_comm.send_term(rdma_param.machine_id, correct, total, 
      query_load, cumu_lat_us, 0);
#endif
    return;
  }

  bool check_term() {
    // Check for termination flag.
    rdma_comm.poll_send();
    rdma_comm.poll_recv();
    if (rdma_comm.check_term()) {
      // printf("term detected .\n");
      // Ending, reset send/recv queue.
      rdma_comm.reset_sendrecv();
      return true;
    }

    return false;
  }

  /*
    Follower termination standby.
  */
  void termStandBy(size_t k, BaseFilterFunctor *isIdAllowed = nullptr) {
    std::vector<QueryMsg<dist_t> *> ret;
    QueryMsg<dist_t> *cur_query;
    for (;;) {
      if (check_term()) break;

      std::deque<QueryMsg<dist_t> *> task_queue;
      rdma_comm.collect_recv_query(task_queue);
      while (!task_queue.empty()) {
        cur_query = task_queue.front();
        task_queue.pop_front();
        if (task_queue.empty()) {
          rdma_comm.collect_recv_query(task_queue);
        }
      }
    }
    return;
  }

  void initTerm() { rdma_comm.reset_term(); }


  void scala_search_init(
      const void *query_data, uint32_t vec_size, uint32_t query_num) {
    // printf("scala search init\n");
#ifdef PROF_DEGREE
    memset(local_distri, 0, sizeof(local_distri));
    memset(remote_distri, 0, sizeof(remote_distri));
    local_cnt = local_deg = remote_cnt = remote_deg = 0;
#endif

    for (uint32_t q = 0; q < query_num; q++) {
      // Init query.
      global_query[q] = new QueryMsg<dist_t>(
          (char *)(query_data + (uint64_t)q * vec_size), vec_size, q, ef_,
          rdma_param.machine_id);
    }
    on_each([&](uint64 tid, uint64 total) {
#ifdef PROFILER
      auto *local_prof = all_profiler.getLocal();
      Profiler &m_profiler = *local_prof;
      m_profiler.clear();
#endif
      // init comm size count.
      auto *ctx = rdma_comm.rdma_ctx.getLocal();
#ifdef COMM_PROFILE
      ctx->clear_size_cnt();
#endif
      auto *tman = task_ma.getLocal();
      tman->clear();

      tman->global_query =
          (QueryMsg<dist_t> **)new QueryMsg<dist_t> *[anns_param.qsize];
      for (uint32_t q = 0; q < query_num; q++) {
        // Init query.
        tman->global_query[q] = global_query[q];
      }

      tman->ba_task_queue = &ba_task_queue;
    });
    // printf("scala search init over\n");
  }

  // TODO: can use fusion to reduce unused chunk.
  void do_task(ComputeTask<dist_t> &task) {
#ifdef DEBUG
    printf("do task %u\n", task.qid);
#endif
#ifdef PROFILER
    Profiler &m_profiler = *all_profiler.getLocal();
    m_profiler.start("g_do_task");
#endif
    _mm_prefetch(global_query[task.qid]->vector, _MM_HINT_T0);
    // auto &task = *global_query[query_id]->cur_task;
    char *vec_ptr = graph_index.scala_graph->vector_ptr;
    uint32_t part_ofs =
        graph_index.scala_graph->scala_graph_get_partition_ofs();
    // printf(
    //     "local task size: %u\n",
    //     global_query[query_id]->local_task.size());
    // printf("%llu %u \n", task.offset, task.deg);
    uint32_t *ngh_arr =
        checked_scala_v3_ngh_ptr(
            task.offset, task.deg, task.qid, anns_param.machine_id,
            "do_task");
    _mm_prefetch((char *)ngh_arr, _MM_HINT_T0);
    _mm_prefetch(vec_ptr + ((*(ngh_arr)) - part_ofs) * data_label_size, _MM_HINT_T0);
    _mm_prefetch((char *)(ngh_arr + 1), _MM_HINT_T0);
    worklists::Chunk<std::pair<dist_t, uint32_t>, MAX_GRAPH_DEG> chunk(
        task.is_count);
    for (uint32_t d = 0; d < task.deg; d++) {
      if (d + 1 < task.deg) {
        _mm_prefetch(
            vec_ptr + ((*(ngh_arr + d + 1)) - part_ofs) * data_label_size,
            _MM_HINT_T0);
      }
      // printf("%u\n", *(ngh_arr + d));
      // TODO: may have problem with concurrent
      if (!global_query[task.qid]->visit_hash.CheckAndSet(*(ngh_arr + d))) {
#ifdef PROF_COMPUTATION
        computation_cnt ++;
#endif
        char *vec_data = vec_ptr + ((*(ngh_arr + d)) - part_ofs) * data_label_size;
        dist_t dist = fstdistfunc_(
            global_query[task.qid]->vector, vec_data, dist_func_param_);
        chunk.emplace_back(dist, *(ngh_arr + d));
      }
    }
    global_query[task.qid]->con_task_queue.push(chunk);
#ifdef PROFILER
    m_profiler.end("g_do_task");
#endif
#ifdef DEBUG
    printf("do task %u over\n", task.qid);
#endif
    return;
  }

  void work_steal() {
#ifdef PROFILER
    Profiler &m_profiler = *all_profiler.getLocal();
    m_profiler.start("g_steal");
#endif
    // Use work stealing across threads.
    auto task = ba_task_queue.pop();
    if (task.has_value()) {
      do_task(task.value());
    }
#ifdef PROFILER
    m_profiler.end("g_steal");
#endif
  }

  /**
   * Add cached topindex search.
   */
  Corobj<bool> coro_sub_query_v5(size_t query_id) {
    TaskManager<dist_t> &tman = *task_ma.getLocal();
#ifdef PROFILER
    Profiler &m_profiler = *all_profiler.getLocal();
#endif

    auto top_proc_candidate = [&](dist_t dist, uint32_t candidate_id) {
      bool flag_consider_candidate =
          global_query[query_id]->pre_top_cand.size() <
              global_query[query_id]->ef ||
          global_query[query_id]->lowerBound > dist;

      if (flag_consider_candidate) {
        global_query[query_id]->pre_cand_set.emplace(-dist, candidate_id);
        global_query[query_id]->pre_top_cand.emplace(dist, candidate_id);

        // if (graph_index.top_index->is_last_level(candidate_id)) {
        //   // In the 5-th level hop, put into final canidate set (prestage).
        //   uint32_t gt_id =
        //   graph_index.top_index->get_global_id(candidate_id);
        //   global_query[query_id]->candidate_set.emplace(-dist, gt_id);
        // }

        while (global_query[query_id]->pre_top_cand.size() >
               global_query[query_id]->ef) {
          global_query[query_id]->pre_top_cand.pop();
        }

        if (!global_query[query_id]->pre_top_cand.empty())
          global_query[query_id]->lowerBound =
              global_query[query_id]->pre_top_cand.top().first;
      }
    };

    auto post_proc_candidate = [&](dist_t dist, uint32_t candidate_id) {
      bool flag_consider_candidate =
          global_query[query_id]->top_candidates.size() <
              global_query[query_id]->ef ||
          global_query[query_id]->lowerBound > dist;

      if (flag_consider_candidate) {
        global_query[query_id]->candidate_set.emplace(-dist, candidate_id);
        global_query[query_id]->top_candidates.emplace(dist, candidate_id);

        // update sync msg.
        global_query[query_id]->sync_msg.local_add_cand.emplace_back(
            dist, candidate_id);

        // do process, update local term
        global_query[query_id]->proc_is_black |= 1;

        while (global_query[query_id]->top_candidates.size() >
               global_query[query_id]->ef) {
          tableint id = global_query[query_id]->top_candidates.top().second;
          global_query[query_id]->top_candidates.pop();
        }

        if (!global_query[query_id]->top_candidates.empty())
          global_query[query_id]->lowerBound =
              global_query[query_id]->top_candidates.top().first;
      }
    };

    std::pair<dist_t, uint32_t> group_nodes[CAND_GROUP_SIZE];
    if (global_query[query_id]->is_new) {
#ifdef DEBUG
      printf("start new query %u, Pre STAGE\n", query_id);
#endif
      global_query[query_id]->state = PRE_STAGE;

#ifdef PROFILER
      global_query[query_id]->profiler.start("q_pre-stage");
      global_query[query_id]->profiler.start("q_over_all");
#endif
      // in scala v3, cur_query_num record the number of query in pre stage.
      tman.cur_query_num++;
      // tableint currObj = graph_index.scala_graph->enter_p;
      uint32_t enterpoint = 0;
      // assume enter point is cached.
      // char *ep_data = graph_index.scala_graph->enter_p_ptr;
      char *ep_data = graph_index.top_index->get_vector(enterpoint);

      // Traverse on base index. (Use one-side migrate)
      dist_t dist = fstdistfunc_(
          global_query[query_id]->vector, ep_data, dist_func_param_);
      global_query[query_id]->lowerBound = dist;
      global_query[query_id]->pre_top_cand.emplace(dist, enterpoint);
      global_query[query_id]->pre_cand_set.emplace(-dist, enterpoint);

      uint32_t gt_id = graph_index.top_index->get_global_id(enterpoint);
      global_query[query_id]->visit_hash.CheckAndSet(gt_id);
      // global_query[query_id]->current_node_id = currObj;
      global_query[query_id]->ef = std::max((uint32_t)ef_, anns_param.res_knn);
      // global_query[query_id]->max_path_id = rdma_param.machine_id;
      global_query[query_id]->is_new = false;

      // Pre-stage.
      while (!global_query[query_id]->pre_cand_set.empty()) {
        std::pair<dist_t, uint32_t> cand =
            global_query[query_id]->pre_cand_set.top();
        dist_t candidate_dist = -cand.first;
        if (candidate_dist > global_query[query_id]->lowerBound) {
          break;
        }
        global_query[query_id]->pre_cand_set.pop();

        _mm_prefetch(global_query[query_id]->vector, _MM_HINT_T0);
        uint32_t *ngh_arr = graph_index.top_index->get_ngh_arr(cand.second);
        _mm_prefetch((char *)ngh_arr, _MM_HINT_T0);

        _mm_prefetch((char *)(ngh_arr + 1), _MM_HINT_T0);
        for (uint32_t d = 0; d < ngh_arr[0]; d++) {
          _mm_prefetch(
              graph_index.top_index->get_vector(ngh_arr[d + 1]), _MM_HINT_T0);

          // TODO: can prefetch here.
          uint32_t gt_id = graph_index.top_index->get_global_id(ngh_arr[d]);
          if (!global_query[query_id]->visit_hash.CheckAndSet(gt_id)) {
#ifdef PROF_COMPUTATION
            computation_cnt ++;
#endif
            char *vec_data = graph_index.top_index->get_vector(ngh_arr[d]);
            dist_t dist = fstdistfunc_(
                global_query[query_id]->vector, vec_data, dist_func_param_);
            top_proc_candidate(dist, ngh_arr[d]);
          }
        }
        // printf("all_result_recved__\n");
      }
      // tman.cur_query_num--;

#ifdef DEBUG
      printf("Start to dispatch sub-query of %u\n", query_id);
#endif
      /**
       * Previous stage over, start to dispatch sub-query.
       * First filterred the candidate set.
       */
      // TODO: maybe can use top_candidates to filter.
      memset(
          global_query[query_id]->filter_num, 0,
          sizeof(global_query[query_id]->filter_num));
      while (!global_query[query_id]->pre_top_cand.empty()) {
        auto cand_top = global_query[query_id]->pre_top_cand.top();
        uint32_t gt_id = graph_index.top_index->get_global_id(cand_top.second);
        int mid = graph_index.scala_graph->get_machine_id(gt_id);
        global_query[query_id]->filterred[mid].emplace_back(
          -cand_top.first, gt_id);
        // if (graph_index.top_index->is_last_level(cand_top.second)) {
          // global_query[query_id]->filterred[mid].emplace_back(
          //     -cand_top.first, gt_id);
          // global_query[query_id]->filter_num[mid]++;
        // }
        global_query[query_id]->filter_num[mid]++;
        global_query[query_id]->top_candidates.emplace(cand_top.first, gt_id);
        global_query[query_id]->pre_top_cand.pop();
      }

      // Count
      uint32_t su_cnt = 0;
      uint32_t mx_cnt = 0, mx_mid;
      for (uint32_t m = 0; m < MACHINE_NUM; m++) {
        su_cnt += global_query[query_id]->filter_num[m];
        if (global_query[query_id]->filter_num[m] > mx_cnt) {
          mx_cnt = global_query[query_id]->filter_num[m];
          mx_mid = m;
        }
      }
      global_query[query_id]->leader_machine = mx_mid;
#ifdef DEBUG
      printf("query %u leader machine is %u\n", query_id, mx_mid);
#endif

      // Identify which partition should be selected.

      for (uint32_t m = 0; m < MACHINE_NUM; m++) {
        float ratio_m = (float)global_query[query_id]->filter_num[m] / su_cnt;
        // Always keep the elected leader as a core machine; otherwise an
        // exactly even split can leave the query with no execution target.
        if (ratio_m > 1.0 / (MACHINE_NUM) || m == mx_mid) {
          global_query[query_id]->core_machine.emplace_back(m);
          global_query[query_id]->is_core_machine[m] = 1;
        } else {
          global_query[query_id]->is_core_machine[m] = 0;
          // Move all non-core machine cand to the leader machine.
          for (auto &cand : global_query[query_id]->filterred[m]) {
            global_query[query_id]->filterred[mx_mid].emplace_back(cand);
          }
        }
      }

      // Dispatch sub-query: fork query to core machine.
      bool local_is_core = false;
      for (uint32_t &m : global_query[query_id]->core_machine) {
        // printf("%u ", m);
        // rdma_comm.migrate_query(m, global_query[query_id]);
        if (m == rdma_param.machine_id) {
          // local machine.
          local_is_core = true;
          for (auto &cand : global_query[query_id]->filterred[m]) {
            global_query[query_id]->candidate_set.emplace(cand);
          }
        } else {
          rdma_comm.fork_query(m, global_query[query_id]);
        }
      }
      global_query[query_id]->visit_hash.clear();

      // Send core machine info to leader machine scheduler.
      // printf("send core machine info to leader machine\n");
      // rdma_comm.send_core_machine_info(tman, global_query[query_id]);

      // printf("\n");
#ifdef PROFILER
      global_query[query_id]->profiler.end("q_pre-stage");
#endif

      if (!local_is_core) {
#ifdef DEBUG
        printf("Local not core, new query %u Pre STAGE over\n", query_id);
#endif
        co_yield false;
        // next time wake up will be END state.
      } else {
#ifdef DEBUG
        printf("Local is core, new query %u Pre STAGE over\n", query_id);
#endif
        if (global_query[query_id]->leader_machine == rdma_param.machine_id) {
          // local is leader, give the token.
          global_query[query_id]->has_token = true;
        }
      }

    } else {
#ifdef DEBUG
      printf("recv fork sub-query q%u\n", query_id);
#endif
      global_query[query_id]->state = PRE_STAGE;
    }

    // if local machine is core machine.
    if (global_query[query_id]->state == PRE_STAGE) {
      // Post stage: after dispatching sub-query.
#ifdef DEBUG
      printf("query %u goto post stage\n", query_id);
#endif
      global_query[query_id]->state = POST_STAGE;
      bool all_result_recved = false;
      auto &post_cnt = global_query[query_id]->post_cnt;
      auto &recv_cnt = global_query[query_id]->recv_cnt;
      auto &nocore_post = global_query[query_id]->nocore_post;
      auto &nocore_recv = global_query[query_id]->nocore_recv;
      post_cnt = recv_cnt = 0;
      nocore_post = nocore_recv = 0;

      // tman.post_async = tman.post_task = 0;
      // tman.rpost_async = tman.rpost_task = 0;

      while (!global_query[query_id]->global_term) {
        for (;;) {
#ifdef PROFILER
          m_profiler.start("post_task");
#endif
          int gsz = 0;
          for (gsz = 0; gsz < CAND_GROUP_SIZE &&
                        !global_query[query_id]->candidate_set.empty();
               gsz++) {
            group_nodes[gsz] = global_query[query_id]->candidate_set.top();
            dist_t candidate_dist = -group_nodes[gsz].first;
            if (candidate_dist > global_query[query_id]->lowerBound) {
              break;
            }
            global_query[query_id]->candidate_set.pop();
            // do not release here, since may in top candidates queue.
          }
          // post micro task to remote machine.
          if (gsz > 0) {
#ifdef DEBUG
              printf("try get remote task\n");
#endif
            // Step 1: split remote and local node push local task to task
            // queue..
            for (int g = 0; g < gsz; g++) {
              tableint ngh_id = group_nodes[g].second;
              uint32_t mid = graph_index.scala_graph->get_machine_id(ngh_id);
              if (mid != rdma_param.machine_id) {
                // remote node task.
                tman.add_remote_task(ngh_id, mid, query_id);
                post_cnt++;
              } else {
                // local node.
                post_cnt += graph_index.scala_graph
                                ->scala_graph_v3_get_subquery_task_lb(
                                    ngh_id, query_id, tman, ba_task_queue);
              }
            }
#ifdef DEBUG
            std::cout<<"get remote task over, post\n";
#endif
            // Post micro task to non-core machine.
            rdma_comm.post_remote_task(
                tman, query_id, data_size_ + sizeof(uint64_t), &global_query[query_id]->profiler);

            // sync msg & task to core machine.
            // untill end sync now.
            global_query[query_id]->sync_step++;
            if (global_query[query_id]->sync_step >= SYNC_BAR) {
              global_query[query_id]->sync_msg.lower_bound =
                  global_query[query_id]->lowerBound;
              rdma_comm.post_subquery_sync(global_query[query_id]);

              global_query[query_id]->sync_msg.clear();
              global_query[query_id]->sync_step = 0;
            }
#ifdef DEBUG
            std::cout<<"remote task post over\n";
#endif

            // put subquery to next iteration global subquery queue.
            if (global_query[query_id]->state == POST_STAGE) {
              tman.next_subquery_queue.push_back(query_id);
              // printf("push q%u to next sub\n", query_id);
            }
#ifdef PROFILER
            m_profiler.end("post_task");
#endif
          } else {
            all_result_recved = (recv_cnt == post_cnt) && (nocore_post == nocore_recv);
            // printf(
            //     "q%d post_cnt %d recv_cnt %d\n", query_id, post_cnt,
            //     recv_cnt);

#ifdef PROFILER
            m_profiler.end("post_task");
#endif
            if (!all_result_recved) {
              // global_query[query_id]->state = PAUSE;
              tman.next_subquery_queue.push_back(query_id);
            } else if (
                all_result_recved &&
                global_query[query_id]->state == POST_STAGE) {
              // printf("try to terminate\n");
              break;
            }
            // else all_result_recved &&
            // global_query[query_id]->state == PAUSE, wait for global term.
          }

          // if (global_query[query_id]->state == END) {
          //   printf("Error: query %u state is END\n", query_id);
          //   abort();
          // }

          /*
            suspend to proc recved remote tasks .
            */
          // printf("post suspend with p %u r %u\n", post_cnt, recv_cnt);
          co_yield false;
          // printf("q%u search RESUME\n", query_id);
#ifdef PROFILER
          m_profiler.start("post_stage");
#endif
// #ifdef DEBUG
//           printf("q%u continue\n", query_id);
// #endif

          _mm_prefetch(global_query[query_id]->vector, _MM_HINT_T0);
          // auto &task = *global_query[query_id]->cur_task;
          char *vec_ptr = graph_index.scala_graph->vector_ptr;
          uint32_t part_ofs =
              graph_index.scala_graph->scala_graph_get_partition_ofs();
          // printf(
          //     "local task size: %u\n",
          //     global_query[query_id]->local_task.size());
          if(global_query[query_id]->local_task.size()){
            #ifdef DEBUG
              printf("proc qlocal task\n");
            #endif 
            for (ComputeTask<dist_t> &task : global_query[query_id]->local_task) {
              // printf("%llu %u \n", task.offset, task.deg);
              uint32_t *ngh_arr =
                  checked_scala_v3_ngh_ptr(
                      task.offset, task.deg, task.qid, anns_param.machine_id,
                      "local_sync_task", true);
              _mm_prefetch((char *)ngh_arr, _MM_HINT_T0);
              _mm_prefetch(
                  vec_ptr + ((*(ngh_arr)) - part_ofs) * data_label_size, _MM_HINT_T0);
              _mm_prefetch((char *)(ngh_arr + 1), _MM_HINT_T0);
              for (uint32_t d = 0; d < task.deg; d++) {
                if (d + 1 < task.deg) {
                  _mm_prefetch(
                      vec_ptr + ((*(ngh_arr + d + 1)) - part_ofs) *
                                    data_label_size,
                      _MM_HINT_T0);
                }
                // printf("%u\n", *(ngh_arr + d));
                if (!global_query[query_id]->visit_hash.CheckAndSet(
                        *(ngh_arr + d))) {
#ifdef PROF_COMPUTATION
                  computation_cnt ++;
#endif
                  char *vec_data =
                      vec_ptr + ((*(ngh_arr + d)) - part_ofs) * data_label_size;
                  dist_t dist = fstdistfunc_(
                      global_query[query_id]->vector, vec_data, dist_func_param_);
                  post_proc_candidate(dist, *(ngh_arr + d));
                }
              }
            }
            global_query[query_id]->local_task.clear();
            #ifdef DEBUG
              printf("proc qlocal task over\n");
            #endif 
          }
          

          // Proc balance task queue. (Shared local task)
          // printf("]]]pop ba_task_queue\n");

          for (;;) {
            auto task = ba_task_queue.pop(false);
            if (!task.has_value()) {
              break;
            }
            // Use work stealing across threads.
            do_task(task.value());
          }

          for (;;) {
            auto chunk = global_query[query_id]->con_task_queue.pop();
            if (!chunk.has_value()) {
              break;
            }
#ifdef DEBUG
            printf("chunk size: %u\n", chunk->size());
#endif
            recv_cnt += chunk.value().isCount();
            for (int i = 0; i < chunk.value().size(); i++) {
              auto &ele = chunk.value().getAt(i);
              post_proc_candidate(ele.first, ele.second);
            }
          }
          // printf("ccnt: %u\n", ccnt);
          // printf("]]]pop con_task_queue over\n");

          // printf(
          //     "local micro task size: %u\n",
          //     global_query[query_id]->local_micro_task.size());
          // local_micro_task: recved ngh from other machine.
          if (global_query[query_id]->local_micro_task.size()) {
            #ifdef DEBUG
              printf("proc qlocal micro task\n");
            #endif 
            // printf("proc local micro task\n");
            const uint32_t local_micro_task_size =
                global_query[query_id]->local_micro_task.size();
            for (uint32_t i = 0; i < local_micro_task_size; i++) {
              uint32_t &vid = global_query[query_id]->local_micro_task[i];
              if (i + 1 < local_micro_task_size) {
                _mm_prefetch(
                    vec_ptr +
                        (global_query[query_id]->local_micro_task[i + 1] -
                         part_ofs) *
                            data_label_size,
                    _MM_HINT_T0);
              }
              if (!global_query[query_id]->visit_hash.CheckAndSet(vid)) {
#ifdef PROF_COMPUTATION
                computation_cnt ++;
#endif
                char *vec_data = vec_ptr + (vid - part_ofs) * data_label_size;
                dist_t dist = fstdistfunc_(
                    global_query[query_id]->vector, vec_data, dist_func_param_);
                post_proc_candidate(dist, vid);
              }
            }
            global_query[query_id]->local_micro_task.clear();
            #ifdef DEBUG
              printf("proc qlocal micro task over\n");
            #endif 
          }

          //  post stage async read queue.
          // printf(
          //     "recv_async_read_queue size: %u\n",
          //     global_query[query_id]->recv_async_read_queue.size());
          if(global_query[query_id]->recv_async_read_queue.size()){
            #ifdef DEBUG
              printf("proc async_read\n");
            #endif 
            for (BufferCache &r : global_query[query_id]->recv_async_read_queue) {
              uint32_t ngh_id = r.vector_id;
              char *vec_ptr = r.buffer_ptr;
              // printf("proc ngh_id: %u bufid: %u\n", ngh_id, r.buffer_id);
              _mm_prefetch(vec_ptr, _MM_HINT_T0);
              if (!global_query[query_id]->visit_hash.CheckAndSet(ngh_id)) {
#ifdef PROF_COMPUTATION
                computation_cnt ++;
#endif
                dist_t dist = fstdistfunc_(
                    global_query[query_id]->vector, vec_ptr, dist_func_param_);
                post_proc_candidate(dist, ngh_id);
              }
              rdma_comm.release_cache(r.buffer_id);
              // free(r);
              recv_cnt++;
              // tman.rpost_async++;
              // printf(
              //     "q%u recv_cnt after async %u post_cnt %u tman t%u a%u rt%u "
              //     "ra%u\n",
              //     query_id, recv_cnt, post_cnt, tman.post_task,
              //     tman.post_async, tman.rpost_task, tman.rpost_async);
            }
            // global_query[query_id]->has_cur_async = false;
            global_query[query_id]->recv_async_read_queue.clear();
            #ifdef DEBUG
              printf("proc async_read over\n");
            #endif 
          }
          

          // if (global_query[query_id]->has_cur_res) {
          // printf(
          //     "local_res size: %u\n",
          //     global_query[query_id]->local_res.size());
          if (global_query[query_id]->local_res.size()) {
            #ifdef DEBUG
              printf("proc qlocal_res\n");
            #endif 
            // printf("proc q%u local res\n", query_id);
            for (ResultMsg<dist_t> &result :
                 global_query[query_id]->local_res) {
              // auto &result = *global_query[query_id]->cur_res;
              for (std::pair<dist_t, uint32_t> &r : result.res) {
                dist_t dist = r.first;
                uint32_t candidate_id = r.second;
                post_proc_candidate(dist, candidate_id);
              }
              // count the res from core and nocore result.
              recv_cnt += result.core_cnt;
              nocore_recv += result.nocore_cnt;
            }
            global_query[query_id]->local_res.clear();
            #ifdef DEBUG
              printf("proc qlocal_res over\n");
            #endif 
          }

#ifdef PROFILER
          m_profiler.end("post_stage");
#endif
        }

        // local termination, prop token
        // printf("q%u start term\n", query_id);
        if (global_query[query_id]->core_machine.size() > 1) {
          if (global_query[query_id]->has_token) {
            // if is leader
            // printf("q%u has token\n", query_id);
            if (global_query[query_id]->leader_machine ==
                rdma_param.machine_id) {
              // printf(
              //     "q%u m%u is leader machine\n", query_id,
              //     rdma_param.machine_id);
              bool failed = global_query[query_id]->token_is_black ||
                            global_query[query_id]->proc_is_black;
              global_query[query_id]->token_is_black =
                  global_query[query_id]->proc_is_black = false;
              if (global_query[query_id]->last_is_white && !failed) {
                // This was the second success
                // printf("The second term success, q%u END\n", query_id);
                global_query[query_id]->global_term = true;
                break;
              }
              global_query[query_id]->last_is_white = !failed;
              // printf("The fisrt term success, second term failed\n");
            }
            bool taint = global_query[query_id]->proc_is_black ||
                         global_query[query_id]->token_is_black;
            global_query[query_id]->proc_is_black =
                global_query[query_id]->token_is_black = false;

            // prop token.
            rdma_comm.post_subquery_sync(global_query[query_id], true, taint);
            global_query[query_id]->sync_msg.clear();
            global_query[query_id]->has_token = false;

            // printf("q%u post PROP sync\n", query_id);
            global_query[query_id]->sync_step = 0;
          }
          global_query[query_id]->state = PAUSE;
          // printf("q%u search PAUSE\n", query_id);
        } else {
          // single core machine, just end.
          global_query[query_id]->global_term = true;
          // printf("The first term success, q%u END\n", query_id);
          break;
        }
      }
    }
    // printf("search END\n");
    global_query[query_id]->state = END;
// printf("q%d over\n", query->query_id);
#ifdef DEBUG
    memset(
        global_query[query_id]->tmp_m_cnt, 0,
        sizeof(global_query[query_id]->tmp_m_cnt));
#endif
    while (global_query[query_id]->top_candidates.size() > anns_param.res_knn) {
#ifdef DEBUG
      auto tt = global_query[query_id]->top_candidates.top();
      int mid = graph_index.scala_graph->get_machine_id(tt.second);
      global_query[query_id]->tmp_m_cnt[mid]++;
#endif
      global_query[query_id]->top_candidates.pop();
    }

    while (global_query[query_id]->top_candidates.size() > 0) {
      std::pair<dist_t, tableint> rez =
          global_query[query_id]->top_candidates.top();
#ifdef DEBUG
      int mid = graph_index.scala_graph->get_machine_id(rez.second);
      global_query[query_id]->tmp_m_cnt[mid]++;
#endif
      global_query[query_id]->result.push(std::pair<dist_t, labeltype>(
          rez.first,
          graph_index.scala_graph->kmeans_pos_label_map[rez.second]));
      global_query[query_id]->top_candidates.pop();
    }

    // migrate to origin machine
    if (global_query[query_id]->leader_machine == rdma_param.machine_id &&
        rdma_param.machine_id != global_query[query_id]->origin_machine) {
      // printf(
      //     "migrate q%u to origin machine m%u...\n", query_id,
      //     global_query[query_id]->origin_machine);
      rdma_comm.migrate_query(
          global_query[query_id]->origin_machine, global_query[query_id]);
      co_yield false;
    }
#ifdef PROFILER
    global_query[query_id]->profiler.end("q_over_all");
#endif
    co_yield true;
  }

  void scala_sub_proc_v5(
      TaskManager<dist_t> &tman, std::vector<QueryMsg<dist_t> *> &ret) {
    auto &task_queue = tman.get_local_task_queue();
    auto &node_queue = tman.get_node_task_queue();
    auto &node_res_queue = tman.get_node_result_queue();

    auto &result_queue = tman.get_result_queue();
    // migrate query queue, in v3 only used for end query.
    auto &query_queue = tman.get_query_queue();
    auto &async_queue = tman.get_async_task_queue();
    auto &subquery_queue = tman.get_subquery_queue();
#ifdef PROFILER
    Profiler &m_profiler = *all_profiler.getLocal();
#endif

    // Leader machine leader thread, do schduler.
    //     if (anns_param.machine_id == 0 && ThreadPool::getTID() == 0) {
    //       // collect all thread's core_info.
    // #ifdef DEBUG
    //       // printf("collect schedule plan\n");
    // #endif
    //       for (uint32_t t = 1; t < getActiveThreads(); t++) {
    //         auto *tma = task_ma.getRemote(t);
    //         // TODO: change to PaddedLock.
    //         tma->schequery_mutex.lock();
    //         for (auto sche_msg : tma->schedule_query_queue) {
    //           tman.schedule_query_queue.push_back(sche_msg);
    //         }
    //         tma->schedule_query_queue.clear();
    //         tma->schequery_mutex.unlock();
    //       }
    // #ifdef DEBUG
    //     // printf("generate schedule plan\n");
    // #endif
    //     // Generate schedule plan.
    //     scheduler.generate_schedule_plan(tman.schedule_query_queue);
    //     // Send schedule plan to machine.
    //     rdma_comm.send_schedule_plan(scheduler);
    //     // printf("send schedule plan over\n");
    //   }

    // printf("proc subquery\n");
    /**
     * Proc remote task, local task, async read, and results.
     */
    if (query_queue.size()) {
#ifdef PROFILER
      m_profiler.start("g_query_Q");
#endif
      for (auto qid : query_queue) {
        if (global_query[qid]->is_new) {
          // new query
          // printf("recv new sub-query %u\n", qid);
          global_query[qid]->is_new = false;
          if (global_query[qid]->leader_machine == rdma_param.machine_id) {
            global_query[qid]->has_token = 1;
            // printf("q%u leader is me.\n", qid);
          }
          global_query[qid]->coro = coro_sub_query_v5(qid);
        } else {
// end query
// printf("recv end q%u\n", qid);
#ifdef PROFILER
          global_query[qid]->profiler.end("q_over_all");
#endif
          ret.push_back(global_query[qid]);
          tman.cur_query_num--;
        }
      }
      query_queue.clear();
#ifdef PROFILER
      m_profiler.end("g_query_Q");
#endif
    }

    if (node_queue.size()) {
#ifdef DEBUG
        std::cout << "proc node task\n";
#endif
      for (NodeTask &task : node_queue) {
        if (task.machine_id == anns_param.machine_id) {  // local task.
          std::cout << "Error: non core machine never recv local node task.\n";
          abort();
        } else {
          /*
            1. For original core machine, local compute, compute and post res &
            post_cnt. use NodeResult queue
            2. For remote other core machines msg, post as sync msg, use sync
            post queue.
            3. For remote non-core machine task, post as remote task,
            node res post_cnt++, use task post queue.
          */

          tman.node_result_queue[task.machine_id].emplace_back(task.qid);
          NodeResult<dist_t> &res_msg =
              tman.node_result_queue[task.machine_id].back();

          // dispatch node task.
          graph_index.scala_graph->scala_graph_v3_get_nodetask_msg(
              tman, task, res_msg);

          // printf("]]post msg post_cnt +=%d\n", res_msg.post_cnt);

          // ResultMsg<dist_t> res_msg(task.qid);
          // Start to traverse local partition ngh, based on task offset got.
          // printf(
          //     "]]] task offset %llu qid %llu deg %llu machine_id %llu\n",
          //     task.offset, task.qid, task.deg, task.machine_id);
          if(task.offset){
            uint32_t *ngh_arr =
                checked_scala_v3_ngh_ptr(
                    task.offset, task.deg, task.qid, task.machine_id,
                    "node_task_local");
            char *vec_ptr = graph_index.scala_graph->vector_ptr;
            uint32_t part_ofs =
                graph_index.scala_graph->scala_graph_get_partition_ofs();
            _mm_prefetch(
                vec_ptr + ((*(ngh_arr)) - part_ofs) * data_label_size, _MM_HINT_T0);
            _mm_prefetch(global_query[task.qid]->vector, _MM_HINT_T0);
            _mm_prefetch((char *)(ngh_arr + 1), _MM_HINT_T0);
            for (uint32_t d = 0; d < task.deg; d++) {
              if (d + 1 < task.deg) {
                _mm_prefetch(
                    vec_ptr + ((*(ngh_arr + d + 1)) - part_ofs) *
                                  data_label_size,
                    _MM_HINT_T0);
              }
              if (!global_query[task.qid]->visit_hash.CheckAndSet(ngh_arr[d])) {
#ifdef PROF_COMPUTATION
                computation_cnt ++;
#endif
                char *vec_data = vec_ptr + ((*(ngh_arr + d)) - part_ofs) * data_label_size;
                dist_t dist = fstdistfunc_(
                    global_query[task.qid]->vector, vec_data, dist_func_param_);
                res_msg.res.emplace_back(dist, ngh_arr[d]);
              }
            }
          }
        }

        if (tman.node_result_queue[task.machine_id].size() >= POST_RES_MERGE) {
          rdma_comm.post_node_result(tman, task.machine_id);
        }
        if (tman.noncore_sync_msg[task.machine_id].size() >=
            POST_NODESYNC_MERGE) {
          rdma_comm.post_node_sync(tman, task.machine_id);
        }
      }
      /*
        Send NodeResult to origin machine (NodeResult queue).
        Send Sync to core machine (sync queue).
        Send Task to non-core machine (task queue).
      */
      // TODO: check if swap queue may have bugs.
      rdma_comm.post_remote_res_task_sync(tman);
      node_queue.clear();
#ifdef DEBUG
        std::cout << "proc node task over\n";
#endif
    }

    if (node_res_queue.size()) {
#ifdef DEBUG
      std::cout << "proc node res\n";
#endif
      for (NodeResult<dist_t> &res_msg : node_res_queue) {
        // printf("proc %d NodeResult ...\n", res_msg.qid);
        if (global_query[res_msg.qid]->state == END) {
          printf("Error: q%u is END when recv res\n", res_msg.qid);
          abort();
        }
        // TODO: can polish here.
        ResultMsg<dist_t> tmp_res_msg(res_msg.qid);
        tmp_res_msg.res = res_msg.res;
        // Add new post count number.

        global_query[res_msg.qid]->nocore_post += res_msg.post_cnt;
        global_query[res_msg.qid]->local_res.push_back(tmp_res_msg);
        if (global_query[res_msg.qid]->state == PAUSE) {
          // printf("voke q%u\n", res_msg.qid);
          subquery_queue.push_back(res_msg.qid);
          global_query[res_msg.qid]->state = POST_STAGE;
        }
      }
      node_res_queue.clear();
#ifdef DEBUG
      std::cout << "proc node res over\n";
#endif
    }

    if (task_queue.size()) {
#ifdef DEBUG
      std::cout << "proc task queue\n";
#endif
#ifdef PROFILER
      m_profiler.start("g_task_Q");
#endif
      // printf("deal task queue\n");
      for (ComputeTask<dist_t> &task : task_queue) {
        if (task.machine_id == anns_param.machine_id) {  // local task.
          global_query[task.qid]->cur_task = &task;
          global_query[task.qid]->has_cur_task = true;
#ifdef DEBUG
          printf("local task\n");
#endif
          // printf("local task from qid: %u\n", task.qid);
          if (global_query[task.qid]->coro()) {
            // query over. proc
            ret.push_back(global_query[task.qid]);
            tman.cur_query_num--;
          }

        } else {
          // #ifdef PROFILER
          //           // printf("start q%u task_remote timer\n", task.qid);
          //           m_profiler.start("g_task_remote");
          //           m_profiler.count("g_task_remote_n", task.deg);
          // #endif
          // printf("remote task from qid: %u\n", task.qid);
          tman.result_queue[task.machine_id].emplace_back(task.qid, task.core_cnt, task.nocore_cnt);
          ResultMsg<dist_t> &res_msg =
              tman.result_queue[task.machine_id].back();
          // ResultMsg<dist_t> res_msg(task.qid);
          uint32_t *ngh_arr;
          // uint32_t mptr[INLINE_DEG_BAR];
#ifdef DEBUG
          printf("remote task q%u deg:%u\n", task.qid, task.deg);
#endif
          if (task.deg <= INLINE_DEG_BAR) {
            // printf("low deg task\n");
            // for (uint32_t d = 0; d < task.deg; d++) {
            //   // tmp_ptr[d] = vid;
            //   // printf("task.offset %llu\n", task.offset);
            //   mptr[d] = (uint32_t)(task.offset & 0xffffffff);
            //   task.offset >>= 32LL;
            // }
            // ngh_arr = mptr;

            ngh_arr = (uint32_t *)(&task.offset);
          } else {
            // printf(
            //     "task offset %llu qid %llu deg %llu machine_id %llu\n",
            //     task.offset, task.qid, task.deg, task.machine_id);
            ngh_arr = checked_scala_v3_ngh_ptr(
                task.offset, task.deg, task.qid, task.machine_id,
                "remote_task", true);
          }

          char *vec_ptr = graph_index.scala_graph->vector_ptr;
          uint32_t part_ofs =
              graph_index.scala_graph->scala_graph_get_partition_ofs();
          _mm_prefetch(
              vec_ptr + ((*(ngh_arr)) - part_ofs) * data_label_size, _MM_HINT_T0);
          _mm_prefetch(global_query[task.qid]->vector, _MM_HINT_T0);
          _mm_prefetch((char *)(ngh_arr + 1), _MM_HINT_T0);
          for (uint32_t d = 0; d < task.deg; d++) {
            // if (task.deg <= INLINE_DEG_BAR) {
            //   printf("low deg d%d ngh%d\n", d, ngh_arr[d]);
            // }
            if (d + 1 < task.deg) {
              _mm_prefetch(
                  vec_ptr + ((*(ngh_arr + d + 1)) - part_ofs) *
                                data_label_size,
                  _MM_HINT_T0);
            }
            if (!global_query[task.qid]->visit_hash.CheckAndSet(ngh_arr[d])) {
#ifdef PROF_COMPUTATION
              computation_cnt ++;
#endif
              char *vec_data = vec_ptr + ((*(ngh_arr + d)) - part_ofs) * data_label_size;
              dist_t dist = fstdistfunc_(
                  global_query[task.qid]->vector, vec_data, dist_func_param_);
              if (!task.ef_fill || dist < task.lower_bound) {
                res_msg.res.emplace_back(dist, ngh_arr[d]);
              }
            }
          }
          // tman.result_queue[task.machine_id].push_back(res_msg);
          if (tman.result_queue[task.machine_id].size() >= POST_RES_MERGE) {
            // #ifdef PROFILER
            //             m_profiler.start("g_post_result");
            // #endif

            rdma_comm.post_remote_result(tman, task.machine_id);
            // #ifdef PROFILER
            //             m_profiler.end("g_post_result");
            // #endif
          }

          // #ifdef PROFILER
          //           m_profiler.end("g_task_remote");
          // #endif
          // rdma_comm.post_result(res_msg, task.machine_id);
        }
      }
      // #ifdef PROFILER
      //       m_profiler.start("g_post_result");
      // #endif
      // printf("post res\n");
      rdma_comm.post_remote_result(tman);
      // printf("post res over\n");
      // #ifdef PROFILER
      //       m_profiler.end("g_post_result");
      // #endif
      task_queue.clear();
#ifdef PROFILER
      m_profiler.end("g_task_Q");
#endif
#ifdef DEBUG
      std::cout << "proc task queue over\n";
#endif
    }

    // async_queue only store the query in pre-stage.
    if (async_queue.size()) {
#ifdef DEBUG
      printf("proc async queue\n");
#endif
#ifdef PROFILER
      m_profiler.start("g_async_Q");
#endif
      for (size_t &q : async_queue) {
        // all async read have finished.
        global_query[q]->has_cur_async = true;

        // printf("local task from qid: %u\n", task.qid);
        if (global_query[q]->coro()) {
          // query over. proc
          ret.push_back(global_query[q]);
          tman.cur_query_num--;
        }
      }
      async_queue.clear();
#ifdef DEBUG
      printf("proc async queue over\n");
#endif
#ifdef PROFILER
      m_profiler.end("g_async_Q");
#endif
    }

    // proc result queue
    if (result_queue.size()) {
#ifdef PROFILER
      m_profiler.start("g_result_Q");
#endif
#ifdef DEBUG
      printf("proc result queue\n");
#endif
      for (ResultMsg<dist_t> &res_msg : result_queue) {
        // if query is core, dont voke.
        // printf("proc res q%u \n", res_msg.qid);
        if (global_query[res_msg.qid]->state == PRE_STAGE) {
          // printf("push q%u PRE STAGE local res\n", res_msg.qid);
          global_query[res_msg.qid]->cur_res = &res_msg;
          global_query[res_msg.qid]->has_cur_res = true;
          // printf("recv res.....\n");
          if (global_query[res_msg.qid]->coro()) {
            // query over. proc
            ret.push_back(global_query[res_msg.qid]);
            tman.cur_query_num--;
          }
        } else {
          // printf("push q%u POST STAGE local res\n", res_msg.qid);
          if (global_query[res_msg.qid]->state == END) {
            printf("WARN: q%u is END when recv res\n", res_msg.qid);
            abort();
          }
          global_query[res_msg.qid]->local_res.push_back(res_msg);
          if (global_query[res_msg.qid]->state == PAUSE) {
            // printf("voke q%u\n", res_msg.qid);
            subquery_queue.push_back(res_msg.qid);
            global_query[res_msg.qid]->state = POST_STAGE;
          }
          // printf("push q%u post-stage res over\n", res_msg.qid);
        }
      }
      result_queue.clear();
#ifdef DEBUG
      printf("proc res over\n");
#endif
#ifdef PROFILER
      m_profiler.end("g_result_Q");
#endif
    }

    /**
     * Proc local sub-query.
     */
    if (subquery_queue.size()) {
#ifdef PROFILER
      m_profiler.start("g_subquery_Q");
#endif
      for (uint32_t &sqid : subquery_queue) {
        if (global_query[sqid]->coro()) {
          ret.push_back(global_query[sqid]);
          tman.cur_query_num--;
        }
      }
      subquery_queue.clear();
#ifdef PROFILER
      m_profiler.end("g_subquery_Q");
#endif
    }
  }

  /**
   * ScalaSearch v3:
   * Try to improve throughput & latency:
   * 1. sub-query
   * ScalaSearch v4:
   * 2. compute-migration
   * 3. load balance sechduler
   *
   * query = corobj()
   * coro[QUERY_NUM]
   */
  std::vector<QueryMsg<dist_t> *> scala_search_sub(size_t qid) {
#ifdef PROFILER
    Profiler &m_profiler = *all_profiler.getLocal();
    m_profiler.start("g_overall");
#endif

    std::vector<QueryMsg<dist_t> *> ret;
    TaskManager<dist_t> &tman = *task_ma.getLocal();

    global_query[qid]->coro = coro_sub_query_v5(qid);

    // uint32_t res_cnt = 0;
    auto &task_queue = tman.get_local_task_queue();
    auto &node_queue = tman.get_node_task_queue();
    auto &result_queue = tman.get_result_queue();
    auto &query_queue = tman.get_query_queue();  // migrate query queue
    auto &async_queue = tman.get_async_task_queue();
    auto &subquery_queue = tman.get_subquery_queue();

    for (;;) {
      // put new (next) local task to cur task queue.
      tman.swap_task_queue();
      // collect task from other machine

#ifdef PROFILER
      rdma_comm.poll_task_result(tman, &m_profiler);
#else
      rdma_comm.poll_task_result(tman);
#endif
      // printf("tman.cur_query_num %u\n", tman.cur_query_num);

      if (!task_queue.size() && !result_queue.size() && !query_queue.size() &&
          !async_queue.size() && !subquery_queue.size() && !node_queue.size()) {
        if (tman.cur_query_num < get_runtime_query_group_size()) {
// all task is empty, add new query.
#ifdef PROFILER
          m_profiler.end("g_overall");
#endif
          return ret;
        } else {
          work_steal();
        }
      } else {
        scala_sub_proc_v5(tman, ret);
      }
    }

    // Should never be here.
    return ret;
  }

  std::vector<QueryMsg<dist_t> *> scala_search_sub_wait_finish() {
    std::vector<QueryMsg<dist_t> *> ret;
    auto &tman = *task_ma.getLocal();
    auto &task_queue = tman.get_local_task_queue();
    auto &node_queue = tman.get_node_task_queue();
    auto &result_queue = tman.get_result_queue();
    auto &query_queue = tman.get_query_queue();  // migrate query queue
    auto &async_queue = tman.get_async_task_queue();
    auto &subquery_queue = tman.get_subquery_queue();

    tman.swap_task_queue();
#ifdef PROFILER
    Profiler &m_profiler = *all_profiler.getLocal();
    rdma_comm.poll_task_result(tman, &m_profiler);
#else
    rdma_comm.poll_task_result(tman);
#endif

    if (query_queue.size() || result_queue.size() || task_queue.size() ||
        async_queue.size() || subquery_queue.size() || node_queue.size()) {
      scala_sub_proc_v5(tman, ret);
    } else {
      work_steal();
    }

    return ret;
  }

  void scala_search_sub_standby(uint32_t query_num) {
#ifdef PROFILER
    Profiler &m_profiler = *all_profiler.getLocal();
    m_profiler.start("g_overall");
#endif
    std::vector<QueryMsg<dist_t> *> ret;
    auto &tman = *task_ma.getLocal();

    auto &task_queue = tman.get_local_task_queue();
    auto &node_queue = tman.get_node_task_queue();
    auto &result_queue = tman.get_result_queue();
    auto &query_queue = tman.get_query_queue();  // migrate query queue
    auto &async_queue = tman.get_async_task_queue();
    auto &subquery_queue = tman.get_subquery_queue();

    for (;;) {
      // put new (next) local task to cur task queue.
      if (check_term()) break;
      tman.swap_task_queue();

#ifdef PROFILER
      rdma_comm.poll_task_result(tman, &m_profiler);
#else
      rdma_comm.poll_task_result(tman);
#endif

      if (query_queue.size() || result_queue.size() || task_queue.size() ||
          async_queue.size() || subquery_queue.size() || node_queue.size()) {
        scala_sub_proc_v5(tman, ret);
      } else {
        work_steal();
      }
    }
#ifdef PROFILER
    m_profiler.end("g_overall");
#endif
  }
};

}  // namespace hnswlib
