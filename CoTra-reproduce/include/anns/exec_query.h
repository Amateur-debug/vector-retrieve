#pragma once

#include <omp.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <queue>
#include <sstream>
#include <unordered_set>
#include <chrono>

#include "anns/scala_search.h"
#include "coromem/include/galois/bag.h"
#include "coromem/include/galois/loops.h"
#include "coromem/include/runtime/range.h"
#include "coromem/include/share_mem.h"
#include "coromem/include/utils.h"
#ifdef DEBUG
const int debug_id = 4;
#endif

using namespace std;
using namespace hnswlib;

static vector<size_t> get_eval_efs() {
  const char *ef_env = std::getenv("COTRA_EF_LIST");
  if (ef_env == nullptr || ef_env[0] == '\0') {
#ifdef DEBUG
    return {30, 400};
#else
    return {3, 4, 5, 6, 8, 10, 15, 20, 30, 40, 50, 70, 100, 150, 200, 250,
            300, 400, 500, 600, 700};
#endif
  }

  vector<size_t> efs;
  std::stringstream ss(ef_env);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (item.empty()) {
      continue;
    }
    efs.push_back(static_cast<size_t>(std::stoul(item)));
  }
  if (efs.empty()) {
    throw std::runtime_error("COTRA_EF_LIST is set but contains no valid ef");
  }
  return efs;
}

static bool cotra_exec_env_flag_enabled(const char *name) {
  const char *env = std::getenv(name);
  if (env == nullptr || env[0] == '\0') {
    return false;
  }
  std::string value(env);
  return value != "0" && value != "false" && value != "False";
}

template<typename dist_t>
static void test_vs_recall(
    AnnsParameter &anns_param, ScalaSearch<dist_t> &appr_alg, 
    vector<std::priority_queue<std::pair<dist_t, labeltype>>> &answers) {
  char *query_ptr = anns_param.query_ptr;
  size_t vecsize = anns_param.vecsize;
  // anns_param.qsize = 10000;
  size_t qsize = anns_param.qsize;
  printf("query size: %llu vecsize: %llu\n", qsize, vecsize);
  // vector<std::priority_queue<std::pair<dist_t, labeltype>>> &answers =
  //     anns_param.answers;
  size_t k = anns_param.res_knn;

  vector<size_t> efs = get_eval_efs();

#if defined(PROF_Q_DISTRI) || defined(PROF_ALL_Q_DISTRI)
  efs = {300};
#endif

  for (size_t ef : efs) {
    appr_alg.set_ef(ef);
    size_t query_load;
    std::atomic<size_t> correct(0);
    std::atomic<size_t> total(0);

    switch (anns_param.app_type) {
      case SINGLE_MACHINE: {
        size_t load_start = 0;
        query_load = qsize;  
#ifdef PROF_Q_DISTRI
        qsize = 3;
        load_start = 0;
#endif
// #ifdef PROF_COMPUTATION
//         qsize = 1;
//         load_start = 0;
// #endif
#ifdef PROF_LAZY
        qsize = 1;
        load_start = 0;
#endif
       

#ifdef DEBUG
        qsize = 10;
        load_start = 0;
#ifdef PROFILER
        on_each([&](uint64 tid, uint64 total) {
          auto *local_prof = appr_alg.all_profiler.getLocal();
          Profiler &m_profiler = *local_prof;
          m_profiler.get_mhz();
        });
#endif
#endif

        // Single machine only.
        RangeIterator all(qsize);
        on_each(all);
        StopW stopw = StopW();

        do_all(
            iterate(all),
            [&](const uint32 &i) {
              std::priority_queue<std::pair<dist_t, labeltype>> result;
              result =
                  appr_alg.localSearchKnn(query_ptr + vecsize * (load_start + i), k, load_start + i);

              std::priority_queue<std::pair<dist_t, labeltype>> gt(answers[i]);
              unordered_set<labeltype> g;
              total.fetch_add(gt.size());
#ifdef DEBUG
              std::cout<<"q "<<load_start + i <<" ground truth:\n";
#endif
              while (gt.size()) {
#ifdef DEBUG
              std::cout<< gt.top().second <<" ";
#endif
                g.insert(gt.top().second);
                gt.pop();
              }
#ifdef DEBUG
              std::cout<<"\n";
#endif
              uint32_t per_q_cnt = 0;
              while (result.size()) {
                if (g.find(result.top().second) != g.end()) {
                  correct.fetch_add(1);
                  per_q_cnt ++;
                } else {
                }
                result.pop();
              }
#ifdef DEBUG
              // Profiler &m_profiler = appr_alg.all_profiler.getLocal();
              std::cout<<"q "<<load_start + i <<" correct: "<<per_q_cnt <<"\n";
              // m_profiler.report();
              // m_profiler.clear();
#endif
            },
            no_stats(), loopname("Reset"));
        float recall = 1.0f * correct.load() / total.load();
        float time_us_per_query = stopw.getElapsedTimeMicro() / query_load;
        float qps = 1000000.0 * query_load / stopw.getElapsedTimeMicro();
        cout << ef << "\t" << recall << "\t" << time_us_per_query << " us\t"
             << qps << " /s\n";
        if (recall > 1.0) {
          cout << recall << "\t" << time_us_per_query << " us\n";
          break;
        }

#ifdef PROF_COMPUTATION
        std::cout<<"Computation Num Tocal : "<< appr_alg.computation_cnt << " Avg: " 
          << appr_alg.computation_cnt / query_load << "\n";
        appr_alg.computation_cnt = 0;
#endif

#ifdef PROF_ALL_Q_DISTRI
        printf("all_q_distr_cnt :\n");
        size_t all_q_distr_sum = 0;
        for(uint32_t m = 0 ;m <MACHINE_NUM; m++){
          all_q_distr_sum += appr_alg.all_q_distr_cnt[m];
        }
        for(uint32_t m = 0 ;m <MACHINE_NUM; m++){
          printf("%llu %.3f %.3f\n", appr_alg.all_q_distr_cnt[m],  
            appr_alg.all_q_distr_cnt[m]*100.0/all_q_distr_sum,
            appr_alg.all_q_dist_cnt[m]/appr_alg.all_q_distr_cnt[m]);
        }
#endif

      } break;

      case B1: {
        // Baseline1
        // Assign query load.
        query_load = qsize / MACHINE_NUM;
        // printf("query_load: %llu\n", query_load);
        size_t load_start = appr_alg.rdma_param.machine_id * query_load;

        appr_alg.initTerm();
        RangeIterator all(query_load);
        on_each(all);

        StopW stopw = StopW();
        do_all(
            iterate(all),
            [&](const uint32 &i) {
              std::priority_queue<std::pair<dist_t, labeltype>> result;
              result = appr_alg.remoteSearchKnn(
                  query_ptr + vecsize * (load_start + i), k);

              std::priority_queue<std::pair<dist_t, labeltype>> gt(
                  answers[load_start + i]);
              unordered_set<labeltype> g;
              total.fetch_add(gt.size());
              while (gt.size()) {
                g.insert(gt.top().second);
                gt.pop();
              }
              while (result.size()) {
                if (g.find(result.top().second) != g.end()) {
                  correct.fetch_add(1);
                } else {
                }
                result.pop();
              }
            },
            no_stats(), loopname("Reset"));

        appr_alg.sendTerm(correct.load(), total.load(), query_load);
        // Wait for end.
        appr_alg.termStandBy(k);

        if (appr_alg.rdma_param.machine_id == 0) {
          appr_alg.printQueryBatchInfo(stopw.getElapsedTimeMicro());
        }
        float recall = 1.0f * correct.load() / total.load();
        float time_us_per_query = stopw.getElapsedTimeMicro() / query_load;
        float qps = 1000000.0 * query_load / stopw.getElapsedTimeMicro();
        cout << ef << "\t" << recall << "\t" << time_us_per_query << " us\t"
             << qps << " /s\n";
        if (recall > 1.0) {
          cout << recall << "\t" << time_us_per_query << " us\n";
          break;
        }

      } break;

      case B2: {
        // Baseline2

#ifdef DEBUG
        qsize = 5;
#endif
        RangeIterator all(qsize);
        on_each(all);
        query_load = qsize;

        double cumu_lat_us = 0.0;
        double avg_lat_us = 0.0;

        StopW stopw = StopW();
        if (appr_alg.rdma_param.machine_id == 0) {
          // baseline2 leader
          do_all(
              iterate(all),
              [&](const uint32 &i) {
                // Send query to other machines ...
                appr_alg.dispatch_query(query_ptr + vecsize * i, i, k);
                // here to change
                std::priority_queue<std::pair<dist_t, labeltype>> result =
                    appr_alg.localSearchKnn(query_ptr + vecsize * i, k);

                std::priority_queue<std::pair<dist_t, labeltype>>
                    sub_result[MACHINE_NUM];

                appr_alg.collect_query(result, k, sub_result);

                std::priority_queue<std::pair<dist_t, labeltype>> gt(answers[i]);
                unordered_set<labeltype> g;
                total.fetch_add(gt.size());
#ifdef DEBUG
                std::cout << "gt: \n ";
#endif
                while (gt.size()) {
                  g.insert(gt.top().second);
#ifdef DEBUG
                  std::cout << gt.top().second << " ";
#endif
                  gt.pop();
                }
#ifdef DEBUG
                std::cout << "\n ";
#endif
                while (result.size()) {
                  if (g.find(result.top().second) != g.end()) {
                    correct.fetch_add(1);
                  } else {
                  }
                  result.pop();
                }
              },
              no_stats(), loopname("Reset"));

          avg_lat_us = cumu_lat_us/query_load;
        } else {
          // baseline2 member
          on_each(
              [&](uint64 tid, uint64 total) {
                for (;;) {
                  // Send query to other machines ...
                  appr_alg.dispatch_query(query_ptr + vecsize * tid, 0, k);

                  std::priority_queue<std::pair<dist_t, labeltype>> result =
                      appr_alg.localSearchKnn(query_ptr + vecsize * tid, k);

                  appr_alg.collect_query(result, k);
                }
              },
              no_stats(), loopname("Reset"));
        }
        float recall = 1.0f * correct.load() / total.load();
        float time_us_per_query = stopw.getElapsedTimeMicro() / query_load;
        float qps = 1000000.0 * query_load / stopw.getElapsedTimeMicro();
        cout << ef << "\t" << recall << "\t" << avg_lat_us << " us\t"
             << qps << " /s\n";
        if (recall > 1.0) {
          cout << recall << "\t" << avg_lat_us << " us\n";
          break;
        }

#ifdef PROF_COMPUTATION
        std::cout<<"Computation Num Tocal : "<< appr_alg.computation_cnt << " Avg: " 
          << appr_alg.computation_cnt/query_load << "\n";
        appr_alg.computation_cnt = 0;
#endif
      } break;

      case B1_ASYNC: {
        // Baseline1
        // Assign query load.
        query_load = qsize / MACHINE_NUM;
        // printf("query_load: %llu\n", query_load);
        size_t load_start = appr_alg.rdma_param.machine_id * query_load;
#ifdef DEBUG
        load_start = 0;
        query_load = 1;
#endif

        appr_alg.initTerm();
        RangeIterator all(query_load);
        on_each(all);

        StopW stopw = StopW();

        do_all(
            iterate(all),
            [&](const uint32 &i) {
              std::priority_queue<std::pair<dist_t, labeltype>> result;
              result = appr_alg.remoteAsyncSearchKnn(
                  query_ptr + vecsize * (load_start + i), k);

              std::priority_queue<std::pair<dist_t, labeltype>> gt(
                  answers[load_start + i]);
              unordered_set<labeltype> g;
              total.fetch_add(gt.size());
              while (gt.size()) {
                g.insert(gt.top().second);
                gt.pop();
              }
              while (result.size()) {
                if (g.find(result.top().second) != g.end()) {
                  correct.fetch_add(1);
                } else {
                }
                result.pop();
              }
            },
            no_stats(), loopname("Reset"));

        appr_alg.sendTerm(correct.load(), total.load(), query_load);
        // Wait for end.
        appr_alg.termStandBy(k);

        if (appr_alg.rdma_param.machine_id == 0) {
          appr_alg.printQueryBatchInfo(stopw.getElapsedTimeMicro());
        }
        float recall = 1.0f * correct.load() / total.load();
        float time_us_per_query = stopw.getElapsedTimeMicro() / query_load;
        float qps = 1000000.0 * query_load / stopw.getElapsedTimeMicro();
        cout << ef << "\t" << recall << "\t" << time_us_per_query << " us\t"
             << qps << " /s\n";
        if (recall > 1.0) {
          cout << recall << "\t" << time_us_per_query << " us\n";
          break;
        }
      } break;

      case B2KmeansBatch: {
        // Baseline2
        query_load = qsize;
        // query_load = qsize / MACHINE_NUM;
        // size_t load_start = appr_alg.rdma_param.machine_id * query_load;

#ifdef DEBUG
        query_load = 5;
#endif
        RangeIterator all(query_load);
        on_each(all);

        double cumu_lat_us = 0.0;
        double avg_lat_us = 0.0;

        appr_alg.scala_search_init(query_ptr, vecsize, query_load);
        // appr_alg.initTerm();
        // printf("search init over\n");

        StopW stopw = StopW();

        std::atomic<size_t> load_cnt(0);
        
        if (appr_alg.rdma_param.machine_id == 0) {
          do_all_standby(
            iterate(all),
            [&](const uint32 &i) {
              // Send query to other machines ...
              // 1. dispatch new query
              uint32_t new_qid = i;
              #ifdef DEBUG
                printf("q%u\n", new_qid);
              #endif
              // printf("]]] new q%u\n", new_qid);

              appr_alg.dispatch_kmeans_query_start(new_qid, k);

              std::vector<uint32_t> b2_finished_queue = appr_alg.b2_deal_local(k);
              for(uint32_t q : b2_finished_queue) {
                // printf("]]] q%u finished\n");
                // proc end query.
                auto &result = appr_alg.global_query[q]->result;
                std::priority_queue<std::pair<dist_t, labeltype>> gt(answers[q]);
                unordered_set<labeltype> g;
                total.fetch_add(gt.size());
                while (gt.size()) {
                  g.insert(gt.top().second);
                  gt.pop();
                }
                while (result.size()) {
                  if (g.find(result.top().second) != g.end()) {
                    correct.fetch_add(1);
                  } else {
                  }
                  result.pop();
                }
                load_cnt.fetch_add(1);
              }
            },
            [&](){
              for(;;){
                std::vector<uint32_t> b2_finished_queue = appr_alg.b2_deal_local_standby(k);
                for(uint32_t q : b2_finished_queue) {
                  // proc end query.
                  auto &result = appr_alg.global_query[q]->result;
                  std::priority_queue<std::pair<dist_t, labeltype>> gt(answers[q]);
                  unordered_set<labeltype> g;
                  total.fetch_add(gt.size());
                  while (gt.size()) {
                    g.insert(gt.top().second);
                    gt.pop(); 
                  }
                  while (result.size()) {
                    if (g.find(result.top().second) != g.end()) {
                      correct.fetch_add(1);
                    } else {
                    }
                    result.pop();
                  }
                  load_cnt.fetch_add(1);
                }
                if(load_cnt.load() == query_load){
                  break;
                }
              }
            },
            no_stats(), loopname("Reset"));
        } else {
          on_each(
            [&](uint64 tid, uint64 total) {
            appr_alg.b2_member_deal_local_standby(k);
          });
        }
        float recall = 1.0f * correct.load() / total.load();
        float time_us_per_query = stopw.getElapsedTimeMicro() / query_load;
        float qps = 1000000.0 * query_load / stopw.getElapsedTimeMicro();
        cout << ef << "\t" << recall << "\t" << avg_lat_us << " us\t"
             << qps << " /s\n";
        if (recall > 1.0) {
          cout << recall << "\t" << avg_lat_us << " us\n";
          break;
        }

#ifdef PROF_COMPUTATION
        std::cout<<"Computation Num Tocal : "<< appr_alg.computation_cnt << " Avg: " 
          << appr_alg.computation_cnt/query_load << "\n";
        appr_alg.computation_cnt = 0;
#endif
      } break;


      case ScalaANN_v3: {
        // ScalaANN v3: sub-query (no-waiting latency-throughput)
        // Note: focus on the upper bound of low-latency-low-work_efficiency.
        // Assign query load.
        query_load = qsize / MACHINE_NUM;
        size_t avg_max_load = qsize / MACHINE_NUM;
        size_t load_start = appr_alg.rdma_param.machine_id * query_load;
#ifdef DEBUG
        if (appr_alg.rdma_param.machine_id == 0) {
          query_load = 20;
        } else {
          query_load = 0;
        }
        load_start = 0;
        
#endif

        appr_alg.initTerm();
        RangeIterator all(query_load);
        on_each(all);
        std::atomic<size_t> load_cnt(0);
        const bool skip_recall = cotra_exec_env_flag_enabled("COTRA_SKIP_RECALL");
        const uint32_t runtime_query_group_size = get_runtime_query_group_size();
        if (appr_alg.rdma_param.machine_id == 0) {
          printf("COTRA_SEARCH_ONLY_CONFIG ef=%zu skip_recall=%u query_group_size=%u query_load_per_rank=%zu qsize=%zu\n",
                 ef, skip_recall ? 1 : 0, runtime_query_group_size,
                 query_load, qsize);
        }
#ifdef LAT
        // for latency record
        std::chrono::high_resolution_clock::time_point start[qsize];
        std::chrono::high_resolution_clock::time_point end[qsize];
#endif

#ifdef PROFILER
        on_each([&](uint64 tid, uint64 total) {
          auto *local_prof = appr_alg.all_profiler.getLocal();
          Profiler &m_profiler = *local_prof;
          m_profiler.get_mhz();
        });
#endif

        auto proc_res = [&](auto &result) {
          for (auto &res : result) {
            if (skip_recall) {
#ifdef LAT
              end[res->query_id] = std::chrono::high_resolution_clock::now();
#endif
              res->release_heavy_state_after_result();
              load_cnt.fetch_add(1);
              continue;
            }
#ifdef PROFILER
            // res->profiler.report();
            auto *local_prof = appr_alg.all_profiler.getLocal();
            Profiler &local_profiler = *local_prof;
            local_profiler += res->profiler;
#endif
            std::priority_queue<std::pair<dist_t, labeltype>> gt(
                answers[res->query_id]);
            unordered_set<labeltype> g;
            total.fetch_add(gt.size());
            // printf("<<>> q%llu gt size: %llu\n", res->query_id, gt.size());
#ifdef DEBUG
            std::vector<uint32_t> gt_distri;
#endif
            while (gt.size()) {
              g.insert(gt.top().second);
#ifdef DEBUG
              gt_distri.push_back(appr_alg.graph_index.scala_graph->get_machine_id(gt.top().second));
#endif
              gt.pop();
            }
#ifdef DEBUG
            printf("end process q%d\n", res->query_id);
            printf("q%u GT machine distribution: \n", res->query_id);
            uint32_t gt_m_cnt[MACHINE_NUM];
            memset(gt_m_cnt, 0, sizeof(gt_m_cnt));
            for(auto m: gt_distri){
              gt_m_cnt[m] ++;
            }
            for(auto m: gt_distri){
              printf("%u ", gt_m_cnt[m]);
            }
            printf("\n");
#endif
            while (res->result.size()) {
              if (g.find(res->result.top().second) != g.end()) {
                correct.fetch_add(1);
              } else {
              }
              res->result.pop();
            }
#ifdef LAT
            end[res->query_id] = std::chrono::high_resolution_clock::now();
#endif
#ifdef DEBUG
            printf("query %u Top Cand Distri\n", res->query_id);
            for (uint32_t m = 0; m < MACHINE_NUM; m++) {
              printf("%u ", res->filter_num[m]);
            }
            printf("\n");
            printf("query %u Final Cand Distri\n", res->query_id);
            for (uint32_t m = 0; m < MACHINE_NUM; m++) {
              printf("%u ", res->tmp_m_cnt[m]);
            }
            printf("\n");
#ifdef PROFILER
            printf("q %u report:\n", res->query_id);
            res->profiler.get_mhz();
            res->profiler.report();
#endif
#endif
            res->release_heavy_state_after_result();
            load_cnt.fetch_add(1);
            // printf("load_cnt ++: %u\n", load_cnt.load());
          }
          return;
        };

        appr_alg.scala_search_init(query_ptr, vecsize, qsize);

        StopW stopw = StopW();
        float exec_time;
#ifdef PROFILER
        on_each([&](uint64 tid, uint64 total) {
          Profiler &m_profiler = *appr_alg.all_profiler.getLocal();
          m_profiler.start("g_actual");
        });
#endif

        do_all_standby(
            iterate(all),
            [&](const uint32 &i) {
        // Stage 1: get query from batch list.
#ifdef DEBUG
              auto tid = ThreadPool::getTID();
              printf("thread %d start process q%d\n", tid, load_start + i);
#endif


#ifdef LAT
              start[load_start + i] = std::chrono::high_resolution_clock::now();
#endif
              // printf(" start process q%d\n", load_start + i);
              auto result = appr_alg.scala_search_sub(load_start + i);
              proc_res(result);
            },
            [&]() {
              // Stage 2: Wait for all my query finished.
              int tid = ThreadPool::getTID();
#ifdef DEBUG
              printf("Thread %d waiting for last query.\n", tid);
#endif
#ifdef PROFILER
              Profiler &m_profiler = *appr_alg.all_profiler.getLocal();
              m_profiler.start("g_overall");
#endif
              while (load_cnt.load() < query_load) {
                auto result = appr_alg.scala_search_sub_wait_finish();
                proc_res(result);
              }
#ifdef PROFILER
              m_profiler.end("g_overall");
#endif
              if (tid == 0) {
                double search_only_us = stopw.getElapsedTimeMicro();
                if (skip_recall) {
                  double local_qps = search_only_us > 0.0
                                         ? 1000000.0 * query_load / search_only_us
                                         : 0.0;
                  printf("SEARCH_ONLY\t%zu\t%d\t%zu\t%.3f\t%.3f\t%u\t%u\n",
                         ef, appr_alg.rdma_param.machine_id, query_load,
                         search_only_us, local_qps, runtime_query_group_size,
                         skip_recall ? 1 : 0);
                  fflush(stdout);
                }
                double cumu_lat_us = 0.0;
                appr_alg.sendTerm(correct.load(), total.load(), query_load, cumu_lat_us);
              }
#ifdef DEBUG
              printf("Thread %d goto standby.\n", tid);
#endif
              // Stage 3: Stand by for all query end
              appr_alg.scala_search_sub_standby(qsize);

              if (tid == 0) {
                exec_time = stopw.getElapsedTimeMicro();
#ifdef COMM_PROFILE
                appr_alg.report_comm_info(exec_time);
#endif
                if (appr_alg.rdma_param.machine_id == 0) {
                  if (anns_param.evaluation_save_path.empty()) {
                    appr_alg.printQueryBatchInfo(exec_time);
                  } else {
                    appr_alg.printQueryBatchInfoSave(
                        exec_time, appr_alg.get_evaluation_save_file());
                  }
                }
              }
            },
            steal(), loopname("Reset"));

#ifdef LAT
      double all_lat = 0.0;
      for(uint32_t q = load_start; q < load_start + query_load; q++){
        all_lat += duration_cast<std::chrono::microseconds>(end[q] - start[q]).count();
      }
      std::cout << "Avg lat: " << all_lat / query_load << "us\n";
#endif


#ifdef PROFILER
        on_each([&](uint64 tid, uint64 total) {
          Profiler &m_profiler = *appr_alg.all_profiler.getLocal();
          m_profiler.end("g_actual");
        });
#endif
        if(appr_alg.rdma_param.machine_id != 0) {
          float recall = total.load() ? 1.0f * correct.load() / total.load() : 0.0f;
          float time_us_per_query = query_load ? exec_time / query_load : 0.0f;
          float qps = exec_time > 0 ? 1000000.0 * query_load / exec_time : 0.0f;
          cout << ef << "\t" << recall << "\t" << time_us_per_query << " us\t"
              << qps << " /s\n";
        }
#ifdef PROFILER
#ifdef DEBUG
        if (appr_alg.rdma_param.machine_id != 0) {
          for (uint32_t q = 0; q < 32; q++) {
            printf("q %u report:\n", q);
            appr_alg.global_query[q]->profiler.get_mhz();
            appr_alg.global_query[q]->profiler.report();
          }
        }
#endif
        Profiler over_all_pro;
        over_all_pro.get_mhz();

        for (uint32_t t = 0; t < getActiveThreads(); ++t) {
          auto *local_prof = appr_alg.all_profiler.getRemote(t);
          Profiler &local_profiler = *local_prof;
          // printf("Profiler report of thread %lu\n", t);
          // local_profiler.report_sum();
          over_all_pro += local_profiler;
        }
        over_all_pro.report_sum();
#endif
#ifdef PROF_DEGREE
        printf(
            "local cnt: %llu deg: %llu\n", appr_alg.local_cnt,
            appr_alg.local_deg);
        printf(
            "remote cnt: %llu deg: %llu\n", appr_alg.remote_cnt,
            appr_alg.remote_deg);
        uint64_t sum_cnt = appr_alg.local_cnt + appr_alg.remote_cnt;
        uint64_t sum_deg = appr_alg.local_deg + appr_alg.remote_deg;
        float l_cnt_ratio = (float)appr_alg.local_cnt * 100.0 / sum_cnt;
        float r_cnt_ratio = (float)appr_alg.remote_cnt * 100.0 / sum_cnt;
        float l_deg_ratio = (float)appr_alg.local_deg * 100.0 / sum_deg;
        float r_deg_ratio = (float)appr_alg.remote_deg * 100.0 / sum_deg;
        printf("local cnt: %.2f%% deg: %.2f%%\n", l_cnt_ratio, l_deg_ratio);
        printf("remote cnt: %.2f%% deg: %.2f%%\n", r_cnt_ratio, r_deg_ratio);

        // deg distribution.
        uint64_t l_sum_dis{0}, r_sum_dis{0};
        for (uint32_t d = 0; d < MAX_GRAPH_DEG; d++) {
          l_sum_dis += appr_alg.local_distri[d];
          r_sum_dis += appr_alg.remote_distri[d];
        }
        printf("deg\t lcnt\t lrat\t rcnt\t rrat\n");
        for (uint32_t d = 0; d < MAX_GRAPH_DEG; d++) {
          printf(
              "%u\t %llu\t %.2f%%\t %llu\t %.2f%%\n", d,
              appr_alg.local_distri[d],
              appr_alg.local_distri[d] * 100.0 / l_sum_dis,
              appr_alg.remote_distri[d],
              appr_alg.remote_distri[d] * 100.0 / r_sum_dis);
        }
#endif

        for (uint32_t q = 0; q < qsize; q++) {
          delete appr_alg.global_query[q];
          appr_alg.global_query[q] = nullptr;
        }

      } break;

      default:
        break;
    }
#ifdef PROFILING
    appr_alg.reportAccessNum(qsize, time_us_per_query);
#endif
  }
}
