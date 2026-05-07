#pragma once

#include <math.h>

#include <fstream>
#include <iostream>
#include <filesystem>

#include "anns/search_interface.h"
#include "anns_config.h"
#include "coromem/include/galois/loops.h"
#include "coromem/include/share_mem.h"
#include "coromem/include/utils.h"
#include "rdma/ip_config.h"
#include "rdma/rdma_param.h"
#include "index/index_param.h"

using namespace hnswlib;
using namespace std;

class StopW {
  std::chrono::steady_clock::time_point time_begin;

 public:
  StopW() { time_begin = std::chrono::steady_clock::now(); }

  float getElapsedTimeMicro() {
    std::chrono::steady_clock::time_point time_end =
        std::chrono::steady_clock::now();
    return (std::chrono::duration_cast<std::chrono::microseconds>(
                time_end - time_begin)
                .count());
  }

  void reset() { time_begin = std::chrono::steady_clock::now(); }
};

/*
 * Author:  David Robert Nadeau
 * Site:    http://NadeauSoftware.com/
 * License: Creative Commons Attribution 3.0 Unported License
 *          http://creativecommons.org/licenses/by/3.0/deed.en_US
 */

#if defined(_WIN32)
#include <psapi.h>
#include <windows.h>

#elif defined(__unix__) || defined(__unix) || defined(unix) || \
    (defined(__APPLE__) && defined(__MACH__))

#include <sys/resource.h>
#include <unistd.h>

#if defined(__APPLE__) && defined(__MACH__)
#include <mach/mach.h>

#elif (defined(_AIX) || defined(__TOS__AIX__)) || \
    (defined(__sun__) || defined(__sun) ||        \
     defined(sun) && (defined(__SVR4) || defined(__svr4__)))
#include <fcntl.h>
#include <procfs.h>

#elif defined(__linux__) || defined(__linux) || defined(linux) || \
    defined(__gnu_linux__)

#endif

#else
#error "Cannot define getPeakRSS( ) or getCurrentRSS( ) for an unknown OS."
#endif

/**
 * Returns the peak (maximum so far) resident set size (physical
 * memory use) measured in bytes, or zero if the value cannot be
 * determined on this OS.
 */
static size_t getPeakRSS() {
#if defined(_WIN32)
  /* Windows -------------------------------------------------- */
  PROCESS_MEMORY_COUNTERS info;
  GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
  return (size_t)info.PeakWorkingSetSize;

#elif (defined(_AIX) || defined(__TOS__AIX__)) || \
    (defined(__sun__) || defined(__sun) ||        \
     defined(sun) && (defined(__SVR4) || defined(__svr4__)))
  /* AIX and Solaris ------------------------------------------ */
  struct psinfo psinfo;
  int fd = -1;
  if ((fd = open("/proc/self/psinfo", O_RDONLY)) == -1)
    return (size_t)0L; /* Can't open? */
  if (read(fd, &psinfo, sizeof(psinfo)) != sizeof(psinfo)) {
    close(fd);
    return (size_t)0L; /* Can't read? */
  }
  close(fd);
  return (size_t)(psinfo.pr_rssize * 1024L);

#elif defined(__unix__) || defined(__unix) || defined(unix) || \
    (defined(__APPLE__) && defined(__MACH__))
  /* BSD, Linux, and OSX -------------------------------------- */
  struct rusage rusage;
  getrusage(RUSAGE_SELF, &rusage);
#if defined(__APPLE__) && defined(__MACH__)
  return (size_t)rusage.ru_maxrss;
#else
  return (size_t)(rusage.ru_maxrss * 1024L);
#endif

#else
  /* Unknown OS ----------------------------------------------- */
  return (size_t)0L; /* Unsupported. */
#endif
}

/**
 * Returns the current resident set size (physical memory use) measured
 * in bytes, or zero if the value cannot be determined on this OS.
 */
static size_t getCurrentRSS() {
#if defined(_WIN32)
  /* Windows -------------------------------------------------- */
  PROCESS_MEMORY_COUNTERS info;
  GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
  return (size_t)info.WorkingSetSize;

#elif defined(__APPLE__) && defined(__MACH__)
  /* OSX ------------------------------------------------------ */
  struct mach_task_basic_info info;
  mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
  if (task_info(
          mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info,
          &infoCount) != KERN_SUCCESS)
    return (size_t)0L; /* Can't access? */
  return (size_t)info.resident_size;

#elif defined(__linux__) || defined(__linux) || defined(linux) || \
    defined(__gnu_linux__)
  /* Linux ---------------------------------------------------- */
  long rss = 0L;
  FILE *fp = NULL;
  if ((fp = fopen("/proc/self/statm", "r")) == NULL)
    return (size_t)0L; /* Can't open? */
  if (fscanf(fp, "%*s%ld", &rss) != 1) {
    fclose(fp);
    return (size_t)0L; /* Can't read? */
  }
  fclose(fp);
  return (size_t)rss * (size_t)sysconf(_SC_PAGESIZE);

#else
  /* AIX, BSD, Solaris, and Unknown OS ------------------------ */
  return (size_t)0L; /* Unsupported. */
#endif
}


template<typename dist_t>
static void get_gt(
    uint32_t *gt_ptr, size_t qsize, 
    vector<std::priority_queue<std::pair<dist_t, labeltype>>> &answers, size_t res_knn,
    size_t gt_knn) {
  (vector<std::priority_queue<std::pair<dist_t, labeltype>>>(qsize)).swap(answers);
  cout <<"qsize: "<< qsize << " gt_knn: " << gt_knn << " res_knn: " << res_knn<< "\n";
  for (int i = 0; i < qsize; i++) {
    for (int j = 0; j < res_knn; j++) {
      answers[i].emplace(0.0f, gt_ptr[gt_knn * i + j]);
    }
  }
}

/**
 * Search parameter.
 *
 *
 *
 */
class AnnsParameter {
 public:
  commandLine cmd;

  uint32_t subset_size_milllions;
  uint32_t efConstruction;
  uint32_t M;
  size_t vecdim;
  size_t qsize;
  size_t vecsize; // one vector size
  size_t vecnum; // all vector number (overall machines)
  uint32_t gt_knn;      // Ground truth KNN number.
  uint32_t res_knn;     // Actually exec knn, Set recall@k = 1,10,100
  L2Space *l2_space;
  L2SpaceI *l2_space_i;
  L2SpaceII *l2_space_ii;

  InnerProductSpace *ip_space;
  int machine_id;   // Machine id
  uint32_t thd_num;      // Yhread number

  std::string data_path;          // Data path
  std::string query_path;         // Query path

  std::string index_path;          // index path
  std::string kmeans_path;         // kmeans path
  std::string map_path;            // map path

  std::string gt_path;         // gt path
  std::string app_type_str;
  std::string config_file;
  std::string data_type_str;
  std::string dist_fn;

  std::string evaluation_save_path;  // evaluation save path

  // char *massb;
  // unsigned char *mass;
  char *query_ptr;

  AppType app_type;

  vector<std::priority_queue<std::pair<int, labeltype>>> int_answers;
  vector<std::priority_queue<std::pair<float, labeltype>>> float_answers;

  AnnsParameter() {}

  AnnsParameter(int argc, char **argv, IndexParameter &index_param);

  ~AnnsParameter() {}

  template<typename dist_t>
  void load_query_gt(vector<std::priority_queue<std::pair<dist_t, labeltype>>> &answers) {
    query_ptr = (char *)malloc(qsize * vecsize);

    printf("Loading GT from: %s\n", gt_path.c_str());
    if(!std::filesystem::exists(gt_path)){
      std::cerr<<"Error: GT file does not exist.\n";
      abort();
    }
    ifstream inputGT(gt_path.c_str(), ios::binary);
    
    uint32_t num_queries;
    inputGT.read((char *)(&num_queries), sizeof(uint32_t));
    inputGT.read((char *)(&gt_knn), sizeof(uint32_t));
    printf("Query number: %d K-NN: %d\n", num_queries, gt_knn);
    
    uint32_t *gt_ptr = (uint32_t *)malloc(qsize * gt_knn * sizeof(uint32_t));
    for (size_t i = 0; i < qsize; i++) {
      inputGT.read((char *)(gt_ptr + gt_knn * i), gt_knn * sizeof(uint32_t));
    }
    inputGT.close();

    printf("Loading queries from: %s\n", query_path.c_str());
    ifstream inputQ(query_path.c_str(), ios::binary);
    if(!std::filesystem::exists(query_path)){
      std::cerr<<"Error: Query file does not exist.\n";
      abort();
    }

    uint32_t num_q, dim;
    inputQ.read((char *)(&num_q), sizeof(uint32_t));
    inputQ.read((char *)(&dim), sizeof(uint32_t));
    printf("Query number: %d Dim: %d\n", num_q, dim);

    for (size_t i = 0; i < qsize; i++) {
      inputQ.read(query_ptr + i * vecsize, vecsize);
    }
    inputQ.close();
    cout << "Parsing gt:\n";
    get_gt<dist_t>(gt_ptr, qsize, answers, res_knn, gt_knn);
    cout << "Loaded gt\n";
    free(gt_ptr);
  }

  void load_query_only() {
    query_ptr = (char *)malloc(qsize * vecsize);

    printf("COTRA_SKIP_RECALL=1: skipping GT load and parsing.\n");
    printf("Loading queries from: %s\n", query_path.c_str());
    if(!std::filesystem::exists(query_path)){
      std::cerr<<"Error: Query file does not exist.\n";
      abort();
    }
    ifstream inputQ(query_path.c_str(), ios::binary);

    uint32_t num_q, dim;
    inputQ.read((char *)(&num_q), sizeof(uint32_t));
    inputQ.read((char *)(&dim), sizeof(uint32_t));
    printf("Query number: %d Dim: %d\n", num_q, dim);

    for (size_t i = 0; i < qsize; i++) {
      inputQ.read(query_ptr + i * vecsize, vecsize);
    }
    inputQ.close();
    printf("Loaded queries without GT.\n");
  }
};
