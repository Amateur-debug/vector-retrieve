#include <malloc.h>
#include <omp.h>

#include <cstring>
#include <iostream>

#include "anns/exec_query.h"
#include "rdma/profiler.h"

// ./tests/rdma/pool

int main() {
  SharedMem coromem;

  //   setActiveThreads(32);
  auto &tp = getThreadPool();
  tp.poolPause();

  omp_set_num_threads(32);

#pragma omp parallel for schedule(dynamic, 2048)
  for (uint32_t i = 0; i < 1000000; i++) {
    for (;;);
  }

  tp.poolContiue();

  on_each([&](uint64 tid, uint64 total) { for (;;); });

  return 0;
}