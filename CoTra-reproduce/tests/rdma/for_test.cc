#include <malloc.h>

#include <cstring>
#include <iostream>

#include "rdma/profiler.h"

// 13728427300

int main() {
  Profiler t;

  t.start("memalign");

  char* partition_data = (char*)memalign(64, 1000000000);

  t.end("memalign");
  t.start("malloc");

  char* partition_data2 = (char*)malloc(1000000000);
  t.end("malloc");
  t.start("memcpy");
  memcpy(partition_data, partition_data2, 1000000000);

  t.end("memcpy");
  printf("its ok\n");

  t.report();

  return 0;
}