#include "anns/exec_query.h"
#include "rdma/profiler.h"

/*
 [set the machine num to 2]
 follower: ./tests/rdma/test_bw -m 1 -t 1
 leader: ./tests/rdma/test_bw -m 0 -t 1
*/
int main(int argc, char** argv) {
  SharedMem coromem;
  coromem.tp.hw_info.show();
  commandLine cmd(argc, argv, "usage: -m <machine_id> -t <thd_num> ");

  Profiler profiler;
  profiler.get_mhz();

  auto machine_id = cmd.getOptionIntValue("-m", 0);
  auto thd_num = cmd.getOptionIntValue("-t", 1);
  auto numThreads = setActiveThreads(thd_num);
  printf("Set thread num = %d\n", numThreads);

  printf("Run RDMA-ANNS machine number: %d\n", MACHINE_NUM);
  RdmaParameter rdma_param;
  int ret_parser = rdma_param.parser(cmd);
  if (ret_parser) {
    printf("Failed to parser parameter.\n");
    exit(0);
  }

  RdmaCommunication rdma_comm;
  size_t vec_num = 1000000;

  size_t vec_partition_size = 128 * vec_num;
  char* vec_partition = (char*)malloc(vec_partition_size);

  const int compute_num = 1000;
  int* random_list = (int*)malloc(100000 * 4);
  srand48(41);
  // generate random read list.
  for (int i = 0; i < 100000; i++) {
    random_list[i] = rand() % vec_num;
  }

  // generate a random vector.
  int* compute_vector = (int*)malloc(128);
  for (int i = 0; i < 32; i++) {
    compute_vector[i] = rand();
  }

  // generate compute data.
  int* data_ptr = (int*)vec_partition;
  for (int i = 0; i < 32 * vec_num; i++) {
    data_ptr[i] = i % 10000;
  }

  rdma_comm.com_init(vec_partition, vec_partition_size, rdma_param);

  rdma_comm.com_write_init();

  if (rdma_param.machine == MEMBER) {
    auto* rbuf = rdma_comm.recv_write_buf.getLocal();
    auto* sbuf = rdma_comm.send_write_buf.getLocal();

    on_each([&](uint64 tid, uint64 total) {
      for (;;) {
        rdma_comm.poll_recv();
        rbuf->recv_task.clear();
      }
    });

    printf("Write recv over..\n");
  } else {
    L2SpaceI l2space(128);
    DISTFUNC<int> fstdistfunc_;
    void* dist_func_param_{nullptr};
    fstdistfunc_ = l2space.get_dist_func();
    dist_func_param_ = l2space.get_dist_func_param();
    std::priority_queue<std::pair<int, int>> top_candidates;

    profiler.start("1000-compute");
    for (int i = 0; i < compute_num; i++) {
      int dist = fstdistfunc_(
          compute_vector, vec_partition + 128 * random_list[i],
          dist_func_param_);
      top_candidates.emplace(dist, i);
    }
    profiler.end("1000-compute");
    profiler.count("1000-compute", 999);

    int target_mid = 1;
    int repeat = 100000;

    std::string name = "write-128-bytes";
    profiler.start(name);
    on_each([&](uint64 tid, uint64 total) {
      auto* sbuf = rdma_comm.send_write_buf.getLocal();
      auto* rbuf = rdma_comm.recv_write_buf.getLocal();

      uint32_t byte_sz = 128;
      for (int i = 0; i < repeat; i++) {
        uint32_t buf_id = rdma_comm.get_write_send_buf(target_mid);
        int* sbuf_ptr = (int*)sbuf->buffer[target_mid][buf_id];
        memcpy(sbuf_ptr, vec_partition + 128 * random_list[i], byte_sz);
        rdma_comm.post_write_send(target_mid, buf_id, byte_sz, TASK);
        rdma_comm.poll_send();
      }
    });
    profiler.end(name);
    profiler.count(name, repeat * getActiveThreads() - 1);

    profiler.report();
  }

  return 0;
}
/*
>> item            num       total(usec)     avg(usec)   throughput
>> 1000-compute    1000      159.422         0.159       6272663.731 /s
>> write-128-bytes 800000    201671.247      0.252       3966852.059 /s
*/
