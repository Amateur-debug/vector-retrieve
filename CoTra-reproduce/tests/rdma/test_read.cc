#include "anns/exec_query.h"
#include "rdma/profiler.h"

// follower: ./tests/rdma/test_read -m 1 -t 1
// leader: ./tests/rdma/test_read -m 0 -t 1
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

  size_t vec_partition_size = MAX_VEC_LEN * vec_num;
  char* vec_partition = (char*)malloc(vec_partition_size);

  const int compute_num = 1000;
  int* random_list = (int*)malloc(compute_num * 4);
  srand48(41);
  // generate random read list.
  for (int i = 0; i < compute_num; i++) {
    random_list[i] = rand() % vec_num;
  }

  // generate a random vector.
  int* compute_vector = (int*)malloc(128);
  for (int i = 0; i < 32; i++) {
    compute_vector[i] = rand();
  }

  // generate compute data.
  int* data_ptr = (int*)vec_partition;
  for (int i = 0; i < (vec_partition_size >> 2); i++) {
    data_ptr[i] = rand() % 1000000;
  }

  rdma_comm.com_init(vec_partition, vec_partition_size, rdma_param);

  if (rdma_param.machine == MEMBER) {
    for (;;);
  } else {
    L2SpaceI l2space(128);
    DISTFUNC<int> fstdistfunc_;
    void* dist_func_param_{nullptr};
    fstdistfunc_ = l2space.get_dist_func();
    dist_func_param_ = l2space.get_dist_func_param();
    std::priority_queue<std::pair<int, int>> top_candidates;

    std::string cmp_name = std::to_string(compute_num) + "-compute";
    profiler.start(cmp_name);
    for (int i = 0; i < compute_num; i++) {
      int dist = fstdistfunc_(
          compute_vector, vec_partition + 128 * random_list[i],
          dist_func_param_);
      top_candidates.emplace(dist, i);
    }
    profiler.end(cmp_name);
    profiler.count(cmp_name, compute_num - 1);

    rdma_comm.com_read_init();
    machine_id = 1;
    int repeat = 100;
    profiler.start("read-268-B");
    for (int i = 0; i < repeat; i++) {
      int* read_data = (int*)rdma_comm.sync_read(machine_id, i, 268);
      assert(memcmp(read_data, vec_partition + i * 268) == 0);
    }
    profiler.end("read-268-B");
    profiler.count("read-268-B", repeat - 1);

    for (int byte_sz = 64; byte_sz <= 8192; byte_sz <<= 1) {
      std::string name = "read-" + std::to_string(byte_sz) + "-B";
      profiler.start(name);
      for (int i = 0; i < repeat; i++) {
        int* read_data = (int*)rdma_comm.sync_read(machine_id, i, byte_sz);
        assert(memcmp(read_data, vec_partition + i * byte_sz) == 0);
      }
      profiler.end(name);
      profiler.count(name, repeat - 1);
    }

    size_t size_data_per_element_ = 128;
    for (int byte_sz = 128; byte_sz <= 8192; byte_sz <<= 1) {
      std::string name = "async-" + std::to_string(byte_sz) + "-B";
      std::string postn = "async-" + std::to_string(byte_sz) + "-post";
      std::string polln = "async-" + std::to_string(byte_sz) + "-poll";
      profiler.start(name);
      for (int i = 0; i < repeat; i++) {
        profiler.start(postn);
        std::vector<BufferCache> remote_vec;
        for (uint32_t k = 0; k < byte_sz / size_data_per_element_; k++) {
          remote_vec.push_back(BufferCache{
              machine_id, random_list[k], random_list[k], -1, NULL});
        }
        rdma_comm.post_read(remote_vec, size_data_per_element_);
        profiler.end(postn);
        profiler.start(polln);
        rdma_comm.poll_read();
        profiler.end(polln);
        for (BufferCache& r : remote_vec) {
          // BufferCache& r = *bc;
          // uint32_t vector_id = r.vector_id;
          // char* vector_ptr = r.buffer_ptr;
          // size_t machine_id = r.machine_id;
          // size_t internal_id = r.internal_id;

          // int* read_data = (int*)(vector_ptr);
          // int* truedata =
          //     (int*)(vec_partition + vector_id * size_data_per_element_);

          // if (memcmp(read_data, truedata, size_data_per_element_)) {
          //   printf("Read operator verify failed.\n");
          //   printf(
          //       "internal_id %llu offset %llu size %llu\n", vector_id,
          //       internal_id * size_data_per_element_,
          //       size_data_per_element_);
          //   printf("read_data\n");
          //   for (int i = 0; i < 10; i++) {
          //     printf("%d ", read_data[i]);
          //   }
          //   printf("\n truedata\n");
          //   for (int i = 0; i < 10; i++) {
          //     printf("%d ", truedata[i]);
          //   }
          //   printf("\n");
          //   abort();
          // }
          rdma_comm.release_cache(r.buffer_id);
        }
      }
      profiler.end(name);
      profiler.count(name, repeat - 1);
    }
  }
  profiler.report();

  return 0;
}

/*
>> item            num       total(usec)     avg(usec)
>> 1000-compute    1000      152.256         0.152

>> async-128-B     100       300.689         3.007
>> async-128-poll  100       198.091         1.981
>> async-128-post  100       48.440          0.484
>> async-256-B     100       446.127         4.461
>> async-256-poll  100       337.995         3.380
>> async-256-post  100       56.570          0.566
>> async-512-B     100       554.939         5.549
>> async-512-poll  100       439.116         4.391
>> async-512-post  100       65.370          0.654
>> async-1024-B    100       827.183         8.272
>> async-1024-poll 100       667.952         6.680
>> async-1024-post 100       97.845          0.978
>> async-2048-B    100       1313.204        13.132
>> async-2048-poll 100       1117.955        11.180
>> async-2048-post 100       136.010         1.360
>> async-4096-B    100       2326.277        23.263
>> async-4096-poll 100       2027.120        20.271
>> async-4096-post 100       227.276         2.273
>> async-8192-B    100       4304.276        43.043
>> async-8192-poll 100       3837.437        38.374
>> async-8192-post 100       377.418         3.774

>> read-64-B       100       257.404         2.574
>> read-128-B      100       261.585         2.616
>> read-256-B      100       280.012         2.800
>> read-268-B      100       278.221         2.782
>> read-512-B      100       289.211         2.892
>> read-1024-B     100       317.766         3.178
>> read-2048-B     100       367.240         3.672
>> read-4096-B     100       474.730         4.747
>> read-8192-B     100       567.911         5.679
*/
