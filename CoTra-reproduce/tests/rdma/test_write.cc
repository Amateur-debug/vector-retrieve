#include "anns/exec_query.h"
#include "rdma/profiler.h"

/*
 [set the machine num to 2]
 follower: ./tests/rdma/test_write -m 1 -t 1
 leader: ./tests/rdma/test_write -m 0 -t 1
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
  int* random_list = (int*)malloc(1000 * 4);
  srand48(41);
  // generate random read list.
  for (int i = 0; i < 1000; i++) {
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
    for (;;) {
      rdma_comm.poll_recv();
      while (rbuf->migrate_queue.size() > 0) {
        int* rbuf_ptr = (int*)rbuf->migrate_queue.front();
        rbuf->migrate_queue.pop_front();

        if (rbuf_ptr[1] == 1) {  // no sg
          int target_mid = 0;    // send request
          uint32_t buf_id = rdma_comm.get_write_send_buf(target_mid);
          int* sbuf_ptr = (int*)sbuf->buffer[target_mid][buf_id];
          sbuf_ptr[0] = 990927;
          // send request.
          rdma_comm.post_write_send(target_mid, buf_id, rbuf_ptr[0], QUERY);
        } else {  // sg

          int target_mid = 0;  // send request

          // printf("request sum size=%u\n", rbuf_ptr[0]);
          std::vector<char*> sg_ptr;
          std::vector<uint32_t> sg_size;
          for (int s = 0; s < rbuf_ptr[0] / 256; s++) {
            sg_ptr.emplace_back(vec_partition + 128 * random_list[s]);
            sg_size.emplace_back(256);
          }
          uint32_t buf_id = rdma_comm.get_write_send_buf(target_mid);
          // send request.
          // printf("send sg num %u..\n", sg_ptr.size());
          rdma_comm.post_sg_write_send(
              target_mid, buf_id, sg_ptr, sg_size, QUERY);

          // rdma_comm.post_write_send(target_mid, buf_id, rbuf_ptr[0], QUERY);
        }

        free(rbuf_ptr);
      }
    }

    printf("Write verify success..\n");
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
    int repeat = 100;

    auto* sbuf = rdma_comm.send_write_buf.getLocal();
    auto* rbuf = rdma_comm.recv_write_buf.getLocal();

    for (int byte_sz = 64; byte_sz <= MAX_QUERYBUFFER_SIZE; byte_sz <<= 1) {
      std::string name = "write-" + std::to_string(byte_sz) + "-bytes";
      profiler.start(name);

      for (int i = 0; i < repeat; i++) {
        uint32_t buf_id = rdma_comm.get_write_send_buf(target_mid);
        int* sbuf_ptr = (int*)sbuf->buffer[target_mid][buf_id];
        sbuf_ptr[0] = byte_sz;
        sbuf_ptr[1] = 1;  // no sg
        // send request.
        rdma_comm.post_write_send(target_mid, buf_id, 256, QUERY);
        for (;;) {
          rdma_comm.poll_recv();
          if (rbuf->migrate_queue.size() > 0) {
            int* rbuf_ptr = (int*)rbuf->migrate_queue.front();
            rbuf->migrate_queue.pop_front();
            if (rbuf_ptr[0] != 990927) {
              printf("Verify failed ..\n");
            }
            break;
          }
        }
      }
      profiler.end(name);
      profiler.count(name, repeat - 1);
    }

    printf("start recv sg res\n");

    // TODO: for now, the sg_num is limited up to 16.
    for (int byte_sz = 256; byte_sz <= MAX_QUERYBUFFER_SIZE >> 1;
         byte_sz <<= 1) {
      std::string name = "sgw-" + std::to_string(byte_sz) + "-bytes";
      profiler.start(name);

      for (int i = 0; i < repeat; i++) {
        uint32_t buf_id = rdma_comm.get_write_send_buf(target_mid);
        int* sbuf_ptr = (int*)sbuf->buffer[target_mid][buf_id];
        sbuf_ptr[0] = byte_sz;
        sbuf_ptr[1] = 2;  // sg
        // send request.
        rdma_comm.post_write_send(target_mid, buf_id, 256, QUERY);
        for (;;) {
          rdma_comm.poll_recv();
          if (rbuf->migrate_queue.size() > 0) {
            int* rbuf_ptr = (int*)rbuf->migrate_queue.front();
            rbuf->migrate_queue.pop_front();
            break;
          }
        }
      }
      profiler.end(name);
      profiler.count(name, repeat - 1);
    }

    profiler.report();

    while (1) {
      rdma_comm.poll_send();
    }
  }

  return 0;
}
/*
>> item            num       total(usec)     avg(usec)
>> 1000-compute    1000      237.202         0.237
// scatter-gather round-trip
>> sgw-256-bytes   100       684.718         6.847
>> sgw-512-bytes   100       663.657         6.637
>> sgw-1024-bytes  100       753.865         7.539
>> sgw-2048-bytes  100       863.083         8.631
>> sgw-4096-bytes  100       1126.519        11.265

// direct round-trip
>> write-64-bytes  100       736.028         7.360
>> write-128-bytes 100       754.952         7.550
>> write-256-bytes 100       755.778         7.558
>> write-512-bytes 100       789.650         7.897
>> write-1024-bytes100       879.390         8.794
>> write-2048-bytes100       983.629         9.836
>> write-4096-bytes100       1299.815        12.998
>> write-8192-bytes100       1824.208        18.242
*/
