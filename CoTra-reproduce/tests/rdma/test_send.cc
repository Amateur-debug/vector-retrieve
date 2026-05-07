#include "anns/exec_query.h"

// w32: ./rdma-hnsw/tests/test_send -m 1 -t 8
// w31: ./rdma-hnsw/tests/test_send -m 0 -t 8
int main(int argc, char** argv) {
  // SharedMem coromem;
  // coromem.tp.hw_info.show();
  // commandLine cmd(argc, argv, "usage: -m <machine_id> -t <thd_num> ");

  // auto machine_id = cmd.getOptionIntValue("-m", 0);
  // auto thd_num = cmd.getOptionIntValue("-t", 1);
  // auto numThreads = setActiveThreads(thd_num);
  // printf("Set thread num = %d\n", numThreads);

  // printf("Run RDMA-ANNS machine number: %d\n", MACHINE_NUM);
  // RdmaParameter rdma_param;
  // int ret_parser = rdma_param.parser(cmd);
  // if (ret_parser) {
  //   printf("Failed to parser parameter.\n");
  //   exit(0);
  // }

  // RdmaCommunication rdma_comm;
  // PerThreadStorage<VectorBuffer> vec_buf;
  // PerThreadStorage<RecvBuffer> recv_buf;
  // PerThreadStorage<SendBuffer> send_buf;

  // char* vec_partition = (char*)malloc(10000);
  // size_t vec_partition_size = 10000;

  // rdma_comm.com_init(vec_partition, vec_partition_size, rdma_param);

  // if (rdma_param.machine == MEMBER) {
  //   int* tmp = (int*)(recv_buf.getLocal()->buffer);
  //   for (int i = 0; i < 10; i++) {
  //     tmp[i] = 1;
  //   }
  //   printf("init\n");
  //   for (int i = 0; i < 10; i++) {
  //     printf("%d ", tmp[i]);
  //   }
  //   printf("\ninit end\n");

  //   rdma_comm.com_send_init();
  //   machine_id = 0;
  //   rdma_comm.sync_recv();

  //   printf("after transfer\n");
  //   bool verify_succuss = true;
  //   tmp = (int*)(recv_buf.getLocal()->buffer);
  //   for (int i = 0; i < 10; i++) {
  //     printf("%d ", tmp[i]);
  //     if (tmp[i] != i * i) {
  //       verify_succuss = false;
  //     }
  //   }
  //   printf("\n");
  //   if (!verify_succuss) {
  //     printf("Send verification failed\n");
  //   } else {
  //     printf("Send verification success\n");
  //   }
  // } else {
  //   int* tmp = (int*)(vec_buf.getLocal()->vector + 268);
  //   for (int i = 0; i < 10; i++) {
  //     tmp[i] = i * i;
  //   }
  //   machine_id = 1;

  //   printf("send read\n");
  //   rdma_comm.com_send_init();

  //   rdma_comm.sync_send(machine_id, 0, 268);
  // }

  return 0;
}
