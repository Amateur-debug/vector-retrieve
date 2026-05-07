#pragma once

#include <netdb.h>
#include <netinet/in.h>
#include <omp.h>
#include <stdlib.h>
#include <sys/socket.h>

#include "anns/vec_buffer.h"
#include "rdma_param.h"

struct MessageContext {
  int lid;
  int out_reads;
  int qpn;
  int psn;
  unsigned read_rkey, write_rkey;
  unsigned long long read_vaddr, write_vaddr;
  union ibv_gid gid;
  unsigned srqn;
  int gid_index;
};

class RdmaContext {
 public:
  struct ibv_context *context;
#ifdef HAVE_AES_XTS
  struct mlx5dv_mkey **mkey;
  struct mlx5dv_dek **dek;
#endif
  struct ibv_comp_channel *recv_channel;
  struct ibv_comp_channel *send_channel;
  struct ibv_pd *pd;

  int machine_num;
  int machine_id;

  uint64_t read_size, read_cnt;  // bytes, number
  uint64_t write_size, write_cnt;

  MessageContext local_mr_msg[MACHINE_NUM];
  MessageContext remote_mr_msg[MACHINE_NUM];

  // Store the vector data for read.
  void *for_read_ptr[MACHINE_NUM];
  struct ibv_mr *for_read_mr[MACHINE_NUM];

  // Store the read vector data.
  void *read_to_ptr[MAX_CACHE_VEC];
  struct ibv_mr *read_to_mr;

  // Store the send write msg.
  void *send_write_ptr[MACHINE_NUM][MAX_WRITE_NUM];
  struct ibv_mr *send_write_mr[MACHINE_NUM];

  // Store the recv write msg.
  void *recv_write_ptr[MACHINE_NUM][MAX_WRITE_NUM];
  struct ibv_mr *recv_write_mr[MACHINE_NUM];

  // Store the send data.
  void *send_ptr[MACHINE_NUM][MAX_RECV_NUM];
  struct ibv_mr *send_mr[MACHINE_NUM];

  // Store the recv data.
  void *recv_ptr[MACHINE_NUM][MAX_RECV_NUM];
  struct ibv_mr *recv_mr[MACHINE_NUM];

  struct ibv_cq *send_cq;
  struct ibv_wc s_wc[MAX_WC_NUM];
  struct ibv_cq *recv_cq;
  struct ibv_wc r_wc[MAX_WC_NUM];

  struct ibv_qp *qp[MACHINE_NUM];
#ifdef HAVE_IBV_WR_API
  struct ibv_qp_ex **qpx;
#ifdef HAVE_MLX5DV
  struct mlx5dv_qp_ex **dv_qp;
#endif
  int (*new_post_send_work_request_func_pointer)(
      int index, RdmaParameter *param);
#endif
  struct ibv_srq *srq;

  // read operation
  struct ibv_sge sge_list[MAX_CACHE_VEC];
  struct ibv_send_wr wr[MACHINE_NUM][MAX_READ_NUM];  // only for read operation.
  size_t wr_cnt[MACHINE_NUM];

  size_t post_num, collect_num;

  // write operation (no need for recv sg list)
  // struct ibv_sge sg_list[MAX_RECV_NUM];  // TODO: for write test only.
  struct ibv_sge write_sge_list[MACHINE_NUM][MAX_WRITE_NUM][MAX_SGE_NUM];
  struct ibv_send_wr wwr[MACHINE_NUM][MAX_WRITE_NUM];
  struct ibv_recv_wr rwwr[MACHINE_NUM][MAX_WRITE_NUM];

  // send/recv operation
  struct ibv_sge send_sge_list[MACHINE_NUM][MAX_RECV_NUM];
  struct ibv_sge recv_sge_list[MACHINE_NUM][MAX_RECV_NUM];
  struct ibv_send_wr swr[MACHINE_NUM][MAX_RECV_NUM];  // only for send op.
  struct ibv_recv_wr rwr[MACHINE_NUM][MAX_RECV_NUM];  // for recv op.

  // uint64_t size;
  int cache_line_size;

  void clear_size_cnt() {
    read_size = read_cnt = 0;
    write_size = write_cnt = 0;
  }

  enum ibv_mtu set_mtu(uint8_t ib_port, int user_mtu);
  int check_mtu(RdmaParameter *param);
  void dealloc_ctx(RdmaParameter *param);
  int alloc_ctx(RdmaParameter *param);

  int create_ctx_mr(
      RdmaParameter *param, RdmaContext *main_ctx, char buf_ptr[][MAX_VEC_LEN],
      size_t buf_length, SendWriteBuffer *local_send_write,
      RecvWriteBuffer *local_recv_write);
  int create_cqs(RdmaContext *main_ctx, RdmaParameter *param);

  struct ibv_qp *qp_create(
      RdmaContext *main_ctx, RdmaParameter *param, int qp_index);

  int modify_qp_to_init(RdmaParameter *param, int qp_index);

  void create_main_partition_mr(
      char *vec_part_ptr, size_t part_length, size_t cacheline);

  int ctx_init(
      RdmaParameter *param, RdmaContext *main_ctx, char buf_ptr[][MAX_VEC_LEN],
      size_t buf_len, SendWriteBuffer *local_send_write,
      RecvWriteBuffer *local_recv_write
      // char send_write_ptr[][MAX_WRITE_NUM][MAX_QUERYBUFFER_SIZE],
      // size_t send_write_length,
      // char recv_write_ptr[][MAX_WRITE_NUM][MAX_QUERYBUFFER_SIZE],
      // size_t recv_write_length,
      // char send_ptr[][MAX_RECV_NUM][MAX_QUERYBUFFER_SIZE], size_t
      // send_length, char recv_ptr[][MAX_RECV_NUM][MAX_QUERYBUFFER_SIZE],
      // size_t recv_len
  );

  int modify_qp_to_rtr(
      struct ibv_qp *qp, struct ibv_qp_attr *attr, RdmaParameter *param,
      struct MessageContext *dest, struct MessageContext *my_dest,
      int qp_index);
  int modify_qp_to_rts(
      struct ibv_qp *qp, struct ibv_qp_attr *attr, RdmaParameter *param,
      struct MessageContext *dest, struct MessageContext *my_dest);

  int connect(RdmaParameter *param);
  uint16_t get_local_lid(int port);

  /**
   * Poll complete queues.
   */
  int poll_SEND();
  int poll_RECV();

  /**
   * Read operation.
   */
  void read_init(RdmaParameter *param);
  // int run_sync_read(
  //     int machine_id, size_t offset, uint32_t size, QueryBuffer *buf =
  //     nullptr);
  int run_post_read(int machine_id);
  // int run_post_read(std::vector<BufferCache> &vectors, size_t size);
  // int run_poll_read(VectorBuffer *vbuf, QueryBuffer *sbuf);

  /**
   * Write operation.
   */
  void write_init(RdmaParameter *param);
  // int run_sync_write(int machine_id, size_t offset, uint32_t size);
  int run_post_write_send(
      int machine_id, uint32_t buf_id, size_t offset, uint32_t size,
      uint32_t control);
  int run_post_write_recv(int machine_id, size_t buff_id);
  int run_post_sg_write_send(
      int machine_id, uint32_t buf_id, size_t offset,
      std::vector<char *> &sg_ptr, std::vector<uint32_t> &sg_size,
      uint32_t control);

  /**
   * Send/Recv operation.
   */
  void send_init(RdmaParameter *param);
  int run_sync_send(int machine_id, uint32_t size);
  int run_sync_recv(uint32_t recv_num);
  int run_post_send(int machine_id, size_t buff_id, uint32_t size);
  int run_post_recv(int machine_id, size_t buff_id, uint32_t size);
};

struct ibv_device *ctx_find_dev(char **ib_devname);
int check_add_port(
    char **service, int port, const char *servername, struct addrinfo *hints,
    struct addrinfo **res);
