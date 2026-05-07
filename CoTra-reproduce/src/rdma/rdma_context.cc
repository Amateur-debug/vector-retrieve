#include "rdma/rdma_context.h"

#include "anns/vec_buffer.h"
#include "coromem/include/backend.h"

struct ibv_device *ctx_find_dev(char **ib_devname) {
  int num_of_device;
  struct ibv_device **dev_list;
  struct ibv_device *ib_dev = NULL;

  dev_list = ibv_get_device_list(&num_of_device);

  // coverity[uninit_use]
  if (num_of_device <= 0) {
    fprintf(stderr, " Did not detect devices \n");
    fprintf(stderr, " If device exists, check if driver is up\n");
    return NULL;
  }

  if (!ib_devname) {
    fprintf(stderr, " Internal error, existing.\n");
    return NULL;
  }

  if (!*ib_devname) {
    ib_dev = dev_list[0];
    if (!ib_dev) {
      fprintf(stderr, "No IB devices found\n");
      exit(1);
    }
  } else {
    for (; (ib_dev = *dev_list); ++dev_list)
      if (!strcmp(ibv_get_device_name(ib_dev), *ib_devname)) break;
    if (!ib_dev) {
      fprintf(stderr, "IB device %s not found\n", *ib_devname);
      return NULL;
    }
  }

  ALLOCATE(*ib_devname, char, (strlen(ibv_get_device_name(ib_dev)) + 1));
  strcpy(*ib_devname, ibv_get_device_name(ib_dev));

  return ib_dev;
}

int check_add_port(
    char **service, int port, const char *servername, struct addrinfo *hints,
    struct addrinfo **res) {
  int number;
  if (asprintf(service, "%d", port) < 0) {
    return FAILURE;
  }
  number = getaddrinfo(servername, *service, hints, res);
  free(*service);
  if (number < 0) {
    fprintf(
        stderr, "%s for ai_family: %x service: %s port: %d\n",
        gai_strerror(number), hints->ai_family, servername, port);
    return FAILURE;
  }
  return SUCCESS;
}

enum ibv_mtu RdmaContext::set_mtu(uint8_t ib_port, int user_mtu) {
  struct ibv_port_attr port_attr;
  enum ibv_mtu curr_mtu;

  if (ibv_query_port(context, ib_port, &port_attr)) {
    fprintf(stderr, " Error when trying to query port\n");
    exit(1);
  }

  /* User did not ask for specific mtu. */
  if (user_mtu == 0) {
    enum ctx_device current_dev = ib_dev_name(context);
    curr_mtu = port_attr.active_mtu;
  }

  else {
    switch (user_mtu) {
      case 256:
        curr_mtu = IBV_MTU_256;
        break;
      case 512:
        curr_mtu = IBV_MTU_512;
        break;
      case 1024:
        curr_mtu = IBV_MTU_1024;
        break;
      case 2048:
        curr_mtu = IBV_MTU_2048;
        break;
      case 4096:
        curr_mtu = IBV_MTU_4096;
        break;
      default:
        fprintf(stderr, " Invalid MTU - %d \n", user_mtu);
        fprintf(stderr, " Please choose mtu from {256,512,1024,2048,4096}\n");
        // coverity[uninit_use_in_call]
        fprintf(
            stderr, " Will run with the port active mtu - %d\n",
            port_attr.active_mtu);
        curr_mtu = port_attr.active_mtu;
    }

    if (curr_mtu > port_attr.active_mtu) {
      fprintf(stdout, "Requested mtu is higher than active mtu \n");
      fprintf(stdout, "Changing to active mtu - %d\n", port_attr.active_mtu);
      curr_mtu = port_attr.active_mtu;
    }
  }
  return curr_mtu;
}

int RdmaContext::check_mtu(RdmaParameter *param) {
  // printf("check_mtu\n");
  int curr_mtu, rem_mtu;
  char cur[sizeof(int)];
  char rem[sizeof(int)];
  int size_of_cur;

  curr_mtu = (int)(set_mtu(param->ib_port, param->mtu));
  param->curr_mtu = (enum ibv_mtu)(curr_mtu);
  return SUCCESS;
}

void RdmaContext::dealloc_ctx(RdmaParameter *param) {
  if (qp != NULL) free(qp);

  if (sge_list != NULL) free(sge_list);
  if (wr != NULL) free(wr);
}

int RdmaContext::alloc_ctx(RdmaParameter *param) {
  memset(local_mr_msg, 0, sizeof(struct MessageContext) * MACHINE_NUM);
  memset(remote_mr_msg, 0, sizeof(struct MessageContext) * MACHINE_NUM);
  cache_line_size = param->cache_line_size;
  // INC(BUFF_SIZE(size, cycle_buffer), cache_line_size) * 2;
  return SUCCESS;
}

void RdmaContext::create_main_partition_mr(
    char *vec_part_ptr, size_t part_length, size_t cacheline) {
  int flags = IBV_ACCESS_LOCAL_WRITE;
  flags |= IBV_ACCESS_REMOTE_WRITE;
  flags |= IBV_ACCESS_REMOTE_READ;
  for_read_ptr[0] = vec_part_ptr;
  for_read_mr[0] =
      ibv_reg_mr(pd, for_read_ptr[0], INC(part_length, cacheline), flags);
  if (!for_read_mr[0]) {
    printf(
        "Failed to create main mr. length: %llu align length: %llu\n",
        part_length, INC(part_length, cacheline));
  } else {
    printf(
        "Create main mr success. length: %llu align length: %llu\n",
        part_length, INC(part_length, cacheline));
  }
}

/*
 Create memory region for recv data.
 */
int RdmaContext::create_ctx_mr(
    RdmaParameter *param, RdmaContext *main_ctx, char buf_ptr[][MAX_VEC_LEN],
    size_t buf_length, SendWriteBuffer *local_send_write,
    RecvWriteBuffer *local_recv_write) {
  int flags = IBV_ACCESS_LOCAL_WRITE;
  flags |= IBV_ACCESS_REMOTE_WRITE;
  flags |= IBV_ACCESS_REMOTE_READ;
  // flags |= IBV_ACCESS_REMOTE_ATOMIC;

  {
    // Use vec_partiton point to the vector patition.
    auto &tp = getThreadPool();
    // printf("thread %d try to alloc mr\n", tp.getTID());
    // for_read_ptr[0] = main_for_read_ptr[0];
    // for_read_mr[0] = main_for_read_mr[0];
    if (!for_read_mr[0]) {
      fprintf(stderr, "Thread %d Couldn't allocate MR\n", tp.getTID());
      return FAILURE;
    } else {
      // printf("Thread %d allocate MR success\n", tp.getTID());
    }
    for (int m = 1; m < MACHINE_NUM; m++) {
      for_read_ptr[m] = for_read_ptr[0];
      for_read_mr[m] = for_read_mr[0];
    }
  }

  // Register read mr
  for (size_t b = 0; b < MAX_CACHE_VEC; b++) {
    read_to_ptr[b] = buf_ptr[b];
  }
  read_to_mr = ibv_reg_mr(
      pd, read_to_ptr[0],
      INC(MAX_CACHE_VEC * MAX_VEC_LEN, param->cache_line_size), flags);
  if (!read_to_mr) {
    fprintf(stderr, "Couldn't allocate MR to partition MR.\n");
    return FAILURE;
  }
  memset(read_to_ptr[0], 0, MAX_CACHE_VEC * MAX_VEC_LEN);

  for (int m = 0; m < MACHINE_NUM; m++) {
    // for (size_t b = 0; b < MAX_READ_NUM; b++) {
    //   read_to_ptr[m * MAX_READ_NUM + b] = buf_ptr[m * MAX_READ_NUM + b];
    // }
    // read_to_mr = ibv_reg_mr(pd, read_to_ptr[0], buf_length, flags);
    // if (!read_to_mr) {
    //   fprintf(stderr, "Couldn't allocate MR to partition MR.\n");
    //   return FAILURE;
    // }
    // memset(read_to_ptr[0], 0, buf_length);

    // Register send_write mr
    for (size_t b = 0; b < MAX_WRITE_NUM; b++) {  //  buffer id
      send_write_ptr[m][b] = local_send_write->buffer[m][b];
    }
    send_write_mr[m] = ibv_reg_mr(
        pd, send_write_ptr[m][0],
        INC(local_send_write->buffer_len, param->cache_line_size), flags);
    if (!send_write_mr[m]) {
      fprintf(stderr, "Couldn't allocate MR to partition MR.\n");
      return FAILURE;
    }
    memset(send_write_ptr[m][0], 0, local_send_write->buffer_len);

    // Register recv_write mr
    for (size_t b = 0; b < MAX_WRITE_NUM; b++) {  //  buffer id
      recv_write_ptr[m][b] = local_recv_write->buffer[m][b];
    }
    recv_write_mr[m] = ibv_reg_mr(
        pd, recv_write_ptr[m][0],
        INC(local_recv_write->buffer_len, param->cache_line_size), flags);
    if (!recv_write_mr[m]) {
      fprintf(stderr, "Couldn't allocate MR to partition MR.\n");
      return FAILURE;
    }
    memset(recv_write_ptr[m][0], 0, local_recv_write->buffer_len);

    // Register send mr
    // for (size_t b = 0; b < MAX_RECV_NUM; b++) {  //  buffer id
    //   send_ptr[m][b] = send_ptr_[m][b];
    // }
    // send_mr[m] = ibv_reg_mr(
    //     pd, send_ptr[m][0], INC(send_length, param->cache_line_size), flags);
    // if (!send_mr[m]) {
    //   fprintf(stderr, "Couldn't allocate MR to partition MR.\n");
    //   return FAILURE;
    // }
    // memset(send_ptr[m][0], 0, send_length);

    // Register recv mr
    // for (size_t b = 0; b < MAX_RECV_NUM; b++) {  //  buffer id
    //   recv_ptr[m][b] = recv_ptr_[m][b];
    // }
    // recv_mr[m] = ibv_reg_mr(
    //     pd, recv_ptr[m][0], INC(recv_length, param->cache_line_size), flags);
    // if (!recv_mr[m]) {
    //   fprintf(stderr, "Couldn't allocate MR to partition MR.\n");
    //   return FAILURE;
    // }
    // memset(recv_ptr[m][0], 0, recv_length);
  }

  return SUCCESS;
}

int RdmaContext::create_cqs(RdmaContext *main_ctx, RdmaParameter *param) {
  // send_cq store all the finished send requests, max number can
  // achieve <max send num> * <machine num> + <max async read op num>
  send_cq = ibv_create_cq(
      context, (MAX_WRITE_NUM + MAX_READ_NUM) * MACHINE_NUM, NULL, send_channel,
      0);
  if (!send_cq) {
    fprintf(stderr, "Couldn't create CQ\n");
    return FAILURE;
  }

  // recv_cq store all the finished recv requests, max number can
  // achieve <max send num> * <machine num>
  recv_cq = ibv_create_cq(
      context, MAX_WRITE_NUM * MACHINE_NUM, NULL, recv_channel, 0);
  if (!recv_cq) {
    fprintf(stderr, "Couldn't create a receiver CQ\n");
    return FAILURE;
  }
  return SUCCESS;
}

struct ibv_qp *RdmaContext::qp_create(
    RdmaContext *main_ctx, RdmaParameter *param, int qp_index) {
  struct ibv_qp *qp = NULL;

  int is_dc_server_side = 0;
  struct ibv_qp_init_attr attr;
  memset(&attr, 0, sizeof(struct ibv_qp_init_attr));
  struct ibv_qp_cap *qp_cap = &attr.cap;

  attr.send_cq = send_cq;
  // attr.recv_cq = send_cq;
  attr.recv_cq = recv_cq;

  attr.cap.max_inline_data = param->inline_size;

  // The max work request number can be in SQ.
  // Rmember that read req and send req are mixed together.
  // MAX_RECV_NUM is the max send req number.
  // MAX_READ_NUM is the max read req number.
  attr.cap.max_send_wr = (MAX_WRITE_NUM + MAX_READ_NUM) * MACHINE_NUM;
  attr.cap.max_send_sge = MAX_SEND_SGE;

  attr.srq = NULL;
  // The max work request number can be in RQ.
  attr.cap.max_recv_wr = MAX_WRITE_NUM * MACHINE_NUM;
  attr.cap.max_recv_sge = MAX_RECV_SGE;

  attr.qp_type = IBV_QPT_RC;

  qp = ibv_create_qp(pd, &attr);

  if (qp == NULL && errno == ENOMEM) {
    fprintf(
        stderr,
        "Requested QP size might be too big. Try reducing TX depth "
        "and/or inline size.\n");
  }

  if (param->inline_size > qp_cap->max_inline_data) {
    printf(
        "  Actual inline-size(%d) < requested inline-size(%d)\n",
        qp_cap->max_inline_data, param->inline_size);
    param->inline_size = qp_cap->max_inline_data;
  }

  return qp;
}

int RdmaContext::modify_qp_to_init(RdmaParameter *param, int qp_index) {
  struct ibv_qp_attr attr;
  int flags = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT;
  int is_dc_server_side = 0;

  int ret = 0;

  memset(&attr, 0, sizeof(struct ibv_qp_attr));
  attr.qp_state = IBV_QPS_INIT;
  attr.pkey_index = param->pkey_index;

  attr.port_num = param->ib_port;
  attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
  // attr.qp_access_flags = IBV_ACCESS_REMOTE_ATOMIC;

  flags |= IBV_QP_ACCESS_FLAGS;
  ret = ibv_modify_qp(qp[qp_index], &attr, flags);

  if (ret) {
    fprintf(stderr, "Failed to modify QP to INIT, ret=%d\n", ret);
    return 1;
  }
  return 0;
}

int RdmaContext::ctx_init(
    RdmaParameter *param, RdmaContext *main_ctx, char buf_ptr[][MAX_VEC_LEN],
    size_t buf_len, SendWriteBuffer *local_send_write,
    RecvWriteBuffer *local_recv_write
    // char send_write_ptr[][MAX_WRITE_NUM][MAX_QUERYBUFFER_SIZE],
    // size_t send_write_length,
    // char recv_write_ptr[][MAX_WRITE_NUM][MAX_QUERYBUFFER_SIZE],
    // size_t recv_write_length,
    // char send_ptr[][MAX_RECV_NUM][MAX_QUERYBUFFER_SIZE], size_t send_length,
    // char recv_ptr[][MAX_RECV_NUM][MAX_QUERYBUFFER_SIZE], size_t recv_len
) {
  // pd = main_pd;

  auto send_write_ptr = local_send_write->buffer;
  auto send_write_length = local_send_write->buffer_len;
  auto recv_write_ptr = local_recv_write->buffer;
  auto recv_write_length = local_recv_write->buffer_len;

  auto &tp = getThreadPool();
  if (!pd) {
    printf("thread %d pd wrong.\n", tp.getTID());
  }

  if (create_ctx_mr(
          param, main_ctx, buf_ptr, buf_len, local_send_write, local_recv_write
          // local_send_write->buffer,local_send_write->buffer_len,
          // local_recv_write->buffer,local_recv_write->buffer_len
          // , send_ptr, send_length, recv_ptr,recv_len
          )) {
    fprintf(stderr, "Failed to create MR\n");
    ibv_dealloc_pd(pd);
    abort();
  }

  if (create_cqs(main_ctx, param)) {
    fprintf(stderr, "Failed to create CQs\n");
  }

  int qp_index = 0;
  for (int i = 0; i < param->machine_num; i++) {
    qp[i] = qp_create(main_ctx, param, i);
    auto &tp = getThreadPool();
    if (qp[i] == NULL) {
      // fprintf(stderr, "thread %d unable to create QP.\n");
      printf("thread %d create qp failed\n", tp.getTID());
      for (int j = 0; j < qp_index; j++) {
        ibv_destroy_qp(qp[j]);
      }
      abort();
    } else {
      // printf("thread %d create qp success\n", tp.getTID());
    }
    modify_qp_to_init(param, i);
    qp_index++;
  }

  return SUCCESS;
}

int RdmaContext::modify_qp_to_rtr(
    struct ibv_qp *qp, struct ibv_qp_attr *attr, RdmaParameter *param,
    struct MessageContext *dest, struct MessageContext *my_dest, int qp_index) {
  int flags = IBV_QP_STATE;

  attr->qp_state = IBV_QPS_RTR;
  attr->ah_attr.src_path_bits = 0;
  attr->ah_attr.port_num = param->ib_port;

  attr->ah_attr.dlid = dest->lid;
  attr->ah_attr.sl = param->sl;
  attr->ah_attr.is_global = 0;

  attr->path_mtu = param->curr_mtu;
  attr->dest_qp_num = dest->qpn;
  attr->rq_psn = dest->psn;

  flags |= (IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN);

  attr->max_dest_rd_atomic = my_dest->out_reads;
  attr->min_rnr_timer = MIN_RNR_TIMER;

  flags |= (IBV_QP_MIN_RNR_TIMER | IBV_QP_MAX_DEST_RD_ATOMIC);

  return ibv_modify_qp(qp, attr, flags);
}

int RdmaContext::modify_qp_to_rts(
    struct ibv_qp *qp, struct ibv_qp_attr *attr, RdmaParameter *param,
    struct MessageContext *dest, struct MessageContext *my_dest) {
  int flags = IBV_QP_STATE;

  attr->qp_state = IBV_QPS_RTS;

  flags |= IBV_QP_SQ_PSN;
  attr->sq_psn = my_dest->psn;

  attr->timeout = param->qp_timeout;
  attr->retry_cnt = 7;
  attr->rnr_retry = 7;
  attr->max_rd_atomic = dest->out_reads;
  flags |=
      (IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
       IBV_QP_MAX_QP_RD_ATOMIC);

  return ibv_modify_qp(qp, attr, flags);
}

int RdmaContext::connect(RdmaParameter *param) {
  struct ibv_qp_attr attr;
  for (int i = 0; i < param->machine_num; i++) {
    if (i == param->machine_id) continue;
    memset(&attr, 0, sizeof attr);

    // printf("modify qp to rtr\n");
    if (modify_qp_to_rtr(
            qp[i], &attr, param, &remote_mr_msg[i], &local_mr_msg[i], i)) {
      fprintf(stderr, "Failed to modify QP %d to RTR\n", qp[i]->qp_num);
      return FAILURE;
    }

    // printf("modify qp to rts\n");
    if (modify_qp_to_rts(
            qp[i], &attr, param, &remote_mr_msg[i], &local_mr_msg[i])) {
      fprintf(stderr, "Failed to modify QP to RTS\n");
      return FAILURE;
    }
  }
  return SUCCESS;
}

uint16_t RdmaContext::get_local_lid(int port) {
  struct ibv_port_attr attr;

  if (ibv_query_port(context, port, &attr)) return 0;

  // coverity[uninit_use]
  return attr.lid;
}

void RdmaContext::read_init(RdmaParameter *param) {
  for (int m = 0; m < MACHINE_NUM; m++) {
    for (int r = 0; r < MAX_READ_NUM; r++) {
      memset(&wr[m][r], 0, sizeof(ibv_send_wr));
      wr[m][r].wr.rdma.remote_addr =
          remote_mr_msg[m].read_vaddr;  // read source
      wr[m][r].wr.rdma.rkey = remote_mr_msg[m].read_rkey;
      wr[m][r].num_sge = MAX_SEND_SGE;
      wr[m][r].next = NULL;
      wr[m][r].send_flags = IBV_SEND_SIGNALED;
      wr[m][r].opcode = IBV_WR_RDMA_READ;
    }
  }

  for (int b = 0; b < MAX_CACHE_VEC; b++) {
    sge_list[b].addr = (uintptr_t)read_to_ptr[b];
    sge_list[b].lkey = read_to_mr->lkey;
  }
}

int RdmaContext::run_post_read(int machine_id) {
  struct ibv_send_wr *bad_wr = NULL;
  int err = ibv_post_send(qp[machine_id], &wr[machine_id][0], &bad_wr);
  if (err) {
    fprintf(stderr, "Couldn't post send");
    return 1;
  }
  return 0;
}

void RdmaContext::write_init(RdmaParameter *param) {
  for (int m = 0; m < param->machine_num; m++) {
    for (int i = 0; i < MAX_WRITE_NUM; i++) {
      // init swr
      memset(&wwr[m][i], 0, sizeof(struct ibv_send_wr));

      wwr[m][i].wr.rdma.rkey = remote_mr_msg[m].write_rkey;

      write_sge_list[m][i][0].addr = (uintptr_t)send_write_ptr[m][i];
      write_sge_list[m][i][0].lkey = send_write_mr[m]->lkey;

      wwr[m][i].sg_list = &write_sge_list[m][i][0];
      wwr[m][i].num_sge = MAX_SEND_SGE;
      // Record the machine id and buff id.
      wwr[m][i].wr_id = (m << 16) | i;
      wwr[m][i].next = NULL;

      wwr[m][i].send_flags = IBV_SEND_SIGNALED;

      wwr[m][i].opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
      // wr[i].send_flags |= IBV_SEND_INLINE;
    }

    for (int i = 0; i < MAX_WRITE_NUM; i++) {
      // Init rwr
      memset(&rwwr[m][i], 0, sizeof(struct ibv_recv_wr));

      rwwr[m][i].sg_list = NULL;
      rwwr[m][i].num_sge = 0;
      // Record the machine id and buff id.
      rwwr[m][i].wr_id = (m << 16) | i;

      rwwr[m][i].next = NULL;
    }
  }
}

int RdmaContext::run_post_write_send(
    int machine_id, uint32_t buf_id, size_t offset, uint32_t size,
    uint32_t control) {
  wwr[machine_id][buf_id].wr.rdma.remote_addr =
      remote_mr_msg[machine_id].write_vaddr + offset;
  wwr[machine_id][buf_id].sg_list->length = size;
  wwr[machine_id][buf_id].num_sge = MAX_SEND_SGE;
  wwr[machine_id][buf_id].imm_data = htonl(control);

  struct ibv_send_wr *bad_wr = NULL;
  int err = ibv_post_send(qp[machine_id], &wwr[machine_id][buf_id], &bad_wr);
  if (err) {
    fprintf(stderr, "post_write_send: Couldn't post send");
    return 1;
  }
  return 0;
}

// TODO: use pointer to link the multi-requests.
int RdmaContext::run_post_sg_write_send(
    int machine_id, uint32_t buf_id, size_t offset, std::vector<char *> &sg_ptr,
    std::vector<uint32_t> &sg_size, uint32_t control) {
  int sg_num = sg_ptr.size();

  // TODO: can not mix with original send method.
  // printf("sg_num:%u sg_size:%u\n", sg_num, size);

  // NOTE: set pos 0 as default buffer ptr, and use corresponding mr.
  if (sg_num > MAX_SGE_NUM) {
    printf("Exceed max sge :%u\n", sg_num);
    abort();
  }
  // printf("post buffer %u sg num %u\n", buf_id, sg_num);
  // printf("sg %u size %u\n", 0, sg_size[0]);
  write_sge_list[machine_id][buf_id][0].length = sg_size[0];
  for (size_t s = 1; s < sg_num; s++) {
    write_sge_list[machine_id][buf_id][s].addr = (uintptr_t)sg_ptr[s];
    write_sge_list[machine_id][buf_id][s].length = sg_size[s];
    write_sge_list[machine_id][buf_id][s].lkey = for_read_mr[machine_id]->lkey;
    // printf("sg %u size %u\n", s, sg_size[s]);
  }

  wwr[machine_id][buf_id].wr.rdma.remote_addr =
      remote_mr_msg[machine_id].write_vaddr + offset;
  wwr[machine_id][buf_id].num_sge = sg_num;
  wwr[machine_id][buf_id].imm_data = htonl(control);

  struct ibv_send_wr *bad_wr = NULL;
  int err = ibv_post_send(qp[machine_id], &wwr[machine_id][buf_id], &bad_wr);
  if (err) {
    fprintf(stderr, "post_write_send: Couldn't post send");
    return 1;
  }
  return 0;
}

int RdmaContext::run_post_write_recv(int machine_id, size_t buff_id) {
  struct ibv_recv_wr *bad_wr_recv;
  if (ibv_post_recv(qp[machine_id], &rwwr[machine_id][buff_id], &bad_wr_recv)) {
    fprintf(stderr, "Couldn't post recv Qp = %d\n", machine_id);
    return 1;
  }
  return 0;
}

int RdmaContext::poll_SEND() {
  int ne = ibv_poll_cq(send_cq, MAX_WC_NUM, s_wc);
  if (ne < 0) {
    fprintf(stderr, "poll CQ failed %d when try recv\n", ne);
    return -1;
  }
  return ne;
}

int RdmaContext::poll_RECV() {
  int ne = ibv_poll_cq(recv_cq, MAX_WC_NUM, r_wc);
  if (ne < 0) {
    fprintf(stderr, "poll CQ failed %d when try recv\n", ne);
    return -1;
  }
  return ne;
}

void RdmaContext::send_init(RdmaParameter *param) {
  for (int m = 0; m < param->machine_num; m++) {
    for (int i = 0; i < MAX_RECV_NUM; i++) {
      // init swr
      memset(&swr[m][i], 0, sizeof(struct ibv_send_wr));
      send_sge_list[m][i].addr =
          (uintptr_t)send_ptr[m][i];  // Send data pointer
      send_sge_list[m][i].lkey = send_mr[m]->lkey;

      swr[m][i].sg_list = &send_sge_list[m][i];
      swr[m][i].num_sge = MAX_SEND_SGE;
      // Record the machine id and buff id.
      swr[m][i].wr_id = (m << 16) | i;
      swr[m][i].next = NULL;

      // TODO: unsignal optimization
      swr[m][i].send_flags = IBV_SEND_SIGNALED;

      swr[m][i].opcode = IBV_WR_SEND;
      // wr[i].send_flags |= IBV_SEND_INLINE;
    }

    for (int i = 0; i < MAX_RECV_NUM; i++) {
      // Init rwr
      memset(&rwr[m][i], 0, sizeof(struct ibv_recv_wr));

      recv_sge_list[m][i].addr = (uintptr_t)recv_ptr[m][i];  // recv destination
      recv_sge_list[m][i].lkey = recv_mr[m]->lkey;

      rwr[m][i].sg_list = &recv_sge_list[m][i];
      rwr[m][i].num_sge = MAX_RECV_SGE;
      // Record the machine id and buff id.
      rwr[m][i].wr_id = (m << 16) | i;

      rwr[m][i].next = NULL;
    }
  }
}

/*
 Sync send.
*/
int RdmaContext::run_sync_send(int machine_id, uint32_t size) {
  swr[machine_id][0].sg_list->length = size;
  swr[machine_id][0].send_flags = IBV_SEND_SIGNALED;

  struct ibv_send_wr *bad_wr = NULL;
  int err = ibv_post_send(qp[machine_id], &swr[machine_id][0], &bad_wr);
  if (err) {
    fprintf(stderr, "Couldn't post send");
    return 1;
  }
  struct ibv_wc s_wc;
  int s_ne;
  do {
    s_ne = ibv_poll_cq(send_cq, 1, &s_wc);
  } while (s_ne == 0);

  if (s_ne < 0) {
    fprintf(stderr, "poll on Send CQ failed %d\n", s_ne);
    return FAILURE;
  }
  if (s_wc.status != IBV_WC_SUCCESS) {
    // coverity[uninit_use_in_call]
    fprintf(stderr, " Completion with error at client\n");
    fprintf(
        stderr, " Failed status %d: wr_id %d syndrom 0x%x\n", s_wc.status,
        (int)s_wc.wr_id, s_wc.vendor_err);
    return 1;
  }
  swr[machine_id][0].send_flags &= ~IBV_SEND_SIGNALED;
  return 0;
}

/*
 Async send.
*/
int RdmaContext::run_post_send(int machine_id, size_t buff_id, uint32_t size) {
  // printf("prepare post send machine %d buff %d\n", machine_id, buff_id);
  swr[machine_id][buff_id].sg_list->length = size;
  swr[machine_id][buff_id].send_flags = IBV_SEND_SIGNALED;

  struct ibv_send_wr *bad_wr = NULL;

  int err = ibv_post_send(qp[machine_id], &swr[machine_id][buff_id], &bad_wr);
  if (err) {
    fprintf(stderr, "Couldn't post send");
    return 1;
  }
  // printf("post send oovver\n");
  swr[machine_id][buff_id].send_flags &= ~IBV_SEND_SIGNALED;
  // return without ack.
  return 0;
}

int RdmaContext::run_sync_recv(uint32_t recv_num) {
  struct ibv_wc wc;
  struct ibv_recv_wr *bad_wr_recv;
  int ne;
  uint32_t recv_cnt = 0;

  do {
    ne = ibv_poll_cq(recv_cq, 1, &wc);
    if (ne > 0) {
      if (wc.status != IBV_WC_SUCCESS) {
        fprintf(stderr, " Completion with error at client\n");
        fprintf(
            stderr, " Failed status %d: wr_id %d syndrom 0x%x\n", wc.status,
            (int)wc.wr_id, wc.vendor_err);
        return 1;
      }
      recv_cnt += ne;
      if (recv_cnt > recv_num) {
        fprintf(stderr, " Recv unexpected number of send message.\n");
        return 1;
      }
    } else if (ne < 0) {
      fprintf(stderr, "poll CQ failed %d\n", ne);
      return 1;
    }
  } while (recv_cnt < recv_num);

  return 0;
}

int RdmaContext::run_post_recv(int machine_id, size_t buff_id, uint32_t size) {
  recv_sge_list[machine_id][buff_id].length = size;
  recv_sge_list[machine_id][buff_id].addr =
      (uintptr_t)recv_ptr[machine_id][buff_id];  // recv destination
  rwr[machine_id][buff_id].sg_list = &recv_sge_list[machine_id][buff_id];

  struct ibv_recv_wr *bad_wr_recv;
  if (ibv_post_recv(qp[machine_id], &rwr[machine_id][buff_id], &bad_wr_recv)) {
    fprintf(stderr, "Couldn't post recv Qp = %d\n", machine_id);
    return 1;
  }
  return 0;
}