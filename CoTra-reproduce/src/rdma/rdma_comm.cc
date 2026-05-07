#include "rdma/rdma_comm.h"

int RdmaCommunication::ethernet_leader_connect() {
  struct addrinfo *res, *t;
  struct addrinfo hints;
  char *service;
  struct sockaddr_in source;

  int sockfd = -1;
  memset(&hints, 0, sizeof hints);
  hints.ai_family = rdma_param.ai_family;
  hints.ai_socktype = SOCK_STREAM;

  for (int i = rdma_param.machine_id + 1; i < rdma_param.machine_num; i++) {
    if (check_add_port(
            &service, rdma_param.port, rdma_param.machine_name[i].c_str(), &hints, &res)) {
      fprintf(stderr, "Problem in resolving basic address and port\n");
      return 1;
    }

    for (t = res; t; t = t->ai_next) {
      sockfd = socket(t->ai_family, t->ai_socktype, t->ai_protocol);
      if (sockfd >= 0) {
        if (!connect(sockfd, t->ai_addr, t->ai_addrlen)) break;
        close(sockfd);
        sockfd = -1;
      }
    }

    freeaddrinfo(res);

    if (sockfd < 0) {
      fprintf(
          stderr, "Couldn't connect to %s:%d\n", rdma_param.machine_name[i].c_str(),
          rdma_param.port);
      return 1;
    }

    rdma_param.sockfd[i] = sockfd;
  }

  return 0;
}

int RdmaCommunication::ethernet_member_connect(int leader_id) {
  printf("Run ethernet member connect\n");
  struct addrinfo *res, *t;
  struct addrinfo hints;
  char *service;
  int n;
  int sockfd = -1, connfd;
  char *src_ip = NULL;

  memset(&hints, 0, sizeof hints);
  hints.ai_flags = AI_PASSIVE;
  hints.ai_family = rdma_param.ai_family;
  hints.ai_socktype = SOCK_STREAM;

  if (check_add_port(&service, rdma_param.port, src_ip, &hints, &res)) {
    fprintf(stderr, "Problem in resolving basic address and port\n");
    return 1;
  }

  for (t = res; t; t = t->ai_next) {
    if (t->ai_family != rdma_param.ai_family) continue;

    sockfd = socket(t->ai_family, t->ai_socktype, t->ai_protocol);

    if (sockfd >= 0) {
      n = 1;
      setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &n, sizeof n);
      if (!bind(sockfd, t->ai_addr, t->ai_addrlen)) break;
      close(sockfd);
      sockfd = -1;
    }
  }
  freeaddrinfo(res);

  if (sockfd < 0) {
    fprintf(stderr, "Couldn't listen to port %d\n", rdma_param.port);
    return 1;
  }

  listen(sockfd, 1);
  connfd = accept(sockfd, NULL, 0);

  if (connfd < 0) {
    perror("server accept");
    fprintf(stderr, "accept() failed\n");
    close(sockfd);
    return 1;
  }
  close(sockfd);
  rdma_param.sockfd[leader_id] = connfd;
  return 0;
}

int RdmaCommunication::establish_connection() {
  printf("Establish eathnet connection ...\n");

  sc = new ServerConnect(rdma_param, rdma_ctx);
  auto myid = sc->get_my_id();
  sc->barrier("RDMA-init");

  // for (int leader_id = 0; leader_id < MACHINE_NUM; leader_id++) {
  //   if (leader_id == rdma_param.machine_id) {
  //     if (ethernet_leader_connect()) {
  //       fprintf(
  //           stderr, "Unable to open file descriptor for socket
  //           connection\n");
  //       return 1;
  //     }
  //   } else if (leader_id < rdma_param.machine_id) {
  //     if (ethernet_member_connect(leader_id)) {
  //       fprintf(
  //           stderr, "Unable to open file descriptor for socket
  //           connection\n");
  //       return 1;
  //     }
  //   }
  //   // if (rdma_param.machine == LEADER) {
  //   //   if (ethernet_leader_connect()) {
  //   //     fprintf(
  //   //         stderr, "Unable to open file descriptor for socket
  //   //         connection\n");
  //   //     return 1;
  //   //   }
  //   // } else {
  //   //   if (ethernet_member_connect()) {
  //   //     fprintf(
  //   //         stderr, "Unable to open file descriptor for socket
  //   //         connection\n");
  //   //     return 1;
  //   //   }
  //   // }
  // }

  return 0;
}

inline int ipv6_addr_v4mapped(const struct in6_addr *a) {
  return ((a->s6_addr32[0] | a->s6_addr32[1]) |
          (a->s6_addr32[2] ^ htonl(LOW_16BIT_MASK))) == 0UL ||
         /* IPv4 encoded multicast addresses */
         (a->s6_addr32[0] == htonl(0xff0e0000) &&
          ((a->s6_addr32[1] | (a->s6_addr32[2] ^ htonl(LOW_16BIT_MASK))) ==
           0UL));
}

int get_best_gid_index(RdmaContext *ctx, struct ibv_port_attr *attr, int port) {
  int gid_index = 0, i;
  union ibv_gid temp_gid, temp_gid_rival;
  int is_ipv4, is_ipv4_rival;

  for (i = 1; i < attr->gid_tbl_len; i++) {
    if (ibv_query_gid(ctx->context, port, gid_index, &temp_gid)) {
      return -1;
    }

    if (ibv_query_gid(ctx->context, port, i, &temp_gid_rival)) {
      return -1;
    }

    is_ipv4 = ipv6_addr_v4mapped((struct in6_addr *)temp_gid.raw);
    is_ipv4_rival = ipv6_addr_v4mapped((struct in6_addr *)temp_gid_rival.raw);

    if (is_ipv4_rival && !is_ipv4) gid_index = i;
  }
  return gid_index;
}

int RdmaCommunication::set_up_connection(RdmaContext *ctx) {
  union ibv_gid temp_gid;
  struct ibv_port_attr attr;

  srand48(getpid() * time(NULL));

  if (rdma_param.gid_index != -1) {
    if (ibv_query_port(ctx->context, rdma_param.ib_port, &attr)) return 0;

    rdma_param.gid_index = get_best_gid_index(ctx, &attr, rdma_param.ib_port);
    if (rdma_param.gid_index < 0) return -1;
    if (ibv_query_gid(
            ctx->context, rdma_param.ib_port, rdma_param.gid_index, &temp_gid))
      return -1;
  }

  for (int i = 0; i < rdma_param.machine_num; i++) {
    // printf("set up qps %d\n",i);
    /*single-port case*/
    ctx->local_mr_msg[i].lid = ctx->get_local_lid(rdma_param.ib_port);
    ctx->local_mr_msg[i].gid_index = rdma_param.gid_index;

    // printf("set qpn\n");
    ctx->local_mr_msg[i].qpn = ctx->qp[i]->qp_num;

    // printf("mr\n");
    ctx->local_mr_msg[i].psn = lrand48() & 0xffffff;
    // my_dest[i].read_rkey = ctx->mr[0]->rkey;
    ctx->local_mr_msg[i].read_rkey = ctx->for_read_mr[i]->rkey;

    ctx->local_mr_msg[i].write_rkey = ctx->recv_write_mr[i]->rkey;

    // printf("set out reads\n");
    /* Each qp gives his receive buffer address.*/
    ctx->local_mr_msg[i].out_reads = rdma_param.out_reads;

    ctx->local_mr_msg[i].read_vaddr = (uintptr_t)ctx->for_read_ptr[i];

    ctx->local_mr_msg[i].write_vaddr = (uintptr_t)ctx->recv_write_ptr[i][0];

    memcpy(ctx->local_mr_msg[i].gid.raw, temp_gid.raw, 16);
  }

  return 0;
}

int RdmaCommunication::ethernet_read_keys(
    struct MessageContext *rem_dest, int sockfd) {
  if (rem_dest->gid_index == -1) {
    int parsed;
    char msg[KEY_MSG_SIZE];

    if (read(sockfd, msg, sizeof msg) != sizeof msg) {
      fprintf(stderr, "ethernet_read_keys: Couldn't read remote address\n");
      return 1;
    }

    parsed = sscanf(
        msg, KEY_PRINT_FMT, (unsigned int *)&rem_dest->lid,
        (unsigned int *)&rem_dest->out_reads, (unsigned int *)&rem_dest->qpn,
        (unsigned int *)&rem_dest->psn, &rem_dest->read_rkey,
        &rem_dest->read_vaddr, &rem_dest->write_rkey, &rem_dest->write_vaddr,
        &rem_dest->srqn);

    // printf("parsed %d\n", parsed);
    if (parsed != 9) {
      // coverity[string_null]
      fprintf(stderr, "Couldn't parse line <%.*s>\n", (int)sizeof msg, msg);
      return 1;
    }

  } else {
    char msg[KEY_MSG_SIZE_GID];
    char *pstr = msg, *term;
    char tmp[120];
    int i;

    if (read(sockfd, msg, sizeof msg) != sizeof msg) {
      fprintf(stderr, "ethernet_read_keys: Couldn't read remote address\n");
      return 1;
    }

    term = strpbrk(pstr, ":");
    memcpy(tmp, pstr, term - pstr);
    tmp[term - pstr] = 0;
    rem_dest->lid = (int)strtol(tmp, NULL, 16); /*LID*/

    pstr += term - pstr + 1;
    term = strpbrk(pstr, ":");
    memcpy(tmp, pstr, term - pstr);
    tmp[term - pstr] = 0;
    rem_dest->out_reads = (int)strtol(tmp, NULL, 16); /*OUT_READS*/

    pstr += term - pstr + 1;
    term = strpbrk(pstr, ":");
    memcpy(tmp, pstr, term - pstr);
    tmp[term - pstr] = 0;
    rem_dest->qpn = (int)strtol(tmp, NULL, 16); /*QPN*/

    pstr += term - pstr + 1;
    term = strpbrk(pstr, ":");
    memcpy(tmp, pstr, term - pstr);
    tmp[term - pstr] = 0;
    rem_dest->psn = (int)strtol(tmp, NULL, 16); /*PSN*/

    pstr += term - pstr + 1;
    term = strpbrk(pstr, ":");
    memcpy(tmp, pstr, term - pstr);
    tmp[term - pstr] = 0;
    rem_dest->read_rkey = (unsigned)strtoul(tmp, NULL, 16); /*RKEY*/

    pstr += term - pstr + 1;
    term = strpbrk(pstr, ":");
    memcpy(tmp, pstr, term - pstr);
    tmp[term - pstr] = 0;

    rem_dest->read_vaddr = strtoull(tmp, NULL, 16); /*VA*/

    for (i = 0; i < 15; ++i) {
      pstr += term - pstr + 1;
      term = strpbrk(pstr, ":");
      memcpy(tmp, pstr, term - pstr);
      tmp[term - pstr] = 0;

      rem_dest->gid.raw[i] = (unsigned char)strtoll(tmp, NULL, 16);
    }

    pstr += term - pstr + 1;

    strcpy(tmp, pstr);
    rem_dest->gid.raw[15] = (unsigned char)strtoll(tmp, NULL, 16);

    pstr += term - pstr + 4;

    term = strpbrk(pstr, ":");
    memcpy(tmp, pstr, term - pstr);
    tmp[term - pstr] = 0;
    rem_dest->srqn = (unsigned)strtoul(tmp, NULL, 16); /*SRQN*/
  }
  return 0;
}

int RdmaCommunication::ethernet_write_keys(
    struct MessageContext *my_dest, int sockfd) {
  if (my_dest->gid_index == -1) {
    char msg[KEY_MSG_SIZE];

    sprintf(
        msg, KEY_PRINT_FMT, my_dest->lid, my_dest->out_reads, my_dest->qpn,
        my_dest->psn, my_dest->read_rkey, my_dest->read_vaddr,
        my_dest->write_rkey, my_dest->write_vaddr, my_dest->srqn);

    if (write(sockfd, msg, sizeof msg) != sizeof msg) {
      perror("client write");
      fprintf(stderr, "Couldn't send local address\n");
      return 1;
    }

  } else {
    printf("write!=-1\n");
    char msg[KEY_MSG_SIZE_GID];
    sprintf(
        msg, KEY_PRINT_FMT_GID, my_dest->lid, my_dest->out_reads, my_dest->qpn,
        my_dest->psn, my_dest->read_rkey, my_dest->read_vaddr,
        my_dest->gid.raw[0], my_dest->gid.raw[1], my_dest->gid.raw[2],
        my_dest->gid.raw[3], my_dest->gid.raw[4], my_dest->gid.raw[5],
        my_dest->gid.raw[6], my_dest->gid.raw[7], my_dest->gid.raw[8],
        my_dest->gid.raw[9], my_dest->gid.raw[10], my_dest->gid.raw[11],
        my_dest->gid.raw[12], my_dest->gid.raw[13], my_dest->gid.raw[14],
        my_dest->gid.raw[15], my_dest->srqn);

    if (write(sockfd, msg, sizeof msg) != sizeof msg) {
      perror("client write");
      fprintf(stderr, "Couldn't send local address\n");
      return 1;
    }
  }

  return 0;
}

int RdmaCommunication::ctx_hand_shake(
    struct MessageContext *my_dest, struct MessageContext *rem_dest,
    int sockfd) {
  // printf("ctx_hand_shake\n");
  // int (*read_func_ptr)(struct MessageContext *);
  // int (*write_func_ptr)(struct MessageContext *);

  // read_func_ptr = &ethernet_read_keys;
  // write_func_ptr = &ethernet_write_keys;

  rem_dest->gid_index = my_dest->gid_index;

  // Client side
  if (rdma_param.machine == LEADER) {
    if (ethernet_write_keys(my_dest, sockfd)) {
      fprintf(stderr, " Unable to write to socket/rdma_cm\n");
      return 1;
    }
    if (ethernet_read_keys(rem_dest, sockfd)) {
      fprintf(stderr, " Unable to read from socket/rdma_cm\n");
      return 1;
    }

    // Server side
    /*Server side will wait for the client side to reach the write function.*/
  } else {
    if (ethernet_read_keys(rem_dest, sockfd)) {
      fprintf(stderr, " Unable to read to socket/rdma_cm\n");
      return 1;
    }
    if (ethernet_write_keys(my_dest, sockfd)) {
      fprintf(stderr, " Unable to write from socket/rdma_cm\n");
      return 1;
    }
  }

  return 0;
}

int RdmaCommunication::ctx_close_connection() {
  /*Signal client is finished.*/
  for (int m = 0; m < rdma_param.machine_num; m++) {
    if (m == rdma_param.machine_id) continue;
    for (int t = 0; t < rdma_param.thread_num; t++) {
      auto *ctx = rdma_ctx.getRemote(t);
      /* shaking hands and gather the other side info. */
      if (ctx_hand_shake(
              &ctx->local_mr_msg[m], &ctx->remote_mr_msg[m],
              rdma_param.sockfd[m])) {
        fprintf(stderr, "Failed to exchange data between server and clients\n");
        exit(0);
      }
    }
  }

  for (int m = 0; m < rdma_param.machine_num; m++) {
    if (m == rdma_param.machine_id) continue;
    close(rdma_param.sockfd[m]);
  }

  return 0;
}

void RdmaCommunication::com_init(
    char *vec_part_ptr, size_t vec_part_len, RdmaParameter &input_param) {
  // on_each([&](uint64 tid, uint64 total) {
  auto *main_ctx = rdma_ctx.getLocal();
  memset(main_ctx, 0, sizeof(RdmaContext));
  rdma_param = input_param;

  struct ibv_device *ib_dev = ctx_find_dev(&rdma_param.ib_devname);
  if (!ib_dev) {
    fprintf(stderr, " Unable to find the Infiniband/RoCE device\n");
    exit(0);
  }

  main_ctx->context = ibv_open_device(ib_dev);
  if (!main_ctx->context) {
    fprintf(stderr, " Couldn't get context for the device\n");
    exit(0);
  }

  if (check_link(main_ctx->context, &rdma_param)) {
    fprintf(stderr, " Couldn't get context for the device\n");
    exit(0);
  }

  // rdma_param.memory_create = host_memory_create;

  for (int t = 1; t < getActiveThreads(); t++) {
    auto *ctx = rdma_ctx.getRemote(t);
    memcpy(ctx, main_ctx, sizeof(RdmaContext));
  }

  // if (rdma_param.machine == MEMBER) {
  //   printf("\n************************************\n");
  //   printf("* Waiting for client to connect... *\n");
  //   printf("************************************\n");
  // }

  on_each([&](uint64 tid, uint64 total) {
    auto *ctx = rdma_ctx.getLocal();
    // auto *&param = rdma_param;

    if (ctx->check_mtu(&rdma_param)) {
      fprintf(stderr, " Couldn't get context for the device\n");
      exit(0);
    }

    if (ctx->alloc_ctx(&rdma_param)) {
      fprintf(stderr, "Couldn't allocate context\n");
      exit(0);
    }

    ctx->pd = ibv_alloc_pd(ctx->context);
    if (!ctx->pd) {
      fprintf(stderr, "Couldn't allocate PD\n");
      exit(-1);
    }
    ctx->create_main_partition_mr(
        vec_part_ptr, vec_part_len, rdma_param.cache_line_size);
  });

  // for(int t = 0; t<getActiveThreads(); t++){
  //   auto* ctx = rdma_ctx.getRemote(t);
  //   printf("Alloc thread %d main mr.\n", t);
  //   create_main_partition_mr(ctx, vec_part_ptr, vec_part_len);
  // }

  // auto *main_ctx = rdma_ctx.getLocal();  // ctx[thread0]
  /* Allocating the Protection domain. */
  // main_ctx->pd = ibv_alloc_pd(main_ctx->context);
  // if (!main_ctx->pd) {
  //   fprintf(stderr, "Couldn't allocate PD\n");
  //   exit(-1);
  // }

  // create_main_partition_mr(main_ctx, vec_part_ptr, vec_part_len);

  on_each([&](uint64 tid, uint64 total) {
    auto *local_buf = read_buf.getLocal();
    auto *local_send_write = send_write_buf.getLocal();
    auto *local_recv_write = recv_write_buf.getLocal();
    // auto *local_send = send_buf.getLocal();
    // auto *local_recv = recv_buf.getLocal();
    auto *ctx = rdma_ctx.getLocal();
    // send_buf recv_buf init
    local_send_write->init();
    local_recv_write->init();
    // local_send->init();
    // local_recv->init();

    // Register the buffers as our machine recv data buffer.
    if (ctx->ctx_init(
            &rdma_param, main_ctx, local_buf->vector,
            MACHINE_NUM * MAX_READ_NUM * MAX_VEC_LEN, local_send_write,
            local_recv_write
            //  local_send_write->buffer,
            // local_send_write->buffer_len, local_recv_write->buffer,
            // local_recv_write->buffer_len, local_send->buffer,
            // local_send->buffer_len, local_recv->buffer,
            // local_recv->buffer_len
            )) {
      fprintf(stderr, " Couldn't create IB resources\n");
      ctx->dealloc_ctx(&rdma_param);
      // free(rem_dest);
      exit(0);
    }
  });

  // auto &tp = getThreadPool();
  // for (int t = 0; t < tp.getMaxThreads(); t++) {
  //   auto *ctx = rdma_ctx.getRemote(t);
  //   auto *param = rdma_param.getRemote(t);
  //   for (int i = 0; i < rdma_param.machine_num; i++) {
  //     ctx->qp[i] = ctx_qp_create(ctx, main_ctx, main_param, i);
  //     auto &tp = getThreadPool();
  //     if (ctx->qp[i] == NULL) {
  //       // fprintf(stderr, "thread %d unable to create QP.\n");
  //       printf("thread %d create qp failed\n", t);
  //     } else {
  //       printf("thread %d create qp success\n", t);
  //     }
  //     modify_qp_to_init(ctx, param, i);
  //   }
  // }

  on_each([&](uint64 tid, uint64 total) {
    // printf("set up connection\n");
    auto *ctx = rdma_ctx.getLocal();
    /* Set up the Connection. */
    if (set_up_connection(ctx)) {
      fprintf(stderr, " Unable to set up socket connection\n");
      exit(0);
    }
  });

  printf("Switch metadata with remote machine..\n");
  if (establish_connection()) {
    fprintf(stderr, " Unable to init the socket connection\n");
    exit(0);
  }

  // for (int leader_id = 0; leader_id < rdma_param.machine_num; leader_id++) {
  //   if (leader_id == rdma_param.machine_id) {
  //     rdma_param.machine = LEADER;
  //     for (int m = 0; m < rdma_param.machine_num; m++) {
  //       if (m == rdma_param.machine_id) continue;
  //       printf("shake hand with machine %d\n", m);
  //       for (int t = 0; t < rdma_param.thread_num; t++) {
  //         auto *ctx = rdma_ctx.getRemote(t);
  //         /* shaking hands and gather the other side info. */
  //         if (ctx_hand_shake(
  //                 &ctx->local_mr_msg[m], &ctx->remote_mr_msg[m],
  //                 rdma_param.sockfd[m])) {
  //           fprintf(
  //               stderr, "Failed to exchange data between server and
  //               clients\n");
  //           exit(0);
  //         }
  //       }
  //     }
  //   } else {
  //     rdma_param.machine = MEMBER;
  //     printf("shake hand with machine %d\n", leader_id);
  //     for (int t = 0; t < rdma_param.thread_num; t++) {
  //       auto *ctx = rdma_ctx.getRemote(t);
  //       /* shaking hands and gather the other side info. */
  //       if (ctx_hand_shake(
  //               &ctx->local_mr_msg[leader_id],
  //               &ctx->remote_mr_msg[leader_id],
  //               rdma_param.sockfd[leader_id])) {
  //         fprintf(
  //             stderr, "Failed to exchange data between server and
  //             clients\n");
  //         exit(0);
  //       }
  //     }
  //   }
  // }

  if (rdma_param.machine_id == 0) {
    rdma_param.machine = LEADER;
  } else {
    rdma_param.machine = MEMBER;
  }

  on_each([&](uint64 tid, uint64 total) {
    RdmaContext *ctx = rdma_ctx.getLocal();
    if (ctx->connect(&rdma_param)) {
      fprintf(stderr, " Unable to Connect the HCA's through the link\n");
      exit(-1);
    }
  });

  printf("RDMA communication init over. \n");
}

/**
 * Poll send cq one time.
 */
int RdmaCommunication::poll_send() {
  auto *ctx = rdma_ctx.getLocal();

  int ne = ctx->poll_SEND();
  if (ne > 0) {
    for (int i = 0; i < ne; i++) {
      auto &wc = ctx->s_wc[i];
      if (wc.status != IBV_WC_SUCCESS) {
        fprintf(stderr, "poll_send: Completion with error at client\n ");
        fprintf(
            stderr, " Failed status %d: qid %d buff %d syndrom 0x%x\n",
            wc.status, (int)wc.wr_id >> 16, (int)(wc.wr_id & LOW_16BIT_MASK),
            wc.vendor_err);
        return -1;
      }
      uint64_t wr_id = wc.wr_id;
      // TODO: change to switch.
      if (wc.opcode == IBV_WC_RDMA_READ) {
        ctx->collect_num++;
        // TODO: reduce copy.
        auto *rd_buf = read_buf.getLocal();
        rd_buf->recv_list.emplace_back(wr_id);  // qid | buf_id
        // printf(
        //     "read finished.q %u buf %d\n", wr_id >> 16, wr_id &
        //     LOW_16BIT_MASK);
      } else if (wc.opcode == IBV_WC_RDMA_WRITE) {
        // printf("write finished. %d\n", wr_id & LOW_16BIT_MASK);
      } else if (wc.opcode == IBV_WC_SEND) {
        printf("!!! recv SEND\n");
        // auto *sbuf = send_buf.getLocal();
        // // the send ack
        // size_t machine_id = wr_id >> 16;
        // size_t buff_id = wr_id & LOW_16BIT_MASK;
        // sbuf->buff_id_list[machine_id]->emplace_back(buff_id);
        // // TODO: does it needed?
        // ctx->swr[machine_id][buff_id].send_flags &= ~IBV_SEND_SIGNALED;
      } else {
        printf("Error: Unexpected opcode.\n");
      }
      // Update buffer list.
    }
  }
  return ne;
}

/**
 * Poll recv cq one time.
 */
// TODO: optimize buffer replicate(copy) efficiency.
int RdmaCommunication::poll_recv() {
  auto *ctx = rdma_ctx.getLocal();

  int ne = ctx->poll_RECV();
  if (ne > 0) {
    // printf("ne: %d\n", ne);
    for (int i = 0; i < ne; i++) {
      auto &wc = ctx->r_wc[i];
      if (wc.status != IBV_WC_SUCCESS) {
        fprintf(stderr, "poll_recv: Completion with error at client\n ");
        fprintf(
            stderr,
            " Failed status %d: machine %d true_wr_id %d syndrom 0x%x\n",
            wc.status, (int)wc.wr_id >> 16, (int)(wc.wr_id & LOW_16BIT_MASK),
            wc.vendor_err);
        return -1;
      }

      uint64_t wr_id = wc.wr_id;
      if (wc.opcode == IBV_WC_RECV) {
        // printf("recv finished.\n");
      } else if (wc.opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
        auto *rbuf = recv_write_buf.getLocal();
        auto *sbuf = send_write_buf.getLocal();

        uint32_t m_id = wr_id >> 16;
        uint32_t true_wr_id = wr_id & LOW_16BIT_MASK;
        uint32_t data_size = wc.byte_len;
        uint32_t imm = ntohl(wc.imm_data);
        ControlType control = ControlType(imm >> 16);
        uint32_t buf_id = imm & LOW_16BIT_MASK;
        char *ptr = (char *)rbuf->buffer[m_id][buf_id];

        // printf(
        //     "recv write imm, bufid %d control %d size %d\n", buf_id, control,
        //     data_size);
        
        // TODO: use switch
        if (control == RELEASE || control == CLEAR) {
          // proc the released send buffer id.
          int *release_ptr = (int *)ptr;
          int num = release_ptr[0];
          // printf("==> get release id ");
          for (int i = 1; i < num + 1; i++) {
            sbuf->buff_id_list[m_id]->push_back(release_ptr[i]);
            // printf("%d ", release_ptr[i]);
          }
          // printf("\n");
        } else if (control == B2_START) {
          char *b2_info = (char *)malloc(data_size);
          memcpy(b2_info, ptr, data_size);
          rbuf->b2_start_queue.push_back(b2_info);
        } else if (control == B2_END) {
          char *b2_info = (char *)malloc(data_size);
          memcpy(b2_info, ptr, data_size);
          rbuf->b2_end_queue.push_back(b2_info);
        } else if (control == PART_INFO) {
          char *part_info = (char *)malloc(data_size);
          memcpy(part_info, ptr, data_size);
          rbuf->part_info_queue.push_back(part_info);
        } else if (control == BUILD_SYNC) {
          char *sync_info = (char *)malloc(data_size);
          memcpy(sync_info, ptr, data_size);
          rbuf->build_sync_queue.push_back(sync_info);
        } else if (control == DISPATCH_META) {
          char *meta_info = (char *)malloc(data_size);
          memcpy(meta_info, ptr, data_size);
          rbuf->dispatch_meta_queue.push_back(meta_info);
        } else if (control == DISPATCH_ID) {
          char *dispatch_info = (char *)malloc(data_size);
          memcpy(dispatch_info, ptr, data_size);
          rbuf->dispatch_id_queue.push_back(dispatch_info);
        } else if (control == DISPATCH_NGH) {
          char *dispatch_info = (char *)malloc(data_size);
          memcpy(dispatch_info, ptr, data_size);
          rbuf->dispatch_ngh_queue.push_back(dispatch_info);
        } else if (control == SCALA_META) {
          char *scala_meta_info = (char *)malloc(data_size);
          memcpy(scala_meta_info, ptr, data_size);
          rbuf->scala_meta_queue.push_back(scala_meta_info);
        } else if (control == SCALA_NGH) {
          char *scala_ngh_info = (char *)malloc(data_size);
          memcpy(scala_ngh_info, ptr, data_size);
          rbuf->scala_ngh_queue.push_back(scala_ngh_info);
        } else if (control == CORE_INFO) {
          char *core_info = (char *)malloc(data_size);
          memcpy(core_info, ptr, data_size);
          rbuf->core_info_queue.push_back(core_info);
        } else if (control == QUERY) {
          char *new_query = (char *)malloc(data_size);
          memcpy(new_query, ptr, data_size);
          rbuf->migrate_queue.push_back(new_query);
        } else if (control == SYNC) {
          // printf("recv sync\n");
          char *sync = (char *)malloc(data_size);
          memcpy(sync, ptr, data_size);
          rbuf->sync_queue.push_back(sync);
        } else if (control == TASK) {
          // printf("recv task form m%u buf %u\n", m_id, buf_id);
          char *new_task = (char *)malloc(data_size);
          // TODO: this maybe optimized.
          memcpy(new_task, ptr, data_size);
          rbuf->recv_task.push_back(new_task);
        } else if (control == RESULT) {
          // printf("recv result\n");
          char *new_res = (char *)malloc(data_size);
          memcpy(new_res, ptr, data_size);
          rbuf->recv_result.push_back(new_res);
        } else if (control == NODE_RESULT) {
          // printf("recv NODE_RESULT\n");
          char *new_res = (char *)malloc(data_size);
          memcpy(new_res, ptr, data_size);
          rbuf->recv_node_res.push_back(new_res);
        } else if (control == NODE_SYNC) {
          // printf("recv NODE_SYNC \n");
          char *new_res = (char *)malloc(data_size);
          memcpy(new_res, ptr, data_size);
          rbuf->recv_node_sync.push_back(new_res);
        } else if (control == COMPUTE) {
          // printf("recv compute\n");
          char *new_cmp = (char *)malloc(data_size);
          memcpy(new_cmp, ptr, data_size);
          rbuf->recv_compute.push_back(new_cmp);
          // printf("recved compute\n");
        } else if (control == TERMINATION) {
          term[m_id] = true;
          // printf("recv m%u term, all term state:\n", m_id);
          // for (uint32_t m = 0; m < MACHINE_NUM; m++) printf("%d ",
          // term[m]); printf("\n");
          size_t offset = 0;
          size_t q_correct{0}, q_total{0}, q_query_load{0};
          double cumu_lat_us{0.0};
          memcpy(
              &q_correct, static_cast<char *>(ptr) + offset, sizeof(q_correct));
          offset += sizeof(q_correct);
          correct += q_correct;
          memcpy(&q_total, static_cast<char *>(ptr) + offset, sizeof(q_total));
          offset += sizeof(q_total);
          total += q_total;
          memcpy(
              &q_query_load, static_cast<char *>(ptr) + offset,
              sizeof(q_query_load));
          offset += sizeof(q_query_load);
          query_load += q_query_load;

          memcpy(
            &cumu_lat_us, static_cast<char *>(ptr) + offset, sizeof(cumu_lat_us));
          offset += sizeof(cumu_lat_us);
          query_lat_sum += cumu_lat_us;

          size_t recv_computation_cnt;
          memcpy(
            &recv_computation_cnt, static_cast<char *>(ptr) + offset, sizeof(size_t));
          offset += sizeof(recv_computation_cnt);
          all_computation_cnt += recv_computation_cnt;
        } else {
          printf("Error: Unexpected buff contorl flag %d.\n", control);
        }
        post_write_recv(m_id, true_wr_id);
        if (control != CLEAR) {
          // If is not clear state.
          send_write_release(m_id, buf_id);
        }
      }
      // Update buffer list.
    }
  }
  return ne;
}

void RdmaCommunication::send_write_release(int machine_id, uint32_t buf_id) {
  auto *rbuf = recv_write_buf.getLocal();
  rbuf->release_id_list[machine_id]->push_back(buf_id);
  // printf(
  //     "rbuf->release_id_list <== %d size %d\n", buf_id,
  //     rbuf->release_id_list[machine_id]->size());
  if (rbuf->release_id_list[machine_id]->size() >= RELEASE_BLOCK) {
    auto *sbuf = send_write_buf.getLocal();
    uint32_t num = 0;
    uint32_t send_buf_id = get_write_send_buf(machine_id, true, true);
    uint32_t *ptr = (uint32_t *)sbuf->buffer[machine_id][send_buf_id];
    // printf("rbuf send release id ==> ");
    while (rbuf->release_id_list[machine_id]->size() > 0) {
      uint32_t release_id = rbuf->release_id_list[machine_id]->front();
      rbuf->release_id_list[machine_id]->pop_front();
      // printf("%d ", release_id);
      ptr[num + 1] = release_id;
      num++;
    }
    // printf("\n");
    ptr[0] = num;
    post_write_send(machine_id, send_buf_id, (num + 1) * 4, RELEASE);
  }
}

void RdmaCommunication::clear_write_release() {
  auto *rbuf = recv_write_buf.getLocal();
  for (uint32_t machine_id = 0; machine_id < MACHINE_NUM; machine_id++) {
    if (rbuf->release_id_list[machine_id]->size() > 0) {
      // printf("[Clear release] clear release\n");

      auto *sbuf = send_write_buf.getLocal();
      uint32_t num = 0;
      uint32_t send_buf_id = get_write_send_buf(machine_id, true, false);
      uint32_t *ptr = (uint32_t *)sbuf->buffer[machine_id][send_buf_id];
      while (rbuf->release_id_list[machine_id]->size() > 0) {
        uint32_t release_id = rbuf->release_id_list[machine_id]->front();
        rbuf->release_id_list[machine_id]->pop_front();
        // printf(" %d\n", release_id);
        ptr[num + 1] = release_id;
        num++;
      }
      ptr[0] = num;
      // printf("%u to m%d bufid %d over\n", num, machine_id, send_buf_id);
      post_write_send(machine_id, send_buf_id, (num + 1) * 4, CLEAR);
    }
  }
}

void RdmaCommunication::com_read_init() {
  on_each([&](uint64 tid, uint64 total) {
    auto *ctx = rdma_ctx.getLocal();
    ctx->read_init(&rdma_param);
    auto *buf = read_buf.getLocal();
    for (int b = 0; b < MAX_CACHE_VEC; b++) {
      buf->buff_id_list.push_back(b);
    }
  });
}

char *RdmaCommunication::sync_read(
    int machine_id, size_t internal_id, uint32_t size) {
  auto *ctx = rdma_ctx.getLocal();
  // Use the general async post and poll.
  // WARN: internal id for now.
  std::vector<BufferCache> vecs{
      BufferCache(machine_id, internal_id, internal_id, -1, NULL)};
  post_read(vecs, size);
  poll_read();
  release_cache(vecs[0].buffer_id);
  return vecs[0].buffer_ptr;
}

/*
  read  wr_id:                <buffer_id>
  write wr_id: <machine_id> | <buffer_id>
  send  wr_id: <machine_id> | <buffer_id>
*/

int RdmaCommunication::post_read(
    std::vector<BufferCache> &vectors, size_t vec_size, uint32_t query_id) {
  auto *ctx = rdma_ctx.getLocal();
  auto *buf = read_buf.getLocal();

  // ctx->run_post_read(vectors, vec_size);
  // TODO: change to buffer queue.
  memset(ctx->wr_cnt, 0, sizeof(ctx->wr_cnt));
  std::vector<uint32_t> machines;
  ctx->post_num = vectors.size();
  // printf("READ offset size %d\n", vectors.size());
  for (BufferCache &vec : vectors) {
    uint32_t m = vec.machine_id;
    size_t offset = vec.internal_id * vec_size;
    assert(buf->buff_id_list.size() > 0);
    vec.buffer_id = buf->buff_id_list.front();
    buf->buff_id_list.pop_front();
    vec.buffer_ptr = (char *)ctx->read_to_ptr[vec.buffer_id];
    buf->bufcache_ptr[vec.buffer_id] = vec;
    // printf("postread buf_id %u vid %u\n", vec.buffer_id, vec.vector_id);
    size_t &r = ctx->wr_cnt[m];
    ctx->sge_list[vec.buffer_id].length = vec_size;
#ifdef COMM_PROFILE
    ctx->read_size += vec_size;
    ctx->read_cnt++;
#endif
    ctx->wr[m][r].sg_list = &ctx->sge_list[vec.buffer_id];
    ctx->wr[m][r].wr.rdma.remote_addr =
        ctx->remote_mr_msg[m].read_vaddr + offset;
    // Keep the query id in the high bits.  Without the cast, uint32_t
    // query_id overflows once qid >= 65536 and large query batches hang.
    ctx->wr[m][r].wr_id = (static_cast<uint64_t>(query_id) << 16) |
                          static_cast<uint64_t>(vec.buffer_id);
    ctx->wr[m][r].next = NULL;
    if (r > 0) {
      ctx->wr[m][r - 1].next = &ctx->wr[m][r];
    } else {
      machines.emplace_back(m);
    }
    r++;
    if (r >= MAX_READ_NUM) {
      printf("Error: Exceed the max read wr buffer.\n");
      abort();
    }
  }

  for (uint32_t m : machines) {
    // printf("READ machine %d\n", m);
    ctx->run_post_read(m);
  }

  return 0;
}

// TODO: add identifier
int RdmaCommunication::poll_read() {
  auto *ctx = rdma_ctx.getLocal();
  // ctx->run_poll_read(vbuf, sbuf);
  ctx->collect_num = 0;
  do {
    poll_send();
  } while (ctx->collect_num < ctx->post_num);

  return 0;
}

void RdmaCommunication::release_cache(int buffer_id) {
  // printf("release cache %d\n", buffer_id);
  auto *buf = read_buf.getLocal();
  buf->buff_id_list.push_back(buffer_id);
  // printf(
  //     "release cache %d buflist sz %u\n", buffer_id,
  //     buf->buff_id_list.size());
  return;
}

char *RdmaCommunication::fetch_vector(
    uint32_t internal_id, size_t machine_id, size_t vec_size) {
  auto *vec_buffer = read_buf.getLocal();
  // if (internal_id != vec_buffer->cur_buffer_id[machine_id]) {
  //   int ret = sync_read(machine_id, internal_id, vec_size);
  //   // update current buffer id
  //   vec_buffer->cur_buffer_id[machine_id] = internal_id;
  // }
  return sync_read(machine_id, internal_id, vec_size);
}

void RdmaCommunication::com_write_init() {
  on_each([&](uint64 tid, uint64 total) {
    auto *ctx = rdma_ctx.getLocal();
    auto *sbuf = send_write_buf.getLocal();
    ctx->write_init(&rdma_param);

    for (int m = 0; m < MACHINE_NUM; m++) {
      for (int buf_id = 0; buf_id < MAX_WRITE_NUM; buf_id++) {
        sbuf->buff_id_list[m]->push_back(buf_id);
      }
    }
    for (int m = 0; m < MACHINE_NUM; m++) {
      for (int buf_id = 0; buf_id < MAX_WRITE_NUM; buf_id++) {
        post_write_recv(m, buf_id);
      }
    }
  });
}

int RdmaCommunication::get_write_send_buf(
    int machine_id, bool send_release, bool pop) {
  auto *sbuf = send_write_buf.getLocal();
  // printf("try get send buf id ...\n");
  if (!send_release) {
    do {
      poll_send();
      poll_recv();  // Recv released buffer indicate by control.
      // reserve buffer for release write send
    } while (sbuf->buff_id_list[machine_id]->size() <= RELEASE_BLOCK);
  } else {
    if (sbuf->buff_id_list[machine_id]->size() == 0) {
      printf("Error: expect sbuf always avaliable.\n");
      exit(0);
    }
  }
  uint32_t buf_id = sbuf->buff_id_list[machine_id]->front();
  if (pop) {  // If not in clear state.
    sbuf->buff_id_list[machine_id]->pop_front();
    // printf(
    //     "sbuf pop ==> %d left size %d\n", buf_id,
    //     sbuf->buff_id_list[machine_id]->size());
  } else {
    // move form front to back
    sbuf->buff_id_list[machine_id]->pop_front();
    sbuf->buff_id_list[machine_id]->push_back(buf_id);
  }
  // printf(
  //     "get %d. left size %d\n", buf_id,
  //     sbuf->buff_id_list[machine_id]->size());
  return buf_id;
}

int RdmaCommunication::post_write_send(
    int machine_id, uint32_t buf_id, uint32_t size, ControlType control) {
  auto *ctx = rdma_ctx.getLocal();

  size_t offset = MAX_QUERYBUFFER_SIZE * buf_id;
// printf(
//     "[post send write] m %d, buf %d, size %d, control %d\n", machine_id,
//     buf_id, size, control);
#ifdef COMM_PROFILE
  ctx->write_size += size;
  ctx->write_cnt++;
#endif
  if (size > MAX_QUERYBUFFER_SIZE) {
    printf("Error: Exceed the max write buffer size.\n");
    abort();
  }
  ctx->run_post_write_send(
      machine_id, buf_id, offset, size, control << 16 | buf_id);
  return 0;
}

int RdmaCommunication::post_sg_write_send(
    int machine_id, uint32_t buf_id, std::vector<char *> &sg_ptr,
    std::vector<uint32_t> &sg_size, ControlType control) {
  auto *ctx = rdma_ctx.getLocal();
  size_t offset = MAX_QUERYBUFFER_SIZE * buf_id;
  // TODO: <<16 use macro
  ctx->run_post_sg_write_send(
      machine_id, buf_id, offset, sg_ptr, sg_size, control << 16 | buf_id);
  return 0;
}

int RdmaCommunication::post_write_recv(int machine_id, uint32_t buf_id) {
  // printf("post recv write buf %d\n", buf_id);
  auto *ctx = rdma_ctx.getLocal();
  ctx->run_post_write_recv(machine_id, buf_id);
  // printf("post recv write buf %d over\n", buf_id);
  return 0;
}

void RdmaCommunication::com_send_init() {
  // on_each([&](uint64 tid, uint64 total) {
  //   auto *ctx = rdma_ctx.getLocal();
  //   auto *buf = send_buf.getLocal();
  //   ctx->send_init(&rdma_param);

  //   for (int m = 0; m < MACHINE_NUM; m++) {
  //     for (int bufid = 0; bufid < MAX_RECV_NUM; bufid++) {
  //       buf->buff_id_list[m]->push_back(bufid);
  //     }
  //     for (int bufid = 0; bufid < MAX_RECV_NUM; bufid++) {
  //       ctx->run_post_recv(m, bufid, MAX_QUERYBUFFER_SIZE);
  //     }
  //   }
  // });
}

int RdmaCommunication::sync_send(int machine_id, size_t offset, uint32_t size) {
  auto *ctx = rdma_ctx.getLocal();
  return ctx->run_sync_send(machine_id, size);
}

int RdmaCommunication::sync_recv(uint32_t recv_num) {
  auto *ctx = rdma_ctx.getLocal();
  return ctx->run_sync_recv(recv_num);
}

int RdmaCommunication::post_send(
    int machine_id, size_t buff_id, uint32_t size) {
  auto *ctx = rdma_ctx.getLocal();
  return ctx->run_post_send(machine_id, buff_id, size);
}

int RdmaCommunication::post_recv(
    int machine_id, size_t buff_id, uint32_t size) {
  auto *ctx = rdma_ctx.getLocal();
  return ctx->run_post_recv(machine_id, buff_id, size);
}

void RdmaCommunication::send_term(
    int my_machine_id, size_t q_correct, size_t q_total, 
    size_t q_query_load, double cumu_lat_us, size_t computation_cnt) {
  // Broadcast term
  auto *buf = send_write_buf.getLocal();
  for (int m = 0; m < MACHINE_NUM; m++) {
    if (m == my_machine_id) {
      term[m] = true;
      correct += q_correct;
      total += q_total;
      query_load += q_query_load;
      continue;
    }
    uint32_t buf_id = get_write_send_buf(m);
    // Write end info
    size_t offset = 0;
    char *ptr = buf->buffer[m][buf_id];
    memcpy(static_cast<char *>(ptr) + offset, &q_correct, sizeof(q_correct));
    offset += sizeof(q_correct);
    memcpy(static_cast<char *>(ptr) + offset, &q_total, sizeof(q_total));
    offset += sizeof(q_total);
    memcpy(
        static_cast<char *>(ptr) + offset, &q_query_load, sizeof(q_query_load));
    offset += sizeof(q_query_load);
    memcpy(
      static_cast<char *>(ptr) + offset, &cumu_lat_us, sizeof(cumu_lat_us));
    offset += sizeof(cumu_lat_us);
    memcpy(
      static_cast<char *>(ptr) + offset, &computation_cnt, sizeof(computation_cnt));
    offset += sizeof(computation_cnt);

    // printf("q%d >> m[%d] buf%d\n", query->query_id, machine_id, buff_id);
    post_write_send(m, buf_id, offset, TERMINATION);
  }
  return;
}

bool RdmaCommunication::check_term() {
  bool all_term = true;
  for (int m = 0; m < MACHINE_NUM; m++) {
    all_term &= term[m];
  }
  return all_term;
}

void RdmaCommunication::reset_term() {
  for (int m = 0; m < MACHINE_NUM; m++) {
    term[m] = false;
  }
  correct = total = query_load = 0;
  query_lat_sum = 0.0;
  all_computation_cnt = 0;
}

void RdmaCommunication::reset_sendrecv() {
  for (int m = 0; m < MACHINE_NUM; m++) {
    auto *sbuf = send_write_buf.getLocal();
    // Collect the sent ack and release the position.
    do {
      poll_send();
      poll_recv();
      clear_write_release();
    } while (sbuf->buff_id_list[m]->size() < MAX_WRITE_NUM);
    // printf(
    //     "sbuf->buff_id_list[%d]->size() %d\n", m,
    //     sbuf->buff_id_list[m]->size());

    auto *rbuf = recv_write_buf.getLocal();
    assert(rbuf->buff_id_list[m]->size() == 0);
  }
}

void RdmaCommunication::post_partition_info(
    IndexParameter &index_param, uint32_t machine_id) {
  // For now, we just send the mera info.
  auto *buf = send_write_buf.getLocal();
  // Collect the sent ack and release the position.
  uint32_t buf_id = get_write_send_buf(machine_id);
  // New migration: no need for visit list.
  size_t len =
      index_param.serialize_info(buf->buffer[machine_id][buf_id], machine_id);
  post_write_send(machine_id, buf_id, len, PART_INFO);

  return;
}

void RdmaCommunication::poll_partition_info(IndexParameter &index_param) {
  auto *rbuf = recv_write_buf.getLocal();  // recv buffer

  for (;;) {
    poll_send();
    poll_recv();

    // Collect recved partition info.
    if (rbuf->part_info_queue.size() > 0) {
      char *part_info_ptr = rbuf->part_info_queue.front();
      rbuf->part_info_queue.pop_front();
      index_param.deserialize_info(part_info_ptr);  // impl
      // printf("[Recv migrate q%d]\n", new_query->query_id);
      free(part_info_ptr);

      std::cout << "Recv partition info :"
                << index_param.local_replica_base_file << std::endl;
      break;
    }
  }
}

void RdmaCommunication::build_sync() {
  uint32_t sync_cnt = 0;
  // send sync to all other machines.
  for (uint32_t m = 0; m < MACHINE_NUM; m++) {
    if (m == rdma_param.machine_id) continue;
    auto *buf = send_write_buf.getLocal();
    uint32_t buf_id = get_write_send_buf(m);
    memcpy(buf->buffer[m][buf_id], &rdma_param.machine_id, sizeof(uint32_t));
    post_write_send(m, buf_id, sizeof(uint32_t), BUILD_SYNC);
  }
  sync_cnt++;

  auto *rbuf = recv_write_buf.getLocal();  // recv buffer
  while (sync_cnt < MACHINE_NUM) {
    poll_send();
    poll_recv();
    // Collect recved partition info.
    // NOTE: sync_cnt may exceed MACHINE_NUM when sync frequent, so we need to
    // check sync_cnt.
    while (rbuf->build_sync_queue.size() > 0 && sync_cnt < MACHINE_NUM) {
      char *build_sync_ptr = rbuf->build_sync_queue.front();
      rbuf->build_sync_queue.pop_front();
      // printf("[Recv migrate q%d]\n", new_query->query_id);
      free(build_sync_ptr);
      sync_cnt++;
    }
  }
}

void RdmaCommunication::post_b2_start(
    uint32_t machine_id, uint32_t qid, char *query, size_t query_size,size_t ef) {
  auto *buf = send_write_buf.getLocal();
  uint32_t buf_id = get_write_send_buf(machine_id);
  size_t len = 0;
  memcpy(buf->buffer[machine_id][buf_id] + len, &query_size, sizeof(size_t));
  len += sizeof(size_t);
  memcpy(buf->buffer[machine_id][buf_id] + len, query, query_size);
  len += query_size;
  memcpy(buf->buffer[machine_id][buf_id] + len, &ef, sizeof(size_t));
  len += sizeof(size_t);
  memcpy(buf->buffer[machine_id][buf_id] + len, &qid, sizeof(uint32_t));
  len += sizeof(uint32_t);
  post_write_send(machine_id, buf_id, len, B2_START);
}

void RdmaCommunication::poll_b2_start(char *query_data, size_t &ef, uint32_t &qid) {
  auto *rbuf = recv_write_buf.getLocal();  // recv buffer

  for (;;) {
    poll_send();
    poll_recv();

    // Collect recved partition info.
    if (rbuf->b2_start_queue.size() > 0) {
      char *b2_start_ptr = rbuf->b2_start_queue.front();
      rbuf->b2_start_queue.pop_front();
      size_t ofs = 0;
      size_t query_size = 0;
      memcpy(&query_size, b2_start_ptr + ofs, sizeof(size_t));
      ofs += sizeof(size_t);
      memcpy(query_data, b2_start_ptr + ofs, query_size);
      ofs += query_size;
      memcpy(&ef, b2_start_ptr + ofs, sizeof(size_t));
      ofs += sizeof(size_t);
      memcpy(&qid, b2_start_ptr + ofs, sizeof(uint32_t));
      ofs += sizeof(uint32_t);
      free(b2_start_ptr);
      // std::cout << "Recv partition info :" << index_param.shard_base_file
      //           << std::endl;
      break;
    }
  }
}

// void RdmaCommunication::poll_b2_start(std::vector<uint32_t> &b2_query_queue, TaskManager<dist_t> &tman){
//   auto *rbuf = recv_write_buf.getLocal();  // recv buffer
//   poll_send();
//   poll_recv();

//   // Collect recved partition info.
//   while (rbuf->b2_start_queue.size() > 0) {
//     char *b2_start_ptr = rbuf->b2_start_queue.front();
//     rbuf->b2_start_queue.pop_front();
//     size_t ofs = 0;
//     size_t query_size = 0;
//     memcpy(&query_size, b2_start_ptr + ofs, sizeof(size_t));
//     ofs += sizeof(size_t);
//     // memcpy(query_data, b2_start_ptr + ofs, query_size);
//     ofs += query_size;
//     memcpy(&tman.global_query[query_id]->ef, b2_start_ptr + ofs, sizeof(size_t));
//     ofs += sizeof(size_t);
//     uint32_t qid;
//     memcpy(&qid, b2_start_ptr + ofs, sizeof(uint32_t));
//     ofs += sizeof(uint32_t);
//     free(b2_start_ptr);
//     // std::cout << "Recv partition info :" << index_param.shard_base_file
//     //           << std::endl;
//     b2_query_queue.push_back(qid);
//   }
// }


/**
 *
 */
void RdmaCommunication::swap_scala_index_meta_info(
    uint32_t machine_id, std::vector<size_t> &out_remote_ngh_num,
    std::vector<std::vector<size_t>> &in_remote_ngh_num) {
  for (uint32_t m = 0; m < MACHINE_NUM; m++) {
    if (m == machine_id) continue;
    printf("send meta to %d\n", m);
    auto *buf = send_write_buf.getLocal();
    uint32_t buf_id = get_write_send_buf(m);
    size_t ofs = 0;
    memcpy(buf->buffer[m][buf_id] + ofs, &machine_id, sizeof(uint32_t));
    ofs += sizeof(uint32_t);
    std::copy(
        out_remote_ngh_num.begin(), out_remote_ngh_num.end(),
        (size_t *)(buf->buffer[m][buf_id] + ofs));
    ofs += out_remote_ngh_num.size() * sizeof(size_t);
    post_write_send(m, buf_id, ofs, SCALA_META);
  }

  // poll meat
  auto *rbuf = recv_write_buf.getLocal();
  uint32_t meta_recv_cnt = 0;
  while (meta_recv_cnt < MACHINE_NUM - 1) {
    poll_send();
    poll_recv();
    while (rbuf->scala_meta_queue.size() > 0) {
      char *ptr = rbuf->scala_meta_queue.front();
      rbuf->scala_meta_queue.pop_front();
      size_t ofs = 0;
      uint32_t send_machine_id;
      memcpy(&send_machine_id, ptr + ofs, sizeof(uint32_t));
      ofs += sizeof(uint32_t);
      printf("recv meta from %d\n", send_machine_id);
      size_t in_ngh_num[MACHINE_NUM];
      memcpy(in_ngh_num, ptr + ofs, sizeof(in_ngh_num));
      ofs += sizeof(in_ngh_num);
      for (uint32_t m = 0; m < MACHINE_NUM; m++) {
        in_remote_ngh_num[send_machine_id].push_back(in_ngh_num[m]);
      }
      free(ptr);
      meta_recv_cnt++;
    }
  }
  printf("]]] swap over\n");

  return;
}

void RdmaCommunication::swap_scalagraph_remote_ngh(
    std::vector<char *> &out_remote_ngh_ptr,
    std::vector<size_t> &out_remote_ngh_num, char *recv_ptr,
    size_t *in_remote_ngh_start_ofs,
    std::vector<std::vector<size_t>> &in_remote_ngh_num) {
  for (uint32_t p = 0; p < MACHINE_NUM; p++) {
    if (p == rdma_param.machine_id) {
      for (uint32_t m = 0; m < MACHINE_NUM; m++) {
        if (p == m) continue;
        printf(
            "INFO: ngh send %d >>> %d size %llu\n", p, m,
            out_remote_ngh_num[m]);
        size_t send_cnt = 0;
        while (send_cnt < out_remote_ngh_num[m]) {
          auto *buf = send_write_buf.getLocal();
          uint32_t buf_id = get_write_send_buf(m);
          char *ptr = buf->buffer[m][buf_id];
          size_t offset = 0;
          memcpy(ptr + offset, &rdma_param.machine_id, sizeof(uint32_t));
          offset += sizeof(uint32_t);
          uint32_t max_trans_num = std::min(
              ((MAX_QUERYBUFFER_SIZE - offset - sizeof(uint32_t)) /
               sizeof(uint32_t)),
              out_remote_ngh_num[m] - send_cnt);
          memcpy(ptr + offset, &max_trans_num, sizeof(uint32_t));
          offset += sizeof(uint32_t);
          memcpy(
              ptr + offset,
              out_remote_ngh_ptr[m] + (size_t)send_cnt * sizeof(uint32_t),
              max_trans_num * sizeof(uint32_t));
          offset += max_trans_num * sizeof(uint32_t);
          post_write_send(m, buf_id, offset, SCALA_NGH);
          send_cnt += max_trans_num;
        }
        printf("INFO: ngh send %d >>> %d send %llu over\n", p, m, send_cnt);
      }
    } else {
      auto *rbuf = recv_write_buf.getLocal();
      uint32_t ngh_recv_cnt = 0;
      uint32_t expect_num = in_remote_ngh_num[p][rdma_param.machine_id];
      printf(
          "INFO: ngh recv %d <<< %d expect %llu\n", rdma_param.machine_id, p,
          expect_num);
      while (ngh_recv_cnt < expect_num) {
        poll_send();
        poll_recv();
        while (rbuf->scala_ngh_queue.size() > 0 && ngh_recv_cnt < expect_num) {
          char *ptr = rbuf->scala_ngh_queue.front();
          rbuf->scala_ngh_queue.pop_front();
          size_t ofs = 0;
          uint32_t send_machine_id;
          memcpy(&send_machine_id, ptr + ofs, sizeof(uint32_t));
          ofs += sizeof(uint32_t);
          uint32_t recv_num = 0;
          memcpy(&recv_num, ptr + ofs, sizeof(recv_num));
          ofs += sizeof(recv_num);
          memcpy(
              recv_ptr + in_remote_ngh_start_ofs[p] +
                  ngh_recv_cnt * sizeof(uint32_t),
              ptr + ofs, recv_num * sizeof(uint32_t));
          ngh_recv_cnt += recv_num;
          free(ptr);
        }
      }
      printf(
          "INFO: ngh recv %d <<< %d num %llu over\n", rdma_param.machine_id, p,
          ngh_recv_cnt);
    }
    build_sync();
    printf("--- Build sync ---\n");
  }

  return;
}
