#include "rdma/rdma_param.h"

#include "rdma/ip_config.h"

RdmaParameter::RdmaParameter() {
  port = DEF_PORT;
  ib_port = DEF_IB_PORT;
  mtu = 0;
  iters = DEF_ITERS;
  link_type = LINK_UNSPEC;
  gid_index = DEF_GID_INDEX;
  inline_size = DEF_INLINE;
  pkey_index = 0;
  ai_family = AF_INET;
  cache_line_size = get_cache_line_size();
  // memory_create = host_memory_create;
  sl = 0;
  qp_timeout = DEF_QP_TIME;
  cpu_freq_f = ON;
  ib_devname = NULL;
  thread_num = 1;
  machine_num = MACHINE_NUM;
  machine_id = 0;
}

RdmaParameter::RdmaParameter(commandLine &cmd) : RdmaParameter() {
  parser(cmd);
}

int RdmaParameter::parser(commandLine &cmd) {
  int c, size_len;
  char *server_ip = NULL;
  char *client_ip = NULL;
  char *not_int_ptr = NULL;

  // get ip config file
  ip_config_file = cmd.getOptionValue("--config_file", "none");

  port = cmd.getOptionIntValue("-p", DEF_PORT);
  auto ib_devname_str = cmd.getOptionValue("-d", "mlx4_0");
  ALLOCATE(ib_devname, char, (ib_devname_str.size() + 1));
  strcpy(ib_devname, ib_devname_str.c_str());

  machine_num = MACHINE_NUM;
  // machine_id = cmd.getOptionIntValue("-m", 0);
  machine_id = get_machine_id(ip_config_file);
  machine_name = get_machine_name(ip_config_file);

  // Set default thread number to max thread to avoid the potential error
  // caused by change of thread number.
  thread_num = cmd.getOptionIntValue("-t", 1);
  printf("qp num per machine = %d\n", machine_num);

  machine = machine_id == 0 ? LEADER : MEMBER;
  return 0;
}

void RdmaParameter::print_para() {
  printf("port            \t%d\n", port);
  printf("ib_port         \t%d\n", ib_port);
  printf("mtu             \t%d\n", mtu);
  printf("curr_mtu        \t%d\n", curr_mtu);
  printf("iters           \t%d\n", iters);
  printf("link_type       \t%d\n", link_type);
  printf("gid_index       \t%d\n", gid_index);
  printf("inline_size     \t%d\n", inline_size);
  printf("out_reads       \t%d\n", out_reads);
  printf("pkey_index      \t%d\n", pkey_index);
  printf("ai_family       \t%d\n", ai_family);
  printf("sockfd          \t%d\n", sockfd);
  printf("cache_line_size \t%d\n", cache_line_size);
  printf("sl              \t%d\n", sl);
  printf("qp_timeout      \t%d\n", qp_timeout);
}

static int get_cache_line_size() {
  int size = 0;
#if !defined(__FreeBSD__)
  size = sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
  if (size == 0) {
#if defined(__sparc__) && defined(__arch64__)
    char *file_name = "/sys/devices/system/cpu/cpu0/l2_cache_line_size";
#else
    std::string file_name =
        "/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size";
#endif

    FILE *fp;
    char line[10];
    fp = fopen(file_name.c_str(), "r");
    if (fp == NULL) {
      return DEF_CACHE_LINE_SIZE;
    }
    if (fgets(line, 10, fp) != NULL) {
      size = atoi(line);
    }
    fclose(fp);
  }
#endif
  // cppcheck-suppress knownConditionTrueFalse
  if (size <= 0) size = DEF_CACHE_LINE_SIZE;

  return size;
}

const char *link_layer_str(int8_t link_layer) {
  switch (link_layer) {
    case IBV_LINK_LAYER_UNSPECIFIED:
    case IBV_LINK_LAYER_INFINIBAND:
      return "IB";
    case IBV_LINK_LAYER_ETHERNET:
      return "Ethernet";
    default:
      return "Unknown";
  }
}

enum ctx_device ib_dev_name(struct ibv_context *context) {
  enum ctx_device dev_fname = UNKNOWN;
  struct ibv_device_attr attr;

  // Get max memoery region register size.
  // printf("max max_mr_size: %llu\n", attr.max_mr_size);

  if (ibv_query_device(context, &attr)) {
    dev_fname = DEVICE_ERROR;
  } else {
    // coverity[uninit_use]
    switch (attr.vendor_part_id) {
      case 4115:
        dev_fname = CONNECTX4;
        break;
      case 4116:
        dev_fname = CONNECTX4;
        break;
      case 4117:
        dev_fname = CONNECTX4LX;
        break;
      case 4118:
        dev_fname = CONNECTX4LX;
        break;
      case 4119:
        dev_fname = CONNECTX5;
        break;
      case 4120:
        dev_fname = CONNECTX5;
        break;
      case 4121:
        dev_fname = CONNECTX5EX;
        break;
      case 4122:
        dev_fname = CONNECTX5EX;
        break;
      case 4123:
        dev_fname = CONNECTX6;
        break;
      case 4124:
        dev_fname = CONNECTX6;
        break;
      case 4125:
        dev_fname = CONNECTX6DX;
        break;
      case 4127:
        dev_fname = CONNECTX6LX;
        break;
      case 4129:
        dev_fname = CONNECTX7;
        break;
      case 4131:
        dev_fname = CONNECTX8;
        break;
      default:
        dev_fname = UNKNOWN;
    }
  }

  return dev_fname;
}

static void ctx_set_max_inline(
    struct ibv_context *context, RdmaParameter *rdma_param) {
  enum ctx_device current_dev = ib_dev_name(context);
  rdma_param->inline_size = DEF_INLINE_WRITE;

  return;
}

static int get_device_max_reads(
    struct ibv_context *context, RdmaParameter *rdma_param) {
  struct ibv_device_attr attr;
  int max_reads = 0;

  if (!max_reads && !ibv_query_device(context, &attr)) {
    // coverity[uninit_use]
    max_reads = attr.max_qp_rd_atom;
  }
  return max_reads;
}

static int ctx_set_out_reads(
    struct ibv_context *context, RdmaParameter *rdma_param) {
  int max_reads = 0;
  int num_user_reads = rdma_param->out_reads;

  max_reads = get_device_max_reads(context, rdma_param);

  printf("max out read num: %d\n", max_reads);

  if (num_user_reads > max_reads) {
    printf(RESULT_LINE);
    fprintf(
        stderr, " Number of outstanding reads is above max = %d\n", max_reads);
    fprintf(stderr, " Changing to that max value\n");
    num_user_reads = max_reads;
  } else if (num_user_reads <= 0) {
    num_user_reads = max_reads;
  }

  return num_user_reads;
}

static int set_link_layer(struct ibv_context *context, RdmaParameter *params) {
  struct ibv_port_attr port_attr;
  int8_t curr_link = params->link_type;

  if (ibv_query_port(context, params->ib_port, &port_attr)) {
    fprintf(stderr, " Unable to query port %d attributes\n", params->ib_port);
    return FAILURE;
  }

  if (curr_link == LINK_UNSPEC) {
    // coverity[uninit_use]
    params->link_type = port_attr.link_layer;
  }

  if (port_attr.state != IBV_PORT_ACTIVE) {
    fprintf(
        stderr, " Port number %d state is %s\n", params->ib_port,
        portStates[port_attr.state]);
    return FAILURE;
  }

  if (strcmp("Unknown", link_layer_str(params->link_type)) == 0) {
    fprintf(stderr, "Link layer on port %d is Unknown\n", params->ib_port);
    return FAILURE;
  }
  return SUCCESS;
}

static int ctx_chk_pkey_index(struct ibv_context *context, int pkey_idx) {
  int idx = 0;
  struct ibv_device_attr attr;

  if (!ibv_query_device(context, &attr)) {
    // coverity[uninit_use]
    if (pkey_idx > attr.max_pkeys - 1) {
      printf(RESULT_LINE);
      fprintf(
          stderr, " Specified PKey Index, %i, greater than allowed max, %i\n",
          pkey_idx, attr.max_pkeys - 1);
      fprintf(stderr, " Changing to 0\n");
      idx = 0;
    } else
      idx = pkey_idx;
  } else {
    fprintf(stderr, " Unable to validata PKey Index, changing to 0\n");
    idx = 0;
  }

  return idx;
}

int check_link(struct ibv_context *context, RdmaParameter *rdma_param) {
  rdma_param->transport_type = context->device->transport_type;
  if (set_link_layer(context, rdma_param) == FAILURE) {
    fprintf(stderr, " Couldn't set the link layer\n");
    return FAILURE;
  }

  if (rdma_param->link_type == IBV_LINK_LAYER_ETHERNET &&
      rdma_param->gid_index == -1) {
    rdma_param->gid_index = 0;
  }

  /* Compute Max inline size with pre found statistics values */
  ctx_set_max_inline(context, rdma_param);
  rdma_param->out_reads = ctx_set_out_reads(context, rdma_param);

  if (rdma_param->pkey_index > 0)
    rdma_param->pkey_index =
        ctx_chk_pkey_index(context, rdma_param->pkey_index);

  return SUCCESS;
}

static int cycles_compare(const void *aptr, const void *bptr) {
  const cycles_t *a = (cycles_t *)aptr;
  const cycles_t *b = (cycles_t *)bptr;
  if (*a < *b) return -1;
  if (*a > *b) return 1;

  return 0;
}

static inline cycles_t get_median(int n, cycles_t *delta) {
  // const cycles_t *delta = (cycles_t *)delta;
  if ((n - 1) % 2)
    return (delta[n / 2] + delta[n / 2 - 1]) / 2;
  else
    return delta[n / 2];
}

void print_report_lat(RdmaParameter *rdma_param) {
  int i;
  int rtt_factor;
  double cycles_to_units, cycles_rtt_quotient;
  cycles_t median;
  cycles_t *delta = NULL;
  const char *units;
  double latency, stdev, average_sum = 0, average, stdev_sum = 0;
  int iters_99, iters_99_9;
  int measure_cnt;

  measure_cnt = rdma_param->iters - 1;
  rtt_factor = 1;
  ALLOCATE(delta, cycles_t, measure_cnt);

  cycles_to_units = get_cpu_mhz(rdma_param->cpu_freq_f);
  units = "usec";

  for (i = 0; i < measure_cnt; ++i) {
    delta[i] = rdma_param->tposted[i + 1] - rdma_param->tposted[i];
  }
  cycles_rtt_quotient = cycles_to_units * rtt_factor;

  qsort(delta, measure_cnt, sizeof *delta, cycles_compare);
  measure_cnt = measure_cnt - LAT_MEASURE_TAIL;
  median = get_median(measure_cnt, delta);

  /* calcualte average sum on sorted array*/
  for (i = 0; i < measure_cnt; ++i)
    average_sum += (delta[i] / cycles_rtt_quotient);

  average = average_sum / measure_cnt;

  /* Calculate stdev by variance*/
  for (i = 0; i < measure_cnt; ++i) {
    int temp_var = average - (delta[i] / cycles_rtt_quotient);
    int pow_var = pow(temp_var, 2);
    stdev_sum += pow_var;
  }

  latency = median / cycles_rtt_quotient;
  stdev = sqrt(stdev_sum / measure_cnt);
  iters_99 = ceil((measure_cnt) * 0.99);
  iters_99_9 = ceil((measure_cnt) * 0.999);

  // printf("%lf\n", average);
  printf(
      REPORT_FMT_LAT, 100, rdma_param->iters, delta[0] / cycles_rtt_quotient,
      delta[measure_cnt] / cycles_rtt_quotient, latency, average, stdev,
      delta[iters_99] / cycles_rtt_quotient,
      delta[iters_99_9] / cycles_rtt_quotient);
  printf(REPORT_EXT);

  free(delta);
}
