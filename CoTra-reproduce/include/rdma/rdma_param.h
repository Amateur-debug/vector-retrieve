#pragma once
#include <getopt.h>
#include <infiniband/verbs.h>
#include <inttypes.h>
#include <malloc.h>
#include <math.h>
#include <omp.h>
#include <rdma/rdma_cma.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/shm.h>
#include <unistd.h>

#include <iostream>

#include "coromem/include/utils.h"
#include "get_clock.h"
#include "rdma_config.h"

#define RESULT_LINE                                                            \
  "--------------------------------------------------------------------------" \
  "-------------\n"
#define RESULT_FMT_LAT                                                     \
  " #bytes #iterations    t_min[usec]    t_max[usec]  t_typical[usec]    " \
  "t_avg[usec]    t_stdev[usec]   99"                                      \
  "%"                                                                      \
  " percentile[usec]   99.9"                                               \
  "%"                                                                      \
  " percentile[usec] "

/* Result print format for latency tests. */
#define REPORT_FMT_LAT                                               \
  " %-7lu %" PRIu64                                                  \
  "          %-7.2f        %-7.2f      %-7.2f  	       %-7.2f     	" \
  "%-7.2f		%-7.2f 		%-7.2f"

#define REPORT_EXT_CPU_UTIL "	    %-3.2f\n"

#define REPORT_EXT "\n"
#define RESULT_EXT "\n"

/* Macro to define the buffer size (according to "Nahalem" chip set).
 * for small message size (under 4K) , we allocate 4K buffer , and the RDMA
 * write verb will write in cycle on the buffer. this improves the BW in
 * "Nahalem" systems.
 */
#define BUFF_SIZE(size, cycle_buffer) \
  ((size < cycle_buffer) ? (cycle_buffer) : (size))

#define ROUND_UP(value, alignment)        \
  (((value) % (alignment) == 0) ? (value) \
                                : ((alignment) * ((value) / (alignment) + 1)))

#define ALLOCATE(var, type, size)                                \
  {                                                              \
    if ((var = (type *)malloc(sizeof(type) * (size))) == NULL) { \
      fprintf(stderr, " Cannot Allocate\n");                     \
      exit(1);                                                   \
    }                                                            \
  }

/* Macro that defines the address where we write in RDMA.
 * If message size is smaller then CACHE_LINE size then we write in CACHE_LINE
 * jumps.
 */
#define INC(size, cache_line_size)                            \
  ((size > cache_line_size) ? ROUND_UP(size, cache_line_size) \
                            : (cache_line_size))

static const char *portStates[] = {"Nop",   "Down", "Init",
                                   "Armed", "",     "Active Defer"};

/* The type of the device */
enum ctx_device {
  DEVICE_ERROR = -1,
  UNKNOWN = 0,
  CONNECTX4 = 10,
  CONNECTX4LX = 11,
  CONNECTX5 = 15,
  CONNECTX5EX = 16,
  CONNECTX6 = 17,
  CONNECTX6DX = 18,
  CONNECTX6LX = 25,
  CONNECTX7 = 26,
  CONNECTX8 = 31
};

typedef enum { LEADER, MEMBER, UNCHOSEN } MachineType;
typedef enum { SEND, WRITE, WRITE_IMM, READ, ATOMIC } VerbType;
typedef enum {
  B2_START,       // for B2
  B2_END,         // for B2
  PART_INFO,      // for build
  BUILD_SYNC,     // for build
  DISPATCH_META,  // for build
  DISPATCH_ID,    // for build
  DISPATCH_NGH,   // for build
  SCALA_META,     // for scala
  SCALA_NGH,      // for scala
  TASK,
  RESULT,
  NODE_RESULT,  // for node task
  NODE_SYNC,    // for node task
  COMPUTE,
  QUERY,
  CORE_INFO,
  SCHEDULE,
  SYNC,
  RELEASE,
  TERMINATION,
  CLEAR
} ControlType;

class RdmaParameter {
 public:
  int port;
  char *ib_devname;
  uint8_t ib_port;
  int mtu;
  enum ibv_mtu curr_mtu;
  uint64_t iters;
  int8_t link_type;
  enum ibv_transport_type transport_type;
  int gid_index;
  int inline_size;
  int out_reads;
  int pkey_index;
  MachineType machine;
  int ai_family;
  int sockfd[MACHINE_NUM];
  int cache_line_size;
  struct memory_ctx *(*memory_create)();
  uint8_t sl;
  uint8_t qp_timeout;
  int cpu_freq_f;
  cycles_t *tposted;
  cycles_t *tcompleted;
  int thread_num;
  int machine_num;
  int machine_id;

  std::string ip_config_file;
  std::vector<std::string> machine_name;

  RdmaParameter();
  RdmaParameter(commandLine &cmd);
  int parser(commandLine &cmd);
  void print_para();
};

static int get_cache_line_size();
const char *link_layer_str(int8_t link_layer);
enum ctx_device ib_dev_name(struct ibv_context *context);
static void ctx_set_max_inline(
    struct ibv_context *context, RdmaParameter *user_param);
static int get_device_max_reads(
    struct ibv_context *context, RdmaParameter *user_param);
static int ctx_set_out_reads(
    struct ibv_context *context, RdmaParameter *user_param);
static int set_link_layer(struct ibv_context *context, RdmaParameter *params);
static int ctx_chk_pkey_index(struct ibv_context *context, int pkey_idx);
int check_link(struct ibv_context *context, RdmaParameter *user_param);
static int cycles_compare(const void *aptr, const void *bptr);
static inline cycles_t get_median(int n, cycles_t *delta);
void print_report_lat(RdmaParameter *user_param);
