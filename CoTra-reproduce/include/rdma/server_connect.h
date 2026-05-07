#pragma once

#include <assert.h>
#include <infiniband/verbs.h>
#include <libmemcached/memcached.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include <functional>
#include <string>
#include <thread>
#include <vector>

#include "rdma/rdma_context.h"
#include "rdma/rdma_param.h"

struct ExchangeMeta {
  MessageContext thd_msg[MAX_THREAD_NUM];
  uint32_t machine_id;
} __attribute__((packed));

class ServerConnect {
 private:
  static const char *server_prefix;
  std::string machine_id_prefix{"Machine-id"};
  bool states[MACHINE_NUM];

  std::vector<std::string> serverList;

  std::string set_key(uint16_t remote_id) {
    return std::to_string(get_my_id()) + "-" + std::to_string(remote_id);
  }

  std::string get_key(uint16_t remote_id) {
    return std::to_string(remote_id) + "-" + std::to_string(get_my_id());
  }

  // order id map to machine id.
  std::string set_machine_key() {
    return machine_id_prefix + "-" + std::to_string(get_my_id());
  }

  std::string get_machine_key(uint16_t remote_id) {
    return machine_id_prefix + "-" + std::to_string(remote_id);
  }

  void init();

  void init_route();

  void show_states();

  void setDataToRemote(uint16_t remote_id);
  void setDataFromRemote(uint16_t remote_id, ExchangeMeta *remoteMeta);

  static const char *machine_num_key;

  uint16_t max_machine;
  uint16_t cur_machine_num;
  uint16_t my_id;       // memcached id
  uint16_t machine_id;  // machine id
  std::string my_ip;

  memcached_st *memc;

 protected:
  bool connect_mc(std::string config_file);
  bool disconnect_mc();
  void connect_machine();
  void add_machine();
  bool exchange_meta(uint16_t remote_id);

 public:
  ExchangeMeta local_meta[MACHINE_NUM];
  PerThreadStorage<RdmaContext> &rdma_ctx;

  ServerConnect(
      RdmaParameter &rdma_param, PerThreadStorage<RdmaContext> &_rdma_ctx)
      : max_machine(MACHINE_NUM),
        cur_machine_num(0),
        memc(NULL),
        machine_id(rdma_param.machine_id),
        rdma_ctx(_rdma_ctx) {
    init();

    if (!connect_mc(rdma_param.ip_config_file)) {
      return;
    }
    add_machine();

    // Set local machine.
    setDataToRemote(machine_id);
    setDataFromRemote(get_my_id(), &local_meta[machine_id]);

    connect_machine();
    init_route();
  }
  ~ServerConnect();

  uint16_t get_my_id() const { return this->my_id; }
  uint32_t get_machine_id() const { return this->machine_id; }
  uint16_t get_machine_num() const { return this->max_machine; }

  std::string get_my_ip() const { return this->my_ip; }

  void mc_set(const char *key, uint32_t klen, const char *val, uint32_t vlen);
  char *mc_get(const char *key, uint32_t klen, size_t *v_size = nullptr);
  uint64_t mc_fetch_add(const char *key, uint32_t klen);

  void barrier(const std::string &barrierKey);
};
