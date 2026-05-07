#include "rdma/server_connect.h"

#include <fstream>
#include <iostream>
#include <random>

const char *ServerConnect::server_prefix = "SPre";

void ServerConnect::show_states() {
  std::cout << "Connect ";
  for (bool state : states) {
    std::cout << (state ? "√ " : "× ");
  }
  std::cout << std::flush;
}

void ServerConnect::init() {
  std::cout << "Machine ";
  for (size_t i = 0; i < MACHINE_NUM; ++i) {
    std::cout << i << " ";
  }
  std::cout << std::endl;
  memset(states, 0, sizeof(states));
  show_states();
}

bool ServerConnect::exchange_meta(uint16_t remote_id) {
  // get the machine id
  std::string key = get_machine_key(remote_id);
  uint16_t m = *mc_get(key.c_str(), key.size());
  setDataToRemote(m);

  std::string setK = set_key(remote_id);
  mc_set(
      setK.c_str(), setK.size(), (char *)(&local_meta[m]),
      sizeof(local_meta[m]));

  std::string getK = get_key(remote_id);
  ExchangeMeta *remoteMeta = (ExchangeMeta *)mc_get(getK.c_str(), getK.size());

  setDataFromRemote(remote_id, remoteMeta);

  free(remoteMeta);
  return true;
}

void ServerConnect::setDataToRemote(uint16_t m) {
  local_meta[m].machine_id = machine_id;
  for (int t = 0; t < getActiveThreads(); t++) {
    auto *ctx = rdma_ctx.getRemote(t);
    local_meta[m].thd_msg[t] = ctx->local_mr_msg[m];
  }
}

void ServerConnect::setDataFromRemote(
    uint16_t remote_id, ExchangeMeta *remoteMeta) {
  // printf("recv remote msg: %s\n", remoteMeta->msg);
  uint32_t m = remoteMeta->machine_id;
  for (int t = 0; t < getActiveThreads(); t++) {
    auto *ctx = rdma_ctx.getRemote(t);
    ctx->remote_mr_msg[m] = remoteMeta->thd_msg[t];
    ctx->remote_mr_msg[m].gid_index = ctx->local_mr_msg[m].gid_index;
  }
  states[m] = true;
  std::cout << "\r";
  show_states();
}

void ServerConnect::init_route() {
  std::cout << "\n";
  std::string k =
      std::string(server_prefix) + std::to_string(this->get_my_id());
  mc_set(k.c_str(), k.size(), get_my_ip().c_str(), get_my_ip().size());
}

void ServerConnect::barrier(const std::string &barrierKey) {
  std::string key = std::string("barrier-") + barrierKey;
  if (this->get_my_id() == 0) {
    mc_set(key.c_str(), key.size(), "0", 1);
  }
  mc_fetch_add(key.c_str(), key.size());
  while (true) {
    uint64_t v = std::stoull(mc_get(key.c_str(), key.size()));
    if (v == this->get_machine_num()) {
      return;
    }
  }
}

std::string trim(const std::string &s) {
  std::string res = s;
  if (!res.empty()) {
    res.erase(0, res.find_first_not_of(" "));
    res.erase(res.find_last_not_of(" ") + 1);
  }
  return res;
}

const char *ServerConnect::machine_num_key = "serverNum";

ServerConnect::~ServerConnect() { disconnect_mc(); }

bool ServerConnect::connect_mc(std::string config_file) {
  memcached_server_st *servers = NULL;
  memcached_return rc;

  // std::ifstream conf("../scripts/memcached.conf");
  std::ifstream conf(config_file);

  if (!conf) {
    fprintf(stderr, "can't open memcached.conf\n");
    return false;
  }

  std::string addr, port;
  std::getline(conf, addr);
  std::getline(conf, port);

  memc = memcached_create(NULL);
  servers = memcached_server_list_append(
      servers, trim(addr).c_str(), std::stoi(trim(port)), &rc);
  rc = memcached_server_push(memc, servers);

  if (rc != MEMCACHED_SUCCESS) {
    fprintf(stderr, "Counld't add server:%s\n", memcached_strerror(memc, rc));
    sleep(1);
    return false;
  }

  memcached_behavior_set(memc, MEMCACHED_BEHAVIOR_BINARY_PROTOCOL, 1);
  return true;
}

bool ServerConnect::disconnect_mc() {
  if (memc) {
    memcached_quit(memc);
    memcached_free(memc);
    memc = NULL;
  }
  return true;
}

void ServerConnect::add_machine() {
  memcached_return rc;
  uint64_t serverNum;

  while (true) {
    rc = memcached_increment(
        memc, machine_num_key, strlen(machine_num_key), 1, &serverNum);
    if (rc == MEMCACHED_SUCCESS) {
      my_id = serverNum - 1;

      std::string id_k = set_machine_key();
      mc_set(
          id_k.c_str(), id_k.size(), (char *)(&machine_id), sizeof(machine_id));
      // printf("I am server %d real machine id: %u\n", my_id, machine_id);
      return;
    }
    fprintf(
        stderr, "Server %d Counld't incr value and get ID: %s, retry...\n",
        my_id, memcached_strerror(memc, rc));
    usleep(10000);
  }
}

void ServerConnect::connect_machine() {
  size_t l;
  uint32_t flags;
  memcached_return rc;

  while (cur_machine_num < max_machine) {
    char *serverNumStr = memcached_get(
        memc, machine_num_key, strlen(machine_num_key), &l, &flags, &rc);
    if (rc != MEMCACHED_SUCCESS) {
      fprintf(
          stderr, "Server %d Counld't get serverNum: %s, retry\n", my_id,
          memcached_strerror(memc, rc));
      continue;
    }
    uint32_t serverNum = atoi(serverNumStr);
    free(serverNumStr);

    // /connect server K
    for (size_t k = cur_machine_num; k < serverNum; ++k) {
      if (k != my_id) {
        exchange_meta(k);
        // printf("I connect server %zu\n", k);
      }
    }
    cur_machine_num = serverNum;
  }
}

void ServerConnect::mc_set(
    const char *key, uint32_t klen, const char *val, uint32_t vlen) {
  memcached_return rc;
  while (true) {
    rc = memcached_set(memc, key, klen, val, vlen, (time_t)0, (uint32_t)0);
    if (rc == MEMCACHED_SUCCESS) {
      break;
    }
    usleep(400);
  }
}

char *ServerConnect::mc_get(const char *key, uint32_t klen, size_t *v_size) {
  size_t l;
  char *res;
  uint32_t flags;
  memcached_return rc;

  while (true) {
    res = memcached_get(memc, key, klen, &l, &flags, &rc);
    if (rc == MEMCACHED_SUCCESS) {
      break;
    }
    usleep(400 * my_id);
  }

  if (v_size != nullptr) {
    *v_size = l;
  }

  return res;
}

uint64_t ServerConnect::mc_fetch_add(const char *key, uint32_t klen) {
  uint64_t res;
  while (true) {
    memcached_return rc = memcached_increment(memc, key, klen, 1, &res);
    if (rc == MEMCACHED_SUCCESS) {
      return res;
    }
    usleep(10000);
  }
}
