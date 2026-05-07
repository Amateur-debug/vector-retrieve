#include "hwtopo.h"

#include <numa.h>
#include <numaif.h>
#include <sched.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>

std::vector<int> parseCPUList(const std::string &line) {
  std::vector<int> vals;

  size_t current;
  size_t next = -1;
  try {
    do {
      current = next + 1;
      next = line.find_first_of(',', current);
      auto buf = line.substr(current, next - current);
      if (!buf.empty()) {
        size_t dash = buf.find_first_of('-', 0);
        if (dash != std::string::npos) {  // range
          auto first = buf.substr(0, dash);
          auto second = buf.substr(dash + 1, std::string::npos);
          unsigned b = std::stoi(first.data());
          unsigned e = std::stoi(second.data());
          while (b <= e) {
            vals.push_back(b++);
          }
        } else {  // singleton
          vals.push_back(std::stoi(buf.data()));
        }
      }
    } while (next != std::string::npos);
  } catch (const std::invalid_argument &) {
    return std::vector<int>{};
  } catch (const std::out_of_range &) {
    return std::vector<int>{};
  }

  return vals;
}

bool operator<(const cpuinfo &lhs, const cpuinfo &rhs) {
  if (lhs.smt != rhs.smt) return lhs.smt < rhs.smt;
  if (lhs.physid != rhs.physid) return lhs.physid < rhs.physid;
  if (lhs.coreid != rhs.coreid) return lhs.coreid < rhs.coreid;
  return lhs.proc < rhs.proc;
}

unsigned getNumaNode(cpuinfo &c) {
  static bool warnOnce = false;
  static bool numaAvail = false;

  if (!warnOnce) {
    warnOnce = true;
    numaAvail = numa_available() >= 0;
    numaAvail = numaAvail && numa_num_configured_nodes() > 0;
    if (!numaAvail) {
      printf(
          "Numa support configured but not present at runtime.\n"
          "Assuming numa topology matches socket topology.");
    }
  }

  if (!numaAvail) return c.physid;
  int i = numa_node_of_cpu(c.proc);
  if (i < 0) {
    printf("failed finding numa node for %u\n", c.proc);
    exit(-1);
  }
  return i;
}

std::vector<cpuinfo> parseCPUInfo() {
  std::vector<cpuinfo> vals;

  const int len = 1024;
  std::array<char, len> line;

  std::ifstream procInfo("/proc/cpuinfo");
  if (!procInfo) {
    printf("failed opening /proc/cpuinfo\n");
    exit(-1);
  }

  int cur = -1;
  while (true) {
    procInfo.getline(line.data(), len);
    if (!procInfo) break;
    int num;
    if (sscanf(line.data(), "processor : %d", &num) == 1) {
      assert(cur < num);
      cur = num;
      vals.resize(cur + 1);
      vals.at(cur).proc = num;
    } else if (sscanf(line.data(), "physical id : %d", &num) == 1) {
      vals.at(cur).physid = num;
    } else if (sscanf(line.data(), "siblings : %d", &num) == 1) {
      vals.at(cur).sib = num;
    } else if (sscanf(line.data(), "core id : %d", &num) == 1) {
      vals.at(cur).coreid = num;
    } else if (sscanf(line.data(), "cpu cores : %d", &num) == 1) {
      vals.at(cur).cpucores = num;
    }
  }

  for (auto &c : vals) c.numaNode = getNumaNode(c);

  return vals;
}

unsigned countSockets(const std::vector<cpuinfo> &info) {
  std::set<unsigned> pkgs;
  for (auto &c : info) pkgs.insert(c.physid);
  return pkgs.size();
}

unsigned countCores(const std::vector<cpuinfo> &info) {
  std::set<std::pair<int, int>> cores;
  for (auto &c : info) cores.insert(std::make_pair(c.physid, c.coreid));
  return cores.size();
}

unsigned countNumaNodes(const std::vector<cpuinfo> &info) {
  std::set<unsigned> nodes;
  for (auto &c : info) nodes.insert(c.numaNode);
  return nodes.size();
}

void markSMT(std::vector<cpuinfo> &info) {
  for (unsigned int i = 1; i < info.size(); ++i) {
    if (info[i - 1].physid == info[i].physid &&
        info[i - 1].coreid == info[i].coreid) {
      info[i].smt = true;
    } else
      info[i].smt = false;
  }
}

std::vector<int> parseCPUSet() {
  std::vector<int> vals;

  std::ifstream data("/proc/self/status");

  if (!data) {
    return vals;
  }

  std::string line;
  std::string prefix("Cpus_allowed_list:");
  bool found = false;
  while (true) {
    std::getline(data, line);
    if (!data) {
      return vals;
    }

    if (line.compare(0, prefix.size(), prefix) == 0) {
      found = true;
      break;
    }
  }

  if (!found) {
    return vals;
  }

  line = line.substr(prefix.size());

  return parseCPUList(line);
}

static std::vector<cpuinfo> makeFallbackCPUInfoFromSet(
    const std::vector<int> &cpuset) {
  std::vector<int> allowed = cpuset;
  std::sort(allowed.begin(), allowed.end());

  std::vector<cpuinfo> info;
  info.reserve(allowed.size());
  for (size_t i = 0; i < allowed.size(); ++i) {
    cpuinfo c;
    c.proc = allowed[i];
    c.physid = 0;
    c.sib = allowed.size();
    c.coreid = i;
    c.cpucores = allowed.size();
    c.valid = true;
    c.smt = false;
    c.numaNode = getNumaNode(c);
    info.push_back(c);
  }
  return info;
}

void markValid(std::vector<cpuinfo> &info) {
  auto v = parseCPUSet();
  if (v.empty()) {
    for (auto &c : info) c.valid = true;
  } else {
    std::sort(v.begin(), v.end());
    for (auto &c : info)
      c.valid = std::binary_search(v.begin(), v.end(), c.proc);
  }
}

HWTopoInfo makeHWTopo() {
  MachineTopoInfo retMTI;

  auto info = parseCPUInfo();
  const auto cpuset = parseCPUSet();
  std::sort(info.begin(), info.end());
  markSMT(info);
  markValid(info);

  info.erase(
      std::partition(
          info.begin(), info.end(), [](const cpuinfo &c) { return c.valid; }),
      info.end());

  // Some Slurm/cgroup environments expose the allowed CPU list correctly in
  // /proc/self/status but provide a truncated /proc/cpuinfo view. Fall back to
  // the cpuset so the thread pool still matches the CPUs allocated to the task.
  if (!cpuset.empty() && info.size() != cpuset.size()) {
    info = makeFallbackCPUInfoFromSet(cpuset);
  }

  std::sort(info.begin(), info.end());
  markSMT(info);
  retMTI.maxSockets = countSockets(info);
  retMTI.maxThreads = info.size();
  retMTI.maxCores = countCores(info);
  retMTI.maxNumaNodes = countNumaNodes(info);

  std::vector<ThreadTopoInfo> retTTI;
  retTTI.reserve(retMTI.maxThreads);
  // compute renumberings
  std::set<unsigned> sockets;
  std::set<unsigned> numaNodes;
  for (auto &i : info) {
    sockets.insert(i.physid);
    numaNodes.insert(i.numaNode);
  }
  unsigned mid = 0;  // max socket id
  for (unsigned i = 0; i < info.size(); ++i) {
    unsigned pid = info[i].physid;
    unsigned repid =
        std::distance(sockets.begin(), sockets.find(info[i].physid));
    mid = std::max(mid, repid);
    unsigned leader = std::distance(
        info.begin(),
        std::find_if(info.begin(), info.end(), [pid](const cpuinfo &c) {
          return c.physid == pid;
        }));
    retTTI.push_back(ThreadTopoInfo{
        i, leader, repid,
        (unsigned)std::distance(
            numaNodes.begin(), numaNodes.find(info[i].numaNode)),
        mid, info[i].proc, info[i].numaNode});
  }

  return {
      .machineTopoInfo = retMTI,
      .threadTopoInfo = retTTI,
  };
}

void HWTopoInfo::show() {
  printf("CPU Info:\n");
  printf("------------------------------------\n");
  printf("maxThreads: %u\n", machineTopoInfo.maxThreads);
  printf("maxCores: %u\n", machineTopoInfo.maxCores);
  printf("maxSockets: %u\n", machineTopoInfo.maxSockets);
  printf("maxNumaNodes: %u\n", machineTopoInfo.maxNumaNodes);
  printf("------------------------------------\n");
  printf("Thread Info:\n");
  printf("Id\tleader\tsocket\tnuma\tcumuSoc\tosCxt\tosNuma\n");
  for (auto thd : threadTopoInfo) {
    printf(
        "%u\t%u\t%u\t%u\t%u\t%u\t%u\n", thd.tid, thd.socketLeader, thd.socket,
        thd.numaNode, thd.cumulativeMaxSocket, thd.osContext, thd.osNumaNode);
  }
}

HWTopoInfo getHWTopo() {
  static std::mutex lock;
  static std::unique_ptr<HWTopoInfo> data;

  std::lock_guard<std::mutex> guard(lock);
  if (!data) {
    data = std::make_unique<HWTopoInfo>(makeHWTopo());
  }
  return *data;
}

//! binds current thread to OS HW context "proc"
bool bindThreadSelf(unsigned osContext) {
  cpu_set_t mask;
  /* CPU_ZERO initializes all the bits in the mask to zero. */
  CPU_ZERO(&mask);

  /* CPU_SET sets only the bit corresponding to cpu. */
  // void to cancel unused result warning
  (void)CPU_SET(osContext, &mask);

  /* sched_setaffinity returns 0 in success */
  if (sched_setaffinity(0, sizeof(mask), &mask) == -1) {
    printf("Could not set CPU affinity to %u (%s)", osContext, strerror(errno));
    return false;
  }
  return true;
}

//! unbinds current thread from any specific OS HW context
bool unbindThreadSelf() {
  cpu_set_t mask;
  /* CPU_ZERO initializes all the bits in the mask to zero. */
  CPU_ZERO(&mask);

  /* Get the number of processors available. */
  int numCPUs = sysconf(_SC_NPROCESSORS_ONLN);
  if (numCPUs <= 0) {
    printf(
        "Failed to get the number of available processors (%s)",
        strerror(errno));
    return false;
  }

  /* CPU_SET all available CPUs in the mask. */
  std::cout << "reset CPU num:" << numCPUs << std::endl;
  for (int i = 0; i < numCPUs; ++i) {
    (void)CPU_SET(i, &mask);
  }

  /* sched_setaffinity returns 0 on success */
  if (sched_setaffinity(0, sizeof(mask), &mask) == -1) {
    printf("Could not unbind CPU affinity (%s)", strerror(errno));
    return false;
  }

  return true;
}
