#ifndef COROGRAPH_HWTOPO_H
#define COROGRAPH_HWTOPO_H

#include <string>
#include <vector>

struct ThreadTopoInfo {
  unsigned tid;                  // thread id.
  unsigned socketLeader;         // first thread id in tid's socket
  unsigned socket;               // socket (L3 normally) of thread
  unsigned numaNode;             // memory bank.  may be different than socket.
  unsigned cumulativeMaxSocket;  // max socket id seen from [0, tid]
  unsigned osContext;            // OS ID to use for thread binding
  unsigned osNumaNode;           // OS ID for numa node
};

struct MachineTopoInfo {
  unsigned maxThreads;
  unsigned maxCores;
  unsigned maxSockets;
  unsigned maxNumaNodes;
};

struct HWTopoInfo {
  MachineTopoInfo machineTopoInfo;
  std::vector<ThreadTopoInfo> threadTopoInfo;

  void show();
};

struct cpuinfo {
  // fields filled in from OS files
  unsigned proc = 0;
  unsigned physid = 0;
  unsigned sib = 0;
  unsigned coreid = 0;
  unsigned cpucores = 0;
  unsigned numaNode = 0;  // from libnuma
  bool valid = false;     // from cpuset
  bool smt = false;       // computed
};

/**
 * getHWTopo determines the machine topology from the process information
 * exposed in /proc and /dev filesystems.
 */
HWTopoInfo getHWTopo();

/**
 * parseCPUList parses cpuset information in "List format" as described in
 * cpuset(7) and available under /proc/self/status
 */
std::vector<int> parseCPUList(const std::string &in);

/**
 * bindThreadSelf binds a thread to an osContext as returned by getHWTopo.
 */
bool bindThreadSelf(unsigned osContext);

bool unbindThreadSelf();

#endif
