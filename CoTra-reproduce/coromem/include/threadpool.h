#ifndef COROGRAPH_THREADPOOL_H
#define COROGRAPH_THREADPOOL_H

#include <atomic>
#include <condition_variable>
#include <functional>
#include <thread>
#include <vector>

#include "hwtopo.h"

class ThreadPool {
 public:
  struct shutdown_ty {};
  std::vector<std::thread> threads;

  HWTopoInfo hw_info;

  MachineTopoInfo mi;

  std::function<void(void)> work;

  struct thd_signal {
    std::condition_variable cv;
    std::mutex m;
    unsigned wbegin, wend;
    std::atomic<int> done;
    ThreadTopoInfo topo;

    void wakeup() {
      std::lock_guard<std::mutex> lg(m);
      done = 0;
      cv.notify_one();
    }

    void wait() {
      std::unique_lock<std::mutex> lg(m);
      cv.wait(lg, [=, this] { return !done; });
    }
  };

  thread_local static thd_signal thd_info;
  std::vector<thd_signal *> signals;

  ThreadPool();
  ~ThreadPool();

  void initThread(unsigned tid);

  void threadLoop(unsigned tid);

  void destroyCommon();

  void cascade();

  void decascade();

  void mainThread(unsigned num);

  template <typename tpl, int s, int r>
  struct ExecuteTupleImpl {
    static inline void execute(tpl &cmds) {
      std::get<s>(cmds)();
      ExecuteTupleImpl<tpl, s + 1, r - 1>::execute(cmds);
    }
  };

  template <typename tpl, int s>
  struct ExecuteTupleImpl<tpl, s, 0> {
    static inline void execute(tpl &) {}
  };

  template <typename... Args>
  void run(unsigned num, Args &&...args) {
    struct ExecuteTuple {
      std::tuple<Args...> cmds;

      void operator()() {
        ExecuteTupleImpl<
            std::tuple<Args...>, 0,
            std::tuple_size<std::tuple<Args...>>::value>::execute(this->cmds);
      }
      ExecuteTuple(Args &&...args) : cmds(std::forward<Args>(args)...) {}
    };
    ExecuteTuple lwork(std::forward<Args>(args)...);
    work = std::ref(lwork);
    mainThread(num);
  }

  void poolPause();
  void poolContiue();

  unsigned getMaxUsableThreads() const { return mi.maxThreads; }
  unsigned getMaxThreads() const { return mi.maxThreads; }
  unsigned getMaxCores() const { return mi.maxCores; }
  unsigned getMaxSockets() const { return mi.maxSockets; }
  unsigned getMaxNumaNodes() const { return mi.maxNumaNodes; }

  unsigned getLeaderForSocket(unsigned pid) const {
    for (unsigned i = 0; i < getMaxThreads(); ++i)
      if (getSocket(i) == pid && isLeader(i)) return i;
    abort();
  }

  bool isLeader(unsigned tid) const {
    return signals[tid]->topo.socketLeader == tid;
  }
  unsigned getSocket(unsigned tid) const { return signals[tid]->topo.socket; }
  unsigned getLeader(unsigned tid) const {
    return signals[tid]->topo.socketLeader;
  }
  unsigned getCumulativeMaxSocket(unsigned tid) const {
    return signals[tid]->topo.cumulativeMaxSocket;
  }
  unsigned getNumaNode(unsigned tid) const {
    return signals[tid]->topo.numaNode;
  }

  static unsigned getTID() { return thd_info.topo.tid; }
  static bool isLeader() {
    return thd_info.topo.tid == thd_info.topo.socketLeader;
  }
  static unsigned getLeader() { return thd_info.topo.socketLeader; }
  static unsigned getSocket() { return thd_info.topo.socket; }
  static unsigned getCumulativeMaxSocket() {
    return thd_info.topo.cumulativeMaxSocket;
  }
  static unsigned getNumaNode() { return thd_info.topo.numaNode; }
};

ThreadPool &getThreadPool(void);

void setThreadPool(ThreadPool *tp);

unsigned int setActiveThreads(unsigned int num) noexcept;

unsigned int getActiveThreads() noexcept;

#endif
