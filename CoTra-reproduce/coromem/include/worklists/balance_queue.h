#pragma once

#include <optional>

#include "../backend.h"
#include "../galois/config.h"
#include "../substrate/padded_lock.h"
#include "../threadpool.h"
#include "../worklists/lockfreeQ.h"
#include "../worklists/worklist.h"

template <typename T>
class BalanceQueue {
 private:
  typedef LockFreeQueue<T> ConQueue;
  // TODO: can use Chunk to increase granularity.
  PerThreadStorage<ConQueue> task_queue;

  COROMEM_ATTRIBUTE_NOINLINE std::optional<T> stealWithinSocket(unsigned tid) {
    auto &tp = getThreadPool();
    const unsigned maxT = getActiveThreads();
    const unsigned my_pack = ThreadPool::getSocket();
    const unsigned per_pack = tp.getMaxThreads() / tp.getMaxSockets();
    const unsigned pack_beg = my_pack * per_pack;
    const unsigned pack_end = (my_pack + 1) * per_pack;

    for (unsigned i = 1; i < pack_end; ++i) {
      // go around the socket in circle starting from the next thread
      unsigned t = (tid + i) % per_pack + pack_beg;
      if (t < maxT) {
        auto ret = task_queue.getRemote(t)->pop();
        if (ret.has_value()) {
          return ret;
        }
      }
    }
    return std::nullopt;
  }

  COROMEM_ATTRIBUTE_NOINLINE std::optional<T> stealOutsideSocket(unsigned tid) {
    auto &tp = getThreadPool();
    unsigned myPkg = ThreadPool::getSocket();
    unsigned maxT = getActiveThreads();

    for (unsigned i = 0; i < maxT; ++i) {
      unsigned t = (tid + i) % maxT;
      if (tp.getSocket(t) != myPkg) {
        auto ret = task_queue.getRemote(t)->pop();
        if (ret.has_value()) {
          return ret;
        }
      }
    }

    return std::nullopt;
  }

  COROMEM_ATTRIBUTE_NOINLINE std::optional<T> trySteal(unsigned tid) {
    auto ret = stealWithinSocket(tid);
    if (ret.has_value()) {
      return ret;
    }
    // if (getThreadPool().isLeader(tid)) {
    //   ret = stealOutsideSocket(tid);
    //   if (ret.has_value()) {
    //     return ret;
    //   }
    // }
    // ret = stealOutsideSocket(tid);
    // if (ret.has_value()) {
    //   return ret;
    // }

    return std::nullopt;
  }

 public:
  BalanceQueue() {}

  ~BalanceQueue() {}

  template <typename... Args>
  void emplace_back(Args &&...args) {
    ConQueue &queue = *task_queue.getLocal();
    queue.emplace_back(std::forward<Args>(args)...);
  }

  void push(T &val) {
    ConQueue &queue = *task_queue.getLocal();
    queue.push(val);
  }

  void push(std::vector<T> &vals) {
    ConQueue &queue = *task_queue.getLocal();
    queue.push(vals);
  }

  std::optional<T> pop(bool steal = true) {
    ConQueue &queue = *task_queue.getLocal();
    auto ret = queue.pop();
    if (ret.has_value()) {
      return ret;
    }
    if (steal) {
      return trySteal(ThreadPool::getTID());
    }
    return std::nullopt;
  }
};
