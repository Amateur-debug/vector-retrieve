#include "threadpool.h"

#include <algorithm>
#include <cassert>
#include <iostream>

#include "hwtopo.h"

// Forward declare this to avoid including PerThreadStorage.
// We avoid this to stress that the thread Pool MUST NOT depend on PTS.

extern void initPTS(unsigned);

thread_local ThreadPool::thd_signal ThreadPool::thd_info;

inline static void asmPause() {
#if defined(__i386__) || defined(__amd64__)
  asm volatile("pause");
#endif
}

static void joinAllThreads(std::vector<std::thread> &threads) {
  for (auto &t : threads) {
    if (t.joinable()) {
      t.join();
    }
  }
  threads.clear();
}

ThreadPool::ThreadPool()
    : mi(getHWTopo().machineTopoInfo), hw_info(getHWTopo()) {
  signals.resize(mi.maxThreads);
  initThread(0);

  for (unsigned i = 1; i < mi.maxThreads; ++i) {
    std::thread t(&ThreadPool::threadLoop, this, i);
    threads.emplace_back(std::move(t));
  }

  while (std::any_of(signals.begin(), signals.end(), [](thd_signal *p) {
    return !p || !p->done;
  })) {
    std::atomic_thread_fence(std::memory_order_seq_cst);
  }
}

ThreadPool::~ThreadPool() {
  destroyCommon();
  joinAllThreads(threads);
}

void ThreadPool::poolPause() {
  destroyCommon();
  unbindThreadSelf();
  joinAllThreads(threads);
}

void ThreadPool::poolContiue() {
  for (unsigned i = 1; i < mi.maxThreads; ++i) {
    std::thread t(&ThreadPool::threadLoop, this, i);
    threads.emplace_back(std::move(t));
  }

  while (std::any_of(signals.begin(), signals.end(), [](thd_signal *p) {
    return !p || !p->done;
  })) {
    std::atomic_thread_fence(std::memory_order_seq_cst);
  }
}

void ThreadPool::destroyCommon() {
  run(mi.maxThreads, []() { throw shutdown_ty(); });
}

void ThreadPool::initThread(unsigned tid) {
  signals[tid] = &thd_info;
  thd_info.topo = getHWTopo().threadTopoInfo[tid];
  // Initialize
  initPTS(mi.maxThreads);

  bindThreadSelf(thd_info.topo.osContext);
  thd_info.done = 1;
}

void ThreadPool::threadLoop(unsigned tid) {
  initThread(tid);
  auto &thd = thd_info;
  do {
    thd.wait();
    cascade();
    try {
      work();
    } catch (const shutdown_ty &) {
      return;
    }
    decascade();
  } while (true);
}

void ThreadPool::decascade() {
  auto &thd = thd_info;
  if (thd.wbegin != thd.wend) {
    auto midpoint = thd.wbegin + (1 + thd.wend - thd.wbegin) / 2;
    auto &left_done = signals[thd.wbegin]->done;
    while (!left_done) {
      asmPause();
    }
    if (midpoint < thd.wend) {
      auto &right_done = signals[midpoint]->done;
      while (!right_done) {
        asmPause();
      }
    }
  }
  thd.done = 1;
}

void ThreadPool::cascade() {
  auto &thd = thd_info;
  assert(thd.wbegin <= thd.wend);

  if (thd.wbegin == thd.wend) {
    return;
  }

  auto midpoint = thd.wbegin + (1 + thd.wend - thd.wbegin) / 2;

  auto left_chd = signals[thd.wbegin];
  left_chd->wbegin = thd.wbegin + 1;
  left_chd->wend = midpoint;
  left_chd->wakeup();

  if (midpoint < thd.wend) {
    auto right_chd = signals[midpoint];
    right_chd->wbegin = midpoint + 1;
    right_chd->wend = thd.wend;
    right_chd->wakeup();
  }
}

void ThreadPool::mainThread(unsigned num) {
  auto &thd = thd_info;
  thd.wbegin = 1;
  thd.wend = num;

  cascade();
  try {
    work();
  } catch (const shutdown_ty &) {
    return;
  }
  decascade();
  work = nullptr;
}

static ThreadPool *TPOOL = nullptr;

void setThreadPool(ThreadPool *tp) { TPOOL = tp; }

ThreadPool &getThreadPool() { return *TPOOL; }

unsigned int activeThreads = 1;

unsigned int setActiveThreads(unsigned int num) noexcept {
  num = std::min(num, getThreadPool().getMaxUsableThreads());
  num = std::max(num, 1U);
  activeThreads = num;
  return num;
}

unsigned int getActiveThreads() noexcept { return activeThreads; }
