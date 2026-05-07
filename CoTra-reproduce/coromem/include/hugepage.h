#ifndef COROGRAPH_HUGEPAGE_H
#define COROGRAPH_HUGEPAGE_H

#include <cstddef>
#include <deque>
#include <numeric>
#include <sys/mman.h>
#include <utility>

#include "substrate/cacheline_storage.h"
#include "substrate/ptr_lock.h"
#include "substrate/simple_lock.h"
#include "threadpool.h"

//! offset type for mmap
typedef off_t offset_t;
static const int _MAP_ANON = MAP_ANONYMOUS;

// size of pages
size_t allocSize();

// allocate contiguous pages, optionally faulting them in
void *allocPages(unsigned num, bool preFault);

// free page range
void freePages(void *ptr, unsigned num);

void *pagePoolAlloc();
void pagePoolFree(void *);
void pagePoolPreAlloc(unsigned);

// Size of returned pages
size_t pagePoolSize();

//! Returns total large pages allocated by Galois memory management subsystem
int numPagePoolAllocTotal();
//! Returns total large pages allocated for thread by Galois memory management
//! subsystem
int numPagePoolAllocForThread(unsigned tid);

struct FreeNode {
  FreeNode *next;
};

typedef substrate::PtrLock<FreeNode> HeadPtr;
typedef substrate::CacheLineStorage<HeadPtr> HeadPtrStorage;

// Tracks pages allocated
template <typename _UNUSED = void> class PageAllocState {
  std::deque<std::atomic<int>> counts;
  std::vector<HeadPtrStorage> pool;
  std::unordered_map<void *, int> ownerMap;
  substrate::SimpleLock mapLock;

  void *allocFromOS() {
    void *ptr = allocPages(1, true);
    assert(ptr);
    auto tid = ThreadPool::getTID();
    counts[tid] += 1;
    std::lock_guard<substrate::SimpleLock> lg(mapLock);
    ownerMap[ptr] = tid;
    return ptr;
  }

public:
  PageAllocState() {
    auto num = getThreadPool().getMaxThreads();
    counts.resize(num);
    pool.resize(num);
  }

  int count(int tid) const { return counts[tid]; }

  int countAll() const {
    return std::accumulate(counts.begin(), counts.end(), 0);
  }

  void *pageAlloc() {
    auto tid = ThreadPool::getTID();
    HeadPtr &hp = pool[tid].data;
    if (hp.getValue()) {
      hp.lock();
      FreeNode *h = hp.getValue();
      if (h) {
        hp.unlock_and_set(h->next);
        return h;
      }
      hp.unlock();
    }
    return allocFromOS();
  }

  void pageFree(void *ptr) {
    assert(ptr);
    mapLock.lock();
    assert(ownerMap.count(ptr));
    int i = ownerMap[ptr];
    mapLock.unlock();
    HeadPtr &hp = pool[i].data;
    hp.lock();
    FreeNode *nh = reinterpret_cast<FreeNode *>(ptr);
    nh->next = hp.getValue();
    hp.unlock_and_set(nh);
  }

  void pagePreAlloc() { pageFree(allocFromOS()); }
};

//! Initialize PagePool, used by runtime::init();
void setPagePoolState(PageAllocState<> *pa);

void preAlloc_impl(unsigned num);

#endif // COROGRAPH_HUGEPAGE_H