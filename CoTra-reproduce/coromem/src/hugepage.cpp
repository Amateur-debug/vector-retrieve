#include "hugepage.h"

#include <sys/mman.h>

#include <mutex>

#include "runtime/mem.h"
#include "substrate/simple_lock.h"

extern unsigned activeThreads;

// figure this out dynamically
const size_t hugePageSize = 2 * 1024 * 1024;
// protect mmap, munmap since linux has issues
static substrate::SimpleLock allocLock;

static void *trymmap(size_t size, int flag) {
  std::lock_guard<substrate::SimpleLock> lg(allocLock);
  const int _PROT = PROT_READ | PROT_WRITE;
  void *ptr = ::mmap(0, size, _PROT, flag, -1, 0);
  if (ptr == MAP_FAILED) ptr = nullptr;
  return ptr;
}

static const int _MAP = _MAP_ANON | MAP_PRIVATE;
#ifdef MAP_POPULATE
static const int _MAP_POP = MAP_POPULATE | _MAP;
static const bool doHandMap = false;
#else
static const int _MAP_POP = _MAP;
static const bool doHandMap = true;
#endif
#ifdef MAP_HUGETLB
static const int _MAP_HUGE_POP = MAP_HUGETLB | _MAP_POP;
static const int _MAP_HUGE = MAP_HUGETLB | _MAP;
#else
static const int _MAP_HUGE_POP = _MAP_POP;
static const int _MAP_HUGE = _MAP;
#endif

size_t allocSize() { return hugePageSize; }

void *allocPages(unsigned num, bool preFault) {
  if (num > 0) {
    void *ptr =
        trymmap(num * hugePageSize, preFault ? _MAP_HUGE_POP : _MAP_HUGE);
    if (!ptr) {
      printf(
          "WARN: Failed to allocate huge pages. Falling back to normal "
          "pages.\n");
      ptr = trymmap(num * hugePageSize, preFault ? _MAP_POP : _MAP);
    }

    if (!ptr) {
      printf("Out of Memory.\n");
      exit(-1);
    }

    // if (preFault && doHandMap)
    //   for (size_t x = 0; x < num * hugePageSize; x += 4096)
    //     static_cast<char *>(ptr)[x] = 0;

    return ptr;
  } else {
    return nullptr;
  }
}

void freePages(void *ptr, unsigned num) {
  std::lock_guard<substrate::SimpleLock> lg(allocLock);
  if (munmap(ptr, num * hugePageSize) != 0) {
    printf("Unmap failed");
    exit(0);
    // GALOIS_SYS_DIE("Unmap failed");
  }
}

#define __is_trivial(type) \
  __has_trivial_constructor(type) && __has_trivial_copy(type)

static PageAllocState<> *PA;

void setPagePoolState(PageAllocState<> *pa) {
  // GALOIS_ASSERT(!(PA && pa),
  // "PagePool.cpp: Double Initialization of PageAllocState");
  PA = pa;
}

int numPagePoolAllocTotal() { return PA->countAll(); }

int numPagePoolAllocForThread(unsigned tid) { return PA->count(tid); }

void *pagePoolAlloc() { return PA->pageAlloc(); }

void pagePoolPreAlloc(unsigned num) {
  while (num--) PA->pagePreAlloc();
}

void pagePoolFree(void *ptr) { PA->pageFree(ptr); }

size_t pagePoolSize() { return allocSize(); }

void preAlloc_impl(unsigned num) {
  unsigned pagesPerThread = (num + activeThreads - 1) / activeThreads;
  getThreadPool().run(
      activeThreads, [=]() { pagePoolPreAlloc(pagesPerThread); });
}