#include "backend.h"

#include <atomic>
#include <mutex>

#include "hugepage.h"
#include "utils.h"

thread_local char *ptsBase;

PerBackend &getPTSBackend() {
  static PerBackend b;
  return b;
}

thread_local char *pssBase;

PerBackend &getPPSBackend() {
  static PerBackend b;
  return b;
}

const size_t ptAllocSize = allocSize() * MAX_MEMPAGE_NUM;
inline void *alloc() {
  // alloc a single page, don't prefault
  void *toReturn = allocPages(1, true);
  if (toReturn == nullptr) {
    printf("per-thread storage out of memory");
    exit(0);
  }
  return toReturn;
}

constexpr unsigned MAX_SIZE = 32;
// PerBackend storage is typically cache-aligned. Simplify bookkeeping at the
// expense of fragmentation by restricting all allocations to be cache-aligned.
constexpr unsigned MIN_SIZE = 7;

static_assert((1 << MIN_SIZE) == substrate::COROMEM_CACHE_LINE_SIZE);

PerBackend::PerBackend() { freeOffsets.resize(MAX_SIZE); }

unsigned PerBackend::nextLog2(unsigned size) {
  unsigned i = MIN_SIZE;
  while ((1U << i) < size) {
    ++i;
  }
  if (i >= MAX_SIZE) {
    abort();
  }
  return i;
}

unsigned PerBackend::allocOffset(const unsigned sz) {
  unsigned ll = nextLog2(sz);
  unsigned size = (1 << ll);

  // TODO: general alloc
  // printf(
  //     "try to alloc size: %d pt size: %d\n",
  //     nextLoc.load(std::memory_order_relaxed) + size, ptAllocSize);
  if (nextLoc.load(std::memory_order_relaxed) + size <= ptAllocSize) {
    // simple path, where we allocate bump ptr style
    unsigned offset = nextLoc.fetch_add(size);
    if (offset + size <= ptAllocSize) {
      return offset;
    }
  }

  if (invalid) {
    // GALOIS_DIE("allocating after delete");
    printf("allocating after delete.");
    exit(0);
    return ptAllocSize;
  }

  // printf("should not have go here\n");

  // find a free offset
  std::lock_guard<Lock> llock(freeOffsetsLock);

  unsigned index = ll;
  if (!freeOffsets[index].empty()) {
    unsigned offset = freeOffsets[index].back();
    freeOffsets[index].pop_back();
    return offset;
  }

  // find a bigger size
  for (; (index < MAX_SIZE) && (freeOffsets[index].empty()); ++index);

  if (index == MAX_SIZE) {
    printf("allocOffset: per-thread storage out of memory");
    exit(-1);
    return ptAllocSize;
  }

  // Found a bigger free offset. Use the first piece equal to required
  // size and produce vending machine change for the rest.
  assert(!freeOffsets[index].empty());
  unsigned offset = freeOffsets[index].back();
  freeOffsets[index].pop_back();

  // remaining chunk
  unsigned end = offset + (1 << index);
  unsigned start = offset + size;
  for (unsigned i = index - 1; start < end; --i) {
    freeOffsets[i].push_back(start);
    start += (1 << i);
  }

  assert(offset != ptAllocSize);

  return offset;
}

void PerBackend::deallocOffset(const unsigned offset, const unsigned sz) {
  unsigned ll = nextLog2(sz);
  unsigned size = (1 << ll);
  unsigned expected = offset + size;

  if (nextLoc.compare_exchange_strong(expected, offset)) {
    // allocation was at the end, so recovered some memory
    return;
  }

  if (invalid) {
    printf("deallocing after delete.\n");
    exit(0);
    return;
  }

  // allocation not at the end
  std::lock_guard<Lock> llock(freeOffsetsLock);
  freeOffsets[ll].push_back(offset);
}

void *PerBackend::getRemote(unsigned thread, unsigned offset) {
  char *rbase = heads[thread].load(std::memory_order_relaxed);
  assert(rbase);
  return &rbase[offset];
}

void PerBackend::initCommon(unsigned maxT) {
  if (!heads) {
    assert(ThreadPool::getTID() == 0);
    heads = new std::atomic<char *>[maxT] {};
  }
}

char *PerBackend::initPerThread(unsigned maxT) {
  initCommon(maxT);
  char *b = heads[ThreadPool::getTID()] =
      (char *)allocPages(MAX_MEMPAGE_NUM, false);
  // TODO: modify this to general config
  memset(b, 0, ptAllocSize);
  return b;
}

char *PerBackend::initPerSocket(unsigned maxT) {
  initCommon(maxT);
  unsigned id = ThreadPool::getTID();
  unsigned leader = ThreadPool::getLeader();
  if (id == leader) {
    // TODO: Changed
    char *b = heads[id] = (char *)allocPages(MAX_MEMPAGE_NUM, true);
    memset(b, 0, ptAllocSize);
    return b;
  }
  char *expected = nullptr;
  // wait for leader to fix up socket
  while (heads[leader].compare_exchange_weak(expected, nullptr)) {
    substrate::asmPause();
  }
  heads[id] = heads[leader].load();
  return heads[id];
}

void initPTS(unsigned maxT) {
  if (!ptsBase) {
    // unguarded initialization as initPTS will run in the master thread
    // before any other threads are generated
    ptsBase = getPTSBackend().initPerThread(maxT);
  }
  if (!pssBase) {
    pssBase = getPPSBackend().initPerSocket(maxT);
  }
}