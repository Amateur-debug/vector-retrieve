#include "runtime/mem.h"

#include <map>
#include <mutex>

using namespace runtime;

// Anchor the class
SystemHeap::SystemHeap() { assert(AllocSize == pagePoolSize()); }

SystemHeap::~SystemHeap() {}

#ifndef COROMEM_FORCE_STANDALONE
thread_local SizedHeapFactory::HeapMap *SizedHeapFactory::localHeaps = 0;

SizedHeapFactory::SizedHeap *
SizedHeapFactory::getHeapForSize(const size_t size) {
  if (size == 0)
    return nullptr;
  return Base::getInstance()->getHeap(size);
}

SizedHeapFactory::SizedHeap *SizedHeapFactory::getHeap(const size_t size) {
  typedef SizedHeapFactory::HeapMap HeapMap;

  if (!localHeaps) {
    std::lock_guard<substrate::SimpleLock> ll(lock);
    localHeaps = new HeapMap;
    allLocalHeaps.push_front(localHeaps);
  }

  auto &lentry = (*localHeaps)[size];
  if (lentry)
    return lentry;

  {
    std::lock_guard<substrate::SimpleLock> ll(lock);
    auto &gentry = heaps[size];
    if (!gentry)
      gentry = new SizedHeap();
    lentry = gentry;
    return lentry;
  }
}

Pow_2_BlockHeap::Pow_2_BlockHeap(void) noexcept : heapTable() {
  populateTable();
}

SizedHeapFactory::SizedHeapFactory() : lock() {}

SizedHeapFactory::~SizedHeapFactory() {
  // TODO destructor ordering problem: there may be pointers to deleted
  // SizedHeap when this Factory is destroyed before dependent
  // FixedSizeHeaps.
  for (auto entry : heaps)
    delete entry.second;
  for (auto mptr : allLocalHeaps)
    delete mptr;
}
#endif
