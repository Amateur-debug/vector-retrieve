#ifndef SUBSTRATE_PADDEDLOCK_H
#define SUBSTRATE_PADDEDLOCK_H

#include "cacheline_storage.h"
#include "simple_lock.h"

namespace substrate {

/// PaddedLock is a spinlock.  If the second template parameter is
/// false, the lock is a noop.
template <bool concurrent> class PaddedLock;

template <> class PaddedLock<true> {
  mutable CacheLineStorage<SimpleLock> Lock;

public:
  void lock() const { Lock.get().lock(); }
  bool try_lock() const { return Lock.get().try_lock(); }
  void unlock() const { Lock.get().unlock(); }
};

template <> class PaddedLock<false> {
public:
  void lock() const {}
  bool try_lock() const { return true; }
  void unlock() const {}
};

} // end namespace substrate

#endif
