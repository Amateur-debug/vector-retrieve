#include "runtime/context.h"
#include "substrate/cacheline_storage.h"
#include "substrate/simple_lock.h"

#include <stdio.h>

//! Global thread context for each active thread
static thread_local runtime::SimpleRuntimeContext *thread_ctx = 0;

thread_local jmp_buf runtime::execFrame;

void runtime::setThreadContext(runtime::SimpleRuntimeContext *ctx) {
  thread_ctx = ctx;
}

runtime::SimpleRuntimeContext *runtime::getThreadContext() {
  return thread_ctx;
}

////////////////////////////////////////////////////////////////////////////////
// LockManagerBase & SimpleRuntimeContext
////////////////////////////////////////////////////////////////////////////////

runtime::LockManagerBase::AcquireStatus
runtime::LockManagerBase::tryAcquire(runtime::Lockable *lockable) {
  assert(lockable);
  if (lockable->owner.try_lock()) {
    lockable->owner.setValue(this);
    return NEW_OWNER;
  } else if (getOwner(lockable) == this) {
    return ALREADY_OWNER;
  }
  return FAIL;
}

void runtime::SimpleRuntimeContext::release(runtime::Lockable *lockable) {
  assert(lockable);
  // The deterministic executor, for instance, steals locks from other
  // iterations
  assert(customAcquire || getOwner(lockable) == this);
  assert(!lockable->next);
  lockable->owner.unlock_and_clear();
}

unsigned runtime::SimpleRuntimeContext::commitIteration() {
  unsigned numLocks = 0;
  while (locks) {
    // ORDER MATTERS!
    Lockable *lockable = locks;
    locks = lockable->next;
    lockable->next = 0;
    substrate::compilerBarrier();
    release(lockable);
    ++numLocks;
  }

  return numLocks;
}

unsigned runtime::SimpleRuntimeContext::cancelIteration() {
  return commitIteration();
}

void runtime::SimpleRuntimeContext::subAcquire(runtime::Lockable *,
                                               galois::MethodFlag) {
  // GALOIS_DIE("unreachable");
}
