#ifndef USERCONTEXT_H
#define USERCONTEXT_H

#include <deque>
#include <functional>

#include "mem.h"
#include "config.h"
#include "../runtime/context.h"

/**
 * This is the object passed to the user's parallel loop.  This
 * provides the in-loop api.
 */
template <typename T> class UserContext : private boost::noncopyable {
protected:
  //! push stuff
  typedef std::deque<T> PushBufferTy;
  static const unsigned int fastPushBackLimit = 64;
  typedef std::function<void(PushBufferTy &)> FastPushBack;

  PushBufferTy pushBuffer;
  //! Allocator stuff
  IterAllocBaseTy IterationAllocatorBase;
  PerIterAllocTy PerIterationAllocator;

  //! used by all
  bool *didBreak = nullptr;
  FastPushBack fastPushBack;

  //! some flags used by deterministic
  bool firstPassFlag = false;
  void *localState = nullptr;

  void __resetAlloc() { IterationAllocatorBase.clear(); }

  void __setFirstPass(void) { firstPassFlag = true; }

  void __resetFirstPass(void) { firstPassFlag = false; }

  PushBufferTy &__getPushBuffer() { return pushBuffer; }

  void __resetPushBuffer() { pushBuffer.clear(); }

  void __setLocalState(void *p) { localState = p; }

  void __setFastPushBack(FastPushBack f) { fastPushBack = f; }

public:
  UserContext()
      : IterationAllocatorBase(),
        PerIterationAllocator(&IterationAllocatorBase), didBreak(0) {}

  //! Signal break in parallel loop, current iteration continues
  //! untill natural termination
  void breakLoop() { *didBreak = true; }

  //! Acquire a per-iteration allocator
  PerIterAllocTy &getPerIterAlloc() { return PerIterationAllocator; }

  //! Push new work
  template <typename... Args> void push(Args &&... args) {
    // galois::runtime::checkWrite(MethodFlag::WRITE, true);
    pushBuffer.emplace_back(std::forward<Args>(args)...);
    if (fastPushBack && pushBuffer.size() > fastPushBackLimit)
      fastPushBack(pushBuffer);
  }

  //! Push new work
  template <typename... Args> inline void push_back(Args &&... args) {
    this->push(std::forward<Args>(args)...);
  }

  //! Push new work
  template <typename... Args> inline void insert(Args &&... args) {
    this->push(std::forward<Args>(args)...);
  }

  //! Force the abort of this iteration
  void abort() { runtime::signalConflict(); }

  //! Store and retrieve local state for deterministic
  template <typename LS> LS *getLocalState(void) {
    return reinterpret_cast<LS *>(localState);
  }

  template <typename LS, typename... Args>
  LS *createLocalState(Args &&... args) {
    new (localState) LS(std::forward<Args>(args)...);
    return getLocalState<LS>();
  }

  //! used by deterministic and ordered
  //! @returns true when the operator is invoked for the first time. The
  //! operator can use this information and choose to expand the neighborhood
  //! only in the first pass.
  bool isFirstPass(void) const { return firstPassFlag; }

  //! declare that the operator has crossed the cautious point.  This
  //! implies all data has been touched thus no new locks will be
  //! acquired.
  void cautiousPoint() {
    if (isFirstPass()) {
      runtime::signalFailSafe();
    }
  }
};

#endif
