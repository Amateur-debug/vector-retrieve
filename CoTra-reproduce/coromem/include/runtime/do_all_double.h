#ifndef DOALL_DOUBLE_H
#define DOALL_DOUBLE_H

#include "../backend.h"
#include "../galois/config.h"
#include "../substrate/barrier.h"
#include "../substrate/compiler_specific.h"
#include "../substrate/padded_lock.h"
#include "../substrate/termination.h"
#include "../threadpool.h"
#include "on_each.h"
#include "optref_types.h"

template <typename R, typename F, typename FD, typename FS, typename ArgsTuple>
class DoAllDoubleExec {
  typedef typename R::local_iterator Iter;
  typedef typename std::iterator_traits<Iter>::difference_type Diff_ty;
  enum StealAmt { HALF, FULL };
  constexpr static const bool USE_TERM = false;

  struct ThreadContext {
    alignas(substrate::COROMEM_CACHE_LINE_SIZE) substrate::SimpleLock
        work_mutex;
    unsigned id;
    Iter shared_beg;
    Iter shared_end;
    Diff_ty m_size;

    ThreadContext()
        : work_mutex(),
          id(getThreadPool().getMaxThreads()),
          shared_beg(),
          shared_end(),
          m_size(0) {}
    ThreadContext(unsigned id, Iter beg, Iter end)
        : work_mutex(),
          id(id),
          shared_beg(beg),
          shared_end(end),
          m_size(std::distance(beg, end)) {}

    bool doWork(F func, const unsigned chunk_size) {
      Iter beg(shared_beg);
      Iter end(shared_end);
      bool didwork = false;
      while (getWork(beg, end, chunk_size)) {
        didwork = true;
        for (; beg != end; ++beg) {
          func(*beg);
        }
      }
      return didwork;
    }
    bool hasWorkWeak() const { return (m_size > 0); }

   private:
    bool getWork(Iter &priv_beg, Iter &priv_end, const unsigned chunk_size) {
      bool succ = false;
      work_mutex.lock();
      {
        if (hasWorkWeak()) {
          succ = true;

          Iter nbeg = shared_beg;
          if (m_size <= chunk_size) {
            nbeg = shared_end;
            m_size = 0;

          } else {
            std::advance(nbeg, chunk_size);
            m_size -= chunk_size;
          }

          priv_beg = shared_beg;
          priv_end = nbeg;
          shared_beg = nbeg;
        }
      }
      work_mutex.unlock();
      return succ;
    }

    void steal_from_end_impl(
        Iter &steal_beg, Iter &steal_end, const Diff_ty sz,
        std::forward_iterator_tag) {
      // steal from front for forward_iterator_tag
      steal_beg = shared_beg;
      std::advance(shared_beg, sz);
      steal_end = shared_beg;
    }

    void steal_from_end_impl(
        Iter &steal_beg, Iter &steal_end, const Diff_ty sz,
        std::bidirectional_iterator_tag) {
      steal_end = shared_end;
      std::advance(shared_end, -sz);
      steal_beg = shared_end;
    }

    void steal_from_end(Iter &steal_beg, Iter &steal_end, const Diff_ty sz) {
      assert(sz > 0);
      steal_from_end_impl(
          steal_beg, steal_end, sz,
          typename std::iterator_traits<Iter>::iterator_category());
    }

    void steal_from_beg(Iter &steal_beg, Iter &steal_end, const Diff_ty sz) {
      steal_beg = shared_beg;
      std::advance(shared_beg, sz);
      steal_end = shared_beg;
    }

   public:
    bool stealWork(
        Iter &steal_beg, Iter &steal_end, Diff_ty &steal_size, StealAmt amount,
        size_t chunk_size) {
      bool succ = false;
      if (work_mutex.try_lock()) {
        if (hasWorkWeak()) {
          succ = true;
          if (amount == HALF && m_size > (Diff_ty)chunk_size) {
            steal_size = m_size / 2;
          } else {
            steal_size = m_size;
          }
          if (m_size <= steal_size) {
            steal_beg = shared_beg;
            steal_end = shared_end;
            shared_beg = shared_end;
            steal_size = m_size;
            m_size = 0;
          } else {
            // steal_from_end (steal_beg, steal_end, steal_size);
            steal_from_beg(steal_beg, steal_end, steal_size);
            m_size -= steal_size;
          }
        }
        work_mutex.unlock();
      }
      return succ;
    }

    void assignWork(const Iter &beg, const Iter &end, const Diff_ty sz) {
      work_mutex.lock();
      {
        shared_beg = beg;
        shared_end = end;
        m_size = sz;
      }
      work_mutex.unlock();
    }
  };

 private:
  COROMEM_ATTRIBUTE_NOINLINE bool transferWork(
      ThreadContext &rich, ThreadContext &poor, StealAmt amount) {
    Iter steal_beg;
    Iter steal_end;
    // stealWork should initialize to a more appropriate value
    Diff_ty steal_size = 0;
    bool succ =
        rich.stealWork(steal_beg, steal_end, steal_size, amount, chunk_size);
    if (succ) {
      poor.assignWork(steal_beg, steal_end, steal_size);
    }
    return succ;
  }

  COROMEM_ATTRIBUTE_NOINLINE bool stealWithinSocket(ThreadContext &poor) {
    bool sawWork = false;
    bool stoleWork = false;

    auto &tp = getThreadPool();

    const unsigned maxT = getActiveThreads();
    const unsigned my_pack = ThreadPool::getSocket();
    const unsigned per_pack = tp.getMaxThreads() / tp.getMaxSockets();
    const unsigned pack_beg = my_pack * per_pack;
    const unsigned pack_end = (my_pack + 1) * per_pack;

    for (unsigned i = 1; i < pack_end; ++i) {
      // go around the socket in circle starting from the next thread
      unsigned t = (poor.id + i) % per_pack + pack_beg;
      if (t < maxT) {
        if (workers.getRemote(t)->hasWorkWeak()) {
          sawWork = true;
          stoleWork = transferWork(*workers.getRemote(t), poor, HALF);
          if (stoleWork) {
            break;
          }
        }
      }
    }

    return sawWork || stoleWork;
  }

  COROMEM_ATTRIBUTE_NOINLINE bool stealOutsideSocket(
      ThreadContext &poor, const StealAmt &amt) {
    bool sawWork = false;
    bool stoleWork = false;
    auto &tp = getThreadPool();
    unsigned myPkg = ThreadPool::getSocket();
    unsigned maxT = getActiveThreads();

    for (unsigned i = 0; i < maxT; ++i) {
      ThreadContext &rich = *(workers.getRemote((poor.id + i) % maxT));
      if (tp.getSocket(rich.id) != myPkg) {
        if (rich.hasWorkWeak()) {
          sawWork = true;
          stoleWork = transferWork(rich, poor, amt);
          // stoleWork = transferWork (rich, poor, HALF);
          if (stoleWork) {
            break;
          }
        }
      }
    }

    return sawWork || stoleWork;
  }

  COROMEM_ATTRIBUTE_NOINLINE bool trySteal(ThreadContext &poor) {
    bool ret = false;
    ret = stealWithinSocket(poor);
    if (ret) {
      return true;
    }
    substrate::asmPause();
    if (getThreadPool().isLeader(poor.id)) {
      ret = stealOutsideSocket(poor, HALF);
      if (ret) {
        return true;
      }
      substrate::asmPause();
    }
    ret = stealOutsideSocket(poor, HALF);
    if (ret) {
      return true;
    }
    substrate::asmPause();

    return ret;
  }

 private:
  R range;
  F func;
  FD dou_func;
  FS stb_func;
  const char *loopname;
  Diff_ty chunk_size;
  PerThreadStorage<ThreadContext> workers;

  substrate::TerminationDetection &term;

 public:
  DoAllDoubleExec(const R &_range, F _func, const ArgsTuple &argsTuple)
      : range(_range),
        func(_func),
        loopname(getLoopName(argsTuple)),
        chunk_size(get_trait_value<chunk_size_tag>(argsTuple).value),
        term(substrate::getSystemTermination(activeThreads)) {}

  DoAllDoubleExec(
      const R &_range, F _func, FD _dou_func, FS _stb_func,
      const ArgsTuple &argsTuple)
      : range(_range),
        func(_func),
        dou_func(_dou_func),
        stb_func(_stb_func),
        loopname(getLoopName(argsTuple)),
        chunk_size(get_trait_value<chunk_size_tag>(argsTuple).value),
        term(substrate::getSystemTermination(activeThreads)) {}

  // parallel call
  void initThread(void) {
    term.initializeThread();
    unsigned id = ThreadPool::getTID();
    *workers.getLocal(id) =
        ThreadContext(id, range.local_begin(), range.local_end());
  }

  ~DoAllDoubleExec() {}

  void operator()(void) {
    ThreadContext &ctx = *workers.getLocal();
    unsigned id = ThreadPool::getTID();
    while (true) {
      // printf("local size: %ld\n", ctx.m_size);
      // ctx = ThreadContext(id, range.local_begin(), range.local_end());
      ctx.doWork(func, chunk_size);
      bool stole = trySteal(ctx);
      if (stole) {
        continue;
      } else {
        break;
      }
    }
    // dou_func();
  }
};

template <bool _STEAL>
struct ChooseDoAllDoubleImpl {
  template <typename R, typename F, typename FD, typename FS, typename ArgsT>
  static void call(
      const R &range, F &&func, FD &&dou_func, FS &&stb_func,
      const ArgsT &argsTuple) {
    DoAllDoubleExec<
        R, runtime::OperatorReferenceType<decltype(std::forward<F>(func))>,
        runtime::OperatorReferenceType<decltype(std::forward<FD>(dou_func))>,
        runtime::OperatorReferenceType<decltype(std::forward<FS>(stb_func))>,
        ArgsT>
        exec(
            range, std::forward<F>(func), std::forward<FD>(dou_func),
            std::forward<FS>(stb_func), argsTuple);
    substrate::Barrier &barrier = substrate::getBarrier(activeThreads);
    getThreadPool().run(
        activeThreads, [&exec](void) { exec.initThread(); }, std::ref(barrier),
        std::ref(exec), std::ref(dou_func));
  }
};

template <typename R, typename F, typename FD, typename FS, typename ArgsTuple>
void do_all_double_gen(
    const R &range, F &&func, FD &&dou_func, FS &&stb_func,
    const ArgsTuple &argsTuple) {
  auto argsT = std::tuple_cat(
      argsTuple, get_default_trait_values(
                     argsTuple, std::make_tuple(chunk_size_tag{}),
                     std::make_tuple(chunk_size<>{})));

  using ArgsT = decltype(argsT);

  constexpr bool STEAL = has_trait<steal_tag, ArgsT>();

  runtime::OperatorReferenceType<decltype(std::forward<F>(func))> func_ref =
      func;

  runtime::OperatorReferenceType<decltype(std::forward<FD>(dou_func))>
      dou_func_ref = dou_func;

  runtime::OperatorReferenceType<decltype(std::forward<FS>(stb_func))>
      stb_func_ref = stb_func;

  ChooseDoAllDoubleImpl<STEAL>::call(
      range, func_ref, dou_func_ref, stb_func_ref, argsT);
}

#endif  // DOALL_DOUBLE_H
