#ifndef COROGRAPH_RUNTIME_EXECUTOR_ONEACH_H
#define COROGRAPH_RUNTIME_EXECUTOR_ONEACH_H

#include "../galois/traits.h"
#include "../galois/config.h"
#include "../runtime/optref_types.h"
#include "../threadpool.h"

namespace runtime {

template <typename FunctionTy, typename ArgsTy>
inline void on_each_impl(FunctionTy &&fn, const ArgsTy &argsTuple) {

  static_assert(!has_trait<char *, ArgsTy>(), "old loopname");
  static_assert(!has_trait<char const *, ArgsTy>(), "old loopname");

  static constexpr bool NEEDS_STATS = has_trait<loopname_tag, ArgsTy>();
  static constexpr bool MORE_STATS =
      NEEDS_STATS && has_trait<more_stats_tag, ArgsTy>();

  const char *const loopname = getLoopName(argsTuple);

  const auto numT = getActiveThreads();

  OperatorReferenceType<decltype(std::forward<FunctionTy>(fn))> fn_ref = fn;

  auto runFun = [&] { fn_ref(ThreadPool::getTID(), numT); };

  getThreadPool().run(numT, runFun);
}

template <typename FunctionTy, typename TupleTy>
inline void on_each_gen(FunctionTy &&fn, const TupleTy &tpl) {
  on_each_impl(std::forward<FunctionTy>(fn), tpl);
}

} // end namespace runtime

#endif
