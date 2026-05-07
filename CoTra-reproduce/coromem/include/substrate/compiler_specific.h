#ifndef COROGRAPH_COMPILERSPECIFIC_H
#define COROGRAPH_COMPILERSPECIFIC_H

#include "../galois/config.h"

namespace substrate {

inline static void asmPause() {
#if defined(__i386__) || defined(__amd64__)
  //  __builtin_ia32_pause();
  asm volatile("pause");
#endif
}

inline static void compilerBarrier() { asm volatile("" ::: "memory"); }

// xeons have 64 byte cache lines, but will prefetch 2 at a time
constexpr int COROMEM_CACHE_LINE_SIZE = 128;

#if defined(__INTEL_COMPILER)
#define COROMEM_ATTRIBUTE_NOINLINE __attribute__((noinline))

#elif defined(__GNUC__)
#define COROMEM_ATTRIBUTE_NOINLINE __attribute__((noinline))

#else
#define COROMEM_ATTRIBUTE_NOINLINE
#endif

} // namespace substrate

#endif
