#ifndef COROGRAPH_NUMAMEM
#define COROGRAPH_NUMAMEM

#include <cstddef>
#include <memory>
#include <vector>

#include "../galois/config.h"

namespace substrate {

struct largeFreer {
  size_t bytes;
  void operator()(void *ptr) const;
};

typedef std::unique_ptr<void, largeFreer> LAptr;

LAptr largeMallocLocal(size_t bytes);    // fault in locally
LAptr largeMallocFloating(size_t bytes); // leave numa mapping undefined
// fault in interleaved mapping
LAptr largeMallocInterleaved(size_t bytes, unsigned numThreads);
// fault in block interleaved mapping
LAptr largeMallocBlocked(size_t bytes, unsigned numThreads);

// fault in specified regions for each thread (threadRanges)
template <typename RangeArrayTy>
LAptr largeMallocSpecified(size_t bytes, uint32_t numThreads,
                           RangeArrayTy &threadRanges, size_t elementSize);

} // namespace substrate

#endif // GALOIS_SUBSTRATE_NUMAMEM
