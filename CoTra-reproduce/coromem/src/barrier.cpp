#include "substrate/barrier.h"

// anchor vtable
substrate::Barrier::~Barrier() {}

// galois::substrate::Barrier& galois::substrate::getSystemBarrier(unsigned
// activeThreads) {
//  return benchmarking::getTopoBarrier(activeThreads);
//}

static substrate::BarrierInstance<> *BI = nullptr;

void substrate::setBarrierInstance(BarrierInstance<> *bi) {
  // GALOIS_ASSERT(!(bi && BI), "Double initialization of BarrierInstance");
  BI = bi;
}

substrate::Barrier &substrate::getBarrier(unsigned numT) {
  // GALOIS_ASSERT(BI, "BarrierInstance not initialized");
  return BI->get(numT);
}
