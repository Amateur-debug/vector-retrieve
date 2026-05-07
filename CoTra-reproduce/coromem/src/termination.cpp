#include "substrate/termination.h"

// vtable anchoring
substrate::TerminationDetection::~TerminationDetection(void) {}

static substrate::TerminationDetection *TERM = nullptr;

void substrate::setTermDetect(substrate::TerminationDetection *t) {
  // GALOIS_ASSERT(!(TERM && t), "Double initialization of
  // TerminationDetection");
  TERM = t;
}

substrate::TerminationDetection &
substrate::getSystemTermination(unsigned activeThreads) {
  TERM->init(activeThreads);
  return *TERM;
}
