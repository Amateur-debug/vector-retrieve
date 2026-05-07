#include "share_mem.h"
#include "substrate/barrier.h"
#include "substrate/termination.h"

SharedMem::SharedMem() {
  setThreadPool(&tp);

  m_biPtr = std::make_unique<substrate::BarrierInstance<>>();
  m_termPtr = std::make_unique<substrate::LocalTerminationDetection<>>();

  setBarrierInstance(m_biPtr.get());
  setTermDetect(m_termPtr.get());
  m_pa = new PageAllocState();
  setPagePoolState(m_pa);
}

SharedMem::~SharedMem() {
  setPagePoolState(nullptr);
  substrate::setTermDetect(nullptr);
  substrate::setBarrierInstance(nullptr);

  m_termPtr.reset();
  m_biPtr.reset();

  setThreadPool(nullptr);
}
