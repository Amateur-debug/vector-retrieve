#ifndef COROGRAPH_SHAREMEM_H
#define COROGRAPH_SHAREMEM_H

#include "hugepage.h"
#include "substrate/barrier.h"
#include "substrate/termination.h"
#include "threadpool.h"

class SharedMem {
public:

  ThreadPool tp;

  std::unique_ptr<substrate::LocalTerminationDetection<>> m_termPtr;
  std::unique_ptr<substrate::BarrierInstance<>> m_biPtr;

  PageAllocState<> *m_pa;


  SharedMem();

  ~SharedMem();

  SharedMem(const SharedMem &) = delete;
  SharedMem &operator=(const SharedMem &) = delete;

  SharedMem(SharedMem &&) = delete;
  SharedMem &operator=(SharedMem &&) = delete;
};

#endif
