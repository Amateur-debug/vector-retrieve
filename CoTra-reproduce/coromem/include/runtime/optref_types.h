#ifndef COROGRAPH_OPERATOR_REFERENCE_TYPES_H
#define COROGRAPH_OPERATOR_REFERENCE_TYPES_H

#include "../galois/config.h"

namespace runtime {

// Helper template for getting the appropriate type of
// reference to hold within each executor based off of the
// type of reference that was passed to it.

// Don't accept operators by value.
template <typename FuncTy> struct OperatorReferenceType_impl;

// Const references are propagated.
// If a user supplies a const reference the operator() on the
// given object must be callable with *this passed as const as well.
template <typename FuncNoRef>
struct OperatorReferenceType_impl<FuncNoRef const &> {
  using type = FuncNoRef const &;
};

// Non-const references continue to be non-const.
template <typename FuncNoRef> struct OperatorReferenceType_impl<FuncNoRef &> {
  using type = FuncNoRef &;
};

// Inside each executor store a reference to a received rvalue reference
// and then use that to pass to the various threads. This must be done in
// a way that keeps the rvalue reference alive throughout the duration of
// the parallel loop (as long as the resulting lvalue reference is used
// anywhere).
template <typename FuncNoRef> struct OperatorReferenceType_impl<FuncNoRef &&> {
  using type = FuncNoRef &;
};

template <typename T>
using OperatorReferenceType = typename OperatorReferenceType_impl<T>::type;

} // namespace runtime

#endif // ifndef(GALOIS_RUNTIME_OPERATOR_REFERENCE_TYPES_H)
