#ifndef COROGRAPH_CACHELINESTORAGE_H
#define COROGRAPH_CACHELINESTORAGE_H

#include <utility>

#include "../galois/config.h"
#include "compiler_specific.h"

namespace substrate {

// Store an item with padding
template <typename T> struct CacheLineStorage {
  alignas(COROMEM_CACHE_LINE_SIZE) T data;

  char buffer[COROMEM_CACHE_LINE_SIZE - (sizeof(T) % COROMEM_CACHE_LINE_SIZE)];
  // static_assert(sizeof(T) < COROMEM_CACHE_LINE_SIZE, "Too large a type");

  CacheLineStorage() : data() {}
  CacheLineStorage(const T &v) : data(v) {}

  template <typename A>
  explicit CacheLineStorage(A &&v) : data(std::forward<A>(v)) {}

  explicit operator T() { return data; }

  T &get() { return data; }
  template <typename V> CacheLineStorage &operator=(const V &v) {
    data = v;
    return *this;
  }
};

} // namespace substrate

#endif
