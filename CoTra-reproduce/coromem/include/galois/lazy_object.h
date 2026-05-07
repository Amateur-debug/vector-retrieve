#ifndef COROMEM_LAZYOBJECT_H
#define COROMEM_LAZYOBJECT_H

#include <type_traits>
#include <utility>

#include "config.h"

/**
 * Single object with specialization for void type. To take advantage of empty
 * member optimization, users should subclass this class, otherwise the
 * compiler will insert non-zero padding for fields (even when empty).
 */
template <typename T> class StrictObject {
  T data;

public:
  typedef T value_type;
  typedef T &reference;
  typedef const T &const_reference;
  const static bool has_value = true;

  StrictObject() {}
  StrictObject(const_reference t) : data(t) {}
  const_reference get() const { return data; }
  reference get() { return data; }
};

template <> struct StrictObject<void> {
  typedef void *value_type;
  typedef void *reference;
  typedef void *const_reference;
  const static bool has_value = false;

  StrictObject() {}
  StrictObject(const_reference) {}
  reference get() const { return 0; }
};

/**
 * Single (uninitialized) object with specialization for void type. To take
 * advantage of empty member optimization, users should subclass this class,
 * otherwise the compiler will insert non-zero padding for fields (even when
 * empty).
 */
template <typename T> class LazyObject {
  typedef
      typename std::aligned_storage<sizeof(T),
                                    std::alignment_of<T>::value>::type CharData;

  union Data {
    CharData buf;
    T value_;

    // Declare constructor explicitly because Data must be default
    // constructable regardless of the constructability of T.
    Data() {}  // NOLINT(modernize-use-equals-default)
    ~Data() {} // NOLINT(modernize-use-equals-default)

    T &value() { return value_; }
    const T &value() const { return value_; }
  };

  Data data_;

  T *cast() { return &data_.value(); }
  const T *cast() const { return &data_.value(); }

public:
  typedef T value_type;
  typedef T &reference;
  typedef const T &const_reference;
  const static bool has_value = true;
  
  struct size_of {
    const static size_t value = sizeof(T);
  };

  void destroy() { cast()->~T(); }
  void construct(const_reference x) { new (cast()) T(x); }

  template <typename... Args> void construct(Args &&... args) {
    new (cast()) T(std::forward<Args>(args)...);
  }

  const_reference get() const { return *cast(); }
  reference get() { return *cast(); }
};

template <> struct LazyObject<void> {
  typedef void *value_type;
  typedef void *reference;
  typedef void *const_reference;
  const static bool has_value = false;
  struct size_of {
    const static size_t value = 0;
  };

  void destroy() {}
  void construct(const_reference) {}

  template <typename... Args> void construct(Args &&...) {}

  const_reference get() const { return 0; }
};

#endif
