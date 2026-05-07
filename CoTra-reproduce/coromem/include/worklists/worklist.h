#ifndef COROGRAPH_WORKLIST_H
#define COROGRAPH_WORKLIST_H

#include <boost/iterator/iterator_facade.hpp>
#include <optional>

#include "../galois/config.h"
#include "../substrate/ptr_lock.h"

namespace worklists {

template <typename T>
class ConExtListNode {
  T *next;

 public:
  ConExtListNode() : next(0) {}
  T *&getNext() { return next; }
  T *const &getNext() const { return next; }
};

template <typename T>
class ConExtIterator : public boost::iterator_facade<
                           ConExtIterator<T>, T, boost::forward_traversal_tag> {
  friend class boost::iterator_core_access;
  T *at;

  template <typename OtherTy>
  bool equal(const ConExtIterator<OtherTy> &o) const {
    return at == o.at;
  }

  T &dereference() const { return *at; }
  void increment() { at = at->getNext(); }

 public:
  ConExtIterator() : at(0) {}

  template <typename OtherTy>
  ConExtIterator(const ConExtIterator<OtherTy> &o) : at(o.at) {}

  explicit ConExtIterator(T *x) : at(x) {}
};

template <typename T, bool concurrent>
class MessageQueue {
  substrate::PtrLock<T> head;
  T *tail;
  // 0 for empty (not in list or execution), 1 for not empty
  bool blockState;
  uint64_t numa_id;  // record the numa id of partition

 public:
  typedef ConExtListNode<T> ListNode;

  MessageQueue() : tail(0), blockState(0), numa_id(1) {}

  bool empty() const { return !tail; }

  bool push(T *C) {  // return true: need to push to gatherQueue
    head.lock();
    C->getNext() = 0;
    if (tail) {
      tail->getNext() = C;
      tail = C;
      head.unlock();
      return false;
    } else {
      tail = C;
      bool ret = !blockState;
      blockState = true;  // will push in list
      head.unlock_and_set(C);
      if (ret) {
        return numa_id;  // if not in list, then push.
      } else {
        return 0;
      }
    }
  }

  void setNuma(uint64_t pop_id) { numa_id = pop_id + 1; }

  T *pop() {
    // lock free Fast path empty case
    //            if (empty())
    //                return 0;
    head.lock();
    T *C = head.getValue();
    if (!C) {
      blockState = false;  // if pop the final element
      head.unlock();
      return 0;
    }
    if (tail == C) {
      tail = 0;
      head.unlock_and_clear();
    } else {
      head.unlock_and_set(C->getNext());
      C->getNext() = 0;
    }

    return C;
  }
};

template <typename T, bool concurrent>
class ConExtLinkedQueue {
  substrate::PtrLock<T> head;
  T *tail;
  size_t sz;

 public:
  typedef ConExtListNode<T> ListNode;

  ConExtLinkedQueue() : tail(0) {}

  bool empty() const { return !tail; }

  void push(T *C) {
    head.lock();
    sz++;
    C->getNext() = 0;
    if (tail) {
      tail->getNext() = C;
      tail = C;
      head.unlock();
    } else {
      tail = C;
      head.unlock_and_set(C);
    }
  }

  std::optional<T *> pop() {
    // lock free Fast path empty case
    if (empty()) return std::nullopt;

    head.lock();
    sz--;
    T *C = head.getValue();
    if (!C) {
      head.unlock();
      return std::nullopt;
    }
    if (tail == C) {
      tail = 0;
      assert(!C->getNext());
      head.unlock_and_clear();
    } else {
      head.unlock_and_set(C->getNext());
      C->getNext() = 0;
    }
    return std::nullopt;
  }

  size_t size() const { return sz; }

  //! iterators not safe with concurrent modifications
  typedef T value_type;
  typedef T &reference;
  typedef ConExtIterator<T> iterator;
  typedef ConExtIterator<const T> const_iterator;

  iterator begin() { return iterator(head.getValue()); }
  iterator end() { return iterator(); }

  const_iterator begin() const { return const_iterator(head.getValue()); }
  const_iterator end() const { return const_iterator(); }
};

template <typename T>
struct DummyIndexer {
  unsigned operator()(const T &) { return 0; }
};

}  // namespace worklists

#endif
