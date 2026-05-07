#pragma once

#include <atomic>
#include <condition_variable>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>

#include "coromem/include/substrate/simple_lock.h"

template <typename T>
class LockFreeQueue {
 private:
  std::queue<T> queue_;
  mutable substrate::SimpleLock lock_; 

 public:
  LockFreeQueue() = default;


  LockFreeQueue(const LockFreeQueue&) = delete;
  LockFreeQueue& operator=(const LockFreeQueue&) = delete;


  void push(const T& value) {
    lock_.lock();
    queue_.push(value);
    lock_.unlock();
  }

  void push(T&& value) {
    lock_.lock();
    queue_.push(std::move(value));
    lock_.unlock();
  }

  void push(std::vector<T>& values) {
    lock_.lock();
    for (auto& value : values) {
      queue_.push(std::move(value));
    }
    lock_.unlock();
  }

  template <typename... Args>
  void emplace_back(Args&&... args) {
    lock_.lock();
    queue_.emplace(std::forward<Args>(args)...);
    lock_.unlock();
  }

  std::optional<T> pop() {
    lock_.lock();
    if (queue_.empty()) {
      lock_.unlock();
      return std::nullopt;
    }
    T value = std::move(queue_.front());
    queue_.pop();
    lock_.unlock();
    return value;
  }

  size_t size() const {
    lock_.lock();
    size_t size = queue_.size();
    lock_.unlock();
    return size;
  }

  bool empty() const {
    lock_.lock();
    bool is_empty = queue_.empty();
    lock_.unlock();
    return is_empty;
  }
};

