#ifndef COROGRAPH_COROBJ_H
#define COROGRAPH_COROBJ_H

#include <coroutine>

template <typename T>
struct Corobj {
  struct promise_type;
  using handle_type = std::coroutine_handle<promise_type>;
  struct promise_type {
    T value_;
    std::exception_ptr exception_;
    Corobj get_return_object() {
      return Corobj(handle_type::from_promise(*this));
    }
    void unhandled_exception() { exception_ = std::current_exception(); }
    std::suspend_never initial_suspend() { return {}; }
    std::suspend_always final_suspend() noexcept { return {}; }
    template <std::convertible_to<T> From>  // C++20 concept
    std::suspend_always yield_value(From&& from) {
      value_ = std::forward<From>(from);
      return {};
    }
    void return_void() {}
  };
  std::coroutine_handle<promise_type> h_;
  Corobj() {}
  Corobj(handle_type h) : h_(h) {}

  Corobj(const Corobj&) = delete;
  Corobj& operator=(const Corobj&) = delete;

  Corobj(Corobj&& other) noexcept : h_(std::exchange(other.h_, {})) {}
  Corobj& operator=(Corobj&& other) noexcept {
    if (this != &other) {
      if (h_) h_.destroy();
      h_ = std::exchange(other.h_, {});
    }
    return *this;
  }

  ~Corobj() {
    if (h_) h_.destroy();
  }

  explicit operator bool() { return !h_.done(); }

  // Overload operator() with no parameters
  T operator()() {
    h_();
    return std::move(h_.promise().value_);
  }
  // Overload operator() to accept a callable (lambda or function)
  template <typename Callable>
  T operator()(Callable&& func) {
    if (func) {
      // Call the function inside the coroutine
      h_.promise().value_ = func();
    }
    h_();
    return std::move(h_.promise().value_);
  }
};

#endif  // COROGRAPH_COROBJ_H
