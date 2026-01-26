#pragma once

#include <cstdarg>
#include <cstdio>
#include <mutex>
#include <string>
#include <utility>

#define DEBUG 1

#ifdef DEBUG

namespace vrlog
{
inline std::mutex &log_mutex()
{
    static std::mutex m;
    return m;
}

inline void debug_printf(const char *fmt, ...)
{
    std::lock_guard<std::mutex> lk(log_mutex());
    va_list args;
    va_start(args, fmt);
    std::vfprintf(stderr, fmt, args);
    std::fprintf(stderr, "\n");
    va_end(args);
}
}

#define LOG_DEBUGF(fmt, ...) ::vrlog::debug_printf((fmt), ##__VA_ARGS__)

#include <iostream>

#define LOG_DEBUG_STREAM(expr)                      \
    do                                              \
    {                                               \
        std::lock_guard<std::mutex> _lk(::vrlog::log_mutex()); \
        std::cerr << expr << std::endl;             \
    } while (0)

#else

#define LOG_DEBUGF(fmt, ...) \
    do                       \
    {                        \
    } while (0)

#define LOG_DEBUG_STREAM(expr) \
    do                          \
    {                           \
    } while (0)

#endif

