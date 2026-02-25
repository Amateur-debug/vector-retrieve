#pragma once

#include <cstdarg>
#include <cstdio>
#include <mutex>
#include <string>
#include <utility>

// 日志级别控制
// VR_LOG_LEVEL_DEBUG=0: debug级别，默认不打印
// VR_LOG_LEVEL_INFO=1: info级别，默认打印
// 可以通过编译选项 -DVR_LOG_LEVEL=1 启用debug日志

#ifndef VR_LOG_LEVEL
#define VR_LOG_LEVEL 0  // 默认: debug级别，不打印debug日志
#endif

namespace vrlog
{

enum LogLevel
{
    LOG_LEVEL_DEBUG = 0,
    LOG_LEVEL_INFO = 1
};

inline std::mutex &log_mutex()
{
    static std::mutex m;
    return m;
}

inline void log_printf(const char *fmt, ...)
{
    std::lock_guard<std::mutex> lk(log_mutex());
    va_list args;
    va_start(args, fmt);
    std::vfprintf(stderr, fmt, args);
    std::fprintf(stderr, "\n");
    va_end(args);
}

inline int &current_log_level()
{
    static int level = VR_LOG_LEVEL;
    return level;
}

inline void set_log_level(int level)
{
    current_log_level() = level;
}

} // namespace vrlog

#include <iostream>

// INFO级别日志 - 默认打印
#define LOG_INFOF(fmt, ...) ::vrlog::log_printf((fmt), ##__VA_ARGS__)
#define LOG_INFO_STREAM(expr)                          \
    do                                                 \
    {                                                  \
        std::lock_guard<std::mutex> _lk(::vrlog::log_mutex()); \
        std::cerr << expr << std::endl;                \
    } while (0)

// DEBUG级别日志 - 根据VR_LOG_LEVEL决定是否打印
#if VR_LOG_LEVEL >= 1
#define LOG_DEBUGF(fmt, ...) ::vrlog::log_printf((fmt), ##__VA_ARGS__)
#define LOG_DEBUG_STREAM(expr)                         \
    do                                                 \
    {                                                  \
        std::lock_guard<std::mutex> _lk(::vrlog::log_mutex()); \
        std::cerr << expr << std::endl;                \
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

