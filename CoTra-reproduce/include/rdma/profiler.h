#pragma once
#include <iostream>
#include <map>
#include <vector>

#include "get_clock.h"

class Profiler {
  // Counter
  std::map<std::string, uint64_t> rec_num;
  // Timer
  std::map<std::string, cycles_t> rec_sum;

  // Timer sum, not serialize
  double cycles_to_units;
  std::map<std::string, double> sum_timer;

  std::map<std::string, cycles_t> rec_start;

 public:
  Profiler() {}

  void get_mhz() { cycles_to_units = get_cpu_mhz(1); }

  Profiler& operator+=(const Profiler& other) {
    for (const auto& entry : other.rec_num) {
      if (!rec_num[entry.first]) rec_num[entry.first] = 0;
      rec_num[entry.first] += entry.second;
    }

    for (const auto& entry : other.rec_sum) {
      if (!sum_timer[entry.first]) sum_timer[entry.first] = 0.0;
      sum_timer[entry.first] += entry.second / cycles_to_units;
    }

    return *this;
  }

  void clear() {
    rec_num.clear();
    rec_sum.clear();
    sum_timer.clear();
    rec_start.clear();
  }

  void count(std::string str, uint64_t num = 1) {
    if (!rec_num[str]) {
      rec_num[str] = 0;
    }
    rec_num[str] += num;
  }

  void start(std::string str) {
    if (!rec_start[str]) {
      rec_start[str] = get_cycles();
    } else {
      printf("Error: timer %s not end.\n", str.c_str());
    }
    return;
  }

  void end(std::string str) {
    if (!rec_start[str]) {
      printf("Error: timer %s not start.\n", str.c_str());
    } else {
      if (!rec_sum[str]) {
        rec_sum[str] = 0;
        sum_timer[str] = 0.0;
      }
      if (!rec_num[str]) {
        rec_num[str] = 0;
      }
      auto diff = get_cycles() - rec_start[str];
      rec_sum[str] += diff;
      sum_timer[str] += diff / cycles_to_units;
      rec_num[str] += 1;
      rec_start[str] = NULL;
    }
    return;
  }

  void report() {
    printf(
        ">> %-16s%-10s%-16s%-12s%-12s\n", "item", "num", "total(usec)",
        "avg(usec)", "throughput");
    for (auto& it : rec_num) {
      // If record the time.
      if (rec_sum[it.first]) {
        printf(
            ">> %-16s%-10llu%-16.3f%-12.3f%-12.3f/s\n", it.first.c_str(),
            it.second, rec_sum[it.first] / cycles_to_units,
            rec_sum[it.first] / cycles_to_units / (double)it.second,
            it.second * 1000000.0 / (rec_sum[it.first] / cycles_to_units));
      } else {
        // Only record the number.
        printf(
            ">> %-16s%-10llu%-16s%-12s%-12s\n", it.first.c_str(), it.second,
            "--", "--", "--");
      }
    }
  }

  void report_sum() {
    printf(
        "------------------------ Profiler info ------------------------\n"
        ">> %-16s%-10s%-16s%-12s%s\n",
        "item", "num", "total(usec)", "avg(usec)", "ratio");
    sum_timer["f_wait_time"] = sum_timer["g_overall"] - sum_timer["g_query_Q"] -
                               sum_timer["g_task_Q"] - sum_timer["g_async_Q"] -
                               sum_timer["g_result_Q"] -
                               sum_timer["g_subquery_Q"] -
                               sum_timer["g_poll_proc"] - sum_timer["g_steal"];
    rec_num["f_wait_time"] = 1;
    sum_timer["f_task_local"] = sum_timer["g_subquery_Q"] +
                                sum_timer["g_poll_proc"] + sum_timer["g_steal"];
    rec_num["f_task_local"] = 1;
    sum_timer["f_task_remote"] =
        sum_timer["g_task_Q"] + sum_timer["g_result_Q"] +
        sum_timer["g_query_Q"] + sum_timer["g_async_Q"];
    rec_num["f_task_remote"] = 1;
    // sum_timer["f_post_time"] =
    //     sum_timer["g_post_result"] + sum_timer["g_post_task"];
    // rec_num["f_post_time"] = 1;

    for (auto& it : rec_num) {
      // If record the time.
      if (sum_timer[it.first]) {
        printf(
            ">> %-16s%-10llu%-16.3f%-12.3f%.3f%%\n", it.first.c_str(),
            it.second, sum_timer[it.first],
            sum_timer[it.first] / (double)it.second,
            sum_timer[it.first] * 100.0 / sum_timer["g_overall"]);
      } else {
        // Only record the number.
        printf(
            ">> %-16s%-10llu%-16s%-12s%s\n", it.first.c_str(), it.second, "--",
            "--", "--");
      }
    }
    printf("---------------------------------------------------------------\n");
  }

  void serialize(void* dest, size_t& offset) const {
    // Serialize rec_sum
    size_t rec_sum_size = rec_sum.size();
    memcpy(
        static_cast<char*>(dest) + offset, &rec_sum_size, sizeof(rec_sum_size));
    offset += sizeof(rec_sum_size);

    for (const auto& pair : rec_sum) {
      size_t key_len = pair.first.size();
      memcpy(static_cast<char*>(dest) + offset, &key_len, sizeof(key_len));
      offset += sizeof(key_len);

      memcpy(static_cast<char*>(dest) + offset, pair.first.data(), key_len);
      offset += key_len;

      memcpy(
          static_cast<char*>(dest) + offset, &pair.second, sizeof(pair.second));
      offset += sizeof(pair.second);
    }

    // Serialize rec_num
    size_t rec_num_size = rec_num.size();
    memcpy(
        static_cast<char*>(dest) + offset, &rec_num_size, sizeof(rec_num_size));
    offset += sizeof(rec_num_size);

    for (const auto& pair : rec_num) {
      size_t key_len = pair.first.size();
      memcpy(static_cast<char*>(dest) + offset, &key_len, sizeof(key_len));
      offset += sizeof(key_len);

      memcpy(static_cast<char*>(dest) + offset, pair.first.data(), key_len);
      offset += key_len;

      memcpy(
          static_cast<char*>(dest) + offset, &pair.second, sizeof(pair.second));
      offset += sizeof(pair.second);
    }

    // Serialize rec_start
    size_t rec_start_size = rec_start.size();
    memcpy(
        static_cast<char*>(dest) + offset, &rec_start_size,
        sizeof(rec_start_size));
    offset += sizeof(rec_start_size);

    for (const auto& pair : rec_start) {
      size_t key_len = pair.first.size();
      memcpy(static_cast<char*>(dest) + offset, &key_len, sizeof(key_len));
      offset += sizeof(key_len);

      memcpy(static_cast<char*>(dest) + offset, pair.first.data(), key_len);
      offset += key_len;

      memcpy(
          static_cast<char*>(dest) + offset, &pair.second, sizeof(pair.second));
      offset += sizeof(pair.second);
    }

    // Serialize sum_timer
    size_t sum_timer_size = sum_timer.size();
    memcpy(
        static_cast<char*>(dest) + offset, &sum_timer_size,
        sizeof(sum_timer_size));
    offset += sizeof(sum_timer_size);

    for (const auto& pair : sum_timer) {
      size_t key_len = pair.first.size();
      memcpy(static_cast<char*>(dest) + offset, &key_len, sizeof(key_len));
      offset += sizeof(key_len);

      memcpy(static_cast<char*>(dest) + offset, pair.first.data(), key_len);
      offset += key_len;

      memcpy(
          static_cast<char*>(dest) + offset, &pair.second, sizeof(pair.second));
      offset += sizeof(pair.second);
    }
  }

  void deserialize(const void* src, size_t& offset) {
    // Deserialize rec_sum
    size_t rec_sum_size;
    memcpy(
        &rec_sum_size, static_cast<const char*>(src) + offset,
        sizeof(rec_sum_size));
    offset += sizeof(rec_sum_size);

    for (size_t i = 0; i < rec_sum_size; ++i) {
      size_t key_len;
      memcpy(&key_len, static_cast<const char*>(src) + offset, sizeof(key_len));
      offset += sizeof(key_len);

      std::string key(key_len, '\0');
      memcpy(&key[0], static_cast<const char*>(src) + offset, key_len);
      offset += key_len;

      cycles_t value;
      memcpy(&value, static_cast<const char*>(src) + offset, sizeof(value));
      offset += sizeof(value);

      rec_sum[key] = value;
    }

    // Deserialize rec_num
    size_t rec_num_size;
    memcpy(
        &rec_num_size, static_cast<const char*>(src) + offset,
        sizeof(rec_num_size));
    offset += sizeof(rec_num_size);

    for (size_t i = 0; i < rec_num_size; ++i) {
      size_t key_len;
      memcpy(&key_len, static_cast<const char*>(src) + offset, sizeof(key_len));
      offset += sizeof(key_len);

      std::string key(key_len, '\0');
      memcpy(&key[0], static_cast<const char*>(src) + offset, key_len);
      offset += key_len;

      uint64_t value;
      memcpy(&value, static_cast<const char*>(src) + offset, sizeof(value));
      offset += sizeof(value);

      rec_num[key] = value;
    }

    // Deserialize rec_start
    size_t rec_start_size;
    memcpy(
        &rec_start_size, static_cast<const char*>(src) + offset,
        sizeof(rec_start_size));
    offset += sizeof(rec_start_size);

    for (size_t i = 0; i < rec_start_size; ++i) {
      size_t key_len;
      memcpy(&key_len, static_cast<const char*>(src) + offset, sizeof(key_len));
      offset += sizeof(key_len);

      std::string key(key_len, '\0');
      memcpy(&key[0], static_cast<const char*>(src) + offset, key_len);
      offset += key_len;

      cycles_t value;
      memcpy(&value, static_cast<const char*>(src) + offset, sizeof(value));
      offset += sizeof(value);

      rec_start[key] = value;
    }

    // Serialize sum_timer
    size_t sum_timer_size;
    memcpy(
        &sum_timer_size, static_cast<const char*>(src) + offset,
        sizeof(sum_timer_size));
    offset += sizeof(sum_timer_size);

    for (size_t i = 0; i < sum_timer_size; ++i) {
      size_t key_len;
      memcpy(&key_len, static_cast<const char*>(src) + offset, sizeof(key_len));
      offset += sizeof(key_len);

      std::string key(key_len, '\0');
      memcpy(&key[0], static_cast<const char*>(src) + offset, key_len);
      offset += key_len;

      double value;
      memcpy(&value, static_cast<const char*>(src) + offset, sizeof(value));
      offset += sizeof(value);

      sum_timer[key] = value;
    }
    return;
  }
};
