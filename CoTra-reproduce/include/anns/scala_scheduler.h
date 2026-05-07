#pragma once
#include <stdlib.h>

#include <cassert>
#include <deque>
#include <vector>

#include "anns/anns_param.h"

struct ScheduleMsg {
  size_t query_id;
  std::vector<uint32_t> core_machine;

  ScheduleMsg(size_t qid, std::vector<uint32_t>& core_m)
      : query_id(qid), core_machine(core_m) {}

  ScheduleMsg(const void* src) {
    size_t offset = 0;
    memcpy(&query_id, static_cast<const char*>(src) + offset, sizeof(query_id));
    offset += sizeof(query_id);
    size_t core_machine_size;
    memcpy(
        &core_machine_size, static_cast<const char*>(src) + offset,
        sizeof(core_machine_size));
    offset += sizeof(core_machine_size);
    core_machine.clear();
    for (size_t i = 0; i < core_machine_size; ++i) {
      uint32_t m;
      memcpy(&m, static_cast<const char*>(src) + offset, sizeof(m));
      offset += sizeof(m);
      core_machine.emplace_back(m);
    }
  }

  /**
   * Send core partiton info to leader machine scheduler.
   */
  size_t serialize(void* ptr) {
    size_t offset = 0;
    memcpy(static_cast<char*>(ptr) + offset, &query_id, sizeof(query_id));
    offset += sizeof(query_id);
    size_t core_machine_size = core_machine.size();
    memcpy(
        static_cast<char*>(ptr) + offset, &core_machine_size,
        sizeof(core_machine_size));
    offset += sizeof(core_machine_size);
    for (auto m : core_machine) {
      memcpy(static_cast<char*>(ptr) + offset, &m, sizeof(m));
      offset += sizeof(m);
    }
    return offset;
  }
};
/**
 * ScalaScheduler: schedule query computation migration.
 */
class ScalaScheduler {
 public:
  void generate_schedule_plan(std::vector<ScheduleMsg>& schedule_query) {
    schedule_query.clear();
    return;
  }
};