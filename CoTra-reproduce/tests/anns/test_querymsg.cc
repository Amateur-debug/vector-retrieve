#include "anns/query_msg.h"
#include "rdma/rdma_param.h"

// ./tests/test_querymsg
int main() {
//   QueryMsg<uint32_t> msg;

//   msg.query_id = 1;
//   msg.current_node_id = 100;
//   msg.vector[0] = '9';
//   msg.vector[1] = '7';
//   msg.state = START;
//   msg.visit_hash.CheckAndSet(1);
//   msg.visit_hash.CheckAndSet(5);
//   msg.top_candidates.push(std::pair<uint32_t, uint32_t>(7, 4));
//   msg.candidate_set.push(std::pair<uint32_t, uint32_t>(4, 3));
//   msg.candidate_set.push(std::pair<uint32_t, uint32_t>(3, 4));
//   msg.result.push(std::pair<uint32_t, uint64_t>(1, 2));
// #ifdef PROFILER
//   msg.profiler.start("111");
// #endif
//   for (int i = 0; i < 1e7; i++) {
//     i ^= (i * i) % 100;
//   }
// #ifdef PROFILER
//   msg.profiler.end("111");

//   msg.profiler.count("222", 3);
//   msg.profiler.report();
// #endif
//   printf("hash table size: %d\n", (msg.visit_hash.m_poolSize + 1) * 2);

//   char* ptr = (char*)malloc(MAX_QUERYBUFFER_SIZE);
//   msg.serialize(ptr);

//   QueryMsg<uint32_t>* newmsg = QueryMsg<uint32_t>::deserialize(ptr);

//   printf("newmsg.query_id %d\n", newmsg->query_id);
//   printf("newmsg.vector %c\n", newmsg->vector[1]);
//   printf("newmsg.visit_hash %d\n", newmsg->visit_hash.CheckAndSet(1));
//   printf("newmsg.visit_hash %d\n", newmsg->visit_hash.CheckAndSet(2));
//   while (newmsg->candidate_set.size() > 0) {
//     printf(
//         "cnd %d %d\n", newmsg->candidate_set.top().first,
//         newmsg->candidate_set.top().second);
//     newmsg->candidate_set.pop();
//   }

// #ifdef PROFILER
//   newmsg->profiler.report();
// #endif
  return 0;
}