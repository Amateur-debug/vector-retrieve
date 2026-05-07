
#include <assert.h>
#include <stdlib.h>

#include <atomic>
#include <fstream>
#include <list>
#include <map>
#include <memory>
#include <random>
#include <unordered_set>

#include "anns/exec_query.h"
#include "coromem/include/backend.h"
#include "coromem/include/galois/loops.h"
#include "coromem/include/runtime/range.h"
#include "index/graph_index.h"
#include "worklists/balance_queue.h"
#include "worklists/chunk.h"
#include "worklists/lockfreeQ.h"

using namespace std;

struct AA {
  AA* next;

 public:
  AA(int _id) : next(nullptr), id(_id) {}
  AA*& getNext() { return next; }
  AA* const& getNext() const { return next; }

  int id;
};

// ./tests/anns/test_bq
int main() {
  SharedMem coromem;

  LockFreeQueue<int> aaa;
  aaa.push(1);

  auto t = aaa.pop();
  if (t) {
    cout << *t << endl;
  } else {
    cout << "empty" << endl;
  }
  //   BalanceQueue<AA> bq;

  //   int a = 1;
  //   int b = 2;
  //   AA aa(1);
  //   AA bb(2);

  //   bq.push(&aa);
  //   printf("bq size: %d\n", bq.size());

  //   auto tt = bq.pop();
  //   if (tt) {
  //     cout << (**tt).id << endl;
  //   } else {
  //     cout << "empty" << endl;
  //   }

  return 0;
}