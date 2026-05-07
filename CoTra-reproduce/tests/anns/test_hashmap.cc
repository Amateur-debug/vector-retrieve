#include "anns/hash_table.h"

int main() {
  OptHashPosVector hash;

  hash.CheckAndSet(1);
  if (hash.CheckAndSet(1) == true) {
    printf("1 has been here\n");
  }
  if (hash.CheckAndSet(2) == false) {
    printf("2 havent been here\n");
  }

  return 0;
}