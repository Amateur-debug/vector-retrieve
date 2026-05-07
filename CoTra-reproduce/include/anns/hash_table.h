#pragma once

#include <dirent.h>
#include <malloc.h>
#include <unistd.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <type_traits>

#define _rotl(x, n) (((x) << (n)) | ((x) >> (32 - (n))))

class OptHashPosVector {
 public:
  typedef std::int32_t SizeType;

  // Max loop number in one hash block.
  static const int m_maxLoop = 8;

  // Could we use the second hash block.
  bool m_secondHash;

  int m_exp;

  // Max pool size.
  int m_poolSize;

  // Record 2 hash tables.
  // [0~m_poolSize + 1) is the first block.
  // [m_poolSize + 1, 2*(m_poolSize + 1)) is the second block;
  std::unique_ptr<SizeType[]> m_hashTable;

  inline unsigned hash_func2(unsigned idx, int poolSize, int loop) {
    return (idx + loop) & poolSize;
  }

  inline unsigned hash_func(unsigned idx, int poolSize) {
    return ((unsigned)(idx * 99991) + _rotl(idx, 2) + 101) & poolSize;
  }

  OptHashPosVector() : m_secondHash(false), m_exp(2), m_poolSize(8191) {
    Init(8191, 1);
  }

  ~OptHashPosVector() {}

  void Init(SizeType size, int exp) {
    int ex = 0;
    while (size != 0) {
      ex++;
      size >>= 1;
    }
    m_secondHash = true;
    m_exp = exp;
    m_poolSize = (1 << (ex + exp)) - 1;
    m_hashTable.reset(new SizeType[(m_poolSize + 1) * 2]);
    clear();
  }

  void clear() {
    if (!m_hashTable) {
      return;
    }
    if (!m_secondHash) {
      // Clear first block.
      memset(m_hashTable.get(), 0, sizeof(SizeType) * (m_poolSize + 1));
    } else {
      // Clear all blocks.
      m_secondHash = false;
      memset(m_hashTable.get(), 0, 2 * sizeof(SizeType) * (m_poolSize + 1));
    }
  }

  void release_storage() {
    m_hashTable.reset();
    m_secondHash = false;
    m_exp = 0;
    m_poolSize = -1;
  }

  inline int HashTableExponent() const { return m_exp; }

  inline int MaxCheck() const {
    return (1 << (int)(std::log2(m_poolSize + 1) - m_exp));
  }

  inline bool CheckAndSet(SizeType idx) {
    if (!m_hashTable) {
      // Released queries should ignore any late duplicate work instead of
      // touching freed query state.
      return true;
    }
    // Inner Index is begin from 1
    return _CheckAndSet(m_hashTable.get(), m_poolSize, true, idx + 1) == 0;
  }

  inline void DoubleSize() {
    int new_poolSize = ((m_poolSize + 1) << 1) - 1;
    SizeType* new_hashTable = new SizeType[(new_poolSize + 1) * 2];
    memset(new_hashTable, 0, sizeof(SizeType) * (new_poolSize + 1) * 2);

    m_secondHash = false;
    for (int i = 0; i <= new_poolSize; i++)
      if (m_hashTable[i])
        _CheckAndSet(new_hashTable, new_poolSize, true, m_hashTable[i]);

    m_exp++;
    m_poolSize = new_poolSize;
    m_hashTable.reset(new_hashTable);
  }

  inline int _CheckAndSet(
      SizeType* hashTable, int poolSize, bool isFirstTable, SizeType idx) {
    unsigned index = hash_func((unsigned)idx, poolSize);
    for (int loop = 0; loop < m_maxLoop; ++loop) {
      if (!hashTable[index]) {
        // index first match and record it.
        hashTable[index] = idx;
        return 1;
      }
      if (hashTable[index] == idx) {
        // Hit this item in hash table.
        return 0;
      }
      // Get next hash position.
      index = hash_func2(index, poolSize, loop);
    }

    if (isFirstTable) {
      // Use second hash block.
      m_secondHash = true;
      return _CheckAndSet(hashTable + poolSize + 1, poolSize, false, idx);
    }

    DoubleSize();
    // printf(
    //     "Hash table is full! Set HashTableExponent to larger value (default is "
    //     "2). NewHashTableExponent=%d NewPoolSize=%d\n",
    //     m_exp, m_poolSize);
    return _CheckAndSet(m_hashTable.get(), m_poolSize, true, idx);
  }
};
