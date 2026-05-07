#ifndef COROGRAPH_CHUNK_H
#define COROGRAPH_CHUNK_H

#include "../galois/config.h"
#include "../galois/fixsize_ring.h"
#include "../runtime/mem.h"
#include "../substrate/padded_lock.h"
#include "worklist.h"

extern unsigned activeThreads;

namespace worklists {

template <typename T, int ChunkSize>
class Chunk : public FixedSizeRing2<T, ChunkSize> {
  Chunk *next;
  bool is_count;  // whether count for recv.

 public:
  Chunk() : next(0), is_count(true) {}
  Chunk(bool _is_count) : next(0), is_count(_is_count) {}
  bool isCount() const { return is_count; }
  Chunk *&getNext() { return next; }
  Chunk *const &getNext() const { return next; }
};

template <typename T, int MessageSize>
struct Block : public ConExtLinkedQueue<Block<T, MessageSize>, true>::ListNode {
 public:
  using Message = Chunk<T, MessageSize>;
  runtime::FixedSizeAllocator<Message> alloc;
  struct MsgBuf {
    Message *cur;
    Message *next;
    MsgBuf() : cur(0), next(0) {}
  };
  PerThreadStorage<MsgBuf> data;
  MessageQueue<Message, true> buffer;
  uint32_t bid;

  Message *mkMessage() {
    Message *ptr = alloc.allocate(1);
    alloc.construct(ptr);
    return ptr;
  }

  void delMessage(Message *ptr) {
    alloc.destroy(ptr);
    alloc.deallocate(ptr, 1);
  }
  bool pushMessage(Message *m) { return buffer.push(m); }
  Message *popMessage() { return buffer.pop(); }

  template <typename... Args>
  bool emplacei(MsgBuf &msg, Args &&...args) {
    if (msg.next && !msg.next->emplace_back(std::forward<Args>(args)...))
      return false;
    if (msg.next) {  // full
      bool ret = pushMessage(msg.next);
      msg.next = 0;
      return ret;
    }
    msg.next = mkMessage();
    msg.next->emplace_back(std::forward<Args>(args)...);
    return false;
  }

  typedef T msg_type;

  explicit Block(uint32_t _bid) : bid(_bid) {}
  Block(const Block &) = delete;
  Block &operator=(const Block &) = delete;

  bool push(const T &val) {
    MsgBuf &msg = *data.getLocal();
    return emplacei(msg, val);
  }

  bool pushNext() {
    bool ret = false;
    MsgBuf &msg = *data.getLocal();
    if (msg.next) {  //&& !msg.next->empty()
      ret = pushMessage(msg.next);
      msg.next = 0;
      // msg.next = mkMessage();
    }
    return ret;
  }

  Message *pop() {
    MsgBuf &msg = *data.getLocal();
    // if (msg.cur && !msg.cur->empty()){
    //     return msg.cur;
    // }
    if (msg.cur) delMessage(msg.cur);
    msg.cur = popMessage();
    if (!msg.cur) {
      msg.cur = msg.next;
      msg.next = 0;
    }
    return msg.cur;
    // if (msg.cur && !msg.cur->empty())
    //     return msg.cur;
    // return nullptr;
  }
};

template <typename T, int FrontierSize>
struct PQueue {
 public:
  using Frontier = Chunk<T, FrontierSize>;
  runtime::FixedSizeAllocator<Frontier> alloc;

  struct FrtBuf {
    Frontier *cur;
    Frontier *next;
    FrtBuf() : cur(0), next(0) {}
  };

  PerThreadStorage<FrtBuf> data;
  PerSocketStorage<ConExtLinkedQueue<Frontier, true>> Q;

  Frontier *mkFrontier() {
    Frontier *ptr = alloc.allocate(1);
    alloc.construct(ptr);
    return ptr;
  }

  void delFrontier(Frontier *ptr) {
    alloc.destroy(ptr);
    alloc.deallocate(ptr, 1);
  }

  void pushFrontier(Frontier *f) {
    auto &I = *Q.getLocal();
    I.push(f);
  }

  Frontier *popFrontierByID(unsigned int i) {
    auto &I = *Q.getRemote(i);
    return I.pop();
  }

  Frontier *popFrontier() {
    int id = ThreadPool::getTID();  // thread id
    Frontier *r = popFrontierByID(id);
    if (r) return r;

    for (int i = id + 1; i < (int)Q.size(); ++i) {
      r = popFrontierByID(i);
      if (r) return r;
    }
    for (int i = 0; i < id; ++i) {
      r = popFrontierByID(i);
      if (r) return r;
    }
    return 0;
  }

  template <typename... Args>
  void emplacei(FrtBuf &frt, Args &&...args) {
    if (frt.next && !frt.next->emplace_back(std::forward<Args>(args)...))
      return;
    if (frt.next) {
      pushFrontier(frt.next);
      frt.next = 0;
      return;
    }
    frt.next = mkFrontier();
    frt.next->emplace_back(std::forward<Args>(args)...);
  }

  typedef T frt_type;

  PQueue() = default;
  PQueue(const PQueue &) = delete;
  PQueue &operator=(const PQueue &) = delete;

  void push(const T &val) {
    FrtBuf &frt = *data.getLocal();
    emplacei(frt, val);
  }

  template <typename Iter>
  void push(Iter b, Iter e) {
    FrtBuf &frt = *data.getLocal();
    while (b != e) emplacei(frt, *b++);
  }

  Frontier *pop() {
    FrtBuf &frt = *data.getLocal();
    // if (frt.cur && !frt.cur->empty()){
    //     return frt.cur;
    // }
    if (frt.cur) delFrontier(frt.cur);
    frt.cur = popFrontier();
    if (!frt.cur) {        // if queue is empty
      frt.cur = frt.next;  // use next as current
      frt.next = 0;
    }
    return frt.cur;
    // if (frt.cur && !frt.cur->empty())
    //     return frt.cur;
    // return nullptr;
  }

  // galois::optional<value_type> pop() {
  //     p& frt = *data.getLocal();
  //     galois::optional<value_type> retval;
  //     if (frt.cur && (retval = frt.cur->extract_front()))
  //         return retval;
  //     if (frt.cur)
  //         delFrontier(frt.cur);
  //     frt.cur = popFrontier();
  //     if (!frt.cur) {
  //         frt.cur  = frt.next;
  //         frt.next = 0;
  //     }
  //     if (frt.cur)
  //         return frt.cur->extract_front();
  //     return galois::optional<value_type>();
  // }
};

template <int ChunkSize = 64, typename T = int>
using PQ = PQueue<T, ChunkSize>;
template <int ChunkSize = 64, typename T = int>
using BL = Block<T, ChunkSize>;
template <int ChunkSize = 64, typename T = int>
using CK = Chunk<T, ChunkSize>;
}  // end namespace worklists

#endif
