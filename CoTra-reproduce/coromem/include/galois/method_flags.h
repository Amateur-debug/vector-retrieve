#ifndef METHODFLAGS_H
#define METHODFLAGS_H

#include "config.h"

namespace galois {

/**
 * What should the runtime do when executing a method.
 *
 * Various methods take an optional parameter indicating what actions
 * the runtime should do on the user's behalf: (1) checking for conflicts,
 * and/or (2) saving undo information. By default, both are performed (ALL).
 */
enum class MethodFlag : char {
  UNPROTECTED   = 0,
  WRITE         = 1,
  READ          = 2,
  INTERNAL_MASK = 3,
  PREVIOUS      = 4,
};

//! Bitwise & for method flags
inline MethodFlag operator&(MethodFlag x, MethodFlag y) {
  return (MethodFlag)(((int)x) & ((int)y));
}

//! Bitwise | for method flags
inline MethodFlag operator|(MethodFlag x, MethodFlag y) {
  return (MethodFlag)(((int)x) | ((int)y));
}
} // namespace galois

#endif // METHODFLAGS_H
