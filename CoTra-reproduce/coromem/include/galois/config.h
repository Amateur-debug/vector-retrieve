#ifndef COROMEM_CONFIG_H
#define COROMEM_CONFIG_H

#define COROMEM_USE_LONGJMP_ABORT
#define COROMEM_ALLOW_WARNINGS                                                  \
  _Pragma("GCC diagnostic push") _Pragma("GCC diagnostic warning \"-Wall\"")   \
      _Pragma("GCC diagnostic warning \"-Wextra\"")
#define COROMEM_END_ALLOW_WARNINGS _Pragma("GCC diagnostic pop")

#define COROMEM_IGNORE_WARNINGS                                                 \
  _Pragma("GCC diagnostic push") _Pragma("GCC diagnostic ignored \"-Wall\"")   \
      _Pragma("GCC diagnostic ignored \"-Wextra\"")
#define COROMEM_END_IGNORE_WARNINGS _Pragma("GCC diagnostic pop")

#define COROMEM_IGNORE_UNUSED_PARAMETERS                                        \
  _Pragma("GCC diagnostic push")                                               \
      _Pragma("GCC diagnostic ignored \"-Wunused-parameter\"")
#define COROMEM_END_IGNORE_UNUSED_PARAMETERS _Pragma("GCC diagnostic pop")

#define COROMEM_IGNORE_MAYBE_UNINITIALIZED
#define COROMEM_END_IGNORE_MAYBE_UNINITIALIZED

#define COROMEM_IGNORE_UNUSED_BUT_SET                                           \
  _Pragma("GCC diagnostic push")                                               \
      _Pragma("GCC diagnostic ignored \"-Wunused-but-set-variable\"")
#define COROMEM_END_IGNORE_UNUSED_BUT_SET _Pragma("GCC diagnostic pop")

#define COROMEM_GCC7_IGNORE_UNUSED_BUT_SET
#define COROMEM_END_GCC7_IGNORE_UNUSED_BUT_SET

#define COROMEM_USED_ONLY_IN_DEBUG(NAME) NAME

#define COROMEM_UNUSED(NAME) NAME[[maybe_unused]]

#define COROMEM_IGNORE_EXTERNAL_UNUSED_PARAMETERS                               \
  _Pragma("GCC diagnostic push")                                               \
      _Pragma("GCC diagnostic ignored \"-Wunused-parameter\"")
#define COROMEM_END_IGNORE_EXTERNAL_UNUSED_PARAMETERS                           \
  _Pragma("GCC diagnostic pop")

#endif
