# Find fmt library
# Once done this will define
#  NUMA_FOUND - libfmt found

if(NOT FMT_FOUND)
  find_library(FMT_LIBRARY NAMES numa PATH_SUFFIXES lib lib64)
  if(FMT_LIBRARY)
    mark_as_advanced(FMT_FOUND)
  endif()
endif()