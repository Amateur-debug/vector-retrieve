cmake_minimum_required(VERSION 3.0)
project(rdma-anns)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")

set(CMAKE_BUILD_TYPE "Release")
set(COTRA_MACHINE_NUM "16" CACHE STRING "Compile-time RDMA machine count")
option(COTRA_ENABLE_ASAN "Enable AddressSanitizer for CoTra memory debugging" OFF)
option(COTRA_ENABLE_BREAKDOWN "Enable CoTra ANN performance breakdown instrumentation" OFF)

message(STATUS "Build CoroMem library")

include(GNUInstallDirs)
include(CheckCXXCompilerFlag)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_definitions(-DMACHINE_NUM=${COTRA_MACHINE_NUM})

SET(CMAKE_CXX_FLAGS  "-Ofast -lrt -DHAVE_CXX0X -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0 -shared-libgcc -fcoroutines -fopenmp -msse2 -mavx2 -mfma" )

if(COTRA_ENABLE_ASAN)
  string(REPLACE "-Ofast" "-O1" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  string(APPEND CMAKE_C_FLAGS " -g -fno-omit-frame-pointer -fno-optimize-sibling-calls -fsanitize=address")
  string(APPEND CMAKE_CXX_FLAGS " -g -fno-omit-frame-pointer -fno-optimize-sibling-calls -fsanitize=address")
  string(APPEND CMAKE_EXE_LINKER_FLAGS " -fsanitize=address")
  string(APPEND CMAKE_SHARED_LINKER_FLAGS " -fsanitize=address")
  message(STATUS "COTRA_ENABLE_ASAN=ON")
endif()

if(COTRA_ENABLE_BREAKDOWN)
  add_compile_definitions(PROFILER COMM_PROFILE PROF_COMPUTATION)
  message(STATUS "COTRA_ENABLE_BREAKDOWN=ON")
endif()

set(CMAKE_THREAD_LIBS_INIT "-lpthread")

find_package(Threads REQUIRED)
find_package(Boost 1.58.0 REQUIRED COMPONENTS serialization iostreams)
find_library(MEMCACHED_LIB memcached REQUIRED)

include_directories(./third_party)
if(DEFINED ENV{CONDA_PREFIX} AND EXISTS "$ENV{CONDA_PREFIX}/include")
  include_directories("$ENV{CONDA_PREFIX}/include")
  link_directories("$ENV{CONDA_PREFIX}/lib" "$ENV{CONDA_PREFIX}/lib64")
endif()
set(THIRD_PARTY_DIR ${CMAKE_SOURCE_DIR}/third_party)
add_subdirectory(${THIRD_PARTY_DIR})

add_custom_target(lib)
add_subdirectory(coromem)

message(STATUS "Build RDMA library")

# for rdma deps
link_libraries(pthread /usr/lib/x86_64-linux-gnu/libibverbs.so.1 /usr/lib/x86_64-linux-gnu/librdmacm.so.1 ${MEMCACHED_LIB})
include_directories("${PROJECT_SOURCE_DIR}" "/data/share/project/grasper/rdma-core/build/include")

add_library(rdma_anns)
file(GLOB sources 
  ${CMAKE_SOURCE_DIR}/src/*.cc 
  ${CMAKE_SOURCE_DIR}/src/rdma/*.cc 
  ${CMAKE_SOURCE_DIR}/src/anns/*.cc 
)

target_sources(rdma_anns PRIVATE ${sources})

target_include_directories(rdma_anns PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)


set(COTRA_LIB_HINTS
    $ENV{CONDA_PREFIX}/lib
    $ENV{CONDA_PREFIX}/lib64
    /usr/lib/x86_64-linux-gnu
    /usr/local/lib
    /opt/intel/oneapi/compiler/2023.1.0/linux/compiler/lib/intel64_lin
    /opt/intel/oneapi/mkl/2023.1.0/lib/intel64
    /opt/intel/2024/oneapi/2024.0/lib
    /opt/intel/2024/oneapi/mkl/2024.0/lib)
set(COTRA_MKL_INCLUDE_HINTS
    $ENV{CONDA_PREFIX}/include
    /usr/include
    /usr/local/include
    /opt/intel/oneapi/mkl/2023.1.0/include
    /opt/intel/2024/oneapi/2024.0/include
    /opt/intel/2024/oneapi/mkl/2024.0/include)

find_library(IOMP5_LIB iomp5 HINTS ${COTRA_LIB_HINTS})
if(NOT IOMP5_LIB)
    message(FATAL_ERROR "Could not find libiomp5; load an Intel OpenMP runtime or point CONDA_PREFIX to one.")
endif()
get_filename_component(OMP_PATH ${IOMP5_LIB} DIRECTORY)
link_directories(${OMP_PATH})

find_library(MKL_CORE_LIB mkl_core HINTS ${COTRA_LIB_HINTS})
if(NOT MKL_CORE_LIB)
    message(FATAL_ERROR "Could not find Intel MKL core library.")
endif()
get_filename_component(MKL_PATH ${MKL_CORE_LIB} DIRECTORY)

find_path(MKL_INCLUDE_PATH mkl.h HINTS ${COTRA_MKL_INCLUDE_HINTS})
if(NOT MKL_INCLUDE_PATH)
    message(FATAL_ERROR "Could not find Intel MKL headers.")
endif()
message(STATUS "index MKL_PATH: ${MKL_PATH}")
link_directories(${MKL_PATH})
include_directories(${MKL_INCLUDE_PATH})

find_package(Boost COMPONENTS program_options system filesystem)

link_libraries(mkl_intel_ilp64 mkl_intel_thread mkl_core iomp5 pthread m dl ${Boost_LIBRARIES})

add_library(rdma_index)
file(GLOB index_sources 
  ${CMAKE_SOURCE_DIR}/src/index/*.cc
)
target_sources(rdma_index PRIVATE ${index_sources})
target_include_directories(rdma_index PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

add_subdirectory(tests)
