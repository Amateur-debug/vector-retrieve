cmake_minimum_required(VERSION 3.12)
project(vector-retrieve)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找 pybind11
find_package(pybind11 REQUIRED)

# 包含目录
include_directories(source/cpp)

include(CheckCXXCompilerFlag)

# 优化与指令集
check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512F)
check_cxx_compiler_flag("-mfma" COMPILER_SUPPORTS_FMA)

set(OPT_FLAGS -O3)
if(COMPILER_SUPPORTS_MARCH_NATIVE)
  list(APPEND OPT_FLAGS -march=native)
endif()
if(COMPILER_SUPPORTS_AVX2)
  list(APPEND OPT_FLAGS -mavx2)
  list(APPEND OPT_FLAGS -DUSE_AVX2)     # 在源码中可检测 USE_AVX2 / USE_AVX
endif()
if(COMPILER_SUPPORTS_FMA)
  list(APPEND OPT_FLAGS -mfma)
endif()
if(COMPILER_SUPPORTS_AVX512F)
  list(APPEND OPT_FLAGS -mavx512f)
  list(APPEND OPT_FLAGS -DUSE_AVX512)
endif()

# 添加 pybind11 模块
pybind11_add_module(vector_retrieve source/cpp/pybind.cpp)

# 仅对目标应用优化/指令集选项
target_compile_options(vector_retrieve PRIVATE ${OPT_FLAGS})

