# =============================================================================
# CMakeLists.txt - Vector Retrieval Library Build Configuration
# =============================================================================
# 
# Description:
#   CMake build configuration for the vector-retrieve Python extension module.
#   This library provides high-performance HNSW (Hierarchical Navigable Small World)
#   graph-based approximate nearest neighbor search with SIMD optimizations.
#
# Features:
#   - Automatic detection of CPU SIMD capabilities (SSE, AVX2, AVX-512)
#   - Python bindings via pybind11
#   - Optimized distance computations with SIMD intrinsics
#
# Usage:
#   mkdir build && cd build
#   cmake ..
#   make -j$(nproc)
#
# Requirements:
#   - CMake >= 3.12
#   - C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
#   - pybind11
#
# Author: CONNECT Research Group
# =============================================================================

cmake_minimum_required(VERSION 3.12)
project(vector-retrieve LANGUAGES CXX)

# =============================================================================
# Compiler Settings
# =============================================================================

# Require C++17 for std::optional, std::string_view, etc.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# =============================================================================
# Dependencies
# =============================================================================

# Find pybind11 for Python bindings
find_package(pybind11 REQUIRED)

# =============================================================================
# Include Directories
# =============================================================================

include_directories(source/cpp)

# =============================================================================
# SIMD and Optimization Flags Detection
# =============================================================================

include(CheckCXXCompilerFlag)

# Check for architecture-specific optimizations
check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)

# Check for SIMD instruction set support
check_cxx_compiler_flag("-msse4.2"    COMPILER_SUPPORTS_SSE42)
check_cxx_compiler_flag("-mavx"      COMPILER_SUPPORTS_AVX)
check_cxx_compiler_flag("-mavx2"     COMPILER_SUPPORTS_AVX2)
check_cxx_compiler_flag("-mfma"      COMPILER_SUPPORTS_FMA)
check_cxx_compiler_flag("-mavx512f"  COMPILER_SUPPORTS_AVX512F)

# -----------------------------------------------------------------------------
# Build optimization flags
# -----------------------------------------------------------------------------

set(OPT_FLAGS 
    -O3                           # Maximum optimization
    -funroll-loops                # Loop unrolling for better ILP
    -ffast-math                   # Fast math (safe for distance computations)
)

# Enable architecture-specific optimizations if supported
if(COMPILER_SUPPORTS_MARCH_NATIVE)
    list(APPEND OPT_FLAGS -march=native)
    message(STATUS "Enabled: -march=native")
endif()

# -----------------------------------------------------------------------------
# SIMD instruction set flags (priority: AVX-512 > AVX2 > AVX > SSE)
# -----------------------------------------------------------------------------

if(COMPILER_SUPPORTS_AVX512F)
    # AVX-512: 512-bit vectors, 16 floats per operation
    list(APPEND OPT_FLAGS -mavx512f -mavx512dq -mavx512vl)
    list(APPEND OPT_FLAGS -DUSE_AVX512)
    message(STATUS "Enabled: AVX-512 SIMD instructions")
    
elseif(COMPILER_SUPPORTS_AVX2 AND COMPILER_SUPPORTS_FMA)
    # AVX2 + FMA: 256-bit vectors, 8 floats per operation with fused multiply-add
    list(APPEND OPT_FLAGS -mavx2 -mfma)
    list(APPEND OPT_FLAGS -DUSE_AVX2)
    message(STATUS "Enabled: AVX2 + FMA SIMD instructions")
    
elseif(COMPILER_SUPPORTS_AVX)
    # AVX: 256-bit vectors, 8 floats per operation
    list(APPEND OPT_FLAGS -mavx)
    list(APPEND OPT_FLAGS -DUSE_AVX)
    message(STATUS "Enabled: AVX SIMD instructions")
    
elseif(COMPILER_SUPPORTS_SSE42)
    # SSE 4.2: 128-bit vectors, 4 floats per operation
    list(APPEND OPT_FLAGS -msse4.2)
    list(APPEND OPT_FLAGS -DUSE_SSE)
    message(STATUS "Enabled: SSE 4.2 SIMD instructions")
    
else()
    message(WARNING "No SIMD instruction set detected, using scalar fallback")
endif()

# =============================================================================
# Python Extension Module
# =============================================================================

# Create the Python extension module using pybind11
pybind11_add_module(vector_retrieve 
    source/cpp/pybind.cpp
)

# Apply optimization flags to the target
target_compile_options(vector_retrieve PRIVATE ${OPT_FLAGS})

# =============================================================================
# Link Libraries (if needed)
# =============================================================================

# Uncomment the following lines if you need to link against additional libraries
# find_package(OpenMP REQUIRED)
# target_link_libraries(vector_retrieve PRIVATE OpenMP::OpenMP_CXX)

# =============================================================================
# Build Information
# =============================================================================

message(STATUS "")
message(STATUS "========================================")
message(STATUS "Vector Retrieval Library Configuration")
message(STATUS "========================================")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Optimization Flags: ${OPT_FLAGS}")
message(STATUS "========================================")
message(STATUS "")

# =============================================================================
# End of CMakeLists.txt
# =============================================================================
