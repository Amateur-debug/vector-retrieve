#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <type_traits>

// SIMD 指令集头文件 —— 始终包含，实际使用哪个指令集由运行时检测决定
#if defined(__x86_64__) || defined(_M_X64)
#ifdef __SSE__
#include <xmmintrin.h>  // SSE 指令集支持
#include <emmintrin.h>  // SSE2 指令集支持
#endif
#ifdef __AVX__
#include <immintrin.h>
#endif
#endif

// 编译宏兼容处理：允许旧版编译选项 USE_AVX512/USE_AVX/USE_SSE 继续生效
#if defined(USE_AVX512) && !defined(__AVX512F__)
// 编译器未启用 AVX512 支持，USE_AVX512 宏定义将被忽略
#endif

// 内存对齐宏定义（跨平台兼容，确保 SIMD 数据的正确对齐）
#if defined(_MSC_VER)
#define PORTABLE_ALIGN64 __declspec(align(64))
#define PORTABLE_ALIGN32 __declspec(align(32))
#else
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#endif

//============================================================
// 距离函数指针类型定义（参考 hnswlib 的设计）
// 统一签名：T(*)(const void* pVect1, const void* pVect2, const void* qty_ptr)
// 其中 qty_ptr 指向 const size_t 类型的维度值（dim）
//============================================================

template<typename T>
using DISTFUNC = T(*)(const void*, const void*, const void*);

//============================================================
// 存储类型 → 距离值类型映射
// 多数类型距离值与存储类型一致；uint8 存储时距离用 int32 累加避免溢出。
//============================================================

template<typename T>
using DistTOf = std::conditional_t<std::is_same_v<T, uint8_t>, int32_t, T>;

//============================================================
// 距离空间抽象基类（参考 hnswlib 的设计）
// 封装距离函数指针、函数参数和数据维度，
// 供 HNSWGraph 通过工厂方法获取对应类型和指令集的最优实现。
//============================================================

template<typename T>
class SpaceInterface {
public:
    virtual ~SpaceInterface() = default;
    // 获取距离计算函数指针（运行时根据 CPU SIMD 能力动态选择）
    virtual DISTFUNC<T> get_dist_func() = 0;
    // 获取距离函数的辅助参数（指向维度值 dim 的指针）
    virtual void* get_dist_func_param() = 0;
    // 获取向量数据的字节大小（dim * sizeof(T)）
    virtual size_t get_data_size() = 0;
};

//============================================================
// 运行时 SIMD 指令集能力检测（参考 hnswlib 的实现）
// 通过 CPUID 指令查询当前 CPU 支持的 SIMD 指令集，
// 返回布尔值表示是否支持 SSE/AVX/AVX-512。
//============================================================

#if defined(__x86_64__) || defined(_M_X64)

#ifdef _MSC_VER
#include <intrin.h>
#include <stdexcept>
static void cpuid(int info[4], int InfoType) {
    __cpuidex(info, InfoType, 0);
}
#else
#include <cpuid.h>
static void cpuid(int info[4], int InfoType) {
    __cpuid_count(InfoType, 0, info[0], info[1], info[2], info[3]);
}
#endif

static bool AVX512Capable() {
    int cpuInfo[4];
    cpuid(cpuInfo, 0);
    int nIds = cpuInfo[0];
    if (nIds >= 7) {
        cpuid(cpuInfo, 7);
    // 检测 AVX-512F 支持：位于 CPUID 功能 7 的 EBX 寄存器第 16 位
        return (cpuInfo[1] & (1 << 16)) != 0;
    }
    return false;
}

static bool AVXCapable() {
    int cpuInfo[4];
    cpuid(cpuInfo, 0);
    int nIds = cpuInfo[0];
    if (nIds >= 1) {
        cpuid(cpuInfo, 1);
    // 检测 AVX 支持：位于 CPUID 功能 1 的 ECX 寄存器第 28 位
        return (cpuInfo[2] & (1 << 28)) != 0;
    }
    return false;
}

static bool SSECapable() {
    int cpuInfo[4];
    cpuid(cpuInfo, 0);
    int nIds = cpuInfo[0];
    if (nIds >= 1) {
        cpuid(cpuInfo, 1);
    // 检测 SSE 支持：位于 CPUID 功能 1 的 EDX 寄存器第 25 位
        return (cpuInfo[3] & (1 << 25)) != 0;
    }
    return false;
}

static bool AVX2Capable() {
    int cpuInfo[4];
    cpuid(cpuInfo, 0);
    int nIds = cpuInfo[0];
    if (nIds >= 7) {
        cpuid(cpuInfo, 7);
    // 检测 AVX2 支持：位于 CPUID 功能 7 的 EBX 寄存器第 5 位
        return (cpuInfo[1] & (1 << 5)) != 0;
    }
    return false;
}

#else
// 非 x86 平台：所有 SIMD 指令集能力检测均返回 false，回退使用标量实现
static bool AVX512Capable() { return false; }
static bool AVXCapable() { return false; }
static bool SSECapable() { return false; }
static bool AVX2Capable() { return false; }
#endif

//============================================================
// L2 距离 kernel 模板族
// 每个 SIMD tier 一个主模板（未定义，靠 (T, DistT) 特化提供实现）。
// 内部循环结构参考 hnswlib；函数体在各特化之间可能差异较大（受指令集限制），
// 但对外统一通过 DISTFUNC<DistT> 函数指针暴露。
//============================================================

template<typename T, typename DistT> DistT L2Sqr_Scalar (const void*, const void*, const void*);
template<typename T, typename DistT> DistT L2Sqr_SSE4   (const void*, const void*, const void*);
template<typename T, typename DistT> DistT L2Sqr_SSE16  (const void*, const void*, const void*);
template<typename T, typename DistT> DistT L2Sqr_AVX8   (const void*, const void*, const void*);
template<typename T, typename DistT> DistT L2Sqr_AVX16  (const void*, const void*, const void*);
template<typename T, typename DistT> DistT L2Sqr_AVX2_8 (const void*, const void*, const void*);
template<typename T, typename DistT> DistT L2Sqr_AVX2_16(const void*, const void*, const void*);
template<typename T, typename DistT> DistT L2Sqr_AVX512 (const void*, const void*, const void*);

//============================================================
// 标量回退实现（Scalar Fallback）
// 不依赖任何 SIMD 指令集，作为 CPU 不支持 SIMD 时的回退方案。
// 符合 DISTFUNC 统一签名，可直接赋值给函数指针。
//============================================================

template<>
inline float L2Sqr_Scalar<float, float>(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    const float* pVect1 = (const float*)pVect1v;
    const float* pVect2 = (const float*)pVect2v;
    size_t qty = *((const size_t*)qty_ptr);
    float res = 0;
    for (size_t i = 0; i < qty; i++) {
        float t = pVect1[i] - pVect2[i];
        res += t * t;
    }
    return res;
}

template<>
inline double L2Sqr_Scalar<double, double>(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    const double* pVect1 = (const double*)pVect1v;
    const double* pVect2 = (const double*)pVect2v;
    size_t qty = *((const size_t*)qty_ptr);
    double res = 0;
    for (size_t i = 0; i < qty; i++) {
        double t = pVect1[i] - pVect2[i];
        res += t * t;
    }
    return res;
}

template<>
inline int32_t L2Sqr_Scalar<int32_t, int32_t>(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    const int32_t* pVect1 = (const int32_t*)pVect1v;
    const int32_t* pVect2 = (const int32_t*)pVect2v;
    size_t qty = *((const size_t*)qty_ptr);
    int32_t res = 0;
    for (size_t i = 0; i < qty; i++) {
        int32_t t = pVect1[i] - pVect2[i];
        res += t * t;
    }
    return res;
}

template<>
inline uint32_t L2Sqr_Scalar<uint32_t, uint32_t>(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    const uint32_t* pVect1 = (const uint32_t*)pVect1v;
    const uint32_t* pVect2 = (const uint32_t*)pVect2v;
    size_t qty = *((const size_t*)qty_ptr);
    uint32_t res = 0;
    for (size_t i = 0; i < qty; i++) {
        int32_t t = (int32_t)pVect1[i] - (int32_t)pVect2[i];
        res += (uint32_t)(t * t);
    }
    return res;
}

// uint8 存储 + int32 累加（参考 CoTra L2SqrI）
// 差值最大 ±255，平方 ≤ 65025；dim=10^6 时累加 ≤ 6.5e10，仍然溢出 int32。
// 实际 HNSW dim ≤ 960，累加 ≤ 6e7，int32 充裕。
template<>
inline int32_t L2Sqr_Scalar<uint8_t, int32_t>(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    const uint8_t* pVect1 = (const uint8_t*)pVect1v;
    const uint8_t* pVect2 = (const uint8_t*)pVect2v;
    size_t qty = *((const size_t*)qty_ptr);
    int32_t res = 0;
    for (size_t i = 0; i < qty; i++) {
        int32_t t = (int32_t)pVect1[i] - (int32_t)pVect2[i];
        res += t * t;
    }
    return res;
}

// 内积距离 —— 仅 float 标量回退（IP 暂不泛化，保持现状）
static float IPSqrScalar(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    const float* pVect1 = (const float*)pVect1v;
    const float* pVect2 = (const float*)pVect2v;
    size_t qty = *((const size_t*)qty_ptr);
    float res = 0;
    for (size_t i = 0; i < qty; i++) {
        res += pVect1[i] * pVect2[i];
    }
    return (1.0f - res);
}

//============================================================
// SSE 指令集优化实现
// 使用 128 位 SSE 寄存器，每次处理 4 个 float。
// 提供 4 元素和 16 元素两种循环展开版本。
//============================================================
#if defined(__SSE__) || defined(USE_SSE)

template<>
inline float L2Sqr_SSE4<float, float>(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    float PORTABLE_ALIGN32 TmpRes[4];
    const float* pVect1 = (const float*)pVect1v;
    const float* pVect2 = (const float*)pVect2v;
    size_t qty = *((const size_t*)qty_ptr);
    size_t qty4 = qty >> 2;

    const float* pEnd1 = pVect1 + (qty4 << 2);

    __m128 diff, v1, v2;
    __m128 sum = _mm_set1_ps(0);

    while (pVect1 < pEnd1) {
        _mm_prefetch((char*)(pVect1 + 16), _MM_HINT_T0);
        _mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
    }

    _mm_store_ps(TmpRes, sum);
    float res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];

    size_t remaining = qty & 3;
    for (size_t i = 0; i < remaining; i++) {
        float t = pVect1[i] - pVect2[i];
        res += t * t;
    }

    return res;
}

template<>
inline float L2Sqr_SSE16<float, float>(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    float PORTABLE_ALIGN32 TmpRes[4];
    const float* pVect1 = (const float*)pVect1v;
    const float* pVect2 = (const float*)pVect2v;
    size_t qty = *((const size_t*)qty_ptr);
    size_t qty16 = qty >> 4;

    const float* pEnd1 = pVect1 + (qty16 << 4);

    __m128 diff, v1, v2;
    __m128 sum = _mm_set1_ps(0);

    while (pVect1 < pEnd1) {
        _mm_prefetch((char*)(pVect1 + 16), _MM_HINT_T0);
        _mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);

        v1 = _mm_loadu_ps(pVect1); pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2); pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

        v1 = _mm_loadu_ps(pVect1); pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2); pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

        v1 = _mm_loadu_ps(pVect1); pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2); pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

        v1 = _mm_loadu_ps(pVect1); pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2); pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
    }

    _mm_store_ps(TmpRes, sum);
    float res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];

    size_t remaining = qty & 15;
    for (size_t i = 0; i < remaining; i++) {
        float t = pVect1[i] - pVect2[i];
        res += t * t;
    }

    return res;
}

// SSE 整数 L2 距离（需要 SSE4.1 的 _mm_mullo_epi32）
// int32 和 uint32 共用同一 SIMD 实现：2's complement 下 sub + mullo 低32位结果一致

template<>
inline int32_t L2Sqr_SSE4<int32_t, int32_t>(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    int32_t PORTABLE_ALIGN32 TmpRes[4];
    const int32_t* pVect1 = (const int32_t*)pVect1v;
    const int32_t* pVect2 = (const int32_t*)pVect2v;
    size_t qty = *((const size_t*)qty_ptr);
    size_t qty4 = qty >> 2;

    const int32_t* pEnd1 = pVect1 + (qty4 << 2);

    __m128i diff, v1, v2;
    __m128i sum = _mm_setzero_si128();

    while (pVect1 < pEnd1) {
        _mm_prefetch((char*)(pVect1 + 16), _MM_HINT_T0);
        _mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);

        v1 = _mm_loadu_si128((const __m128i*)pVect1); pVect1 += 4;
        v2 = _mm_loadu_si128((const __m128i*)pVect2); pVect2 += 4;
        diff = _mm_sub_epi32(v1, v2);
        sum = _mm_add_epi32(sum, _mm_mullo_epi32(diff, diff));
    }

    _mm_store_si128((__m128i*)TmpRes, sum);
    int32_t res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];

    size_t remaining = qty & 3;
    for (size_t i = 0; i < remaining; i++) {
        int32_t t = pVect1[i] - pVect2[i];
        res += t * t;
    }

    return res;
}

template<>
inline int32_t L2Sqr_SSE16<int32_t, int32_t>(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    int32_t PORTABLE_ALIGN32 TmpRes[4];
    const int32_t* pVect1 = (const int32_t*)pVect1v;
    const int32_t* pVect2 = (const int32_t*)pVect2v;
    size_t qty = *((const size_t*)qty_ptr);
    size_t qty16 = qty >> 4;

    const int32_t* pEnd1 = pVect1 + (qty16 << 4);

    __m128i diff, v1, v2;
    __m128i sum = _mm_setzero_si128();

    while (pVect1 < pEnd1) {
        _mm_prefetch((char*)(pVect1 + 16), _MM_HINT_T0);
        _mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);

        v1 = _mm_loadu_si128((const __m128i*)pVect1); pVect1 += 4;
        v2 = _mm_loadu_si128((const __m128i*)pVect2); pVect2 += 4;
        diff = _mm_sub_epi32(v1, v2);
        sum = _mm_add_epi32(sum, _mm_mullo_epi32(diff, diff));

        v1 = _mm_loadu_si128((const __m128i*)pVect1); pVect1 += 4;
        v2 = _mm_loadu_si128((const __m128i*)pVect2); pVect2 += 4;
        diff = _mm_sub_epi32(v1, v2);
        sum = _mm_add_epi32(sum, _mm_mullo_epi32(diff, diff));

        v1 = _mm_loadu_si128((const __m128i*)pVect1); pVect1 += 4;
        v2 = _mm_loadu_si128((const __m128i*)pVect2); pVect2 += 4;
        diff = _mm_sub_epi32(v1, v2);
        sum = _mm_add_epi32(sum, _mm_mullo_epi32(diff, diff));

        v1 = _mm_loadu_si128((const __m128i*)pVect1); pVect1 += 4;
        v2 = _mm_loadu_si128((const __m128i*)pVect2); pVect2 += 4;
        diff = _mm_sub_epi32(v1, v2);
        sum = _mm_add_epi32(sum, _mm_mullo_epi32(diff, diff));
    }

    _mm_store_si128((__m128i*)TmpRes, sum);
    int32_t res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];

    size_t remaining = qty & 15;
    for (size_t i = 0; i < remaining; i++) {
        int32_t t = pVect1[i] - pVect2[i];
        res += t * t;
    }

    return res;
}

// uint32 版本：内部位模式与 int32 完全一致，仅返回类型不同
template<>
inline uint32_t L2Sqr_SSE4<uint32_t, uint32_t>(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    return (uint32_t)L2Sqr_SSE4<int32_t, int32_t>(pVect1v, pVect2v, qty_ptr);
}

template<>
inline uint32_t L2Sqr_SSE16<uint32_t, uint32_t>(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    return (uint32_t)L2Sqr_SSE16<int32_t, int32_t>(pVect1v, pVect2v, qty_ptr);
}

// uint8 → int32：一次 8 字节（8 维）：cvtepu8_epi16（8×u8→8×i16）+ sub + madd_epi16（成对 i16² → 4×i32）
// SSE4.1 起支持 _mm_cvtepu8_epi16；SSE2 就支持 _mm_sub_epi16 / _mm_madd_epi16。
template<>
inline int32_t L2Sqr_SSE4<uint8_t, int32_t>(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    int32_t PORTABLE_ALIGN32 TmpRes[4];
    const uint8_t* pVect1 = (const uint8_t*)pVect1v;
    const uint8_t* pVect2 = (const uint8_t*)pVect2v;
    size_t qty = *((const size_t*)qty_ptr);
    size_t qty8 = qty >> 3;  // 每次处理 8 维

    const uint8_t* pEnd1 = pVect1 + (qty8 << 3);

    __m128i sum = _mm_setzero_si128();

    while (pVect1 < pEnd1) {
        _mm_prefetch((const char*)(pVect1 + 32), _MM_HINT_T0);
        _mm_prefetch((const char*)(pVect2 + 32), _MM_HINT_T0);

        // 加载 8 字节并零扩展到 8×int16
        __m128i v1_8 = _mm_loadl_epi64((const __m128i*)pVect1);
        __m128i v2_8 = _mm_loadl_epi64((const __m128i*)pVect2);
        __m128i v1 = _mm_cvtepu8_epi16(v1_8);
        __m128i v2 = _mm_cvtepu8_epi16(v2_8);
        __m128i diff = _mm_sub_epi16(v1, v2);
        // madd_epi16: 4 对 (i16, i16) → 4×i32，每项是 a*a+b*b
        sum = _mm_add_epi32(sum, _mm_madd_epi16(diff, diff));

        pVect1 += 8;
        pVect2 += 8;
    }

    _mm_store_si128((__m128i*)TmpRes, sum);
    int32_t res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];

    size_t remaining = qty & 7;
    for (size_t i = 0; i < remaining; i++) {
        int32_t t = (int32_t)pVect1[i] - (int32_t)pVect2[i];
        res += t * t;
    }

    return res;
}

// uint8 → int32：dim%16==0 变体，内层展开 2 倍
template<>
inline int32_t L2Sqr_SSE16<uint8_t, int32_t>(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    int32_t PORTABLE_ALIGN32 TmpRes[4];
    const uint8_t* pVect1 = (const uint8_t*)pVect1v;
    const uint8_t* pVect2 = (const uint8_t*)pVect2v;
    size_t qty = *((const size_t*)qty_ptr);
    size_t qty16 = qty >> 4;  // 每次处理 16 维

    const uint8_t* pEnd1 = pVect1 + (qty16 << 4);

    __m128i sum = _mm_setzero_si128();

    while (pVect1 < pEnd1) {
        _mm_prefetch((const char*)(pVect1 + 32), _MM_HINT_T0);
        _mm_prefetch((const char*)(pVect2 + 32), _MM_HINT_T0);

        __m128i v1_8 = _mm_loadl_epi64((const __m128i*)pVect1);
        __m128i v2_8 = _mm_loadl_epi64((const __m128i*)pVect2);
        __m128i v1 = _mm_cvtepu8_epi16(v1_8);
        __m128i v2 = _mm_cvtepu8_epi16(v2_8);
        __m128i diff = _mm_sub_epi16(v1, v2);
        sum = _mm_add_epi32(sum, _mm_madd_epi16(diff, diff));
        pVect1 += 8;
        pVect2 += 8;

        v1_8 = _mm_loadl_epi64((const __m128i*)pVect1);
        v2_8 = _mm_loadl_epi64((const __m128i*)pVect2);
        v1 = _mm_cvtepu8_epi16(v1_8);
        v2 = _mm_cvtepu8_epi16(v2_8);
        diff = _mm_sub_epi16(v1, v2);
        sum = _mm_add_epi32(sum, _mm_madd_epi16(diff, diff));
        pVect1 += 8;
        pVect2 += 8;
    }

    _mm_store_si128((__m128i*)TmpRes, sum);
    int32_t res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];

    size_t remaining = qty & 15;
    for (size_t i = 0; i < remaining; i++) {
        int32_t t = (int32_t)pVect1[i] - (int32_t)pVect2[i];
        res += t * t;
    }

    return res;
}

#endif // __SSE__

//============================================================
// AVX 指令集优化实现
// 使用 256 位 AVX 寄存器，每次处理 8 个 float 或 4 个 double。
// 提供 L2 距离和内积距离的多种循环展开版本。
//============================================================
#if defined(__AVX__) || defined(USE_AVX) || defined(USE_AVX2)

template<>
inline float L2Sqr_AVX8<float, float>(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    float PORTABLE_ALIGN32 TmpRes[8];
    const float* pVect1 = (const float*)pVect1v;
    const float* pVect2 = (const float*)pVect2v;
    size_t qty = *((const size_t*)qty_ptr);
    size_t qty8 = qty >> 3;

    const float* pEnd1 = pVect1 + (qty8 << 3);

    __m256 diff, v1, v2;
    __m256 sum = _mm256_set1_ps(0);

    while (pVect1 < pEnd1) {
        _mm_prefetch((char*)(pVect1 + 32), _MM_HINT_T0);
        _mm_prefetch((char*)(pVect2 + 32), _MM_HINT_T0);

        v1 = _mm256_loadu_ps(pVect1); pVect1 += 8;
        v2 = _mm256_loadu_ps(pVect2); pVect2 += 8;
        diff = _mm256_sub_ps(v1, v2);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
    }

    _mm256_store_ps(TmpRes, sum);
    float res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] +
                TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];

    size_t remaining = qty & 7;
    for (size_t i = 0; i < remaining; i++) {
        float t = pVect1[i] - pVect2[i];
        res += t * t;
    }

    return res;
}

template<>
inline float L2Sqr_AVX16<float, float>(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    float PORTABLE_ALIGN32 TmpRes[8];
    const float* pVect1 = (const float*)pVect1v;
    const float* pVect2 = (const float*)pVect2v;
    size_t qty = *((const size_t*)qty_ptr);
    size_t qty16 = qty >> 4;

    const float* pEnd1 = pVect1 + (qty16 << 4);

    __m256 diff, v1, v2;
    __m256 sum = _mm256_set1_ps(0);

    while (pVect1 < pEnd1) {
        _mm_prefetch((char*)(pVect1 + 32), _MM_HINT_T0);
        _mm_prefetch((char*)(pVect2 + 32), _MM_HINT_T0);

        v1 = _mm256_loadu_ps(pVect1); pVect1 += 8;
        v2 = _mm256_loadu_ps(pVect2); pVect2 += 8;
        diff = _mm256_sub_ps(v1, v2);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));

        v1 = _mm256_loadu_ps(pVect1); pVect1 += 8;
        v2 = _mm256_loadu_ps(pVect2); pVect2 += 8;
        diff = _mm256_sub_ps(v1, v2);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
    }

    _mm256_store_ps(TmpRes, sum);
    float res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] +
                TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];

    size_t remaining = qty & 15;
    for (size_t i = 0; i < remaining; i++) {
        float t = pVect1[i] - pVect2[i];
        res += t * t;
    }

    return res;
}

template<>
inline double L2Sqr_AVX8<double, double>(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    double PORTABLE_ALIGN32 TmpRes[4];
    const double* pVect1 = (const double*)pVect1v;
    const double* pVect2 = (const double*)pVect2v;
    size_t qty = *((const size_t*)qty_ptr);
    size_t qty4 = qty >> 2;

    const double* pEnd1 = pVect1 + (qty4 << 2);

    __m256d diff, v1, v2;
    __m256d sum = _mm256_set1_pd(0);

    while (pVect1 < pEnd1) {
        _mm_prefetch((char*)(pVect1 + 8), _MM_HINT_T0);
        _mm_prefetch((char*)(pVect2 + 8), _MM_HINT_T0);

        v1 = _mm256_loadu_pd(pVect1); pVect1 += 4;
        v2 = _mm256_loadu_pd(pVect2); pVect2 += 4;
        diff = _mm256_sub_pd(v1, v2);
        sum = _mm256_add_pd(sum, _mm256_mul_pd(diff, diff));
    }

    _mm256_store_pd(TmpRes, sum);
    double res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];

    size_t remaining = qty & 3;
    for (size_t i = 0; i < remaining; i++) {
        double t = pVect1[i] - pVect2[i];
        res += t * t;
    }

    return res;
}

// AVX 内积距离（保留现状，未模板化）
static float IPSqrSIMD8ExtAVX(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    float PORTABLE_ALIGN32 TmpRes[8];
    const float* pVect1 = (const float*)pVect1v;
    const float* pVect2 = (const float*)pVect2v;
    size_t qty = *((const size_t*)qty_ptr);
    size_t qty8 = qty >> 3;

    const float* pEnd1 = pVect1 + (qty8 << 3);

    __m256 v1, v2;
    __m256 sum = _mm256_set1_ps(0);

    while (pVect1 < pEnd1) {
        _mm_prefetch((char*)(pVect1 + 32), _MM_HINT_T0);
        _mm_prefetch((char*)(pVect2 + 32), _MM_HINT_T0);

        v1 = _mm256_loadu_ps(pVect1); pVect1 += 8;
        v2 = _mm256_loadu_ps(pVect2); pVect2 += 8;
        sum = _mm256_add_ps(sum, _mm256_mul_ps(v1, v2));
    }

    _mm256_store_ps(TmpRes, sum);
    float res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] +
                TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];

    size_t remaining = qty & 7;
    for (size_t i = 0; i < remaining; i++) {
        res += pVect1[i] * pVect2[i];
    }

    return (1.0f - res);
}

static float IPSqrSIMD16ExtAVX(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    float PORTABLE_ALIGN32 TmpRes[8];
    const float* pVect1 = (const float*)pVect1v;
    const float* pVect2 = (const float*)pVect2v;
    size_t qty = *((const size_t*)qty_ptr);
    size_t qty16 = qty >> 4;

    const float* pEnd1 = pVect1 + (qty16 << 4);

    __m256 v1, v2;
    __m256 sum = _mm256_set1_ps(0);

    while (pVect1 < pEnd1) {
        _mm_prefetch((char*)(pVect1 + 32), _MM_HINT_T0);
        _mm_prefetch((char*)(pVect2 + 32), _MM_HINT_T0);

        v1 = _mm256_loadu_ps(pVect1); pVect1 += 8;
        v2 = _mm256_loadu_ps(pVect2); pVect2 += 8;
        sum = _mm256_add_ps(sum, _mm256_mul_ps(v1, v2));

        v1 = _mm256_loadu_ps(pVect1); pVect1 += 8;
        v2 = _mm256_loadu_ps(pVect2); pVect2 += 8;
        sum = _mm256_add_ps(sum, _mm256_mul_ps(v1, v2));
    }

    _mm256_store_ps(TmpRes, sum);
    float res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] +
                TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];

    size_t remaining = qty & 15;
    for (size_t i = 0; i < remaining; i++) {
        res += pVect1[i] * pVect2[i];
    }

    return (1.0f - res);
}

// AVX2 整数 L2 距离（需要 AVX2 的 _mm256_mullo_epi32）

template<>
inline int32_t L2Sqr_AVX2_8<int32_t, int32_t>(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    int32_t PORTABLE_ALIGN32 TmpRes[8];
    const int32_t* pVect1 = (const int32_t*)pVect1v;
    const int32_t* pVect2 = (const int32_t*)pVect2v;
    size_t qty = *((const size_t*)qty_ptr);
    size_t qty8 = qty >> 3;

    const int32_t* pEnd1 = pVect1 + (qty8 << 3);

    __m256i diff, v1, v2;
    __m256i sum = _mm256_setzero_si256();

    while (pVect1 < pEnd1) {
        _mm_prefetch((char*)(pVect1 + 32), _MM_HINT_T0);
        _mm_prefetch((char*)(pVect2 + 32), _MM_HINT_T0);

        v1 = _mm256_loadu_si256((const __m256i*)pVect1); pVect1 += 8;
        v2 = _mm256_loadu_si256((const __m256i*)pVect2); pVect2 += 8;
        diff = _mm256_sub_epi32(v1, v2);
        sum = _mm256_add_epi32(sum, _mm256_mullo_epi32(diff, diff));
    }

    _mm256_store_si256((__m256i*)TmpRes, sum);
    int32_t res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] +
                  TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];

    size_t remaining = qty & 7;
    for (size_t i = 0; i < remaining; i++) {
        int32_t t = pVect1[i] - pVect2[i];
        res += t * t;
    }

    return res;
}

template<>
inline int32_t L2Sqr_AVX2_16<int32_t, int32_t>(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    int32_t PORTABLE_ALIGN32 TmpRes[8];
    const int32_t* pVect1 = (const int32_t*)pVect1v;
    const int32_t* pVect2 = (const int32_t*)pVect2v;
    size_t qty = *((const size_t*)qty_ptr);
    size_t qty16 = qty >> 4;

    const int32_t* pEnd1 = pVect1 + (qty16 << 4);

    __m256i diff, v1, v2;
    __m256i sum = _mm256_setzero_si256();

    while (pVect1 < pEnd1) {
        _mm_prefetch((char*)(pVect1 + 32), _MM_HINT_T0);
        _mm_prefetch((char*)(pVect2 + 32), _MM_HINT_T0);

        v1 = _mm256_loadu_si256((const __m256i*)pVect1); pVect1 += 8;
        v2 = _mm256_loadu_si256((const __m256i*)pVect2); pVect2 += 8;
        diff = _mm256_sub_epi32(v1, v2);
        sum = _mm256_add_epi32(sum, _mm256_mullo_epi32(diff, diff));

        v1 = _mm256_loadu_si256((const __m256i*)pVect1); pVect1 += 8;
        v2 = _mm256_loadu_si256((const __m256i*)pVect2); pVect2 += 8;
        diff = _mm256_sub_epi32(v1, v2);
        sum = _mm256_add_epi32(sum, _mm256_mullo_epi32(diff, diff));
    }

    _mm256_store_si256((__m256i*)TmpRes, sum);
    int32_t res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] +
                  TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];

    size_t remaining = qty & 15;
    for (size_t i = 0; i < remaining; i++) {
        int32_t t = pVect1[i] - pVect2[i];
        res += t * t;
    }

    return res;
}

// uint32 版本：委托给 int32 实现（位模式一致）
template<>
inline uint32_t L2Sqr_AVX2_8<uint32_t, uint32_t>(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    return (uint32_t)L2Sqr_AVX2_8<int32_t, int32_t>(pVect1v, pVect2v, qty_ptr);
}

template<>
inline uint32_t L2Sqr_AVX2_16<uint32_t, uint32_t>(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    return (uint32_t)L2Sqr_AVX2_16<int32_t, int32_t>(pVect1v, pVect2v, qty_ptr);
}

// uint8 → int32：一次 16 字节（16 维）：cvtepu8_epi16（16×u8→16×i16）+ sub + madd_epi16（8 对 → 8×i32）
template<>
inline int32_t L2Sqr_AVX2_8<uint8_t, int32_t>(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    int32_t PORTABLE_ALIGN32 TmpRes[8];
    const uint8_t* pVect1 = (const uint8_t*)pVect1v;
    const uint8_t* pVect2 = (const uint8_t*)pVect2v;
    size_t qty = *((const size_t*)qty_ptr);
    size_t qty16 = qty >> 4;  // 每次处理 16 维

    const uint8_t* pEnd1 = pVect1 + (qty16 << 4);

    __m256i sum = _mm256_setzero_si256();

    while (pVect1 < pEnd1) {
        _mm_prefetch((const char*)(pVect1 + 64), _MM_HINT_T0);
        _mm_prefetch((const char*)(pVect2 + 64), _MM_HINT_T0);

        __m128i v1_8 = _mm_loadu_si128((const __m128i*)pVect1);
        __m128i v2_8 = _mm_loadu_si128((const __m128i*)pVect2);
        __m256i v1 = _mm256_cvtepu8_epi16(v1_8);
        __m256i v2 = _mm256_cvtepu8_epi16(v2_8);
        __m256i diff = _mm256_sub_epi16(v1, v2);
        // madd_epi16: 8 对 (i16, i16) → 8×i32
        sum = _mm256_add_epi32(sum, _mm256_madd_epi16(diff, diff));

        pVect1 += 16;
        pVect2 += 16;
    }

    _mm256_store_si256((__m256i*)TmpRes, sum);
    int32_t res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] +
                  TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];

    size_t remaining = qty & 15;
    for (size_t i = 0; i < remaining; i++) {
        int32_t t = (int32_t)pVect1[i] - (int32_t)pVect2[i];
        res += t * t;
    }

    return res;
}

// uint8 → int32：dim%32==0 变体，每次处理 32 维（2 次 256-bit madd_epi16）
template<>
inline int32_t L2Sqr_AVX2_16<uint8_t, int32_t>(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    int32_t PORTABLE_ALIGN32 TmpRes[8];
    const uint8_t* pVect1 = (const uint8_t*)pVect1v;
    const uint8_t* pVect2 = (const uint8_t*)pVect2v;
    size_t qty = *((const size_t*)qty_ptr);
    size_t qty32 = qty >> 5;  // 每次处理 32 维

    const uint8_t* pEnd1 = pVect1 + (qty32 << 5);

    __m256i sum = _mm256_setzero_si256();

    while (pVect1 < pEnd1) {
        _mm_prefetch((const char*)(pVect1 + 64), _MM_HINT_T0);
        _mm_prefetch((const char*)(pVect2 + 64), _MM_HINT_T0);

        __m128i v1_8 = _mm_loadu_si128((const __m128i*)pVect1);
        __m128i v2_8 = _mm_loadu_si128((const __m128i*)pVect2);
        __m256i v1 = _mm256_cvtepu8_epi16(v1_8);
        __m256i v2 = _mm256_cvtepu8_epi16(v2_8);
        __m256i diff = _mm256_sub_epi16(v1, v2);
        sum = _mm256_add_epi32(sum, _mm256_madd_epi16(diff, diff));
        pVect1 += 16;
        pVect2 += 16;

        v1_8 = _mm_loadu_si128((const __m128i*)pVect1);
        v2_8 = _mm_loadu_si128((const __m128i*)pVect2);
        v1 = _mm256_cvtepu8_epi16(v1_8);
        v2 = _mm256_cvtepu8_epi16(v2_8);
        diff = _mm256_sub_epi16(v1, v2);
        sum = _mm256_add_epi32(sum, _mm256_madd_epi16(diff, diff));
        pVect1 += 16;
        pVect2 += 16;
    }

    _mm256_store_si256((__m256i*)TmpRes, sum);
    int32_t res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] +
                  TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];

    size_t remaining = qty & 31;
    for (size_t i = 0; i < remaining; i++) {
        int32_t t = (int32_t)pVect1[i] - (int32_t)pVect2[i];
        res += t * t;
    }

    return res;
}

#endif // __AVX__

//============================================================
// AVX-512 指令集优化实现
// 使用 512 位 AVX-512 寄存器，每次处理 16 个 float 或 8 个 double。
// 提供 L2 距离和内积距离的实现，利用 _mm512_reduce_add 进行高效归约。
//============================================================
#if defined(__AVX512F__) || defined(USE_AVX512)

template<>
inline float L2Sqr_AVX512<float, float>(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    const float* pVect1 = (const float*)pVect1v;
    const float* pVect2 = (const float*)pVect2v;
    size_t qty = *((const size_t*)qty_ptr);
    size_t qty16 = qty >> 4;

    const float* pEnd1 = pVect1 + (qty16 << 4);

    __m512 diff, v1, v2;
    __m512 sum = _mm512_set1_ps(0);

    while (pVect1 < pEnd1) {
        _mm_prefetch((char*)(pVect1 + 64), _MM_HINT_T0);
        _mm_prefetch((char*)(pVect2 + 64), _MM_HINT_T0);

        v1 = _mm512_loadu_ps(pVect1); pVect1 += 16;
        v2 = _mm512_loadu_ps(pVect2); pVect2 += 16;
        diff = _mm512_sub_ps(v1, v2);
        sum = _mm512_add_ps(sum, _mm512_mul_ps(diff, diff));
    }

    float res = _mm512_reduce_add_ps(sum);

    size_t remaining = qty & 15;
    for (size_t i = 0; i < remaining; i++) {
        float t = pVect1[i] - pVect2[i];
        res += t * t;
    }

    return res;
}

template<>
inline double L2Sqr_AVX512<double, double>(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    const double* pVect1 = (const double*)pVect1v;
    const double* pVect2 = (const double*)pVect2v;
    size_t qty = *((const size_t*)qty_ptr);
    size_t qty8 = qty >> 3;

    const double* pEnd1 = pVect1 + (qty8 << 3);

    __m512d diff, v1, v2;
    __m512d sum = _mm512_set1_pd(0);

    while (pVect1 < pEnd1) {
        _mm_prefetch((char*)(pVect1 + 16), _MM_HINT_T0);
        _mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);

        v1 = _mm512_loadu_pd(pVect1); pVect1 += 8;
        v2 = _mm512_loadu_pd(pVect2); pVect2 += 8;
        diff = _mm512_sub_pd(v1, v2);
        sum = _mm512_add_pd(sum, _mm512_mul_pd(diff, diff));
    }

    double res = _mm512_reduce_add_pd(sum);

    size_t remaining = qty & 7;
    for (size_t i = 0; i < remaining; i++) {
        double t = pVect1[i] - pVect2[i];
        res += t * t;
    }

    return res;
}

// AVX-512 内积距离（保留现状，未模板化）
static float IPSqrSIMD16ExtAVX512(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    const float* pVect1 = (const float*)pVect1v;
    const float* pVect2 = (const float*)pVect2v;
    size_t qty = *((const size_t*)qty_ptr);
    size_t qty16 = qty >> 4;

    const float* pEnd1 = pVect1 + (qty16 << 4);

    __m512 v1, v2;
    __m512 sum = _mm512_set1_ps(0);

    while (pVect1 < pEnd1) {
        _mm_prefetch((char*)(pVect1 + 64), _MM_HINT_T0);
        _mm_prefetch((char*)(pVect2 + 64), _MM_HINT_T0);

        v1 = _mm512_loadu_ps(pVect1); pVect1 += 16;
        v2 = _mm512_loadu_ps(pVect2); pVect2 += 16;
        sum = _mm512_add_ps(sum, _mm512_mul_ps(v1, v2));
    }

    float res = _mm512_reduce_add_ps(sum);

    size_t remaining = qty & 15;
    for (size_t i = 0; i < remaining; i++) {
        res += pVect1[i] * pVect2[i];
    }

    return (1.0f - res);
}

// AVX-512 整数 L2 距离

template<>
inline int32_t L2Sqr_AVX512<int32_t, int32_t>(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    const int32_t* pVect1 = (const int32_t*)pVect1v;
    const int32_t* pVect2 = (const int32_t*)pVect2v;
    size_t qty = *((const size_t*)qty_ptr);
    size_t qty16 = qty >> 4;

    const int32_t* pEnd1 = pVect1 + (qty16 << 4);

    __m512i diff, v1, v2;
    __m512i sum = _mm512_setzero_si512();

    while (pVect1 < pEnd1) {
        _mm_prefetch((char*)(pVect1 + 64), _MM_HINT_T0);
        _mm_prefetch((char*)(pVect2 + 64), _MM_HINT_T0);

        v1 = _mm512_loadu_si512((const __m512i*)pVect1); pVect1 += 16;
        v2 = _mm512_loadu_si512((const __m512i*)pVect2); pVect2 += 16;
        diff = _mm512_sub_epi32(v1, v2);
        sum = _mm512_add_epi32(sum, _mm512_mullo_epi32(diff, diff));
    }

    int32_t res = _mm512_reduce_add_epi32(sum);

    size_t remaining = qty & 15;
    for (size_t i = 0; i < remaining; i++) {
        int32_t t = pVect1[i] - pVect2[i];
        res += t * t;
    }

    return res;
}

template<>
inline uint32_t L2Sqr_AVX512<uint32_t, uint32_t>(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    return (uint32_t)L2Sqr_AVX512<int32_t, int32_t>(pVect1v, pVect2v, qty_ptr);
}

// uint8 → int32：一次 32 字节（32 维）：cvtepu8_epi16（32×u8→32×i16 in 512-bit）+ sub + madd_epi16（16 对 → 16×i32）
// 需要 AVX-512BW 提供 _mm512_sub_epi16 / _mm512_madd_epi16。
template<>
inline int32_t L2Sqr_AVX512<uint8_t, int32_t>(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    const uint8_t* pVect1 = (const uint8_t*)pVect1v;
    const uint8_t* pVect2 = (const uint8_t*)pVect2v;
    size_t qty = *((const size_t*)qty_ptr);
    size_t qty32 = qty >> 5;  // 每次处理 32 维

    const uint8_t* pEnd1 = pVect1 + (qty32 << 5);

    __m512i sum = _mm512_setzero_si512();

    while (pVect1 < pEnd1) {
        _mm_prefetch((const char*)(pVect1 + 64), _MM_HINT_T0);
        _mm_prefetch((const char*)(pVect2 + 64), _MM_HINT_T0);

        __m256i v1_8 = _mm256_loadu_si256((const __m256i*)pVect1);
        __m256i v2_8 = _mm256_loadu_si256((const __m256i*)pVect2);
        __m512i v1 = _mm512_cvtepu8_epi16(v1_8);
        __m512i v2 = _mm512_cvtepu8_epi16(v2_8);
        __m512i diff = _mm512_sub_epi16(v1, v2);
        sum = _mm512_add_epi32(sum, _mm512_madd_epi16(diff, diff));

        pVect1 += 32;
        pVect2 += 32;
    }

    int32_t res = _mm512_reduce_add_epi32(sum);

    size_t remaining = qty & 31;
    for (size_t i = 0; i < remaining; i++) {
        int32_t t = (int32_t)pVect1[i] - (int32_t)pVect2[i];
        res += t * t;
    }

    return res;
}

#endif // __AVX512F__

//============================================================
// L2 距离空间 —— 统一模板实现
// 合并原 L2SpaceFloat / L2SpaceDouble / L2SpaceInt32 / L2SpaceUInt32 为单一模板。
// 运行时根据 CPU SIMD 能力和向量维度的对齐情况，动态选择性能最优的距离函数。
// 优先级：AVX-512 > AVX/AVX2 > SSE > Scalar。
// 浮点类型走 AVX 分支；整数类型走 AVX2 分支（依赖 _mm256_mullo_epi32）。
//============================================================

template<typename T, typename DistT = DistTOf<T>>
class L2Space : public SpaceInterface<DistT> {
    DISTFUNC<DistT> dist_func_;
    size_t data_size_;
    size_t dim_;

public:
    L2Space(size_t dim) : dim_(dim), data_size_(dim * sizeof(T)) {
        dist_func_ = &L2Sqr_Scalar<T, DistT>;

#if defined(__AVX512F__) || defined(USE_AVX512)
        if (AVX512Capable()) {
            dist_func_ = &L2Sqr_AVX512<T, DistT>;
        } else
#endif
        {
            if constexpr (std::is_floating_point_v<T>) {
#if defined(__AVX__) || defined(USE_AVX) || defined(USE_AVX2)
                if (AVXCapable()) {
                    if constexpr (std::is_same_v<T, double>) {
                        // double 仅有 AVX8（4 元素）变体
                        dist_func_ = &L2Sqr_AVX8<T, DistT>;
                    } else {
                        // float 有 AVX16 / AVX8 两种变体，按维度对齐选择
                        if (dim % 16 == 0) {
                            dist_func_ = &L2Sqr_AVX16<T, DistT>;
                        } else {
                            dist_func_ = &L2Sqr_AVX8<T, DistT>;
                        }
                    }
                } else
#endif
#if defined(__SSE__) || defined(USE_SSE)
                if constexpr (std::is_same_v<T, float>) {
                    if (SSECapable()) {
                        if (dim % 16 == 0) {
                            dist_func_ = &L2Sqr_SSE16<T, DistT>;
                        } else if (dim % 4 == 0) {
                            dist_func_ = &L2Sqr_SSE4<T, DistT>;
                        }
                    }
                }
#endif
                ;
            } else {
                // 整数类型：AVX2 > SSE4.1 > Scalar
                // 注意 uint8 和 int32/uint32 的 SIMD 展开步长不同（uint8 每 SIMD 通道装更多元素）
#if defined(__AVX2__) || defined(USE_AVX2)
                if (AVX2Capable()) {
                    if constexpr (std::is_same_v<T, uint8_t>) {
                        // uint8: AVX2_16 每次 32 维（dim%32 最佳）；AVX2_8 每次 16 维
                        if (dim % 32 == 0) {
                            dist_func_ = &L2Sqr_AVX2_16<T, DistT>;
                        } else {
                            dist_func_ = &L2Sqr_AVX2_8<T, DistT>;
                        }
                    } else {
                        // int32/uint32: AVX2_16 每次 16 维；AVX2_8 每次 8 维
                        if (dim % 16 == 0) {
                            dist_func_ = &L2Sqr_AVX2_16<T, DistT>;
                        } else {
                            dist_func_ = &L2Sqr_AVX2_8<T, DistT>;
                        }
                    }
                } else
#endif
#if defined(__SSE4_1__) || defined(__SSE__) || defined(USE_SSE)
                if (SSECapable()) {
                    if constexpr (std::is_same_v<T, uint8_t>) {
                        // uint8: SSE16 每次 16 维；SSE4 每次 8 维
                        if (dim % 16 == 0) {
                            dist_func_ = &L2Sqr_SSE16<T, DistT>;
                        } else if (dim % 8 == 0) {
                            dist_func_ = &L2Sqr_SSE4<T, DistT>;
                        }
                    } else {
                        // int32/uint32: SSE16 每次 16 维；SSE4 每次 4 维
                        if (dim % 16 == 0) {
                            dist_func_ = &L2Sqr_SSE16<T, DistT>;
                        } else if (dim % 4 == 0) {
                            dist_func_ = &L2Sqr_SSE4<T, DistT>;
                        }
                    }
                }
#endif
                ;
            }
        }
    }

    DISTFUNC<DistT> get_dist_func() override { return dist_func_; }
    void* get_dist_func_param() override { return &dim_; }
    size_t get_data_size() override { return data_size_; }
};

// 兼容别名：保留现有代码中对 L2SpaceFloat / L2SpaceDouble / L2SpaceInt32 / L2SpaceUInt32 的引用
using L2SpaceFloat  = L2Space<float>;
using L2SpaceDouble = L2Space<double>;
using L2SpaceInt32  = L2Space<int32_t>;
using L2SpaceUInt32 = L2Space<uint32_t>;

//============================================================
// 内积距离空间 —— float 特化（未模板化，保持现状）
// 返回 (1.0 - 内积) 作为距离度量，适用于归一化向量的余弦相似度搜索。
//============================================================

class InnerProductSpaceFloat : public SpaceInterface<float> {
    DISTFUNC<float> dist_func_;
    size_t data_size_;
    size_t dim_;

public:
    InnerProductSpaceFloat(size_t dim) : dim_(dim), data_size_(dim * sizeof(float)) {
        dist_func_ = IPSqrScalar;  // 默认 scalar

#if defined(__AVX512F__) || defined(USE_AVX512)
        if (AVX512Capable()) {
            dist_func_ = IPSqrSIMD16ExtAVX512;
        } else
#endif
#if defined(__AVX__) || defined(USE_AVX) || defined(USE_AVX2)
        if (AVXCapable()) {
            if (dim % 16 == 0) {
                dist_func_ = IPSqrSIMD16ExtAVX;
            } else {
                dist_func_ = IPSqrSIMD8ExtAVX;
            }
        }
#endif
        ;
    }

    DISTFUNC<float> get_dist_func() override { return dist_func_; }
    void* get_dist_func_param() override { return &dim_; }
    size_t get_data_size() override { return data_size_; }
};

//============================================================
// 便捷工厂函数：根据模板类型 T 自动创建对应的 L2 距离空间实例。
// 返回类型为 SpaceInterface<DistTOf<T>>：对 float/double/int32/uint32 等价于 SpaceInterface<T>；
// 对 uint8_t 为 SpaceInterface<int32_t>（存储 uint8，距离累加为 int32）。
//============================================================

template<typename T>
inline std::unique_ptr<SpaceInterface<DistTOf<T>>> create_l2_space(size_t dim) {
    return std::make_unique<L2Space<T>>(dim);
}

