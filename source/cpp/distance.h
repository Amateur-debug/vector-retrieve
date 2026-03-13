#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>

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

#else
// 非 x86 平台：所有 SIMD 指令集能力检测均返回 false，回退使用标量实现
static bool AVX512Capable() { return false; }
static bool AVXCapable() { return false; }
static bool SSECapable() { return false; }
#endif

//============================================================
// 标量回退实现（Scalar Fallback）
// 不依赖任何 SIMD 指令集，作为 CPU 不支持 SIMD 时的回退方案。
// 符合 DISTFUNC 统一签名，可直接赋值给函数指针。
//============================================================

static float L2SqrScalar(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
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

static double L2SqrScalarDouble(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
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

static float L2SqrSIMD4ExtSSE(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
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

static float L2SqrSIMD16ExtSSE(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
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

#endif // __SSE__

//============================================================
// AVX 指令集优化实现
// 使用 256 位 AVX 寄存器，每次处理 8 个 float 或 4 个 double。
// 提供 L2 距离和内积距离的多种循环展开版本。
//============================================================
#if defined(__AVX__) || defined(USE_AVX) || defined(USE_AVX2)

static float L2SqrSIMD8ExtAVX(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
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

static float L2SqrSIMD16ExtAVX(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
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

static double L2SqrSIMD4ExtAVX(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
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

#endif // __AVX__

//============================================================
// AVX-512 指令集优化实现
// 使用 512 位 AVX-512 寄存器，每次处理 16 个 float 或 8 个 double。
// 提供 L2 距离和内积距离的实现，利用 _mm512_reduce_add 进行高效归约。
//============================================================
#if defined(__AVX512F__) || defined(USE_AVX512)

static float L2SqrSIMD16ExtAVX512(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
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

static double L2SqrSIMD8ExtAVX512(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
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

#endif // __AVX512F__

//============================================================
// L2 距离空间 —— float 特化
// 运行时根据 CPU SIMD 能力和向量维度的对齐情况，动态选择性能最优的距离函数。
// 优先级：AVX-512 > AVX > SSE > Scalar。
//============================================================

class L2SpaceFloat : public SpaceInterface<float> {
    DISTFUNC<float> dist_func_;
    size_t data_size_;
    size_t dim_;

public:
    L2SpaceFloat(size_t dim) : dim_(dim), data_size_(dim * sizeof(float)) {
        // 运行时动态选择性能最优的距离计算函数
        dist_func_ = L2SqrScalar;  // 默认使用标量实现作为回退方案

#if defined(__AVX512F__) || defined(USE_AVX512)
        if (AVX512Capable()) {
            // AVX-512：每次迭代处理 16 个 float 元素
            dist_func_ = L2SqrSIMD16ExtAVX512;
        } else
#endif
#if defined(__AVX__) || defined(USE_AVX) || defined(USE_AVX2)
        if (AVXCapable()) {
            if (dim % 16 == 0) {
                dist_func_ = L2SqrSIMD16ExtAVX;
            } else {
                dist_func_ = L2SqrSIMD8ExtAVX;
            }
        } else
#endif
#if defined(__SSE__) || defined(USE_SSE)
        if (SSECapable()) {
            if (dim % 16 == 0) {
                dist_func_ = L2SqrSIMD16ExtSSE;
            } else if (dim % 4 == 0) {
                dist_func_ = L2SqrSIMD4ExtSSE;
            }
        }
#endif
        ;  // if-else 链结尾（防止编译器对悬空 else 报警）
    }

    DISTFUNC<float> get_dist_func() override { return dist_func_; }
    void* get_dist_func_param() override { return &dim_; }
    size_t get_data_size() override { return data_size_; }
};

//============================================================
// L2 距离空间 —— double 特化
// 与 float 版本类似，根据 CPU SIMD 能力动态选择最优的 double 距离函数。
//============================================================

class L2SpaceDouble : public SpaceInterface<double> {
    DISTFUNC<double> dist_func_;
    size_t data_size_;
    size_t dim_;

public:
    L2SpaceDouble(size_t dim) : dim_(dim), data_size_(dim * sizeof(double)) {
        dist_func_ = L2SqrScalarDouble;  // 默认 scalar

#if defined(__AVX512F__) || defined(USE_AVX512)
        if (AVX512Capable()) {
            dist_func_ = L2SqrSIMD8ExtAVX512;
        } else
#endif
#if defined(__AVX__) || defined(USE_AVX) || defined(USE_AVX2)
        if (AVXCapable()) {
            dist_func_ = L2SqrSIMD4ExtAVX;
        }
#endif
        ;
    }

    DISTFUNC<double> get_dist_func() override { return dist_func_; }
    void* get_dist_func_param() override { return &dim_; }
    size_t get_data_size() override { return data_size_; }
};

//============================================================
// 内积距离空间 —— float 特化
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
// 在 HNSWGraph 的模板代码中使用，根据 T=float/double 自动分发。
//============================================================

template<typename T>
inline std::unique_ptr<SpaceInterface<T>> create_l2_space(size_t dim);

template<>
inline std::unique_ptr<SpaceInterface<float>> create_l2_space<float>(size_t dim) {
    return std::make_unique<L2SpaceFloat>(dim);
}

template<>
inline std::unique_ptr<SpaceInterface<double>> create_l2_space<double>(size_t dim) {
    return std::make_unique<L2SpaceDouble>(dim);
}

//============================================================
// 旧接口兼容包装函数（供过渡期使用）
// 封装为直接接受指针/维度参数的调用形式，
// 内部通过 Scalar 实现计算，与原有调用方式完全兼容。
//============================================================

inline float L2Sqr(const float* pVect1, const float* pVect2, const void* qty_ptr) {
    return L2SqrScalar((const void*)pVect1, (const void*)pVect2, qty_ptr);
}

inline float L2Sqr(const float* pVect1, const float* pVect2, size_t qty) {
    return L2SqrScalar((const void*)pVect1, (const void*)pVect2, (const void*)&qty);
}

inline double L2Sqr(const double* pVect1, const double* pVect2, const void* qty_ptr) {
    return L2SqrScalarDouble((const void*)pVect1, (const void*)pVect2, qty_ptr);
}

inline double L2Sqr(const double* pVect1, const double* pVect2, size_t qty) {
    return L2SqrScalarDouble((const void*)pVect1, (const void*)pVect2, (const void*)&qty);
}

inline float IPSqr(const float* pVect1, const float* pVect2, size_t qty) {
    return IPSqrScalar((const void*)pVect1, (const void*)pVect2, (const void*)&qty);
}

inline float IPSqr(const float* pVect1, const float* pVect2, const void* qty_ptr) {
    return IPSqrScalar((const void*)pVect1, (const void*)pVect2, qty_ptr);
}
