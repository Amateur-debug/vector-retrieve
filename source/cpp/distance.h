#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>

// SIMD headers — 始终包含，运行时决定使用哪个
#if defined(__x86_64__) || defined(_M_X64)
#ifdef __SSE__
#include <xmmintrin.h>  // SSE
#include <emmintrin.h>  // SSE2
#endif
#ifdef __AVX__
#include <immintrin.h>
#endif
#endif

// 兼容旧的编译宏：如果使用旧的 USE_AVX512/USE_AVX/USE_SSE 也能编译
#if defined(USE_AVX512) && !defined(__AVX512F__)
// 编译期没有 AVX512 支持，USE_AVX512 宏无效
#endif

// Alignment macro
#if defined(_MSC_VER)
#define PORTABLE_ALIGN64 __declspec(align(64))
#define PORTABLE_ALIGN32 __declspec(align(32))
#else
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#endif

//============================================================
// DISTFUNC 函数指针类型定义（参考 hnswlib）
// 签名：T(*)(const void* pVect1, const void* pVect2, const void* qty_ptr)
// qty_ptr 指向 const size_t（维度 dim）
//============================================================

template<typename T>
using DISTFUNC = T(*)(const void*, const void*, const void*);

//============================================================
// SpaceInterface 抽象基类（参考 hnswlib）
//============================================================

template<typename T>
class SpaceInterface {
public:
    virtual ~SpaceInterface() = default;
    // 获取距离函数指针
    virtual DISTFUNC<T> get_dist_func() = 0;
    // 获取距离函数参数（指向 dim 的指针）
    virtual void* get_dist_func_param() = 0;
    // 获取维度
    virtual size_t get_data_size() = 0;
};

//============================================================
// 运行时 SIMD 能力检测（参考 hnswlib）
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
        // AVX512F 位于 EBX bit 16
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
        // AVX 位于 ECX bit 28
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
        // SSE 位于 EDX bit 25
        return (cpuInfo[3] & (1 << 25)) != 0;
    }
    return false;
}

#else
// 非 x86 平台，所有 SIMD 能力返回 false
static bool AVX512Capable() { return false; }
static bool AVXCapable() { return false; }
static bool SSECapable() { return false; }
#endif

//============================================================
// Scalar fallback implementations（DISTFUNC 签名）
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
// SSE implementation（DISTFUNC 签名）
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
// AVX implementation（DISTFUNC 签名）
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
// AVX-512 implementation（DISTFUNC 签名）
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
// L2Space — float 特化
// 运行时根据 CPU SIMD 能力和 dim 对齐情况动态选择最优距离函数
//============================================================

class L2SpaceFloat : public SpaceInterface<float> {
    DISTFUNC<float> dist_func_;
    size_t data_size_;
    size_t dim_;

public:
    L2SpaceFloat(size_t dim) : dim_(dim), data_size_(dim * sizeof(float)) {
        // 运行时动态选择最优距离函数
        dist_func_ = L2SqrScalar;  // 默认 scalar

#if defined(__AVX512F__) || defined(USE_AVX512)
        if (AVX512Capable()) {
            // AVX512: 16 floats per iteration
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
        ;  // 结束 if-else 链
    }

    DISTFUNC<float> get_dist_func() override { return dist_func_; }
    void* get_dist_func_param() override { return &dim_; }
    size_t get_data_size() override { return data_size_; }
};

//============================================================
// L2Space — double 特化
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
// InnerProductSpace — float 特化
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
// 便捷函数：创建 SpaceInterface（保持向后兼容）
// 在模板代码中使用，根据 T 类型自动选择对应的 Space 类
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
// 保留旧接口的兼容包装函数（供过渡期使用）
// 这些函数内部通过编译期宏选择，与原有调用方式兼容
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
