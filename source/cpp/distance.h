#pragma once

#include <cstddef>
#include <cstdint>

// SIMD headers
#if defined(USE_AVX512)
#include <immintrin.h>
#elif defined(USE_AVX) || defined(USE_AVX2)
#include <immintrin.h>
#elif defined(USE_SSE)
#include <xmmintrin.h>  // SSE
#include <emmintrin.h>  // SSE2
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
// Scalar fallback implementations
//============================================================

inline float L2SqrScalar(const float* pVect1, const float* pVect2, size_t qty) {
    float res = 0;
    for (size_t i = 0; i < qty; i++) {
        float t = *pVect1 - *pVect2;
        pVect1++;
        pVect2++;
        res += t * t;
    }
    return res;
}

inline double L2SqrScalar(const double* pVect1, const double* pVect2, size_t qty) {
    double res = 0;
    for (size_t i = 0; i < qty; i++) {
        double t = *pVect1 - *pVect2;
        pVect1++;
        pVect2++;
        res += t * t;
    }
    return res;
}

//============================================================
// SSE implementation (128-bit, 4 floats per iteration)
//============================================================
#if defined(USE_SSE)

inline float L2SqrSIMD4ExtSSE(const float* pVect1, const float* pVect2, size_t qty) {
    float PORTABLE_ALIGN32 TmpRes[4];
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

// Process 16 floats at a time (4 SSE operations)
inline float L2SqrSIMD16ExtSSE(const float* pVect1, const float* pVect2, size_t qty) {
    float PORTABLE_ALIGN32 TmpRes[4];
    size_t qty16 = qty >> 4;
    
    const float* pEnd1 = pVect1 + (qty16 << 4);
    
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
        
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
        
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
        
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
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

#endif // USE_SSE

//============================================================
// AVX implementation (256-bit, 8 floats per iteration)
//============================================================
#if defined(USE_AVX) || defined(USE_AVX2)

inline float L2SqrSIMD8ExtAVX(const float* pVect1, const float* pVect2, size_t qty) {
    float PORTABLE_ALIGN32 TmpRes[8];
    size_t qty8 = qty >> 3;
    
    const float* pEnd1 = pVect1 + (qty8 << 3);
    
    __m256 diff, v1, v2;
    __m256 sum = _mm256_set1_ps(0);
    
    while (pVect1 < pEnd1) {
        _mm_prefetch((char*)(pVect1 + 32), _MM_HINT_T0);
        _mm_prefetch((char*)(pVect2 + 32), _MM_HINT_T0);
        
        v1 = _mm256_loadu_ps(pVect1);
        pVect1 += 8;
        v2 = _mm256_loadu_ps(pVect2);
        pVect2 += 8;
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

// Process 16 floats at a time (2 AVX operations)
inline float L2SqrSIMD16ExtAVX(const float* pVect1, const float* pVect2, size_t qty) {
    float PORTABLE_ALIGN32 TmpRes[8];
    size_t qty16 = qty >> 4;
    
    const float* pEnd1 = pVect1 + (qty16 << 4);
    
    __m256 diff, v1, v2;
    __m256 sum = _mm256_set1_ps(0);
    
    while (pVect1 < pEnd1) {
        _mm_prefetch((char*)(pVect1 + 32), _MM_HINT_T0);
        _mm_prefetch((char*)(pVect2 + 32), _MM_HINT_T0);
        
        v1 = _mm256_loadu_ps(pVect1);
        pVect1 += 8;
        v2 = _mm256_loadu_ps(pVect2);
        pVect2 += 8;
        diff = _mm256_sub_ps(v1, v2);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
        
        v1 = _mm256_loadu_ps(pVect1);
        pVect1 += 8;
        v2 = _mm256_loadu_ps(pVect2);
        pVect2 += 8;
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

inline double L2SqrSIMD4ExtAVX(const double* pVect1, const double* pVect2, size_t qty) {
    double PORTABLE_ALIGN32 TmpRes[4];
    size_t qty4 = qty >> 2;
    
    const double* pEnd1 = pVect1 + (qty4 << 2);
    
    __m256d diff, v1, v2;
    __m256d sum = _mm256_set1_pd(0);
    
    while (pVect1 < pEnd1) {
        _mm_prefetch((char*)(pVect1 + 8), _MM_HINT_T0);
        _mm_prefetch((char*)(pVect2 + 8), _MM_HINT_T0);
        
        v1 = _mm256_loadu_pd(pVect1);
        pVect1 += 4;
        v2 = _mm256_loadu_pd(pVect2);
        pVect2 += 4;
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

#endif // USE_AVX

//============================================================
// AVX-512 implementation (512-bit, 16 floats per iteration)
//============================================================
#if defined(USE_AVX512)

inline float L2SqrSIMD16ExtAVX512(const float* pVect1, const float* pVect2, size_t qty) {
    float PORTABLE_ALIGN64 TmpRes[16];
    size_t qty16 = qty >> 4;
    
    const float* pEnd1 = pVect1 + (qty16 << 4);
    
    __m512 diff, v1, v2;
    __m512 sum = _mm512_set1_ps(0);
    
    while (pVect1 < pEnd1) {
        _mm_prefetch((char*)(pVect1 + 64), _MM_HINT_T0);
        _mm_prefetch((char*)(pVect2 + 64), _MM_HINT_T0);
        
        v1 = _mm512_loadu_ps(pVect1);
        pVect1 += 16;
        v2 = _mm512_loadu_ps(pVect2);
        pVect2 += 16;
        diff = _mm512_sub_ps(v1, v2);
        sum = _mm512_add_ps(sum, _mm512_mul_ps(diff, diff));
    }
    
    _mm512_store_ps(TmpRes, sum);
    float res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + 
                TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7] + 
                TmpRes[8] + TmpRes[9] + TmpRes[10] + TmpRes[11] + 
                TmpRes[12] + TmpRes[13] + TmpRes[14] + TmpRes[15];
    
    size_t remaining = qty & 15;
    for (size_t i = 0; i < remaining; i++) {
        float t = pVect1[i] - pVect2[i];
        res += t * t;
    }
    
    return res;
}

inline double L2SqrSIMD8ExtAVX512(const double* pVect1, const double* pVect2, size_t qty) {
    double PORTABLE_ALIGN64 TmpRes[8];
    size_t qty8 = qty >> 3;
    
    const double* pEnd1 = pVect1 + (qty8 << 3);
    
    __m512d diff, v1, v2;
    __m512d sum = _mm512_set1_pd(0);
    
    while (pVect1 < pEnd1) {
        _mm_prefetch((char*)(pVect1 + 16), _MM_HINT_T0);
        _mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);
        
        v1 = _mm512_loadu_pd(pVect1);
        pVect1 += 8;
        v2 = _mm512_loadu_pd(pVect2);
        pVect2 += 8;
        diff = _mm512_sub_pd(v1, v2);
        sum = _mm512_add_pd(sum, _mm512_mul_pd(diff, diff));
    }
    
    _mm512_store_pd(TmpRes, sum);
    double res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + 
                 TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];
    
    size_t remaining = qty & 7;
    for (size_t i = 0; i < remaining; i++) {
        double t = pVect1[i] - pVect2[i];
        res += t * t;
    }
    
    return res;
}

#endif // USE_AVX512

//============================================================
// Main L2 distance function - float specialization
//============================================================

inline float L2Sqr(const float* pVect1, const float* pVect2, const void* qty_ptr) {
    size_t qty = *((size_t*)qty_ptr);
    
#if defined(USE_AVX512)
    return L2SqrSIMD16ExtAVX512(pVect1, pVect2, qty);
#elif defined(USE_AVX) || defined(USE_AVX2)
    return L2SqrSIMD16ExtAVX(pVect1, pVect2, qty);
#elif defined(USE_SSE)
    return L2SqrSIMD16ExtSSE(pVect1, pVect2, qty);
#else
    return L2SqrScalar(pVect1, pVect2, qty);
#endif
}

// Overload without void* (more convenient)
inline float L2Sqr(const float* pVect1, const float* pVect2, size_t qty) {
#if defined(USE_AVX512)
    return L2SqrSIMD16ExtAVX512(pVect1, pVect2, qty);
#elif defined(USE_AVX) || defined(USE_AVX2)
    return L2SqrSIMD16ExtAVX(pVect1, pVect2, qty);
#elif defined(USE_SSE)
    return L2SqrSIMD16ExtSSE(pVect1, pVect2, qty);
#else
    return L2SqrScalar(pVect1, pVect2, qty);
#endif
}

//============================================================
// Main L2 distance function - double specialization
//============================================================

inline double L2Sqr(const double* pVect1, const double* pVect2, const void* qty_ptr) {
    size_t qty = *((size_t*)qty_ptr);
    
#if defined(USE_AVX512)
    return L2SqrSIMD8ExtAVX512(pVect1, pVect2, qty);
#elif defined(USE_AVX) || defined(USE_AVX2)
    return L2SqrSIMD4ExtAVX(pVect1, pVect2, qty);
#else
    return L2SqrScalar(pVect1, pVect2, qty);
#endif
}

inline double L2Sqr(const double* pVect1, const double* pVect2, size_t qty) {
#if defined(USE_AVX512)
    return L2SqrSIMD8ExtAVX512(pVect1, pVect2, qty);
#elif defined(USE_AVX) || defined(USE_AVX2)
    return L2SqrSIMD4ExtAVX(pVect1, pVect2, qty);
#else
    return L2SqrScalar(pVect1, pVect2, qty);
#endif
}

//============================================================
// Inner Product distance
//============================================================

inline float IPSqrScalar(const float* pVect1, const float* pVect2, size_t qty) {
    float res = 0;
    for (size_t i = 0; i < qty; i++) {
        res += *pVect1 * *pVect2;
        pVect1++;
        pVect2++;
    }
    return res;
}

#if defined(USE_AVX) || defined(USE_AVX2) || defined(USE_AVX512)

inline float IPSqrAVX(const float* pVect1, const float* pVect2, size_t qty) {
    float PORTABLE_ALIGN32 TmpRes[8];
    size_t qty8 = qty >> 3;
    
    const float* pEnd1 = pVect1 + (qty8 << 3);
    
    __m256 v1, v2;
    __m256 sum = _mm256_set1_ps(0);
    
    while (pVect1 < pEnd1) {
        _mm_prefetch((char*)(pVect1 + 32), _MM_HINT_T0);
        _mm_prefetch((char*)(pVect2 + 32), _MM_HINT_T0);
        
        v1 = _mm256_loadu_ps(pVect1);
        pVect1 += 8;
        v2 = _mm256_loadu_ps(pVect2);
        pVect2 += 8;
        sum = _mm256_add_ps(sum, _mm256_mul_ps(v1, v2));
    }
    
    _mm256_store_ps(TmpRes, sum);
    float res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + 
                TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];
    
    size_t remaining = qty & 7;
    for (size_t i = 0; i < remaining; i++) {
        res += pVect1[i] * pVect2[i];
    }
    
    return res;
}

inline float IPSqr(const float* pVect1, const float* pVect2, size_t qty) {
    return IPSqrAVX(pVect1, pVect2, qty);
}

inline float IPSqr(const float* pVect1, const float* pVect2, const void* qty_ptr) {
    return IPSqrAVX(pVect1, pVect2, *((size_t*)qty_ptr));
}

#else

inline float IPSqr(const float* pVect1, const float* pVect2, size_t qty) {
    return IPSqrScalar(pVect1, pVect2, qty);
}

inline float IPSqr(const float* pVect1, const float* pVect2, const void* qty_ptr) {
    return IPSqrScalar(pVect1, pVect2, *((size_t*)qty_ptr));
}

#endif
