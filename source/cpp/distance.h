#pragma once

#include <vector>

#if defined(USE_AVX) || defined(USE_AVX2) || defined(USE_AVX512)
#include <immintrin.h>
#endif

template <typename T>

T distance(const std::vector<T> &a, const std::vector<T> &b)
{
    const std::size_t n = std::min(a.size(), b.size());

#ifdef USE_AVX512
    if constexpr (std::is_same<T, float>::value) {
        const float *pa = a.data();
        const float *pb = b.data();
        std::size_t i = 0;
        __m512 vsum = _mm512_setzero_ps();
        for (; i + 16 <= n; i += 16) {
            __m512 va = _mm512_loadu_ps(pa + i);
            __m512 vb = _mm512_loadu_ps(pb + i);
            __m512 d = _mm512_sub_ps(va, vb);
            __m512 m = _mm512_mul_ps(d, d);
            vsum = _mm512_add_ps(vsum, m);
        }
        alignas(64) float tmp[16];
        _mm512_store_ps(tmp, vsum);
        double s = 0.0;
        for (int k = 0; k < 16; ++k) s += tmp[k];
        for (; i < n; ++i) { double d = static_cast<double>(pa[i] - pb[i]); s += d * d; }
        return static_cast<T>(s);
    } else if constexpr (std::is_same<T, double>::value) {
        const double *pa = a.data();
        const double *pb = b.data();
        std::size_t i = 0;
        __m512d vsum = _mm512_setzero_pd();
        for (; i + 8 <= n; i += 8) {
            __m512d va = _mm512_loadu_pd(pa + i);
            __m512d vb = _mm512_loadu_pd(pb + i);
            __m512d d = _mm512_sub_pd(va, vb);
            __m512d m = _mm512_mul_pd(d, d);
            vsum = _mm512_add_pd(vsum, m);
        }
        alignas(64) double tmp[8];
        _mm512_store_pd(tmp, vsum);
        double s = 0.0;
        for (int k = 0; k < 8; ++k) s += tmp[k];
        for (; i < n; ++i) { double d = pa[i] - pb[i]; s += d * d; }
        return static_cast<T>(s);
    }
#elif defined(USE_AVX)
    if constexpr (std::is_same<T, float>::value) {
        const float *pa = a.data();
        const float *pb = b.data();
        std::size_t i = 0;
        __m256 vsum = _mm256_setzero_ps();
        for (; i + 8 <= n; i += 8) {
            __m256 va = _mm256_loadu_ps(pa + i);
            __m256 vb = _mm256_loadu_ps(pb + i);
            __m256 d = _mm256_sub_ps(va, vb);
            __m256 m = _mm256_mul_ps(d, d);
            vsum = _mm256_add_ps(vsum, m);
        }
        float tmp[8];
        _mm256_storeu_ps(tmp, vsum);
        double s = 0.0;
        for (int k = 0; k < 8; ++k) s += tmp[k];
        for (; i < n; ++i) { double d = static_cast<double>(pa[i] - pb[i]); s += d * d; }
        return static_cast<T>(s);
    } else if constexpr (std::is_same<T, double>::value) {
        const double *pa = a.data();
        const double *pb = b.data();
        std::size_t i = 0;
        __m256d vsum = _mm256_setzero_pd();
        for (; i + 4 <= n; i += 4) {
            __m256d va = _mm256_loadu_pd(pa + i);
            __m256d vb = _mm256_loadu_pd(pb + i);
            __m256d d = _mm256_sub_pd(va, vb);
            __m256d m = _mm256_mul_pd(d, d);
            vsum = _mm256_add_pd(vsum, m);
        }
        double tmp[4];
        _mm256_storeu_pd(tmp, vsum);
        double s = tmp[0] + tmp[1] + tmp[2] + tmp[3];
        for (; i < n; ++i) { double d = pa[i] - pb[i]; s += d * d; }
        return static_cast<T>(s);
    }
#endif

    // fallback scalar implementation (also used when T not float/double or AVX not available)
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i)
    {
        double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        s += d * d;
    }
    return static_cast<T>(s);
}