#include <immintrin.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "anns/search_interface.h"

namespace {

using Clock = std::chrono::steady_clock;

#if defined(__GNUC__)
__attribute__((optimize("no-tree-vectorize")))
#endif
int l2_u8_scalar(const uint8_t *a, const uint8_t *b, size_t dim) {
  int sum = 0;
  for (size_t i = 0; i < dim; ++i) {
    int diff = static_cast<int>(a[i]) - static_cast<int>(b[i]);
    sum += diff * diff;
  }
  return sum;
}

int l2_u8_avx2(const uint8_t *a, const uint8_t *b, size_t dim) {
  size_t i = 0;
  __m256i total = _mm256_setzero_si256();
  const __m256i zero = _mm256_setzero_si256();
  for (; i + 32 <= dim; i += 32) {
    __m256i av = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + i));
    __m256i bv = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + i));
    __m256i alo = _mm256_unpacklo_epi8(av, zero);
    __m256i ahi = _mm256_unpackhi_epi8(av, zero);
    __m256i blo = _mm256_unpacklo_epi8(bv, zero);
    __m256i bhi = _mm256_unpackhi_epi8(bv, zero);
    __m256i dlo = _mm256_sub_epi16(alo, blo);
    __m256i dhi = _mm256_sub_epi16(ahi, bhi);
    total = _mm256_add_epi32(total, _mm256_madd_epi16(dlo, dlo));
    total = _mm256_add_epi32(total, _mm256_madd_epi16(dhi, dhi));
  }

  alignas(32) int partial[8];
  _mm256_store_si256(reinterpret_cast<__m256i *>(partial), total);
  int sum = 0;
  for (int value : partial) {
    sum += value;
  }
  for (; i < dim; ++i) {
    int diff = static_cast<int>(a[i]) - static_cast<int>(b[i]);
    sum += diff * diff;
  }
  return sum;
}

template <typename Fn>
double bench_l2(const char *name, Fn &&fn, const std::vector<uint8_t> &data,
                const std::vector<uint8_t> &query, size_t dim, size_t repeats,
                size_t threads, uint64_t &sink) {
  const size_t vectors = data.size() / dim;
  omp_set_num_threads(static_cast<int>(threads));
  auto start = Clock::now();
  uint64_t total = 0;
#pragma omp parallel for reduction(+ : total) schedule(static)
  for (size_t r = 0; r < repeats; ++r) {
    for (size_t i = 0; i < vectors; ++i) {
      total += static_cast<uint32_t>(fn(query.data(), data.data() + i * dim, dim));
    }
  }
  auto end = Clock::now();
  sink += total;
  double sec = std::chrono::duration<double>(end - start).count();
  double distances = static_cast<double>(vectors) * static_cast<double>(repeats);
  double gb = distances * static_cast<double>(dim) / 1e9;
  std::cout << name << '\t' << threads << '\t' << vectors << '\t' << dim << '\t'
            << repeats << '\t' << sec << '\t' << distances / sec << '\t'
            << gb / sec << '\n';
  return sec;
}

double bench_read(const std::vector<uint8_t> &data, size_t dim, size_t repeats,
                  size_t threads, uint64_t &sink) {
  const size_t vectors = data.size() / dim;
  omp_set_num_threads(static_cast<int>(threads));
  auto start = Clock::now();
  uint64_t total = 0;
#pragma omp parallel for reduction(+ : total) schedule(static)
  for (size_t r = 0; r < repeats; ++r) {
    for (size_t i = 0; i < data.size(); ++i) {
      total += data[i];
    }
  }
  auto end = Clock::now();
  sink += total;
  double sec = std::chrono::duration<double>(end - start).count();
  double gb = static_cast<double>(data.size()) * static_cast<double>(repeats) / 1e9;
  std::cout << "mem_read" << '\t' << threads << '\t' << vectors << '\t' << dim
            << '\t' << repeats << '\t' << sec << '\t' << 0 << '\t' << gb / sec
            << '\n';
  return sec;
}

size_t get_arg(int argc, char **argv, const std::string &name, size_t fallback) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (argv[i] == name) {
      return std::stoull(argv[i + 1]);
    }
  }
  return fallback;
}

}  // namespace

int main(int argc, char **argv) {
  size_t dim = get_arg(argc, argv, "--dim", 128);
  size_t vectors = get_arg(argc, argv, "--vectors", 1000000);
  size_t repeats = get_arg(argc, argv, "--repeats", 10);
  size_t threads = get_arg(argc, argv, "--threads", 8);

  std::vector<uint8_t> data(vectors * dim);
  std::vector<uint8_t> query(dim);
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> dist(0, 255);
  std::generate(data.begin(), data.end(), [&] { return static_cast<uint8_t>(dist(rng)); });
  std::generate(query.begin(), query.end(), [&] { return static_cast<uint8_t>(dist(rng)); });

  hnswlib::L2SpaceI space(dim);
  auto current_func = space.get_dist_func();
  auto *current_param = space.get_dist_func_param();
  uint64_t sink = 0;

  std::cout << "bench\tthreads\tvectors\tdim\trepeats\tseconds\tdist_per_s\tgb_per_s\n";
  bench_read(data, dim, repeats, threads, sink);
  bench_l2("l2_current", [&](const uint8_t *a, const uint8_t *b, size_t) {
    return current_func(a, b, current_param);
  }, data, query, dim, repeats, threads, sink);
  bench_l2("l2_scalar_no_vec", l2_u8_scalar, data, query, dim, repeats, threads, sink);
  bench_l2("l2_avx2_manual", l2_u8_avx2, data, query, dim, repeats, threads, sink);

  std::cerr << "sink=" << sink << '\n';
  return 0;
}
