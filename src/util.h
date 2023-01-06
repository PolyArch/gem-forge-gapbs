// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#ifndef UTIL_H_
#define UTIL_H_

#include <cinttypes>
#include <omp.h>
#include <stdio.h>
#include <string>

#include "timer.h"

#ifdef GEM_FORGE
#include "gem5/m5ops.h"
#endif

/*
GAP Benchmark Suite
Author: Scott Beamer

Miscellaneous helpers that don't fit into classes
*/

static const int64_t kRandSeed = 27491095;

void PrintLabel(const std::string &label, const std::string &val) {
  printf("%-21s%7s\n", (label + ":").c_str(), val.c_str());
}

void PrintTime(const std::string &s, double seconds) {
  printf("%-21s%3.5lf\n", (s + ":").c_str(), seconds);
}

void PrintStep(const std::string &s, int64_t count) {
  printf("%-14s%14" PRId64 "\n", (s + ":").c_str(), count);
}

void PrintStep(int step, double seconds, int64_t count = -1) {
  if (count != -1)
    printf("%5d%11" PRId64 "  %10.5lf\n", step, count, seconds);
  else
    printf("%5d%23.5lf\n", step, seconds);
}

void PrintStep(const std::string &s, double seconds, int64_t count = -1) {
  if (count != -1)
    printf("%5s%11" PRId64 "  %10.5lf\n", s.c_str(), count, seconds);
  else
    printf("%5s%23.5lf\n", s.c_str(), seconds);
}

// Runs op and prints the time it took to execute labelled by label
#define TIME_PRINT(label, op)                                                  \
  {                                                                            \
    Timer t_;                                                                  \
    t_.Start();                                                                \
    (op);                                                                      \
    t_.Stop();                                                                 \
    PrintTime(label, t_.Seconds());                                            \
  }

template <typename T_> class RangeIter {
  T_ x_;

public:
  explicit RangeIter(T_ x) : x_(x) {}
  bool operator!=(RangeIter const &other) const { return x_ != other.x_; }
  T_ const &operator*() const { return x_; }
  RangeIter &operator++() {
    ++x_;
    return *this;
  }
};

template <typename T_> class Range {
  T_ from_;
  T_ to_;

public:
  explicit Range(T_ to) : from_(0), to_(to) {}
  Range(T_ from, T_ to) : from_(from), to_(to) {}
  RangeIter<T_> begin() const { return RangeIter<T_>(from_); }
  RangeIter<T_> end() const { return RangeIter<T_>(to_); }
};

static constexpr std::size_t alignBytes = 4096;
template <typename T> T *alignedAllocAndTouch(size_t numElements) {
  auto totalBytes = sizeof(T) * numElements;
  if (totalBytes % alignBytes) {
    totalBytes = (totalBytes / alignBytes + 1) * alignBytes;
  }
  auto p = reinterpret_cast<T *>(aligned_alloc(alignBytes, totalBytes));

  auto raw = reinterpret_cast<char *>(p);
  for (unsigned long Byte = 0; Byte < totalBytes; Byte += alignBytes) {
    raw[Byte] = 0;
  }
  return p;
}

uint32_t roundUpPow2(uint32_t v) {
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;
  return v;
}

uint32_t log2Pow2(uint32_t v) {
  int log2;
  for (log2 = 0; v != 1; v >>= 1, log2++) {
  }
  return log2;
}

#ifdef GEM_FORGE

__attribute__((noinline)) void gf_warm_impl(char *buffer, uint64_t totalBytes) {
#pragma clang loop unroll(disable) vectorize(disable) interleave(disable)
  for (uint64_t i = 0; i < totalBytes; i += 64) {
    __attribute__((unused)) volatile char v = buffer[i];
  }
}

void gf_warm_array(const char *name, void *buffer, uint64_t totalBytes) {
  uint64_t cachedBytes = m5_stream_nuca_get_cached_bytes(buffer);
  printf("[GF_WARM] Region %s TotalBytes %lu CachedBytes %lu Cached %.2f%%.\n",
         name, totalBytes, cachedBytes,
         static_cast<float>(cachedBytes) / static_cast<float>(totalBytes) *
             100.f);
  assert(cachedBytes <= totalBytes);

  char *data = reinterpret_cast<char *>(buffer);
#pragma omp parallel firstprivate(data, cachedBytes)
  {
    int threads = omp_get_num_threads();
    int threadId = omp_get_thread_num();
    uint64_t bytesPerThread = (cachedBytes + threads - 1) / threads;
    char *lhs = data + threadId * bytesPerThread;
    char *rhs = lhs + bytesPerThread;
    char *end = data + cachedBytes;
    if (lhs < end) {
      if (rhs <= end) {
        gf_warm_impl(lhs, rhs - lhs);
      } else {
        gf_warm_impl(lhs, end - lhs);
      }
    }
  }
  // #pragma omp parallel for firstprivate(buffer)
  //   for (uint64_t i = 0; i < cachedBytes; i += 64) {
  //     __attribute__((unused)) volatile uint8_t data =
  //         reinterpret_cast<uint8_t *>(buffer)[i];
  //   }
  printf("[GF_WARM] Region %s Warmed %.2f%%.\n", name,
         static_cast<float>(cachedBytes) / static_cast<float>(totalBytes) *
             100.f);
}
#endif

#endif // UTIL_H_
