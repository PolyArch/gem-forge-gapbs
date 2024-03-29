// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#ifndef UTIL_H_
#define UTIL_H_

#include <cassert>
#include <cinttypes>
#include <fstream>
#include <omp.h>
#include <stdio.h>
#include <string>
#include <vector>

#include "timer.h"

#ifdef GEM_FORGE
#include "gem5/m5ops.h"
#endif

#ifdef GEM_FORGE
#define gf_detail_sim_start() m5_detail_sim_start()
#define gf_detail_sim_end() m5_detail_sim_end()
#define gf_reset_stats() m5_reset_stats(0, 0)
#define gf_dump_stats() m5_dump_stats(0, 0)
#define gf_panic() m5_panic()
#define gf_work_begin(x) m5_work_begin(x, 0)
#define gf_work_end(x) m5_work_end(x, 0)

#define gf_stream_nuca_region1d(name, start, elementSize, dim1)                \
  m5_stream_nuca_region(name, start, elementSize, dim1, 0, 0)
#define gf_stream_nuca_region2d(name, start, elementSize, dim1, dim2)          \
  m5_stream_nuca_region(name, start, elementSize, dim1, dim2, 0)
#define gf_stream_nuca_region3d(name, start, elementSize, dim1, dim2, dim3)    \
  m5_stream_nuca_region(name, start, elementSize, dim1, dim2, dim3)
#define get_4th_arg(name, start, elemSize, arg1, arg2, arg3, arg4, ...) arg4
#define gf_stream_nuca_region(...)                                             \
  get_4th_arg(__VA_ARGS__, gf_stream_nuca_region3d, gf_stream_nuca_region2d,   \
              gf_stream_nuca_region1d)(__VA_ARGS__)

#define gf_stream_nuca_align(A, B, elementOffset)                              \
  m5_stream_nuca_align(A, B, elementOffset)
#define gf_stream_nuca_set_property(start, property, value)                    \
  m5_stream_nuca_set_property(start, property, value)
#define gf_stream_nuca_remap() m5_stream_nuca_remap()

#else
#define gf_detail_sim_start()
#define gf_detail_sim_end()
#define gf_reset_stats()
#define gf_dump_stats()
#define gf_panic() assert(0 && "gf_panic")
#define gf_work_begin(x)
#define gf_work_end(x)
#define gf_stream_nuca_region(args...)
#define gf_stream_nuca_align(args...)
#define gf_stream_nuca_set_property(args...)
#define gf_stream_nuca_remap()
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

uint32_t roundUp(uint32_t v, uint32_t k) { return ((v + k - 1) / k) * k; }

uint32_t log2Pow2(uint32_t v) {
  int log2;
  for (log2 = 0; v != 1; v >>= 1, log2++) {
  }
  return log2;
}

__attribute__((noinline)) uint64_t gf_warm_impl(char *buffer,
                                                uint64_t totalBytes) {
  auto N = totalBytes / 64;
  uint64_t x = 0;
#pragma clang loop unroll(disable) vectorize(disable) interleave(disable)
  for (uint64_t i = 0; i < N; i++) {
    x += buffer[i * 64];
  }
  return x;
}

void gf_warm_array(const char *name, void *buffer, uint64_t totalBytes,
                   int checkCached = 1) {
  // Default assume all cached.
  uint64_t cachedBytes = totalBytes;
#ifdef GEM_FORGE
  if (checkCached) {
    cachedBytes = m5_stream_nuca_get_cached_bytes(buffer);
  }
#endif
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
        __attribute__((unused)) volatile auto x = gf_warm_impl(lhs, rhs - lhs);
      } else {
        __attribute__((unused)) volatile auto x = gf_warm_impl(lhs, end - lhs);
      }
    }
  }
  printf("[GF_WARM] Region %s Warmed %.2f%%.\n", name,
         static_cast<float>(cachedBytes) / static_cast<float>(totalBytes) *
             100.f);
}

/**
 * Read the partiton file.
 */
std::vector<int64_t> getNodePartitionSizes(const std::string &graphFn) {
  auto pos = graphFn.rfind('.');
  assert(pos != std::string::npos);
  auto prefix = graphFn.substr(0, pos);
  auto partitionFn = prefix + ".part.txt";
  std::ifstream f(partitionFn);
  std::vector<int64_t> partSizes;
  if (!f.is_open()) {
    return partSizes;
  }
  std::string field;
  f >> field;
  assert(field == "PartSize");
  int nParts;
  f >> nParts;
  assert(nParts > 0);
  for (int i = 0; i < nParts; ++i) {
    size_t partSize;
    f >> partSize;
    partSizes.push_back(partSize);
  }
  return partSizes;
}

__attribute__((noinline)) void startThreads(int num_threads) {
  omp_set_num_threads(num_threads);
  float v;
  float *pv = &v;
#pragma omp parallel for schedule(static)
  for (int i = 0; i < num_threads; ++i) {
    __attribute__((unused)) volatile float v = *pv;
  }
}

using ThreadWorkVecT = std::vector<std::pair<int64_t, int64_t>>;

__attribute__((noinline)) ThreadWorkVecT generateThreadWork(int64_t total,
                                                            int threads) {
  auto part = (total + threads - 1) / threads;
  ThreadWorkVecT ret;
  int64_t accWork = 0;
  for (int i = 0; i < threads; ++i) {
    auto work = std::min(part, total - accWork);
    ret.emplace_back(accWork, accWork + work);
    accWork += work;
  }
  assert(accWork == total);
  return ret;
}

__attribute__((noinline)) ThreadWorkVecT
fuseWork(const std::vector<int64_t> &part, int threads) {
  auto numParts = part.size();
  auto partsPerThread = (numParts + threads - 1) / threads;
  ThreadWorkVecT ret;
  int64_t accWork = 0;
  for (int i = 0; i < threads; ++i) {
    int64_t work = 0;
    for (int j = i * partsPerThread;
         j < std::min(numParts, (i + 1) * partsPerThread); ++j) {
      work += part[j];
    }
    ret.emplace_back(accWork, accWork + work);
    accWork += work;
  }
  int64_t total = 0;
  for (auto x : part) {
    total += x;
  }
  assert(accWork == total);
  return ret;
}

template <typename NodeID>
__attribute__((noinline)) NodeID *initShuffledNodes(int64_t num_nodes) {
  NodeID *nodes = alignedAllocAndTouch<NodeID>(num_nodes);
  for (NodeID n = 0; n < num_nodes; n++) {
    nodes[n] = n;
  }
  for (NodeID i = 0; i + 1 < num_nodes; ++i) {
    // Shuffle a little bit to make it not always linear access.
    long long j = (rand() % (num_nodes - i)) + i;
    NodeID tmp = nodes[i];
    nodes[i] = nodes[j];
    nodes[j] = tmp;
  }
  return nodes;
}

#endif // UTIL_H_
