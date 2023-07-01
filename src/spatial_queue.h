#ifndef SPATIAL_QUEUE_H_
#define SPATIAL_QUEUE_H_

#include "platform_atomics.h"
#include "util.h"

/**
 * Implements a spatially distributed queue.
 * Each queue can further be divided into bins (used for sssp).
 */

template <typename T> class SpatialQueue {
public:
  /**
   * For now simply assume at most 128 bins per queue.
   */
  static constexpr int MaxBinsPerQueue = 128;
  struct QueueMetaInfo {
    int size[MaxBinsPerQueue];
    // Used to implement scout count in bfs push.
    int weightedSize[MaxBinsPerQueue];
  };

  SpatialQueue(int _num_queues, int _num_bins, int _queue_capacity,
               int _hash_div, int _hash_mask)
      : num_queues(_num_queues), num_bins(_num_bins),
        queue_capacity(_queue_capacity),
        bin_capacity(_queue_capacity / _num_bins), hash_div(_hash_div),
        hash_mask(_hash_mask) {

    this->data = alignedAllocAndTouch<T>(num_queues * queue_capacity);
    this->meta = alignedAllocAndTouch<QueueMetaInfo>(num_queues);
#pragma clang loop unroll(disable) vectorize(disable) interleave(disable)
    for (int i = 0; i < this->num_queues; ++i) {
#pragma clang loop unroll(disable) vectorize(disable) interleave(disable)
      for (int j = 0; j < this->num_bins; ++j) {
        this->meta[i].size[j] = 0;
      }
    }
  }

  ~SpatialQueue() {
    free(this->data);
    free(this->meta);
  }

  int size(int queue_idx, int bin_idx) const {
    return this->meta[queue_idx].size[bin_idx];
  }
  int totalSize() const {
    int ret = 0;
    for (int queue_idx = 0; queue_idx < this->num_queues; ++queue_idx) {
      for (int bin_idx = 0; bin_idx < this->num_bins; ++bin_idx) {
        ret += this->size(queue_idx, bin_idx);
      }
    }
    return ret;
  }
  int weightedSize(int queue_idx, int bin_idx) const {
    return this->meta[queue_idx].weightedSize[bin_idx];
  }
  int getAndClearTotalWeightedSize() const {
    int ret = 0;
    for (int queue_idx = 0; queue_idx < this->num_queues; ++queue_idx) {
      for (int bin_idx = 0; bin_idx < this->num_bins; ++bin_idx) {
        ret += this->weightedSize(queue_idx, bin_idx);
        this->meta[queue_idx].weightedSize[bin_idx] = 0;
      }
    }
    return ret;
  }
  void clear(int queue_idx, int bin_idx) {
    this->meta[queue_idx].size[bin_idx] = 0;
    this->meta[queue_idx].weightedSize[bin_idx] = 0;
  }
  void clear() {
    for (int queue_idx = 0; queue_idx < this->num_queues; ++queue_idx) {
      for (int bin_idx = 0; bin_idx < this->num_bins; ++bin_idx) {
        this->clear(queue_idx, bin_idx);
      }
    }
  }
  bool empty() const {
    for (int queue_idx = 0; queue_idx < this->num_queues; ++queue_idx) {
      for (int bin_idx = 0; bin_idx < this->num_bins; ++bin_idx) {
        if (this->size(queue_idx, bin_idx) > 0) {
          return false;
        }
      }
    }
    return true;
  }

  int getQueueIdx(T v) const { return (v / hash_div) & hash_mask; }
  void enque(T v, int bin_idx) {
    auto queue_idx = getQueueIdx(v);
    auto loc = this->meta[queue_idx].size[bin_idx]++;
    this->data[queue_idx * queue_capacity + bin_idx * bin_capacity + loc] = v;
  }

  const int num_queues;
  const int num_bins;
  const int queue_capacity;
  const int bin_capacity;

  const int hash_div;
  const int hash_mask;

  T *data;
  QueueMetaInfo *meta;
};

#endif