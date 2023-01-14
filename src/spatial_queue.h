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
  };

  SpatialQueue(int _num_queues, int _num_bins, int _queue_capacity,
               int _hash_shift, int _hash_mask)
      : num_queues(_num_queues), num_bins(_num_bins),
        queue_capacity(_queue_capacity),
        bin_capacity(_queue_capacity / _num_bins), hash_shift(_hash_shift),
        hash_mask(_hash_mask) {

    this->data = alignedAllocAndTouch<T>(num_queues * queue_capacity);
    this->meta = alignedAllocAndTouch<QueueMetaInfo>(num_queues);
    for (int i = 0; i < this->num_queues; ++i) {
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
  void clear(int queue_idx, int bin_idx) {
    this->meta[queue_idx].size[bin_idx] = 0;
  }

  int getQueueIdx(int v) const { return (v >> hash_mask) & hash_mask; }

  const int num_queues;
  const int num_bins;
  const int queue_capacity;
  const int bin_capacity;

  const int hash_shift;
  const int hash_mask;

  T *data;
  QueueMetaInfo *meta;
};

#endif