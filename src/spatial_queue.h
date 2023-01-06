#ifndef SPATIAL_QUEUE_H_
#define SPATIAL_QUEUE_H_

#include "platform_atomics.h"
#include "util.h"

template <typename T> class SpatialQueue {
public:
  struct QueueMetaInfo {
    // Making this one cache line.
    int size;
    int padd[15];
  };

  SpatialQueue(int _num_queues, int _queue_capacity, int _hash_shift,
               int _hash_mask)
      : num_queues(_num_queues), queue_capacity(_queue_capacity),
        hash_shift(_hash_shift), hash_mask(_hash_mask) {

    this->data = alignedAllocAndTouch<T>(num_queues * queue_capacity);
    this->meta = alignedAllocAndTouch<QueueMetaInfo>(num_queues);
    for (int i = 0; i < this->num_queues; ++i) {
      this->meta[i].size = 0;
    }
  }

  ~SpatialQueue() {
    free(this->data);
    free(this->meta);
  }

  int size(int queue_idx) const { return this->meta[queue_idx].size; }
  void clear(int queue_idx) { this->meta[queue_idx].size = 0; }

  int getQueueIdx(int v) const { return (v >> hash_mask) & hash_mask; }

  const int num_queues;
  const int queue_capacity;

  const int hash_shift;
  const int hash_mask;

  T *data;
  QueueMetaInfo *meta;
};

#endif