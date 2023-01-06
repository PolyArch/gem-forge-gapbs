#ifndef SIZED_ARRAY_H_
#define SIZED_ARRAY_H_

/**
 * This is just std::array but with size remembered.
 * @author Zhengrong Wang
 */

#include <array>
#include <cassert>
#include <cstdlib>
#include <cstring>

template <typename T> class SizedArray {
public:
  SizedArray(uint64_t _capacity)
      : capacity(_capacity), buffer(nullptr), num_elements(0) {
    this->buffer =
        reinterpret_cast<T *>(aligned_alloc(64, sizeof(T) * this->capacity));
  }

  SizedArray(const SizedArray<T> &other) : SizedArray<T>(other.capacity) {
    memcpy(this->buffer, other.buffer, other.capacity * sizeof(T));
  }
  SizedArray &operator=(const SizedArray<T> &other) {
    this->release();
    this->capacity = other.capacity;
    this->num_elements = other.num_elements;
    this->buffer =
        reinterpret_cast<T *>(aligned_alloc(64, sizeof(T) * this->capacity));
    memcpy(this->buffer, other.buffer, other.capacity * sizeof(T));
    return *this;
  }

  SizedArray(SizedArray<T> &&other)
      : capacity(other.capacity), buffer(other.buffer),
        num_elements(other.num_elements) {
    other.buffer = nullptr;
    other.capacity = 0;
    other.num_elements = 0;
  }
  SizedArray &operator=(SizedArray<T> &&other) {
    this->release();
    this->capacity = other.capacity;
    this->num_elements = other.num_elements;
    this->buffer = other.buffer;
    other.buffer = nullptr;
    other.capacity = 0;
    other.num_elements = 0;
    return *this;
  }

  ~SizedArray() { this->release(); }

  using iterator = T *;

  iterator begin() { return buffer; }
  iterator end() { return buffer + this->num_elements; }

  void resize(size_t n) {
    assert(n <= this->capacity);
    this->num_elements = n;
  }

  void push_back(const T &v) {
    assert(this->num_elements < this->capacity);
    this->buffer[this->num_elements] = v;
    this->num_elements++;
  }

  bool empty() const { return this->num_elements == 0; }

  int32_t size() const { return this->num_elements; }

  void clear() { this->num_elements = 0; }

  int32_t capacity;
  iterator buffer;
  int32_t num_elements;

  void release() {
    free(this->buffer);
    this->buffer = nullptr;
    this->capacity = 0;
    this->num_elements = 0;
  }
};

#endif