#ifndef AFFINITY_ALLOCATOR_HH
#define AFFINITY_ALLOCATOR_HH

#include <cstdlib>
#include <vector>

/**
 * @brief
 * This template assumes a bidirection linked list,
 * i.e. with next and prev pointers.
 *
 * It allocates space in the granularity of arena from malloc.
 *
 * @tparam NodeT
 */
template <class NodeT, int ArenaSize> class AffinityAllocator {
public:
  constexpr static int MaxBanks = 128;

  AffinityAllocator() = default;
  AffinityAllocator(AffinityAllocator &&other) : arenas(other.arenas) {
    other.arenas = nullptr;
  }
  AffinityAllocator &operator=(AffinityAllocator &&other) {
    this->deallocArenas();
    this->arenas = other.arenas;
    other.arenas = nullptr;
  }

  ~AffinityAllocator() { this->deallocArenas(); }

  NodeT *alloc() { return this->allocFromArena(); }

  /*********************************************************
   * Some internal data structure used to manage free space.
   *********************************************************/

  struct AffinityAllocatorArena {
    NodeT data[ArenaSize];
    NodeT *freeList;
    int64_t usedNodes = 0;
    AffinityAllocatorArena *next = nullptr;
    AffinityAllocatorArena *prev = nullptr;

    AffinityAllocatorArena() {

      auto curNode = new (this->data + initFreeNodeIndexes.indexes[0]) NodeT();

      // Connect them into a free list.
      for (int64_t i = 1; i < ArenaSize; ++i) {
        auto newNode =
            new (this->data + initFreeNodeIndexes.indexes[i]) NodeT();
        curNode->prev = newNode;
        newNode->next = curNode;
        curNode = newNode;
      }

      this->freeList = curNode;
      this->usedNodes = 0;
    }
  };

  AffinityAllocatorArena *arenas = nullptr;

  NodeT *allocFromArena() {
    if (!this->arenas || this->arenas->usedNodes == ArenaSize) {
      this->allocArena();
    }
    auto newNode = this->arenas->freeList;
    this->arenas->freeList = newNode->next;
    newNode->next = nullptr;
    if (this->arenas->freeList) {
      this->arenas->freeList->prev = nullptr;
    }
    this->arenas->usedNodes++;
    return newNode;
  }

  void allocArena() {

    auto arenaRaw = alignedAllocAndTouch<AffinityAllocatorArena>(1);

    auto arena = new (arenaRaw) AffinityAllocatorArena();

    // Connect arenas together.
    arena->next = this->arenas;
    if (this->arenas) {
      this->arenas->prev = arena;
    }
    this->arenas = arena;
  }

  void deallocArenas() {
    while (this->arenas) {
      auto arena = this->arenas;
      this->arenas = this->arenas->next;
      free(arena);
    }
  }

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

  struct InitFreeNodeIndexes {
    std::array<int, ArenaSize> indexes;
    InitFreeNodeIndexes() {
      for (int i = 0; i < ArenaSize; ++i) {
        indexes[i] = ArenaSize - i - 1;
      }
#ifdef RANDOMIZE_ADJ_LIST
      for (int i = 0; i + 1 < ArenaSize; ++i) {
        // Randomize the free list.
        long long j = (rand() % (ArenaSize - i)) + i;
        auto tmp = indexes[i];
        indexes[i] = indexes[j];
        indexes[j] = tmp;
      }
#endif
    }
  };

  static InitFreeNodeIndexes initFreeNodeIndexes;
};

template <class NodeT, int ArenaSize>
typename AffinityAllocator<NodeT, ArenaSize>::InitFreeNodeIndexes
    AffinityAllocator<NodeT, ArenaSize>::initFreeNodeIndexes;

template <class NodeT, int ArenaSize> class MultiThreadAffinityAllocator {

public:
  using Allocator = AffinityAllocator<NodeT, ArenaSize>;

  MultiThreadAffinityAllocator(int threads) {
    this->allocators.reserve(threads);
    for (int i = 0; i < threads; ++i) {
      this->allocators.emplace_back();
    }
  }

  NodeT *alloc(int tid) { return this->allocators.at(tid).alloc(); }

  std::vector<Allocator> allocators;
};

#endif