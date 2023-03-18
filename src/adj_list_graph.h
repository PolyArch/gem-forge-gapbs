#ifndef ADJ_LIST_GRAPH_H
#define ADJ_LIST_GRAPH_H

#ifdef USE_AFFINITY_ALLOCATOR
#include "affinity_allocator.hh"
#elif defined(USE_AFFINITY_ALLOC)
#include "affinity_alloc.h"
#endif

#include "util.h"

#include "omp.h"

#ifdef GEM_FORGE
#include "gem5/m5ops.h"
#endif

template <class NodeID_, class DestID_ = NodeID_, bool MakeInverse_ = true,
          int EdgeOffset_ = 0, int EdgeSize_ = sizeof(DestID_)>
class AdjListGraph {
public:
  constexpr static int EdgesPerNode = sizeof(DestID_) == 8 ? 5 : 10;

  struct AdjListNode {
    int64_t numEdges = 0;
    AdjListNode *next = nullptr;
    AdjListNode *prev = nullptr;
    DestID_ edges[EdgesPerNode];
  };

  const int64_t N;
  AdjListNode **adjList = nullptr;
  int64_t *degrees = nullptr;

#ifdef USE_AFFINITY_ALLOCATOR
  constexpr static int ArenaSize = 8192;
  using AllocatorT = MultiThreadAffinityAllocator<AdjListNode, ArenaSize>;
  AllocatorT *allocator = nullptr;
#endif

  AdjListGraph(int threads, int64_t _N, NodeID_ *offsets, DestID_ *edges,
               void *properties)
      : N(_N) {

#ifdef USE_AFFINITY_ALLOCATOR
    this->allocator = new AllocatorT(threads);
#endif

    this->adjList = alignedAllocAndTouch<AdjListNode *>(_N);
    this->degrees = alignedAllocAndTouch<int64_t>(_N);

    auto allocNodes = std::vector<int>(threads, 0);

    auto adjList = this->adjList;
    auto degrees = this->degrees;

#pragma omp parallel firstprivate(_N, offsets, edges, adjList, degrees)
    {
      auto tid = omp_get_thread_num();

#pragma omp for schedule(static)
      for (int64_t i = 0; i < _N; ++i) {
        auto lhs = offsets[i];
        auto rhs = offsets[i + 1];
        degrees[i] = rhs - lhs;
        if (lhs == rhs) {
          adjList[i] = nullptr;
          continue;
        }

        adjList[i] = this->allocNode(tid);
        allocNodes[tid]++;

        auto *curNode = adjList[i];
        for (int64_t j = lhs; j < rhs; ++j) {
          if (curNode->numEdges == EdgesPerNode) {
            // Allocate a new node.
            auto *newNode = this->allocNode(tid);
            allocNodes[tid]++;
            curNode->next = newNode;
            newNode->prev = curNode;
            curNode = newNode;
          }
          // Push the edge.
          curNode->edges[curNode->numEdges] = edges[j];
          curNode->numEdges++;
        }
      }
    }

    int totalNodes = 0;
    for (auto node : allocNodes) {
      totalNodes += node;
    }
    printf("Allocated %d AdjList Nodes.\n", totalNodes);

/*****************************************************************
 * Register the region to SNUCA.
 *****************************************************************/
#ifdef GEM_FORGE
    {
      m5_stream_nuca_region("gap.adj.ptr", adjList, sizeof(*adjList), N, 0, 0);
      m5_stream_nuca_region("gap.adj.degree", degrees, sizeof(*degrees), N, 0,
                            0);
      m5_stream_nuca_align(adjList, properties, 0);
      m5_stream_nuca_align(degrees, properties, 0);
#ifdef USE_AFFINITY_ALLOCATOR
      auto dummyNode = reinterpret_cast<AdjListNode *>(0);
      int arenaIdx = 0;
      for (const auto &threadAlloc : this->allocator->allocators) {
        auto arena = threadAlloc.arenas;
        while (arena) {
          auto regionName = "gap.pr_push.adj/" + std::to_string(arenaIdx);
          m5_stream_nuca_region(regionName.c_str(), arena, sizeof(AdjListNode),
                                ArenaSize, 0, 0);
          m5_stream_nuca_align(arena, properties,
                               m5_stream_nuca_encode_multi_ind(
                                   offsetof(AdjListNode, numEdges),
                                   sizeof(dummyNode->numEdges),
                                   offsetof(AdjListNode, edges) + EdgeOffset_,
                                   EdgeSize_, sizeof(*dummyNode->edges)));

          arena = arena->next;
          arenaIdx++;
        }
      }
#endif
    }
#endif
  }

  ~AdjListGraph() {
#ifdef USE_AFFINITY_ALLOCATOR
    delete allocator;
#else
    for (int i = 0; i < N; ++i) {
      // Release all nodes.
      auto node = adjList[i];
      while (node) {
        auto next = node->next;
#ifdef USE_AFFINITY_ALLOC
        free_aff(node);
#else
        free(node);
#endif
        node = next;
      }
    }
#endif
    free(adjList);
    free(degrees);
  }

  AdjListNode *allocNode(int tid) {
#ifdef USE_AFFINITY_ALLOCATOR
    return this->allocator->alloc(tid);
#else
// In-place new.
#ifdef USE_AFFINITY_ALLOC
    auto *n = malloc_aff(sizeof(AdjListNode));
#else
    auto *n = malloc(sizeof(AdjListNode));
#endif
    auto *node = new (n) AdjListNode();
    return node;
#endif
  }

  void warmAdjList() {
    // Warm up the adjacent list.
    printf("Start warming AdjList.\n");
    auto adj_list = this->adjList;
#pragma omp parallel for schedule(static) firstprivate(adj_list)
    for (int64_t i = 0; i < this->N; i++) {

      int64_t n = i;

      auto *cur_node = adj_list[n];

#pragma clang loop unroll(disable) vectorize(disable) interleave(disable)
      while (cur_node) {
        volatile auto next_node = cur_node->next;
        cur_node = next_node;
      }
    }
    printf("Warmed AdjList.\n");
  }
};

#endif