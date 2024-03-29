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

using NumEdgeInNodeT = int64_t;

#ifndef ADJ_LIST_GRAPH_MIX_CSR_THRESHOLD
#define ADJ_LIST_GRAPH_MIX_CSR_THRESHOLD 2
#endif

template <
    // Type of edge.
    typename EdgeT,
    // Number of edges.
    int NumEdge,
    // Whether this is bidirection.
    bool Bidirection,
    // Whether this has edge num.
    bool HasEdgeNum,
    // Whether this stores the offset instead of pull ptr.
    bool UseOffsetPtr>
struct __attribute__((packed)) AdjListNodeT {
  using NodeT =
      AdjListNodeT<EdgeT, NumEdge, Bidirection, HasEdgeNum, UseOffsetPtr>;
  NumEdgeInNodeT numEdges = 0;
  NodeT *next = nullptr;
  NodeT *prev = nullptr;
  EdgeT edges[NumEdge];
  void setPrev(NodeT *prevNode) { this->prev = prevNode; }
  void setEdgeNum(NumEdgeInNodeT numEdges) { this->numEdges = numEdges; }
};

template <typename EdgeT, int NumEdge>
struct __attribute__((packed))
AdjListNodeT<EdgeT, NumEdge, false, true, false> {
  using NodeT = AdjListNodeT<EdgeT, NumEdge, false, true, false>;
  NumEdgeInNodeT numEdges = 0;
  NodeT *next = nullptr;
  EdgeT edges[NumEdge];
  void setPrev(NodeT *prevNode) {}
  void setEdgeNum(NumEdgeInNodeT numEdges) { this->numEdges = numEdges; }
};

template <typename EdgeT, int NumEdge>
struct __attribute__((packed)) AdjListNodeT<EdgeT, NumEdge, false, true, true> {
  using NodeT = AdjListNodeT<EdgeT, NumEdge, false, true, true>;
  NumEdgeInNodeT numEdges = 0;
  int32_t nextOffset = 0;
  EdgeT edges[NumEdge];
  void setPrev(NodeT *prevNode) {}
  void setEdgeNum(NumEdgeInNodeT numEdges) { this->numEdges = numEdges; }
};

template <typename EdgeT, int NumEdge>
struct __attribute__((packed))
AdjListNodeT<EdgeT, NumEdge, false, false, false> {
  using NodeT = AdjListNodeT<EdgeT, NumEdge, false, false, false>;
  NodeT *next = nullptr;
  EdgeT edges[NumEdge];
  void setPrev(NodeT *prevNode) {}
  void setEdgeNum(NumEdgeInNodeT numEdges) {}
};

enum AdjListTypeE {
  /**
   * This is the single data structure: one link list per node.
   * Each node could remember number of edges or terminated by special value.
   * This is the dynamic data structure.
   */
  OneListPerNode = 0,
  /**
   * This is the static data structure. The entire edge list is broken into
   * a single link list. The vertex remembers a pointer into the middle of
   * the list. You need some pointer arithmetic to chase the pointer.
   */
  SingleListPerGraph = 1,
  /**
   * Similar to OneListPerNode, but use offset ptr.
   */
  OneListPerNodeWithOffsetPtr = 2,
  /**
   * Mixed CSR and AdjList.
   */
  OneListPerNodeMixCSR = 3,
};

template <
    // Type of the node id.
    class NodeID_,
    // Type of the edge.
    class DestID_ = NodeID_,
    // The edge value offset within the edge type.
    int EdgeOffset_ = 0,
    // The size of the edge.
    int EdgeSize_ = sizeof(DestID_),
    // The size of the adj list node.
    int AdjNodeSize = 64,
    // Whether we have prev pointer.
    bool HasPrevPtr = true,
    // Wether the node has number of edges.
    bool HasEdgeNum = true,
    // Wheter partion by vertex or by edge.
    AdjListTypeE ListType = AdjListTypeE::OneListPerNode>
class AdjListGraph {
public:
  using EdgeT = DestID_;
  constexpr static AdjListTypeE ListT = ListType;

  constexpr static bool IsOneListPerNode =
      ListType == AdjListTypeE::OneListPerNode ||
      ListType == AdjListTypeE::OneListPerNodeWithOffsetPtr ||
      ListType == AdjListTypeE::OneListPerNodeMixCSR;

  constexpr static bool IsSingleList =
      ListType == AdjListTypeE::SingleListPerGraph;

  constexpr static bool UseOffsetPtr =
      ListType == AdjListTypeE::OneListPerNodeWithOffsetPtr;

  constexpr static int NodePtrSize =
      UseOffsetPtr ? sizeof(int32_t) : sizeof(void *);

  constexpr static int MetaDataSize =
      NodePtrSize                                 // NextPtr
      + (HasEdgeNum ? sizeof(NumEdgeInNodeT) : 0) // NumEdges
      + (HasPrevPtr ? NodePtrSize : 0)            // PrevPtr
      ;
  constexpr static int EdgesPerNode =
      (AdjNodeSize - MetaDataSize) / sizeof(DestID_);

  using AdjListNode =
      AdjListNodeT<DestID_, EdgesPerNode, HasPrevPtr, HasEdgeNum, UseOffsetPtr>;

  constexpr static uint64_t AdjListNodeOffsetMask =
      (static_cast<uint64_t>(AdjNodeSize) - 1);
  constexpr static uint64_t AdjListNodePtrMask = ~AdjListNodeOffsetMask;

  static AdjListNode *getNodePtr(DestID_ *ptr) {
    return reinterpret_cast<AdjListNode *>(reinterpret_cast<uint64_t>(ptr) &
                                           AdjListNodePtrMask);
  }
  static uint64_t getNodeOffset(DestID_ *ptr) {
    return (reinterpret_cast<uint64_t>(ptr) & AdjListNodeOffsetMask) -
           offsetof(AdjListNode, edges);
  }

  /**
   * If the degree of a node is beyond this threshold, we use AdjList.
   */
  constexpr static int MixCSRThreshold =
      ADJ_LIST_GRAPH_MIX_CSR_THRESHOLD * EdgesPerNode;

  const int64_t N;
  AdjListNode **adjList = nullptr;
  AdjListNode **adjListEnd = nullptr;
  int64_t *degrees = nullptr;

#ifdef USE_AFFINITY_ALLOCATOR
  constexpr static int ArenaSize = 8192;
  using AllocatorT = MultiThreadAffinityAllocator<AdjListNode, ArenaSize>;
  AllocatorT *allocator = nullptr;
#endif

  template <typename P>
  AdjListGraph(int threads, int64_t _N, NodeID_ *offsets, DestID_ *edges,
               const P *properties)
      : N(_N) {

#ifdef USE_AFFINITY_ALLOCATOR
    this->allocator = new AllocatorT(threads);
#endif

#ifdef GEM_FORGE
    this->adjList = alignedAllocAndTouch<AdjListNode *>(_N + 1);
    this->degrees = alignedAllocAndTouch<int64_t>(_N + 1);
#else
    this->adjList = reinterpret_cast<AdjListNode **>(
        malloc(sizeof(AdjListNode *) * (_N + 1)));
    this->degrees =
        reinterpret_cast<int64_t *>(malloc(sizeof(int64_t) * (_N + 1)));
#endif

    uint64_t totalNodes;
    if (IsOneListPerNode) {
      totalNodes = this->allocateByVertex(threads, offsets, edges, properties);
    } else if (IsSingleList) {
      totalNodes = this->allocateByEdges(threads, offsets, edges, properties);
    }
    auto totalEdges = offsets[_N];
    auto ratio = static_cast<float>(totalNodes * sizeof(AdjListNode)) /
                 (totalEdges * sizeof(DestID_));
    printf("Allocated %lu AdjList Nodes Overhead %7.2f.\n", totalNodes,
           ratio * 100.f);

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
      m5_stream_nuca_remap();
    }
#endif
  }

  void releaseAdjListPerNode() {
    // Default one AdjList per node.
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
  }

  void releaseSingleAdjList() {
    // Release the single edge list.
    if (N == 0) {
      return;
    }
    auto firstNode = adjList[0];
    auto node = reinterpret_cast<AdjListNode *>(
        reinterpret_cast<uint64_t>(firstNode) & AdjListNodePtrMask);
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

  ~AdjListGraph() {
#ifdef USE_AFFINITY_ALLOCATOR
    delete allocator;
#else
    if (ListType == AdjListTypeE::OneListPerNode) {
      releaseAdjListPerNode();
    } else if (ListType == AdjListTypeE::SingleListPerGraph) {
      releaseSingleAdjList();
    }
#endif
    free(adjList);
    free(adjListEnd);
    free(degrees);
  }

  AdjListNode *allocNode(int tid, int edges, const void **addrs) {
#ifdef USE_AFFINITY_ALLOCATOR
    return this->allocator->alloc(tid);
#else
// In-place new.
#ifdef USE_AFFINITY_ALLOC
    auto *n = malloc_aff(sizeof(AdjListNode), edges, addrs);
#else
    auto *n = malloc(sizeof(AdjListNode));
#endif
    auto *node = new (n) AdjListNode();
    return node;
#endif
  }

  __attribute__((noinline)) void warmAdjList() {
#ifdef GEM_FORGE
    gf_warm_array("degrees", degrees, N * sizeof(degrees[0]));
    gf_warm_array("adj_list", adjList, N * sizeof(adjList[0]));
#endif
    if (ListType == AdjListTypeE::SingleListPerGraph) {
      this->warmSingleAdjList();
    } else if (this->IsOneListPerNode) {
      this->warmOneAdjListPerNode();
    }
  }

  __attribute__((noinline)) int64_t warmOneAdjListPerNode() {
    // Warm up the adjacent list.
    printf("Start warming AdjList.\n");
    auto adj_list = this->adjList;
    auto degrees = this->degrees;
    int64_t edges = 0;
#pragma omp parallel for schedule(static) firstprivate(adj_list, degrees)      \
    reduction(+ : edges)
    for (int64_t i = 0; i < this->N; i++) {

      auto degree = degrees[i];
      auto *cur_node = adj_list[i];

      auto sum = 0;

      if (ListType == AdjListTypeE::OneListPerNodeMixCSR &&
          degree < MixCSRThreshold) {
        // This is just normal CSR edges.
        EdgeT *edges = reinterpret_cast<EdgeT *>(cur_node);
#pragma clang loop unroll(disable) vectorize(disable) interleave(disable)
        for (int i = 0; i < degree; i += 64 / sizeof(EdgeT)) {
          NodeID_ out_v = edges[i];
          sum += out_v;
        }
      } else {
        // This is adj list.
#pragma clang loop unroll(disable) vectorize(disable) interleave(disable)
        while (cur_node) {
          auto next_node = cur_node->next;
          sum += reinterpret_cast<uint64_t>(next_node);
          cur_node = next_node;
        }
      }

      edges += sum;
    }
    printf("Warmed AdjList.\n");
    return edges;
  }

  __attribute__((noinline)) int64_t warmSingleAdjList() {
    // Warm up the adjacent list.
    printf("Start warming AdjList.\n");
    int64_t edges = 0;
    auto adj_list = this->adjList;
    if (this->N > 0) {
      auto ptr = reinterpret_cast<DestID_ *>(adj_list[0]);
      auto node = getNodePtr(ptr);
      while (node) {
        auto next_node = node->next;
        edges += reinterpret_cast<int64_t>(next_node);
        node = next_node;
      }
    }
    printf("Warmed AdjList.\n");
    return edges;
  }

  uint64_t estimateMixCSRHops(NodeID_ *offsets, DestID_ *edges) const {

    uint64_t totalCSRHops = 0;

    /**
     * Assume the CSR edges and nodes are evenly distributed across banks.
     */
    const int rows = 8;
    const int cols = 8;
    const int banks = rows * cols;

    const auto numNodes = this->N;
    const auto numEdges = offsets[this->N];

    const auto nodeIntrlv = numNodes / banks;
    const auto edgeIntrlv = numEdges / banks;

    auto getBank = [&banks](uint64_t idx, uint64_t intrlv) -> int {
      return (idx / intrlv) % banks;
    };

    auto getHops = [&rows, &cols](int bankA, int bankB) -> int {
      auto rowA = bankA / cols;
      auto rowB = bankB / cols;
      auto colA = bankA % cols;
      auto colB = bankB % cols;
      return std::abs(rowA - rowB) + std::abs(colA - colB);
    };

    if (ListType == AdjListTypeE::OneListPerNodeMixCSR) {

      for (int64_t i = 0; i < N; ++i) {
        auto lhs = offsets[i];
        auto rhs = offsets[i + 1];
        auto degree = rhs - lhs;

        if (degree >= MixCSRThreshold) {
          // This node is broken into AdjList.
          continue;
        }

        // Estimate the CSR hops.
        for (int j = 0; j < degree; ++j) {
          auto edgeBank = getBank(lhs + j, edgeIntrlv);
          auto v = edges[lhs + j];
          auto nodeBank = getBank(v, nodeIntrlv);
          auto hops = getHops(edgeBank, nodeBank);
          totalCSRHops += hops;
        }
      }
    }

    return totalCSRHops;
  }

  template <typename P>
  uint64_t allocateByVertex(int threads, NodeID_ *offsets, DestID_ *edges,
                            const P *properties) {

    auto allocNodes = std::vector<int>(threads, 0);

    auto N = this->N;
    auto adjList = this->adjList;
    auto degrees = this->degrees;

#pragma omp parallel firstprivate(N, offsets, edges, adjList, degrees)
    {
      auto tid = omp_get_thread_num();

      const void *addrs[EdgesPerNode];

#pragma omp for schedule(static)
      for (int64_t i = 0; i < N; ++i) {
        auto lhs = offsets[i];
        auto rhs = offsets[i + 1];
        auto degree = rhs - lhs;
        degrees[i] = degree;
        if (lhs == rhs) {
          adjList[i] = nullptr;
          continue;
        }

        if (ListType == AdjListTypeE::OneListPerNodeMixCSR &&
            degree < MixCSRThreshold) {
          // Use CSR for this node.
          adjList[i] = reinterpret_cast<AdjListNode *>(edges + lhs);
          continue;
        }

        AdjListNode *curNode = nullptr;

#ifdef GEM_FORGE
#pragma clang loop unroll(disable) vectorize(disable) interleave(disable)
#endif
        for (int64_t j = lhs; j < rhs; j += EdgesPerNode) {

          auto numEdges = std::min(rhs - j, static_cast<int64_t>(EdgesPerNode));

#ifdef GEM_FORGE
#pragma clang loop unroll(disable) vectorize(disable) interleave(disable)
#endif
          for (int64_t k = 0; k < numEdges; ++k) {
            // Calculate the address
            addrs[k] = properties + edges[j + k];
          }

          // Allocate a new node.
          auto *newNode = this->allocNode(tid, numEdges, addrs);
          allocNodes[tid]++;
          newNode->setPrev(curNode);
          if (curNode == nullptr) {
            adjList[i] = newNode;
          } else {
            curNode->next = newNode;
          }
          curNode = newNode;

          curNode->setEdgeNum(numEdges);

#ifdef GEM_FORGE
#pragma clang loop unroll(disable) vectorize(disable) interleave(disable)
#endif
          for (int64_t k = 0; k < numEdges; ++k) {
            // Push the edge.
            curNode->edges[k] = edges[j + k];
          }
        }
      }
    }

    int totalNodes = 0;
    for (auto node : allocNodes) {
      totalNodes += node;
    }
    return totalNodes;
  }

  template <typename P>
  uint64_t allocateByEdges(int threads, NodeID_ *offsets, DestID_ *edges,
                           const P *properties) {

    auto allocNodes = std::vector<int>(threads, 0);

    auto N = this->N;
    auto adjList = this->adjList;
    auto degrees = this->degrees;
    auto total_edges = offsets[N];

    const void *addrs[EdgesPerNode];

    int tid = 0;

    AdjListNode *curNode = nullptr;

    NodeID_ curVertex = 0;
    NodeID_ curOffset = 0;

    for (int64_t i = 0; i <= total_edges; i += EdgesPerNode) {

      // Construct the addresses.
      auto numEdges =
          std::min(total_edges - i, static_cast<int64_t>(EdgesPerNode));

#ifdef GEM_FORGE
#pragma clang loop unroll(disable) vectorize(disable) interleave(disable)
#endif
      for (int64_t k = 0; k < numEdges; ++k) {
        addrs[k] = properties + edges[i + k];
      }

      // Allocate a new node.
      auto *newNode = this->allocNode(tid, numEdges, addrs);
      allocNodes[tid]++;
      newNode->setPrev(curNode);
      if (curNode) {
        curNode->next = newNode;
      }
      curNode = newNode;

      curNode->setEdgeNum(numEdges);
#ifdef GEM_FORGE
#pragma clang loop unroll(disable) vectorize(disable) interleave(disable)
#endif
      for (int64_t k = 0; k < numEdges; ++k) {
        // Push the edge.
        curNode->edges[k] = edges[i + k];
      }

// Advance the pointers of vertex.
#ifdef GEM_FORGE
#pragma clang loop unroll(disable) vectorize(disable) interleave(disable)
#endif
      while (curOffset < i + numEdges) {
        auto ptr = curNode->edges + (curOffset - i);
        auto nextOffset = offsets[curVertex + 1];
        adjList[curVertex] = reinterpret_cast<AdjListNode *>(ptr);
        degrees[curVertex] = nextOffset - curOffset;
        curVertex++;
        curOffset = nextOffset;
      }
    }

    // Set the last ptr for the rest nodes.
    assert(curVertex <= N);
    assert(curOffset == total_edges);
    auto lastPtr = curNode->edges + total_edges % EdgesPerNode;
#ifdef GEM_FORGE
#pragma clang loop unroll(disable) vectorize(disable) interleave(disable)
#endif
    for (auto i = curVertex; i <= N; ++i) {
      adjList[i] = reinterpret_cast<AdjListNode *>(lastPtr);
      degrees[i] = 0;
    }

    int totalNodes = 0;
    for (auto node : allocNodes) {
      totalNodes += node;
    }
    return totalNodes;
  }

  template <bool PositiveDegree, typename U, typename Op>
  static inline void iterate(U u, AdjListNode **adj_list, Op op) {

    auto adj_ptr = adj_list + u;

#pragma ss stream_name "gap.adj_iter.lhs.ld"
    auto *lhs = reinterpret_cast<NodeID_ *>(adj_ptr[0]);

#pragma ss stream_name "gap.adj_iter.rhs.ld"
    auto *rhs = reinterpret_cast<NodeID_ *>(adj_ptr[1]);

    auto *rhs_node = getNodePtr(rhs);
    auto rhs_offset = getNodeOffset(rhs);

    if (PositiveDegree || lhs != rhs) {

      while (true) {
        auto *lhs_node = getNodePtr(lhs);
        const auto local_rhs =
            lhs_node != rhs_node ? (lhs_node->edges + EdgesPerNode) : rhs;
        const auto numEdges = local_rhs - lhs;

        int64_t j = 0;

#pragma clang loop unroll(disable) vectorize(disable) interleave(disable)
        while (true) {

#pragma ss stream_name "gap.adj_iter.v.ld"
          auto v = lhs[j];

          op(v);

          j++;
          if (j == numEdges) {
            break;
          }
        }

#pragma ss stream_name "gap.adj_iter.next_node.ld"
        auto next_node = lhs_node->next;

        lhs = next_node->edges;

        /**
         * I need to write this weird loop break condition to to distinguish
         * lhs and next_lhs for LoopBound.
         * TODO: Fix this in the compiler.
         */
        bool rhs_zero = next_node == rhs_node && rhs_offset == 0;
        bool should_break = rhs_zero || (lhs_node == rhs_node);
        if (should_break) {
          break;
        }
      }
    }
  }
};

#endif