#ifndef BFS_KERNELS_H
#define BFS_KERNELS_H

#include "benchmark.h"
#include "sized_array.h"
#include "sliding_queue.h"
#include "spatial_queue.h"

#ifdef USE_EDGE_INDEX_OFFSET
#define EdgeIndexT NodeID
#else
#define EdgeIndexT NodeID *
#endif // USE_EDGE_INDEX_OFFSET

#ifndef OMP_SCHEDULE
#define OMP_SCHEDULE static
#endif

const NodeID InitParentId = -1;

__attribute__((noinline)) void bfsPush(

#ifdef USE_ADJ_LIST
    AdjGraph &graph,
#else
    const Graph &g,
#endif

    pvector<NodeID> &parent,

#ifdef USE_SPATIAL_QUEUE
    SpatialQueue<NodeID> &squeue,
#endif

#ifdef USE_SPATIAL_FRONTIER
    SpatialQueue<NodeID> &sfrontier
#else
    SlidingQueue<NodeID> &queue
#endif

) {

  /**************************************************************************
   * Define the private object list.
   **************************************************************************/

#define PushPrivObj _Pragma("push_macro(\"PRIV_OBJ_LIST\")") // for convenience
#define PopPrivObj _Pragma("pop_macro(\"PRIV_OBJ_LIST\")")

  /**************************************************************************
   * Initialization: Get all the variables.
   **************************************************************************/
  auto parent_v = parent.data();
#define PRIV_OBJ_LIST parent_v

#ifdef USE_SPATIAL_QUEUE
  const auto squeue_data = squeue.data;
  const auto squeue_meta = squeue.meta;
  const auto squeue_capacity = squeue.queue_capacity;
  const auto squeue_hash_shift = squeue.hash_shift;
  const auto squeue_hash_mask = squeue.hash_mask;

  PushPrivObj
#undef PRIV_OBJ_LIST
#define PRIV_OBJ_LIST                                                          \
  PopPrivObj PRIV_OBJ_LIST, squeue_data, squeue_meta, squeue_capacity,         \
      squeue_hash_shift, squeue_hash_mask
      ;
#endif

#ifdef USE_SPATIAL_FRONTIER
  const auto frontier_v = sfrontier.data;
  const auto frontier_meta = sfrontier.meta;
  const auto frontier_cap = sfrontier.queue_capacity;
  const auto frontier_n = sfrontier.num_queues;

  PushPrivObj
#undef PRIV_OBJ_LIST
#define PRIV_OBJ_LIST                                                          \
  PopPrivObj PRIV_OBJ_LIST, frontier_v, frontier_meta, frontier_cap, frontier_n
      ;

#else
  auto queue_v = queue.begin();
  auto queue_e = queue.end();
  int64_t queue_size = queue_e - queue_v;

  PushPrivObj
#undef PRIV_OBJ_LIST
#define PRIV_OBJ_LIST PopPrivObj PRIV_OBJ_LIST, queue_v, queue_e, queue_size
      ;
#endif

#ifdef USE_ADJ_LIST
  auto adj_list = graph.adjList;

  PushPrivObj
#undef PRIV_OBJ_LIST
#define PRIV_OBJ_LIST PopPrivObj PRIV_OBJ_LIST, adj_list
      ;
#else
  NodeID *out_edges = g.out_edges();
#ifdef USE_EDGE_INDEX_OFFSET
  EdgeIndexT *out_neigh_index = g.out_neigh_index_offset();
#else
  EdgeIndexT *out_neigh_index = g.out_neigh_index();
#endif

  PushPrivObj
#undef PRIV_OBJ_LIST
#define PRIV_OBJ_LIST PopPrivObj PRIV_OBJ_LIST, out_edges, out_neigh_index
      ;
#endif

  /**
   * Helper function for debug perpose.
   */
  // {
  //   static int iter = 0;
  //   uint64_t totalIdx = 0;
  //   for (auto iter = queue_v; iter < queue_e; iter++) {
  //     NodeID u = *iter;
  //     totalIdx += u;
  //   }
  //   printf(" - TD %d Size %ld Total %lu.\n", iter, queue_e - queue_v,
  //   totalIdx); iter++;
  // }

  /**************************************************************************
   * Start the OpenMP Parallel Loop.
   **************************************************************************/

#pragma omp parallel firstprivate(PRIV_OBJ_LIST)
  {

#ifndef USE_SPATIAL_QUEUE
    SizedArray<NodeID> lqueue(g.num_nodes());
#endif

    /**************************************************************************
     * Outer loop to iterate through frontier.
     **************************************************************************/
#ifdef USE_SPATIAL_FRONTIER
#pragma omp for schedule(OMP_SCHEDULE)
    for (int64_t f = 0; f < frontier_n; ++f) {

      const int64_t frontier_size = frontier_meta[f].size[0];
      frontier_meta[f].size[0] = 0;

#pragma clang loop unroll(disable) vectorize(disable) interleave(disable)
      for (int64_t i = 0; i < frontier_size; ++i) {

#pragma ss stream_name "gap.bfs_push.u.ld"
        NodeID u = frontier_v[f * frontier_cap + i];

#else
#pragma omp for schedule(OMP_SCHEDULE)
    for (int64_t i = 0; i < queue_size; ++i) {

#pragma ss stream_name "gap.bfs_push.u.ld"
      NodeID u = queue_v[i];
#endif // USE_SPATIAL_FRONTIER

        /**************************************************************************
         * Get the out edge list. Either AdjList or CSR edge list.
         **************************************************************************/

#ifdef USE_ADJ_LIST

#pragma ss stream_name "gap.bfs_push.adj.node.ld"
        auto *cur_node = adj_list[u];

#pragma clang loop unroll(disable) vectorize(disable) interleave(disable)
        while (cur_node) {

#pragma ss stream_name "gap.bfs_push.adj.n_edges.ld"
          const auto numEdges = cur_node->numEdges;

#pragma clang loop unroll(disable) vectorize(disable) interleave(disable)
          for (int64_t j = 0; j < numEdges; ++j) {

#pragma ss stream_name "gap.bfs_push.out_v.ld"
            NodeID v = cur_node->edges[j];

#else

      // Explicit write this out to aviod u + 1.
      EdgeIndexT *out_neigh_ptr = out_neigh_index + u;

#pragma ss stream_name "gap.bfs_push.out_begin.ld"
      EdgeIndexT out_begin = out_neigh_ptr[0];

#pragma ss stream_name "gap.bfs_push.out_end.ld"
      EdgeIndexT out_end = out_neigh_ptr[1];

#ifdef USE_EDGE_INDEX_OFFSET
      NodeID *out_ptr = out_edges + out_begin;
#else
      NodeID *out_ptr = out_begin;
#endif

      const auto out_degree = out_end - out_begin;

      for (int64_t j = 0; j < out_degree; ++j) {

#pragma ss stream_name "gap.bfs_push.out_v.ld"
        NodeID v = out_ptr[j];

#endif

            /**************************************************************************
             * Perform atomic swap.
             **************************************************************************/

            NodeID temp = InitParentId;

#pragma ss stream_name "gap.bfs_push.swap.at"
            bool swapped = __atomic_compare_exchange_n(
                parent_v + v, &temp, u, false /* weak */, __ATOMIC_RELAXED,
                __ATOMIC_RELAXED);
            if (swapped) {

              /**************************************************************************
               * Push into local or spatial queue.
               **************************************************************************/

#ifdef USE_SPATIAL_QUEUE

              /**
               * Hash into the spatial queue.
               */
              auto queue_idx = (v >> squeue_hash_shift) & squeue_hash_mask;

#pragma ss stream_name "gap.bfs_push.enque.at"
              auto queue_loc = __atomic_fetch_add(
                  &squeue_meta[queue_idx].size[0], 1, __ATOMIC_RELAXED);

#pragma ss stream_name "gap.bfs_push.enque.st"
              squeue_data[queue_idx * squeue_capacity + queue_loc] = v;

#else
          lqueue.buffer[lqueue.num_elements++] = v;
#endif
            }
          }

          /**************************************************************************
           * Move to next node if AdjList.
           **************************************************************************/

#ifdef USE_ADJ_LIST
#pragma ss stream_name "gap.bfs_push.adj.next_node.ld"
          auto next_node = cur_node->next;

          cur_node = next_node;
        }
#endif
      }

      // There is an implicit barrier after pragma for.
      // Flush into the global queue.

      /**************************************************************************
       * Post-processing:
       * 1. For spatial frontier, nothing to do.
       * 2. For spatial queue, copy to global queue.
       * 3. For local queue, copy to global queue.
       **************************************************************************/

#ifdef USE_SPATIAL_QUEUE

#ifdef USE_SPATIAL_FRONTIER
      // Nothing to do for spatial frontier.
    }
#else
#pragma omp for schedule(static)
      for (int queue_idx = 0; queue_idx < squeue.num_queues; ++queue_idx) {
        queue.append(squeue.data + queue_idx * squeue.queue_capacity,
                     squeue.size(queue_idx, 0));
        squeue.clear(queue_idx, 0);
      }
#endif
#else
    queue.append(lqueue.begin(), lqueue.size());
    lqueue.clear();
#endif
  }

  return;
}

// Sanity check that pull has no queue.
#if defined(USE_PULL) || !defined(USE_PUSH)
#ifdef USE_SPATIAL_QUEUE
#error "BFS pull no spatial queue."
#endif
#ifdef USE_SPATIAL_FRONTIER
#error "BFS pull no spatial queue."
#endif
#endif

__attribute__((noinline)) int64_t bfsPullCSR(const Graph &g, NodeID *parent,
                                             NodeID *next_parent) {

  NodeID *in_edges = g.in_edges();
#ifdef USE_EDGE_INDEX_OFFSET
  EdgeIndexT *in_neigh_index = g.in_neigh_index_offset();
#else
  EdgeIndexT *in_neigh_index = g.in_neigh_index();
#endif

  int64_t awake_count = 0;

#pragma omp parallel for schedule(static) reduction(+ : awake_count) \
  firstprivate(in_edges, in_neigh_index, parent, next_parent)
  for (NodeID u = 0; u < g.num_nodes(); u++) {

#pragma ss stream_name "gap.bfs_pull.parent.ld"
    NodeID p = parent[u];

    // Explicit write this to avoid u + 1 and misplaced stream_step.
    auto *in_neigh_ptr = in_neigh_index + u;

#pragma ss stream_name "gap.bfs_pull.in_begin.ld"
    auto in_begin = in_neigh_ptr[0];

#pragma ss stream_name "gap.bfs_pull.in_end.ld"
    auto in_end = in_neigh_ptr[1];

    int64_t in_degree = in_end - in_begin;

#ifdef USE_EDGE_INDEX_OFFSET
    NodeID *in_ptr = in_edges + in_begin;
#else
    NodeID *in_ptr = in_begin;
#endif

    // This helps nest the inner loop.
    auto needProcess = p < 0 && in_degree > 0;
    if (needProcess) {

      // Better to reduce from zero.
      NodeID np = 0;
      int64_t i = 0;
      do {

#pragma ss stream_name "gap.bfs_pull.v.ld"
        NodeID v = in_ptr[i];

#pragma ss stream_name "gap.bfs_pull.v_parent.ld"
        NodeID v_parent = parent[v];

        np = (v_parent > -1) ? (v + 1) : np;

        i++;
      } while (i != in_degree);

      if (np != 0) {
        next_parent[u] = np - 1;
        awake_count++;
      }
    }
  }

  return awake_count;
}

__attribute__((noinline)) void bfsPullUpdate(NodeID num_nodes, NodeID *parent,
                                             NodeID *next_parent) {

// Copy next_parent into parent.
#pragma omp parallel for schedule(static) firstprivate(parent, next_parent)
  for (NodeID u = 0; u < num_nodes; u++) {
    parent[u] = next_parent[u];
  }
}

#endif