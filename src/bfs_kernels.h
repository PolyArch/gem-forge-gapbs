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

const NodeID InitDepth = -1;

__attribute__((noinline)) NodeID bfsPushCSR(

    const Graph &g,

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
  const auto squeue_hash_div = squeue.hash_div;
  const auto squeue_hash_mask = squeue.hash_mask;

  PushPrivObj
#undef PRIV_OBJ_LIST
#define PRIV_OBJ_LIST                                                          \
  PopPrivObj PRIV_OBJ_LIST, squeue_data, squeue_meta, squeue_capacity,         \
      squeue_hash_div, squeue_hash_mask
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

  __attribute__((unused)) auto num_nodes = g.num_nodes();
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

  /**************************************************************************
   * Start the OpenMP Parallel Loop.
   **************************************************************************/

  NodeID scout_count = 0;

#pragma omp parallel firstprivate(PRIV_OBJ_LIST)
  {

#ifndef USE_SPATIAL_QUEUE
    SizedArray<NodeID> lqueue(num_nodes);
#ifdef COMPUTE_SCOUT_COUNT
    NodeID local_scout_count = 0;
#endif
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

        auto op = [&](NodeID v) -> void {
          /**************************************************************************
           * Perform atomic swap.
           **************************************************************************/

          NodeID temp = InitDepth;

#ifdef CHECK_BEFORE_ATOMIC
          if (parent_v[v] == InitDepth) {
#endif

#pragma ss stream_name "gap.bfs_push.swap.at"
            bool swapped = __atomic_compare_exchange_n(
                parent_v + v, &temp, u, false /* weak */, __ATOMIC_RELAXED,
                __ATOMIC_RELAXED);
            if (swapped) {

              /**************************************************************************
               * Push into local or spatial queue.
               **************************************************************************/

#ifdef COMPUTE_SCOUT_COUNT
              // Write in this way to ensure that offsets are recognized by
              // LLVM.
              auto out_degree_ptr = out_neigh_index + v;
#pragma ss stream_name "gap.bfs_push.scout_lhs.ld"
              auto out_degree_lhs = out_degree_ptr[0];
#pragma ss stream_name "gap.bfs_push.scout_rhs.ld"
              auto out_degree_rhs = out_degree_ptr[1];
              auto out_degree = out_degree_rhs - out_degree_lhs;
#endif

#ifdef USE_SPATIAL_QUEUE

              /**
               * Hash into the spatial queue.
               */
              auto queue_idx = (v / squeue_hash_div) & squeue_hash_mask;

#pragma ss stream_name "gap.bfs_push.enque.at"
              auto queue_loc = __atomic_fetch_add(
                  &squeue_meta[queue_idx].size[0], 1, __ATOMIC_RELAXED);

#pragma ss stream_name "gap.bfs_push.enque.st"
              squeue_data[queue_idx * squeue_capacity + queue_loc] = v;

#ifdef COMPUTE_SCOUT_COUNT
#pragma ss stream_name "gap.bfs_push.scout.at"
              __atomic_fetch_add(&squeue_meta[queue_idx].weightedSize[0],
                                 out_degree, __ATOMIC_RELAXED);
#endif

#else
          lqueue.buffer[lqueue.num_elements++] = v;
#ifdef COMPUTE_SCOUT_COUNT
          local_scout_count += out_degree;
#endif
#endif
            }

#ifdef CHECK_BEFORE_ATOMIC
          }
#endif
        };

        csrIterate<false>(u, out_neigh_index, op);
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
#ifdef COMPUTE_SCOUT_COUNT
        __atomic_fetch_add(&scout_count, squeue_meta[queue_idx].weightedSize[0],
                           __ATOMIC_RELAXED);
#endif
        queue.append(squeue.data + queue_idx * squeue.queue_capacity,
                     squeue.size(queue_idx, 0));
        squeue.clear(queue_idx, 0);
      }
#endif
#else
#ifdef COMPUTE_SCOUT_COUNT
    __atomic_fetch_add(&scout_count, local_scout_count, __ATOMIC_RELAXED);
#endif
    queue.append(lqueue.begin(), lqueue.size());
    lqueue.clear();
#endif
  }

#ifdef USE_SPATIAL_FRONTIER
#ifdef COMPUTE_SCOUT_COUNT
  // Simpy summarize scout_count if we are using spatial frontier.
  scout_count = squeue.getAndClearTotalWeightedSize();
#endif
#endif

  return scout_count;
}

__attribute__((noinline)) void bfsPushAdjList(

    AdjGraph &graph,

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
  const auto squeue_hash_div = squeue.hash_div;
  const auto squeue_hash_mask = squeue.hash_mask;

  PushPrivObj
#undef PRIV_OBJ_LIST
#define PRIV_OBJ_LIST                                                          \
  PopPrivObj PRIV_OBJ_LIST, squeue_data, squeue_meta, squeue_capacity,         \
      squeue_hash_div, squeue_hash_mask
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

  __attribute__((unused)) auto num_nodes = graph.N;
  auto adj_list = graph.adjList;

  PushPrivObj
#undef PRIV_OBJ_LIST
#define PRIV_OBJ_LIST PopPrivObj PRIV_OBJ_LIST, adj_list
      ;

  /**************************************************************************
   * Start the OpenMP Parallel Loop.
   **************************************************************************/

#pragma omp parallel firstprivate(PRIV_OBJ_LIST)
  {

#ifndef USE_SPATIAL_QUEUE
    SizedArray<NodeID> lqueue(num_nodes);
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

#pragma ss stream_name "gap.bfs_push.adj.node.ld"
        auto *cur_node = adj_list[u];

#pragma clang loop unroll(disable) vectorize(disable) interleave(disable)
        while (cur_node) {

#pragma ss stream_name "gap.bfs_push.adj.n_edges.ld"
          const auto num_edges = cur_node->numEdges;

#pragma clang loop unroll(disable) vectorize(disable) interleave(disable)
          for (int64_t j = 0; j < num_edges; ++j) {

#pragma ss stream_name "gap.bfs_push.out_v.ld"
            NodeID v = cur_node->edges[j];

            /**************************************************************************
             * Perform atomic swap.
             **************************************************************************/

            NodeID temp = InitDepth;

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
              auto queue_idx = (v / squeue_hash_div) & squeue_hash_mask;

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

#pragma ss stream_name "gap.bfs_push.adj.next_node.ld"
          auto next_node = cur_node->next;

          cur_node = next_node;
        }
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

__attribute__((noinline)) NodeID bfsPushSingleAdjList(

    AdjGraphSingleAdjListT &graph,

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
  const auto squeue_hash_div = squeue.hash_div;
  const auto squeue_hash_mask = squeue.hash_mask;

  PushPrivObj
#undef PRIV_OBJ_LIST
#define PRIV_OBJ_LIST                                                          \
  PopPrivObj PRIV_OBJ_LIST, squeue_data, squeue_meta, squeue_capacity,         \
      squeue_hash_div, squeue_hash_mask
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

  __attribute__((unused)) auto num_nodes = graph.N;
  auto adj_list = graph.adjList;

  PushPrivObj
#undef PRIV_OBJ_LIST
#define PRIV_OBJ_LIST PopPrivObj PRIV_OBJ_LIST, adj_list
      ;

  /**************************************************************************
   * Start the OpenMP Parallel Loop.
   **************************************************************************/

  NodeID scout_count = 0;
#pragma omp parallel firstprivate(PRIV_OBJ_LIST)
  {

#ifndef USE_SPATIAL_QUEUE
    SizedArray<NodeID> lqueue(num_nodes);
#ifdef COMPUTE_SCOUT_COUNT
    NodeID local_scout_count = 0;
#endif
#endif

#ifdef COMPUTE_SCOUT_COUNT
    auto out_degrees = graph.degrees;
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

        auto op = [&](NodeID v) -> void {
          /**************************************************************************
           * Perform atomic swap.
           **************************************************************************/

          NodeID temp = InitDepth;

#pragma ss stream_name "gap.bfs_push.swap.at"
          bool swapped = __atomic_compare_exchange_n(
              parent_v + v, &temp, u, false /* weak */, __ATOMIC_RELAXED,
              __ATOMIC_RELAXED);
          if (swapped) {

            /**************************************************************************
             * Push into local or spatial queue.
             **************************************************************************/

#ifdef COMPUTE_SCOUT_COUNT
            auto out_degree = out_degrees[v];
#endif

#ifdef USE_SPATIAL_QUEUE

            /**
             * Hash into the spatial queue.
             */
            auto queue_idx = (v / squeue_hash_div) & squeue_hash_mask;

#pragma ss stream_name "gap.bfs_push.enque.at"
            auto queue_loc = __atomic_fetch_add(&squeue_meta[queue_idx].size[0],
                                                1, __ATOMIC_RELAXED);

#pragma ss stream_name "gap.bfs_push.enque.st"
            squeue_data[queue_idx * squeue_capacity + queue_loc] = v;

#ifdef COMPUTE_SCOUT_COUNT
#pragma ss stream_name "gap.bfs_push.scout.at"
            __atomic_fetch_add(&squeue_meta[queue_idx].weightedSize[0],
                               out_degree, __ATOMIC_RELAXED);
#endif

#else
          lqueue.buffer[lqueue.num_elements++] = v;
#ifdef COMPUTE_SCOUT_COUNT
          local_scout_count += out_degree;
#endif
#endif
          }
        };

        AdjGraphSingleAdjListT::iterate<false>(u, adj_list, op);
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
#ifdef COMPUTE_SCOUT_COUNT
            __atomic_fetch_add(&scout_count,
                               squeue_meta[queue_idx].weightedSize[0],
                               __ATOMIC_RELAXED);
#endif
            queue.append(squeue.data + queue_idx * squeue.queue_capacity,
                         squeue.size(queue_idx, 0));
            squeue.clear(queue_idx, 0);
          }
#endif
#else
#ifdef COMPUTE_SCOUT_COUNT
    __atomic_fetch_add(&scout_count, local_scout_count, __ATOMIC_RELAXED);
#endif
    queue.append(lqueue.begin(), lqueue.size());
    lqueue.clear();
#endif
  }

#ifdef USE_SPATIAL_FRONTIER
#ifdef COMPUTE_SCOUT_COUNT
  // Simpy summarize scout_count if we are using spatial frontier.
  scout_count = squeue.getAndClearTotalWeightedSize();
#endif
#endif

  return scout_count;
}

// Sanity check that pull has no queue.
#if defined(USE_PULL) && !defined(USE_PUSH)
#ifdef USE_SPATIAL_QUEUE
#error "BFS pull no spatial queue."
#endif
#ifdef USE_SPATIAL_FRONTIER
#error "BFS pull no spatial queue."
#endif
#endif

__attribute__((noinline)) int64_t bfsPullCSR(const Graph &g,
#ifdef SHUFFLE_NODES
                                             NodeID *nodes,
#endif
                                             NodeID *parent,
                                             NodeID *next_parent) {

  NodeID *in_edges = g.in_edges();
#ifdef USE_EDGE_INDEX_OFFSET
  EdgeIndexT *in_neigh_index = g.in_neigh_index_offset();
#else
  EdgeIndexT *in_neigh_index = g.in_neigh_index();
#endif

  int64_t awake_count = 0;

#ifdef SHUFFLE_NODES

#pragma omp parallel for schedule(static) reduction(+ : awake_count)           \
    firstprivate(nodes, in_edges, in_neigh_index, parent, next_parent)
  for (int64_t k = 0; k < g.num_nodes(); k++) {

    int64_t u = nodes[k];

#else
#pragma omp parallel for schedule(static) reduction(+ : awake_count)           \
    firstprivate(in_edges, in_neigh_index, parent, next_parent)
  for (int64_t k = 0; k < g.num_nodes(); k++) {

    int64_t u = k;

#endif

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
      while (true) {

#ifdef BFS_PULL_ANALYZE_MAX_TRIP_COUNT
#pragma ss stream_name "gap.bfs_pull.v.ld/analyze-max-trip-count"
#else
#pragma ss stream_name "gap.bfs_pull.v.ld"
#endif
        NodeID v = in_ptr[i];

#pragma ss stream_name "gap.bfs_pull.v_parent.ld"
        NodeID v_parent = parent[v];

        np = (v_parent > -1) ? (v + 1) : np;

        i++;

#ifdef NO_EARLY_BREAK
        bool broken = i == in_degree;
#else
        bool broken = i == in_degree || v_parent != InitDepth;
#endif
        if (broken) {
          break;
        }
      }

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

__attribute__((noinline)) void bfsPullToFrontier(NodeID num_nodes,
                                                 NodeID *parent,
                                                 NodeID *next_parent,

#ifdef USE_SPATIAL_FRONTIER
                                                 SpatialQueue<NodeID> &squeue
#else
                                                 SlidingQueue<NodeID> &queue
#endif
) {

#pragma omp parallel firstprivate(num_nodes, parent, next_parent)
  {

#ifdef USE_SPATIAL_FRONTIER
    const auto squeue_data = squeue.data;
    const auto squeue_meta = squeue.meta;
    const auto squeue_capacity = squeue.queue_capacity;
    const auto squeue_hash_div = squeue.hash_div;
    const auto squeue_hash_mask = squeue.hash_mask;
#else
    SizedArray<NodeID> lqueue(num_nodes);
#endif

#pragma omp for schedule(static)
    for (NodeID u = 0; u < num_nodes; u++) {

#pragma ss stream_name "gap.bfs_frontier.np.ld"
      auto np = next_parent[u];

#pragma ss stream_name "gap.bfs_frontier.p.ld/fake-addr-base=np.ld"
      auto p = parent[u];
      if (np != p) {

        // Push to the frontier.
#ifdef USE_SPATIAL_FRONTIER

        // Hash into the spatial queue.
        auto queue_idx = (u / squeue_hash_div) & squeue_hash_mask;

#pragma ss stream_name "gap.bfs_frontier.enque.at/fake-addr-base=np.ld"
        auto queue_loc = __atomic_fetch_add(&squeue_meta[queue_idx].size[0], 1,
                                            __ATOMIC_RELAXED);

#pragma ss stream_name "gap.bfs_frontier.enque.st"
        squeue_data[queue_idx * squeue_capacity + queue_loc] = u;

#else
        lqueue.buffer[lqueue.num_elements++] = u;
#endif
      }
    }

#ifdef USE_SPATIAL_FRONTIER
// Nothing to do for spatial frontier.
#else
    // Push to the global queue.
    queue.append(lqueue.begin(), lqueue.size());
    lqueue.clear();
#endif
  }
}

__attribute__((noinline)) int64_t
bfsPullAdjList(const AdjGraph &g, NodeID *parent, NodeID *next_parent) {

  auto num_nodes = g.N;
  auto adj_list = g.adjList;

  int64_t awake_count = 0;

#pragma omp parallel for schedule(static) reduction(+ : awake_count)           \
    firstprivate(adj_list, parent, next_parent)
  for (NodeID u = 0; u < num_nodes; u++) {

#pragma ss stream_name "gap.bfs_pull.parent.ld"
    NodeID p = parent[u];

#pragma ss stream_name "gap.bfs_pull.head.ld"
    auto *cur_node = adj_list[u];

    // This helps nest the inner loop.
    auto needProcess = p < 0 && cur_node != nullptr;
    if (needProcess) {

      // Better to reduce from zero.
      NodeID np = 0;
      while (true) {

#pragma ss stream_name "gap.bfs_pull.n_edges.ld"
        const auto num_edges = cur_node->numEdges;

        int64_t i = 0;
        NodeID local_np = 0;
        do {

#pragma ss stream_name "gap.bfs_pull.v.ld"
          NodeID v = cur_node->edges[i];

#pragma ss stream_name "gap.bfs_pull.v_parent.ld"
          NodeID v_parent = parent[v];

          local_np = (v_parent > InitDepth) ? (v + 1) : local_np;

          i++;

          /**
           * Do not break not in this loop level, as num_edges is very small.
           * LoopBound will actually cause the stream trying to do more work
           * as it goes beyond num_edges.
           * Ideally the LoopBound in stream should be enhanced with an upper
           * bound, i.e. num_edges here.
           */
        } while (i < num_edges);

#pragma ss stream_name "gap.bfs_pull.next_node.ld"
        auto next_node = cur_node->next;

        np = (local_np != 0) ? local_np : np;

        cur_node = next_node;

#ifdef NO_EARLY_BREAK
        bool broken = cur_node == nullptr;
#else
        bool broken = cur_node == nullptr || local_np > 0;
#endif
        if (broken) {
          break;
        }
      }

      if (np != 0) {
        next_parent[u] = np - 1;
        awake_count++;
      }
    }
  }

  return awake_count;
}

__attribute__((noinline)) int64_t
bfsPullSingleAdjList(const AdjGraphSingleAdjListT &g, NodeID *parent,
                     NodeID *next_parent) {

  auto num_nodes = g.N;
  auto adj_list = g.adjList;

  int64_t awake_count = 0;

#pragma omp parallel for schedule(static) reduction(+ : awake_count)           \
    firstprivate(adj_list, parent, next_parent)
  for (NodeID u = 0; u < num_nodes; u++) {

#pragma ss stream_name "gap.bfs_pull.parent.ld"
    NodeID p = parent[u];

    auto adj_ptr = adj_list + u;

#pragma ss stream_name "gap.bfs_pull.adj.lhs.ld"
    auto *lhs = reinterpret_cast<NodeID *>(adj_ptr[0]);

#pragma ss stream_name "gap.bfs_pull.adj.rhs.ld"
    auto *rhs = reinterpret_cast<NodeID *>(adj_ptr[1]);

    auto *rhs_node = AdjGraphSingleAdjListT::getNodePtr(rhs);
    auto rhs_offset = AdjGraphSingleAdjListT::getNodeOffset(rhs);

    // This helps nest the inner loop.
    auto needProcess = p < 0 && lhs != rhs;
    if (needProcess) {

      // Better to reduce from zero.
      NodeID np = 0;
      while (true) {

        auto *lhs_node = AdjGraphSingleAdjListT::getNodePtr(lhs);
        const auto local_rhs =
            lhs_node != rhs_node
                ? (lhs_node->edges + AdjGraphSingleAdjListT::EdgesPerNode)
                : rhs;
        const auto numEdges = local_rhs - lhs;

        int64_t i = 0;
        NodeID local_np = 0;
#pragma clang loop unroll(disable) vectorize(disable) interleave(disable)
        do {

#pragma ss stream_name "gap.bfs_pull.v.ld"
          NodeID v = lhs[i];

#pragma ss stream_name "gap.bfs_pull.v_parent.ld"
          NodeID v_parent = parent[v];

          local_np = (v_parent > InitDepth) ? (v + 1) : local_np;

          i++;

          /**
           * Do not break not in this loop level, as num_edges is very small.
           * LoopBound will actually cause the stream trying to do more work
           * as it goes beyond num_edges.
           * Ideally the LoopBound in stream should be enhanced with an upper
           * bound, i.e. num_edges here.
           */
        } while (i < numEdges);

        np = (local_np != 0) ? local_np : np;

#pragma ss stream_name "gap.bfs_pull.next_node.ld"
        auto next_node = lhs_node->next;

        lhs = next_node->edges;

        /**
         * I need to write this werid loop break condition to to distinguish
         * lhs and next_lhs for LoopBound.
         * TODO: Fix this in the compiler.
         */
        bool rhs_zero = next_node == rhs_node && rhs_offset == 0;

#ifdef NO_EARLY_BREAK
        bool broken = rhs_zero || (lhs_node == rhs_node);
#else
        bool broken = rhs_zero || (lhs_node == rhs_node) || local_np > 0;
#endif
        if (broken) {
          break;
        }
      }

      if (np != 0) {
        next_parent[u] = np - 1;
        awake_count++;
      }
    }
  }

  return awake_count;
}

#endif