// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#include <cassert>
#include <cinttypes>
#include <iostream>
#include <limits>
#include <queue>
#include <vector>

#include <omp.h>

#include "benchmark.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "platform_atomics.h"
#include "pvector.h"
#include "sized_array.h"
#include "source_generator.h"
#include "spatial_queue.h"
#include "timer.h"

#ifdef GEM_FORGE
#include "gem5/m5ops.h"
#endif

/*
GAP Benchmark Suite
Kernel: Single-source Shortest Paths (SSSP)
Author: Scott Beamer, Yunming Zhang

Returns array of distances for all vertices from given source vertex

This SSSP implementation makes use of the ∆-stepping algorithm [1]. The type
used for weights and distances (WeightT) is typedefined in benchmark.h. The
delta parameter (-d) should be set for each input graph. This implementation
incorporates a new bucket fusion optimization [2] that significantly reduces
the number of iterations (& barriers) needed.

The bins of width delta are actually all thread-local and of type std::vector
so they can grow but are otherwise capacity-proportional. Each iteration is
done in two phases separated by barriers. In the first phase, the current
shared bin is processed by all threads. As they find vertices whose distance
they are able to improve, they add them to their thread-local bins. During this
phase, each thread also votes on what the next bin should be (smallest
non-empty bin). In the next phase, each thread copies its selected
thread-local bin into the shared bin.

Once a vertex is added to a bin, it is not removed, even if its distance is
later updated and it now appears in a lower bin. We find ignoring vertices if
their distance is less than the min distance for the current bin removes
enough redundant work to be faster than removing the vertex from older bins.

The bucket fusion optimization [2] executes the next thread-local bin in
the same iteration if the vertices in the next thread-local bin have the
same priority as those in the current shared bin. This optimization greatly
reduces the number of iterations needed without violating the priority-based
execution order, leading to significant speedup on large diameter road networks.

[1] Ulrich Meyer and Peter Sanders. "δ-stepping: a parallelizable shortest path
    algorithm." Journal of Algorithms, 49(1):114–152, 2003.

[2] Yunming Zhang, Ajay Brahmakshatriya, Xinyi Chen, Laxman Dhulipala,
    Shoaib Kamil, Saman Amarasinghe, and Julian Shun. "Optimizing ordered graph
    algorithms with GraphIt." The 18th International Symposium on Code
Generation and Optimization (CGO), pages 158-170, 2020.

Inline RelaxEdges.

Control flags:

USE_EDGE_INDEX_OFFSET: Use index_offset instead of the pointer index.

USE_SPATIAL_QUEUE: Use a spatially localized queue instead of default thread
    localized queue.
    So far we assume that there N banks, and vertexes are divided into
    N clusters with one cluster per bank.

USE_SPATIAL_FRONTIER: Instead of copying the localized queue into a global
    frontier, we copy them into another spatial queue, this helps avoid the
    indirect traffic for the outer loop.
    NOTE: This only works when USE_SPATIAL_QUEUE is enabled.

DELTA_STEP: Specify the delta step.
    If too large, this will fall back to a single bucket.
*
*/

using namespace std;

using BinIndexT = uint32_t;
const WeightT kDistInf = numeric_limits<WeightT>::max() / 2;
const BinIndexT kMaxBin = numeric_limits<BinIndexT>::max() / 2;

#ifndef DELTA_STEP
#define DELTA_STEP 1
#endif

const WeightT kDelta = DELTA_STEP;

// const BinIndexT kBinSizeThreshold = 1000;
/**
 * Zhengrong: We simply set a maximum number of bins, and a maximum
 * number of vertexes in the bins.
 */
const BinIndexT kMaxNumBin = 64 / kDelta;
using BinT = SizedArray<NodeID>;

#ifdef USE_EDGE_INDEX_OFFSET
#define EdgeIndexT NodeID
#else
#define EdgeIndexT WNode *
#endif // USE_EDGE_INDEX_OFFSET

__attribute__((noinline)) void findMinBin(BinIndexT curr_bin_index,
                                          BinIndexT &next_bin_index,
                                          std::vector<BinT> &local_bins) {

  for (BinIndexT bin_idx = curr_bin_index; bin_idx < kMaxNumBin; bin_idx++) {

    if (!local_bins[bin_idx].empty()) {

#ifdef __clang__
      __atomic_fetch_min(&next_bin_index, bin_idx, __ATOMIC_RELAXED);
#else
#pragma omp critical
      next_bin_index = min(next_bin_index, bin_idx);
#endif

      break;
    }
  }
}

__attribute__((noinline)) void findMinBinSpatial(BinIndexT curr_bin_index,
                                                 BinIndexT &next_bin_index,
                                                 SpatialQueue<NodeID> &squeue) {

  for (BinIndexT bin_idx = curr_bin_index; bin_idx < kMaxNumBin; bin_idx++) {

    bool bin_empty = true;

    // If the bin in any queue is not empty.
    for (int queue_idx = 0; queue_idx < squeue.num_queues; ++queue_idx) {
      if (squeue.size(queue_idx, bin_idx)) {
        bin_empty = false;
        break;
      }
    }

    if (!bin_empty) {

#ifdef __clang__
      __atomic_fetch_min(&next_bin_index, bin_idx, __ATOMIC_RELAXED);
#else
#pragma omp critical
      next_bin_index = min(next_bin_index, bin_idx);
#endif

      break;
    }
  }
}

__attribute__((noinline)) void
copySpatialQueueToGlobalFrontier(SpatialQueue<NodeID> &squeue,
                                 BinIndexT next_bin_index,
                                 size_t &next_frontier_tail, NodeID *frontier) {

  const auto squeue_data = squeue.data;
  const auto squeue_queue_capacity = squeue.queue_capacity;
  const auto squeue_bin_capacity = squeue.bin_capacity;

#pragma omp for schedule(static)
  for (int queue_idx = 0; queue_idx < squeue.num_queues; ++queue_idx) {
    auto bin_size = squeue.size(queue_idx, next_bin_index);
    if (bin_size == 0) {
      continue;
    }
    auto bin_begin = squeue_data + queue_idx * squeue_queue_capacity +
                     next_bin_index * squeue_bin_capacity;
    auto bin_end = bin_begin + bin_size;

    size_t copy_start = fetch_and_add(next_frontier_tail, bin_size);
    copy(bin_begin, bin_end, frontier + copy_start);

    squeue.clear(queue_idx, next_bin_index);
  }
}

__attribute__((noinline)) void
copySpatialQueueToSpatialFrontier(SpatialQueue<NodeID> &squeue,
                                  BinIndexT next_bin_index,
                                  SpatialQueue<NodeID> &sfrontier) {

  const auto squeue_data = squeue.data;
  const auto squeue_queue_capacity = squeue.queue_capacity;
  const auto squeue_bin_capacity = squeue.bin_capacity;

  const auto sfrontier_data = sfrontier.data;
  const auto sfrontier_queue_capacity = sfrontier.queue_capacity;

#pragma omp for schedule(static)
  for (int queue_idx = 0; queue_idx < squeue.num_queues; ++queue_idx) {

    auto bin_size = squeue.size(queue_idx, next_bin_index);

    // Set the size.
    sfrontier.meta[queue_idx].size[0] = bin_size;

    if (bin_size == 0) {
      // Avoid the memcpy if nothing to move.
      continue;
    }
    auto bin_begin = squeue_data + queue_idx * squeue_queue_capacity +
                     next_bin_index * squeue_bin_capacity;
    auto bin_end = bin_begin + bin_size;

    auto frontier_start = sfrontier_data + queue_idx * sfrontier_queue_capacity;

    copy(bin_begin, bin_end, frontier_start);

    squeue.clear(queue_idx, next_bin_index);
  }
}

__attribute__((noinline)) void DeltaStepImpl(

#ifdef USE_ADJ_LIST
    WAdjGraph &adjGraph,
#else
    const WGraph &g,
#endif

    BinIndexT *shared_indexes,

#ifdef USE_SPATIAL_QUEUE
    SpatialQueue<NodeID> &squeue,
#ifdef USE_SPATIAL_FRONTIER
    SpatialQueue<NodeID> &sfrontier,
#endif
#endif

#ifndef USE_SPATIAL_FRONTIER
    pvector<NodeID> &frontier, size_t *frontier_tails,
#endif

    pvector<WeightT> &dist

) {

#ifndef GEM_FORGE
  uint64_t processed_vertexes = 0;
#endif

#ifdef USE_ADJ_LIST
  auto adj_list = adjGraph.adjList;
#pragma omp parallel firstprivate(adj_list)
#else

#ifdef USE_EDGE_INDEX_OFFSET
  EdgeIndexT *out_neigh_index = g.out_neigh_index_offset();
#else
  EdgeIndexT *out_neigh_index = g.out_neigh_index();
#endif // USE_EDGE_INDEX_OFFSET
  WNode *out_edges = g.out_edges();

#pragma omp parallel firstprivate(out_neigh_index, out_edges)
#endif
  {

#ifdef USE_SPATIAL_QUEUE
    const auto squeue_data = squeue.data;
    const auto squeue_meta = squeue.meta;
    const auto squeue_queue_capacity = squeue.queue_capacity;
    const auto squeue_bin_capacity = squeue.bin_capacity;
    const auto squeue_hash_shift = squeue.hash_shift;
    const auto squeue_hash_mask = squeue.hash_mask;

#ifdef USE_SPATIAL_FRONTIER
    const auto sfrontier_data = sfrontier.data;
    const auto sfrontier_queue_capacity = sfrontier.queue_capacity;
#endif

#else

    vector<BinT> local_bins;
    local_bins.reserve(kMaxNumBin);
    for (int i = 0; i < kMaxNumBin; ++i) {
      local_bins.emplace_back(g.num_edges_directed());
    }
#endif

    WeightT *dist_data = dist.data();

    size_t iter = 0;
    while (shared_indexes[iter & 1] != kMaxBin) {

#ifndef GEM_FORGE
      uint64_t local_processed_vertexes = 0;
#endif

      // Get the current bucket distance.
      WeightT curr_bin_weight =
          kDelta * static_cast<WeightT>(shared_indexes[iter & 1]);

#ifdef USE_SPATIAL_FRONTIER
/**
 * We need to iterate through all spatial frontiers.
 */
#pragma omp for schedule(static) nowait
      for (size_t frontier_idx = 0; frontier_idx < sfrontier.num_queues;
           ++frontier_idx) {

        const size_t frontier_size =
            sfrontier.size(frontier_idx, 0 /* binIdx */);
        NodeID *frontier_data =
            sfrontier_data + frontier_idx * sfrontier_queue_capacity;
        for (size_t i = 0; i < frontier_size; i++) {

#else
      const size_t frontier_size = frontier_tails[iter & 1];
      NodeID *frontier_data = frontier.data();

#pragma omp for schedule(static) nowait
      for (size_t i = 0; i < frontier_size; i++) {
#endif

#pragma ss stream_name "gap.sssp.frontier.ld"
          NodeID u = frontier_data[i];

#pragma ss stream_name "gap.sssp.dist.ld"
          WeightT u_dist = dist_data[u];

#ifdef USE_ADJ_LIST

#pragma ss stream_name "gap.sssp.adj.node.ld"
          auto *cur_node = adj_list[u];

          /**
           * I need to fuse them into a single condition check with
           * do while loop below so that BBPredDataGraph can handle it.
           */
          if (cur_node != nullptr && u_dist >= curr_bin_weight) {

#else

        EdgeIndexT *out_neigh_ptr = out_neigh_index + u;

#pragma ss stream_name "gap.sssp.out_begin.ld"
        EdgeIndexT out_begin = out_neigh_ptr[0];

#pragma ss stream_name "gap.sssp.out_end.ld"
        EdgeIndexT out_end = out_neigh_ptr[1];

#ifdef USE_EDGE_INDEX_OFFSET
        WNode *out_ptr = out_edges + out_begin;
#else
        WNode *out_ptr = out_begin;
#endif

        int64_t out_degree = out_end - out_begin;

        if (u_dist >= curr_bin_weight) {

#endif // USE_ADJ_LIST

#ifndef GEM_FORGE
            local_processed_vertexes++;
#endif

#ifdef USE_ADJ_LIST

#pragma clang loop unroll(disable) vectorize(disable) interleave(disable)
            do {

#pragma ss stream_name "gap.sssp.adj.n_edges.ld"
              const auto numEdges = cur_node->numEdges;

#pragma clang loop unroll(disable) vectorize(disable) interleave(disable)
              for (int64_t j = 0; j < numEdges; ++j) {

                const WNode &wn = cur_node->edges[j];
#else

          /**
           * Zhengrong: Rewrite using iterators to introduce IV.
           * Avoid the fake pointer chasing pattern in original IR.
           */
          for (int64_t i = 0; i < out_degree; ++i) {

            const WNode &wn = out_ptr[i];

#endif // USE_ADJ_LIST

#pragma ss stream_name "gap.sssp.out_w.ld"
                const WeightT weight = wn.w;

#pragma ss stream_name "gap.sssp.out_v.ld"
                const NodeID v = wn.v;

                WeightT new_dist = u_dist + weight;

#pragma ss stream_name "gap.sssp.min.at"
                WeightT old_dist = __atomic_fetch_min(&dist_data[v], new_dist,
                                                      __ATOMIC_RELAXED);

                if (old_dist > new_dist) {

#ifdef USE_SPATIAL_QUEUE
                  /**
                   * Hash into the spatial queue and bin within each queue.
                   */
                  auto queue_idx = (v >> squeue_hash_shift) & squeue_hash_mask;
                  auto bin_idx = new_dist / kDelta;

#ifndef GEM_FORGE
                  assert(bin_idx < kMaxNumBin);
#endif

#pragma ss stream_name "gap.sssp.enque.at"
                  auto queue_loc =
                      __atomic_fetch_add(&squeue_meta[queue_idx].size[bin_idx],
                                         1, __ATOMIC_RELAXED);

#pragma ss stream_name "gap.sssp.enque.st"
                  squeue_data[queue_idx * squeue_queue_capacity +
                              bin_idx * squeue_bin_capacity + queue_loc] = v;

#else
              size_t dest_bin = new_dist / kDelta;
#ifndef GEM_FORGE
              assert(dest_bin < local_bins.size());
#endif
              local_bins[dest_bin].push_back(v);
#endif // USE_SPATIAL_QUEUE

                } // old_dist > new_dist
              }   // Out vertex.

#ifdef USE_ADJ_LIST
#pragma ss stream_name "gap.sssp.adj.next_node.ld"
              auto next_node = cur_node->next;

              cur_node = next_node;
            } while (cur_node);
#endif
          } // u_dist > cur_bin_dist.
        }

#ifdef USE_SPATIAL_FRONTIER
      }
#endif

      BinIndexT &curr_bin_index = shared_indexes[iter & 1];
      BinIndexT &next_bin_index = shared_indexes[(iter + 1) & 1];

      /**
       * Search for the local bin with smallest weight.
       */
#ifdef USE_SPATIAL_QUEUE

      {
        BinIndexT old_bin_index = curr_bin_index;
        findMinBinSpatial(old_bin_index, next_bin_index, squeue);
      }

#ifndef GEM_FORGE
      fetch_and_add(processed_vertexes, local_processed_vertexes);
#endif

// Copy to frontier.
#pragma omp barrier

#ifndef GEM_FORGE
#ifdef USE_SPATIAL_FRONTIER
      size_t frontier_size = sfrontier.totalSize();
#endif
      if (omp_get_thread_num() == 0) {
        t.Stop();
        printf("%6zu%5d%11" PRId64 " %4.2f %10.5lf\n", iter, curr_bin_weight,
               frontier_size,
               static_cast<float>(processed_vertexes) /
                   static_cast<float>(frontier_size),
               t.Millisecs());
        processed_vertexes = 0;
        t.Start();
      }
#endif

      // Clear the curr_bin_index.
      curr_bin_index = kMaxBin;

#ifdef USE_SPATIAL_FRONTIER
      // Clear the spatial frontier.
      if (next_bin_index < kMaxBin) {
        copySpatialQueueToSpatialFrontier(squeue, next_bin_index, sfrontier);
      }

#else
      size_t &curr_frontier_tail = frontier_tails[iter & 1];
      size_t &next_frontier_tail = frontier_tails[(iter + 1) & 1];

      // Clear the frontier end.
      curr_frontier_tail = 0;

      if (next_bin_index < kMaxNumBin) {
        copySpatialQueueToGlobalFrontier(squeue, next_bin_index,
                                         next_frontier_tail, frontier.data());
      }
#endif

      iter++;

#else

      findMinBin(curr_bin_index, next_bin_index, local_bins);

      size_t &curr_frontier_tail = frontier_tails[iter & 1];
      size_t &next_frontier_tail = frontier_tails[(iter + 1) & 1];

#pragma omp barrier
      if (next_bin_index < kMaxNumBin) {
        size_t copy_start = fetch_and_add(next_frontier_tail,
                                          local_bins[next_bin_index].size());
        copy(local_bins[next_bin_index].begin(),
             local_bins[next_bin_index].end(), frontier.data() + copy_start);
        local_bins[next_bin_index].resize(0);
      }

#pragma omp single
      {
        curr_bin_index = kMaxBin;
        curr_frontier_tail = 0;
      }

      iter++;

#endif
    }

#pragma omp single
    cout << "took " << iter << " iterations" << endl;
  }
}

pvector<WeightT> DeltaStep(const WGraph &g, NodeID source, int num_threads,
                           int warm_cache, bool graph_partition) {
  Timer t;

  // two element arrays for double buffering curr=iter&1, next=(iter+1)&1
  BinIndexT shared_indexes[2] = {0, kMaxBin};

#ifndef USE_SPATIAL_FRONTIER
  pvector<NodeID> frontier(g.num_edges_directed());
  size_t frontier_tails[2] = {1, 0};
  frontier[0] = source;
#else
#ifndef USE_SPATIAL_QUEUE
#error "Spatial frontier must be used with spatial queue."
#endif
#endif

#ifdef USE_EDGE_INDEX_OFFSET
  EdgeIndexT *out_neigh_index = g.out_neigh_index_offset();
#else
  EdgeIndexT *out_neigh_index = g.out_neigh_index();
#endif // USE_EDGE_INDEX_OFFSET
  WNode *out_edges = g.out_edges();
  auto num_edges = g.num_edges_directed();

  auto num_nodes = g.num_nodes();
  pvector<WeightT> dist(g.num_nodes(), kDistInf);
  dist[source] = 0;

  const int num_banks = 64;
  const auto num_nodes_per_bank = roundUpPow2(num_nodes / num_banks);

#ifdef USE_SPATIAL_QUEUE
  /**
   * We have a spatial queue for each bank, and with bins.
   */
  const auto node_hash_mask = num_banks - 1;
  const auto node_hash_shift = log2Pow2(num_nodes_per_bank);
  SpatialQueue<NodeID> squeue(num_banks, kMaxNumBin,
                              num_nodes_per_bank * kMaxNumBin * kDelta,
                              node_hash_shift, node_hash_mask);

#ifdef USE_SPATIAL_FRONTIER
  /**
   * We also have a spatial queue for the frontier.
   */
  SpatialQueue<NodeID> sfrontier(num_banks, 1 /* nBins */,
                                 num_nodes_per_bank * kDelta, node_hash_shift,
                                 node_hash_mask);
  sfrontier.enque(source, 0 /* binIdx */);

#endif

#endif

#ifdef GEM_FORGE
  {
    WeightT *dist_data = dist.data();

    m5_stream_nuca_region("gap.sssp.dist", dist_data, sizeof(WeightT),
                          num_nodes, 0, 0);
    m5_stream_nuca_region("gap.sssp.out_neigh_index", out_neigh_index,
                          sizeof(EdgeIndexT), num_nodes, 0, 0);
    m5_stream_nuca_region("gap.sssp.out_edge", out_edges, sizeof(WNode),
                          num_edges, 0, 0);
    m5_stream_nuca_align(out_neigh_index, dist_data, 0);

    if (graph_partition) {
      g.setStreamNUCAPartition(dist_data, g.node_parts);
      g.setStreamNUCAPartition(out_edges, g.out_edge_parts);
    } else {
      m5_stream_nuca_set_property(dist_data,
                                  STREAM_NUCA_REGION_PROPERTY_INTERLEAVE,
                                  num_nodes_per_bank * sizeof(WeightT));
      m5_stream_nuca_align(out_edges, dist_data,
                           m5_stream_nuca_encode_ind_align(
                               offsetof(WNode, v), sizeof(((WNode *)0)->v)));
      m5_stream_nuca_align(out_edges, out_neigh_index,
                           m5_stream_nuca_encode_csr_index());
    }

#ifdef USE_SPATIAL_QUEUE
    m5_stream_nuca_region("gap.sssp.squeue", squeue.data,
                          sizeof(NodeID) * kMaxNumBin * kDelta, num_nodes, 0,
                          0);
    m5_stream_nuca_region("gap.sssp.squeue_meta", squeue.meta,
                          sizeof(*squeue.meta), num_banks, 0, 0);
    m5_stream_nuca_align(squeue.data, dist_data, 0);
    m5_stream_nuca_align(squeue.meta, dist_data, num_nodes_per_bank);
    m5_stream_nuca_set_property(squeue.meta,
                                STREAM_NUCA_REGION_PROPERTY_INTERLEAVE,
                                sizeof(*squeue.meta));

#ifdef USE_SPATIAL_FRONTIER
    m5_stream_nuca_region("gap.sssp.sfrontier", sfrontier.data,
                          sizeof(NodeID) * kDelta, num_nodes, 0, 0);
    m5_stream_nuca_region("gap.sssp.sfrontier_meta", sfrontier.meta,
                          sizeof(*sfrontier.meta), num_banks, 0, 0);
    m5_stream_nuca_align(sfrontier.data, dist_data, 0);
    m5_stream_nuca_align(sfrontier.meta, dist_data, num_nodes_per_bank);
    m5_stream_nuca_set_property(sfrontier.meta,
                                STREAM_NUCA_REGION_PROPERTY_INTERLEAVE,
                                sizeof(*sfrontier.meta));

#endif // USE_SPATIAL_FRONTIER
#endif // USE_SPATIAL_QUEUE
    m5_stream_nuca_remap();
  }
#endif // GEM_FORGE

#ifdef USE_ADJ_LIST
  printf("Start to build AdjListGraph, node %luB.\n",
         sizeof(WAdjGraph::AdjListNode));
#ifndef GEM_FORGE
  Timer adjBuildTimer;
  adjBuildTimer.Start();
#endif // GEM_FORGE
  WAdjGraph adjGraph(num_threads, g.num_nodes(), g.out_neigh_index_offset(),
                     g.out_edges(), dist.data());
#ifndef GEM_FORGE
  adjBuildTimer.Stop();
  printf("AdjListGraph built %10.5lfs.\n", adjBuildTimer.Seconds());
#else
  printf("AdjListGraph built.\n");
#endif // GEM_FORGE
#endif // USE_ADJ_LIST

  t.Start();

#ifdef GEM_FORGE
  m5_detail_sim_start();
  if (warm_cache > 0) {
    WeightT *dist_data = dist.data();
    gf_warm_array("dist", dist_data, num_nodes * sizeof(dist_data[0]));

#ifdef USE_ADJ_LIST
    gf_warm_array("adj_list", adjGraph.adjList,
                  num_nodes * sizeof(adjGraph.adjList[0]));

    if (warm_cache > 1) {
      adjGraph.warmAdjList();
    }
#else
    gf_warm_array("out_neigh_index", out_neigh_index,
                  num_nodes * sizeof(out_neigh_index[0]));
    if (warm_cache > 1) {
      WNode *out_edges = g.out_edges();
      gf_warm_array("out_edges", out_edges, num_edges * sizeof(out_edges[0]));
    }
#endif
    std::cout << "Warm up done.\n";
  }

  startThreads(num_threads);

  m5_reset_stats(0, 0);
#endif

  DeltaStepImpl(
#ifdef USE_ADJ_LIST
      adjGraph,
#else
      g,
#endif
      shared_indexes,

#ifdef USE_SPATIAL_QUEUE
      squeue,
#ifdef USE_SPATIAL_FRONTIER
      sfrontier,
#endif
#endif

#ifndef USE_SPATIAL_FRONTIER
      frontier, frontier_tails,
#endif
      dist);

#ifdef GEM_FORGE
  m5_detail_sim_end();
  exit(0);
#endif
  return dist;
}

void PrintSSSPStats(const WGraph &g, const pvector<WeightT> &dist) {
  auto NotInf = [](WeightT d) { return d != kDistInf; };
  int64_t num_reached = count_if(dist.begin(), dist.end(), NotInf);
  cout << "SSSP Tree reaches " << num_reached << " nodes" << endl;
}

// Compares against simple serial implementation
bool SSSPVerifier(const WGraph &g, NodeID source,
                  const pvector<WeightT> &dist_to_test) {
  // Serial Dijkstra implementation to get oracle distances
  pvector<WeightT> oracle_dist(g.num_nodes(), kDistInf);
  oracle_dist[source] = 0;
  typedef pair<WeightT, NodeID> WN;
  priority_queue<WN, vector<WN>, greater<WN>> mq;
  mq.push(make_pair(0, source));
  while (!mq.empty()) {
    WeightT td = mq.top().first;
    NodeID u = mq.top().second;
    mq.pop();
    if (td == oracle_dist[u]) {
      for (WNode wn : g.out_neigh(u)) {
        if (td + wn.w < oracle_dist[wn.v]) {
          oracle_dist[wn.v] = td + wn.w;
          mq.push(make_pair(td + wn.w, wn.v));
        }
      }
    }
  }
  // Report any mismatches
  bool all_ok = true;
  for (NodeID n : g.vertices()) {
    if (dist_to_test[n] != oracle_dist[n]) {
      cout << n << ": " << dist_to_test[n] << " != " << oracle_dist[n] << endl;
      all_ok = false;
    }
  }
  return all_ok;
}

int main(int argc, char *argv[]) {
  CLDelta<WeightT> cli(argc, argv, "single-source shortest-path");
  if (!cli.ParseArgs())
    return -1;

  printf("Num threads %d Delta %d.\n", cli.num_threads(), kDelta);
  if (cli.num_threads() != -1) {
    // We always start with a single thread.
    omp_set_num_threads(1);
  }

  WeightedBuilder b(cli);
  WGraph g = b.MakeGraph();
  std::vector<NodeID> given_sources;
  if (cli.start_vertex() != -1) {
    // CLI has higher priority.
    given_sources.push_back(cli.start_vertex());
  } else {
    // Try to get the source from file.
    given_sources = SourceGenerator<Graph>::loadSource(cli.filename());
  }
  SourcePicker<WGraph> sp(g, given_sources);
  auto SSSPBound = [&sp, &cli](const WGraph &g) {
    return DeltaStep(g, sp.PickNext(), cli.num_threads(), cli.warm_cache(),
                     cli.graph_partition());
  };
  SourcePicker<WGraph> vsp(g, given_sources);
  auto VerifierBound = [&vsp](const WGraph &g, const pvector<WeightT> &dist) {
    return SSSPVerifier(g, vsp.PickNext(), dist);
  };
  BenchmarkKernel(cli, g, SSSPBound, PrintSSSPStats, VerifierBound);
  return 0;
}
