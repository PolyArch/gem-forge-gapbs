// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#include <functional>
#include <iostream>
#include <vector>

#include <omp.h>

#include "benchmark.h"
#include "bitmap.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "platform_atomics.h"
#include "pvector.h"
#include "sized_array.h"
#include "sliding_queue.h"
#include "source_generator.h"
#include "spatial_queue.h"
#include "timer.h"
#include "util.h"

#include "immintrin.h"

#ifdef GEM_FORGE
#include "gem5/m5ops.h"
#endif

/*
GAP Benchmark Suite
Kernel: Betweenness Centrality (BC)
Author: Scott Beamer

Will return array of approx betweenness centrality scores for each vertex

This BC implementation makes use of the Brandes [1] algorithm with
implementation optimizations from Madduri et al. [2]. It is only an approximate
because it does not compute the paths from every start vertex, but only a small
subset of them. Additionally, the scores are normalized to the range [0,1].

As an optimization to save memory, this implementation uses a Bitmap to hold
succ (list of successors) found during the BFS phase that are used in the back-
propagation phase.

[1] Ulrik Brandes. "A faster algorithm for betweenness centrality." Journal of
    Mathematical Sociology, 25(2):163–177, 2001.

[2] Kamesh Madduri, David Ediger, Karl Jiang, David A Bader, and Daniel
    Chavarria-Miranda. "A faster parallel algorithm and efficient multithreaded
    implementations for evaluating betweenness centrality on massive datasets."
    International Symposium on Parallel & Distributed Processing (IPDPS), 2009.
*/

using namespace std;
typedef float ScoreT;
typedef double CountT;

const NodeID InitDepth = -1;

__attribute__((noinline)) void
PBFS(const Graph &g, NodeID source, NodeID *depths, CountT *path_counts,

#ifdef USE_SPATIAL_QUEUE
     SpatialQueue<NodeID> &squeue,
#endif

     vector<SlidingQueue<NodeID>::iterator> &depth_index,
     SlidingQueue<NodeID> &queue) {

  depths[source] = 0;
  path_counts[source] = 1;
  queue.push_back(source);
  queue.slide_window();

  auto num_nodes = g.num_nodes();

  auto out_neigh_index = g.out_neigh_index();

#pragma omp parallel firstprivate(num_nodes, depths, path_counts,              \
                                      out_neigh_index)
  {

#ifdef USE_SPATIAL_QUEUE
    const auto squeue_data = squeue.data;
    const auto squeue_meta = squeue.meta;
    const auto squeue_capacity = squeue.queue_capacity;
    const auto squeue_hash_div = squeue.hash_div;
    const auto squeue_hash_mask = squeue.hash_mask;
#else
    SizedArray<NodeID> lqueue(num_nodes);
#endif

    NodeID depth = 0;

    while (!queue.empty()) {
      depth++;

      auto q = queue.begin();
      auto N = queue.end() - q;

#pragma omp for schedule(static) nowait
      for (int64_t i = 0; i < N; ++i) {

#pragma ss stream_name "gap.bc.bfs.u.ld"
        NodeID u = q[i];

#pragma ss stream_name "gap.bc.bfs.path_u.ld"
        CountT path_u = path_counts[u];

        auto pushOp = [&](NodeID v) -> void {

#ifndef IN_CORE_IMPL
          NodeID oldDepth = InitDepth;

#pragma ss stream_name "gap.bc.bfs.swap.at"
          bool swapped = __atomic_compare_exchange_n(
              &depths[v], &oldDepth, depth, false /* weak */, __ATOMIC_RELAXED,
              __ATOMIC_RELAXED);

          // Use bitwise or to avoid compiler messing with our control flow.
          bool updatePath = swapped | (oldDepth == depth);
          if (updatePath) {
#pragma ss stream_name "gap.bc.bfs.cnt.at"
            __atomic_fetch_fadd(&path_counts[v], path_u, __ATOMIC_RELAXED);
          }

          if (swapped) {

#ifdef USE_SPATIAL_QUEUE

            /**
             * Hash into the spatial queue.
             */
            auto queue_idx = (v / squeue_hash_div) & squeue_hash_mask;

            // printf("Enqueue %d to %d.\n", v, queue_idx);

#pragma ss stream_name "gap.bc.bfs.enque.at"
            auto queue_loc = __atomic_fetch_add(&squeue_meta[queue_idx].size[0],
                                                1, __ATOMIC_RELAXED);

#pragma ss stream_name "gap.bc.bfs.enque.st"
            squeue_data[queue_idx * squeue_capacity + queue_loc] = v;

#else

#pragma ss stream_name "gap.bc.bfs.enque.at"
            auto tail =
                __atomic_fetch_add(&lqueue.num_elements, 1, __ATOMIC_RELAXED);

#pragma ss stream_name "gap.bc.bfs.enque.st"
            lqueue.buffer[tail] = v;

#endif
          }

#else
          /// Optimized for In-Core. This is the original implementation.
          if ((depths[v] == -1) &&
              (compare_and_swap(depths[v], static_cast<NodeID>(InitDepth),
                                depth))) {
            lqueue.buffer[lqueue.num_elements++] = v;
          }
          if (depths[v] == depth) {
            __atomic_fetch_fadd(&path_counts[v], path_u, __ATOMIC_RELAXED);
          }
#endif
        };

        csrIterate<false>(u, out_neigh_index, pushOp);
      }

// Move to global queue.
#ifdef USE_SPATIAL_QUEUE

#pragma omp barrier

#pragma omp for schedule(static)
      for (int queue_idx = 0; queue_idx < squeue.num_queues; ++queue_idx) {
        auto squeue_size = squeue.size(queue_idx, 0);
        if (squeue_size > 0) {
          // printf("Depth %d SpatialQueue %d Size %d.\n", depth, queue_idx,
          //        squeue.size(queue_idx, 0));
          queue.append(squeue.data + queue_idx * squeue.queue_capacity,
                       squeue_size);
          squeue.clear(queue_idx, 0);
        }
      }

#else
      queue.append(lqueue.begin(), lqueue.size());
      lqueue.clear();

#pragma omp barrier
#endif

#pragma omp single
      {
        depth_index.push_back(queue.begin());
        queue.slide_window();
      }
    }
  }
  depth_index.push_back(queue.begin());
  // for (int i = 0; i + 1 < depth_index.size(); ++i) {
  //   auto lhs = depth_index[i];
  //   auto rhs = depth_index[i + 1];
  //   printf("DepthIndex %d %ld %p %p.\n", i, rhs - lhs, lhs, rhs);
  // }
}

__attribute__((noinline)) void
computeScoresPull(const Graph &g, CountT *path_counts, NodeID *depths,
                  ScoreT *scores, ScoreT *deltas,
                  vector<SlidingQueue<NodeID>::iterator> &depth_index) {

  int n = depth_index.size();
  for (int d = n - 3; d >= 0; d--) {
#pragma omp parallel for schedule(static)                                      \
    firstprivate(d, path_counts, depths, scores, deltas)
    for (auto it = depth_index[d]; it < depth_index[d + 1]; it++) {
      NodeID u = *it;
      ScoreT delta_u = 0;
      CountT path_counts_u = path_counts[u];
      for (NodeID &v : g.out_neigh(u)) {
        if (depths[v] == d + 1) {
          delta_u += (path_counts_u / path_counts[v]) * (1 + deltas[v]);
        }
      }
      deltas[u] = delta_u;
      scores[u] += delta_u;
    }
  }
}

__attribute__((noinline)) void
computeScoresPush(const Graph &g, CountT *path_counts, NodeID *depths,
                  ScoreT *scores, ScoreT *deltas,
                  vector<SlidingQueue<NodeID>::iterator> &depth_index) {

  auto in_neigh_index = g.in_neigh_index();

  int depth_index_n = depth_index.size();
  for (int d = depth_index_n - 2; d > 0; d--) {

    auto lhs = depth_index[d];
    auto rhs = depth_index[d + 1];
    auto cnt = rhs - lhs;

#pragma omp parallel for schedule(static)                                      \
    firstprivate(d, path_counts, depths, scores, deltas, lhs, in_neigh_index)
    for (int64_t i = 0; i < cnt; ++i) {

#pragma ss stream_name "gap.bc.score.u.ld"
      NodeID u = lhs[i];

#pragma ss stream_name "gap.bc.score.path_u.ld"
      CountT path_counts_u = path_counts[u];

#pragma ss stream_name "gap.bc.score.delta_u.ld"
      ScoreT delta_u = deltas[u];

      auto pushOp = [&](NodeID v) -> void {

#pragma ss stream_name "gap.bc.score.depth_v.ld"
        auto depth_v = depths[v];

        if (depth_v == d - 1) {

#pragma ss stream_name "gap.bc.score.path_v.ld"
          CountT path_counts_v = path_counts[v];

          auto value = (path_counts_v / path_counts_u) * (1 + delta_u);

#pragma ss stream_name "gap.bc.score.delta_v.at"
          __atomic_fetch_fadd(deltas + v, value, __ATOMIC_RELAXED);

#pragma ss stream_name "gap.bc.score.score_v.at"
          __atomic_fetch_fadd(scores + v, value, __ATOMIC_RELAXED);
        }
      };

      /**
       * Since these are visited vertices, the degree must be > 0.
       */
      csrIterate<true>(u, in_neigh_index, pushOp);
    }
  }
}

__attribute__((noinline)) void normalizeScores(NodeID num_nodes,
                                               ScoreT *scores) {

  // normalize scores
  ScoreT biggest_score = 0;

#pragma omp parallel for schedule(static) reduction(max : biggest_score)       \
    firstprivate(scores)
  for (NodeID n = 0; n < num_nodes; n++) {
    biggest_score = max(biggest_score, scores[n]);
  }

#pragma omp parallel for schedule(static) firstprivate(scores, biggest_score)
  for (NodeID n = 0; n < num_nodes; n++) {
    scores[n] = scores[n] / biggest_score;
  }

  printf("MaxScore %f.\n", biggest_score);
}

pvector<ScoreT> Brandes(const Graph &g, SourcePicker<Graph> &sp,
                        NodeID num_iters, int warm_cache = 2,
                        int num_threads = 1, bool graph_partition = false) {

  Timer t;
  t.Start();
  pvector<ScoreT> scores(g.num_nodes(), 0);
  pvector<CountT> path_counts(g.num_nodes());
  pvector<NodeID> depths(g.num_nodes(), -1);
  pvector<ScoreT> deltas(g.num_nodes(), 0);
  vector<SlidingQueue<NodeID>::iterator> depth_index;
  SlidingQueue<NodeID> queue(g.num_nodes());
  t.Stop();
  PrintStep("a", t.Seconds());

  const auto num_nodes = g.num_nodes();

#ifdef USE_SPATIAL_QUEUE
  const int num_banks = 64;
  const auto num_nodes_per_bank =
      roundUp(num_nodes / num_banks, 128 / sizeof(NodeID));
  const auto node_hash_mask = num_banks - 1;
  const auto node_hash_div = num_nodes_per_bank;
  SpatialQueue<NodeID> squeue(num_banks, 1 /* num_bins */, num_nodes_per_bank,
                              node_hash_div, node_hash_mask);
#endif

#ifdef GEM_FORGE

  g.declareNUCARegions(graph_partition);

  m5_stream_nuca_region("gap.bc.score", scores.data(), sizeof(ScoreT),
                        num_nodes, 0, 0);
  m5_stream_nuca_region("gap.bc.path", path_counts.data(), sizeof(CountT),
                        num_nodes, 0, 0);
  m5_stream_nuca_region("gap.bc.depth", depths.data(), sizeof(NodeID),
                        num_nodes, 0, 0);
  m5_stream_nuca_region("gap.bc.deltas", deltas.data(), sizeof(ScoreT),
                        num_nodes, 0, 0);
  m5_stream_nuca_align(scores.data(), g.out_neigh_index(), 0);
  m5_stream_nuca_align(path_counts.data(), g.out_neigh_index(), 0);
  m5_stream_nuca_align(depths.data(), g.out_neigh_index(), 0);
  m5_stream_nuca_align(deltas.data(), g.out_neigh_index(), 0);

#ifdef USE_SPATIAL_QUEUE
  m5_stream_nuca_region("gap.bc.squeue", squeue.data, sizeof(NodeID), num_nodes,
                        0, 0);
  m5_stream_nuca_region("gap.bc.squeue_meta", squeue.meta, sizeof(*squeue.meta),
                        num_banks, 0, 0);
  m5_stream_nuca_align(squeue.data, g.out_neigh_index(), 0);
  m5_stream_nuca_align(squeue.meta, g.out_neigh_index(), num_nodes_per_bank);
  m5_stream_nuca_set_property(squeue.meta,
                              STREAM_NUCA_REGION_PROPERTY_INTERLEAVE,
                              sizeof(*squeue.meta));
#endif

  m5_stream_nuca_remap();

#endif

#ifdef GEM_FORGE
  m5_detail_sim_start();
#endif

#ifdef GEM_FORGE
  if (warm_cache > 0) {
    gf_warm_array("scores", scores.data(), num_nodes * sizeof(scores[0]));
    gf_warm_array("depth", depths.data(), num_nodes * sizeof(depths[0]));
    gf_warm_array("delta", deltas.data(), num_nodes * sizeof(deltas[0]));
    gf_warm_array("path", path_counts.data(),
                  num_nodes * sizeof(path_counts[0]));
    gf_warm_array("out_neigh_index", g.out_neigh_index(),
                  num_nodes * sizeof(g.out_neigh_index()[0]));

    if (warm_cache > 1) {
      const auto num_edges = g.num_edges_directed();
      gf_warm_array("out_edges", g.out_edges(),
                    num_edges * sizeof(g.out_edges()[0]));
    }

    printf("Warm up done.\n");
  }
#endif

  startThreads(num_threads);

#ifdef GEM_FORGE
  m5_reset_stats(0, 0);
#endif

  for (NodeID iter = 0; iter < num_iters; iter++) {
    NodeID source = sp.PickNext();
    cout << "source: " << source << endl;
    t.Start();
    depths.fill(-1);
    path_counts.fill(0);
    deltas.fill(0);
    depth_index.resize(0);
    queue.reset();

#ifdef GEM_FORGE
    m5_work_begin(0, 0);
#endif

    PBFS(g, source, depths.data(), path_counts.data(),
#ifdef USE_SPATIAL_QUEUE
         squeue,
#endif
         depth_index, queue);

#ifdef GEM_FORGE
    m5_work_end(0, 0);
#endif

    t.Stop();
    PrintStep("b", t.Seconds());

    t.Start();

#ifdef GEM_FORGE
    m5_work_begin(1, 0);
#endif

#ifdef IN_CORE_IMPL
    computeScoresPull(g, path_counts.data(), depths.data(), scores.data(),
                      deltas.data(), depth_index);
#else
    computeScoresPush(g, path_counts.data(), depths.data(), scores.data(),
                      deltas.data(), depth_index);
#endif

#ifdef GEM_FORGE
    m5_work_end(1, 0);
#endif
    t.Stop();
    PrintStep("p", t.Seconds());
  }

  normalizeScores(num_nodes, scores.data());

#ifdef GEM_FORGE
  m5_detail_sim_end();
  exit(0);
#endif

  return scores;
}

void PrintTopScores(const Graph &g, const pvector<ScoreT> &scores) {
  vector<pair<NodeID, ScoreT>> score_pairs(g.num_nodes());
  for (NodeID n : g.vertices())
    score_pairs[n] = make_pair(n, scores[n]);
  int k = 5;
  vector<pair<ScoreT, NodeID>> top_k = TopK(score_pairs, k);
  for (auto kvp : top_k)
    cout << kvp.second << ":" << kvp.first << endl;
}

// Still uses Brandes algorithm, but has the following differences:
// - serial (no need for atomics or dynamic scheduling)
// - uses vector for BFS queue
// - regenerates farthest to closest traversal order from depths
// - regenerates successors from depths
bool BCVerifier(const Graph &g, SourcePicker<Graph> &sp, NodeID num_iters,
                const pvector<ScoreT> &scores_to_test) {
  pvector<ScoreT> scores(g.num_nodes(), 0);
  for (int iter = 0; iter < num_iters; iter++) {
    NodeID source = sp.PickNext();
    // BFS phase, only records depth & path_counts
    pvector<int> depths(g.num_nodes(), -1);
    depths[source] = 0;
    vector<CountT> path_counts(g.num_nodes(), 0);
    path_counts[source] = 1;
    vector<NodeID> to_visit;
    to_visit.reserve(g.num_nodes());
    to_visit.push_back(source);
    for (auto it = to_visit.begin(); it != to_visit.end(); it++) {
      NodeID u = *it;
      for (NodeID v : g.out_neigh(u)) {
        if (depths[v] == -1) {
          depths[v] = depths[u] + 1;
          to_visit.push_back(v);
        }
        if (depths[v] == depths[u] + 1)
          path_counts[v] += path_counts[u];
      }
    }
    // Get lists of vertices at each depth
    vector<vector<NodeID>> verts_at_depth;
    for (NodeID n : g.vertices()) {
      if (depths[n] != -1) {
        if (depths[n] >= static_cast<int>(verts_at_depth.size()))
          verts_at_depth.resize(depths[n] + 1);
        verts_at_depth[depths[n]].push_back(n);
      }
    }
    // Going from farthest to clostest, compute "depencies" (deltas)
    pvector<ScoreT> deltas(g.num_nodes(), 0);
    for (int depth = verts_at_depth.size() - 1; depth >= 0; depth--) {
      for (NodeID u : verts_at_depth[depth]) {
        for (NodeID v : g.out_neigh(u)) {
          if (depths[v] == depths[u] + 1) {
            deltas[u] += (path_counts[u] / path_counts[v]) * (1 + deltas[v]);
          }
        }
        scores[u] += deltas[u];
      }
    }
  }
  // Normalize scores
  ScoreT biggest_score = *max_element(scores.begin(), scores.end());
  for (NodeID n : g.vertices())
    scores[n] = scores[n] / biggest_score;
  // Compare scores
  bool all_ok = true;
  for (NodeID n : g.vertices()) {
    ScoreT delta = abs(scores_to_test[n] - scores[n]);
    if (delta > std::numeric_limits<ScoreT>::epsilon()) {
      cout << n << ": " << scores[n] << " != " << scores_to_test[n];
      cout << "(" << delta << ")" << endl;
      all_ok = false;
    }
  }
  return all_ok;
}

int main(int argc, char *argv[]) {
  CLIterApp cli(argc, argv, "betweenness-centrality", 1);
  if (!cli.ParseArgs())
    return -1;

  if (cli.num_threads() != -1) {
    printf("NumThreads = %d.\n", cli.num_threads());
    // It begins with 1 thread.
    omp_set_num_threads(1);
  }

  if (cli.num_iters() > 1 && cli.start_vertex() != -1)
    cout << "Warning: iterating from same source (-r & -i)" << endl;
  Builder b(cli);
  Graph g = b.MakeGraph();
  std::vector<NodeID> given_sources;
  if (cli.start_vertex() != -1) {
    // CLI has higher priority.
    given_sources.push_back(cli.start_vertex());
  } else {
    // Try to get the source from file.
    given_sources = SourceGenerator<Graph>::loadSource(cli.filename());
  }
  SourcePicker<Graph> sp(g, given_sources);
  auto BCBound = [&sp, &cli](const Graph &g) {
    return Brandes(g, sp, cli.num_iters(), cli.warm_cache(), cli.num_threads(),
                   cli.graph_partition());
  };
  SourcePicker<Graph> vsp(g, given_sources);
  auto VerifierBound = [&vsp, &cli](const Graph &g,
                                    const pvector<ScoreT> &scores) {
    return BCVerifier(g, vsp, cli.num_iters(), scores);
  };
  BenchmarkKernel(cli, g, BCBound, PrintTopScores, VerifierBound);
  return 0;
}
