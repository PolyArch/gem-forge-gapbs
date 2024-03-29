// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#include <algorithm>
#include <iostream>
#include <vector>

#include <omp.h>

#include "benchmark.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "pvector.h"

#include "pr_kernels.h"

#ifdef GEM_FORGE
#include "gem5/m5ops.h"
#endif

#if defined(USE_ADJ_LIST_SINGLE_LIST)
using AdjGraphT = AdjGraphSingleAdjListT;
#define PRPushAdjFunc pageRankPushSingleAdjList
#define PRPullAdjFunc pageRankPullSingleAdjList

#elif defined(USE_ADJ_LIST_MIX_CSR)
using AdjGraphT = AdjGraphMixCSRT;
#define PRPushAdjFunc pageRankPushAdjListMixCSR
#define PRPullAdjFunc pageRankPullAdjListMixCSR

#elif defined(USE_ADJ_LIST_NO_PREV)
using AdjGraphT = AdjGraphNoPrevT;
#define PRPushAdjFunc pageRankPushAdjList
#define PRPullAdjFunc pageRankPullAdjList

#else
using AdjGraphT = AdjGraph;
#define PRPushAdjFunc pageRankPushAdjList
#define PRPullAdjFunc pageRankPullAdjList
#endif

/*
GAP Benchmark Suite
Kernel: PageRank (PR)
Author: Scott Beamer

Will return pagerank scores for all vertices once total change < epsilon

This PR implementation uses the traditional iterative approach. This is done
to ease comparisons to other implementations (often use same algorithm), but
it is not necesarily the fastest way to implement it. It does perform the
updates in the pull direction to remove the need for atomics.

Control flags:
USE_EDGE_INDEX_OFFSET: Use index_offset instead of the pointer index.
OMP_SCHEDULE_TYPE: Control how OpenMP schedules the computation.
SCORES_OFFSET_BYTES: Bytes offset between scores and next_scores.
USE_DOUBLE_SCORE_T: Use double for scores.
USE_ADJ_LIST: Use adjacent list instead of CSR.
*/

using namespace std;

pvector<ScoreT> PageRank(const Graph &g, int max_iters, double epsilon = 0,
                         int warm_cache = 2, int num_threads = 1,
                         bool graph_partition = false) {
  const auto num_nodes = g.num_nodes();
  const auto __attribute__((unused)) num_edges = g.num_edges_directed();

  const ScoreT __attribute__((unused)) init_score = 1.0f / num_nodes;
  const ScoreT __attribute__((unused)) base_score = (1.0f - kDamp) / num_nodes;

  pvector<ScoreT> scores0(num_nodes, init_score);
  pvector<ScoreT> scores1(num_nodes, 0);

  ScoreT *scores_data0 = scores0.data();
  ScoreT *scores_data1 = scores1.data();

#ifdef SHUFFLE_NODES
  // Shuffle the nodes.
  auto nodes_data = initShuffledNodes<NodeID>(num_nodes);
#endif

  /**
   * Either evenly distribute the work or by graph partition.
   */
  ThreadWorkVecT thread_works;
  if (g.hasPartition()) {
    thread_works = fuseWork(g.getNodePartition(), num_threads);
  } else {
    thread_works = generateThreadWork(g.num_nodes(), num_threads);
  }

  auto *__attribute__((unused)) out_edges = g.out_edges();
  auto *__attribute__((unused)) out_neigh_index_offset =
      g.out_neigh_index_offset();
  auto *__attribute__((unused)) out_neigh_index_ptr = g.out_neigh_index();
  auto *__attribute__((unused)) in_edges = g.in_edges();
  auto *__attribute__((unused)) in_neigh_index_offset =
      g.in_neigh_index_offset();
  auto *__attribute__((unused)) in_neigh_index_ptr = g.in_neigh_index();

#ifdef USE_EDGE_INDEX_OFFSET
  auto *__attribute__((unused)) out_neigh_index = out_neigh_index_offset;
  auto *__attribute__((unused)) in_neigh_index = in_neigh_index_offset;
#else
  auto *__attribute__((unused)) out_neigh_index = out_neigh_index_ptr;
  auto *__attribute__((unused)) in_neigh_index = in_neigh_index_ptr;
#endif

#ifdef GEM_FORGE

  g.declareNUCARegions(graph_partition);

#ifdef SHUFFLE_NODES
  m5_stream_nuca_region("gap.pr.nodes", nodes_data, sizeof(nodes_data[0]),
                        num_nodes, 0, 0);
#endif

  m5_stream_nuca_region("gap.pr.score0", scores_data0, sizeof(ScoreT),
                        num_nodes, 0, 0);
  m5_stream_nuca_region("gap.pr.score1", scores_data1, sizeof(ScoreT),
                        num_nodes, 0, 0);
  m5_stream_nuca_align(scores_data0, g.out_neigh_index(), 0);
  m5_stream_nuca_align(scores_data1, g.out_neigh_index(), 0);

  // Do a remap first.
  m5_stream_nuca_remap();
#endif // GEM_FORGE

#ifdef USE_ADJ_LIST
  printf("Start to build AdjListGraph, node %luB.\n",
         sizeof(AdjGraph::AdjListNode));

#ifndef GEM_FORGE
  Timer adjBuildTimer;
  adjBuildTimer.Start();
#endif // GEM_FORGE

#ifdef USE_PUSH
  AdjGraphT adjGraph(num_threads, num_nodes, out_neigh_index_offset, out_edges,
                     scores_data1);
#else
  AdjGraphT adjGraph(num_threads, num_nodes, in_neigh_index_offset, in_edges,
                     scores_data1);
#endif

#ifndef GEM_FORGE
  adjBuildTimer.Stop();
  printf("AdjListGraph built %10.5lfs.\n", adjBuildTimer.Seconds());
#else
  printf("AdjListGraph built.\n");
#endif // GEM_FORGE
#endif // USE_ADJ_LIST

#ifdef GEM_FORGE
  m5_detail_sim_start();
  if (warm_cache > 0) {
#ifdef SHUFFLE_NODES
    gf_warm_array("nodes", nodes_data, num_nodes * sizeof(nodes_data[0]));
#endif
    gf_warm_array("scores0", scores_data0, num_nodes * sizeof(scores_data0[0]));
    gf_warm_array("scores1", scores_data1, num_nodes * sizeof(scores_data1[0]));
    gf_warm_array("real_out_degrees", (void *)(g.getRealOutDegrees()),
                  num_nodes * sizeof(*g.getRealOutDegrees()));

    if (g.hasInterPartitionEdges()) {
      const auto &inter_part_edges = g.getInterPartEdges();
      gf_warm_array("inter_part_edges", inter_part_edges.data(),
                    inter_part_edges.size() * sizeof(inter_part_edges[0]));
    }
#ifndef USE_PUSH
    // Pull uses out_neigh_index for outgoing degree.
    gf_warm_array("out_neigh_index", out_neigh_index,
                  num_nodes * sizeof(out_neigh_index[0]));
#endif

#if !defined(DISABLE_KERNEL1) || !defined(DISABLE_KERNEL2)
#ifdef USE_ADJ_LIST // Warm up the adjacent list.
    adjGraph.warmAdjList();
#else // Warm up CSR list.

#ifdef USE_PUSH
    // Push CSR uses the out_edges.
    gf_warm_array("out_neigh_index", out_neigh_index,
                  num_nodes * sizeof(out_neigh_index[0]));
    if (warm_cache > 1) {
      gf_warm_array("out_edges", out_edges, num_edges * sizeof(out_edges[0]));
    }
#else
    // Pull CSR uses the in_edges.
    gf_warm_array("in_neigh_index", in_neigh_index,
                  num_nodes * sizeof(in_neigh_index[0]));
    if (warm_cache > 1) {
      gf_warm_array("in_edges", in_edges, num_edges * sizeof(in_edges[0]));
    }
#endif
#endif
#endif

    std::cout << "Warm up done.\n";
  }
#endif // GEM_FORGE

  startThreads(num_threads);

#ifdef GEM_FORGE
  m5_reset_stats(0, 0);
#endif

  std::vector<ScoreT> errs(max_iters, 0);

  int iter = 0;
  for (; iter < max_iters; iter++) {

#ifdef GEM_FORGE
    m5_work_begin(0, 0);
#endif

#ifndef GEM_FORGE
    // Testing purpose.
    for (NodeID n = 0; n < std::min(4l, g.num_nodes()); n++) {
      printf(" - Iter %d-0 %d Score0 %f Score1 %f.\n", iter, n, scores_data0[n],
             scores_data1[n]);
    }
#endif

#ifndef DISABLE_KERNEL1

#ifdef USE_PUSH

#ifdef USE_ADJ_LIST
    /*********************************************************************
     * Push adj list.
     ********************************************************************/
    PRPushAdjFunc(adjGraph, thread_works,
#ifdef SHUFFLE_NODES
                  nodes_data,
#endif
                  scores_data0, scores_data1);

#else

    /*********************************************************************
     * Push CSR.
     ********************************************************************/
    pageRankPushCSR(g.num_nodes(), thread_works,
#ifdef SHUFFLE_NODES
                    nodes_data,
#endif
                    scores_data0, scores_data1, out_neigh_index, out_edges,
                    g.getRealOutDegrees());

#endif // USE_ADJ_LIST

#else

    /*********************************************************************
     * Pull update.
     ********************************************************************/
    pageRankPullUpdate(num_nodes, scores_data0, scores_data1,
                       g.getRealOutDegrees());

#endif

#endif // DISABLE_KERNEL1

#ifdef GEM_FORGE
    m5_work_end(0, 0);
#endif
// Testing purpose.
#ifndef GEM_FORGE
    for (NodeID n = 0; n < std::min(4l, g.num_nodes()); n++) {
      printf(" - Iter %d-1 %d Score0 %f Score1 %f.\n", iter, n, scores_data0[n],
             scores_data1[n]);
    }
#endif

#ifndef DISABLE_INTER_PART

#ifdef GEM_FORGE
    m5_work_begin(2, 0);
#endif
    if (g.hasInterPartitionEdges()) {

      // Handle inter-partition update.
      const auto &inter_part_edges = g.getInterPartEdges();
      pageRankPushInterPartUpdate(inter_part_edges.size(),
                                  inter_part_edges.data(), scores_data1);
    }
#ifdef GEM_FORGE
    m5_work_end(2, 0);
#endif
#endif

#ifdef GEM_FORGE
    m5_work_begin(1, 0);
#endif

    float error = 0;

#ifndef DISABLE_KERNEL2
#ifdef USE_PUSH
    /*********************************************************************
     * Push update.
     ********************************************************************/
    error = pageRankPushUpdate(num_nodes, scores_data0, scores_data1,
                               base_score, kDamp);
#else
#ifdef USE_ADJ_LIST
    /*********************************************************************
     * Pull adj list.
     ********************************************************************/
    error = PRPullAdjFunc(adjGraph,
#ifdef SHUFFLE_NODES
                          nodes_data,
#endif
                          scores_data0, scores_data1, base_score, kDamp);

#else
    /*********************************************************************
     * Pull CSR.
     ********************************************************************/
    error = pageRankPullCSR(num_nodes,
#ifdef SHUFFLE_NODES
                            nodes_data,
#endif
                            scores_data0, scores_data1, base_score, kDamp,
                            in_neigh_index, in_edges);
#endif
#endif
#endif
#ifndef GEM_FORGE
    // Testing purpose.
    for (NodeID n = 0; n < std::min(4l, g.num_nodes()); n++) {
      printf(" - Iter %d-2 %d Score0 %f Score1 %f.\n", iter, n, scores_data0[n],
             scores_data1[n]);
    }

#endif

#ifdef GEM_FORGE
    errs[iter] = error;
#else
    printf(" %2d    %f\n", iter, error);
#endif

#ifdef GEM_FORGE
    m5_work_end(1, 0);
#endif

    if (error < epsilon)
      break;
  }

#ifdef GEM_FORGE
  m5_detail_sim_end();

  for (int i = 0; i <= std::min(iter, max_iters - 1); ++i) {
    printf(" %d     %f.\n", i, errs[i]);
  }
  exit(0);
#endif

  return scores0;
}

void PrintTopScores(const Graph &g, const pvector<ScoreT> &scores) {
  vector<pair<NodeID, ScoreT>> score_pairs(g.num_nodes());
  for (NodeID n = 0; n < g.num_nodes(); n++) {
    score_pairs[n] = make_pair(n, scores[n]);
  }
  int k = 5;
  vector<pair<ScoreT, NodeID>> top_k = TopK(score_pairs, k);
  k = min(k, static_cast<int>(top_k.size()));
  for (auto kvp : top_k)
    cout << kvp.second << ":" << kvp.first << endl;
}

// Verifies by asserting a single serial iteration in push direction has
//   error < target_error
bool PRVerifier(const Graph &g, const pvector<ScoreT> &scores,
                double target_error) {
  const ScoreT base_score = (1.0f - kDamp) / g.num_nodes();
  pvector<ScoreT> incomming_sums(g.num_nodes(), 0);
  double error = 0;
  for (NodeID u : g.vertices()) {
    ScoreT outgoing_contrib = scores[u] / g.out_degree(u);
    for (NodeID v : g.out_neigh(u))
      incomming_sums[v] += outgoing_contrib;
  }
  for (NodeID n : g.vertices()) {
    error += fabs(base_score + kDamp * incomming_sums[n] - scores[n]);
    incomming_sums[n] = 0;
  }
  PrintTime("Total Error", error);
  return error < target_error;
}

int main(int argc, char *argv[]) {
  CLPageRank cli(argc, argv, "pagerank", 1e-4, 20);
  if (!cli.ParseArgs())
    return -1;

  if (cli.num_threads() != -1) {
    printf("NumThreads = %d.\n", cli.num_threads());
    // It begins with 1 thread.
    omp_set_num_threads(1);
  }

  Builder b(cli);
  Graph g = b.MakeGraph();
  auto PRBound = [&cli](const Graph &g) {
    return PageRank(g, cli.max_iters(), cli.tolerance(), cli.warm_cache(),
                    cli.num_threads(), cli.graph_partition());
  };
  auto VerifierBound = [&cli](const Graph &g, const pvector<ScoreT> &scores) {
    return PRVerifier(g, scores, cli.tolerance());
  };
  BenchmarkKernel(cli, g, PRBound, PrintTopScores, VerifierBound);
  return 0;
}
