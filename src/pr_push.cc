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

pvector<ScoreT> PageRankPush(const Graph &g, int max_iters, double epsilon = 0,
                             int warm_cache = 2, int num_threads = 1) {
  const auto num_nodes = g.num_nodes();
  const auto __attribute__((unused)) num_edges = g.num_edges_directed();

  const ScoreT init_score = 1.0f / num_nodes;
  const ScoreT base_score = (1.0f - kDamp) / num_nodes;

  pvector<ScoreT> scores0(num_nodes, init_score);
  pvector<ScoreT> scores1(num_nodes, 0);

  ScoreT *scores_data0 = scores0.data();
  ScoreT *scores_data1 = scores1.data();

#ifdef SHUFFLE_NODES
  // Shuffle the nodes.
  int64_t num_nodes = g.num_nodes();
  pvector<NodeID> nodes(num_nodes);
  for (NodeID i = 0; i < num_nodes; ++i) {
    nodes[i] = i;
  }
  for (NodeID i = 0; i + 1 < num_nodes; ++i) {
    // Shuffle a little bit to make it not always linear access.
    long long j = (rand() % (num_nodes - i)) + i;
    NodeID tmp = nodes[i];
    nodes[i] = nodes[j];
    nodes[j] = tmp;
  }
  NodeID *nodes_data = nodes.data();
#endif

#ifdef USE_PUSH
  auto *edges = g.out_edges();
  auto *__attribute__((unused)) neigh_index_offset = g.out_neigh_index_offset();
  auto *__attribute__((unused)) neigh_index_ptr = g.out_neigh_index();
#else
  auto *edges = g.in_edges();
  auto *__attribute__((unused)) neigh_index_offset = g.in_neigh_index_offset();
  auto *__attribute__((unused)) neigh_index_ptr = g.in_neigh_index();
#endif

#ifdef USE_EDGE_INDEX_OFFSET
  auto *__attribute__((unused)) neigh_index = neigh_index_offset;
#else
  auto *__attribute__((unused)) neigh_index = neigh_index_ptr;
#endif

#ifdef GEM_FORGE
  m5_stream_nuca_region("gap.pr.score0", scores_data0, sizeof(ScoreT),
                        num_nodes, 0, 0);
  m5_stream_nuca_region("gap.pr.score1", scores_data1, sizeof(ScoreT),
                        num_nodes, 0, 0);
  m5_stream_nuca_set_property(scores_data1,
                              STREAM_NUCA_REGION_PROPERTY_INTERLEAVE,
                              roundUpPow2(num_nodes / 64) * sizeof(ScoreT));
  m5_stream_nuca_align(scores_data0, scores_data1, 0);

#ifndef USE_ADJ_LIST

  m5_stream_nuca_region("gap.pr.neigh_index", neigh_index, sizeof(EdgeIndexT),
                        num_nodes, 0, 0);
  m5_stream_nuca_region("gap.pr.edge", edges, sizeof(NodeID), num_edges, 0, 0);
  m5_stream_nuca_align(neigh_index, scores_data1, 0);
  m5_stream_nuca_align(edges, scores_data1,
                       m5_stream_nuca_encode_ind_align(0, sizeof(NodeID)));
  m5_stream_nuca_align(edges, neigh_index, m5_stream_nuca_encode_csr_index());
#endif

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
  AdjGraph adjGraph(num_threads, num_nodes, neigh_index_offset, edges,
                    scores_data1);
#ifndef GEM_FORGE
  adjBuildTimer.Stop();
  printf("AdjListGraph built %10.5lfs.\n", adjBuildTimer.Seconds());
#else
  printf("AdjListGraph built.\n");
#endif // GEM_FORGE
#endif // USE_ADJ_LIST

  // Start the threads.
  {
    omp_set_num_threads(num_threads);
    float v;
    float *pv = &v;
#pragma omp parallel for schedule(static)
    for (uint64_t i = 0; i < num_threads; ++i) {
      __attribute__((unused)) volatile float v = *pv;
    }
  }

#ifdef GEM_FORGE
  m5_detail_sim_start();
  if (warm_cache > 0) {
#ifdef SHUFFLE_NODES
    gf_warm_array("nodes", nodes_data, num_nodes * sizeof(nodes_data[0]));
#endif
    gf_warm_array("scores0", scores_data0, num_nodes * sizeof(scores_data0[0]));
    gf_warm_array("scores1", scores_data1, num_nodes * sizeof(scores_data1[0]));

#ifdef USE_ADJ_LIST
    gf_warm_array("degrees", adjGraph.degrees,
                  num_nodes * sizeof(adjGraph.degrees[0]));
    gf_warm_array("adj_list", adjGraph.adjList,
                  num_nodes * sizeof(adjGraph.adjList[0]));

    // Warm up the adjacent list.
    adjGraph.warmAdjList();

#else // Warm up CSR list.
    gf_warm_array("neigh_index", neigh_index,
                  num_nodes * sizeof(neigh_index[0]));
    if (warm_cache > 1) {
      gf_warm_array("edges", edges, num_edges * sizeof(edges[0]));
    }
#endif

    std::cout << "Warm up done.\n";
  }
#endif // GEM_FORGE

#ifdef GEM_FORGE
  m5_reset_stats(0, 0);
#endif

  for (int iter = 0; iter < max_iters; iter++) {

#ifdef GEM_FORGE
    m5_work_begin(0, 0);
#endif

#ifndef DISABLE_KERNEL1

#ifdef USE_ADJ_LIST
    pageRankPushAdjList(adjGraph,
#ifdef SHUFFLE_NODES
                        nodes_data,
#endif
                        scores_data0, scores_data1);

#else

    pageRankPushCSR(g.num_nodes(),
#ifdef SHUFFLE_NODES
                    nodes_data,
#endif
                    scores_data0, scores_data1, neigh_index, edges);

#endif // USE_ADJ_LIST

#endif // DISABLE_KERNEL1

#ifdef GEM_FORGE
    m5_work_end(0, 0);
    m5_work_begin(1, 0);
#endif
    // // Testing purpose.
    // for (NodeID n = 0; n < g.num_nodes(); n++) {
    //   printf(" - Iter %d-1 %d Score %f NextScore %f.\n", iter, n,
    //          scores_data0[n], scores_data1[n]);
    // }

    float error = 0;
#ifndef DISABLE_KERNEL2
    error = pageRankPushUpdate(num_nodes, scores_data0, scores_data1,
                               base_score, kDamp);
#endif
    // // Testing purpose.
    // for (NodeID n = 0; n < g.num_nodes(); n++) {
    //   printf(" - Iter %d-2 %d Score %f NextScore %f.\n", iter, n,
    //          scores_data[n], scores_data1[n]);
    // }

    printf(" %2d    %f\n", iter, error);

#ifdef GEM_FORGE
    m5_work_end(1, 0);
#endif

    if (error < epsilon)
      break;
  }

#ifdef GEM_FORGE
  m5_detail_sim_end();
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
    return PageRankPush(g, cli.max_iters(), cli.tolerance(), cli.warm_cache(),
                        cli.num_threads());
  };
  auto VerifierBound = [&cli](const Graph &g, const pvector<ScoreT> &scores) {
    return PRVerifier(g, scores, cli.tolerance());
  };
  BenchmarkKernel(cli, g, PRBound, PrintTopScores, VerifierBound);
  return 0;
}
