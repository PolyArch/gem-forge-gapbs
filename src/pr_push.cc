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
*/

using namespace std;

typedef float ScoreT;
const float kDamp = 0.85;

pvector<ScoreT> PageRankPush(const Graph &g, int max_iters,
                             double epsilon = 0) {
  const ScoreT init_score = 1.0f / g.num_nodes();
  const ScoreT base_score = (1.0f - kDamp) / g.num_nodes();
  pvector<ScoreT> scores(g.num_nodes(), init_score);
  pvector<ScoreT> next_scores(g.num_nodes(), 0);
  ScoreT *scores_data = scores.data();
  ScoreT *next_scores_data = next_scores.data();
  NodeID **out_neigh_index = g.out_neigh_index();
  NodeID *out_edges = g.out_edges();

#ifdef GEM_FORGE
  m5_detail_sim_start();
#ifdef GEM_FORGE_WARM_CACHE
  {
#pragma omp parallel for firstprivate(scores_data, next_scores_data,           \
                                      out_neigh_index)
    for (NodeID n = 0; n < g.num_nodes(); n += 64 / sizeof(ScoreT)) {
      __attribute__((unused)) volatile ScoreT score = scores_data[n];
      __attribute__((unused)) volatile ScoreT next_score = next_scores_data[n];
      __attribute__((unused)) volatile NodeID *out_neigh = out_neigh_index[n];
    }
#pragma omp parallel for firstprivate(out_edges)
    for (NodeID e = 0; e < g.num_edges(); e += 64 / sizeof(NodeID)) {
      // We also warm up the out edge list.
      __attribute__((unused)) volatile NodeID edge = out_edges[e];
    }
  }
  std::cout << "Warm up done.\n";
#endif
  m5_reset_stats(0, 0);
#endif

  for (int iter = 0; iter < max_iters; iter++) {

#ifdef GEM_FORGE
    m5_work_begin(0, 0);
#endif

#ifndef DISABLE_KERNEL1
#pragma omp parallel for schedule(static)                                      \
    firstprivate(scores_data, out_neigh_index, next_scores_data)
    for (NodeID n = 0; n < g.num_nodes(); n++) {
      NodeID **out_neigh_ptr = out_neigh_index + n;
      NodeID *out_neigh = out_neigh_ptr[0];
      NodeID *out_neigh_next = out_neigh_ptr[1];
      ScoreT score = scores_data[n];
      int64_t out_degree = out_neigh_next - out_neigh;
      ScoreT outgoing_contrib = score / out_degree;
      for (int64_t i = 0; i < out_degree; ++i) {
        NodeID v = out_neigh[i];
        __atomic_fetch_fadd(next_scores_data + v, outgoing_contrib,
                            __ATOMIC_RELAXED);
      }
    }
#endif

#ifdef GEM_FORGE
    m5_work_end(0, 0);
    m5_work_begin(1, 0);
#endif

    float error = 0;
#ifndef DISABLE_KERNEL2
#pragma omp parallel for reduction(+ : error) schedule(static) \
  firstprivate(scores_data, next_scores_data, base_score, kDamp)
    for (NodeID n = 0; n < g.num_nodes(); n++) {
      ScoreT score = scores_data[n];
      ScoreT next_score = base_score + kDamp * next_scores_data[n];
      error += next_score > score ? (next_score - score) : (score - next_score);
      scores_data[n] = next_score;
      next_scores_data[n] = 0;
    }
#endif

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

  return scores;
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
    printf("%d.\n", cli.num_threads());
    omp_set_num_threads(cli.num_threads());
  }

  Builder b(cli);
  Graph g = b.MakeGraph();
  auto PRBound = [&cli](const Graph &g) {
    return PageRankPush(g, cli.max_iters(), cli.tolerance());
  };
  auto VerifierBound = [&cli](const Graph &g, const pvector<ScoreT> &scores) {
    return PRVerifier(g, scores, cli.tolerance());
  };
  BenchmarkKernel(cli, g, PRBound, PrintTopScores, VerifierBound);
  return 0;
}
