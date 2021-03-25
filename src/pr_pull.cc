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

#ifndef GEM_FORGE
#define PROFILE_EDGES
#endif

#ifdef PROFILE_EDGES
std::vector<uint64_t> edge_lengths;
#endif

pvector<ScoreT> PageRankPull(const Graph &g, int max_iters,
                             double epsilon = 0) {
  const ScoreT init_score = 1.0f / g.num_nodes();
  const ScoreT base_score = (1.0f - kDamp) / g.num_nodes();
  pvector<ScoreT> scores(g.num_nodes(), init_score);
  pvector<ScoreT> outgoing_contribs(g.num_nodes());
  ScoreT *scores_data = scores.data();
  ScoreT *outgoing_contribs_data = outgoing_contribs.data();
  NodeID **in_neigh_index = g.in_neigh_index();
  NodeID **out_neigh_index = g.out_neigh_index();

#ifdef GEM_FORGE
  m5_detail_sim_start();
#ifdef GEM_FORGE_WARM_CACHE
  {
    NodeID *in_edges = g.in_edges();
#pragma omp parallel for firstprivate(scores_data, outgoing_contribs_data,     \
                                      out_neigh_index, in_neigh_index)
    for (NodeID n = 0; n < g.num_nodes(); n += 64 / sizeof(ScoreT)) {
      __attribute__((unused)) volatile ScoreT score = scores_data[n];
      __attribute__((unused)) volatile ScoreT outgoing_contrib =
          outgoing_contribs_data[n];
      __attribute__((unused)) volatile NodeID *in_neigh = in_neigh_index[n];
      __attribute__((unused)) volatile NodeID *out_neigh = out_neigh_index[n];
    }
#pragma omp parallel for firstprivate(in_edges)
    for (NodeID e = 0; e < g.num_edges(); e += 64 / sizeof(NodeID)) {
      // We also warm up the out edge list.
      __attribute__((unused)) volatile NodeID edge = in_edges[e];
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

#ifndef DISABLE_KERNEL_1
#pragma omp parallel for schedule(static)                                      \
    firstprivate(scores_data, out_neigh_index, outgoing_contribs_data)
    for (NodeID n = 0; n < g.num_nodes(); n++) {
      NodeID **out_neigh_ptr = out_neigh_index + n;
      NodeID *out_neigh = out_neigh_ptr[0];
      NodeID *out_neigh_next = out_neigh_ptr[1];
      ScoreT score = scores_data[n];
      int64_t out_degree = out_neigh_next - out_neigh;
      ScoreT outgoing_contrib = score / out_degree;
      outgoing_contribs_data[n] = outgoing_contrib;
    }
#endif

#ifdef GEM_FORGE
    m5_work_end(0, 0);
    m5_work_begin(1, 0);
#endif

    double error = 0;
#ifndef DISABLE_KERNEL2
#pragma omp parallel for schedule(static) reduction(+ : error) \
    firstprivate(scores_data, in_neigh_index, outgoing_contribs_data, base_score, kDamp)
    for (NodeID u = 0; u < g.num_nodes(); u++) {
      NodeID **in_neigh_ptr = in_neigh_index + u;
      NodeID *in_neigh = in_neigh_ptr[0];
      NodeID *in_neigh_next = in_neigh_ptr[1];
      int64_t in_degree = in_neigh_next - in_neigh;
#ifdef PROFILE_EDGES
      {
        size_t bucket = 0;
        int64_t remainder = in_degree;
        while (remainder >= 2) {
          remainder /= 2;
          bucket++;
        }
        if (bucket >= edge_lengths.size()) {
          bucket = edge_lengths.size() - 1;
        }
        edge_lengths[bucket] += in_degree;
      }
#endif
      ScoreT incoming_total = 0;
      for (int64_t i = 0; i < in_degree; ++i) {
        NodeID v = in_neigh[i];
        incoming_total += outgoing_contribs_data[v];
      }
      ScoreT old_score = scores_data[u];
      ScoreT new_score = base_score + kDamp * incoming_total;
      scores_data[u] = new_score;
      error += fabs(new_score - old_score);
    }
#endif
    printf(" %2d    %lf\n", iter, error);

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
    omp_set_num_threads(cli.num_threads());
  }

#ifdef PROFILE_EDGES
  edge_lengths.resize(12, 0);
#endif

  Builder b(cli);
  Graph g = b.MakeGraph();
  auto PRBound = [&cli](const Graph &g) {
    return PageRankPull(g, cli.max_iters(), cli.tolerance());
  };
  auto VerifierBound = [&cli](const Graph &g, const pvector<ScoreT> &scores) {
    return PRVerifier(g, scores, cli.tolerance());
  };
  BenchmarkKernel(cli, g, PRBound, PrintTopScores, VerifierBound);

#ifdef PROFILE_EDGES
  uint64_t sum_edges = 0;
  for (auto count : edge_lengths) {
    sum_edges += count;
  }
  for (size_t i = 0; i < edge_lengths.size(); ++i) {
    auto count = edge_lengths[i];
    auto ratio = static_cast<double>(count) / static_cast<double>(sum_edges);
    std::cout << "Edge Bucket > " << ((1 << i) >> 1) << " " << count << " "
              << ratio << '\n';
  }
#endif

  return 0;
}
