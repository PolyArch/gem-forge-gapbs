// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#include <algorithm>
#include <cstdlib>
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

#ifdef USE_EDGE_INDEX_OFFSET
#define EdgeIndexT NodeID
#else
#define EdgeIndexT NodeID *
#endif // USE_EDGE_INDEX_OFFSET

pvector<ScoreT> PageRankPull(const Graph &g, int max_iters, int warm_cache,
                             double epsilon = 0) {
  const auto num_nodes = g.num_nodes();
  const auto num_edges = g.num_edges_directed();
  NodeID *in_edges = g.in_edges();

  const ScoreT init_score = 1.0f / num_nodes;
  const ScoreT base_score = (1.0f - kDamp) / num_nodes;
  pvector<ScoreT> scores(num_nodes, init_score);
  pvector<ScoreT> out_contribs(num_nodes);
  ScoreT *scores_data = scores.data();
  ScoreT *out_contribs_data = out_contribs.data();

  // Shuffle the nodes.
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

  // Precompute the out_degree.
  pvector<NodeID> out_degrees(num_nodes);
  {
    NodeID **out_neigh_index = g.out_neigh_index();
    for (NodeID i = 0; i < num_nodes; ++i) {
      NodeID **out_neigh_ptr = out_neigh_index + i;
      NodeID *out_neigh = out_neigh_ptr[0];
      NodeID *out_neigh_next = out_neigh_ptr[1];
      int64_t out_degree = out_neigh_next - out_neigh;
      out_degrees[i] = out_degree;
    }
  }
  NodeID *out_degrees_data = out_degrees.data();

#ifdef USE_EDGE_INDEX_OFFSET
  EdgeIndexT *in_neigh_index = g.in_neigh_index_offset();
#else
  EdgeIndexT *in_neigh_index = g.in_neigh_index();
#endif // USE_EDGE_INDEX_OFFSET

#ifdef GEM_FORGE

  m5_stream_nuca_region("gap.pr_pull.score", scores_data, sizeof(ScoreT),
                        num_nodes);
  m5_stream_nuca_region("gap.pr_pull.out_contrib", out_contribs_data,
                        sizeof(ScoreT), num_nodes);
  m5_stream_nuca_region("gap.pr_pull.out_degree", out_degrees_data,
                        sizeof(NodeID), num_nodes);
  m5_stream_nuca_region("gap.pr_pull.in_neigh_index", in_neigh_index,
                        sizeof(EdgeIndexT), num_nodes);
  m5_stream_nuca_region("gap.pr_pull.in_edges", in_edges, sizeof(NodeID),
                        num_edges);
  m5_stream_nuca_align(scores_data, out_contribs_data, 0);
  m5_stream_nuca_align(out_degrees_data, out_contribs_data, 0);
  m5_stream_nuca_align(in_neigh_index, out_contribs_data, 0);
  m5_stream_nuca_align(in_edges, out_contribs_data,
                       m5_stream_nuca_encode_ind_align(0, sizeof(NodeID)));
  m5_stream_nuca_remap();

#endif

#ifdef GEM_FORGE
  m5_detail_sim_start();
  if (warm_cache > 0) {
    gf_warm_array("nodes", nodes_data, num_nodes * sizeof(nodes_data[0]));
    gf_warm_array("scores", scores_data, num_nodes * sizeof(scores_data[0]));
    gf_warm_array("out_contrib", out_contribs_data,
                  num_nodes * sizeof(out_contribs_data[0]));
    gf_warm_array("out_degree", out_degrees_data,
                  num_nodes * sizeof(out_degrees_data[0]));
    gf_warm_array("in_neigh_index", in_neigh_index,
                  num_nodes * sizeof(in_neigh_index[0]));
    if (warm_cache > 1) {
      gf_warm_array("in_edges", in_edges, num_edges * sizeof(in_edges[0]));
    }
  }
  std::cout << "Warm up done.\n";
  m5_reset_stats(0, 0);
#endif

  for (int iter = 0; iter < max_iters; iter++) {

#ifdef GEM_FORGE
    m5_work_begin(0, 0);
#endif

#ifndef DISABLE_KERNEL_1
#pragma omp parallel for schedule(static)                                      \
    firstprivate(scores_data, out_degrees_data, out_contribs_data)
    for (NodeID n = 0; n < g.num_nodes(); n++) {

#pragma ss stream_name "gap.pr_pull.contrib.score.ld"
      ScoreT score = scores_data[n];

#pragma ss stream_name "gap.pr_pull.contrib.degree.ld"
      NodeID out_degree = out_degrees_data[n];

      ScoreT out_contrib = score / out_degree;

#pragma ss stream_name "gap.pr_pull.contrib.contrib.st"
      out_contribs_data[n] = out_contrib;
    }
#endif

#ifdef GEM_FORGE
    m5_work_end(0, 0);
    m5_work_begin(1, 0);
#endif

    double error = 0;
#ifndef DISABLE_KERNEL2
#pragma omp parallel for schedule(static) reduction(+ : error) \
    firstprivate(nodes_data, scores_data, in_edges, in_neigh_index, out_contribs_data, base_score, kDamp)
    for (int64_t x = 0; x < g.num_nodes(); ++x) {

#pragma ss stream_name "gap.pr_pull.acc.node.ld"
      NodeID u = nodes_data[x];

      EdgeIndexT *in_neigh_ptr = in_neigh_index + u;

#pragma ss stream_name "gap.pr_pull.acc.in_begin.ld"
      EdgeIndexT in_begin = in_neigh_ptr[0];

#pragma ss stream_name "gap.pr_pull.acc.in_end.ld"
      EdgeIndexT in_end = in_neigh_ptr[1];

#ifdef USE_EDGE_INDEX_OFFSET
      NodeID *in_ptr = in_edges + in_begin;
#else
      NodeID *in_ptr = in_begin;
#endif

      int64_t in_degree = in_end - in_begin;
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

#pragma ss stream_name "gap.pr_pull.acc.in_v.ld"
        NodeID v = in_ptr[i];

#pragma ss stream_name "gap.pr_pull.acc.contrib.ld"
        ScoreT contrib = out_contribs_data[v];

        incoming_total += contrib;
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
    ScoreT out_contrib = scores[u] / g.out_degree(u);
    for (NodeID v : g.out_neigh(u))
      incomming_sums[v] += out_contrib;
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
    return PageRankPull(g, cli.max_iters(), cli.warm_cache(), cli.tolerance());
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
