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
  NodeID *out_edges = g.out_edges();
  const ScoreT init_score = 1.0f / num_nodes;
  const ScoreT base_score = (1.0f - kDamp) / num_nodes;

  const uint64_t page_bytes = 4096;
  const uint64_t llc_wrap_bytes = 1024 * 1024;
  const auto llc_wrap_pages = (llc_wrap_bytes + page_bytes - 1) / page_bytes;
  auto scores_bytes = num_nodes * sizeof(ScoreT);
  auto scores_pages = (scores_bytes + page_bytes - 1) / page_bytes;
  /**
   * We consider the extra meta data before the allocated area.
   * One page for next_scores, we added to scores_pages.
   * One page for gap array, we charge if gap_pages > 1.
   */
  scores_pages++;
  auto scores_remain_pages = scores_pages % llc_wrap_pages;
  auto offset_pages = (SCORES_OFFSET_BYTES + page_bytes - 1) / page_bytes;
  auto gap_pages = llc_wrap_pages - scores_remain_pages + offset_pages;
  printf("Pages: LLC %lu Score %lu Remain %lu Offset %lu Gap %lu.\n",
         llc_wrap_pages, scores_pages, scores_remain_pages, offset_pages,
         gap_pages);

  pvector<ScoreT> scores(num_nodes, init_score);
  if (gap_pages > 0) {
    auto gap_bytes = gap_pages * page_bytes;
    if (gap_pages > 1) {
      gap_bytes = (gap_pages - 1) * page_bytes;
    } else {
      gap_bytes = page_bytes / 2;
    }
    pvector<uint8_t> scores_gap(gap_bytes, 0);
  }
  pvector<ScoreT> next_scores(num_nodes, 0);

  ScoreT *scores_data = scores.data();
  ScoreT *next_scores_data = next_scores.data();

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

#ifdef GEM_FORGE
  m5_stream_nuca_region("gap.pr_push.score", scores_data, sizeof(ScoreT),
                        num_nodes, 0, 0);
  m5_stream_nuca_region("gap.pr_push.next_score", next_scores_data,
                        sizeof(ScoreT), num_nodes, 0, 0);
  m5_stream_nuca_set_property(next_scores_data,
                              STREAM_NUCA_REGION_PROPERTY_INTERLEAVE,
                              roundUpPow2(num_nodes / 64));
  m5_stream_nuca_align(scores_data, next_scores_data, 0);
#ifndef USE_ADJ_LIST
  const auto num_edges = g.num_edges_directed();

#ifdef USE_EDGE_INDEX_OFFSET
  EdgeIndexT *out_neigh_index = g.out_neigh_index_offset();
#else
  EdgeIndexT *out_neigh_index = g.out_neigh_index();
#endif // USE_EDGE_INDEX_OFFSET

  m5_stream_nuca_region("gap.pr_push.out_neigh_index", out_neigh_index,
                        sizeof(EdgeIndexT), num_nodes, 0, 0);
  m5_stream_nuca_region("gap.pr_push.out_edge", out_edges, sizeof(NodeID),
                        num_edges, 0, 0);
  m5_stream_nuca_align(out_neigh_index, next_scores_data, 0);
  m5_stream_nuca_align(out_edges, next_scores_data,
                       m5_stream_nuca_encode_ind_align(0, sizeof(NodeID)));
  m5_stream_nuca_align(out_edges, out_neigh_index,
                       m5_stream_nuca_encode_csr_index());
#endif
#endif // GEM_FORGE

#ifdef USE_ADJ_LIST
  printf("Start to build AdjListGraph, node %luB.\n",
         sizeof(AdjGraph::AdjListNode));

#ifndef GEM_FORGE
  Timer adjBuildTimer;
  adjBuildTimer.Start();
#endif // GEM_FORGE
  AdjGraph adjGraph(num_threads, g.num_nodes(), g.out_neigh_index_offset(),
                    out_edges, next_scores_data);
#ifndef GEM_FORGE
  adjBuildTimer.Stop();
  printf("AdjListGraph built %10.5lfs.\n", adjBuildTimer.Seconds());
#else
  printf("AdjListGraph built.\n");
#endif // GEM_FORGE
#endif // USE_ADJ_LIST

#ifdef GEM_FORGE
  m5_stream_nuca_remap();
#endif // GEM_FORGE

#ifdef GEM_FORGE
  m5_detail_sim_start();
  if (warm_cache > 0) {
#ifdef SHUFFLE_NODES
    gf_warm_array("nodes", nodes_data, num_nodes * sizeof(nodes_data[0]));
#endif
    gf_warm_array("scores", scores_data, num_nodes * sizeof(scores_data[0]));
    gf_warm_array("next_scores", next_scores_data,
                  num_nodes * sizeof(next_scores_data[0]));

#ifdef USE_ADJ_LIST
    gf_warm_array("degrees", adjGraph.degrees,
                  num_nodes * sizeof(adjGraph.degrees[0]));
    gf_warm_array("adj_list", adjGraph.adjList,
                  num_nodes * sizeof(adjGraph.adjList[0]));

    // Warm up the adjacent list.
    {
      printf("Start warming AdjList.\n");
      auto adj_list = adjGraph.adjList;
#pragma omp parallel for schedule(static) firstprivate(adj_list)
      for (int64_t i = 0; i < num_nodes; i++) {

        int64_t n = i;

        auto *cur_node = adj_list[n];

#pragma clang loop unroll(disable) vectorize(disable) interleave(disable)
        while (cur_node) {
          volatile auto next_node = cur_node->next;
          cur_node = next_node;
        }
      }
      printf("Warmed AdjList.\n");
    }

#else // Warm up CSR list.
    gf_warm_array("out_neigh_index", out_neigh_index,
                  num_nodes * sizeof(out_neigh_index[0]));
    if (warm_cache > 1) {
      gf_warm_array("out_edges", out_edges, num_edges * sizeof(out_edges[0]));
    }
#endif

    std::cout << "Warm up done.\n";
  }
#endif // GEM_FORGE

  // Start the threads.
  {
    float v;
    float *pv = &v;
#pragma omp parallel for schedule(static)
    for (uint64_t i = 0; i < num_threads; ++i) {
      __attribute__((unused)) volatile float v = *pv;
    }
  }

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
                        scores_data, next_scores_data);

#else

    pageRankPushCSR(g.num_nodes(),
#ifdef SHUFFLE_NODES
                    nodes_data,
#endif
                    scores_data, next_scores_data, out_neigh_index, out_edges);

#endif // USE_ADJ_LIST

#endif // DISABLE_KERNEL1

#ifdef GEM_FORGE
    m5_work_end(0, 0);
    m5_work_begin(1, 0);
#endif
    // // Testing purpose.
    // for (NodeID n = 0; n < g.num_nodes(); n++) {
    //   printf(" - Iter %d-1 %d Score %f NextScore %f.\n", iter, n,
    //          scores_data[n], next_scores_data[n]);
    // }

    float error = 0;
#ifndef DISABLE_KERNEL2
#pragma omp parallel for reduction(+ : error) schedule(static) \
  firstprivate(scores_data, next_scores_data, base_score, kDamp)
    for (NodeID n = 0; n < g.num_nodes(); n++) {

#pragma ss stream_name "gap.pr_push.update.score.ld"
      ScoreT score = scores_data[n];

#pragma ss stream_name "gap.pr_push.update.next.ld"
      ScoreT next = next_scores_data[n];

      ScoreT next_score = base_score + kDamp * next;
      error += next_score > score ? (next_score - score) : (score - next_score);

#pragma ss stream_name "gap.pr_push.update.score.st"
      scores_data[n] = next_score;

#pragma ss stream_name "gap.pr_push.update.next.st"
      next_scores_data[n] = 0;
    }
#endif
    // // Testing purpose.
    // for (NodeID n = 0; n < g.num_nodes(); n++) {
    //   printf(" - Iter %d-2 %d Score %f NextScore %f.\n", iter, n,
    //          scores_data[n], next_scores_data[n]);
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
    printf("NumThreads = %d.\n", cli.num_threads());
    omp_set_num_threads(cli.num_threads());
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
