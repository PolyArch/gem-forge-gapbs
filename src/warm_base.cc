#include <algorithm>
#include <iostream>
#include <vector>

#include <omp.h>

#include "benchmark.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "pvector.h"

#ifdef USE_DOUBLE_SCORE_T
typedef double ScoreT;
#else
typedef float ScoreT;
#endif

#ifndef OMP_SCHEDULE_TYPE
#define OMP_SCHEDULE_TYPE schedule(static)
#endif

#ifndef SCORES_OFFSET_BYTES
#define SCORES_OFFSET_BYTES 0
#endif

#ifdef USE_WEIGHT_GRAPH
#define EdgeType WNode
#define AdjGraphT WAdjGraph
#define GraphT WGraph
#define BuilderT WeightedBuilder
#else
#define EdgeType NodeID
#define AdjGraphT AdjGraph
#define GraphT Graph
#define BuilderT Builder
#endif

#ifdef USE_EDGE_INDEX_OFFSET
#define EdgeIndexT NodeID
#else
#define EdgeIndexT EdgeType *
#endif // USE_EDGE_INDEX_OFFSET

#ifdef GEM_FORGE
#include "gem5/m5ops.h"
#endif

using namespace std;

pvector<ScoreT> BuildAdjGraph(const GraphT &g, int warm_cache = 2,
                          int num_threads = 1) {
  const auto num_nodes = g.num_nodes();
  const auto __attribute__((unused)) num_edges = g.num_edges_directed();

  const ScoreT init_score = 1.0f / num_nodes;

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
  m5_stream_nuca_region("gap.pr.score0", scores_data0, sizeof(ScoreT),
                        num_nodes, 0, 0);
  m5_stream_nuca_region("gap.pr.score1", scores_data1, sizeof(ScoreT),
                        num_nodes, 0, 0);
  m5_stream_nuca_set_property(scores_data1,
                              STREAM_NUCA_REGION_PROPERTY_INTERLEAVE,
                              roundUpPow2(num_nodes / 64) * sizeof(ScoreT));
  m5_stream_nuca_align(scores_data0, scores_data1, 0);

  m5_stream_nuca_region("gap.pr.out_neigh_index", out_neigh_index,
                        sizeof(EdgeIndexT), num_nodes, 0, 0);
  m5_stream_nuca_region("gap.pr.out_edge", out_edges, sizeof(NodeID), num_edges,
                        0, 0);
  m5_stream_nuca_align(out_neigh_index, scores_data1, 0);
  m5_stream_nuca_align(out_edges, scores_data1,
                       m5_stream_nuca_encode_ind_align(0, sizeof(NodeID)));
  m5_stream_nuca_align(out_edges, out_neigh_index,
                       m5_stream_nuca_encode_csr_index());
  if (in_neigh_index != out_neigh_index) {
    // This is directed graph.
    m5_stream_nuca_region("gap.pr.in_neigh_index", in_neigh_index,
                          sizeof(EdgeIndexT), num_nodes, 0, 0);
    m5_stream_nuca_region("gap.pr.in_edge", in_edges, sizeof(NodeID), num_edges,
                          0, 0);
    m5_stream_nuca_align(in_neigh_index, scores_data1, 0);
    m5_stream_nuca_align(in_edges, scores_data1,
                         m5_stream_nuca_encode_ind_align(0, sizeof(NodeID)));
    m5_stream_nuca_align(in_edges, in_neigh_index,
                         m5_stream_nuca_encode_csr_index());
  }

  // Do a remap first.
  m5_stream_nuca_remap();
#endif // GEM_FORGE

#ifdef USE_ADJ_LIST
  printf("Start to build AdjListGraph, node %luB.\n",
         sizeof(AdjGraphT::AdjListNode));

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

#ifndef USE_PUSH
    // Pull uses out_neigh_index for outgoing degree.
    gf_warm_array("out_neigh_index", out_neigh_index,
                  num_nodes * sizeof(out_neigh_index[0]));
#endif

#ifdef USE_ADJ_LIST
    // Warm up the adjacent list.
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

    std::cout << "Warm up done.\n";
  }
#endif // GEM_FORGE

  exit(0);

  return scores0;
}

void PrintTopScores(const GraphT &g, const pvector<ScoreT> &scores) {
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

int main(int argc, char *argv[]) {
  CLApp cli(argc, argv, "graphwarmer");
  if (!cli.ParseArgs())
    return -1;

  if (cli.num_threads() != -1) {
    printf("NumThreads = %d.\n", cli.num_threads());
    // It begins with 1 thread.
    omp_set_num_threads(1);
  }

  BuilderT b(cli);
  GraphT g = b.MakeGraph();
  auto PRBound = [&cli](const GraphT &g) {
    return BuildAdjGraph(g, cli.warm_cache(), cli.num_threads());
  };
  auto VerifierBound = [](const GraphT &g,
                          const pvector<ScoreT> &scores) -> bool {
    return true;
  };
  BenchmarkKernel(cli, g, PRBound, PrintTopScores, VerifierBound);
  return 0;
}
