#include <algorithm>
#include <iostream>
#include <map>
#include <unordered_map>
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

using AdjGraphNoPrevT =
    AdjListGraph<NodeID, NodeID, 0, sizeof(NodeID), 64, false>;
using AdjGraphNoPrevNoNumEdgeT =
    AdjListGraph<NodeID, NodeID, 0, sizeof(NodeID), 64, false, false>;

#endif

#include "gem5/m5ops.h"

using namespace std;

void AnalyzerStorageOverhead(const GraphT &g) {

  std::map<int, std::pair<int, int>> bucket_degree_dist;
  for (int i = 1; i < g.num_edges_directed(); i *= 2) {
    bucket_degree_dist.emplace(i, std::make_pair(0, 0));
  }
  bucket_degree_dist.emplace(g.num_edges_directed(), std::make_pair(0, 0));
  std::unordered_map<int, int> degree_dist;
  for (int u = 0; u < g.num_nodes(); ++u) {
    auto out_degree = g.out_degree(u);
    degree_dist.emplace(out_degree, 0).first->second++;
    auto iter = bucket_degree_dist.upper_bound(out_degree);
    assert(iter != bucket_degree_dist.end());
    iter->second.first++;
    iter->second.second += out_degree;
  }
  for (const auto &bucket : bucket_degree_dist) {
    auto degree = bucket.first - 1;
    auto node_count = bucket.second.first;
    auto edge_count = bucket.second.second;
    if (node_count == 0) {
      continue;
    }
    float node_ratio = static_cast<float>(node_count) / g.num_nodes();
    float edge_ratio = static_cast<float>(edge_count) / g.num_edges_directed();
    printf("[Analyze] Degree <= %10d %10d %.4f E %10d %.4f.\n", degree,
           node_count, node_ratio, edge_count, edge_ratio);
  }
  int edge_size = sizeof(EdgeType);
  // next/prev points + cnt.
  int node_meta_sizes[] = {12, 16, 24};

  for (int node_size = 64; node_size < 128; node_size *= 2) {
    for (auto node_meta_size : node_meta_sizes) {
      const int edges_per_node = (node_size - node_meta_size) / edge_size;
      int num_adj_nodes = 0;
      std::map<int, int> node_dist;
      for (const auto &entry : degree_dist) {
        auto degree = entry.first;
        auto count = entry.second;
        auto nodes = (degree + edges_per_node - 1) / edges_per_node;
        num_adj_nodes += nodes * count;

        auto full_nodes = degree / edges_per_node;
        node_dist.emplace(edges_per_node, 0).first->second +=
            full_nodes * count;

        auto remainder_edges = degree % edges_per_node;
        if (remainder_edges) {
          node_dist.emplace(remainder_edges, 0).first->second += count;
        }
      }
      auto overhead = static_cast<float>(num_adj_nodes * node_size) /
                      static_cast<float>(g.num_edges_directed() * edge_size);
      printf("[Analyze] NodeSize %5d Edge/Node %5d Total AdjNodes %10d %2f.\n",
             node_size, edges_per_node, num_adj_nodes, overhead * 100.f);
      for (const auto &entry : node_dist) {
        auto node_degree = entry.first;
        auto node_count = entry.second;
        auto node_ratio = static_cast<float>(node_count) / num_adj_nodes;
        printf("[Analyze]   AdjNodeSize %5d %10d %7.2f.\n", node_degree,
               node_count, node_ratio * 100.f);
      }
    }
  }
}

template <typename AdjGraphT>
void BuildImpl(int num_threads, int64_t num_nodes, NodeID *neigh_index_offset,
               EdgeType *edges, ScoreT *props) {

  printf("Start to build AdjListGraph, node %luB edge/node %d.\n",
         sizeof(typename AdjGraphT::AdjListNode), AdjGraphT::EdgesPerNode);

#ifndef GEM_FORGE
  Timer adjBuildTimer;
  adjBuildTimer.Start();
#endif // GEM_FORGE

  AdjGraphT adjGraph(num_threads, num_nodes, neigh_index_offset, edges, props);

#ifndef GEM_FORGE
  adjBuildTimer.Stop();
  printf("AdjListGraph built %10.5lfs.\n", adjBuildTimer.Seconds());
#else
  printf("AdjListGraph built.\n");
#endif // GEM_FORGE
}

void BuildAdjGraph(const GraphT &g, int warm_cache = 2, int num_threads = 1) {
  const auto num_nodes = g.num_nodes();
  const auto __attribute__((unused)) num_edges = g.num_edges_directed();

  pvector<ScoreT> props(num_nodes, 0);

  ScoreT *props_ptr = props.data();

#ifndef USE_PULL
  auto *edges = g.out_edges();
  auto *neigh_index_offset = g.out_neigh_index_offset();
#else
  auto *edges = g.in_edges();
  auto *neigh_index_offset = g.in_neigh_index_offset();
#endif

  m5_stream_nuca_region("gap.pr.score1", props_ptr, sizeof(ScoreT), num_nodes,
                        0, 0);
  m5_stream_nuca_set_property(props_ptr, STREAM_NUCA_REGION_PROPERTY_INTERLEAVE,
                              roundUpPow2(num_nodes / 64) * sizeof(ScoreT));
  // Do a remap first.
  m5_stream_nuca_remap();

  printf(">>>>>>>>>>>>>>>>>>> Testing AdjListGraph with 24B MetaData.\n");
  clear_affinity_alloc();
  BuildImpl<AdjGraphT>(num_threads, num_nodes, neigh_index_offset, edges,
                       props_ptr);
  print_affinity_alloc_stats();

  printf(">>>>>>>>>>>>>>>>>>> Testing AdjListGraph with 16B MetaData "
         "(NoPrevPtr).\n");
  clear_affinity_alloc();
  BuildImpl<AdjGraphNoPrevT>(num_threads, num_nodes, neigh_index_offset, edges,
                             props_ptr);
  print_affinity_alloc_stats();

  printf(">>>>>>>>>>>>>>>>>>> Testing AdjListGraph with 8B MetaData "
         "(NoPrevPtr NoNumEdge).\n");
  clear_affinity_alloc();
  BuildImpl<AdjGraphNoPrevNoNumEdgeT>(num_threads, num_nodes,
                                      neigh_index_offset, edges, props_ptr);
  print_affinity_alloc_stats();

  printf(">>>>>>>>>>>>>>>>>>> Testing AdjListGraph with SingleAdjList\n");
  clear_affinity_alloc();
  BuildImpl<AdjGraphSingleAdjListT>(num_threads, num_nodes, neigh_index_offset,
                                    edges, props_ptr);
  print_affinity_alloc_stats();
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

  printf(">>>> Graph %s\n", cli.filename().c_str());

  BuilderT b(cli);
  GraphT g = b.MakeGraph();

  AnalyzerStorageOverhead(g);

  BuildAdjGraph(g, cli.warm_cache(), cli.num_threads());
  return 0;
}
