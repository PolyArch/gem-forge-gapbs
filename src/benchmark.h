// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#ifndef BENCHMARK_H_
#define BENCHMARK_H_

#include <algorithm>
#include <cinttypes>
#include <functional>
#include <random>
#include <utility>
#include <vector>

#include "adj_list_graph.h"
#include "builder.h"
#include "cluster.h"
#include "graph.h"
#include "timer.h"
#include "util.h"
#include "writer.h"

/*
GAP Benchmark Suite
File:   Benchmark
Author: Scott Beamer

Various helper functions to ease writing of kernels
*/

// Default type signatures for commonly used types
typedef int32_t NodeID;
typedef int32_t WeightT;
typedef NodeWeight<NodeID, WeightT> WNode;

typedef CSRGraph<NodeID> Graph;
typedef CSRGraph<NodeID, WNode> WGraph;

typedef AdjListGraph<NodeID> AdjGraph;
typedef AdjListGraph<NodeID, WNode, offsetof(WNode, v), sizeof(WNode::v)>
    WAdjGraph;

using AdjGraphNoPrevT = AdjListGraph<NodeID, NodeID, 0, sizeof(NodeID), 64,
                                     false, true, AdjListTypeE::OneListPerNode>;
using AdjGraphSingleAdjListT =
    AdjListGraph<NodeID, NodeID, 0, sizeof(NodeID), 64, false, false,
                 AdjListTypeE::SingleListPerGraph>;
using WAdjGraphSingleAdjListT =
    AdjListGraph<NodeID, WNode, offsetof(WNode, v), sizeof(WNode::v), 64, false,
                 false, AdjListTypeE::SingleListPerGraph>;

using AdjGraphMixCSRT =
    AdjListGraph<NodeID, NodeID, 0, sizeof(NodeID), 64, false, true,
                 AdjListTypeE::OneListPerNodeMixCSR>;

typedef BuilderBase<NodeID, NodeID, WeightT> Builder;
typedef BuilderBase<NodeID, WNode, WeightT> WeightedBuilder;

typedef WriterBase<NodeID, NodeID> Writer;
typedef WriterBase<NodeID, WNode> WeightedWriter;

typedef ClusterBase<NodeID, NodeID> Cluster;
typedef ClusterBase<NodeID, WNode> WeightedCluster;

// Used to pick random non-zero degree starting points for search algorithms
template <typename GraphT_> class SourcePicker {
public:
  explicit SourcePicker(const GraphT_ &g,
                        const std::vector<NodeID> &_given_sources)
      : given_sources(_given_sources), current_given_source_idx(0),
        rng(kRandSeed), udist(0, g.num_nodes() - 1), g_(g) {}

  NodeID PickNext() {
    if (!given_sources.empty() && given_sources.front() != -1) {
      if (current_given_source_idx == given_sources.size()) {
        current_given_source_idx = 0;
      }
      return given_sources.at(current_given_source_idx++);
    }
    NodeID source;
    do {
      source = udist(rng);
    } while (g_.out_degree(source) == 0);
    return source;
  }

private:
  std::vector<NodeID> given_sources;
  std::size_t current_given_source_idx;
  std::mt19937 rng;
  std::uniform_int_distribution<NodeID> udist;
  const GraphT_ &g_;
};

// Returns k pairs with largest values from list of key-value pairs
template <typename KeyT, typename ValT>
std::vector<std::pair<ValT, KeyT>>
TopK(const std::vector<std::pair<KeyT, ValT>> &to_sort, size_t k) {
  std::vector<std::pair<ValT, KeyT>> top_k;
  ValT min_so_far = 0;
  for (auto kvp : to_sort) {
    if ((top_k.size() < k) || (kvp.second > min_so_far)) {
      top_k.push_back(std::make_pair(kvp.second, kvp.first));
      std::sort(top_k.begin(), top_k.end(),
                std::greater<std::pair<ValT, KeyT>>());
      if (top_k.size() > k)
        top_k.resize(k);
      min_so_far = top_k.back().first;
    }
  }
  return top_k;
}

bool VerifyUnimplemented(...) {
  std::cout << "** verify unimplemented **" << std::endl;
  return false;
}

// Calls (and times) kernel according to command line arguments
template <typename GraphT_, typename GraphFunc, typename AnalysisFunc,
          typename VerifierFunc>
void BenchmarkKernel(const CLApp &cli, const GraphT_ &g, GraphFunc kernel,
                     AnalysisFunc stats, VerifierFunc verify) {
  g.PrintStats();
  double total_seconds = 0;
  Timer trial_timer;
  for (int iter = 0; iter < cli.num_trials(); iter++) {
    trial_timer.Start();
    auto result = kernel(g);
    trial_timer.Stop();
    PrintTime("Trial Time(s) ", trial_timer.Seconds());
    total_seconds += trial_timer.Seconds();
    if (cli.do_analysis() && (iter == (cli.num_trials() - 1)))
      stats(g, result);
    if (cli.do_verify()) {
      trial_timer.Start();
      PrintLabel("Verification",
                 verify(std::ref(g), std::ref(result)) ? "PASS" : "FAIL");
      trial_timer.Stop();
      PrintTime("Verification Time(s)", trial_timer.Seconds());
    }
  }
  PrintTime("Average Time(s)", total_seconds / cli.num_trials());
}

#endif // BENCHMARK_H_
