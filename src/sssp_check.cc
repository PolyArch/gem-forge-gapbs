// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#include <cassert>
#include <cinttypes>
#include <iostream>
#include <limits>
#include <queue>
#include <vector>

#include <omp.h>

#include "benchmark.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "platform_atomics.h"
#include "pvector.h"
#include "sized_array.h"
#include "source_generator.h"
#include "timer.h"

#ifdef GEM_FORGE
#include "gem5/m5ops.h"
#endif

/*
GAP Benchmark Suite
Kernel: Single-source Shortest Paths (SSSP)
Author: Scott Beamer, Yunming Zhang

Returns array of distances for all vertices from given source vertex

This SSSP implementation makes use of the ∆-stepping algorithm [1]. The type
used for weights and distances (WeightT) is typedefined in benchmark.h. The
delta parameter (-d) should be set for each input graph. This implementation
incorporates a new bucket fusion optimization [2] that significantly reduces
the number of iterations (& barriers) needed.

The bins of width delta are actually all thread-local and of type std::vector
so they can grow but are otherwise capacity-proportional. Each iteration is
done in two phases separated by barriers. In the first phase, the current
shared bin is processed by all threads. As they find vertices whose distance
they are able to improve, they add them to their thread-local bins. During this
phase, each thread also votes on what the next bin should be (smallest
non-empty bin). In the next phase, each thread copies its selected
thread-local bin into the shared bin.

Once a vertex is added to a bin, it is not removed, even if its distance is
later updated and it now appears in a lower bin. We find ignoring vertices if
their distance is less than the min distance for the current bin removes
enough redundant work to be faster than removing the vertex from older bins.

The bucket fusion optimization [2] executes the next thread-local bin in
the same iteration if the vertices in the next thread-local bin have the
same priority as those in the current shared bin. This optimization greatly
reduces the number of iterations needed without violating the priority-based
execution order, leading to significant speedup on large diameter road networks.

[1] Ulrich Meyer and Peter Sanders. "δ-stepping: a parallelizable shortest path
    algorithm." Journal of Algorithms, 49(1):114–152, 2003.

[2] Yunming Zhang, Ajay Brahmakshatriya, Xinyi Chen, Laxman Dhulipala,
    Shoaib Kamil, Saman Amarasinghe, and Julian Shun. "Optimizing ordered graph
    algorithms with GraphIt." The 18th International Symposium on Code
Generation and Optimization (CGO), pages 158-170, 2020.
*/

using namespace std;

using BinIndexT = uint32_t;
const WeightT kDistInf = numeric_limits<WeightT>::max() / 2;
const BinIndexT kMaxBin = numeric_limits<BinIndexT>::max() / 2;
// const BinIndexT kBinSizeThreshold = 1000;
/**
 * Zhengrong: We simply set a maximum number of bins, and a maximum
 * number of vertexes in the bins.
 */
const BinIndexT kMaxNumBin = 100;
using BinT = SizedArray<NodeID>;

__attribute__((noinline)) void RelaxEdges(const WGraph &g, NodeID u,
                                          WeightT dist_u, WeightT delta,
                                          pvector<WeightT> &dist,
                                          vector<BinT> &local_bins) {
  /**
   * Zhengrong: Rewrite using iterators to introduce IV.
   * Avoid the fake pointer chasing pattern in original IR.
   */
  WeightT *dist_data = dist.data();
  auto out_neighbor = g.out_neigh(u);
  const WNode *out_begin = out_neighbor.begin();
  const WNode *out_end = out_neighbor.end();
  const int64_t N = out_end - out_begin;
  for (int64_t i = 0; i < N; ++i) {
    const WNode &wn = out_begin[i];
    const WeightT weight = wn.w;
    const NodeID v = wn.v;
    WeightT new_dist = dist_u + weight;
    if (dist_data[v] > new_dist) {
      // Use clang's __atomic_fetch_min.
      WeightT old_dist =
          __atomic_fetch_min(&dist_data[v], new_dist, __ATOMIC_RELAXED);
      if (old_dist > new_dist) {
        size_t dest_bin = new_dist / delta;
        assert(dest_bin < local_bins.size());
        local_bins[dest_bin].push_back(v);
      }
    }
#if 0
    WeightT old_dist = dist[v];
    while (new_dist < old_dist) {
      if (compare_and_swap(dist[v], old_dist, new_dist)) {
        size_t dest_bin = new_dist / delta;
        assert(dest_bin < local_bins.size());
        local_bins[dest_bin].push_back(v);
        break;
      }
      old_dist = dist[v]; // swap failed, recheck dist update & retry
    }
#endif
  }
}

pvector<WeightT> DeltaStep(const WGraph &g, NodeID source, WeightT delta,
                           int warm_cahce) {
  Timer t;
  pvector<WeightT> dist(g.num_nodes(), kDistInf);
  dist[source] = 0;
  pvector<NodeID> frontier(g.num_edges_directed());
  // two element arrays for double buffering curr=iter&1, next=(iter+1)&1
  BinIndexT shared_indexes[2] = {0, kMaxBin};
  size_t frontier_tails[2] = {1, 0};
  frontier[0] = source;
  t.Start();

#ifdef GEM_FORGE
  m5_detail_sim_start();
  if (warm_cahce > 0) {
    WNode **out_neigh_index = g.out_neigh_index();
    WNode *out_edges = g.out_edges();
    WeightT *dist_data = dist.data();
#pragma omp parallel for firstprivate(out_neigh_index)
    for (NodeID n = 0; n < g.num_nodes(); n += 64 / sizeof(*out_neigh_index)) {
      __attribute__((unused)) volatile WNode *out_neigh = out_neigh_index[n];
      __attribute__((unused)) volatile WeightT dist = dist_data[n];
    }
    if (warm_cahce > 1) {
#pragma omp parallel for firstprivate(out_edges)
      for (NodeID e = 0; e < g.num_edges(); e += 64 / sizeof(WNode)) {
        // We also warm up the out edge list.
        __attribute__((unused)) volatile WNode edge = out_edges[e];
      }
    }
  }
  std::cout << "Warm up done.\n";
  m5_reset_stats(0, 0);
#endif

#pragma omp parallel
  {
    vector<BinT> local_bins;
    local_bins.reserve(kMaxNumBin);
    for (int i = 0; i < kMaxNumBin; ++i) {
      local_bins.emplace_back(g.num_edges_directed());
    }
    size_t iter = 0;
    while (shared_indexes[iter & 1] != kMaxBin) {
#ifdef GEM_FORGE
#pragma omp single nowait
      m5_work_begin(0, 0);
#endif
      BinIndexT &curr_bin_index = shared_indexes[iter & 1];
      BinIndexT &next_bin_index = shared_indexes[(iter + 1) & 1];
      size_t &curr_frontier_tail = frontier_tails[iter & 1];
      size_t &next_frontier_tail = frontier_tails[(iter + 1) & 1];
#pragma omp for nowait schedule(static)
      for (size_t i = 0; i < curr_frontier_tail; i++) {
        NodeID u = frontier[i];
        WeightT dist_u = dist[u];
        if (dist_u >= delta * static_cast<WeightT>(curr_bin_index)) {
          RelaxEdges(g, u, dist_u, delta, dist, local_bins);
        }
      }
/**
 * From our testing, this seems not critical. Disable for now.
 */
#if 0
      while (curr_bin_index < local_bins.size() &&
             !local_bins[curr_bin_index].empty() &&
             local_bins[curr_bin_index].size() < kBinSizeThreshold) {
        vector<NodeID> curr_bin_copy = local_bins[curr_bin_index];
        local_bins[curr_bin_index].resize(0);
        for (NodeID u : curr_bin_copy)
          RelaxEdges(g, u, delta, dist, local_bins);
      }
#endif
      for (BinIndexT i = curr_bin_index; i < local_bins.size(); i++) {
        if (!local_bins[i].empty()) {
#ifdef __clang__
          __atomic_fetch_min(&next_bin_index, i, __ATOMIC_RELAXED);
#else
#pragma omp critical
          next_bin_index = min(next_bin_index, i);
#endif
          break;
        }
      }
#pragma omp barrier
#pragma omp single nowait
      {
        t.Stop();
        PrintStep(curr_bin_index, t.Millisecs(), curr_frontier_tail);
        t.Start();
        curr_bin_index = kMaxBin;
        curr_frontier_tail = 0;
      }
      if (next_bin_index < local_bins.size()) {
        size_t copy_start = fetch_and_add(next_frontier_tail,
                                          local_bins[next_bin_index].size());
        copy(local_bins[next_bin_index].begin(),
             local_bins[next_bin_index].end(), frontier.data() + copy_start);
        local_bins[next_bin_index].resize(0);
      }
      iter++;
#pragma omp barrier
#ifdef GEM_FORGE
#pragma omp single nowait
      m5_work_end(0, 0);
#endif
    }
#pragma omp single
    cout << "took " << iter << " iterations" << endl;
  }
#ifdef GEM_FORGE
  m5_detail_sim_end();
  exit(0);
#endif
  return dist;
}

void PrintSSSPStats(const WGraph &g, const pvector<WeightT> &dist) {
  auto NotInf = [](WeightT d) { return d != kDistInf; };
  int64_t num_reached = count_if(dist.begin(), dist.end(), NotInf);
  cout << "SSSP Tree reaches " << num_reached << " nodes" << endl;
}

// Compares against simple serial implementation
bool SSSPVerifier(const WGraph &g, NodeID source,
                  const pvector<WeightT> &dist_to_test) {
  // Serial Dijkstra implementation to get oracle distances
  pvector<WeightT> oracle_dist(g.num_nodes(), kDistInf);
  oracle_dist[source] = 0;
  typedef pair<WeightT, NodeID> WN;
  priority_queue<WN, vector<WN>, greater<WN>> mq;
  mq.push(make_pair(0, source));
  while (!mq.empty()) {
    WeightT td = mq.top().first;
    NodeID u = mq.top().second;
    mq.pop();
    if (td == oracle_dist[u]) {
      for (WNode wn : g.out_neigh(u)) {
        if (td + wn.w < oracle_dist[wn.v]) {
          oracle_dist[wn.v] = td + wn.w;
          mq.push(make_pair(td + wn.w, wn.v));
        }
      }
    }
  }
  // Report any mismatches
  bool all_ok = true;
  for (NodeID n : g.vertices()) {
    if (dist_to_test[n] != oracle_dist[n]) {
      cout << n << ": " << dist_to_test[n] << " != " << oracle_dist[n] << endl;
      all_ok = false;
    }
  }
  return all_ok;
}

int main(int argc, char *argv[]) {
  CLDelta<WeightT> cli(argc, argv, "single-source shortest-path");
  if (!cli.ParseArgs())
    return -1;

  if (cli.num_threads() != -1) {
    omp_set_num_threads(cli.num_threads());
  }

  WeightedBuilder b(cli);
  WGraph g = b.MakeGraph();
  std::vector<NodeID> given_sources;
  if (cli.start_vertex() != -1) {
    // CLI has higher priority.
    given_sources.push_back(cli.start_vertex());
  } else {
    // Try to get the source from file.
    given_sources = SourceGenerator<Graph>::loadSource(cli.filename());
  }
  SourcePicker<WGraph> sp(g, given_sources);
  auto SSSPBound = [&sp, &cli](const WGraph &g) {
    return DeltaStep(g, sp.PickNext(), cli.delta(), cli.warm_cache());
  };
  SourcePicker<WGraph> vsp(g, given_sources);
  auto VerifierBound = [&vsp](const WGraph &g, const pvector<WeightT> &dist) {
    return SSSPVerifier(g, vsp.PickNext(), dist);
  };
  BenchmarkKernel(cli, g, SSSPBound, PrintSSSPStats, VerifierBound);
  return 0;
}
