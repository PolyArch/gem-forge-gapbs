// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#include <iostream>
#include <vector>

#include <omp.h>

#include "benchmark.h"
#include "bitmap.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "platform_atomics.h"
#include "pvector.h"
#include "sliding_queue.h"
#include "source_generator.h"
#include "timer.h"

#ifdef GEM_FORGE
#include "gem5/m5ops.h"
#endif

/*
GAP Benchmark Suite
Kernel: Breadth-First Search (BFS)
Author: Scott Beamer

Will return parent array for a BFS traversal from a source vertex

This BFS implementation makes use of the Direction-Optimizing approach [1].
It uses the alpha and beta parameters to determine whether to switch search
directions. For representing the frontier, it uses a SlidingQueue for the
top-down approach and a Bitmap for the bottom-up approach. To reduce
false-sharing for the top-down approach, thread-local QueueBuffer's are used.

To save time computing the number of edges exiting the frontier, this
implementation precomputes the degrees in bulk at the beginning by storing
them in parent array as negative numbers. Thus the encoding of parent is:
  parent[x] < 0 implies x is unvisited and parent[x] = -out_degree(x)
  parent[x] >= 0 implies x been visited

[1] Scott Beamer, Krste Asanović, and David Patterson. "Direction-Optimizing
    Breadth-First Search." International Conference on High Performance
    Computing, Networking, Storage and Analysis (SC), Salt Lake City, Utah,
    November 2012.
*/

using namespace std;

int64_t BUStep(const Graph &g, const pvector<NodeID> &nodes,
               pvector<NodeID> &parent, pvector<NodeID> &next_parent) {
  int64_t awake_count = 0;
  NodeID **graph_in_neigh_index = g.in_neigh_index();
  auto *nodes_v = nodes.data();
  auto *parent_v = parent.data();
  auto *next_parent_v = next_parent.data();
#pragma omp parallel for schedule(static) reduction(+ : awake_count) \
  firstprivate(graph_in_neigh_index, nodes_v, parent_v, next_parent_v)
  for (int64_t x = 0; x < g.num_nodes(); ++x) {
    NodeID u = nodes_v[x];
    if (parent[u] < 0) {
      // Explicit write this to avoid u + 1 and misplaced stream_step.
      NodeID *const *in_neigh_index = graph_in_neigh_index + u;
      const NodeID *in_neigh_begin = in_neigh_index[0];
      const NodeID *in_neigh_end = in_neigh_index[1];
      const auto N = in_neigh_end - in_neigh_begin;
      // Better to reduce from zero.
      NodeID p = 0;
      for (int64_t i = 0; i < N; ++i) {
        NodeID v = in_neigh_begin[i];
        NodeID vParent = parent_v[v];
        p = (vParent > -1) ? (v + 1) : p;
      }
      if (p != 0) {
        next_parent_v[u] = p - 1;
        awake_count++;
      }
    }
  }

// Copy next_parent into parent.
#pragma omp parallel for schedule(static) firstprivate(parent_v, next_parent_v)
  for (NodeID u = 0; u < g.num_nodes(); u++) {
    parent_v[u] = next_parent_v[u];
  }
  return awake_count;
}

pvector<NodeID> InitParent(const Graph &g) {
  pvector<NodeID> parent(g.num_nodes());
#pragma omp parallel for
  for (NodeID n = 0; n < g.num_nodes(); n++) {
    parent[n] = -1;
  }
  return parent;
}

pvector<NodeID> InitShuffledNodes(const Graph &g) {
  pvector<NodeID> nodes(g.num_nodes());
  for (NodeID n = 0; n < g.num_nodes(); n++) {
    nodes[n] = n;
  }
  for (NodeID i = 0; i + 1 < g.num_nodes(); ++i) {
    // Shuffle a little bit to make it not always linear access.
    long long j = (rand() % (g.num_nodes() - i)) + i;
    NodeID tmp = nodes[i];
    nodes[i] = nodes[j];
    nodes[j] = tmp;
  }
  return nodes;
}

pvector<NodeID> DOBFS(const Graph &g, NodeID source, int alpha = 15,
                      int beta = 18) {
  PrintStep("Source", static_cast<int64_t>(source));
  Timer t;
  t.Start();
  pvector<NodeID> nodes = InitShuffledNodes(g);
  pvector<NodeID> parent = InitParent(g);
  pvector<NodeID> next_parent = InitParent(g);
  t.Stop();
  PrintStep("i", t.Seconds());
  parent[source] = source;
  next_parent[source] = source;

#ifdef GEM_FORGE
  m5_detail_sim_start();
#ifdef GEM_FORGE_WARM_CACHE
  {
    NodeID **in_neigh_index = g.in_neigh_index();
    NodeID *in_edges = g.in_edges();
    NodeID *nodes_data = nodes.data();
    NodeID *parent_data = parent.data();
    NodeID *next_parent_data = parent.data();
#pragma omp parallel for firstprivate(in_neigh_index)
    for (NodeID n = 0; n < g.num_nodes(); n += 64 / sizeof(*in_neigh_index)) {
      __attribute__((unused)) volatile NodeID *in_neigh = in_neigh_index[n];
      __attribute__((unused)) volatile NodeID node = nodes_data[n];
      __attribute__((unused)) volatile NodeID parent = parent_data[n];
      __attribute__((unused)) volatile NodeID next_parent = next_parent_data[n];
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

  PrintStep("e", t.Seconds());
  int64_t awake_count = 1;
  while (awake_count > 0) {
#ifdef GEM_FORGE
    m5_work_begin(0, 0);
#else
    t.Start();
#endif
    awake_count = BUStep(g, nodes, parent, next_parent);
#ifdef GEM_FORGE
    m5_work_end(0, 0);
#else
    t.Stop();
    PrintStep("bu", t.Seconds(), awake_count);
#endif
  }
  PrintStep("c", t.Seconds());

#ifdef GEM_FORGE
  m5_detail_sim_end();
  exit(0);
#endif

#pragma omp parallel for
  for (NodeID n = 0; n < g.num_nodes(); n++)
    if (parent[n] < -1)
      parent[n] = -1;
  return parent;
}

void PrintBFSStats(const Graph &g, const pvector<NodeID> &bfs_tree) {
  int64_t tree_size = 0;
  int64_t n_edges = 0;
  for (NodeID n : g.vertices()) {
    if (bfs_tree[n] >= 0) {
      n_edges += g.out_degree(n);
      tree_size++;
    }
  }
  cout << "BFS Tree has " << tree_size << " nodes and ";
  cout << n_edges << " edges" << endl;
}

// BFS verifier does a serial BFS from same source and asserts:
// - parent[source] = source
// - parent[v] = u  =>  depth[v] = depth[u] + 1 (except for source)
// - parent[v] = u  => there is edge from u to v
// - all vertices reachable from source have a parent
bool BFSVerifier(const Graph &g, NodeID source, const pvector<NodeID> &parent) {
  pvector<int> depth(g.num_nodes(), -1);
  depth[source] = 0;
  vector<NodeID> to_visit;
  to_visit.reserve(g.num_nodes());
  to_visit.push_back(source);
  for (auto it = to_visit.begin(); it != to_visit.end(); it++) {
    NodeID u = *it;
    for (NodeID v : g.out_neigh(u)) {
      if (depth[v] == -1) {
        depth[v] = depth[u] + 1;
        to_visit.push_back(v);
      }
    }
  }
  for (NodeID u : g.vertices()) {
    if ((depth[u] != -1) && (parent[u] != -1)) {
      if (u == source) {
        if (!((parent[u] == u) && (depth[u] == 0))) {
          cout << "Source wrong" << endl;
          return false;
        }
        continue;
      }
      bool parent_found = false;
      for (NodeID v : g.in_neigh(u)) {
        if (v == parent[u]) {
          if (depth[v] != depth[u] - 1) {
            cout << "Wrong depths for " << u << " & " << v << endl;
            return false;
          }
          parent_found = true;
          break;
        }
      }
      if (!parent_found) {
        cout << "Couldn't find edge from " << parent[u] << " to " << u << endl;
        return false;
      }
    } else if (depth[u] != parent[u]) {
      cout << "Reachability mismatch" << endl;
      return false;
    }
  }
  return true;
}

int main(int argc, char *argv[]) {
  CLApp cli(argc, argv, "breadth-first search");
  if (!cli.ParseArgs())
    return -1;

  if (cli.num_threads() != -1) {
    omp_set_num_threads(cli.num_threads());
  }

  Builder b(cli);
  Graph g = b.MakeGraph();
  std::vector<NodeID> given_sources;
  if (cli.start_vertex() != -1) {
    // CLI has higher priority.
    given_sources.push_back(cli.start_vertex());
  } else {
    // Try to get the source from file.
    given_sources = SourceGenerator<Graph>::loadSource(cli.filename());
  }
  SourcePicker<Graph> sp(g, given_sources);
  auto BFSBound = [&sp](const Graph &g) { return DOBFS(g, sp.PickNext()); };
  SourcePicker<Graph> vsp(g, given_sources);
  auto VerifierBound = [&vsp](const Graph &g, const pvector<NodeID> &parent) {
    return BFSVerifier(g, vsp.PickNext(), parent);
  };
  BenchmarkKernel(cli, g, BFSBound, PrintBFSStats, VerifierBound);
  return 0;
}
