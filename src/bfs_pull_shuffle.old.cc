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

#ifdef USE_EDGE_INDEX_OFFSET
#define EdgeIndexT NodeID
#else
#define EdgeIndexT NodeID *
#endif // USE_EDGE_INDEX_OFFSET

int64_t BUStep(const Graph &g, const pvector<NodeID> &nodes,
               pvector<NodeID> &parent, pvector<NodeID> &next_parent) {
  int64_t awake_count = 0;

#ifdef USE_EDGE_INDEX_OFFSET
  EdgeIndexT *in_neigh_index = g.in_neigh_index_offset();
#else
  EdgeIndexT *in_neigh_index = g.in_neigh_index();
#endif // USE_EDGE_INDEX_OFFSET

  auto *nodes_v = nodes.data();
  auto *parent_v = parent.data();
  auto *next_parent_v = next_parent.data();
  auto *in_edges = g.in_edges();

#pragma omp parallel for schedule(static) reduction(+ : awake_count) \
  firstprivate(in_neigh_index, in_edges, nodes_v, parent_v, next_parent_v)
  for (int64_t x = 0; x < g.num_nodes(); ++x) {

#pragma ss stream_name "gap.bfs_pull.node.ld"
    NodeID u = nodes_v[x];

    // Explicit write this to avoid u + 1 and misplaced stream_step.
    EdgeIndexT *in_neigh_ptr = in_neigh_index + u;

#pragma ss stream_name "gap.bfs_pull.in_begin.ld"
    EdgeIndexT in_begin = in_neigh_ptr[0];

#pragma ss stream_name "gap.bfs_pull.in_end.ld"
    EdgeIndexT in_end = in_neigh_ptr[1];

#ifdef USE_EDGE_INDEX_OFFSET
    NodeID *in_ptr = in_edges + in_begin;
#else
    NodeID *in_ptr = in_begin;
#endif

    const auto in_degree = in_end - in_begin;

#pragma ss stream_name "gap.bfs_pull.parent_u.ld"
    auto parent_u = parent_v[u];

    if (parent_u < 0 && in_degree > 0) {
      // Better to reduce from zero.
      NodeID p = 0;
      for (int64_t i = 0; i < in_degree; ++i) {

#pragma ss stream_name "gap.bfs_pull.in_v.ld"
        NodeID v = in_ptr[i];

#pragma ss stream_name "gap.bfs_pull.parent_v.ld"
        NodeID vParent = parent_v[v];

        p = (vParent > -1) ? (v + 1) : p;
      }

      if (p != 0) {
#pragma ss stream_name "gap.bfs_pull.next_parent.st"
        next_parent_v[u] = p - 1;
        awake_count++;
      }
    }
  }

// Copy next_parent into parent.
#pragma omp parallel for schedule(static) firstprivate(parent_v, next_parent_v)
  for (NodeID u = 0; u < g.num_nodes(); u++) {
#pragma ss stream_name "gap.bfs_pull.copy.next_parent.ld"
    auto next_parent = next_parent_v[u];

#pragma ss stream_name "gap.bfs_pull.copy.parent.st"
    parent_v[u] = next_parent;
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

pvector<NodeID> DOBFS(const Graph &g, NodeID source, int warm_cache,
                      int alpha = 15, int beta = 18) {
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

  const auto num_nodes = g.num_nodes();
  const auto num_edges = g.num_edges_directed();
  NodeID *in_edges = g.in_edges();
  NodeID *nodes_data = nodes.data();
  NodeID *parent_data = parent.data();
  NodeID *next_parent_data = next_parent.data();

#ifdef USE_EDGE_INDEX_OFFSET
  EdgeIndexT *in_neigh_index = g.in_neigh_index_offset();
#else
  EdgeIndexT *in_neigh_index = g.in_neigh_index();
#endif // USE_EDGE_INDEX_OFFSET

#ifdef GEM_FORGE
  m5_stream_nuca_region("gap.bfs_pull.node", nodes_data, sizeof(NodeID),
                        num_nodes, 0, 0);
  m5_stream_nuca_region("gap.bfs_pull.parent", parent_data, sizeof(NodeID),
                        num_nodes, 0, 0);
  m5_stream_nuca_region("gap.bfs_pull.next_degree", next_parent_data,
                        sizeof(NodeID), num_nodes, 0, 0);
  m5_stream_nuca_region("gap.bfs_pull.in_neigh_index", in_neigh_index,
                        sizeof(EdgeIndexT), num_nodes, 0, 0);
  m5_stream_nuca_region("gap.bfs_pull.in_edges", in_edges, sizeof(NodeID),
                        num_edges, 0, 0);
  m5_stream_nuca_align(parent_data, next_parent_data, 0);
  m5_stream_nuca_align(in_neigh_index, next_parent_data, 0);
  m5_stream_nuca_align(in_edges, next_parent_data,
                       m5_stream_nuca_encode_ind_align(0, sizeof(NodeID)));
  m5_stream_nuca_remap();
#endif

#ifdef GEM_FORGE
  m5_detail_sim_start();
  if (warm_cache > 0) {
    gf_warm_array("nodes", nodes_data, num_nodes * sizeof(nodes_data[0]));
    gf_warm_array("parents", parent_data, num_nodes * sizeof(parent_data[0]));
    gf_warm_array("next_parents", next_parent_data,
                  num_nodes * sizeof(next_parent_data[0]));
    gf_warm_array("in_neigh_index", in_neigh_index,
                  num_nodes * sizeof(in_neigh_index[0]));
    if (warm_cache > 1) {
      gf_warm_array("in_edges", in_edges, num_edges * sizeof(in_edges[0]));
    }
  }
  std::cout << "Warm up done.\n";
  m5_reset_stats(0, 0);
#endif

  PrintStep("e", t.Seconds());
  int64_t awake_count = 1;
  uint64_t iter = 0;
  while (awake_count > 0) {
#ifdef GEM_FORGE
    m5_work_begin(0, 0);
#endif
    t.Start();
    awake_count = BUStep(g, nodes, parent, next_parent);
#ifdef GEM_FORGE
    m5_work_end(0, 0);
#endif
    t.Stop();
    printf("%6zu bu%11" PRId64 "  %10.5lfms\n", iter, awake_count,
           t.Millisecs());
    iter++;
  }
  PrintStep("c", t.Seconds());

#ifdef GEM_FORGE
  m5_detail_sim_end();
  exit(0);
#endif

#pragma omp parallel for
  for (NodeID n = 0; n < g.num_nodes(); n++) {
    if (parent[n] < -1)
      parent[n] = -1;
  }
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
  auto BFSBound = [&sp, &cli](const Graph &g) {
    return DOBFS(g, sp.PickNext(), cli.warm_cache());
  };
  SourcePicker<Graph> vsp(g, given_sources);
  auto VerifierBound = [&vsp](const Graph &g, const pvector<NodeID> &parent) {
    return BFSVerifier(g, vsp.PickNext(), parent);
  };
  BenchmarkKernel(cli, g, BFSBound, PrintBFSStats, VerifierBound);
  return 0;
}
