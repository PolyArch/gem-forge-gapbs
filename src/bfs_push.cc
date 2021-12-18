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
#include "sized_array.h"
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

Control flags:
USE_EDGE_INDEX_OFFSET: Use index_offset instead of the pointer index.
*/

using namespace std;

#ifdef USE_EDGE_INDEX_OFFSET
#define EdgeIndexT NodeID
#else
#define EdgeIndexT NodeID *
#endif // USE_EDGE_INDEX_OFFSET

const NodeID InitParentId = -1;

void TDStep(const Graph &g, pvector<NodeID> &parent,
            SlidingQueue<NodeID> &queue) {
  auto parent_v = parent.data();
  auto queue_v = queue.begin();
  auto queue_e = queue.end();
  int64_t queue_size = queue_e - queue_v;

  NodeID *out_edges = g.out_edges();
#ifdef USE_EDGE_INDEX_OFFSET
  EdgeIndexT *out_neigh_index = g.out_neigh_index_offset();
#else
  EdgeIndexT *out_neigh_index = g.out_neigh_index();
#endif

  /**
   * Helper function for debug perpose.
   */
  // {
  //   static int iter = 0;
  //   uint64_t totalIdx = 0;
  //   for (auto iter = queue_v; iter < queue_e; iter++) {
  //     NodeID u = *iter;
  //     totalIdx += u;
  //   }
  //   printf(" - TD %d Size %ld Total %lu.\n", iter, queue_e - queue_v,
  //   totalIdx); iter++;
  // }
#pragma omp parallel firstprivate(queue_v, queue_size, parent_v,               \
                                  out_neigh_index, out_edges)
  {
    SizedArray<NodeID> lqueue(g.num_nodes());

#pragma omp for schedule(static)
    for (int64_t i = 0; i < queue_size; ++i) {

#pragma ss stream_name "gap.bfs_push.u.ld"
      NodeID u = queue_v[i];

      // Explicit write this out to aviod u + 1.
      EdgeIndexT *out_neigh_ptr = out_neigh_index + u;

#pragma ss stream_name "gap.bfs_push.out_begin.ld"
      EdgeIndexT out_begin = out_neigh_ptr[0];

#pragma ss stream_name "gap.bfs_push.out_end.ld"
      EdgeIndexT out_end = out_neigh_ptr[1];

#ifdef USE_EDGE_INDEX_OFFSET
      NodeID *out_ptr = out_edges + out_begin;
#else
      NodeID *out_ptr = out_begin;
#endif

      const auto out_degree = out_end - out_begin;

      for (int64_t j = 0; j < out_degree; ++j) {

#pragma ss stream_name "gap.bfs_push.out_v.ld"
        NodeID v = out_ptr[j];

        NodeID temp = InitParentId;

#pragma ss stream_name "gap.bfs_push.swap.at"
        bool swapped = __atomic_compare_exchange_n(
            parent_v + v, &temp, u, false /* weak */, __ATOMIC_RELAXED,
            __ATOMIC_RELAXED);
        if (swapped) {
          lqueue.push_back(v);
        }
      }
    }
    // Flush into the global queue.
    queue.append(lqueue.begin(), lqueue.size());
  }

  return;
}

pvector<NodeID> InitParent(const Graph &g) {
  pvector<NodeID> parent(g.num_nodes());
#pragma omp parallel for
  for (NodeID n = 0; n < g.num_nodes(); n++) {
    parent[n] = -1;
  }
  return parent;
}

pvector<NodeID> DOBFS(const Graph &g, NodeID source, int alpha = 15,
                      int beta = 18, int warm_cache = 2) {
  PrintStep("Source", static_cast<int64_t>(source));
  Timer t;
  t.Start();
  pvector<NodeID> parent = InitParent(g);
  t.Stop();
  PrintStep("i", t.Seconds());
  parent[source] = source;
  SlidingQueue<NodeID> queue(g.num_nodes());
  queue.push_back(source);
  queue.slide_window();
  Bitmap curr(g.num_nodes());
  curr.reset();
  Bitmap front(g.num_nodes());
  front.reset();

#ifdef GEM_FORGE
  {
    const auto num_nodes = g.num_nodes();
    const auto num_edges = g.num_edges();
#ifdef USE_EDGE_INDEX_OFFSET
    EdgeIndexT *out_neigh_index = g.out_neigh_index_offset();
#else
    EdgeIndexT *out_neigh_index = g.out_neigh_index();
#endif // USE_EDGE_INDEX_OFFSET
    NodeID *out_edges = g.out_edges();
    NodeID *parent_data = parent.data();
    m5_stream_nuca_region("gap.bfs_push.parent", parent_data, sizeof(NodeID),
                          num_nodes);
    m5_stream_nuca_region("gap.bfs_push.out_neigh_index", out_neigh_index,
                          sizeof(EdgeIndexT), num_nodes);
    m5_stream_nuca_region("gap.bfs_push.out_edge", out_edges, sizeof(NodeID),
                          num_edges);
    m5_stream_nuca_align(out_neigh_index, parent_data, 0);
    m5_stream_nuca_align(out_edges, parent_data,
                         STREAM_NUCA_IND_ALIGN_EVERY_ELEMENT);
    m5_stream_nuca_remap();
  }
#endif

#ifdef GEM_FORGE
  m5_detail_sim_start();
#endif

  if (warm_cache > 0) {
    auto num_nodes = g.num_nodes();
    auto num_edges = g.num_edges();
#ifdef USE_EDGE_INDEX_OFFSET
    EdgeIndexT *out_neigh_index = g.out_neigh_index_offset();
#else
    EdgeIndexT *out_neigh_index = g.out_neigh_index();
#endif // USE_EDGE_INDEX_OFFSET
    NodeID *out_edges = g.out_edges();
    NodeID *parent_data = parent.data();
    gf_warm_array("out_neigh_index", out_neigh_index,
                  num_nodes * sizeof(out_neigh_index[0]));
    gf_warm_array("parent", parent_data, num_nodes * sizeof(parent_data[0]));
    if (warm_cache > 1) {
      gf_warm_array("out_edges", out_edges, num_edges * sizeof(out_edges[0]));
    }
    std::cout << "Warm up done.\n";
  }

#ifdef GEM_FORGE
  m5_reset_stats(0, 0);
#endif

  while (!queue.empty()) {
#ifdef GEM_FORGE
    m5_work_begin(0, 0);
#else
    t.Start();
#endif
    TDStep(g, parent, queue);
    queue.slide_window();
#ifdef GEM_FORGE
    m5_work_end(0, 0);
#else
    t.Stop();
    PrintStep("td", t.Seconds(), queue.size());
#endif
  }

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
  auto BFSBound = [&sp, &cli](const Graph &g) {
    int alpha = 15;
    int beta = 18;
    int warm_cache = cli.warm_cache();
    return DOBFS(g, sp.PickNext(), alpha, beta, warm_cache);
  };
  SourcePicker<Graph> vsp(g, given_sources);
  auto VerifierBound = [&vsp](const Graph &g, const pvector<NodeID> &parent) {
    return BFSVerifier(g, vsp.PickNext(), parent);
  };
  BenchmarkKernel(cli, g, BFSBound, PrintBFSStats, VerifierBound);
  return 0;
}
