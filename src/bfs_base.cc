// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#include "bfs_kernels.h"

#include <iostream>
#include <vector>

#include <omp.h>

#include "bitmap.h"
#include "command_line.h"
#include "platform_atomics.h"
#include "sliding_queue.h"
#include "source_generator.h"

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
USE_SPATIAL_QUEUE: Use a spatially localized queue instead of default thread
    localized queue.
    So far we assume that there N banks, and vertexes are divided into
    N clusters with one cluster per bank.
*/

#ifdef USE_ADJ_LIST
#ifdef USE_ADJ_LIST_SINGLE_LIST
using AdjGraphT = AdjGraphSingleAdjListT;
#define BFSPushFunc bfsPushSingleAdjList
#define BFSPullFunc bfsPullSingleAdjList
#else
using AdjGraphT = AdjGraph;
#define BFSPushFunc bfsPushAdjList
#define BFSPullFunc bfsPullAdjList
#endif
#else
#define BFSPushFunc bfsPushCSR
#define BFSPullFunc bfsPullCSR
#endif

using namespace std;

pvector<NodeID> DOBFS(const Graph &g, NodeID source, int num_threads, int alpha,
                      int beta, int delta, int warm_cache = 2,
                      bool graph_partition = false) {
  PrintStep("Source", static_cast<int64_t>(source));
  Timer t;
  Bitmap curr(g.num_nodes());
  curr.reset();
  Bitmap front(g.num_nodes());
  front.reset();

  t.Start();
  pvector<NodeID> parent(g.num_nodes(), InitDepth);
  pvector<NodeID> next_parent(g.num_nodes(), InitDepth);
  t.Stop();
  PrintStep("i", t.Seconds());
  parent[source] = source;
  next_parent[source] = source;

#ifdef SHUFFLE_NODES
  // Shuffle the nodes.
  auto nodes_data = initShuffledNodes<NodeID>(g.num_nodes());
#endif

  const int num_banks = 64;
  const auto num_nodes = g.num_nodes();
  const auto __attribute__((unused)) num_nodes_per_bank =
      roundUp(num_nodes / num_banks, 128 / sizeof(NodeID));

#ifdef USE_SPATIAL_QUEUE
  const auto node_hash_mask = num_banks - 1;
  const auto node_hash_div = num_nodes_per_bank;
  SpatialQueue<NodeID> squeue(num_banks, 1 /* num_bins */, num_nodes_per_bank,
                              node_hash_div, node_hash_mask);
#endif

#ifdef USE_SPATIAL_FRONTIER
  // Another queue for frontier.
  SpatialQueue<NodeID> squeue2(num_banks, 1 /* num_bins */, num_nodes_per_bank,
                               node_hash_div, node_hash_mask);
  squeue2.enque(source, 0);
  {
    auto queue_idx = squeue2.getQueueIdx(source);
    printf("Source mask %d shift %d nodes_per_bank %d %d %d %d %d.\n",
           node_hash_mask, node_hash_div, num_nodes_per_bank, source, queue_idx,
           squeue2.size(0, 0), squeue2.size(queue_idx, 0));
  }
#else
  SlidingQueue<NodeID> queue(g.num_nodes());
  queue.push_back(source);
  queue.slide_window();
#endif

  const auto __attribute__((unused)) num_edges = g.num_edges_directed();
#ifdef USE_EDGE_INDEX_OFFSET
  EdgeIndexT *__attribute__((unused)) out_neigh_index =
      g.out_neigh_index_offset();
  EdgeIndexT *__attribute__((unused)) in_neigh_index =
      g.in_neigh_index_offset();
#else
  EdgeIndexT *__attribute__((unused)) out_neigh_index = g.out_neigh_index();
  EdgeIndexT *__attribute__((unused)) in_neigh_index = g.in_neigh_index();
#endif // USE_EDGE_INDEX_OFFSET

  NodeID *__attribute__((unused)) out_edges = g.out_edges();
  NodeID *__attribute__((unused)) in_edges = g.in_edges();

#ifdef GEM_FORGE
  {
    NodeID *parent_data = parent.data();
    NodeID *next_parent_data = next_parent.data();

#ifdef SHUFFLE_NODES
    m5_stream_nuca_region("gap.bfs.nodes", nodes_data, sizeof(nodes_data[0]),
                          num_nodes, 0, 0);
#endif

    m5_stream_nuca_region("gap.bfs.parent", parent_data, sizeof(NodeID),
                          num_nodes, 0, 0);
    m5_stream_nuca_region("gap.bfs.next_parent", next_parent_data,
                          sizeof(NodeID), num_nodes, 0, 0);
    m5_stream_nuca_align(next_parent_data, parent_data, 0);

    m5_stream_nuca_region("gap.bfs.out_neigh_index", out_neigh_index,
                          sizeof(EdgeIndexT), num_nodes, 0, 0);
    m5_stream_nuca_region("gap.bfs.out_edge", out_edges, sizeof(NodeID),
                          num_edges, 0, 0);
    m5_stream_nuca_align(out_neigh_index, parent_data, 0);

    if (graph_partition) {
      g.setStreamNUCAPartition(parent_data, g.node_parts);
      g.setStreamNUCAPartition(out_edges, g.out_edge_parts);
    } else {
      m5_stream_nuca_set_property(parent_data,
                                  STREAM_NUCA_REGION_PROPERTY_INTERLEAVE,
                                  num_nodes_per_bank * sizeof(NodeID));
      m5_stream_nuca_align(out_edges, parent_data,
                           m5_stream_nuca_encode_ind_align(0, sizeof(NodeID)));
      m5_stream_nuca_align(out_edges, out_neigh_index,
                           m5_stream_nuca_encode_csr_index());
    }

    if (in_neigh_index != out_neigh_index) {
      // This is a directed graph.
      m5_stream_nuca_region("gap.bfs.in_neigh_index", in_neigh_index,
                            sizeof(EdgeIndexT), num_nodes, 0, 0);
      m5_stream_nuca_region("gap.bfs.in_edge", in_edges, sizeof(NodeID),
                            num_edges, 0, 0);
      m5_stream_nuca_align(in_neigh_index, parent_data, 0);
      if (graph_partition) {
        g.setStreamNUCAPartition(in_edges, g.in_edge_parts);
      } else {
        m5_stream_nuca_align(
            in_edges, parent_data,
            m5_stream_nuca_encode_ind_align(0, sizeof(NodeID)));
        m5_stream_nuca_align(in_edges, in_neigh_index,
                             m5_stream_nuca_encode_csr_index());
      }
    }

#ifdef USE_SPATIAL_QUEUE
    m5_stream_nuca_region("gap.bfs_push.squeue", squeue.data, sizeof(NodeID),
                          num_nodes, 0, 0);
    m5_stream_nuca_region("gap.bfs_push.squeue_meta", squeue.meta,
                          sizeof(*squeue.meta), num_banks, 0, 0);
    m5_stream_nuca_align(squeue.data, parent_data, 0);
    m5_stream_nuca_align(squeue.meta, parent_data, num_nodes_per_bank);
    m5_stream_nuca_set_property(squeue.meta,
                                STREAM_NUCA_REGION_PROPERTY_INTERLEAVE,
                                sizeof(*squeue.meta));

#ifdef USE_SPATIAL_FRONTIER
    m5_stream_nuca_region("gap.bfs_push.squeue2", squeue2.data, sizeof(NodeID),
                          num_nodes, 0, 0);
    m5_stream_nuca_region("gap.bfs_push.squeue2_meta", squeue2.meta,
                          sizeof(*squeue2.meta), num_banks, 0, 0);
    m5_stream_nuca_align(squeue2.data, parent_data, 0);
    m5_stream_nuca_align(squeue2.meta, parent_data, num_nodes_per_bank);
    m5_stream_nuca_set_property(squeue2.meta,
                                STREAM_NUCA_REGION_PROPERTY_INTERLEAVE,
                                sizeof(*squeue2.meta));
#endif
#endif
    m5_stream_nuca_remap();
  }
#endif

#ifdef USE_ADJ_LIST

  printf("Start to build AdjListGraph, node %luB.\n",
         sizeof(AdjGraphT::AdjListNode));

#ifndef GEM_FORGE
  Timer adjBuildTimer;
  adjBuildTimer.Start();
#endif // GEM_FORGE

#ifdef USE_PUSH
  AdjGraphT adjGraph(num_threads, g.num_nodes(), g.out_neigh_index_offset(),
                     g.out_edges(), parent.data());
#else
  AdjGraphT adjGraph(num_threads, g.num_nodes(), g.in_neigh_index_offset(),
                     g.in_edges(), parent.data());
#endif

#ifndef GEM_FORGE
  adjBuildTimer.Stop();
  printf("AdjListGraph built %10.5lfs.\n", adjBuildTimer.Seconds());
#else
  printf("AdjListGraph built.\n");
#endif // GEM_FORGE

#endif // USE_ADJ_LIST

  gf_detail_sim_start();

  if (warm_cache > 0) {
    auto num_nodes = g.num_nodes();

#ifdef SHUFFLE_NODES
    gf_warm_array("nodes", nodes_data, num_nodes * sizeof(nodes_data[0]));
#endif

    NodeID *parent_data = parent.data();
    gf_warm_array("parent", parent_data, num_nodes * sizeof(parent_data[0]));

#ifdef USE_PULL
    gf_warm_array("next_parent", next_parent.data(),
                  num_nodes * sizeof(next_parent[0]));
#endif

#ifdef USE_SPATIAL_QUEUE
    gf_warm_array("squeue.data", squeue.data,
                  squeue.num_queues * squeue.queue_capacity *
                      sizeof(squeue.data[0]));
#ifdef USE_SPATIAL_FRONTIER
    gf_warm_array("squeue2.data", squeue2.data,
                  squeue2.num_queues * squeue2.queue_capacity *
                      sizeof(squeue2.data[0]));
#endif
#endif

#ifdef USE_ADJ_LIST
    adjGraph.warmAdjList();

#else

#ifdef USE_PUSH
    gf_warm_array("out_neigh_index", out_neigh_index,
                  num_nodes * sizeof(out_neigh_index[0]));
    if (warm_cache > 1) {
      gf_warm_array("out_edges", out_edges, num_edges * sizeof(out_edges[0]));
    }
#else
    gf_warm_array("in_neigh_index", in_neigh_index,
                  num_nodes * sizeof(in_neigh_index[0]));
    if (warm_cache > 1) {
      gf_warm_array("in_edges", in_edges, num_edges * sizeof(in_edges[0]));
    }
#endif

#endif

    std::cout << "Warm up done.\n";
  }

  startThreads(num_threads);

  gf_reset_stats();

  uint64_t iter = 0;

#if defined(USE_PUSH) && !defined(USE_PULL)

  /********************** Push Version ***********************************/

#ifdef USE_SPATIAL_FRONTIER
  // Push spatial frontier loop.
  auto *frontier = &squeue2;
  auto *next_frontier = &squeue;
  while (!frontier->empty()) {
#else
  // Push global queue loop.
  while (!queue.empty()) {
#endif

    gf_work_begin(0);

#ifndef GEM_FORGE
    t.Start();
#endif

    BFSPushFunc(
#ifdef USE_ADJ_LIST
        adjGraph,
#else
        g,
#endif
        parent,
#if defined(USE_SPATIAL_FRONTIER)
        *next_frontier, *frontier
#elif defined(USE_SPATIAL_QUEUE)
        squeue, queue
#else
      queue
#endif
    );

#ifdef USE_SPATIAL_FRONTIER
    // Swap two spatial queues.
    {
      auto *tmp = frontier;
      frontier = next_frontier;
      next_frontier = tmp;
    }
#else
    queue.slide_window();
#endif

    gf_work_end(0);

#ifndef GEM_FORGE
    t.Stop();
#endif

#ifndef GEM_FORGE
#ifndef USE_SPATIAL_FRONTIER
    printf("%6zu  td%11" PRId64 "  %10.5lfms %lu-%lu\n", iter, queue.size(),
           t.Millisecs(), queue.shared_out_start, queue.shared_in);
#else
    printf("%6zu.\n", iter);
#endif
#endif

    iter++;
  }

#elif !defined(USE_PUSH) && defined(USE_PULL)

  /********************** Pull Version ***********************************/

  int64_t awake_count = 1;
  while (awake_count > 0) {

    gf_work_begin(1);

#ifndef GEM_FORGE
    t.Start();
#endif

#ifdef USE_ADJ_LIST
    awake_count = BFSPullFunc(adjGraph, parent.data(), next_parent.data());
#else
    awake_count = bfsPullCSR(g,
#ifdef SHUFFLE_NODES
                             nodes_data,
#endif
                             parent.data(), next_parent.data());
#endif
    bfsPullUpdate(g.num_nodes(), parent.data(), next_parent.data());

    gf_work_end(1);

#ifndef GEM_FORGE
    t.Stop();
#endif

    iter++;
  }

#elif defined(USE_PUSH) && defined(USE_PULL)

  /********************** Push-Pull Version ******************************/

  int scout_count = g.out_degree(source);
  int total_visited = 1;

  int scout_threshold = alpha > 0 ? (num_edges * (alpha / 100.f)) : 1;
  int awake_threshold = beta > 0 ? (num_nodes * (beta / 100.f)) : 1;
  int visit_threshold = num_nodes * (delta / 100.f);

#ifdef USE_SPATIAL_FRONTIER
  // Push spatial frontier loop.
  auto *frontier = &squeue2;
  auto *next_frontier = &squeue;
  while (!frontier->empty()) {
#else
  // Push global queue loop.
  while (!queue.empty()) {
#endif

    // Check if we want to switch to pull.
    // ! Negative alpha/beta directly specifies the iterations.
    // ! Use pull for [-alpha, -beta) iterations.
    if ((alpha > 0 && scout_count > scout_threshold &&
         total_visited > visit_threshold) ||
        (alpha < 0 && iter >= -alpha && iter < -beta)) {

// Get awake_count as the queue size.
#ifdef USE_SPATIAL_FRONTIER
      int64_t awake_count = frontier->totalSize();
      frontier->clear();
#else
      int64_t awake_count = queue.size();
#endif
      int64_t old_awake_count;

      NodeID *parent_data = next_parent.data();
      NodeID *next_parent_data = parent.data();

      do {

        gf_work_begin(1);

#ifndef GEM_FORGE
        t.Start();
#endif

        // First sync the two parent arrays.
        bfsPullUpdate(g.num_nodes(), parent_data, next_parent_data);

        old_awake_count = awake_count;

        // Perform the pull-based BFS.
#ifdef USE_ADJ_LIST
        awake_count = BFSPullFunc(adjGraph, parent_data, next_parent_data);
#else
        awake_count = bfsPullCSR(g,
#ifdef SHUFFLE_NODES
                                 nodes_data,
#endif
                                 parent_data, next_parent_data);
#endif

        gf_work_end(1);

        total_visited += awake_count;

#ifndef GEM_FORGE
        t.Stop();
#endif

#ifndef GEM_FORGE
        printf("Iter %6zu Pull Awake %8ld(%.2f) Visited %8d(%.2f).\n", iter,
               awake_count, static_cast<float>(awake_count) / num_nodes,
               total_visited, static_cast<float>(total_visited) / num_nodes);
#endif

        iter++;
      } while ((beta > 0 && ((awake_count >= old_awake_count) ||
                             (awake_count > awake_threshold))) ||
               (beta < 0 && iter < -beta && awake_count > 0));

      if (awake_count == 0) {
        // we are done.
        break;
      }

      gf_work_begin(2);
      // Convert back to frontier by comparing parent and next_parent.
      bfsPullToFrontier(g.num_nodes(), parent_data, next_parent_data,
#if defined(USE_SPATIAL_FRONTIER)
                        *frontier
#else
                        queue
#endif
      );

#ifndef USE_SPATIAL_FRONTIER
      queue.slide_window();
#endif
      gf_work_end(2);
      scout_count = 1;

    } else {

      gf_work_begin(0);

#ifndef GEM_FORGE
      t.Start();
#endif

      scout_count = BFSPushFunc(
#ifdef USE_ADJ_LIST
          adjGraph,
#else
          g,
#endif
          parent,
#if defined(USE_SPATIAL_FRONTIER)
          *next_frontier, *frontier
#elif defined(USE_SPATIAL_QUEUE)
          squeue, queue
#else
        queue
#endif
      );

#ifdef USE_SPATIAL_FRONTIER
      // Swap two spatial queues.
      {
        auto *tmp = frontier;
        frontier = next_frontier;
        next_frontier = tmp;
        total_visited += frontier->totalSize();
      }
#else
      queue.slide_window();
      total_visited += queue.size();
#endif

      gf_work_end(0);

#ifndef GEM_FORGE
      t.Stop();
#endif

#ifndef GEM_FORGE
#ifndef USE_SPATIAL_FRONTIER
      printf("Iter %6zu  td%11" PRId64 "  %10.5lfms %lu-%lu\n", iter, queue.size(),
             t.Millisecs(), queue.shared_out_start, queue.shared_in);
#else
      printf("Iter %6zu Push Scout %8d(%.2f) Visited %8d(%.2f).\n", iter,
             scout_count, static_cast<float>(scout_count) / num_edges,
             total_visited, static_cast<float>(total_visited) / num_nodes);
#endif
#endif

      iter++;
    }
  }

#else

#error "No Push/Pull specified?"

#endif

  gf_detail_sim_end();
#ifdef GEM_FORGE
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
  CLBFS cli(argc, argv, "breadth-first search");
  if (!cli.ParseArgs())
    return -1;

  if (cli.num_threads() != -1) {
    // Start with one thread.
    omp_set_num_threads(1);
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
    int alpha = cli.alpha();
    int beta = cli.beta();
    int delta = cli.delta();
    int warm_cache = cli.warm_cache();
    int num_threads = cli.num_threads();
    bool graph_partition = cli.graph_partition();
    return DOBFS(g, sp.PickNext(), num_threads, alpha, beta, delta, warm_cache,
                 graph_partition);
  };
  SourcePicker<Graph> vsp(g, given_sources);
  auto VerifierBound = [&vsp](const Graph &g, const pvector<NodeID> &parent) {
    return BFSVerifier(g, vsp.PickNext(), parent);
  };
  BenchmarkKernel(cli, g, BFSBound, PrintBFSStats, VerifierBound);
  return 0;
}
