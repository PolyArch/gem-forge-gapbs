#ifndef PR_KERNELS_H
#define PR_KERNELS_H

#include "benchmark.h"

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

#ifdef USE_EDGE_INDEX_OFFSET
#define EdgeIndexT NodeID
#else
#define EdgeIndexT NodeID *
#endif // USE_EDGE_INDEX_OFFSET

const float kDamp = 0.85;

__attribute__((noinline)) void pageRankPushCSR(NodeID num_nodes,
#ifdef SHUFFLE_NODES
                                               NodeID *nodes,
#endif
                                               ScoreT *scores,
                                               ScoreT *next_scores,
                                               EdgeIndexT *out_neigh_index,
                                               NodeID *out_edges) {

#ifdef SHUFFLE_NODES

#pragma omp parallel for OMP_SCHEDULE_TYPE firstprivate(                       \
    scores, out_neigh_index, out_edges, next_scores, nodes)
  for (int64_t i = 0; i < num_nodes; i++) {

#pragma ss stream_name "gap.pr_push.atomic.node.ld"
    NodeID n = nodes[i];

#else

#pragma omp parallel for OMP_SCHEDULE_TYPE firstprivate(                       \
    scores, out_neigh_index, out_edges, next_scores)
  for (int64_t i = 0; i < num_nodes; i++) {

    int64_t n = i;

#endif // SHUFFLE_NODES

    EdgeIndexT *out_neigh_ptr = out_neigh_index + n;

#pragma ss stream_name "gap.pr_push.atomic.score.ld"
    ScoreT score = scores[n];

#pragma ss stream_name "gap.pr_push.atomic.out_begin.ld"
    EdgeIndexT out_begin = out_neigh_ptr[0];

#pragma ss stream_name "gap.pr_push.atomic.out_end.ld"
    EdgeIndexT out_end = out_neigh_ptr[1];

#ifdef USE_EDGE_INDEX_OFFSET
    NodeID *out_ptr = out_edges + out_begin;
#else
    NodeID *out_ptr = out_begin;
#endif

    int64_t out_degree = out_end - out_begin;
    ScoreT outgoing_contrib = score / out_degree;
    for (int64_t j = 0; j < out_degree; ++j) {

#pragma ss stream_name "gap.pr_push.atomic.out_v.ld"
      NodeID v = out_ptr[j];

#pragma ss stream_name "gap.pr_push.atomic.next.at"
      __atomic_fetch_fadd(next_scores + v, outgoing_contrib, __ATOMIC_RELAXED);
    }
  }
}

__attribute__((noinline)) void pageRankPushAdjList(AdjGraph &graph,
#ifdef SHUFFLE_NODES
                                                   NodeID *nodes,
#endif
                                                   ScoreT *scores,
                                                   ScoreT *next_scores) {

  auto num_nodes = graph.N;
  auto adj_list = graph.adjList;
  auto degrees = graph.degrees;

#ifdef SHUFFLE_NODES

#pragma omp parallel for OMP_SCHEDULE_TYPE firstprivate(                       \
    scores, next_scores, nodes, adj_list, degrees)
  for (int64_t i = 0; i < num_nodes; i++) {

#pragma ss stream_name "gap.pr_push.atomic.node.ld"
    NodeID n = nodes[i];

#else

#pragma omp parallel for OMP_SCHEDULE_TYPE firstprivate(scores, next_scores,   \
                                                        adj_list, degrees)
  for (int64_t i = 0; i < num_nodes; i++) {

    int64_t n = i;

#endif // SHUFFLE_NODES

#pragma ss stream_name "gap.pr_push.adj.node.ld"
    auto *cur_node = adj_list[n];

#pragma ss stream_name "gap.pr_push.adj.score.ld"
    ScoreT score = scores[n];

#pragma ss stream_name "gap.pr_push.adj.degree.ld"
    int64_t out_degree = degrees[n];

    ScoreT outgoing_contrib = score / out_degree;

#pragma clang loop unroll(disable) vectorize(disable) interleave(disable)
    while (cur_node) {

#pragma ss stream_name "gap.pr_push.adj.n_edges.ld"
      const auto numEdges = cur_node->numEdges;

#pragma clang loop unroll(disable) vectorize(disable) interleave(disable)
      for (int64_t j = 0; j < numEdges; ++j) {

#pragma ss stream_name "gap.pr_push.adj.out_v.ld"
        NodeID v = cur_node->edges[j];

#pragma ss stream_name "gap.pr_push.adj.score.at"
        __atomic_fetch_fadd(next_scores + v, outgoing_contrib,
                            __ATOMIC_RELAXED);
      }

#pragma ss stream_name "gap.pr_push.adj.next_node.ld"
      auto next_node = cur_node->next;

      cur_node = next_node;
    }
  }
}

__attribute__((noinline)) ScoreT
pageRankPushUpdate(NodeID num_nodes, ScoreT *scores, ScoreT *next_scores,
                   ScoreT base_score, ScoreT kDamp) {

  ScoreT error = 0;

#pragma omp parallel for reduction(+ : error) schedule(static) \
  firstprivate(num_nodes, scores, next_scores, base_score, kDamp)
  for (NodeID n = 0; n < num_nodes; n++) {

#pragma ss stream_name "gap.pr_push.update.score.ld"
    ScoreT score = scores[n];

#pragma ss stream_name "gap.pr_push.update.next.ld"
    ScoreT next = next_scores[n];

    ScoreT next_score = base_score + kDamp * next;
    error += next_score > score ? (next_score - score) : (score - next_score);

#pragma ss stream_name "gap.pr_push.update.score.st"
    scores[n] = next_score;

#pragma ss stream_name "gap.pr_push.update.next.st"
    next_scores[n] = 0;
  }

  return error;
}

#endif