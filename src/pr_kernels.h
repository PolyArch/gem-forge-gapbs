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

__attribute__((noinline)) void pageRankPushAdjList(
#ifdef USE_ADJ_LIST_NO_PREV
    AdjGraphNoPrevT &graph,
#else
    AdjGraph &graph,
#endif
#ifdef SHUFFLE_NODES
    NodeID *nodes,
#endif
    ScoreT *scores, ScoreT *next_scores) {

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

__attribute__((noinline)) void
pageRankPushSingleAdjList(AdjGraphSingleAdjListT &graph,
#ifdef SHUFFLE_NODES
                          NodeID *nodes,
#endif
                          ScoreT *scores, ScoreT *next_scores) {

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

    auto adj_ptr = adj_list + n;

#pragma ss stream_name "gap.pr_push.adj.lhs.ld"
    auto *lhs = reinterpret_cast<NodeID *>(adj_ptr[0]);

#pragma ss stream_name "gap.pr_push.adj.rhs.ld"
    auto *rhs = reinterpret_cast<NodeID *>(adj_ptr[1]);

#pragma ss stream_name "gap.pr_push.adj.score.ld"
    ScoreT score = scores[n];

#pragma ss stream_name "gap.pr_push.adj.degree.ld"
    int64_t out_degree = degrees[n];

    ScoreT outgoing_contrib = score / out_degree;

    auto *rhs_node = AdjGraphSingleAdjListT::getNodePtr(rhs);
    auto rhs_offset = AdjGraphSingleAdjListT::getNodeOffset(rhs);

    if (lhs != rhs) {

      while (true) {
        auto *lhs_node = AdjGraphSingleAdjListT::getNodePtr(lhs);
        const auto local_rhs =
            lhs_node != rhs_node
                ? (lhs_node->edges + AdjGraphSingleAdjListT::EdgesPerNode)
                : rhs;
        const auto numEdges = local_rhs - lhs;

        int64_t j = 0;

#pragma clang loop unroll(disable) vectorize(disable) interleave(disable)
        while (true) {

#pragma ss stream_name "gap.pr_push.adj.out_v.ld"
          NodeID v = lhs_node->edges[j];

#pragma ss stream_name "gap.pr_push.adj.score.at"
          __atomic_fetch_fadd(next_scores + v, outgoing_contrib,
                              __ATOMIC_RELAXED);

          j++;
          if (j == numEdges) {
            break;
          }
        }

#pragma ss stream_name "gap.pr_push.adj.next_node.ld"
        auto next_node = lhs_node->next;

        lhs = next_node->edges;

        /**
         * I need to write this werid loop break condition to to distinguish lhs
         * and next_lhs for LoopBound.
         * TODO: Fix this in the compiler.
         */
        bool rhs_zero = next_node == rhs_node && rhs_offset == 0;
        bool should_break = rhs_zero || (lhs_node == rhs_node);
        if (should_break) {
          break;
        }
      }
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

__attribute__((noinline)) void pageRankPullUpdate(NodeID num_nodes,
                                                  ScoreT *scores,
                                                  ScoreT *out_contribs,
                                                  EdgeIndexT *out_neigh_index) {

#pragma omp parallel for schedule(static)                                      \
    firstprivate(num_nodes, scores, out_contribs, out_neigh_index)
  for (NodeID n = 0; n < num_nodes; n++) {

    auto *out_neigh_ptr = out_neigh_index + n;

#pragma ss stream_name "gap.pr_pull.update.score.ld"
    ScoreT score = scores[n];

#pragma ss stream_name "gap.pr_pull.update.out_begin.ld"
    EdgeIndexT out_begin = out_neigh_ptr[0];

#pragma ss stream_name "gap.pr_pull.update.out_end.ld"
    EdgeIndexT out_end = out_neigh_ptr[1];

    int64_t out_degree = out_end - out_begin;
    ScoreT contrib = score / out_degree;

#pragma ss stream_name "gap.pr_pull.update.out_contrib.st"
    out_contribs[n] = contrib;
  }
}

__attribute__((noinline)) ScoreT
pageRankPullCSR(NodeID num_nodes,
#ifdef SHUFFLE_NODES
                NodeID *nodes,
#endif
                ScoreT *scores, ScoreT *out_contribs, ScoreT base_score,
                ScoreT kDamp, EdgeIndexT *in_neigh_index, NodeID *in_edges) {

  ScoreT error = 0;

#ifdef SHUFFLE_NODES
#pragma omp parallel for OMP_SCHEDULE_TYPE reduction(+ : error) \
    firstprivate(num_nodes, nodes, scores, out_contribs, in_neigh_index, in_edges, base_score, kDamp)
  for (NodeID i = 0; i < num_nodes; i++) {

    NodeID u = nodes[i];

#else

#pragma omp parallel for OMP_SCHEDULE_TYPE reduction(+ : error) \
    firstprivate(num_nodes, scores, out_contribs, in_neigh_index, in_edges, base_score, kDamp)
  for (NodeID u = 0; u < num_nodes; u++) {

#endif

    auto *in_neigh_ptr = in_neigh_index + u;

#pragma ss stream_name "gap.pr_pull.rdc.in_begin.ld"
    EdgeIndexT in_begin = in_neigh_ptr[0];

#pragma ss stream_name "gap.pr_pull.rdc.in_end.ld"
    EdgeIndexT in_end = in_neigh_ptr[1];

    int64_t in_degree = in_end - in_begin;

#ifdef USE_EDGE_INDEX_OFFSET
    NodeID *in_ptr = in_edges + in_begin;
#else
    NodeID *in_ptr = in_begin;
#endif

    ScoreT incoming_total = 0;
    for (int64_t j = 0; j < in_degree; ++j) {

#pragma ss stream_name "gap.pr_pull.rdc.v.ld"
      NodeID v = in_ptr[j];

#pragma ss stream_name "gap.pr_pull.rdc.contrib.ld"
      ScoreT contrib = out_contribs[v];

      incoming_total += contrib;
    }

#pragma ss stream_name "gap.pr_pull.rdc.score.ld"
    ScoreT score = scores[u];

    ScoreT new_score = base_score + kDamp * incoming_total;

    error += new_score > score ? (new_score - score) : (score - new_score);

#pragma ss stream_name "gap.pr_pull.rdc.score.st"
    scores[u] = new_score;
  }

  return error;
}

__attribute__((noinline)) ScoreT pageRankPullAdjList(AdjGraph &graph,
#ifdef SHUFFLE_NODES
                                                     NodeID *nodes,
#endif
                                                     ScoreT *scores,
                                                     ScoreT *out_contribs,
                                                     ScoreT base_score,
                                                     ScoreT kDamp) {

  auto num_nodes = graph.N;
  auto adj_list = graph.adjList;

  ScoreT error = 0;

#ifdef SHUFFLE_NODES

#pragma omp parallel for OMP_SCHEDULE_TYPE reduction(+:error) firstprivate(                       \
    scores, out_contribs, nodes, num_nodes, adj_list, base_score, kDamp)
  for (int64_t i = 0; i < num_nodes; i++) {

#pragma ss stream_name "gap.pr_pull.rdc.node.ld"
    NodeID n = nodes[i];

#else

#pragma omp parallel for OMP_SCHEDULE_TYPE reduction(+:error) firstprivate(                       \
    scores, out_contribs, num_nodes, adj_list, base_score, kDamp)
  for (int64_t i = 0; i < num_nodes; i++) {

    int64_t n = i;

#endif // SHUFFLE_NODES

#pragma ss stream_name "gap.pr_pull.rdc.head.ld"
    auto *cur_node = adj_list[n];

    ScoreT income_total = 0;

#pragma clang loop unroll(disable) vectorize(disable) interleave(disable)
    while (cur_node) {

#pragma ss stream_name "gap.pr_pull.rdc.n_edges.ld"
      const auto numEdges = cur_node->numEdges;

      ScoreT income = 0;

      /**
       * It is guaranteed that numEdges > 0. And we need to write this as
       * a do while loop so that income_total always get the value of income.
       */
      int64_t j = 0;
#pragma clang loop unroll(disable) vectorize(disable) interleave(disable)
      do {

#pragma ss stream_name "gap.pr_pull.rdc.v.ld"
        NodeID v = cur_node->edges[j];

#pragma ss stream_name "gap.pr_pull.rdc.contrib.ld"
        ScoreT contrib = out_contribs[v];

        income += contrib;

        ++j;
      } while (j < numEdges);

#pragma ss stream_name "gap.pr_pull.rdc.next_node.ld"
      auto next_node = cur_node->next;

      income_total += income;

      cur_node = next_node;
    }

#pragma ss stream_name "gap.pr_pull.rdc.score.ld"
    ScoreT score = scores[n];

    ScoreT new_score = base_score + kDamp * income_total;

    error += new_score > score ? (new_score - score) : (score - new_score);

#pragma ss stream_name "gap.pr_pull.rdc.score.st"
    scores[n] = new_score;
  }

  return error;
}

__attribute__((noinline)) ScoreT
pageRankPullSingleAdjList(AdjGraphSingleAdjListT &graph,
#ifdef SHUFFLE_NODES
                          NodeID *nodes,
#endif
                          ScoreT *scores, ScoreT *out_contribs,
                          ScoreT base_score, ScoreT kDamp) {

  auto num_nodes = graph.N;
  auto adj_list = graph.adjList;

  ScoreT error = 0;

#ifdef SHUFFLE_NODES

#pragma omp parallel for OMP_SCHEDULE_TYPE reduction(+:error) firstprivate(                       \
    scores, out_contribs, nodes, num_nodes, adj_list, base_score, kDamp)
  for (int64_t i = 0; i < num_nodes; i++) {

#pragma ss stream_name "gap.pr_pull.rdc.node.ld"
    NodeID n = nodes[i];

#else

#pragma omp parallel for OMP_SCHEDULE_TYPE reduction(+:error) firstprivate(                       \
    scores, out_contribs, num_nodes, adj_list, base_score, kDamp)
  for (int64_t i = 0; i < num_nodes; i++) {

    int64_t n = i;

#endif // SHUFFLE_NODES

    auto adj_ptr = adj_list + n;

#pragma ss stream_name "gap.pr_pull.rdc.lhs.ld"
    auto *lhs = reinterpret_cast<NodeID *>(adj_ptr[0]);

#pragma ss stream_name "gap.pr_pull.rdc.rhs.ld"
    auto *rhs = reinterpret_cast<NodeID *>(adj_ptr[1]);

    auto *rhs_node = AdjGraphSingleAdjListT::getNodePtr(rhs);
    auto rhs_offset = AdjGraphSingleAdjListT::getNodeOffset(rhs);

    ScoreT income_total = 0;

    if (lhs != rhs) {

      while (true) {
        auto *lhs_node = AdjGraphSingleAdjListT::getNodePtr(lhs);
        const auto local_rhs =
            lhs_node != rhs_node
                ? (lhs_node->edges + AdjGraphSingleAdjListT::EdgesPerNode)
                : rhs;
        const auto numEdges = local_rhs - lhs;

        ScoreT income = 0;

        int64_t j = 0;

#pragma clang loop unroll(disable) vectorize(disable) interleave(disable)
        while (true) {

#pragma ss stream_name "gap.pr_pull.rdc.v.ld"
          NodeID v = lhs_node->edges[j];

#pragma ss stream_name "gap.pr_pull.rdc.contrib.ld"
          ScoreT contrib = out_contribs[v];

          income += contrib;

          j++;
          if (j == numEdges) {
            break;
          }
        }

        income_total += income;

#pragma ss stream_name "gap.pr_pull.rdc.next_node.ld"
        auto next_node = lhs_node->next;

        lhs = next_node->edges;

        /**
         * I need to write this werid loop break condition to to distinguish lhs
         * and next_lhs for LoopBound.
         * TODO: Fix this in the compiler.
         */
        bool rhs_zero = next_node == rhs_node && rhs_offset == 0;
        bool should_break = rhs_zero || (lhs_node == rhs_node);
        if (should_break) {
          break;
        }
      }
    }

#pragma ss stream_name "gap.pr_pull.rdc.score.ld"
    ScoreT score = scores[n];

    ScoreT new_score = base_score + kDamp * income_total;

    error += new_score > score ? (new_score - score) : (score - new_score);

#pragma ss stream_name "gap.pr_pull.rdc.score.st"
    scores[n] = new_score;
  }

  return error;
}

#endif