// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#ifndef GRAPH_H_
#define GRAPH_H_

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cstddef>
#include <iostream>
#include <limits>
#include <type_traits>

#include "pvector.h"
#include "util.h"

#ifdef GEM_FORGE
#include "gem5/m5ops.h"
#endif

/*
GAP Benchmark Suite
Class:  CSRGraph
Author: Scott Beamer

Simple container for graph in CSR format
 - Intended to be constructed by a Builder
 - To make weighted, set DestID_ template type to NodeWeight
 - MakeInverse parameter controls whether graph stores its inverse
*/

// Used to hold node & weight, with another node it makes a weighted edge
template <typename NodeID_, typename WeightT_> struct NodeWeight {
  WeightT_ w;
  NodeID_ v;
  NodeWeight() {}
  NodeWeight(NodeID_ v) : w(1), v(v) {}
  NodeWeight(NodeID_ v, WeightT_ w) : w(w), v(v) {}

  bool operator<(const NodeWeight &rhs) const {
    return v == rhs.v ? w < rhs.w : v < rhs.v;
  }

  // doesn't check WeightT_s, needed to remove duplicate edges
  bool operator==(const NodeWeight &rhs) const { return v == rhs.v; }

  // doesn't check WeightT_s, needed to remove self edges
  bool operator==(const NodeID_ &rhs) const { return v == rhs; }

  operator NodeID_() { return v; }
};

template <typename NodeID_, typename WeightT_>
std::ostream &operator<<(std::ostream &os,
                         const NodeWeight<NodeID_, WeightT_> &nw) {
  os << nw.v << " " << nw.w;
  return os;
}

template <typename NodeID_, typename WeightT_>
std::istream &operator>>(std::istream &is, NodeWeight<NodeID_, WeightT_> &nw) {
  is >> nw.v >> nw.w;
  return is;
}

// Syntatic sugar for an edge
template <typename SrcT, typename DstT = SrcT> struct EdgePair {
  SrcT u;
  DstT v;

  EdgePair() {}

  EdgePair(SrcT u, DstT v) : u(u), v(v) {}
};

// SG = serialized graph, these types are for writing graph to file
typedef int32_t SGID;
typedef EdgePair<SGID> SGEdge;
typedef int64_t SGOffset;

/**
 * Control the layout of edge list.
 * Default: CSR format.
 * Aligned: Each vertex's edge list start aligns with specified granularity.
 */

template <class NodeID_, class DestID_ = NodeID_, bool MakeInverse_ = true>
class CSRGraph {
  // Used for *non-negative* offsets within a neighborhood
  typedef std::make_unsigned<std::ptrdiff_t>::type OffsetT;

  // Used to access neighbors of vertex, basically sugar for iterators
  class Neighborhood {
    NodeID_ n_;
    DestID_ **g_index_;
    OffsetT start_offset_;

  public:
    Neighborhood(NodeID_ n, DestID_ **g_index, OffsetT start_offset)
        : n_(n), g_index_(g_index), start_offset_(0) {
      OffsetT max_offset = end() - begin();
      start_offset_ = std::min(start_offset, max_offset);
    }
    typedef DestID_ *iterator;
    iterator begin() { return g_index_[n_] + start_offset_; }
    iterator end() { return g_index_[n_ + 1]; }
  };

  void ReleaseResources() {
    if (out_index_ != nullptr)
      free(out_index_);
    if (out_index_offset_ != nullptr)
      free(out_index_offset_);
    if (out_neighbors_ != nullptr)
      free(out_neighbors_);
    if (directed_) {
      if (in_index_ != nullptr)
        free(in_index_);
      if (in_index_offset_ != nullptr)
        free(in_index_offset_);
      if (in_neighbors_ != nullptr)
        free(in_neighbors_);
    }
  }

public:
  CSRGraph() : directed_(false), num_nodes_(-1), num_edges_(-1) {}

  CSRGraph(int64_t num_nodes, DestID_ **index, DestID_ *neighs)
      : directed_(false), num_nodes_(num_nodes), out_index_(index),
        out_neighbors_(neighs), in_index_(index), in_neighbors_(neighs) {
    num_edges_ = (out_index_[num_nodes_] - out_index_[0]) / 2;
    this->out_index_offset_ = this->GenIndexOffset(out_index_, out_neighbors_);
    this->in_index_offset_ = this->out_index_offset_;
    this->initRealDegrees();
  }

  CSRGraph(int64_t num_nodes, DestID_ **out_index, DestID_ *out_neighs,
           DestID_ **in_index, DestID_ *in_neighs)
      : directed_(true), num_nodes_(num_nodes), out_index_(out_index),
        out_neighbors_(out_neighs), in_index_(in_index),
        in_neighbors_(in_neighs) {
    num_edges_ = out_index_[num_nodes_] - out_index_[0];
    this->out_index_offset_ = this->GenIndexOffset(out_index_, out_neighbors_);
    this->in_index_offset_ = this->GenIndexOffset(in_index_, in_neighbors_);
    this->initRealDegrees();
  }

  CSRGraph(CSRGraph &&other)
      : directed_(other.directed_), num_nodes_(other.num_nodes_),
        num_edges_(other.num_edges_), out_index_(other.out_index_),
        out_index_offset_(other.out_index_offset_),
        out_neighbors_(other.out_neighbors_), in_index_(other.in_index_),
        in_index_offset_(other.in_index_offset_),
        in_neighbors_(other.in_neighbors_),
        inter_part_edges(std::move(other.inter_part_edges)),
        real_in_degrees(other.real_in_degrees),
        real_out_degrees(other.real_out_degrees) {
    other.num_edges_ = -1;
    other.num_nodes_ = -1;
    other.out_index_ = nullptr;
    other.out_index_offset_ = nullptr;
    other.out_neighbors_ = nullptr;
    other.in_index_ = nullptr;
    other.in_index_offset_ = nullptr;
    other.in_neighbors_ = nullptr;
    other.real_in_degrees = nullptr;
    other.real_out_degrees = nullptr;
  }

  ~CSRGraph() { ReleaseResources(); }

  CSRGraph &operator=(CSRGraph &&other) {
    if (this != &other) {
      ReleaseResources();
      directed_ = other.directed_;
      num_edges_ = other.num_edges_;
      num_nodes_ = other.num_nodes_;
      out_index_ = other.out_index_;
      out_index_offset_ = other.out_index_offset_;
      out_neighbors_ = other.out_neighbors_;
      in_index_ = other.in_index_;
      in_index_offset_ = other.in_index_offset_;
      in_neighbors_ = other.in_neighbors_;
      other.num_edges_ = -1;
      other.num_nodes_ = -1;
      other.out_index_ = nullptr;
      other.out_index_offset_ = nullptr;
      other.out_neighbors_ = nullptr;
      other.in_index_ = nullptr;
      other.in_index_offset_ = nullptr;
      other.in_neighbors_ = nullptr;
    }
    return *this;
  }

  bool directed() const { return directed_; }

  int64_t num_nodes() const { return num_nodes_; }

  int64_t num_edges() const { return num_edges_; }

  int64_t num_edges_directed() const {
    return directed_ ? num_edges_ : 2 * num_edges_;
  }

  int64_t out_degree(NodeID_ v) const {
    return out_index_[v + 1] - out_index_[v];
  }

  int64_t in_degree(NodeID_ v) const {
    static_assert(MakeInverse_, "Graph inversion disabled but reading inverse");
    return in_index_[v + 1] - in_index_[v];
  }

  Neighborhood out_neigh(NodeID_ n, OffsetT start_offset = 0) const {
    return Neighborhood(n, out_index_, start_offset);
  }

  Neighborhood in_neigh(NodeID_ n, OffsetT start_offset = 0) const {
    static_assert(MakeInverse_, "Graph inversion disabled but reading inverse");
    return Neighborhood(n, in_index_, start_offset);
  }

  DestID_ **out_neigh_index() const { return out_index_; }
  NodeID_ *out_neigh_index_offset() const { return out_index_offset_; }

  DestID_ **in_neigh_index() const {
    static_assert(MakeInverse_, "Graph inversion disabled but reading inverse");
    return in_index_;
  }
  NodeID_ *in_neigh_index_offset() const {
    static_assert(MakeInverse_, "Graph inversion disabled but reading inverse");
    return in_index_offset_;
  }

  DestID_ *out_edges() const { return out_neighbors_; }
  DestID_ *in_edges() const {
    static_assert(MakeInverse_, "Graph inversion disabled but reading inverse");
    return in_neighbors_;
  }

  void PrintStats() const {
    std::cout << "Graph has " << num_nodes_ << " nodes ("
              << this->num_nodes() * sizeof(NodeID_) / 1024 << "kB) and "
              << num_edges_ << " ";
    if (!directed_)
      std::cout << "un";
    std::cout << "directed edges ("
              << this->num_edges_directed() * sizeof(DestID_) / 1024
              << "kB) for degree: ";
    std::cout << static_cast<float>(num_edges_) / static_cast<float>(num_nodes_)
              << std::endl;
  }

  void PrintTopology() const {
    for (NodeID_ i = 0; i < num_nodes_; i++) {
      std::cout << i << ": ";
      for (DestID_ j : out_neigh(i)) {
        std::cout << j << " ";
      }
      std::cout << std::endl;
    }
  }

  static DestID_ **GenIndex(const pvector<SGOffset> &offsets, DestID_ *neighs) {
    NodeID_ length = offsets.size();
    DestID_ **index = alignedAllocAndTouch<DestID_ *>(length);
#pragma omp parallel for
    for (NodeID_ n = 0; n < length; n++) {
      index[n] = neighs + offsets[n];
    }
    return index;
  }

  NodeID_ *GenIndexOffset(DestID_ **indexes, DestID_ *base) {
    NodeID_ *offsets = alignedAllocAndTouch<NodeID_>(num_nodes_ + 1);
#pragma omp parallel for
    for (NodeID_ i = 0; i < num_nodes_ + 1; ++i) {
      auto index = indexes[i];
      auto offset = index - base;
      assert(offset <= std::numeric_limits<NodeID_>::max());
      offsets[i] = offset;
    }
    printf("Done IndexOffset.\n");
    return offsets;
  }

  pvector<SGOffset> VertexOffsets(bool in_graph = false) const {
    pvector<SGOffset> offsets(num_nodes_ + 1);
    for (NodeID_ n = 0; n < num_nodes_ + 1; n++)
      if (in_graph)
        offsets[n] = in_index_[n] - in_index_[0];
      else
        offsets[n] = out_index_[n] - out_index_[0];
    return offsets;
  }

  Range<NodeID_> vertices() const { return Range<NodeID_>(num_nodes()); }

private:
  bool directed_;
  int64_t num_nodes_;
  int64_t num_edges_;
  DestID_ **out_index_ = nullptr;
  NodeID_ *out_index_offset_ = nullptr;
  DestID_ *out_neighbors_ = nullptr;
  DestID_ **in_index_ = nullptr;
  NodeID_ *in_index_offset_ = nullptr;
  DestID_ *in_neighbors_ = nullptr;

public:
  using Edge = EdgePair<NodeID_, DestID_>;
  using EdgeList = pvector<Edge>;
  using PartitionT = std::vector<int64_t>;
  void setPartitions(const PartitionT &node_part_sizes) {
    NodeID_ node_acc = 0;
    NodeID_ in_acc = 0;
    NodeID_ out_acc = 0;
    auto out_begin = this->out_index_[0];
    auto in_begin = this->out_index_[0];
    for (auto part_size : node_part_sizes) {
      node_acc += part_size;
      assert(node_acc > 0);
      if (node_acc > this->num_nodes_) {
        node_acc = this->num_nodes_;
      }
      auto out_end = this->out_index_[node_acc];
      out_acc = out_end - out_begin;
      auto in_end = this->in_index_[node_acc];
      in_acc = in_end - in_begin;
      // Adjust to bytes.
      this->node_parts.push_back(node_acc);
      this->in_edge_parts.push_back(in_acc);
      this->out_edge_parts.push_back(out_acc);
    }
    assert(node_acc == this->num_nodes_);
  }

  void setStreamNUCAPartition(void *ptr, const PartitionT &parts) const {
#ifdef GEM_FORGE
    auto parts_ptr = parts.data();
    auto parts_int = reinterpret_cast<int64_t>(parts_ptr);
    assert(parts_int > 0);
    m5_stream_nuca_set_property(ptr, STREAM_NUCA_REGION_PROPERTY_INTERLEAVE,
                                -parts_int);
#endif
  }

  PartitionT convertToSizePartition(const PartitionT &parts,
                                    int elemSize) const {
    PartitionT ret;
    for (int i = 0; i < parts.size(); ++i) {
      auto x = parts[i];
      if (i == 0) {
        ret.push_back(x / elemSize);
      } else {
        auto prev = parts[i - 1];
        ret.push_back((x - prev) / elemSize);
      }
    }
    return ret;
  }

  PartitionT getNodePartition() const {
    return this->convertToSizePartition(this->node_parts, sizeof(NodeID_));
  }

  bool hasPartition() const { return !this->node_parts.empty(); }

  // Data structures to remember the partition.
  PartitionT node_parts;
  PartitionT in_edge_parts;
  PartitionT out_edge_parts;

  /**
   * Information for cross partition edges.
   * Notice that we have to remember the real degree as nodes are duplicated
   * between partitions.
   */
  using InterPartEdge = EdgePair<NodeID_, NodeID_>;
  using InterPartEdgeList = pvector<InterPartEdge>;
  bool hasInterPartitionEdges() const {
    return !this->inter_part_edges.empty();
  }

  void initRealDegrees() {
    this->real_out_degrees = alignedAllocAndTouch<NodeID_>(this->num_nodes());
#pragma clang loop vectorize(disable)
    for (NodeID_ u = 0; u < this->num_nodes(); ++u) {
      this->real_out_degrees[u] = this->out_degree(u);
    }
    if (this->in_neighbors_ == this->out_neighbors_) {
      // Undirect graph.
      this->real_in_degrees = this->real_out_degrees;
    } else {
      this->real_in_degrees = alignedAllocAndTouch<NodeID_>(this->num_nodes());
#pragma clang loop vectorize(disable)
      for (NodeID_ u = 0; u < this->num_nodes(); ++u) {
        this->real_in_degrees[u] = this->in_degree(u);
      }
    }
  }

  void setInterPartitionEdges(InterPartEdgeList inter_part_edges) {
    this->inter_part_edges = std::move(inter_part_edges);
    /**
     * Compute real degrees degress.
     */
    bool directed = this->in_neighbors_ != this->out_neighbors_;
    for (const auto &edge : this->inter_part_edges) {
      this->real_in_degrees[edge.v] += this->real_in_degrees[edge.u];
      if (directed) {
        this->real_out_degrees[edge.v] += this->real_out_degrees[edge.u];
      }
    }
    for (const auto &edge : this->inter_part_edges) {
      this->real_in_degrees[edge.u] = this->real_in_degrees[edge.v];
      if (directed) {
        this->real_out_degrees[edge.u] = this->real_out_degrees[edge.v];
      }
    }
  }

  const InterPartEdgeList &getInterPartEdges() const {
    return this->inter_part_edges;
  }

  const NodeID_ *getRealInDegrees() const { return this->real_in_degrees; }
  const NodeID_ *getRealOutDegrees() const { return this->real_out_degrees; }

  // Inter partition edges are undirected.
  InterPartEdgeList inter_part_edges;
  NodeID_ *real_in_degrees = nullptr;
  NodeID_ *real_out_degrees = nullptr;

  // Specialize on set indirect alignment on edge types.
  template <typename EdgeT>
  void alignEdgesToVertices(EdgeT *const edges, EdgeT **const vertices) const {
#ifdef GEM_FORGE
    m5_stream_nuca_align(
        edges, vertices,
        m5_stream_nuca_encode_ind_align(offsetof(EdgeT, v), sizeof(EdgeT::v)));
#endif
  }
  template <>
  void alignEdgesToVertices<NodeID_>(NodeID_ *const edges,
                                     NodeID_ **const vertices) const {
#ifdef GEM_FORGE
    m5_stream_nuca_align(edges, vertices,
                         m5_stream_nuca_encode_ind_align(0, sizeof(NodeID_)));
#endif
  }

  void declareNUCARegions(bool enableNonUniformPartition) const {
#ifdef GEM_FORGE

    const auto num_nodes = this->num_nodes();
    const auto num_edges = this->num_edges_directed();

    m5_stream_nuca_region("gap.out_neigh_index", out_neigh_index(),
                          sizeof(*out_neigh_index()), num_nodes, 0, 0);

#define AlignVertexArray(name, ptr)                                            \
  m5_stream_nuca_region(name, ptr, sizeof(*ptr), num_nodes, 0, 0);             \
  m5_stream_nuca_align(ptr, this->out_neigh_index(), 0);

    AlignVertexArray("gap.out_neigh_index_offset", out_neigh_index_offset());

    AlignVertexArray("gap.real_out_degree", getRealOutDegrees());

    m5_stream_nuca_region("gap.out_edge", out_edges(), sizeof(*out_edges()),
                          num_edges, 0, 0);

    if (hasInterPartitionEdges()) {
      const auto &inter_part_edges = getInterPartEdges();
      m5_stream_nuca_region("gap.inter_part_edges", inter_part_edges.data(),
                            sizeof(inter_part_edges[0]),
                            inter_part_edges.size(), 0, 0);
    }

    if (enableNonUniformPartition) {
      // Inform the GemForge about the partitioned graph.
      setStreamNUCAPartition(out_neigh_index(), node_parts);
      setStreamNUCAPartition(out_edges(), out_edge_parts);
    } else {
      m5_stream_nuca_set_property(
          out_neigh_index(), STREAM_NUCA_REGION_PROPERTY_INTERLEAVE,
          roundUp(num_nodes / 64, 128 / sizeof(NodeID_)) *
              sizeof(*out_neigh_index()));
      alignEdgesToVertices(out_edges(), out_neigh_index());
      m5_stream_nuca_align(out_edges(), out_neigh_index(),
                           m5_stream_nuca_encode_csr_index());
    }

    if (in_neigh_index() != out_neigh_index()) {
      // This is directed graph.
      AlignVertexArray("gap.in_neigh_index", in_neigh_index());
      AlignVertexArray("gap.in_neigh_index_offset", in_neigh_index_offset());

      m5_stream_nuca_region("gap.in_edge", in_edges(), sizeof(*in_edges()),
                            num_edges, 0, 0);

      if (enableNonUniformPartition) {
        // Inform the GemForge about the partitioned graph.
        setStreamNUCAPartition(in_edges(), in_edge_parts);
      } else {
        alignEdgesToVertices(in_edges(), in_neigh_index());
        m5_stream_nuca_align(in_edges(), in_neigh_index(),
                             m5_stream_nuca_encode_csr_index());
      }
    }

#undef AlignVertexArray

#endif
  }
};

/**
 * @brief Template to perform CSR Push traverse on unweighted graph.
 */
template <bool PositiveDegree, typename U, typename NodeID, typename PushOp>
inline void csrIterate(U u, NodeID **neigh_index, PushOp pushOp) {

  auto neigh_ptr = neigh_index + u;

#pragma ss stream_name "gap.csr_push.begin.ld"
  auto begin = neigh_ptr[0];

#pragma ss stream_name "gap.csr_push.end.ld"
  auto end = neigh_ptr[1];

  int64_t degree = end - begin;

  // Short circuit the degree > 0 check if PositiveDegree is true.
  int64_t j = 0;
  if (PositiveDegree || degree > 0) {
    while (true) {

#pragma ss stream_name "gap.csr_push.v.ld"
      auto v = begin[j];

      pushOp(v);

      j++;
      if (j >= degree) {
        break;
      }
    }
  }
}

#endif // GRAPH_H_
