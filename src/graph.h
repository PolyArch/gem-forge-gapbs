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
  }

  CSRGraph(int64_t num_nodes, DestID_ **out_index, DestID_ *out_neighs,
           DestID_ **in_index, DestID_ *in_neighs)
      : directed_(true), num_nodes_(num_nodes), out_index_(out_index),
        out_neighbors_(out_neighs), in_index_(in_index),
        in_neighbors_(in_neighs) {
    num_edges_ = out_index_[num_nodes_] - out_index_[0];
    this->out_index_offset_ = this->GenIndexOffset(out_index_, out_neighbors_);
    this->in_index_offset_ = this->GenIndexOffset(in_index_, in_neighbors_);
  }

  CSRGraph(CSRGraph &&other)
      : directed_(other.directed_), num_nodes_(other.num_nodes_),
        num_edges_(other.num_edges_), out_index_(other.out_index_),
        out_index_offset_(other.out_index_offset_),
        out_neighbors_(other.out_neighbors_), in_index_(other.in_index_),
        in_index_offset_(other.in_index_offset_),
        in_neighbors_(other.in_neighbors_) {
    other.num_edges_ = -1;
    other.num_nodes_ = -1;
    other.out_index_ = nullptr;
    other.out_index_offset_ = nullptr;
    other.out_neighbors_ = nullptr;
    other.in_index_ = nullptr;
    other.in_index_offset_ = nullptr;
    other.in_neighbors_ = nullptr;
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
      this->node_parts.push_back(node_acc * sizeof(NodeID_));
      this->in_edge_parts.push_back(in_acc * sizeof(DestID_));
      this->out_edge_parts.push_back(out_acc * sizeof(DestID_));
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

  bool hasPartition() const {
    return !this->node_parts.empty();
  }

  // Data structures to remember the partition.
  PartitionT node_parts;
  PartitionT in_edge_parts;
  PartitionT out_edge_parts;
};

#endif // GRAPH_H_
