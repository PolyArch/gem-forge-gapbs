// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#ifndef CLUSTER_H_
#define CLUSTER_H_

#include <algorithm>
#include <cassert>
#include <deque>
#include <fstream>
#include <iostream>
#include <list>
#include <set>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "graph.h"

/*
GAP Benchmark Suite
Class:  Cluster
Author: Zhengrong Wang

Given a graph, partition the graph with bounded depth-first search.
*/

template <typename NodeID_, typename DestID_ = NodeID_> class ClusterBase {
public:
  using ClusterNodes = std::vector<NodeID_>;
  using Clusters = std::list<ClusterNodes>;

  CSRGraph<NodeID_, DestID_> ClusterByBoundedDFS(CSRGraph<NodeID_, DestID_> &g,
                                                 int bounded_depth,
                                                 uint64_t cluster_size) {

    std::unordered_map<NodeID_, NodeID_> clustered_nodes;
    std::vector<NodeID_> reverse_map;

    std::vector<std::vector<NodeID_>> clusters;

    auto num_nodes = g.num_nodes();

    while (reverse_map.size() < static_cast<uint64_t>(num_nodes)) {

      std::deque<std::pair<NodeID_, int>> stack;
      for (auto i = 0; i < num_nodes; ++i) {
        if (!clustered_nodes.count(i)) {
          stack.emplace_front(i, 0);
          break;
        }
      }

      while (!stack.empty()) {
        auto v = stack.front().first;
        auto depth = stack.front().second;
        stack.pop_front();

        if (clustered_nodes.count(v)) {
          // Visited.
          continue;
        }

        if (depth == 0) {
          clusters.emplace_back();
        }

        clustered_nodes.emplace(v, clustered_nodes.size());
        reverse_map.push_back(v);
        clusters.back().push_back(v);

        for (auto u : g.out_neigh(v)) {
          if (!clustered_nodes.count(u)) {
            if (depth + 1 < bounded_depth) {
              stack.emplace_front(u, depth + 1);
            } else {
              stack.emplace_back(u, 0);
            }
          }
        }
      }
    }

    this->analyzeCluster(g, bounded_depth, clusters);

    assert(!g.directed());

    return this->ReorderGraph(g, reverse_map, clustered_nodes);
  }

  void analyzeCluster(CSRGraph<NodeID_, DestID_> &g, int depth,
                      const std::vector<std::vector<NodeID_>> &clusters) {

    uint64_t avgClusterSize = g.num_nodes() / clusters.size();

    uint64_t sum = 0;
    for (const auto &c : clusters) {
      auto s = c.size();
      auto diff =
          (s < avgClusterSize) ? (avgClusterSize - s) : (s - avgClusterSize);
      sum += diff * diff;
    }
    uint64_t stdClusterSize = static_cast<uint64_t>(std::sqrt(
        static_cast<double>(sum) / static_cast<double>(clusters.size())));

    printf("Depth %4d NumClusters %8lu, AvgSize %4lu, Std %6lu.\n", depth,
           clusters.size(), avgClusterSize, stdClusterSize);
  }

  class GScorePriorityQueue {
  public:
    GScorePriorityQueue(uint64_t _numNodes) : numNodes(_numNodes) {

      queue.resize(this->numNodes);

      QueueEntry *prev = nullptr;
      for (uint64_t i = 0; i < this->numNodes; ++i) {
        QueueEntry *newEntry = &queue.at(i);
        newEntry->key = 0;
        newEntry->update = 0;
        newEntry->prev = prev;
        newEntry->next = nullptr;
        if (prev) {
          prev->next = newEntry;
        }
        prev = newEntry;
      }
      auto &qHead = this->getQueueHead(0);
      qHead.head = &queue.at(0);
      qHead.end = &queue.at(this->numNodes - 1);

      // Simply make the top pointed to 0.
      this->top = &queue.front();
      printf("Init Top -> %d -> %d.\n", this->getNodeFromEntry(this->top),
             this->getNodeFromEntry(this->top->next));

      this->verify();
    }

    void decKey(NodeID_ v) { this->queue.at(v).update--; }

    void incKey(NodeID_ v) {
      QueueEntry *entry = &this->queue.at(v);
      entry->update++;
      // printf("IncKey %d update %ld.\n", v, entry->update);
      if (entry->update > 0) {
        auto oldKey = entry->key;
        auto &oldQHead = this->getQueueHeadNoAlloc(oldKey);
        entry->key++;
        entry->update = 0;
        if (oldQHead.head == entry) {
          /**
           * This is the head of this oldKey. We only adjust the qHead.
           * 1. oldQHead.end == entry.
           *    This means we have only one entry with oldKey. We just
           *    clear qHead.
           * 2. Otherwise, there are multiple entries with oldKey. We
           *    just shift head to next one.
           */
          if (oldQHead.end == entry) {
            oldQHead.head = nullptr;
            oldQHead.end = nullptr;
          } else {
            assert(oldQHead.head->next);
            assert(oldQHead.head->next->key == oldKey);
            oldQHead.head = oldQHead.head->next;
          }

        } else {
          /**
           * We only need to adjust qHead.end.
           * Then insert entry before qHead.head.
           */
          if (oldQHead.end == entry) {
            assert(entry->prev);
            oldQHead.end = entry->prev;
          }
          this->removeFromQueue(entry);
          this->insertQueueBefore(entry, oldQHead.head);
        }

        /**
         * Append entry to qHead of new key.
         */
        auto &newQHead = this->getQueueHead(entry->key);
        newQHead.end = entry;
        if (!newQHead.head) {
          newQHead.head = entry;
        }

        if (entry->key > this->top->key) {
          printf("Replace Top -> %d -> %d.\n",
                 this->getNodeFromEntry(this->top),
                 this->getNodeFromEntry(entry));
          this->top = entry;
        }
      }

      this->verify();
    }

    NodeID_ pop() {
      assert(this->popped < this->numNodes);
      assert(this->top);
      while (this->top->update < 0) {
        auto entry = this->top;

        auto oldKey = entry->key;

        entry->key += entry->update;
        entry->update = 0;

        auto &oldQHead = this->getQueueHeadNoAlloc(oldKey);
        auto &newQHead = this->getQueueHeadNoAlloc(entry->key);
        assert(oldQHead.head == entry);

        auto topNext = this->top->next;
        if (topNext && this->top->key <= topNext->key) {

          /**
           * Need to adjust the position and qHead.
           * 1. If newQHead.head, then we know where to insert.
           * 2. Otherwise, we have to search between oldKey and newKey.
           */
          // printf("Adjust Top -> %d -> %d.\n",
          // this->getNodeFromEntry(this->top),
          //        this->getNodeFromEntry(topNext));
          this->top = topNext;

          if (newQHead.end) {
            this->removeFromQueue(entry);
            this->insertQueueAfter(entry, newQHead.end);

            newQHead.end = entry;
          } else {
            // Search for key to insert.
            QueueEntry *insertAfter = nullptr;
            for (int64_t midKey = entry->key + 1; midKey <= oldKey; ++midKey) {
              auto &midQHead = this->getQueueHeadNoAlloc(midKey);
              if (midQHead.end) {
                insertAfter = midQHead.end;
                break;
              }
            }
            if (!insertAfter) {
              printf("No InsertAfter OldKey %ld NewKey %ld TopNextKey %ld.\n",
                     oldKey, entry->key, topNext->key);
            }
            assert(insertAfter);

            this->removeFromQueue(entry);
            this->insertQueueAfter(entry, insertAfter);

            newQHead.head = entry;
            newQHead.end = entry;
          }

          // Remove from the old QHead.
          if (oldQHead.end == entry) {
            // This is the last one in oldQHead.
            oldQHead.head = nullptr;
            oldQHead.end = nullptr;
          } else {
            assert(topNext->key == oldKey);
            oldQHead.head = topNext;
          }

        } else {

          /**
           * No need to adjust the position.
           * oldQHead must only have one entry, newQHead must be empty.
           */
          assert(oldQHead.end == entry);
          assert(!newQHead.head);
          assert(!newQHead.end);
          oldQHead.head = nullptr;
          oldQHead.end = nullptr;
          newQHead.head = entry;
          newQHead.end = entry;
        }
        this->verify();
      }
      QueueEntry *vt = this->top;
      this->top = this->top->next;

      auto &qHead = this->getQueueHeadNoAlloc(vt->key);
      assert(qHead.head == vt);
      if (qHead.end == vt) {
        qHead.head = nullptr;
        qHead.end = nullptr;
      } else {
        assert(vt->next);
        assert(vt->next->key == vt->key);
        qHead.head = vt->next;
      }
      this->removeFromQueue(vt);
      this->popped++;

      printf("Pop %lu of %lu, node %d. NewTop %d.\n", this->popped,
             this->numNodes, this->getNodeFromEntry(vt),
             this->getNodeFromEntry(this->top));
      this->verify();
      return this->getNodeFromEntry(vt);
    }

  private:
    struct QueueEntry {
      int64_t key = 0;
      int64_t update = 0;
      QueueEntry *prev = nullptr;
      QueueEntry *next = nullptr;
    };
    struct QueueHead {
      QueueEntry *head = nullptr;
      QueueEntry *end = nullptr;
    };
    std::vector<QueueHead> queueHead;
    std::vector<QueueEntry> queue;
    QueueEntry *top = nullptr;
    const uint64_t numNodes;
    uint64_t popped = 0;

    QueueHead &getQueueHead(int64_t key) {
      assert(key >= 0 && "Negative key.");
      if (key >= static_cast<int64_t>(queueHead.size())) {
        queueHead.resize(key + 1);
      }
      return queueHead.at(key);
    }

    QueueHead &getQueueHeadNoAlloc(int64_t key) {
      assert(key >= 0 && "Negative key.");
      assert(key < static_cast<int64_t>(queueHead.size()));
      return queueHead.at(key);
    }

    NodeID_ getNodeFromEntry(QueueEntry *entry) {
      if (!entry) {
        return -1;
      }
      return entry - (&this->queue.front());
    }

    void verifyTop() {
      uint64_t cnt = 0;
      auto entry = this->top;
      while (entry) {
        cnt++;
        entry = entry->next;
      }
      if (cnt + this->popped != this->numNodes) {
        printf("Invalid Top %lu + %lu < %lu.\n", cnt, this->popped,
               this->numNodes);
        assert(false);
      }
    }

    void verifyQueue() {
      auto entry = this->top;
      while (entry) {
        if (entry->update > 0) {
          printf("Positive Update.\n");
          assert(false);
        }
        if (entry->next) {
          if (entry->next->prev != entry) {
            printf("Broken Link.\n");
            assert(false);
          }
          if (entry->next->key > entry->key) {
            printf("Ascending Key.\n");
            assert(false);
          }
        }
        entry = entry->next;
      }
    }

    void verifyQHead() {
      for (int64_t key = this->queueHead.size() - 1; key >= 0; --key) {
        auto &qHead = this->queueHead.at(key);
        if (qHead.head || qHead.end) {
          assert(qHead.head);
          assert(qHead.end);
          assert(qHead.head->key == key);
          assert(qHead.end->key == key);
          if (qHead.head->prev) {
            assert(qHead.head->prev->key > key);
          }
          if (qHead.end->next) {
            assert(qHead.end->next->key < key);
          }
        }
      }
    }

    void verify() {
      // this->verifyTop();
      // this->verifyQueue();
      // this->verifyQHead();
    }

    void prepend(QueueEntry *entry) {
      auto &qHead = this->getQueueHead(entry->key);
      // Prepend to qHead.head.
      entry->next = qHead.head;
      if (qHead.head) {
        entry->prev = qHead.head->prev;
        qHead.head->prev = entry;
      }
      printf("Prepend %d -> %d.\n", this->getNodeFromEntry(entry),
             this->getNodeFromEntry(qHead.head));
      qHead.head = entry;
      if (qHead.end == nullptr) {
        qHead.end = entry;
      }
    }

    void insertQueueBefore(QueueEntry *entry, QueueEntry *before) {
      assert(before);
      entry->next = before;
      entry->prev = before->prev;
      before->prev = entry;
      if (entry->prev) {
        entry->prev->next = entry;
      }
    }

    void insertQueueAfter(QueueEntry *entry, QueueEntry *after) {
      assert(after);
      entry->prev = after;
      entry->next = after->next;
      after->next = entry;
      if (entry->next) {
        entry->next->prev = entry;
      }
    }

    void removeFromQueue(QueueEntry *entry) {
      if (entry->prev) {
        entry->prev->next = entry->next;
      }
      if (entry->next) {
        entry->next->prev = entry->prev;
      }
      entry->prev = nullptr;
      entry->next = nullptr;
    }
  };

  CSRGraph<NodeID_, DestID_> ClusterByGScorePQ(CSRGraph<NodeID_, DestID_> &g,
                                               int window_size) {

    auto num_nodes = g.num_nodes();

    GScorePriorityQueue pqueue(num_nodes);
    std::vector<NodeID_> permutation;

    // Start with the first node.
    auto firstNode = pqueue.pop();
    permutation.push_back(firstNode);

    std::set<NodeID_> remaining_nodes;
    for (NodeID_ i = 0; i < num_nodes; ++i) {
      remaining_nodes.insert(i);
    }
    remaining_nodes.erase(firstNode);

    for (NodeID_ i = 1; i < num_nodes; ++i) {
      if (i % 10 == 0) {
        printf("Current progress %d Total %ld.\n", i, num_nodes);
      }
      auto ve = permutation.back();
      for (auto u : g.out_neigh(ve)) {
        if (remaining_nodes.count(u)) {
          pqueue.incKey(u);
        }
      }
      for (auto u : g.in_neigh(ve)) {
        if (remaining_nodes.count(u)) {
          pqueue.incKey(u);
        }
        for (auto v : g.out_neigh(u)) {
          if (remaining_nodes.count(v)) {
            pqueue.incKey(v);
          }
        }
      }

      if (i > window_size) {
        auto vb = permutation.at(i - window_size - 1);
        for (auto u : g.out_neigh(vb)) {
          if (remaining_nodes.count(u)) {
            pqueue.decKey(u);
          }
        }
        for (auto u : g.in_neigh(vb)) {
          if (remaining_nodes.count(u)) {
            pqueue.decKey(u);
          }
          for (auto v : g.out_neigh(u)) {
            if (remaining_nodes.count(v)) {
              pqueue.decKey(v);
            }
          }
        }
      }

      auto v_max = pqueue.pop();
      permutation.push_back(v_max);
      remaining_nodes.erase(v_max);
    }

    // Construct the forward map.
    std::unordered_map<NodeID_, NodeID_> reordered;
    for (NodeID_ i = 0; i < num_nodes; ++i) {
      NodeID_ u = permutation.at(i);
      reordered.emplace(u, i);
    }

    // Reorder the graph.
    return this->ReorderGraph(g, permutation, reordered);
  }

  CSRGraph<NodeID_, DestID_> ClusterByGScore(CSRGraph<NodeID_, DestID_> &g,
                                             int window_size) {

    auto num_nodes = g.num_nodes();

    // Start with 0.
    std::vector<NodeID_> permutation;
    permutation.push_back(0);

    std::set<NodeID_> remaining_nodes;
    for (NodeID_ i = 1; i < num_nodes; ++i) {
      remaining_nodes.insert(i);
    }

    for (NodeID_ i = 1; i < num_nodes; ++i) {
      if (i % 10 == 0) {
        printf("Current progress %d Total %ld.\n", i, num_nodes);
      }
      NodeID_ v_max = *(remaining_nodes.begin());
      uint64_t k_max = 0;
      for (NodeID_ v : remaining_nodes) {
        uint64_t kv = 0;
        for (int64_t s = i - 1; s >= 0 && s >= i - window_size; --s) {
          kv += GScore(g, permutation[s], v);
        }
        if (kv > k_max) {
          k_max = kv;
          v_max = v;
        }
      }
      permutation.push_back(v_max);
      remaining_nodes.erase(v_max);
    }

    // Construct the forward map.
    std::unordered_map<NodeID_, NodeID_> reordered;
    for (NodeID_ i = 0; i < num_nodes; ++i) {
      NodeID_ u = permutation.at(i);
      reordered.emplace(u, i);
    }

    // Reorder the graph.
    return this->ReorderGraph(g, permutation, reordered);
  }

  uint64_t GScore(CSRGraph<NodeID_, DestID_> &g, NodeID_ u, NodeID_ v) {
    uint64_t s = 0;
    for (NodeID_ u_in : g.in_neigh(u)) {
      for (NodeID_ v_in : g.in_neigh(v)) {
        if (v_in == u_in) {
          s++;
        }
      }
    }
    return s;
  }

  CSRGraph<NodeID_, DestID_>
  ReorderGraph(CSRGraph<NodeID_, DestID_> &g,
               const std::vector<NodeID_> &permutation,
               const std::unordered_map<NodeID_, NodeID_> &reordered) {

    auto num_nodes = g.num_nodes();
    auto num_edges = g.out_neigh_index()[num_nodes] - g.out_neigh_index()[0];

    DestID_ *edges = alignedAllocAndTouch<DestID_>(num_edges);
    DestID_ **indexes = alignedAllocAndTouch<DestID_ *>(num_nodes + 1);
    int64_t cur_edges = 0;
    for (auto v = 0; v < num_nodes; ++v) {
      indexes[v] = edges + cur_edges;
      auto original_v = permutation.at(v);
      for (auto original_u : g.out_neigh(original_v)) {
        auto u = reordered.at(original_u);
        edges[cur_edges] = u;
        cur_edges++;
      }
    }
    assert(cur_edges == num_edges);
    indexes[num_nodes] = edges + num_edges;

    CSRGraph<NodeID_, DestID_> ret(num_nodes, indexes, edges);
    return ret;
  }
};

#endif // WRITER_H_
