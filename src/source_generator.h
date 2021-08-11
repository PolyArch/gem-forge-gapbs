#ifndef SOURCE_GENERATOR_H
#define SOURCE_GENERATOR_H

#include "graph.h"

#include <fstream>
#include <queue>
#include <unordered_map>
#include <vector>

// Pick up a source that covers most of the nodes.
template <typename GraphT_> class SourceGenerator {
public:
  explicit SourceGenerator(const GraphT_ &g) : g_(g) {
    pickedSource = pickSource();
  }

  void writeToFile(const std::string &graph_fn) {
    std::ofstream o(graph_fn + src_suffix());
    o << pickedSource;
    o.close();
  }

  static std::vector<NodeID> loadSource(const std::string &graph_fn) {
    std::ifstream i(graph_fn + src_suffix());
    std::vector<NodeID> sources;
    if (i.is_open()) {
      NodeID source;
      i >> source;
      sources.push_back(source);
    }
    return sources;
  }

  NodeID getSource() const { return this->pickedSource; }

private:
  const GraphT_ &g_;
  NodeID pickedSource;

  static constexpr const char *src_suffix() { return ".src.txt"; }

  NodeID pickSource() {

    // Perform 10 BFS to pick up a suitable root.
    const size_t numIters = 10;

    auto numNodes = g_.num_nodes();
    NodeID pickedSource = 0;
    size_t maxVisitedEdges = 0;

    for (size_t i = 0; i < numIters; ++i) {
      NodeID root = numNodes * i / numIters;
      std::unordered_map<NodeID, size_t> visited;
      std::queue<NodeID> q;
      size_t visitedEdges = 0;
      q.push(root);
      visited.emplace(root, 0);
      while (!q.empty()) {
        auto node = q.front();
        q.pop();
        auto dist = visited.at(node);
        for (NodeID v : g_.out_neigh(node)) {
          if (visited.count(v) == 0) {
            visited.emplace(v, dist + 1);
            q.push(v);
          }
          visitedEdges++;
        }
      }
      if (visitedEdges > maxVisitedEdges) {
        pickedSource = root;
        maxVisitedEdges = visitedEdges;
      }
    }
    std::cout << "Picked root " << pickedSource << ", coverage "
              << static_cast<float>(maxVisitedEdges) /
                     static_cast<float>(g_.num_edges())
              << '\n';
    return pickedSource;
  }
};

#endif