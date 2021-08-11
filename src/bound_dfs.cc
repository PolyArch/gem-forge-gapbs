// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#include <iostream>

#include "benchmark.h"
#include "builder.h"
#include "cluster.h"
#include "command_line.h"
#include "graph.h"
#include "reader.h"
#include "source_generator.h"
#include "writer.h"

using namespace std;

int main(int argc, char *argv[]) {
  CLApp cli(argc, argv, "bound_dfs");
  if (!cli.ParseArgs())
    return -1;

  Builder b(cli);
  Graph g = b.MakeGraph();

  Cluster c;

#pragma omp parallel for
  for (int depth_pow = 1; depth_pow <= 10; ++depth_pow) {

    int depth = (1 << depth_pow);

    printf("Constructing Bounded DFS Graph %d.\n", depth);
    auto bdfs = c.ClusterByBoundedDFS(g, depth, 1024);
    Writer clustered_w(bdfs);
    auto pos = cli.filename().find_last_of(".");
    const bool serialized = true;
    clustered_w.WriteGraph(cli.filename().substr(0, pos) + "-bdfs-d" +
                               std::to_string(depth) +
                               cli.filename().substr(pos),
                           serialized);
  }

  return 0;
}
