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
  CLApp cli(argc, argv, "gscore");
  if (!cli.ParseArgs())
    return -1;

  Builder b(cli);
  Graph g = b.MakeGraph();

  Cluster c;

#pragma omp parallel for
  for (int window_size_pow = 3; window_size_pow <= 12; ++window_size_pow) {
    int window_size = (1 << window_size_pow);
    auto gscore = c.ClusterByGScorePQ(g, window_size);
    Writer clustered_w(gscore);
    auto pos = cli.filename().find_last_of(".");
    const bool serialized = true;
    clustered_w.WriteGraph(cli.filename().substr(0, pos) + "-gs-w" +
                               std::to_string(window_size) +
                               cli.filename().substr(pos),
                           serialized);
  }

  return 0;
}
