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
  CLConvert cli(argc, argv, "converter");
  cli.ParseArgs();
  if (cli.out_weighted()) {
    WeightedBuilder bw(cli);
    WGraph wg = bw.MakeGraph();
    wg.PrintStats();
    WeightedWriter ww(wg);
    ww.WriteGraph(cli.out_filename() + ".wsg", true /* serialized */);
    ww.WriteGraph(cli.out_filename() + ".wel", false /* serialized */);
    SourceGenerator<WGraph> sg(wg, cli.src());
    sg.writeToFile(cli.out_filename());
  } else {
    Builder b(cli);
    Graph g = b.MakeGraph();
    g.PrintStats();
    Writer w(g);
    // Dump both version.
    w.WriteGraph(cli.out_filename() + ".sg", true /* serialized */);
    w.WriteGraph(cli.out_filename() + ".el", false /* serialized */);
    SourceGenerator<Graph> sg(g, cli.src());
    sg.writeToFile(cli.out_filename());
  }
  return 0;
}
