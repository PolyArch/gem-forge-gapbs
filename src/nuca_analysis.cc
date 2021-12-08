// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

#include "benchmark.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "pvector.h"

/**
 * Analyze the NUCA traffic reduction.
 */

using namespace std;

typedef float ScoreT;

class NUCAIndirectTraffic {
public:
  enum EdgeBankTopology {
    EVERY_BANK = 0, // Edges can be stored at each bank.
    CENTRAL_HALF,   // Only the center half banks store edge.
    CORNER,         // Only the four corners.
    DIAGONAL,       // Diagonal banks.
    TILE_2,         // Top left corner of each 2x2 tile.
    TILE_2_VAR,     // Variant bank of each 2x2 tile.
    NUM_TOPOLOGIES,
  };
  static const char *edgeBankTopologyToString(EdgeBankTopology e) {
#define Case(x)                                                                \
  case x:                                                                      \
    return #x
    switch (e) {
      Case(EVERY_BANK);
      Case(CENTRAL_HALF);
      Case(CORNER);
      Case(DIAGONAL);
      Case(TILE_2);
      Case(TILE_2_VAR);
#undef Case
    default:
      assert(false && "Invalid EdgeBankTopology.");
    }
  }

  NUCAIndirectTraffic(int _EdgeBankInterleave, int _ValBankInterleave,
                      EdgeBankTopology _EdgeTopology)
      : EdgeBankInterleave(_EdgeBankInterleave),
        ValBankInterleave(_ValBankInterleave), EdgeTopology(_EdgeTopology),
        EdgePerBank(_EdgeBankInterleave / sizeof(NodeID)),
        ValPerBank(_ValBankInterleave / sizeof(ScoreT)) {
    this->mid_banks.resize(this->BankPerRow * this->BankPerCol, 0);
    this->freq_banks.resize(this->BankPerRow * this->BankPerCol, 0);
    this->cur_val_bank_count.resize(this->BankPerRow * this->BankPerCol, 0);
    this->populateValidEdgeBanks();
  }

  void addEdge(NodeID edge) {
    cur_edge_banks.push_back(getEdgeBank(cur_edge));
    auto val_bank = getValBank(edge);
    cur_val_banks.push_back(val_bank);
    cur_val_bank_count.at(val_bank)++;
    cur_val_bank_rows += val_bank / BankPerRow;
    cur_val_bank_cols += val_bank % BankPerRow;
    cur_edge++;
    if ((cur_edge % EdgePerBank) == 0) {
      accumulateHops();
    }
  }

  void print() {
    printf("base_hops %lu mid_hops %lu freq_hops %lu.\n", this->base_hops,
           this->mid_hops, this->freq_hops);
    for (int i = 0; i < BankPerRow * BankPerCol; ++i) {
      printf("MidBank %3d %8lu %8lu.\n", i, this->mid_banks.at(i),
             this->freq_banks.at(i));
    }
  }

  struct Result {
    uint64_t base_hops = 0;
    uint64_t mid_hops = 0;
    uint64_t mid_edge_bytes_per_bank_avg = 0;
    uint64_t mid_edge_bytes_per_bank_std = 0;
    uint64_t freq_hops = 0;
    uint64_t freq_edge_bytes_per_bank_avg = 0;
    uint64_t freq_edge_bytes_per_bank_std = 0;
  };

  Result getResult() const {
    Result ret;
    ret.base_hops = this->base_hops;
    ret.mid_hops = this->mid_hops;
    ret.mid_edge_bytes_per_bank_avg = this->computeAvg(this->mid_banks);
    ret.mid_edge_bytes_per_bank_std =
        this->computeStdVar(this->mid_banks, ret.mid_edge_bytes_per_bank_avg);

    ret.freq_hops = this->freq_hops;
    ret.freq_edge_bytes_per_bank_avg = this->computeAvg(this->freq_banks);
    ret.freq_edge_bytes_per_bank_std =
        this->computeStdVar(this->freq_banks, ret.freq_edge_bytes_per_bank_avg);

    return ret;
  }

  uint64_t computeAvg(const std::vector<uint64_t> &vs) const {
    uint64_t sum = 0;
    for (auto v : vs) {
      sum += v;
    }
    return sum / vs.size();
  }

  uint64_t computeStdVar(const std::vector<uint64_t> &vs, uint64_t avg) const {
    uint64_t sum = 0;
    for (auto v : vs) {
      uint64_t diff = 0;
      if (v < avg) {
        diff = avg - v;
      } else {
        diff = v - avg;
      }
      sum += diff * diff;
    }
    uint64_t var = sum / vs.size();
    return static_cast<uint64_t>(std::sqrt(static_cast<double>(var)));
  }

private:
  mutable std::default_random_engine generator;
  const int EdgeBankInterleave;
  const int ValBankInterleave;
  const EdgeBankTopology EdgeTopology;
  const int EdgePerBank;
  const int ValPerBank;
  const int BankPerRow = 4;
  const int BankPerCol = 4;
  std::vector<uint64_t> validEdgeBanks;

  std::vector<uint64_t> mid_banks;
  std::vector<uint64_t> freq_banks;
  uint64_t base_hops = 0;
  uint64_t mid_hops = 0;
  uint64_t freq_hops = 0;

  uint64_t cur_edge = 0;

  std::vector<uint64_t> cur_val_bank_count;
  std::vector<int> cur_edge_banks;
  std::vector<int> cur_val_banks;
  uint64_t cur_val_bank_rows = 0;
  uint64_t cur_val_bank_cols = 0;

  int getEdgeBank(uint64_t i) const {
    auto rawEdgeBank = i / EdgePerBank;
    switch (this->EdgeTopology) {
    case EVERY_BANK: {
      return rawEdgeBank % (BankPerRow * BankPerCol);
    }
    case CENTRAL_HALF: {
      auto halfBankPerRow = BankPerRow / 2;
      auto halfBankPerCol = BankPerCol / 2;
      auto quarterBankPerRow = halfBankPerRow / 2;
      auto quarterBankPerCol = halfBankPerCol / 2;
      auto edgeRow = (rawEdgeBank / halfBankPerRow) % halfBankPerCol;
      auto edgeCol = rawEdgeBank % halfBankPerRow;
      return (edgeRow + quarterBankPerCol) * BankPerRow + edgeCol +
             quarterBankPerRow;
    }
    case CORNER: {
      auto edgeBank = rawEdgeBank % 4;
      switch (edgeBank) {
      case 0:
        return 0;
      case 1:
        return BankPerRow - 1;
      case 2:
        return (BankPerCol - 1) * BankPerRow;
      case 3:
        return BankPerCol * BankPerRow - 1;
      }
    }
    case DIAGONAL: {
      auto minDim = std::min(BankPerCol, BankPerRow);
      auto edgeBank = rawEdgeBank % minDim;
      return edgeBank * BankPerRow + edgeBank;
    }
    case TILE_2: {
      auto row = (rawEdgeBank / BankPerRow) % BankPerCol;
      auto col = rawEdgeBank % BankPerRow;
      auto tileRow = row - row % 2;
      auto tileCol = col - col % 2;
      return tileRow * BankPerRow + tileCol;
    }
    case TILE_2_VAR: {
      auto row = (rawEdgeBank / BankPerRow) % BankPerCol;
      auto col = rawEdgeBank % BankPerRow;
      auto tileRow = row - row % 2 + col % 2;
      auto tileCol = col - col % 2 + row % 2;
      return tileRow * BankPerRow + tileCol;
    }
    default: {
      assert(false && "Invalid EdgeTopology.");
    }
    }
  }
  int getValBank(uint64_t i) const {
    return (i / ValPerBank) % (BankPerRow * BankPerCol);
  }
  int getHopsBetween(int bankA, int bankB) const {
    int rowA = bankA / BankPerRow;
    int rowB = bankB / BankPerRow;
    int colA = bankA % BankPerRow;
    int colB = bankB % BankPerRow;
    return std::abs(rowA - rowB) + std::abs(colA - colB);
  }

  uint64_t computeMid(uint64_t acc, uint64_t size) const {
    return static_cast<int>(static_cast<float>(acc) / static_cast<float>(size) +
                            0.5f);
  }
  uint64_t selectMiddle() const {
    /**
     * Select the cloest valid edge bank to the middle of the current node
     * banks.
     */
    auto mid_bank_row =
        this->computeMid(cur_val_bank_rows, cur_val_banks.size());
    auto mid_bank_col =
        this->computeMid(cur_val_bank_cols, cur_val_banks.size());
    int mid_bank = mid_bank_row * BankPerRow + mid_bank_col;
    return this->selectClosestEdgeBank(mid_bank);
  }

  uint64_t selectFrequent() const {
    /**
     * Select the cloest valid edge bank to the most frequent node
     * bank.
     */
    auto max_count = this->cur_val_bank_count.front();
    auto max_bank = 0;
    uint64_t max_banks = 1;
    for (auto i = 0; i < this->BankPerRow * this->BankPerCol; ++i) {
      auto count = this->cur_val_bank_count.at(i);
      if (count > max_count) {
        max_bank = i;
        max_count = count;
        max_banks = 1;
      } else if (count == max_count) {
        // Flip a coin to pick this new bank.
        max_banks++;
        std::uniform_int_distribution<uint64_t> r(1, max_banks);
        auto v = r(this->generator);
        if (v == max_banks) {
          max_bank = i;
        }
      }
    }
    return this->selectClosestEdgeBank(max_bank);
  }

  uint64_t selectClosestEdgeBank(int bank) const {
    /**
     * Get the cloest edge bank to this bank.
     */
    auto edgeBank = this->BankPerRow * this->BankPerCol;
    auto minHops = INT32_MAX;
    for (auto e : this->validEdgeBanks) {
      auto hops = this->getHopsBetween(bank, e);
      if (hops < minHops) {
        minHops = hops;
        edgeBank = e;
      }
    }
    assert(minHops != INT32_MAX && "Failed to select closest edge bank.");
    return edgeBank;
  }

  void populateValidEdgeBanks() {
    switch (this->EdgeTopology) {
    case EVERY_BANK: {
      for (int i = 0; i < BankPerCol; ++i) {
        for (int j = 0; j < BankPerRow; ++j) {
          this->validEdgeBanks.push_back(i * BankPerRow + j);
        }
      }
      break;
    }
    case DIAGONAL: {
      auto minDim = std::min(BankPerCol, BankPerRow);
      for (int i = 0; i < minDim; ++i) {
        this->validEdgeBanks.push_back(i * BankPerRow + i);
      }
      break;
    }
    case CORNER: {
      this->validEdgeBanks.push_back(0);
      this->validEdgeBanks.push_back(BankPerRow - 1);
      this->validEdgeBanks.push_back((BankPerCol - 1) * BankPerRow);
      this->validEdgeBanks.push_back(BankPerCol * BankPerRow - 1);
      break;
    }
    case TILE_2: {
      for (int i = 0; i < BankPerCol; i += 2) {
        for (int j = 0; j < BankPerRow; j += 2) {
          this->validEdgeBanks.push_back(i * BankPerRow + j);
        }
      }
      break;
    }
    case TILE_2_VAR: {
      for (int i = 0; i < BankPerCol; i += 2) {
        for (int j = 0; j < BankPerRow; j += 2) {
          int blockRow = i / 2;
          int blockCol = j / 2;
          this->validEdgeBanks.push_back((i + blockCol % 2) * BankPerRow + j +
                                         blockRow % 2);
        }
      }
      break;
    }
    case CENTRAL_HALF: {
      auto halfBankPerRow = BankPerRow / 2;
      auto halfBankPerCol = BankPerCol / 2;
      auto quarterBankPerRow = halfBankPerRow / 2;
      auto quarterBankPerCol = halfBankPerCol / 2;
      for (int i = quarterBankPerCol; i < quarterBankPerCol + halfBankPerCol;
           ++i) {
        for (int j = quarterBankPerRow; j < quarterBankPerRow + halfBankPerRow;
             ++j) {
          this->validEdgeBanks.push_back(i * BankPerRow + j);
        }
      }
      break;
    }
    default: {
      assert(false && "Invalid EdgeTopology Type.");
    }
    }
    std::stringstream ss;
    ss << "EdgeTopology " << edgeBankTopologyToString(this->EdgeTopology)
       << " ValidEdgeBanks: ";
    for (auto e : this->validEdgeBanks) {
      ss << e << " ";
    }
    // printf("%s.\n", ss.str().c_str());
  }

  void accumulateHops() {
    int mid_bank = this->selectMiddle();
    this->mid_banks.at(mid_bank) += EdgeBankInterleave;

    int freq_bank = this->selectFrequent();
    this->freq_banks.at(freq_bank) += EdgeBankInterleave;

    uint64_t base_hops = 0;
    uint64_t mid_hops = 0;
    uint64_t freq_hops = 0;
    for (size_t i = 0; i < cur_edge_banks.size(); ++i) {
      auto edge_bank = cur_edge_banks.at(i);
      auto val_bank = cur_val_banks.at(i);
      base_hops += getHopsBetween(edge_bank, val_bank);
      mid_hops += getHopsBetween(mid_bank, val_bank);
      freq_hops += getHopsBetween(freq_bank, val_bank);
    }
    this->base_hops += base_hops;
    this->mid_hops += mid_hops;
    this->freq_hops += freq_hops;
    cur_edge_banks.clear();
    cur_val_banks.clear();
    cur_val_bank_cols = 0;
    cur_val_bank_rows = 0;
    for (auto &v : this->cur_val_bank_count) {
      v = 0;
    }
  }
};

NUCAIndirectTraffic::Result
CountNUCATraffic(const Graph &g, int edge_interleave, int node_interleave,
                 NUCAIndirectTraffic::EdgeBankTopology edge_topology) {
  NUCAIndirectTraffic traffic(edge_interleave, node_interleave, edge_topology);
  for (NodeID e = 0; e < g.num_edges_directed(); e++) {
    NodeID edge = g.out_edges()[e];
    traffic.addEdge(edge);
  }
  // traffic.print();
  return traffic.getResult();
}

std::string formatStorage(uint64_t bytes) {
  if (bytes >= 1024 * 1024) {
    return std::to_string(bytes / 1024 / 1024) + "MiB";
  } else if (bytes >= 1024) {
    return std::to_string(bytes / 1024) + "KiB";
  } else {
    return std::to_string(bytes) + "B";
  }
}

int main(int argc, char *argv[]) {
  CLApp cli(argc, argv, "nuca");
  if (!cli.ParseArgs())
    return -1;

  if (cli.num_threads() != -1) {
    printf("%d.\n", cli.num_threads());
  }

  Builder b(cli);
  Graph g = b.MakeGraph();
  g.PrintStats();

  const int node_intrl_pow_start = 6;
  const int node_intrl_pow_end = 11;
  const int node_intrl_num = node_intrl_pow_end - node_intrl_pow_start;
  const int edge_intrl_pow_start = 6;
  const int edge_intrl_pow_end = 13;
  const int edge_intrl_num = edge_intrl_pow_end - edge_intrl_pow_start;

  NUCAIndirectTraffic::Result
      results[node_intrl_num][edge_intrl_num]
             [NUCAIndirectTraffic::EdgeBankTopology::NUM_TOPOLOGIES];

#pragma omp parallel for
  for (int node_intrl_pow = node_intrl_pow_start;
       node_intrl_pow < node_intrl_pow_end; ++node_intrl_pow) {

    auto node_intrl_idx = node_intrl_pow - node_intrl_pow_start;

    for (int edge_intrl_pow = edge_intrl_pow_start;
         edge_intrl_pow < edge_intrl_pow_end; ++edge_intrl_pow) {

      auto edge_intrl_idx = edge_intrl_pow - edge_intrl_pow_start;

      auto node_interleave = (1 << node_intrl_pow);
      auto edge_interleave = (1 << edge_intrl_pow);

      for (int et = 0;
           et < NUCAIndirectTraffic::EdgeBankTopology::NUM_TOPOLOGIES; ++et) {

        auto edge_topology =
            static_cast<NUCAIndirectTraffic::EdgeBankTopology>(et);

        printf("Eval NodeIntrl %s EdgeIntrl %s EdgeTopology %s.\n",
               formatStorage(node_interleave).c_str(),
               formatStorage(edge_interleave).c_str(),
               NUCAIndirectTraffic::edgeBankTopologyToString(edge_topology));
        auto ret = CountNUCATraffic(g, edge_interleave, node_interleave,
                                    edge_topology);

        results[node_intrl_idx][edge_intrl_idx][edge_topology] = ret;
      }
    }
  }

  for (int et = 0; et < NUCAIndirectTraffic::EdgeBankTopology::NUM_TOPOLOGIES;
       ++et) {

    auto edge_topology = static_cast<NUCAIndirectTraffic::EdgeBankTopology>(et);

    for (int node_intrl_pow = node_intrl_pow_start;
         node_intrl_pow < node_intrl_pow_end; ++node_intrl_pow) {

      auto node_intrl_idx = node_intrl_pow - node_intrl_pow_start;

      for (int edge_intrl_pow = edge_intrl_pow_start;
           edge_intrl_pow < edge_intrl_pow_end; ++edge_intrl_pow) {

        auto edge_intrl_idx = edge_intrl_pow - edge_intrl_pow_start;

        auto node_interleave = (1 << node_intrl_pow);
        auto edge_interleave = (1 << edge_intrl_pow);

        const auto &ret =
            results[node_intrl_idx][edge_intrl_idx][edge_topology];

        printf("EdgeTopo %s NodeIntrl %6s EdgeIntrl %6s BaseHop %10lu Mid "
               "%10lu Avg %6s "
               "Std %6s "
               "Freq %10lu Std %6s.\n",
               NUCAIndirectTraffic::edgeBankTopologyToString(edge_topology),
               formatStorage(node_interleave).c_str(),
               formatStorage(edge_interleave).c_str(), ret.base_hops,
               ret.mid_hops,
               formatStorage(ret.mid_edge_bytes_per_bank_avg).c_str(),
               formatStorage(ret.mid_edge_bytes_per_bank_std).c_str(),
               ret.freq_hops,
               formatStorage(ret.freq_edge_bytes_per_bank_std).c_str());
      }
    }
  }
}
