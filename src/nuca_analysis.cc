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
const float kDamp = 0.85;

class NUCAIndirectTraffic {
public:
  NUCAIndirectTraffic(int _EdgeBankInterleave, int _ValBankInterleave)
      : EdgeBankInterleave(_EdgeBankInterleave),
        ValBankInterleave(_ValBankInterleave),
        EdgePerBank(_EdgeBankInterleave / sizeof(NodeID)),
        ValPerBank(_ValBankInterleave / sizeof(ScoreT)) {
    this->mid_banks.resize(this->BankPerRow * this->BankPerCol, 0);
    this->freq_banks.resize(this->BankPerRow * this->BankPerCol, 0);
    this->cur_val_bank_count.resize(this->BankPerRow * this->BankPerCol, 0);
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
  const int EdgePerBank;
  const int ValPerBank;
  const int BankPerRow = 8;
  const int BankPerCol = 8;

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
    return (i / EdgePerBank) % (BankPerRow * BankPerCol);
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

  uint64_t selectMid(uint64_t acc, uint64_t size) const {
    return static_cast<int>(static_cast<float>(acc) / static_cast<float>(size) +
                            0.5f);
  }

  uint64_t selectFrequent() const {
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
    return max_bank;
  }

  void accumulateHops() {
    auto mid_bank_row =
        this->selectMid(cur_val_bank_rows, cur_val_banks.size());
    auto mid_bank_col =
        this->selectMid(cur_val_bank_cols, cur_val_banks.size());
    int mid_bank = mid_bank_row * BankPerRow + mid_bank_col;
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
CountNUCATraffic(const Graph &g, int edge_interleave, int node_interleave) {
  NUCAIndirectTraffic traffic(edge_interleave, node_interleave);
  for (NodeID e = 0; e < g.num_edges(); e++) {
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

  const int node_intrl_pow_start = 6;
  const int node_intrl_pow_end = 15;
  const int node_intrl_num = node_intrl_pow_end - node_intrl_pow_start;
  const int edge_intrl_pow_start = 6;
  const int edge_intrl_pow_end = 15;
  const int edge_intrl_num = edge_intrl_pow_end - edge_intrl_pow_start;

  NUCAIndirectTraffic::Result results[node_intrl_num][edge_intrl_num];

#pragma omp parallel for
  for (int node_intrl_pow = node_intrl_pow_start;
       node_intrl_pow < node_intrl_pow_end; ++node_intrl_pow) {

    auto node_intrl_idx = node_intrl_pow - node_intrl_pow_start;

    for (int edge_intrl_pow = edge_intrl_pow_start;
         edge_intrl_pow < edge_intrl_pow_end; ++edge_intrl_pow) {

      auto edge_intrl_idx = edge_intrl_pow - edge_intrl_pow_start;

      auto node_interleave = (1 << node_intrl_pow);
      auto edge_interleave = (1 << edge_intrl_pow);

      printf("Eval NodeIntrl %s EdgeIntrl %s.\n",
             formatStorage(node_interleave).c_str(),
             formatStorage(edge_interleave).c_str());
      auto ret = CountNUCATraffic(g, edge_interleave, node_interleave);

      results[node_intrl_idx][edge_intrl_idx] = ret;
    }
  }

  for (int node_intrl_pow = node_intrl_pow_start;
       node_intrl_pow < node_intrl_pow_end; ++node_intrl_pow) {

    auto node_intrl_idx = node_intrl_pow - node_intrl_pow_start;

    for (int edge_intrl_pow = edge_intrl_pow_start;
         edge_intrl_pow < edge_intrl_pow_end; ++edge_intrl_pow) {

      auto edge_intrl_idx = edge_intrl_pow - edge_intrl_pow_start;

      auto node_interleave = (1 << node_intrl_pow);
      auto edge_interleave = (1 << edge_intrl_pow);

      const auto &ret = results[node_intrl_idx][edge_intrl_idx];

      printf(
          "NodeIntrl %6s EdgeIntrl %6s BaseHop %10lu Mid %10lu Avg %6s Std %6s "
          "Freq %10lu Std %6s.\n",
          formatStorage(node_interleave).c_str(),
          formatStorage(edge_interleave).c_str(), ret.base_hops, ret.mid_hops,
          formatStorage(ret.mid_edge_bytes_per_bank_avg).c_str(),
          formatStorage(ret.mid_edge_bytes_per_bank_std).c_str(), ret.freq_hops,
          formatStorage(ret.freq_edge_bytes_per_bank_std).c_str());
    }
  }
}
