// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#ifndef COMMAND_LINE_H_
#define COMMAND_LINE_H_

#include <getopt.h>

#include <algorithm>
#include <cinttypes>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

/*
GAP Benchmark Suite
Class:  CLBase
Author: Scott Beamer

Handles command line argument parsing
 - Through inheritance, can add more options to object
 - For example, most kernels will use CLApp
*/

class CLBase {
protected:
  int argc_;
  char **argv_;
  std::string name_;
  std::string get_args_ = "f:g:hk:su:c:j";
  std::vector<std::string> help_strings_;

  int scale_ = -1;
  int degree_ = 16;
  std::string filename_ = "";
  bool symmetrize_ = false;
  bool uniform_ = false;
  /**
   * How to warm up cache:
   * 0: don't warm up.
   * 1: warm up except the edge list.
   * 2: warm up everything.
   */
  int warm_cache_ = 2;
  bool graph_partition_ = false;

  void AddHelpLine(char opt, std::string opt_arg, std::string text,
                   std::string def = "") {
    const int kBufLen = 100;
    char buf[kBufLen];
    if (opt_arg != "")
      opt_arg = "<" + opt_arg + ">";
    if (def != "")
      def = "[" + def + "]";
    snprintf(buf, kBufLen, " -%c %-9s: %-54s%10s", opt, opt_arg.c_str(),
             text.c_str(), def.c_str());
    help_strings_.push_back(buf);
  }

public:
  CLBase(int argc, char **argv, std::string name = "")
      : argc_(argc), argv_(argv), name_(name) {
    AddHelpLine('h', "", "print this help message");
    AddHelpLine('f', "file", "load graph from file");
    AddHelpLine('s', "", "symmetrize input edge list", "false");
    AddHelpLine('g', "scale", "generate 2^scale kronecker graph");
    AddHelpLine('u', "scale", "generate 2^scale uniform-random graph");
    AddHelpLine('k', "degree", "average degree for synthetic graph",
                std::to_string(degree_));
    AddHelpLine('c', "cache", "warm up cache to certain level",
                std::to_string(warm_cache_));
    AddHelpLine('j', "partition", "distribute graph in partition", "false");
  }

  bool ParseArgs() {
    signed char c_opt;
    extern char *optarg; // from and for getopt
    while ((c_opt = getopt(argc_, argv_, get_args_.c_str())) != -1) {
      HandleArg(c_opt, optarg);
    }
    if ((filename_ == "") && (scale_ == -1)) {
      std::cout << "No graph input specified. (Use -h for help)" << std::endl;
      return false;
    }
    if (scale_ != -1)
      symmetrize_ = true;
    return true;
  }

  void virtual HandleArg(signed char opt, char *opt_arg) {
    switch (opt) {
    case 'f':
      filename_ = std::string(opt_arg);
      break;
    case 'g':
      scale_ = atoi(opt_arg);
      break;
    case 'h':
      PrintUsage();
      break;
    case 'k':
      degree_ = atoi(opt_arg);
      break;
    case 's':
      symmetrize_ = true;
      break;
    case 'u':
      uniform_ = true;
      scale_ = atoi(opt_arg);
      break;
    case 'c':
      warm_cache_ = atoi(opt_arg);
      break;
    case 'j':
      graph_partition_ = true;
      break;
    }
  }

  void PrintUsage() {
    std::cout << name_ << std::endl;
    // std::sort(help_strings_.begin(), help_strings_.end());
    for (std::string h : help_strings_)
      std::cout << h << std::endl;
    std::exit(0);
  }

  int scale() const { return scale_; }
  int degree() const { return degree_; }
  std::string filename() const { return filename_; }
  bool symmetrize() const { return symmetrize_; }
  bool uniform() const { return uniform_; }
  int warm_cache() const { return warm_cache_; }
  bool graph_partition() const { return graph_partition_; }
};

class CLApp : public CLBase {
  bool do_analysis_ = false;
  int num_trials_ = 16;
  int num_threads_ = -1;
  int64_t start_vertex_ = -1;
  bool do_verify_ = false;

public:
  CLApp(int argc, char **argv, std::string name) : CLBase(argc, argv, name) {
    get_args_ += "an:p:r:v";
    AddHelpLine('a', "", "output analysis of last run", "false");
    AddHelpLine('n', "n", "perform n trials", std::to_string(num_trials_));
    AddHelpLine('p', "p", "parallel p threads", std::to_string(num_threads_));
    AddHelpLine('r', "node", "start from node r", "rand");
    AddHelpLine('v', "", "verify the output of each run", "false");
  }

  void HandleArg(signed char opt, char *opt_arg) override {
    switch (opt) {
    case 'a':
      do_analysis_ = true;
      break;
    case 'n':
      num_trials_ = atoi(opt_arg);
      break;
    case 'p':
      num_threads_ = atoi(opt_arg);
      break;
    case 'r':
      start_vertex_ = atol(opt_arg);
      break;
    case 'v':
      do_verify_ = true;
      break;
    default:
      CLBase::HandleArg(opt, opt_arg);
    }
  }

  bool do_analysis() const { return do_analysis_; }
  int num_trials() const { return num_trials_; }
  int num_threads() const { return num_threads_; }
  int64_t start_vertex() const { return start_vertex_; }
  bool do_verify() const { return do_verify_; }
};

class CLIterApp : public CLApp {
  int num_iters_;

public:
  CLIterApp(int argc, char **argv, std::string name, int num_iters)
      : CLApp(argc, argv, name), num_iters_(num_iters) {
    get_args_ += "i:";
    AddHelpLine('i', "i", "perform i iterations", std::to_string(num_iters_));
  }

  void HandleArg(signed char opt, char *opt_arg) override {
    switch (opt) {
    case 'i':
      num_iters_ = atoi(opt_arg);
      break;
    default:
      CLApp::HandleArg(opt, opt_arg);
    }
  }

  int num_iters() const { return num_iters_; }
};

class CLPageRank : public CLApp {
  int max_iters_;
  double tolerance_;

public:
  CLPageRank(int argc, char **argv, std::string name, double tolerance,
             int max_iters)
      : CLApp(argc, argv, name), max_iters_(max_iters), tolerance_(tolerance) {
    get_args_ += "i:t:";
    AddHelpLine('i', "i", "perform at most i iterations",
                std::to_string(max_iters_));
    AddHelpLine('t', "t", "use tolerance t", std::to_string(tolerance_));
  }

  void HandleArg(signed char opt, char *opt_arg) override {
    switch (opt) {
    case 'i':
      max_iters_ = atoi(opt_arg);
      break;
    case 't':
      tolerance_ = std::stod(opt_arg);
      break;
    default:
      CLApp::HandleArg(opt, opt_arg);
    }
  }

  int max_iters() const { return max_iters_; }
  double tolerance() const { return tolerance_; }
};

class CLBFS : public CLApp {
  int alpha_ = 6;
  int beta_ = 5;
  int delta_ = 0;

public:
  CLBFS(int argc, char **argv, std::string name) : CLApp(argc, argv, name) {
    get_args_ += "i:t:d:";
    AddHelpLine('i', "i", "alpha: active edge percentage to pull",
                std::to_string(alpha_));
    AddHelpLine('t', "t", "beta: active node percentage to push",
                std::to_string(beta_));
    AddHelpLine('d', "d", "delta: total visited node percentage to pull",
                std::to_string(delta_));
  }

  void HandleArg(signed char opt, char *opt_arg) override {
    switch (opt) {
    case 'i':
      alpha_ = atoi(opt_arg);
      break;
    case 't':
      beta_ = std::stod(opt_arg);
      break;
    case 'd':
      delta_ = std::stod(opt_arg);
      break;
    default:
      CLApp::HandleArg(opt, opt_arg);
    }
  }

  int alpha() const { return alpha_; }
  int beta() const { return beta_; }
  int delta() const { return delta_; }
};

template <typename WeightT_> class CLDelta : public CLApp {
  WeightT_ delta_ = 1;

public:
  CLDelta(int argc, char **argv, std::string name) : CLApp(argc, argv, name) {
    get_args_ += "d:";
    AddHelpLine('d', "d", "delta parameter", std::to_string(delta_));
  }

  void HandleArg(signed char opt, char *opt_arg) override {
    switch (opt) {
    case 'd':
      if (std::is_floating_point<WeightT_>::value)
        delta_ = static_cast<WeightT_>(atof(opt_arg));
      else
        delta_ = static_cast<WeightT_>(atol(opt_arg));
      break;
    default:
      CLApp::HandleArg(opt, opt_arg);
    }
  }

  WeightT_ delta() const { return delta_; }
};

class CLConvert : public CLBase {
  std::string out_filename_ = "";
  bool out_weighted_ = false;
  bool out_el_ = false;
  bool out_sg_ = false;
  int64_t src_ = -1;

public:
  CLConvert(int argc, char **argv, std::string name)
      : CLBase(argc, argv, name) {
    get_args_ += "e:b:wr:";
    AddHelpLine('b', "file", "output serialized graph to file");
    AddHelpLine('e', "file", "output edge list to file");
    AddHelpLine('w', "", "make output weighted");
    AddHelpLine('r', "-1", "Source/Root node");
  }

  void HandleArg(signed char opt, char *opt_arg) override {
    switch (opt) {
    case 'b':
      out_sg_ = true;
      out_filename_ = std::string(opt_arg);
      break;
    case 'e':
      out_el_ = true;
      out_filename_ = std::string(opt_arg);
      break;
    case 'w':
      out_weighted_ = true;
      break;
    case 'r':
      src_ = std::atoll(opt_arg);
    default:
      CLBase::HandleArg(opt, opt_arg);
    }
  }

  std::string out_filename() const { return out_filename_; }
  bool out_weighted() const { return out_weighted_; }
  bool out_el() const { return out_el_; }
  bool out_sg() const { return out_sg_; }
  int64_t src() const { return src_; }
};

#endif // COMMAND_LINE_H_
