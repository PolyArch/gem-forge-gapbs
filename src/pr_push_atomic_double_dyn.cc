
// Use double ScoreT with only the atomic kernel
#define DISABLE_KERNEL2
#define USE_DOUBLE_SCORE_T
#define OMP_SCHEDULE_TYPE schedule(dynamic, 1024)
#include "pr_push.cc"