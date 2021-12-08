
// Use double ScoreT with only the atomic kernel
#define DISABLE_KERNEL2
#define OMP_SCHEDULE_TYPE schedule(dynamic, 1024)
#include "pr_push.cc"