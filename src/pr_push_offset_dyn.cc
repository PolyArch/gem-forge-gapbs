
// Use edge index offset and dynamic schedule.
#define USE_EDGE_INDEX_OFFSET
#define OMP_SCHEDULE_TYPE schedule(dynamic, 1024)
#include "pr_push.cc"