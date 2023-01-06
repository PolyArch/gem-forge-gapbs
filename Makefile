# See LICENSE.txt for license details.

CXX_FLAGS += -std=c++11 -O3 -Wall
PAR_FLAG = -fopenmp

ifneq (,$(findstring icpc,$(CXX)))
	PAR_FLAG = -openmp
endif

ifneq (,$(findstring sunCC,$(CXX)))
	CXX_FLAGS = -std=c++11 -xO3 -m64 -xtarget=native
	PAR_FLAG = -xopenmp
endif

ifneq ($(SERIAL), 1)
	CXX_FLAGS += $(PAR_FLAG)
endif

KERNELS = bc bfs bfs_pull bfs_pull_shuffle cc cc_sv pr_pull pr_pull_shuffle pr_push sssp sssp_check sssp_inline tc
KERNELS += bfs_push
KERNELS += bfs_push_spatial
KERNELS += bfs_push_offset
KERNELS += bfs_push_check
KERNELS += bfs_push_spatial_dyn128
KERNELS += bfs_push_spatial_dyn256
KERNELS += bfs_push_spatial_dyn512
KERNELS += bfs_push_spatial_guided
SUITE = $(KERNELS) converter gscore bound_dfs nuca_analysis

.PHONY: all
all: $(SUITE)

% : src/%.cc src/*.h
	$(CXX) $(CXX_FLAGS) $< -o $@

# Testing
include test/test.mk

# Benchmark Automation
include benchmark/bench.mk


.PHONY: clean
clean:
	rm -f $(SUITE) test/out/*
