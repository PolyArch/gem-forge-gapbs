# See LICENSE.txt for license details.

AFFINITY_ALLOC_INC_PATH=${GEM_FORGE_TOP}/lib/affinity_alloc
AFFINITY_ALLOC_LIB_PATH=${GEM_FORGE_TOP}/lib/affinity_alloc/build

GEM_FORGE_INC_PATH=${GEM_FORGE_TOP}/gem5/include

CXX_FLAGS += -std=c++11 -O3 -Wall -I${AFFINITY_ALLOC_INC_PATH} -I${GEM_FORGE_INC_PATH}
LINK_FLAGS += -L${AFFINITY_ALLOC_LIB_PATH} -lAffinityAlloc
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

KERNELS = bc bfs bfs_pull bfs_pull_shuffle cc cc_sv tc
KERNELS += pr_pull
KERNELS += pr_pull_shuffle
KERNELS += pr_push
KERNELS += pr_push_inter_part
KERNELS += pr_push_adj_rnd
KERNELS += pr_push_adj_lnr
KERNELS += pr_push_adj_aff
KERNELS += pr_push_adj_uno_aff
KERNELS += pr_push_adj_uno_aff_dyn256
KERNELS += pr_push_adj_mix2_aff
KERNELS += bfs_push
KERNELS += bfs_push_adj_rnd_spatial
KERNELS += bfs_push_adj_rnd_sf
KERNELS += bfs_push_spatial
KERNELS += bfs_push_offset
KERNELS += bfs_push_check
KERNELS += bfs_push_spatial_dyn128
KERNELS += bfs_push_spatial_dyn256
KERNELS += bfs_push_spatial_dyn512
KERNELS += bfs_push_spatial_guided
KERNELS += sssp_outline
KERNELS += sssp_check
KERNELS += sssp
KERNELS += sssp_sq_delta1
KERNELS += sssp_sq_delta2
KERNELS += sssp_sq_delta4
KERNELS += sssp_sq_delta8
KERNELS += sssp_sq_delta16
KERNELS += sssp_sq_delta32
KERNELS += sssp_sf_delta1
KERNELS += sssp_sf_delta2
KERNELS += sssp_sf_delta4
KERNELS += sssp_sf_delta8
KERNELS += sssp_sf_delta16
KERNELS += sssp_sf_delta32
KERNELS += sssp_adj_rnd_sf_delta1
KERNELS += sssp_adj_rnd_sf_delta2
KERNELS += sssp_adj_rnd_sf_delta4
KERNELS += sssp_adj_rnd_sf_delta8
KERNELS += sssp_adj_rnd_sf_delta16
KERNELS += sssp_adj_rnd_sf_delta32
KERNELS += sssp_adj_aff_sf_delta1
KERNELS += sssp_adj_uno_aff_sf_delta1

SUITE = $(KERNELS) converter gscore bound_dfs nuca_analysis

.PHONY: all
all: $(SUITE)

build_adj_list_graph: src/build_adj_list_graph.cc src/*.hh
	$(CXX) -DUSE_AFFINITY_ALLOC $(CXX_FLAGS) $< $(LINK_FLAGS) -o $@

%: src/%.cc src/*.h src/*.cc
	$(CXX) $(CXX_FLAGS) $< $(LINK_FLAGS) -o $@

# Testing
include test/test.mk

# Benchmark Automation
include benchmark/bench.mk


.PHONY: clean
clean:
	rm -f $(SUITE) test/out/*
