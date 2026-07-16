CC ?= cc
NATIVE_CPU_FLAG ?= -march=native

DEBUG_FLAGS ?= -g
CFLAGS ?= -O3 -ffast-math $(DEBUG_FLAGS) $(NATIVE_CPU_FLAG) -Wall -Wextra -std=c99
CFLAGS += -D_GNU_SOURCE -fno-finite-math-only

CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?=
ifneq ($(strip $(CUDA_ARCH)),)
NVCC_ARCH_FLAGS := -arch=$(CUDA_ARCH)
endif
NVCCFLAGS ?= -O3 -g -lineinfo --use_fast_math --default-stream per-thread $(NVCC_ARCH_FLAGS) -Xcompiler $(NATIVE_CPU_FLAG) -Xcompiler -pthread

CUTLASS_DIR ?= $(CURDIR)/cutlass
CUTLASS_INC ?= -I$(CUTLASS_DIR)/include -I$(CUTLASS_DIR)/tools/util/include
CUDA_LDLIBS ?= -lm -Xcompiler -pthread -L$(CUDA_HOME)/targets/sbsa-linux/lib -L$(CUDA_HOME)/lib64 -lcudart -lcublas -lcublasLt

DS4_INC = -Isrc -Isrc/lib -Isrc/vendor

ENGINE_SRCS = $(wildcard src/engine/*.c)
ENGINE_OBJS = $(ENGINE_SRCS:.c=.o)
AGENT_SRCS = $(wildcard src/agent/*.c)
AGENT_OBJS = $(AGENT_SRCS:.c=.o)
SERVER_SRCS = $(wildcard src/server/*.c)
SERVER_OBJS = $(SERVER_SRCS:.c=.o)
# CUTLASS TUs need the CUTLASS include path + c++17; they build via dedicated rules below,
# so keep them out of the generic src/cuda/%.o rule.
CUTLASS_CUDA_OBJS = src/cuda/ds4_mxfp4_cutlass.o
CUDA_SRCS = $(filter-out src/cuda/ds4_mxfp4_cutlass.cu,$(wildcard src/cuda/*.cu))
CUDA_OBJS = $(CUDA_SRCS:.cu=.o)
LIB_HDRS = src/lib/ds4_help.h src/lib/ds4_kvstore.h
CORE_OBJS = $(ENGINE_OBJS) $(CUDA_OBJS) $(CUTLASS_CUDA_OBJS)
DS4_LINK ?= $(NVCC) $(NVCCFLAGS)
DS4_LINK_LIBS ?= $(CUDA_LDLIBS)

.PHONY: all help clean test cuda-spark cuda-regression cuda-frontier-gate cuda-multiseq-gate cuda-multiseq-gate-nodspark cuda-spec-sampling-gate

all: help

help:
	@echo "DS4 build targets (CUDA-only fork for DGX Spark / GB10):"
	@echo "  make cuda-spark          Build for the DGX Spark / GB10 (sm_120f)"
	@echo "  make test                Build and run tests"
	@echo "  make cuda-regression     Kernel smokes vs synthetic slabs (modelless)"
	@echo "  make cuda-frontier-gate  Multiseq frontier-isolation gate (needs the model;"
	@echo "                           FRONTIER_MODEL=./ds4flash.gguf by default)"
	@echo "  make cuda-multiseq-gate  Multiseq-vs-solo token-stream gate + aggregate"
	@echo "                           throughput at N=1..3 (needs the model)"
	@echo "  make cuda-multiseq-gate-nodspark"
	@echo "                           The same gate with speculation disabled"
	@echo "                           (--no-dspark config; needs the model)"
	@echo "  make cuda-spec-sampling-gate"
	@echo "                           Speculative-sampling chi-square exactness"
	@echo "                           oracle + acceptance alpha (needs the model)"
	@echo "  make clean               Remove build outputs"

cuda-spark:
	$(MAKE) -B ds4-server CUDA_ARCH=sm_120f

ds4-server: $(SERVER_OBJS) src/lib/ds4_help.o src/lib/ds4_kvstore.o src/vendor/rax.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

# Development tools, not part of the shipped release. The release is just
# ds4-server (the HTTP API). Source is kept; build these by name.
ds4: src/cli/ds4_cli.o src/lib/ds4_help.o src/vendor/linenoise.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

ds4-bench: src/cli/ds4_bench.o src/lib/ds4_help.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

ds4-eval: src/cli/ds4_eval.o src/lib/ds4_help.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

ds4-agent: $(AGENT_OBJS) src/lib/ds4_help.o src/lib/ds4_kvstore.o src/vendor/linenoise.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

cuda-regression: tests/cuda_long_context_smoke
	./tests/cuda_long_context_smoke

# Multiseq frontier-isolation gate (engine-side wrong-bank wiring; see the
# header of tests/multiseq_frontier_gate.c).  MODEL-DEPENDENT — run manually
# on the GB10, not part of `make test`.  Discipline before running: no
# ds4-server/ds4_test process, `sync; echo 3 > /proc/sys/vm/drop_caches`,
# check `free -g` headroom (the model is ~87 GB).
FRONTIER_MODEL ?= ./ds4flash.gguf
cuda-frontier-gate: tests/multiseq_frontier_gate
	DS4_MSEQ_BANKS=2 ./tests/multiseq_frontier_gate $(FRONTIER_MODEL)

# Multiseq-vs-solo token-stream gate + first aggregate-throughput measurement
# (see the header of tests/multiseq_decode_gate.c).  MODEL-DEPENDENT — run
# manually on the GB10 with the same memory discipline as the frontier gate.
cuda-multiseq-gate: tests/multiseq_decode_gate
	DS4_MSEQ_BANKS=3 ./tests/multiseq_decode_gate $(FRONTIER_MODEL) 3 512

# The same gate with speculation DISABLED — the ds4-bench/ds4-eval/agent and
# `ds4-server --no-dspark` config, and a different allocation shape (no DSpark
# graph state).  The driver must work there; it used to reject every step.
# Shorter (N=2, 64 steps): this is a config gate, not a throughput run.
cuda-multiseq-gate-nodspark: tests/multiseq_decode_gate
	DS4_MSEQ_BANKS=2 DS4_GATE_NO_DSPARK=1 ./tests/multiseq_decode_gate $(FRONTIER_MODEL) 2 64

# Speculative-sampling exactness oracle: chi-square of plain-sampled vs
# speculative per-position marginals, plus the greedy prefix gate and the
# acceptance rate alpha (see the header of tests/spec_sampling_gate.c).
# Proposal-agnostic, so it gates both the deterministic and the
# temperature-matched (p/q) accept rules.  MODEL-DEPENDENT — same memory
# discipline as the gates above; not part of `make test`.
SPEC_GATE_MODEL ?= gguf/model.gguf
cuda-spec-sampling-gate: tests/spec_sampling_gate
	./tests/spec_sampling_gate $(SPEC_GATE_MODEL) 0.95

src/engine/%.o: src/engine/%.c src/engine/ds4_engine_internal.h src/ds4.h src/ds4_gpu.h
	$(CC) $(CFLAGS) $(DS4_INC) -c -o $@ $<

src/agent/%.o: src/agent/%.c src/agent/ds4_agent_internal.h src/ds4.h $(LIB_HDRS) src/vendor/linenoise.h
	$(CC) $(CFLAGS) $(DS4_INC) -c -o $@ $<

src/server/%.o: src/server/%.c src/server/ds4_server_internal.h src/ds4.h $(LIB_HDRS) src/vendor/rax.h
	$(CC) $(CFLAGS) $(DS4_INC) -c -o $@ $<

src/cli/%.o: src/cli/%.c src/ds4.h src/lib/ds4_help.h src/vendor/linenoise.h
	$(CC) $(CFLAGS) $(DS4_INC) -c -o $@ $<

src/lib/%.o: src/lib/%.c src/ds4.h $(LIB_HDRS)
	$(CC) $(CFLAGS) $(DS4_INC) -c -o $@ $<

src/vendor/%.o: src/vendor/%.c src/vendor/linenoise.h src/vendor/rax.h src/vendor/rax_malloc.h
	$(CC) $(CFLAGS) -c -o $@ $<

tests/ds4_test.o: tests/ds4_test.c $(SERVER_SRCS) src/server/ds4_server_internal.h src/ds4.h $(LIB_HDRS) src/vendor/rax.h
	$(CC) $(CFLAGS) $(DS4_INC) -Wno-unused-function -c -o $@ tests/ds4_test.c

tests/ds4_agent_test.o: tests/ds4_agent_test.c $(AGENT_SRCS) src/agent/ds4_agent_internal.h src/ds4.h $(LIB_HDRS) src/vendor/linenoise.h
	$(CC) $(CFLAGS) $(DS4_INC) -Wno-unused-function -c -o $@ tests/ds4_agent_test.c

tests/cuda_long_context_smoke.o: tests/cuda_long_context_smoke.c src/ds4_gpu.h
	$(CC) $(CFLAGS) -Isrc -c -o $@ tests/cuda_long_context_smoke.c

tests/multiseq_frontier_gate.o: tests/multiseq_frontier_gate.c src/engine/ds4_engine_internal.h src/ds4.h src/ds4_gpu.h
	$(CC) $(CFLAGS) $(DS4_INC) -Isrc/engine -c -o $@ tests/multiseq_frontier_gate.c

tests/multiseq_decode_gate.o: tests/multiseq_decode_gate.c src/engine/ds4_engine_internal.h src/ds4.h src/ds4_gpu.h
	$(CC) $(CFLAGS) $(DS4_INC) -Isrc/engine -c -o $@ tests/multiseq_decode_gate.c

tests/spec_sampling_gate.o: tests/spec_sampling_gate.c src/ds4.h
	$(CC) $(CFLAGS) $(DS4_INC) -c -o $@ tests/spec_sampling_gate.c

src/cuda/%.o: src/cuda/%.cu src/cuda/ds4_cuda_internal.h src/ds4_gpu.h src/cuda/ds4_iq2_tables_cuda.inc
	$(NVCC) $(NVCCFLAGS) -Isrc -c -o $@ $<

# CUTLASS MXFP4 tensor-core expert FFN (GB10/sm_120f). Requires -arch=sm_120f (family mode) for the
# mxf4 block-scale MMA; build the whole engine with CUDA_ARCH=sm_120f so all objects match arch.
src/cuda/ds4_mxfp4_cutlass.o: src/cuda/ds4_mxfp4_cutlass.cu src/ds4_gpu.h
	$(NVCC) $(NVCCFLAGS) -std=c++17 --expt-relaxed-constexpr --expt-extended-lambda -Isrc $(CUTLASS_INC) -c -o $@ src/cuda/ds4_mxfp4_cutlass.cu

tests/cuda_long_context_smoke: tests/cuda_long_context_smoke.o $(CUDA_OBJS) $(CUTLASS_CUDA_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/multiseq_frontier_gate: tests/multiseq_frontier_gate.o src/lib/ds4_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/multiseq_decode_gate: tests/multiseq_decode_gate.o src/lib/ds4_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/spec_sampling_gate: tests/spec_sampling_gate.o src/lib/ds4_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4_test: tests/ds4_test.o src/lib/ds4_help.o src/lib/ds4_kvstore.o src/vendor/rax.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ tests/ds4_test.o src/lib/ds4_help.o src/lib/ds4_kvstore.o src/vendor/rax.o $(CORE_OBJS) $(CUDA_LDLIBS)

ds4_agent_test: tests/ds4_agent_test.o src/lib/ds4_help.o src/lib/ds4_kvstore.o src/vendor/linenoise.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ tests/ds4_agent_test.o src/lib/ds4_help.o src/lib/ds4_kvstore.o src/vendor/linenoise.o $(CORE_OBJS) $(CUDA_LDLIBS)

test: ds4_test
	./ds4_test

clean:
	rm -f ds4 ds4-server ds4-bench ds4-eval ds4-agent ds4_test ds4_agent_test src/engine/*.o src/agent/*.o src/server/*.o src/cuda/*.o src/cli/*.o src/lib/*.o src/vendor/*.o tests/*.o tests/cuda_long_context_smoke tests/multiseq_frontier_gate tests/multiseq_decode_gate tests/spec_sampling_gate
