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
NVCCFLAGS ?= -O3 -g -lineinfo --use_fast_math $(NVCC_ARCH_FLAGS) -Xcompiler $(NATIVE_CPU_FLAG) -Xcompiler -pthread

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
LIB_HDRS = src/lib/ds4_help.h src/lib/ds4_kvstore.h src/lib/ds4_ssd.h src/lib/ds4_web.h
CORE_OBJS = $(ENGINE_OBJS) src/lib/ds4_ssd.o $(CUDA_OBJS) $(CUTLASS_CUDA_OBJS)
DS4_LINK ?= $(NVCC) $(NVCCFLAGS)
DS4_LINK_LIBS ?= $(CUDA_LDLIBS)

.PHONY: all help clean test cuda cuda-spark cuda-generic cuda-regression

all: help

help:
	@echo "DS4 build targets (CUDA-only fork for DGX Spark / GB10):"
	@echo "  make cuda-spark          Build CUDA for DGX Spark / GB10"
	@echo "  make cuda-generic        Build CUDA for a generic local CUDA GPU"
	@echo "  make cuda CUDA_ARCH=sm_N Build CUDA with an explicit nvcc -arch value"
	@echo "  make test                Build and run tests"
	@echo "  make clean               Remove build outputs"

cuda-spark:
	$(MAKE) -B ds4 ds4-server ds4-bench ds4-eval ds4-agent CUDA_ARCH=sm_120f

cuda-generic:
	$(MAKE) -B ds4 ds4-server ds4-bench ds4-eval ds4-agent CUDA_ARCH=native

cuda:
	@if [ -z "$(strip $(CUDA_ARCH))" ]; then \
		echo "error: specify CUDA_ARCH, for example: make cuda CUDA_ARCH=sm_120"; \
		echo "       or use make cuda-spark / make cuda-generic"; \
		exit 2; \
	fi
	$(MAKE) -B ds4 ds4-server ds4-bench ds4-eval ds4-agent CUDA_ARCH="$(CUDA_ARCH)"

ds4: src/cli/ds4_cli.o src/lib/ds4_help.o src/vendor/linenoise.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

ds4-server: $(SERVER_OBJS) src/lib/ds4_help.o src/lib/ds4_kvstore.o src/vendor/rax.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

ds4-bench: src/cli/ds4_bench.o src/lib/ds4_help.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

ds4-eval: src/cli/ds4_eval.o src/lib/ds4_help.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

ds4-agent: $(AGENT_OBJS) src/lib/ds4_help.o src/lib/ds4_web.o src/lib/ds4_kvstore.o src/vendor/linenoise.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

cuda-regression: tests/cuda_long_context_smoke
	./tests/cuda_long_context_smoke

src/engine/%.o: src/engine/%.c src/engine/ds4_engine_internal.h src/ds4.h src/lib/ds4_ssd.h src/ds4_gpu.h
	$(CC) $(CFLAGS) $(DS4_INC) -c -o $@ $<

src/agent/%.o: src/agent/%.c src/agent/ds4_agent_internal.h src/ds4.h $(LIB_HDRS) src/vendor/linenoise.h
	$(CC) $(CFLAGS) $(DS4_INC) -c -o $@ $<

src/server/%.o: src/server/%.c src/server/ds4_server_internal.h src/ds4.h $(LIB_HDRS) src/vendor/rax.h
	$(CC) $(CFLAGS) $(DS4_INC) -c -o $@ $<

src/cli/%.o: src/cli/%.c src/ds4.h src/lib/ds4_ssd.h src/lib/ds4_help.h src/vendor/linenoise.h
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

src/cuda/%.o: src/cuda/%.cu src/cuda/ds4_cuda_internal.h src/ds4_gpu.h src/cuda/ds4_iq2_tables_cuda.inc
	$(NVCC) $(NVCCFLAGS) -Isrc -c -o $@ $<

# CUTLASS MXFP4 tensor-core expert FFN (GB10/sm_120f). Requires -arch=sm_120f (family mode) for the
# mxf4 block-scale MMA; build the whole engine with CUDA_ARCH=sm_120f so all objects match arch.
src/cuda/ds4_mxfp4_cutlass.o: src/cuda/ds4_mxfp4_cutlass.cu src/ds4_gpu.h
	$(NVCC) $(NVCCFLAGS) -std=c++17 --expt-relaxed-constexpr --expt-extended-lambda -Isrc $(CUTLASS_INC) -c -o $@ src/cuda/ds4_mxfp4_cutlass.cu

tests/cuda_long_context_smoke: tests/cuda_long_context_smoke.o $(CUDA_OBJS) $(CUTLASS_CUDA_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4_test: tests/ds4_test.o src/lib/ds4_help.o src/lib/ds4_kvstore.o src/vendor/rax.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ tests/ds4_test.o src/lib/ds4_help.o src/lib/ds4_kvstore.o src/vendor/rax.o $(CORE_OBJS) $(CUDA_LDLIBS)

ds4_agent_test: tests/ds4_agent_test.o src/lib/ds4_help.o src/lib/ds4_web.o src/lib/ds4_kvstore.o src/vendor/linenoise.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ tests/ds4_agent_test.o src/lib/ds4_help.o src/lib/ds4_web.o src/lib/ds4_kvstore.o src/vendor/linenoise.o $(CORE_OBJS) $(CUDA_LDLIBS)

test: ds4_test ds4_agent_test ds4-eval
	./ds4-eval --self-test-extractors
	./ds4_agent_test
	./ds4_test

clean:
	rm -f ds4 ds4-server ds4-bench ds4-eval ds4-agent ds4_test ds4_agent_test src/engine/*.o src/agent/*.o src/server/*.o src/cuda/*.o src/cli/*.o src/lib/*.o src/vendor/*.o tests/*.o tests/cuda_long_context_smoke
