CC ?= cc
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
NATIVE_CPU_FLAG ?= -mcpu=native
QUALITY_LDLIBS := -lm -pthread -framework Foundation -framework Metal
QUALITY_TARGETS := ds4.o ds4_metal.o
QUALITY_OBJS := ../ds4.o ../ds4_metal.o
QUALITY_LINK := $(CC) $(CFLAGS) -I.. -o
else
NATIVE_CPU_FLAG ?= -march=native
CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?=
ifneq ($(strip $(CUDA_ARCH)),)
NVCC_ARCH_FLAGS := -arch=$(CUDA_ARCH)
endif
NVCCFLAGS ?= -O3 --use_fast_math $(NVCC_ARCH_FLAGS) -Xcompiler $(NATIVE_CPU_FLAG) -Xcompiler -pthread
QUALITY_LDLIBS ?= -lm -Xcompiler -pthread -L$(CUDA_HOME)/targets/sbsa-linux/lib -L$(CUDA_HOME)/lib64 -lcudart -lcublas -lcublasLt
# post-restructure: link against the split engine/cuda TUs (built via the
# ds4-bench target, which forces every core object)
QUALITY_TARGETS := ds4-bench
QUALITY_OBJS = ../src/engine/*.o ../src/cuda/*.o ../src/lib/ds4_help.o ../src/vendor/rax.o ../src/lib/ds4_kvstore.o
QUALITY_LINK := $(NVCC) $(NVCCFLAGS) -I../src -o
endif

CFLAGS ?= -O3 -Wall -Wextra -std=c11 $(NATIVE_CPU_FLAG)
CPPFLAGS ?= -D_GNU_SOURCE

QUANTIZE_SRCS := $(wildcard quantize/dsq_*.c)
QUANTIZE_OBJS := $(QUANTIZE_SRCS:.c=.o)
QUANTS_SRCS := quants_common.c quants_fp.c quants_kquants.c
QUANTS_OBJS := $(QUANTS_SRCS:.c=.o)

.PHONY: all clean quality-score test test-cutlass-mxfp4

all: deepseek4-quantize

test_quants: test_quants.c $(QUANTS_OBJS)
	$(CC) $(CFLAGS) $(CPPFLAGS) -I. -o $@ test_quants.c $(QUANTS_OBJS) -lm -pthread

# test also rebuilds the production binary: a stale deepseek4-quantize
# once shipped sign-stripped MXFP4 donors while test_quants (rebuilt) passed.
test: test_quants deepseek4-quantize
	./test_quants

# Byte-compare the quantizer's native cutlass_mxfp4 (type 40) output against
# a reference GGUF whose type-40 tensors were produced by the validated
# CUTLASS packer (the retired converter/splice_cutlass_mxfp4.py pipeline).
# Needs the QAT HF snapshot and a reference GGUF on disk; override the paths
# as needed. Checks one gate and one down tensor (different N/K orientation).
DSV4_HF ?= $(HOME)/models/dsv4-flash-dspark
CUTLASS_REF_GGUF ?= ../gguf/dspark.gguf
test-cutlass-mxfp4: deepseek4-quantize
	./deepseek4-quantize --hf $(DSV4_HF) --template $(CUTLASS_REF_GGUF) \
	    --compare-tensor dspark.0.ffn_gate_exps.weight | tee /tmp/dsq-t40-gate.txt
	grep -q '^byte_compare: OK' /tmp/dsq-t40-gate.txt
	./deepseek4-quantize --hf $(DSV4_HF) --template $(CUTLASS_REF_GGUF) \
	    --compare-tensor dspark.0.ffn_down_exps.weight | tee /tmp/dsq-t40-down.txt
	grep -q '^byte_compare: OK' /tmp/dsq-t40-down.txt
	@echo "cutlass_mxfp4 byte-compare: OK"

quantize/%.o: quantize/%.c quantize/dsq_internal.h quants.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -I. -Iquantize -c -o $@ $<

%.o: %.c quants_internal.h quants.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -I. -c -o $@ $<

deepseek4-quantize: $(QUANTIZE_OBJS) $(QUANTS_OBJS)
	$(CC) $(CFLAGS) -o $@ $(QUANTIZE_OBJS) $(QUANTS_OBJS) -lm -pthread

quality-score:
	$(MAKE) -C .. $(QUALITY_TARGETS)
	$(QUALITY_LINK) quality-testing/score_official quality-testing/score_official.c $(QUALITY_OBJS) $(QUALITY_LDLIBS)

clean:
	rm -f deepseek4-quantize test_quants quality-testing/score_official quantize/*.o *.o
