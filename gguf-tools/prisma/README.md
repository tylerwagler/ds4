# Prisma allocator for ds4 mixed quants

Per-layer expert-format allocation for the ds4 GGUF quantizer, after
RobTand/PrismaQuant. Two cost models:

- **Measured KL (preferred, AURA-era):** real end-to-end KL divergence per
  layer promotion, measured on this engine with `measure_layer_kl.py`.
  Upstream's finding applies verbatim here: for routed MoE experts, gradient
  and imatrix proxies are blind to router flips — measure, don't estimate.
- **Proxy (fallback):** `dKL ~= 0.5 * sensitivity * mse` (imatrix Fisher proxy
  × round-trip MSE from `deepseek4-quantize --mse-probe`).

## Pipeline

```
# 0. one-time assets: all-cheap base + all-rich (lossless MXFP4) donor
deepseek4-quantize --hf HF_DIR --template T.gguf \
    --routed-w1 iq2_xxs --routed-w2 q2_k --routed-w3 iq2_xxs \
    --imatrix imatrix.dat --out allcheap-base.gguf
deepseek4-quantize --hf HF_DIR --template T.gguf --experts mxfp4 \
    --out allrich-donor.gguf

# 1. measure per-layer promotion KL (43 serial engine runs; overnight)
prisma/measure_layer_kl.py --ds4 ./ds4 --base allcheap-base.gguf \
    --donor allrich-donor.gguf \
    --calib gguf-tools/imatrix/dataset/rendered_prompts.txt --out kl.json

# 2. allocate at byte budgets (exact 0/1 DP knapsack)
dump_inventory.py MODEL.gguf > tensors.json
prisma_alloc.py --tensors tensors.json --kl kl.json --pareto 90 99 105
#   optional: --ucb-z 1.0 charges 1×stderr against measured benefits

# 3. produce the GGUF from the chosen manifest
deepseek4-quantize --hf HF_DIR --template T.gguf \
    --imatrix imatrix.dat --format-map manifest.99gb.json --out OUT.gguf
```

## Notes

- Allocation unit = whole MoE layer (gate+up+down pick one preset: **cheap**
  IQ2_XXS/Q2_K ~2.2 bpw, or **rich** MXFP4 4.25 bpw, byte-lossless vs the HF
  source). Per-expert formats are structurally unavailable: the GGUF stacks
  all 256 experts per tensor.
- `measure_layer_kl.py` needs the engine's `--expert-overlay` +
  `--kl-file/--kl-ref-dump/--kl-score` (see `ds4 --help-full`); resumable,
  one instance at a time.
- Rich layers served on the tensor-core prefill path additionally want the
  CUTLASS type-40 splice (`converter/splice_cutlass_mxfp4.py`) as a post-step;
  plain type-39 MXFP4 decodes identically but skips prefill batching.
- Proxy-path inputs: `extract_sens.py imatrix.dat > sens.json`, and
  `deepseek4-quantize --mse-probe mse.json` (writes `.sens.json` sidecar).
  Without any inputs a documented SYNTHETIC fallback runs so the machinery is
  testable — those manifests are size-driven only, not real allocations.
