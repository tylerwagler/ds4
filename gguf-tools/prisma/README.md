# Prisma allocator (Rob's method) for ds4 mixed quants

Sensitivity-driven per-layer format allocation for the ds4 GGUF quantizer,
after RobTand/PrismaQuant: cost `dKL ~= 0.5 * sensitivity * mse`, solved as a
multiple-choice knapsack (greedy-upgrade) at a target size budget.

- `dump_inventory.py MODEL.gguf > tensors.json` — tensor name/shape/family/type.
- `prisma_alloc.py --tensors tensors.json [--sens sens.json] [--mse mse.json]
   --pareto 90 99 105` — emits `manifest.<gb>gb.json` (per-tensor format).
- Lever = the 129 routed-expert tensors (~91% of the model); non-experts pinned.
- Each MoE layer picks one preset: **cheap** (IQ2_XXS gate/up + Q2_K down,
  ~2.2 bpw) or **rich** (MXFP4 all three, 4.25 bpw, lossless vs source).
- `sens.json` from the imatrix (Fisher proxy); `mse.json` = per-(tensor,format)
  round-trip MSE from `deepseek4-quantize` (the MSE oracle). Without them a
  documented SYNTHETIC fallback runs so the machinery is testable.
- Feed a chosen manifest to `deepseek4-quantize --format-map manifest.json`.
