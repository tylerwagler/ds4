"""Run on pve2 CT117. Emit SOURCE MXFP4 arrays (E2M1 + E8M0, SEPARATE, stacked over 256 experts)
for a rich layer, for the CUTLASS pack step on Sparky. No 17-byte interleave — the CUTLASS packer
consumes the source's native separate arrays directly.
  usage: python3 pve_extract_source.py LAYER SHARD OUTDIR
Produces per tgt(gate/up/down): L{LAYER}_{tgt}.e2m1  [n_expert,out,in/2]  +  .e8m0 [n_expert,out,in/32]."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # portable: p0conv sits beside this script
from p0conv import SafeTensors

LAYER = int(sys.argv[1]); SHARD = sys.argv[2]; OUTDIR = sys.argv[3]
st = SafeTensors(SHARD)
for tgt, wk in [("gate", "w1"), ("up", "w3"), ("down", "w2")]:
    e2, e8, n = [], [], 0
    while True:
        k = f"layers.{LAYER}.ffn.experts.{n}.{wk}.weight"
        if k not in st.hdr: break
        e2.append(st.get(k))                                             # [out, in/2] uint8 (E2M1)
        e8.append(st.get(f"layers.{LAYER}.ffn.experts.{n}.{wk}.scale"))  # [out, in/32] uint8 (E8M0)
        n += 1
    e2 = np.stack(e2).astype(np.uint8); e8 = np.stack(e8).astype(np.uint8)   # [n_expert, out, *]
    e2.tofile(f"{OUTDIR}/L{LAYER}_{tgt}.e2m1"); e8.tofile(f"{OUTDIR}/L{LAYER}_{tgt}.e8m0")
    print(f"L{LAYER} {tgt:4} experts={n} out={e2.shape[1]} in={e2.shape[2]*2} "
          f"e2m1={e2.nbytes/1e6:.1f}MB e8m0={e8.nbytes/1e6:.1f}MB", flush=True)
