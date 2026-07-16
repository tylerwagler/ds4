#!/usr/bin/env python3
"""Parse an nsys cuda_gpu_kern_sum CSV into prefill composition buckets.
Usage: parse_prefill.py <nsys-rep-or-csv> [label]
Buckets kernels by substring match; prints ms-per-bucket and % of tracked GPU time.
"""
import sys, subprocess, csv, io, os

# ordered list: (bucket, [substrings]); first match wins
RULES = [
    ("MoE MXFP4 gate/up", ["moe_gate_up_mid_mxfp4_expert_ntile", "moe_gate_up_mid_mxfp4_qwarp", "moe_gate_up_mid_decode_lut"]),
    ("MoE MXFP4 down",    ["moe_down_mxfp4_expert_tile16", "moe_down_mxfp4_sum6", "moe_down_mxfp4_qwarp"]),
    ("MoE IQ2 gate/up",   ["moe_gate_up_mid_expert_tile8_rowspan", "moe_gate_up_mid_expert_tile8_row32", "moe_gate_up_mid_qwarp32"]),
    ("MoE IQ2 down",      ["moe_down_iq2_expert_tile16", "moe_down_iq2_expert_tile8", "moe_down_iq2_sum6", "moe_down_iq2_qwarp", "moe_down_expert_tile16", "moe_down_expert_tile8", "moe_down_sum6_qwarp", "moe_down_qwarp32"]),
    ("MoE support/sort",  ["moe_build_expert_tile", "moe_count_sorted", "moe_prefix_sorted", "moe_scatter_sorted", "moe_sum_kernel", "moe_padded", "moe_cutlass_gather", "moe_cutlass_scatter", "expert_gemv"]),
    ("Attn core",         ["attention_prefill", "attention_indexed", "attention_static", "attention_decode"]),
    ("Proj/dense GEMM (attn lora+shared+hc)", ["mxfp8_mmvq", "matmul_fp8mx_hc_expand", "grouped_fp8mx_a", "pack_act_rowmajor", "matmul_f16", "matmul_bf16", "matmul_f32", "cutlass", "device_kernel", "cublas", "ampere_", "sm120_", "sm100_", "nvjet", "gemm", "Gemm", "wgrad", "s16816", "cgemm"]),
    ("Indexer",           ["indexer_", "indexed_topk"]),
    ("Compressor/KV",     ["compressor_", "mxkv_", "fp8_kv", "pack_fp8_kv", "attn_pack"]),
    ("Norms",             ["rms_norm", "qkv_rms", "head_rms"]),
    ("HC / hyper-conn",   ["hc_expand", "hc_split", "hc_weighted", "output_hc_weights", "embed_token", "embed_tokens_hc"]),
    ("RoPE",              ["rope_tail"]),
    ("Staging quant/cast", ["f32_to_f16", "f32_to_bf16", "mxfp8_quant_act", "q8_K_quantize", "swiglu", "pack_weight_f32", "fill_f32", "batched_copy", "fp4_act_roundtrip"]),
    ("Router", ["router_select", "argmax"]),
    ("Weight convert (1-time)", ["mxfp8_weight_convert"]),
    ("Router/topk",       ["argmax", "indexed_topk_sort_512"]),
    ("Steering/dspark",   ["directional_steering", "dspark_"]),
    ("misc/add",          ["add_kernel"]),
]

def load_csv(path):
    if path.endswith(".csv"):
        return open(path).read()
    # run nsys stats
    out = subprocess.run(["nsys","stats","--report","cuda_gpu_kern_sum","--format","csv","--force-export=true",path],
                         capture_output=True, text=True)
    return out.stdout

def main():
    path = sys.argv[1]
    label = sys.argv[2] if len(sys.argv)>2 else os.path.basename(path)
    txt = load_csv(path)
    # find header line
    lines = txt.splitlines()
    start = next((i for i,l in enumerate(lines) if l.startswith('"Time') or l.startswith('Time') or 'Total Time' in l), 0)
    reader = csv.DictReader(io.StringIO("\n".join(lines[start:])))
    buckets = {}
    unmatched = {}
    total = 0.0
    for row in reader:
        name = row.get('Name') or row.get('name') or ''
        # total time column varies
        tkey = next((k for k in row if k and ('Total Time' in k or k.strip()=='Total Time (ns)')), None)
        if tkey is None:
            tkey = next((k for k in row if k and 'ns' in k.lower() and 'total' in k.lower()), None)
        try:
            t_ns = float((row[tkey] or '0').replace(',',''))
        except: continue
        ms = t_ns/1e6
        total += ms
        placed=False
        for bucket,subs in RULES:
            if any(s in name for s in subs):
                buckets[bucket]=buckets.get(bucket,0.0)+ms; placed=True; break
        if not placed:
            unmatched[name]=unmatched.get(name,0.0)+ms
            buckets["UNMATCHED"]=buckets.get("UNMATCHED",0.0)+ms
    print(f"\n===== {label} =====")
    print(f"total tracked GPU kernel time: {total:.2f} ms")
    for bucket,_ in RULES + [("UNMATCHED",None)]:
        if bucket in buckets:
            ms=buckets[bucket]
            print(f"  {bucket:28s} {ms:9.2f} ms  {100*ms/total:5.1f}%")
    if unmatched:
        print("  -- top unmatched --")
        for n,ms in sorted(unmatched.items(),key=lambda x:-x[1])[:15]:
            print(f"     {ms:8.2f} ms  {n[:80]}")

if __name__=="__main__":
    main()
