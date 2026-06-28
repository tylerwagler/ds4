#!/usr/bin/env python3
"""extract_sens.py — per-tensor sensitivity (Fisher proxy) from a legacy
llama.cpp/ds4 imatrix .dat. Format: int32 n_entries; per entry int32 name_len,
name, int32 ncall, int32 nval, float[nval] (values divided by ncall).

Sensitivity = mean importance over the tensor's columns (activation 2nd-moment).
Absolute scale is irrelevant — the allocator uses relative dKL ratios.
Usage: python3 extract_sens.py imatrix.dat > sens.json
"""
import struct, sys, json

def ri32(f): return struct.unpack("<i", f.read(4))[0]

f = open(sys.argv[1], "rb")
n = ri32(f)
sens = {}
for _ in range(n):
    L = ri32(f); name = f.read(L).decode("utf-8", "replace")
    ncall = ri32(f); nval = ri32(f)
    vals = struct.unpack("<%df" % nval, f.read(4 * nval))
    if ncall > 0:
        s = sum(vals) / (ncall * nval)
    else:
        s = sum(vals) / nval
    sens[name] = s
json.dump(sens, sys.stdout)
