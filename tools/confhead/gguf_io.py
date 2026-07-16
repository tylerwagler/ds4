#!/usr/bin/env python3
"""Minimal GGUF read/patch helpers for the confidence-head pipeline.

Only what the pipeline needs: parse the tensor directory (honoring
general.alignment), read an F32 tensor, and byte-patch an F32 tensor in place.
Works on both the small sidecar drafter ggufs and the 91 GB merged artifacts
(the parse never reads tensor data, only the header walk).
"""
import struct
import numpy as np

def parse(path):
    """Return ({name: (type, dims, rel_offset)}, data_start)."""
    f = open(path, "rb")
    assert f.read(4) == b"GGUF", path
    struct.unpack("<I", f.read(4))                       # version
    nt = struct.unpack("<Q", f.read(8))[0]
    nkv = struct.unpack("<Q", f.read(8))[0]
    align = 32
    def rs():
        n = struct.unpack("<Q", f.read(8))[0]
        return f.read(n)
    def rv(t):
        if t == 8:
            return rs()
        if t == 9:
            et = struct.unpack("<I", f.read(4))[0]
            n = struct.unpack("<Q", f.read(8))[0]
            return [rv(et) for _ in range(n)]
        sz = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
        fm = {0:'<b',1:'<B',2:'<h',3:'<H',4:'<i',5:'<I',6:'<f',7:'<?',10:'<q',11:'<Q',12:'<d'}
        return struct.unpack(fm[t], f.read(sz[t]))[0]
    for _ in range(nkv):
        k = rs()
        t = struct.unpack("<I", f.read(4))[0]
        v = rv(t)
        if k == b"general.alignment":
            align = int(v)
    tens = {}
    for _ in range(nt):
        nb = struct.unpack("<Q", f.read(8))[0]
        name = f.read(nb).decode()
        nd = struct.unpack("<I", f.read(4))[0]
        dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]
        typ = struct.unpack("<I", f.read(4))[0]
        off = struct.unpack("<Q", f.read(8))[0]
        tens[name] = (typ, dims, off)
    data_start = (f.tell() + align - 1) // align * align
    f.close()
    return tens, data_start

def read_f32(path, name):
    tens, ds = parse(path)
    typ, dims, off = tens[name]
    assert typ == 0, f"{name}: expected F32 (0), got {typ}"
    ne = int(np.prod(dims))
    arr = np.fromfile(path, dtype=np.float32, count=ne, offset=ds + off)
    assert arr.size == ne, (name, arr.size, ne)
    return arr, dims

def patch_f32(path, name, arr):
    """Overwrite tensor `name` in `path` with `arr` (must match element count)."""
    tens, ds = parse(path)
    typ, dims, off = tens[name]
    assert typ == 0, f"{name}: expected F32 (0), got {typ}"
    ne = int(np.prod(dims))
    a = np.ascontiguousarray(arr, dtype=np.float32).reshape(-1)
    assert a.size == ne, f"{name}: {a.size} elements vs tensor {ne}"
    with open(path, "r+b") as f:
        f.seek(ds + off)
        f.write(a.tobytes())
    return ds + off, ne
