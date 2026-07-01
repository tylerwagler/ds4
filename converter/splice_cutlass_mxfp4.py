"""Sparky-side assembly: build the CUTLASS-MXFP4 GGUF from SOURCE. Stream oracle-zeroq8 verbatim,
but rewrite the 18 rich-layer expert tensors (layers 30,34,37,39,41,42 x gate/up/down) into CUTLASS
B layout (new type 40) packed FROM SOURCE E2M1+E8M0 (pve_extract_source.py output in SRCDIR).
Uses p0conv.pack_mxfp4_cutlass -> mxfp4_pack_source_cli. Per rich tensor: expert-major (data||sf) per
expert; per-expert stride = data(N*K/2) + sf(sf_count). Everything else copied byte-verbatim.

NOTE(cleanup): GB10-only fork drops the non-CUTLASS tiers; kept verbatim here for now.
  env: MXFP4_SRCDIR (source arrays), DS4_MXFP4_PACK_CLI (packer). Two-pass: gen blob files, then splice."""
import struct, os, sys, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
PACK_CLI = os.environ.get("DS4_MXFP4_PACK_CLI", os.path.expanduser("~/Projects/AI/temp/p0/mxfp4_pack_source_cli"))

SRC = "/home/tyler/Projects/AI/ds4-gguf/oracle-zeroq8-99gb.gguf"
OUT = "/home/tyler/Projects/AI/ds4-gguf/oracle-cutlass-mxfp4-99gb.gguf"
SRCDIR = os.environ.get("MXFP4_SRCDIR", "/home/tyler/Projects/AI/temp/p0/mxfp4_src")
BLOBDIR = os.environ.get("MXFP4_BLOBDIR", "/home/tyler/Projects/AI/temp/p0/cutlass_blobs")
ALIGN = 32; T_CUTLASS = 40; RICH = [30,34,37,39,41,42]
BLK = {0:(1,4),1:(1,2),8:(32,34),10:(256,84),12:(256,144),16:(256,66),26:(1,4),30:(1,2),38:(32,33),39:(32,17)}
TGT = {"ffn_gate_exps":"gate","ffn_up_exps":"up","ffn_down_exps":"down"}
os.makedirs(BLOBDIR, exist_ok=True)

def rich_of(name):
    for il in RICH:
        for t in TGT:
            if name == f"blk.{il}.{t}.weight": return il, TGT[t]
    return None

# ---- pass 1: generate CUTLASS blob file per rich tensor (dims come from the GGUF, [K,N,n_exp]) ----
def gen_blob(il, tgt, dims):
    K, N, n_exp = dims[0], dims[1], dims[2]
    e2f, e8f = f"{SRCDIR}/L{il}_{tgt}.e2m1", f"{SRCDIR}/L{il}_{tgt}.e8m0"
    out = f"{BLOBDIR}/L{il}_{tgt}.cutlass"
    subprocess.run([PACK_CLI, "--stacked", e2f, e8f, str(N), str(K), str(n_exp), out],
                   check=True, stdout=subprocess.DEVNULL)   # one call packs all experts; expert-major (data||sf)
    print(f"  blob L{il} {tgt}: {os.path.getsize(out)/1e9:.2f}GB (N={N} K={K} experts={n_exp})", flush=True)
    return out

# ---- read base GGUF header/KV/tensor-infos (verbatim KV) ----
f = open(SRC, "rb"); assert f.read(4) == b"GGUF"
ver, = struct.unpack("<I", f.read(4)); nt, = struct.unpack("<Q", f.read(8)); nkv, = struct.unpack("<Q", f.read(8))
def rstr(): n, = struct.unpack("<Q", f.read(8)); return f.read(n)
def skipval(t):
    sz = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
    if t == 8: n, = struct.unpack("<Q", f.read(8)); f.read(n); return
    if t == 9:
        et, = struct.unpack("<I", f.read(4)); n, = struct.unpack("<Q", f.read(8))
        for _ in range(n): skipval(et)
        return
    f.read(sz[t])
kv_start = f.tell()
for _ in range(nkv): rstr(); t, = struct.unpack("<I", f.read(4)); skipval(t)
kv_end = f.tell(); f.seek(kv_start); kv_bytes = f.read(kv_end - kv_start)
infos = []
for _ in range(nt):
    n, = struct.unpack("<Q", f.read(8)); name = f.read(n).decode()
    nd, = struct.unpack("<I", f.read(4)); dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]
    typ, = struct.unpack("<I", f.read(4)); off, = struct.unpack("<Q", f.read(8))
    infos.append([name, typ, dims, off])
pos = f.tell(); data_base = (pos + ALIGN - 1)//ALIGN*ALIGN
def nbytes(typ, dims):
    be, bb = BLK[typ]; nel = 1
    for d in dims: nel *= d
    return nel//be*bb

# ---- pass 1: blobs ----
print("generating CUTLASS blobs from source...")
blobmap = {}
for name, typ, dims, o in infos:
    r = rich_of(name)
    if r: blobmap[name] = gen_blob(r[0], r[1], dims)
assert len(blobmap) == 18, len(blobmap)

# ---- pass 2: recompute offsets + write ----
new = []; off = 0
for name, typ, dims, o in infos:
    if name in blobmap:
        sz = os.path.getsize(blobmap[name]); ntyp = T_CUTLASS
    else:
        sz = nbytes(typ, dims); ntyp = typ
    new.append([name, ntyp, dims, off, sz, blobmap.get(name)])
    off = (off + sz + ALIGN - 1)//ALIGN*ALIGN
def w_str(s): b = s.encode(); return struct.pack("<Q", len(b)) + b
hdr = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", nt) + struct.pack("<Q", nkv) + kv_bytes
ib = b""
for name, typ, dims, o, sz, bp in new:
    ib += w_str(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
    ib += struct.pack("<I", typ) + struct.pack("<Q", o)
pre = hdr + ib; pad = (-len(pre)) % ALIGN
CH = 64*1024*1024
o = open(OUT, "wb"); o.write(pre); o.write(b"\x00"*pad)
for (name, typ, dims, o_off, sz, bp), (sn, sty, sd, soff) in zip(new, infos):
    if bp:
        with open(bp, "rb") as bf:
            while (c := bf.read(CH)): o.write(c)
    else:
        f.seek(data_base + soff); left = sz
        while left > 0: c = f.read(min(CH, left)); o.write(c); left -= len(c)
    o.write(b"\x00" * ((-sz) % ALIGN))
o.close(); f.close()
print(f"WROTE {OUT} size={os.path.getsize(OUT)/1e9:.2f}GB tensors={nt} rich={len(blobmap)} type={T_CUTLASS}")
