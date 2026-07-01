"""P0: HF DeepSeek-V4-Flash -> ds4 GGUF converter (library).
Minimal pure-numpy safetensors + GGUF readers and dequant codecs.
Stage 1: the mtp.0.* draft layer."""
import json, struct, numpy as np

# ---------- safetensors reader ----------
class SafeTensors:
    DT = {"F32":np.float32,"F16":np.float16,"BF16":None,"F8_E4M3":np.uint8,
          "F8_E8M0":np.uint8,"I8":np.int8,"U8":np.uint8}
    def __init__(self, path):
        self.path=path
        with open(path,"rb") as f:
            n,=struct.unpack("<Q",f.read(8)); self.hdr=json.loads(f.read(n)); self.data0=8+n
    def names(self): return [k for k in self.hdr if k!="__metadata__"]
    def meta(self,name): return self.hdr[name]
    def raw(self,name):
        m=self.hdr[name]; a,b=m["data_offsets"]
        with open(self.path,"rb") as f:
            f.seek(self.data0+a); buf=f.read(b-a)
        return m["dtype"], m["shape"], buf
    def get(self,name):
        """Return a numpy array in the tensor's NATIVE numeric interpretation.
        BF16->f32; F8/I8 returned as raw uint8/int8 (caller dequants with scale)."""
        dt,shp,buf=self.raw(name)
        if dt=="BF16":
            u=np.frombuffer(buf,dtype=np.uint16).astype(np.uint32)
            return (u<<16).view(np.float32).reshape(shp)
        if dt=="F32": return np.frombuffer(buf,dtype=np.float32).reshape(shp)
        if dt=="F16": return np.frombuffer(buf,dtype=np.float16).astype(np.float32).reshape(shp)
        if dt in ("F8_E4M3","F8_E8M0","U8"): return np.frombuffer(buf,dtype=np.uint8).reshape(shp)
        if dt=="I8": return np.frombuffer(buf,dtype=np.int8).reshape(shp)
        raise ValueError(dt)

# ---------- fp8 / mx codecs ----------
def e4m3_to_f32(b):  # b: uint8 array
    b=b.astype(np.uint32)
    s=np.where(b&0x80,-1.0,1.0).astype(np.float32)
    e=((b>>3)&0xF).astype(np.int32); m=(b&0x7).astype(np.float32)
    sub = s*(m/8.0)*(2.0**-6)
    nrm = s*(1.0+m/8.0)*(2.0**(e-7))
    out=np.where(e==0, sub, nrm).astype(np.float32)
    out=np.where((e==15)&((b&0x7)==7), np.nan, out)  # NaN
    return out
def e8m0_to_scale(b):  # uint8 -> 2^(b-127); 255 is NaN in E8M0
    b=b.astype(np.int32); return np.where(b==255, np.nan, 2.0**(b-127)).astype(np.float32)

def q8_0_encode(w_f32):
    """f32 [out,in] -> GGUF Q8_0 bytes, row-major, in/32 blocks/row of 34B = [f16 d][32 int8].
    The fused attn-output kernels are NOT FP8-aware, so attn_output must be Q8_0 (lossy)."""
    out,IN=w_f32.shape; assert IN%32==0, IN
    x=w_f32.reshape(out, IN//32, 32).astype(np.float32)
    amax=np.abs(x).max(axis=2)                          # [out, in/32]
    d=(amax/127.0).astype(np.float32)
    idd=np.where(d>0, 1.0/d, 0.0).astype(np.float32)
    q=np.rint(x*idd[:,:,None]).astype(np.int32)
    q=np.clip(q,-127,127).astype(np.int8)               # [out, in/32, 32]
    dh=d.astype(np.float16).view(np.uint8).reshape(out, IN//32, 2)
    blocks=np.concatenate([dh, q.view(np.uint8)], axis=2)  # [out, in/32, 34]
    return blocks.tobytes()

# ---------- GGUF reader (for validation vs off-the-shelf) ----------
GGUF_T={0:"F32",1:"F16",8:"Q8_0",10:"Q2_K",12:"Q4_K",14:"Q6_K",16:"IQ2_XXS",38:"FP8_E4M3"}
def gguf_read(path):
    f=open(path,"rb"); f.read(4); ver,=struct.unpack("<I",f.read(4))
    nt,=struct.unpack("<Q",f.read(8)); nkv,=struct.unpack("<Q",f.read(8))
    def rs():
        n,=struct.unpack("<Q",f.read(8)); return f.read(n).decode("utf-8","replace")
    def rv(t):
        sz={0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
        if t==8: return rs()
        if t==9:
            et,=struct.unpack("<I",f.read(4)); n,=struct.unpack("<Q",f.read(8)); return [rv(et) for _ in range(n)]
        fmt={0:"<b",1:"<B",2:"<h",3:"<H",4:"<i",5:"<I",6:"<f",7:"<?",10:"<q",11:"<Q",12:"<d"}[t]
        return struct.unpack(fmt,f.read(sz[t]))[0]
    md={}
    for _ in range(nkv):
        k=rs(); t,=struct.unpack("<I",f.read(4)); md[k]=rv(t)
    infos=[]
    for _ in range(nt):
        name=rs(); nd,=struct.unpack("<I",f.read(4))
        dims=[struct.unpack("<Q",f.read(8))[0] for _ in range(nd)]
        typ,=struct.unpack("<I",f.read(4)); off,=struct.unpack("<Q",f.read(8))
        infos.append((name,GGUF_T.get(typ,typ),dims,off))
    align=md.get("general.alignment",32)
    pos=f.tell(); base=(pos+align-1)//align*align
    return f,md,infos,base,align
def gguf_get_f32(path,want):
    f,md,infos,base,align=gguf_read(path)
    out={}
    for name,t,dims,off in infos:
        if name not in want or t!="F32": continue
        n=1
        for d in dims: n*=d
        f.seek(base+off); out[name]=np.frombuffer(f.read(n*4),dtype=np.float32).reshape(list(reversed(dims)))
    f.close(); return out

# ---------- MXFP4 (E2M1) codec + ds4 block repack ----------
# E2M1 nibble -> value: sign(1) exp(2) mantissa(1); levels {0,.5,1,1.5,2,3,4,6}
_E2M1=np.array([0,0.5,1,1.5,2,3,4,6, 0,-0.5,-1,-1.5,-2,-3,-4,-6],dtype=np.float32)
def e2m1_unpack(b_i8):           # int8/uint8 [.., in/2] -> float nibbles [.., in]
    u=b_i8.astype(np.uint8)
    lo=u&0xF; hi=(u>>4)&0xF
    out=np.empty(u.shape[:-1]+(u.shape[-1]*2,),dtype=np.float32)
    out[...,0::2]=_E2M1[lo]; out[...,1::2]=_E2M1[hi]
    return out
def dequant_mxfp4(w_i8, scale_e8m0):
    """w_i8 [out, in/2] E2M1-packed; scale_e8m0 [out, in/32] E8M0. -> f32 [out,in]"""
    vals=e2m1_unpack(w_i8)                       # [out, in]
    out,IN=vals.shape
    sc=e8m0_to_scale(scale_e8m0)                 # [out, in/32]
    sc=np.repeat(sc, 32, axis=1)                 # [out, in]
    return vals*sc

# ds4 MXFP4 block: per 32 input elems, 17 bytes = [1 E8M0 scale][16 E2M1 bytes]
def repack_mxfp4_row(w_i8_row, scale_row):       # w_i8_row [in/2], scale_row [in/32] -> bytes
    IN2=w_i8_row.shape[0]; nblk=scale_row.shape[0]
    assert IN2==nblk*16, (IN2,nblk)
    out=np.empty((nblk,17),dtype=np.uint8)
    out[:,0]=scale_row.astype(np.uint8)
    out[:,1:]=w_i8_row.reshape(nblk,16).astype(np.uint8)  # 16 bytes = 32 E2M1 vals
    return out.tobytes()

# ---------- CUTLASS MXFP4 codec (source E2M1+E8M0 -> ColumnMajor data + swizzled SF) ----------
# The GB10 tensor-core path stores rich experts in CUTLASS B layout. The SF swizzle is CUTLASS's
# tile-atom layout, so we shell out to the validated sm_120f packer (needs a GB10 box w/ CUTLASS —
# run this on Sparky). NOTE(cleanup): once the fork is GB10-only, THIS + repack_mxfp8 + BF16 are the
# only codecs we keep; q8_0_encode / Q4_K / Q2_K / IQ2 readers+codecs are dead code to strip.
import subprocess as _sp, tempfile as _tf, os as _os
_PACK_CLI = _os.environ.get("DS4_MXFP4_PACK_CLI", _os.path.expanduser("~/Projects/AI/temp/p0/mxfp4_pack_source_cli"))
def pack_mxfp4_cutlass(e2m1_i8, scale_e8m0, N, K):
    """One expert: E2M1 [N,K/2] + E8M0 [N,K/32] -> (data_bytes, sf_bytes) in CUTLASS B layout (ColumnMajor data + swizzled SF)."""
    assert e2m1_i8.shape == (N, K//2) and scale_e8m0.shape == (N, K//32), (e2m1_i8.shape, scale_e8m0.shape)
    with _tf.TemporaryDirectory() as d:
        pe, ps = f"{d}/e2.bin", f"{d}/e8.bin"; pd, pf = f"{d}/data.bin", f"{d}/sf.bin"
        e2m1_i8.astype(np.uint8).tofile(pe); scale_e8m0.astype(np.uint8).tofile(ps)
        _sp.run([_PACK_CLI, pe, ps, str(N), str(K), pd, pf], check=True, stdout=_sp.DEVNULL)
        return open(pd, "rb").read(), open(pf, "rb").read()
def stack_experts_cutlass(get_expert, n_expert, N, K):
    """get_expert(i) -> (e2m1[N,K/2], e8m0[N,K/32]). Returns stacked tensor: per expert (data||sf), expert-major."""
    parts = []
    for i in range(n_expert):
        e2, e8 = get_expert(i)
        data, sf = pack_mxfp4_cutlass(e2, e8, N, K)
        parts.append(data); parts.append(sf)   # per-expert stride = len(data)+len(sf); engine indexes expert i at i*stride
    return b"".join(parts)

# ---------- MXFP8 repack (source 128x128-block FP8 -> ds4 per-32 block) ----------
# ds4 MXFP8 block: 33 bytes/32 = [1 E8M0 scale][32 E4M3]. Reuse the parent 128x128
# scale for each per-32 sub-block + copy E4M3 bytes verbatim => byte-lossless.
def repack_mxfp8(w_e4m3, scale_e8m0):
    """w_e4m3 uint8 [out,in]; scale_e8m0 uint8 [out/128, in/128] -> bytes, ds4 33B/32 blocks (row-major)."""
    out,IN=w_e4m3.shape; nb=IN//32
    sc=np.repeat(scale_e8m0, 4, axis=1)[:, :nb]          # in/128 -> in/32 (each 128-blk = 4 of 32)
    sc=np.repeat(sc, 128, axis=0)[:out, :]               # out/128 -> out (each 128 rows share)
    blocks=np.empty((out, nb, 33), dtype=np.uint8)
    blocks[:,:,0]=sc
    blocks[:,:,1:]=w_e4m3.reshape(out, nb, 32)
    return blocks.tobytes()
def dequant_block128_fp8(w_e4m3, scale_e8m0):
    """reference source dequant (128x128-block FP8) -> f32"""
    v=e4m3_to_f32(w_e4m3)
    sc=e8m0_to_scale(np.repeat(np.repeat(scale_e8m0,128,axis=0),128,axis=1)[:w_e4m3.shape[0],:w_e4m3.shape[1]])
    return v*sc
