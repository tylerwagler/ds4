#!/usr/bin/env python3
"""Reference driver for the native MXFP8_LT (type 41) byte-equality proof.

Runs the numpy reference transform repack_tensor() from the (now retirable)
tools/mxfp8_prestore/repack_mxfp8_lt.py on a raw type-38 payload and writes the
resulting MXFP8_LT bytes, so the native C packer's output can be byte-compared
against it (make test-mxfp8-lt). The numpy repack is the single reference for
the swizzle; if the native packer matches it, the post-hoc bandaid is redundant.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# gguf-tools/quantize/tests -> repo root -> tools/mxfp8_prestore
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "tools", "mxfp8_prestore"))
from repack_mxfp8_lt import repack_tensor  # noqa: E402


def main():
    if len(sys.argv) != 5:
        sys.exit("usage: mxfp8_lt_ref.py OUT IN RAW38_FILE LT_FILE")
    out_dim = int(sys.argv[1])
    in_dim = int(sys.argv[2])
    with open(sys.argv[3], "rb") as f:
        raw = f.read()
    payload = repack_tensor(raw, in_dim, out_dim)
    with open(sys.argv[4], "wb") as f:
        f.write(payload)
    print(f"ref: out={out_dim} in={in_dim} lt={len(payload)}")


if __name__ == "__main__":
    main()
