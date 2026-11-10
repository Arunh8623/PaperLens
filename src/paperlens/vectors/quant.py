"""RaBitQ-style 1-4 bit vector quantization (numpy reference implementation).

What it does and why (plain language)
-------------------------------------
A paper's meaning is a vector of, say, 384 numbers (4 bytes each = 1536 bytes).
Comparing a query with 50,000 of them takes a while and the vectors take memory.
Quantization keeps only a few BITS per number (1 bit = "is this number above or
below zero?"), which is 32x smaller (1-bit) and much faster to scan. The price is
a small error, so we scan the cheap codes first to find ~100 candidates, then
RE-SCORE only those with the exact half-precision copy (see store/segment.py).

Steps (this is the idea of RaBitQ, SIGMOD 2024, written from memory of the
paper, NOT a line-by-line port; the error bound of the paper is not implemented):
  1. centre: subtract the segment's mean vector c; keep the length d_o = |o - c|
  2. rotate the direction with a fixed random orthogonal matrix (spreads the
     information evenly over all coordinates)
  3. code: per coordinate keep b bits (b = 1 is just the sign)
  4. store per vector two numbers that fix the estimator's bias:
        inv  = 1 / (|u| * <obar, o'>)      (obar = the vector the code stands for)
        oc   = <o, c>                      (to undo the centring)
     Estimated cosine of the centred directions = inv * <u, q'>, where q' is the
     query centred and rotated the same way and u are the integer code levels.
  5. inner product <o, q> = d_o * d_q * est + <o, c> + <q, c> - |c|^2

Honest limits: the codes make scans cheaper in MEMORY and TIME; on DISK we also
keep the half-precision copy, so disk use is not reduced. 2-4 bit codes use a
simple uniform scalar quantizer (not the optimised "Extended RaBitQ" search).
The rotation is generated with Python's own random module (a documented-stable
stream) plus numpy QR, so a numpy upgrade does not change old segments' codes.
"""
from __future__ import annotations

import math
import random
from functools import lru_cache

import numpy as np

from .. import native

QVERSION = 1
ROT_SEED = 20240601
CHUNK = 16384


@lru_cache(maxsize=8)
def rotation_matrix(dim: int, seed: int = ROT_SEED) -> np.ndarray:
    """Deterministic random orthogonal (dim x dim) matrix, float32."""
    rng = random.Random(f"paperlens-rotation-{seed}-{dim}")
    a = np.empty((dim, dim), dtype=np.float64)
    for i in range(dim):
        for j in range(0, dim, 2):                       # Box-Muller on the stable random() stream
            u1, u2 = rng.random() or 1e-12, rng.random()
            r = math.sqrt(-2.0 * math.log(u1))
            a[i, j] = r * math.cos(2 * math.pi * u2)
            if j + 1 < dim:
                a[i, j + 1] = r * math.sin(2 * math.pi * u2)
    q, r = np.linalg.qr(a)
    q = q * np.sign(np.diag(r))                           # make the distribution uniform (Haar)
    return q.astype(np.float32)


def code_bytes(dim: int, bits: int) -> int:
    return (dim * bits + 7) // 8


def _check_bits(bits: int) -> None:
    if bits not in (1, 2, 3, 4):
        raise ValueError("bits must be 1, 2, 3 or 4")


def pack_codes(codes: np.ndarray, bits: int) -> np.ndarray:
    """codes: (n, dim) ints in [0, 2^bits) -> (n, code_bytes) uint8."""
    n, dim = codes.shape
    planes = ((codes[:, :, None].astype(np.uint8) >> np.arange(bits, dtype=np.uint8)) & 1)
    return np.packbits(planes.reshape(n, dim * bits), axis=1)


def unpack_levels(packed: np.ndarray, dim: int, bits: int) -> np.ndarray:
    """(n, code_bytes) -> (n, dim) float32 odd integer levels u = 2c + 1 - 2^bits."""
    n = packed.shape[0]
    flat = np.unpackbits(packed, axis=1)[:, :dim * bits].reshape(n, dim, bits)
    codes = (flat.astype(np.int16) << np.arange(bits, dtype=np.int16)).sum(axis=2)
    return (2 * codes + 1 - (1 << bits)).astype(np.float32)


def encode(vecs: np.ndarray, centroid: np.ndarray, bits: int, rot: np.ndarray) -> dict:
    """vecs: (n, dim) float32 UNIT vectors. Returns arrays: codes, d_o, inv, oc."""
    _check_bits(bits)
    vecs = np.asarray(vecs, dtype=np.float32)
    n, dim = vecs.shape
    r = vecs - centroid
    d_o = np.linalg.norm(r, axis=1)
    safe = np.where(d_o > 1e-12, d_o, 1.0)
    o_rot = (r / safe[:, None]) @ rot                      # rotated unit direction o'
    levels = 1 << bits
    if bits == 1:
        codes = (o_rot > 0).astype(np.uint8)
    else:
        s = np.maximum(np.abs(o_rot).max(axis=1), 1e-12)   # per-vector range; cancels in the estimator
        codes = np.clip(np.floor((o_rot / s[:, None] + 1.0) * 0.5 * levels), 0, levels - 1).astype(np.uint8)
    u = (2 * codes.astype(np.int16) + 1 - levels).astype(np.float32)
    u_norm = np.linalg.norm(u, axis=1)
    x0 = (u * o_rot).sum(axis=1) / u_norm                  # <obar, o'>, obar = u / |u|
    x0 = np.where(np.abs(x0) > 1e-6, x0, 1e-6)
    inv = (1.0 / (u_norm * x0)).astype(np.float32)
    inv = np.where(d_o > 1e-12, inv, 0.0).astype(np.float32)   # vector == centroid: direction unknown
    oc = (vecs @ centroid).astype(np.float32)
    return {"codes": pack_codes(codes, bits), "d_o": d_o.astype(np.float32), "inv": inv, "oc": oc}


class QueryPrep:
    """Everything about one query that does not depend on the document."""

    def __init__(self, q: np.ndarray, centroid: np.ndarray, rot: np.ndarray):
        q = np.asarray(q, dtype=np.float32)
        qc = q - centroid
        self.d_q = float(np.linalg.norm(qc))
        self.cc = float(centroid @ centroid)
        self.qcen = float(q @ centroid)
        self.q_rot = ((qc / self.d_q) @ rot).astype(np.float32) if self.d_q > 1e-12 else np.zeros_like(q)


def estimate(packed: np.ndarray, d_o, inv, oc, prep: QueryPrep, dim: int, bits: int) -> np.ndarray:
    """Estimated inner products <o, q> for the given rows (chunked to bound memory)."""
    n = packed.shape[0]
    out = np.empty(n, dtype=np.float32)
    for a in range(0, n, CHUNK):
        b = min(a + CHUNK, n)
        est = inv[a:b] * native.level_dot(packed[a:b], prep.q_rot, dim, bits)
        out[a:b] = d_o[a:b] * prep.d_q * est + oc[a:b] + prep.qcen - prep.cc
    return out


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n == 0.0 or not np.all(np.isfinite(v)):
        raise ValueError("vector must be finite and non-zero")
    return v / n
