"""STAND-IN embedder (Phase 2 only; the real model arrives in Phase 3).

What it does and why: the pipeline must carry a vector per paper through shared
memory with a checksum, to prove that protocol works under crashes. The real
semantic model (SPECTER2 or similar) is Phase 3, so here a vector is made by
"feature hashing": every term picks a position (crc32 of the term, modulo dim)
and a sign, and adds a weight. The result is L2-normalised. It is
deterministic on every OS and needs no numpy. It carries NO real meaning: do
not evaluate search quality with it.
"""
from __future__ import annotations

import array
import math
import zlib


def embed_terms(term_freqs: dict, dim: int) -> array.array:
    vec = array.array("f", [0.0]) * dim
    for term, tf in term_freqs.items():
        h = zlib.crc32(term.encode("utf-8"))
        vec[h % dim] += (1.0 if (h >> 16) & 1 else -1.0) * (1.0 + math.log(tf))
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        for i in range(dim):
            vec[i] /= norm
    return vec


def vector_bytes(vec: array.array) -> bytes:
    return vec.tobytes()


def vector_crc(raw: bytes) -> int:
    return zlib.crc32(raw) & 0xFFFFFFFF
