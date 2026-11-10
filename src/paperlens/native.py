"""Optional C++ accelerator with pure-Python twins.

What it does and why (plain language)
-------------------------------------
Profiling (docs/evidence/profile-20261101-*.json) showed most of a search is spent decoding postings in Python.
`paperlens._native` (C++17, pybind11, built by scripts/build_native.py) does that work faster. RULES:
  * every function exists twice with the same arguments and results: `<name>_py` (reference) and the C++ one;
  * tests/test_native.py runs both on the same inputs and demands equal outputs (differential testing);
  * if the extension is not built, or PAPERLENS_NATIVE=0, the Python versions are used and a warning says so once.
`set_enabled(False)` switches to the Python twins at run time (used by tests and benchmarks).
"""
from __future__ import annotations

import os
import warnings

import numpy as np

from .text.varint import VarintError, decode_postings as _decode_postings_py

_native = None
_import_error = None
if os.environ.get("PAPERLENS_NATIVE", "1") != "0":
    try:
        from . import _native            # type: ignore[attr-defined]
    except ImportError as e:             # not built (or built for another Python / platform)
        _import_error = str(e)
        _native = None
        warnings.warn("PaperLens C++ extension not available (" + _import_error + "); using the slower pure-Python "
                      "implementation. Build it with: python scripts/build_native.py", RuntimeWarning, stacklevel=2)

HAVE_NATIVE = _native is not None
_enabled = HAVE_NATIVE


def set_enabled(flag: bool) -> bool:
    """Use the C++ functions (True) or the Python twins (False). Returns the state now in effect."""
    global _enabled
    _enabled = bool(flag) and HAVE_NATIVE
    return _enabled


def enabled() -> bool:
    return _enabled


def status() -> dict:
    return {"have_native": HAVE_NATIVE, "enabled": _enabled, "import_error": _import_error,
            "env_PAPERLENS_NATIVE": os.environ.get("PAPERLENS_NATIVE")}


# ------------------------------------------------------------------ Python twins (the reference)
def decode_postings_py(buf, pos: int, count: int):
    return _decode_postings_py(buf, pos, count)


def bm25_accumulate_py(postings, pos, count, doclens, dead, mask, idf, k1, b, avgdl, out) -> int:
    """out[doc] += BM25 contribution for every non-dead, non-filtered posting. Returns how many were scored."""
    pairs, _ = _decode_postings_py(postings, pos, count)
    n = len(doclens)
    scored = 0
    for local, tf in pairs:
        if local >= n:
            raise IndexError("document id out of range")
        if (dead is not None and dead[local]) or (mask is not None and not mask[local]):
            continue
        dl = doclens[local]
        out[local] += idf * tf * (k1 + 1.0) / (tf + k1 * (1.0 - b + b * dl / avgdl))
        scored += 1
    return scored


def level_dot_py(packed, q, dim: int, bits: int) -> np.ndarray:
    from .vectors.quant import unpack_levels
    return (unpack_levels(np.asarray(packed, dtype=np.uint8), dim, bits) @ np.asarray(q, dtype=np.float32))


# ------------------------------------------------------------------ dispatchers
def decode_postings(buf, pos: int, count: int):
    if _enabled:
        try:
            return _native.decode_postings(buf, pos, count)
        except RuntimeError as e:
            raise VarintError(str(e)) from None
    return decode_postings_py(buf, pos, count)


def bm25_accumulate(postings, pos, count, doclens, dead, mask, idf, k1, b, avgdl, out) -> int:
    if _enabled:
        try:
            return _native.bm25_accumulate(postings, pos, count, doclens, dead, mask, idf, k1, b, avgdl, out)
        except RuntimeError as e:
            raise VarintError(str(e)) from None
    return bm25_accumulate_py(postings, pos, count, doclens, dead, mask, idf, k1, b, avgdl, out)


def level_dot(packed, q, dim: int, bits: int) -> np.ndarray:
    if _enabled:
        return _native.level_dot(np.ascontiguousarray(packed, dtype=np.uint8), np.ascontiguousarray(q, dtype=np.float32), dim, bits)
    return level_dot_py(packed, q, dim, bits)
