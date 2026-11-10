"""Variable-byte integers and the postings list codec (Python reference).

What it does and why (plain language)
-------------------------------------
A postings list says "term X appears in documents 3, 7, 8, 20 ...". Storing
each number in 4 or 8 bytes wastes space, because most numbers are small.

1. Gaps: store the DIFFERENCE from the previous doc id (3, 4, 1, 12) - small.
2. Varint (LEB128): a number is split into 7-bit groups, lowest first; the top
   bit of each byte means "more bytes follow". Values < 128 take 1 byte.

A postings list is a sequence of (gap, tf) pairs, both varints. The first gap
is measured from -1, so every gap is >= 1 (doc 0 -> gap 1).

This is the pure-Python reference. A C++ twin with the same API arrives in
Phase 1b only if profiling shows it is worth it (differential tests then).
"""
from __future__ import annotations

MAX_VARINT_BYTES = 10          # a u64 needs at most 10 groups of 7 bits


class VarintError(ValueError):
    pass


def encode_uvarint(n: int) -> bytes:
    if n < 0:
        raise VarintError("negative number")
    if n >> 64:
        raise VarintError("number does not fit in 64 bits")
    out = bytearray()
    while n >= 0x80:
        out.append((n & 0x7F) | 0x80)
        n >>= 7
    out.append(n)
    return bytes(out)


def decode_uvarint(buf, pos: int = 0):
    """Return (value, new_pos). Raises VarintError on truncated/overlong input."""
    result = shift = 0
    n = len(buf)
    for i in range(MAX_VARINT_BYTES):
        if pos + i >= n:
            raise VarintError("truncated varint")
        b = buf[pos + i]
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            if result >> 64:
                raise VarintError("varint exceeds 64 bits")
            return result, pos + i + 1
        shift += 7
    raise VarintError("varint too long")


def encode_postings(pairs) -> bytes:
    """pairs: iterable of (doc_id, tf) with strictly increasing doc_id >= 0, tf >= 1."""
    out = bytearray()
    prev = -1
    for doc, tf in pairs:
        if doc <= prev:
            raise VarintError("doc ids must be strictly increasing")
        if tf < 1:
            raise VarintError("tf must be >= 1")
        out += encode_uvarint(doc - prev)
        out += encode_uvarint(tf)
        prev = doc
    return bytes(out)


def decode_postings(buf, pos: int, count: int):
    """Decode exactly `count` pairs. Returns (list_of_(doc, tf), new_pos)."""
    out, prev = [], -1
    for _ in range(count):
        gap, pos = decode_uvarint(buf, pos)
        tf, pos = decode_uvarint(buf, pos)
        if gap < 1 or tf < 1:
            raise VarintError("corrupt postings (gap or tf is zero)")
        prev += gap
        out.append((prev, tf))
    return out, pos
