"""Citation graph in CSR (compressed sparse row) form.

What it does and why (plain language)
-------------------------------------
Papers are nodes; an edge a -> b means "paper a cites paper b". A graph with
millions of edges as Python dicts of lists is slow and big. CSR stores it as
two flat arrays:

    offsets[i] .. offsets[i+1]   the slice of `targets` that holds node i's citations

so "who does node i cite?" is one slice, and a whole PageRank step is a few
vectorised numpy operations. Node ids are dense integers 0..n-1; `ids` maps
them back to arxiv ids.

File format (little endian, one file `graph.bin`, checksummed in the MANIFEST
snapshot record, see store/graphfile.py):
    magic 'PLG1' | u32 n | u32 m | offsets u32[n+1] | targets u32[m] | ids (JSON, length-prefixed u32)
Citations to papers we do not have are dropped (we only rank papers in the index).
Self-citations and duplicate edges are removed.
"""
from __future__ import annotations

import json
import struct

import numpy as np

MAGIC = b"PLG1"


class Graph:
    def __init__(self, ids: list, offsets: np.ndarray, targets: np.ndarray):
        self.ids = list(ids)
        self.offsets = np.asarray(offsets, dtype=np.uint32)
        self.targets = np.asarray(targets, dtype=np.uint32)
        self.index = {a: i for i, a in enumerate(self.ids)}
        if len(self.offsets) != len(self.ids) + 1 or self.offsets[-1] != len(self.targets):
            raise ValueError("inconsistent CSR arrays")
        if len(self.targets) and int(self.targets.max()) >= len(self.ids):
            raise ValueError("edge points outside the graph")

    @property
    def n(self) -> int:
        return len(self.ids)

    @property
    def m(self) -> int:
        return len(self.targets)

    def out(self, i: int) -> np.ndarray:
        return self.targets[self.offsets[i]:self.offsets[i + 1]]

    def out_degree(self) -> np.ndarray:
        return np.diff(self.offsets.astype(np.int64))

    def in_degree(self) -> np.ndarray:
        return np.bincount(self.targets, minlength=self.n)

    def sources(self) -> np.ndarray:
        """Source node of every edge (parallel to `targets`)."""
        return np.repeat(np.arange(self.n), self.out_degree())

    def reverse(self) -> "Graph":
        src, dst = self.sources(), self.targets.astype(np.int64)
        order = np.argsort(dst, kind="stable")
        counts = np.bincount(dst, minlength=self.n)
        offs = np.concatenate([[0], np.cumsum(counts)])
        return Graph(self.ids, offs, src[order])

    # ---------------------------------------------------------------- build / serialize
    @classmethod
    def from_edges(cls, ids: list, edges) -> "Graph":
        """ids: node names; edges: iterable of (citing_id, cited_id). Unknown ids, self-loops and
        repeated edges are dropped."""
        idx = {a: i for i, a in enumerate(ids)}
        if len(idx) != len(ids):
            raise ValueError("duplicate node ids")
        pairs = set()
        for a, b in edges:
            if a in idx and b in idx and a != b:
                pairs.add((idx[a], idx[b]))
        arr = np.array(sorted(pairs), dtype=np.int64).reshape(-1, 2)
        counts = np.bincount(arr[:, 0], minlength=len(ids)) if len(arr) else np.zeros(len(ids), dtype=np.int64)
        offs = np.concatenate([[0], np.cumsum(counts)])
        return cls(ids, offs, arr[:, 1] if len(arr) else np.zeros(0))

    def to_bytes(self) -> bytes:
        ids = json.dumps(self.ids, separators=(",", ":")).encode("utf-8")
        return (MAGIC + struct.pack("<II", self.n, self.m) + self.offsets.astype("<u4").tobytes()
                + self.targets.astype("<u4").tobytes() + struct.pack("<I", len(ids)) + ids)

    @classmethod
    def from_bytes(cls, data: bytes) -> "Graph":
        if data[:4] != MAGIC:
            raise ValueError("not a PaperLens graph file")
        n, m = struct.unpack_from("<II", data, 4)
        o = 12
        offsets = np.frombuffer(data, dtype="<u4", count=n + 1, offset=o)
        o += 4 * (n + 1)
        targets = np.frombuffer(data, dtype="<u4", count=m, offset=o)
        o += 4 * m
        (ln,) = struct.unpack_from("<I", data, o)
        ids = json.loads(data[o + 4:o + 4 + ln].decode("utf-8"))
        if o + 4 + ln != len(data) or len(ids) != n:
            raise ValueError("graph file has the wrong size")
        return cls(ids, offsets.copy(), targets.copy())
