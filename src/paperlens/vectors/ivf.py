"""IVF (inverted file) over one segment's vectors: cluster, then search only the nearest clusters.

What it does and why (plain language)
-------------------------------------
Instead of comparing the query with EVERY vector, group similar vectors into
`nlist` clusters (k-means). At query time compare the query with the cluster
centres, pick the `nprobe` closest clusters and scan only the vectors inside.
Fewer vectors scanned = faster, at the risk of missing a neighbour that sits in
a cluster we did not visit (that is why recall is measured against nprobe).

File ivf.bin (little-endian): centroids f32 [nlist x dim] | offsets u32 [nlist+1] | order u32 [n_valid]
`order` lists the segment-local doc ids grouped by cluster; offsets[i]:offsets[i+1] is cluster i.
Only vectors that exist (non-zero) are clustered.
"""
from __future__ import annotations

import numpy as np


def kmeans(x: np.ndarray, k: int, iters: int = 10, seed: int = 0) -> np.ndarray:
    """Spherical k-means on unit vectors; returns unit centroids (k, dim) float32."""
    n = x.shape[0]
    rng = np.random.default_rng(seed)
    cent = x[rng.choice(n, size=k, replace=False)].copy()
    for _ in range(iters):
        assign = np.concatenate([np.argmax(x[a:a + 8192] @ cent.T, axis=1) for a in range(0, n, 8192)])
        for j in range(k):
            members = x[assign == j]
            if len(members):
                m = members.sum(axis=0)
                nm = np.linalg.norm(m)
                cent[j] = m / nm if nm > 0 else cent[j]
            else:
                cent[j] = x[rng.integers(0, n)]                  # re-seed an empty cluster
    return cent.astype(np.float32)


def build(x: np.ndarray, local_ids: np.ndarray, nlist: int, seed: int = 0) -> bytes:
    cent = kmeans(x, nlist, seed=seed)
    assign = np.concatenate([np.argmax(x[a:a + 8192] @ cent.T, axis=1) for a in range(0, len(x), 8192)])
    order_idx = np.argsort(assign, kind="stable")
    counts = np.bincount(assign, minlength=nlist)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype("<u4")
    order = local_ids[order_idx].astype("<u4")
    return cent.astype("<f4").tobytes() + offsets.tobytes() + order.tobytes()


class Ivf:
    def __init__(self, data: bytes, nlist: int, dim: int, n_valid: int):
        a = nlist * dim * 4
        b = a + (nlist + 1) * 4
        if len(data) != b + n_valid * 4:
            raise ValueError("ivf.bin has the wrong size")
        self.centroids = np.frombuffer(data, dtype="<f4", count=nlist * dim).reshape(nlist, dim)
        self.offsets = np.frombuffer(data, dtype="<u4", count=nlist + 1, offset=a)
        self.order = np.frombuffer(data, dtype="<u4", count=n_valid, offset=b)
        self.nlist = nlist

    def candidates(self, q: np.ndarray, nprobe: int) -> np.ndarray:
        nprobe = max(1, min(nprobe, self.nlist))
        lists = np.argpartition(-(self.centroids @ q), nprobe - 1)[:nprobe]
        return np.concatenate([self.order[self.offsets[i]:self.offsets[i + 1]] for i in lists]).astype(np.int64)
