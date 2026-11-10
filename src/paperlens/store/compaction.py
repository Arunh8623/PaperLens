"""Compaction planning and merging (pure functions; IndexStore.compact does the publishing).

What it does and why (plain language)
-------------------------------------
Every flush adds a segment, and queries must look in all of them, so segments pile up. Compaction
(like an LSM tree's merge) rewrites several small segments as ONE bigger segment:
  * documents that were deleted or replaced are dropped for good (the space and the BM25 statistics
    N / df / avgdl stop counting them: Lucene does the same on merge),
  * tombstones that only pointed at documents inside the merged group are dropped; tombstones that
    point at OLDER segments outside the group are kept,
  * vectors are carried over from their half-precision copies and re-quantized against the merged
    segment's own centroid (a merged segment gets a fresh IVF index when it is big enough).
Policy (size-tiered): merge the first `fan_in` neighbouring segments whose sizes are within `ratio`
of each other; otherwise rewrite one segment that is mostly dead (>= `dead_ratio`).
Only NEIGHBOURS are merged, because a segment's tombstones apply to segments before it.
"""
from __future__ import annotations

import numpy as np


def plan_run(sizes: list, dead_counts: list, fan_in: int = 4, ratio: float = 4.0,
             dead_ratio: float = 0.5, force: bool = False):
    """Returns (first, last) inclusive segment indexes to merge, or None."""
    n = len(sizes)
    if n == 0:
        return None
    if force:
        return (0, n - 1) if (n > 1 or any(dead_counts)) else None
    for i in range(0, n - fan_in + 1):
        w = sizes[i:i + fan_in]
        if max(w) <= ratio * max(1, min(w)):
            return (i, i + fan_in - 1)
    for i in range(n):
        if sizes[i] and dead_counts[i] / sizes[i] >= dead_ratio:
            return (i, i)
    return None


def merge_run(segs: list, dead_sets: list, older_keys: set, ivf_min_docs: int, default_bits: int):
    """segs: Segment objects (oldest first); dead_sets: per segment, local ids to drop.
    older_keys: keys present in segments BEFORE the run (tombstones for them must survive).
    Returns (docs, tombstones, vec_cfg) ready for write_segment_tmp."""
    docs = []
    vec_dim = bits = 0
    embedder = ""
    for seg in segs:
        if seg.has_vectors:
            vec_dim = vec_dim or seg.vec_dim
            bits = bits or seg.meta["vectors"]["bits"]
            embedder = embedder or seg.meta["vectors"].get("embedder", "")
    for seg, dead in zip(segs, dead_sets):
        tfs = [dict() for _ in range(seg.num_docs)]
        for term in seg.terms:
            for local, tf in seg.postings(term):
                tfs[local][term] = tf
        valid = vecs = None
        if seg.has_vectors:
            valid = seg.valid_vectors()
            vecs = seg.all_vectors().astype("<f2")
        for local in range(seg.num_docs):
            if local in dead:
                continue
            v = vecs[local].tobytes() if valid is not None and valid[local] else None
            docs.append((seg.keys[local], seg.paper(local), tfs[local], int(seg.doclens[local]), v))
    tomb = set()
    for seg in segs:
        tomb.update(seg.tombstones)
    tomb &= older_keys
    vec_cfg = None
    if vec_dim and any(d[4] is not None for d in docs):
        vec_cfg = {"dim": vec_dim, "bits": bits or default_bits, "ivf_min_docs": ivf_min_docs, "embedder": embedder}
    return docs, sorted(tomb), vec_cfg
