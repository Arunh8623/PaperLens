"""MinHash + LSH near-duplicate detection (pure-Python/numpy reference; a C++ twin may come in Phase 1b).

What it does and why (plain language)
-------------------------------------
Preprints often appear twice (a v1 and a lightly edited copy, a conference/journal re-post).
Two papers are "near duplicates" when their word 3-grams ("shingles") overlap a lot:
Jaccard = |A and B| / |A or B|.  Comparing every pair is too slow, so:
  * MinHash squeezes each paper into 128 small numbers; two papers agree on a fraction of
    positions equal to their Jaccard (an estimate, not exact).
  * LSH ("banding") splits those 128 numbers into 16 bands of 8; two papers that match in
    ANY whole band become a candidate pair. Similar pairs almost surely collide; dissimilar
    pairs almost never do.
  * Every candidate is then CHECKED with the exact Jaccard, so a reported duplicate is real
    (no false positives); the only possible error is a missed pair (probability printed by
    `miss_probability`).
Clusters are the connected groups of verified pairs (union-find).

Everything is deterministic: shingle hashes use blake2b (NOT Python's per-process salted
hash()), and the hash functions come from a seeded random generator, so signatures are
identical across processes, machines and Windows/Linux.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np

P = (1 << 31) - 1                      # Mersenne prime: a*x+b stays below 2**63 for a,b,x < P
_WORD = re.compile(r"[a-z0-9]+")


def shingles(text: str, k: int = 3) -> set:
    words = _WORD.findall(text.lower())
    if len(words) < k:
        grams = [" ".join(words)] if words else []
    else:
        grams = [" ".join(words[i:i + k]) for i in range(len(words) - k + 1)]
    return {int.from_bytes(hashlib.blake2b(g.encode("utf-8"), digest_size=8).digest(), "little") % P for g in grams}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


class MinHasher:
    def __init__(self, num_perm: int = 128, seed: int = 1):
        rng = np.random.default_rng(seed)
        self.num_perm = num_perm
        self.a = rng.integers(1, P, size=num_perm, dtype=np.int64)
        self.b = rng.integers(0, P, size=num_perm, dtype=np.int64)

    def signature(self, sh: set) -> np.ndarray:
        if not sh:
            return np.full(self.num_perm, P, dtype=np.int64)       # empty text: never equals a real signature
        x = np.fromiter(sh, dtype=np.int64, count=len(sh))
        return ((self.a[:, None] * x[None, :] + self.b[:, None]) % P).min(axis=1)

    @staticmethod
    def estimate(s1: np.ndarray, s2: np.ndarray) -> float:
        return float(np.mean(s1 == s2))


def miss_probability(jaccard_value: float, bands: int = 16, rows: int = 8) -> float:
    """Chance that LSH never proposes a pair of this true similarity."""
    return (1.0 - jaccard_value ** rows) ** bands


class _UnionFind:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)      # smallest key is the representative: deterministic


def near_duplicate_pairs(texts: dict, threshold: float = 0.8, num_perm: int = 128, bands: int = 16,
                         seed: int = 1, k: int = 3) -> list:
    """texts: {key: text}. Returns sorted [(key_a, key_b, exact_jaccard)] with exact_jaccard >= threshold."""
    if num_perm % bands:
        raise ValueError("bands must divide num_perm")
    rows = num_perm // bands
    hasher = MinHasher(num_perm, seed)
    shs = {key: shingles(t, k) for key, t in texts.items()}
    buckets: dict = {}
    for key in sorted(texts):
        sig = hasher.signature(shs[key])
        for bi in range(bands):
            buckets.setdefault((bi, sig[bi * rows:(bi + 1) * rows].tobytes()), []).append(key)
    seen, out = set(), []
    for members in buckets.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pair = (members[i], members[j])
                if pair in seen:
                    continue
                seen.add(pair)
                if not shs[pair[0]] and not shs[pair[1]]:
                    continue                                        # two empty texts are not "duplicates"
                jv = jaccard(shs[pair[0]], shs[pair[1]])
                if jv >= threshold:
                    out.append((pair[0], pair[1], jv))
    return sorted(out)


def near_duplicate_clusters(texts: dict, **kw) -> list:
    """Groups (size >= 2) of keys connected by verified near-duplicate pairs; each sorted, list sorted."""
    uf = _UnionFind()
    for a, b, _ in near_duplicate_pairs(texts, **kw):
        uf.union(a, b)
    groups: dict = {}
    for key in list(uf.p):
        groups.setdefault(uf.find(key), []).append(key)
    return sorted(sorted(g) for g in groups.values() if len(g) > 1)


def dedupe_ranked(ranked_keys: list, texts: dict, **kw) -> tuple:
    """Keep the best-ranked member of every near-duplicate group.
    Returns (kept_keys_in_order, {dropped_key: kept_key}) so the UI can say why a result vanished."""
    present = {k: texts[k] for k in ranked_keys}
    rep = {}
    for g in near_duplicate_clusters(present, **kw):
        best = min(g, key=ranked_keys.index)
        for k in g:
            if k != best:
                rep[k] = best
    return [k for k in ranked_keys if k not in rep], rep
