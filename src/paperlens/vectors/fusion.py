"""Reciprocal Rank Fusion (Cormack, Clarke, Buettcher, 2009).

What it does and why (plain language)
-------------------------------------
BM25 scores and vector similarities live on different scales, so adding them is
meaningless. RRF ignores scores and uses only RANKS: a paper at rank r in a list
gets 1 / (k + r); add up over the lists. Being near the top of both lists beats
being first in only one. k = 60 is the value from the paper.
"""
from __future__ import annotations


def rrf(lists: list, k: int = 60) -> list:
    """lists: [[key, ...] best first, ...] -> [(key, fused_score, [rank or None per list])]
    sorted by score desc, then key (deterministic). Ranks are 1-based."""
    score: dict = {}
    ranks: dict = {}
    for li, lst in enumerate(lists):
        seen = set()
        for pos, key in enumerate(lst, 1):
            if key in seen:
                continue
            seen.add(key)
            score[key] = score.get(key, 0.0) + 1.0 / (k + pos)
            ranks.setdefault(key, [None] * len(lists))[li] = pos
    return sorted(((key, s, ranks[key]) for key, s in score.items()), key=lambda t: (-t[1], t[0]))
