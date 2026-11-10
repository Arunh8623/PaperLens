"""PageRank, personalized PageRank and label propagation on a CSR citation graph.

What they do and why (plain language)
-------------------------------------
PageRank: a paper is important if important papers cite it. Imagine a reader who
follows a random citation with probability `alpha` (0.85) and otherwise jumps to a
random paper; PageRank is how often that reader ends up at each paper. We find it by
POWER ITERATION: start uniform, repeat  r <- alpha * (spread r along citations)
+ (1 - alpha) * teleport  until it stops changing.

Dangling papers (cite nothing) would leak probability; their mass is spread by the
teleport distribution so the scores always add up to 1.

Personalized PageRank: the same, but the reader jumps back to a chosen SET of seed
papers instead of anywhere. The scores then mean "importance as seen from these
papers": a way to find papers related to the top search hits.

Label propagation: every paper starts with its own label (or a known one); repeatedly
each paper adopts the most common label among its neighbours (ties -> smallest label).
Papers that cite each other a lot end up sharing a label: a cheap topic community.
Reproducible: nodes are visited in a seeded random order each round and updated in place
("asynchronous"; a fully synchronous version makes two linked nodes swap labels forever), and a node
keeps its label when it is among the most common ones, to avoid oscillation.
"""
from __future__ import annotations

import numpy as np

from .csr import Graph


def pagerank(g: Graph, alpha: float = 0.85, tol: float = 1e-10, max_iter: int = 200,
             personalization=None) -> tuple:
    """Returns (scores float64[n] summing to 1, iterations used).
    personalization: None (uniform teleport) or a float array of non-negative weights (normalised here)."""
    n = g.n
    if n == 0:
        return np.zeros(0), 0
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if personalization is None:
        tele = np.full(n, 1.0 / n)
    else:
        tele = np.asarray(personalization, dtype=np.float64)
        if tele.shape != (n,) or (tele < 0).any() or tele.sum() <= 0:
            raise ValueError("personalization must be n non-negative weights, not all zero")
        tele = tele / tele.sum()
    outdeg = g.out_degree().astype(np.float64)
    src = g.sources()
    dst = g.targets.astype(np.int64)
    dangling = outdeg == 0
    r = tele.copy()
    for it in range(1, max_iter + 1):
        share = np.where(dangling, 0.0, r / np.where(dangling, 1.0, outdeg))
        spread = np.bincount(dst, weights=share[src], minlength=n)
        new = alpha * (spread + r[dangling].sum() * tele) + (1 - alpha) * tele
        delta = np.abs(new - r).sum()
        r = new
        if delta < tol:
            return r, it
    return r, max_iter


def personalized_pagerank(g: Graph, seeds: dict, alpha: float = 0.85, **kw) -> np.ndarray:
    """seeds: {arxiv_id: weight}. Unknown ids are ignored; at least one must exist."""
    w = np.zeros(g.n)
    for a, x in seeds.items():
        if a in g.index:
            w[g.index[a]] += x
    if w.sum() <= 0:
        raise ValueError("none of the seed papers is in the graph")
    return pagerank(g, alpha=alpha, personalization=w, **kw)[0]


def label_propagation(g: Graph, max_iter: int = 50, labels=None, seed: int = 0) -> tuple:
    """Community detection on the UNDIRECTED version of the graph (asynchronous; nodes visited in a seeded random order each round, as in Raghavan et al. 2007, so results are reproducible for a given seed). Returns (labels int64[n], rounds)."""
    n = g.n
    lab = np.arange(n, dtype=np.int64) if labels is None else np.asarray(labels, dtype=np.int64).copy()
    rev = g.reverse()
    rng = np.random.default_rng(seed)
    for rnd in range(1, max_iter + 1):
        changed = False
        for i in rng.permutation(n):
            nb = np.concatenate([g.out(i), rev.out(i)])
            if len(nb) == 0:
                continue
            vals, counts = np.unique(lab[nb], return_counts=True)
            best = vals[counts == counts.max()]
            if lab[i] in best:
                continue                                  # already among the most common: keep (no oscillation)
            lab[i] = best.min()                           # in place (asynchronous)
            changed = True                                # (fully synchronous updates make two linked nodes swap labels forever)
        if not changed:
            return lab, rnd
    return lab, max_iter
