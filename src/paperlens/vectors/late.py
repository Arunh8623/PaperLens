"""Late-interaction (ColBERT-style) reranking with compressed token vectors  (Phase 6, stretch, OPTIONAL).

What it does and why (plain language)
-------------------------------------
One vector per paper squeezes everything into a single point. Late interaction keeps ONE VECTOR PER WORD and scores a
paper by letting every query word find its best-matching paper word:  MaxSim = sum over query tokens of
(max over document tokens of the dot product). It is slower, so it only re-orders the top ~100 candidates.
Token vectors are big, so we compress them (8-bit per number here; the scoring code works on the dequantized values).

HONEST STATUS
  * The token encoder here (`HashTokenEncoder`) is a STAND-IN: each word becomes a vector built from hashed character
    3-grams, so words that share letters ("index", "indexing") are close. It has no real meaning. A real ColBERT / SPECTER2
    token encoder is NOT implemented, so no claim about quality on real text is made.
  * Token vectors are computed at query time from the stored title+abstract; they are NOT stored in segments (the brief's
    planned `tokens` file is not built).
  * Like every ranking stage it must earn its place: `bench/late_ablation.py` measures it and prints keep / switch off.
"""
from __future__ import annotations

import numpy as np

from ..pipeline.embed import embed_terms
from ..text.tokenizer import tokenize


class HashTokenEncoder:
    name = "hash-trigram-tokens"

    def __init__(self, dim: int = 32, max_tokens: int = 96):
        self.dim, self.max_tokens = dim, max_tokens

    def token_vector(self, token: str) -> np.ndarray:
        padded = f"#{token}#"
        grams = {padded[i:i + 3]: 1 for i in range(max(1, len(padded) - 2))}
        return np.frombuffer(embed_terms(grams, self.dim).tobytes(), dtype=np.float32).copy()

    def encode(self, text: str) -> np.ndarray:
        toks = list(dict.fromkeys(tokenize(text)))[: self.max_tokens]          # distinct tokens, in order
        if not toks:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self.token_vector(t) for t in toks])


def quantize8(mat: np.ndarray):
    """Symmetric 8-bit: int8 values + one float32 scale per row. 4x smaller than float32."""
    scale = np.maximum(np.abs(mat).max(axis=1), 1e-12) / 127.0 if mat.size else np.zeros(0, dtype=np.float32)
    q = np.rint(mat / scale[:, None]).astype(np.int8) if mat.size else mat.astype(np.int8)
    return q, scale.astype(np.float32)


def dequantize8(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (q.astype(np.float32) * scale[:, None]) if q.size else np.zeros(q.shape, dtype=np.float32)


def maxsim(query_tokens: np.ndarray, doc_tokens: np.ndarray) -> float:
    if query_tokens.shape[0] == 0 or doc_tokens.shape[0] == 0:
        return 0.0
    return float((query_tokens @ doc_tokens.T).max(axis=1).sum())


def late_rerank(store, query: str, hits: list, encoder=None, depth: int = 100, compress: bool = True,
                weight: float = 1.0, rrf_k: int = 60) -> list:
    """Re-order the first `depth` hits by fusing their original rank with their MaxSim rank (RRF).
    Returns [(arxiv_id, fused_score, maxsim, original_rank)] best first; hits beyond `depth` keep their order after."""
    enc = encoder or HashTokenEncoder()
    q = enc.encode(query)
    head, tail = hits[:depth], hits[depth:]
    sims = []
    for h in head:
        p = store.get(h.arxiv_id)
        d = enc.encode(f"{p.title}. {p.abstract}") if p else np.zeros((0, enc.dim), dtype=np.float32)
        if compress and d.size:
            d = dequantize8(*quantize8(d))
        sims.append(maxsim(q, d))
    order = sorted(range(len(head)), key=lambda i: (-sims[i], i))
    ms_rank = {i: r for r, i in enumerate(order, 1)}
    fused = [(1.0 / (rrf_k + i + 1) + weight / (rrf_k + ms_rank[i]), i) for i in range(len(head))]
    fused.sort(key=lambda t: (-t[0], t[1]))
    out = [(head[i].arxiv_id, s, sims[i], i + 1) for s, i in fused]
    out += [(h.arxiv_id, 0.0, 0.0, len(head) + j + 1) for j, h in enumerate(tail)]
    return out
