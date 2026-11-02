"""Ranking metrics: nDCG@k, Recall@k, MRR@k (and their means over queries).

What they mean (plain language)
-------------------------------
For one query we have a ranked list of document ids and a judgement table
`qrels` {doc_id: relevance}, relevance 0 = not relevant, 1, 2, ... = more relevant.

* Recall@k  : of all relevant documents, what fraction appears in the top k?
* MRR@k     : 1 / (rank of the FIRST relevant document in the top k), else 0.
* nDCG@k    : rewards putting highly relevant documents near the top.
              DCG@k  = sum over ranks i=1..k of  rel_i / log2(i + 1)
              IDCG@k = DCG of the best possible ordering of the judged documents
              nDCG@k = DCG / IDCG   (0 if nothing is relevant)
              Gain is the relevance grade itself (linear), the trec_eval /
              pytrec_eval default that BEIR reports.

Documents missing from qrels count as relevance 0. Only relevance > 0 counts as
"relevant" for Recall and MRR.
"""
from __future__ import annotations

import math


def _rel(qrels: dict, doc_id) -> int:
    r = qrels.get(doc_id, 0)
    return r if r > 0 else 0


def dcg_at_k(rels, k: int) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels[:k]))


def ndcg_at_k(ranked: list, qrels: dict, k: int) -> float:
    ideal = sorted((r for r in qrels.values() if r > 0), reverse=True)
    idcg = dcg_at_k(ideal, k)
    if idcg == 0:
        return 0.0
    return dcg_at_k([_rel(qrels, d) for d in ranked[:k]], k) / idcg


def recall_at_k(ranked: list, qrels: dict, k: int) -> float:
    relevant = {d for d, r in qrels.items() if r > 0}
    if not relevant:
        return 0.0
    return len(relevant.intersection(ranked[:k])) / len(relevant)


def mrr_at_k(ranked: list, qrels: dict, k: int) -> float:
    for i, d in enumerate(ranked[:k], 1):
        if _rel(qrels, d) > 0:
            return 1.0 / i
    return 0.0


def evaluate(run: dict, qrels: dict, ndcg_k=(10,), recall_k=(100,), mrr_k=(10,)) -> dict:
    """run: {query_id: [doc_id, ...] ranked best first}; qrels: {query_id: {doc_id: rel}}.
    Averages over the queries present in qrels (a query missing from `run` scores 0)."""
    qids = sorted(qrels)
    if not qids:
        raise ValueError("no queries with judgements to evaluate")
    out = {}
    for k in ndcg_k:
        out[f"ndcg@{k}"] = sum(ndcg_at_k(run.get(q, []), qrels[q], k) for q in qids) / len(qids)
    for k in recall_k:
        out[f"recall@{k}"] = sum(recall_at_k(run.get(q, []), qrels[q], k) for q in qids) / len(qids)
    for k in mrr_k:
        out[f"mrr@{k}"] = sum(mrr_at_k(run.get(q, []), qrels[q], k) for q in qids) / len(qids)
    out["queries"] = len(qids)
    return out
