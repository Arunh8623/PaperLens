"""Citation-aware reranking (optional stage; must earn its place in the ablation or be switched off).

What it does and why (plain language)
-------------------------------------
Embeddings find papers on the same TOPIC; citations hint at papers on the same AGENDA. Idea:
take the top few results as seeds, run Personalized PageRank from them over the citation graph
(a random surfer who keeps hopping to cited / citing papers and keeps teleporting back to the
seeds), and give a candidate a boost when it sits in the seeds' citation neighbourhood.
The graph ordering of the candidate list is then merged with the original ordering by
Reciprocal Rank Fusion; `weight` < 1 lets the original ranking count for more.
Candidates with no graph signal (score 0 or not in the graph) keep only their original rank
signal. The function returns reasons so the UI can explain every move.
"""
from __future__ import annotations

import numpy as np

from .rank import personalized_pagerank


def citation_rerank(ranked_ids: list, graph, seeds: int = 5, weight: float = 1.0, rrf_k: int = 60,
                    alpha: float = 0.85, undirected: bool = True) -> list:
    """ranked_ids: arxiv ids best-first. Returns [(arxiv_id, fused_score, graph_rank_or_None, original_rank)]."""
    if graph is None or not ranked_ids:
        return [(a, 1.0 / (rrf_k + i), None, i) for i, a in enumerate(ranked_ids, 1)]
    in_graph = [a for a in ranked_ids[:seeds] if a in graph.index]
    out = {}
    graph_rank = {}
    if in_graph:
        seed_w = {a: 1.0 / (i + 1) for i, a in enumerate(in_graph)}
        g = graph
        if undirected:                       # citing and cited papers are both "near" in agenda
            from .csr import Graph
            edges = [(g.ids[i], g.ids[int(j)]) for i in range(g.n) for j in g.out(i)]
            g = Graph.from_edges(g.ids, edges + [(b, a) for a, b in edges])
        ppr = personalized_pagerank(g, seed_w, alpha=alpha)
        scored = [(float(ppr[g.index[a]]), a) for a in ranked_ids if a in g.index]
        scored.sort(key=lambda t: (-t[0], ranked_ids.index(t[1])))
        for r, (s, a) in enumerate(scored, 1):
            graph_rank[a] = r
    for i, a in enumerate(ranked_ids, 1):
        s = 1.0 / (rrf_k + i)
        if a in graph_rank:
            s += weight / (rrf_k + graph_rank[a])
        out[a] = (s, graph_rank.get(a), i)
    fused = sorted(out.items(), key=lambda kv: (-kv[1][0], kv[1][2]))
    return [(a, s, gr, orig) for a, (s, gr, orig) in fused]
