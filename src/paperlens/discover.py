"""Discovery features built on the index: "More like this" and "Key papers"  (Phase 6, FR11).

More like this: nearest papers to ONE paper. With a stored vector we search by that vector; without one we search by the
paper's title with BM25. The paper itself is never returned.
Key papers: among the top results of a query, the ones the field itself treats as central, measured by PageRank on the
citation graph (needs `citations` + `graph-build`); each also reports how many other results cite it.
"""
from __future__ import annotations

from .graph.rank import pagerank


class NoGraph(Exception):
    pass


def more_like_this(store, arxiv_id: str, k: int = 10, flt=None) -> list:
    p = store.get(arxiv_id)
    if p is None:
        raise KeyError(arxiv_id)
    v = store.get_vector(arxiv_id)
    if v is not None:
        hits = store.search_vector(v, k=k + 1, flt=flt)
        how = "meaning vector"
    else:
        hits = store.search(p.title, k=k + 1, flt=flt)
        how = "title keywords"
    return [(h, how) for h in hits if h.arxiv_id != arxiv_id][:k]


def key_papers(store, query: str, qvec=None, k: int = 5, depth: int = 100, flt=None) -> list:
    g = store.graph
    if g is None:
        raise NoGraph("no citation graph in this index")
    hits = store.search_hybrid(query, qvec, k=depth, flt=flt) if qvec is not None else store.search(query, k=depth, flt=flt)
    ids = [h.arxiv_id for h in hits if h.arxiv_id in g.index]
    if not ids:
        return []
    pr = getattr(g, "_pagerank_cache", None)
    if pr is None:
        pr, _ = pagerank(g)
        g._pagerank_cache = pr
    cand = set(ids)
    cited_by_results = {a: 0 for a in ids}
    for a in ids:
        for j in g.out(g.index[a]):
            b = g.ids[int(j)]
            if b in cand and b != a:
                cited_by_results[b] += 1
    ranked = sorted(ids, key=lambda a: (-float(pr[g.index[a]]), a))[:k]
    return [{"arxiv_id": a, "pagerank": float(pr[g.index[a]]), "cited_by_results": cited_by_results[a]} for a in ranked]
