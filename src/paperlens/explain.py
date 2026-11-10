"""Plain-language reasons for a search result (FR9).

Every result says WHY it is there, using only facts we really have: which query words occur in the title / abstract,
which list(s) found it (keyword, meaning) and at what rank, and whether the citation graph moved it.
"""
from __future__ import annotations

from .text.tokenizer import tokenize


def reasons(query: str, paper, mode: str, bm25_rank=None, vec_rank=None, graph_move=None, late_move=None) -> list:
    q_terms = list(dict.fromkeys(tokenize(query)))
    out = []
    if q_terms and paper is not None:
        in_title = [t for t in q_terms if t in set(tokenize(paper.title))]
        in_abs = [t for t in q_terms if t in set(tokenize(paper.abstract)) and t not in in_title]
        if in_title:
            out.append("Title contains: " + ", ".join(in_title))
        if in_abs:
            out.append("Abstract contains: " + ", ".join(in_abs))
        missing = [t for t in q_terms if t not in in_title and t not in in_abs]
        if (in_title or in_abs) and missing:
            out.append("Does not mention: " + ", ".join(missing))
    if mode == "hybrid":
        if bm25_rank and vec_rank:
            out.append(f"Found by both keyword search (#{bm25_rank}) and meaning search (#{vec_rank}), which ranks it higher")
        elif bm25_rank:
            out.append(f"Found by keyword search (#{bm25_rank}); meaning search did not rank it near the top")
        elif vec_rank:
            out.append(f"Found by meaning search (#{vec_rank}) although few query words appear in it")
    elif mode == "vector":
        out.append("Ranked by closeness of meaning to your query (not by exact words)")
    if graph_move:
        was, now = graph_move
        if now < was:
            out.append(f"Moved up from #{was} to #{now}: it sits close to the top results in the citation graph")
        elif now > was:
            out.append(f"Moved down from #{was} to #{now}: other results are better connected by citations")
    if late_move:
        was, now = late_move
        if was != now:
            out.append(f"Word-by-word matching moved it from #{was} to #{now}")
    return out
