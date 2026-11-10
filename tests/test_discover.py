import numpy as np
import pytest

from paperlens.discover import NoGraph, key_papers, more_like_this
from paperlens.graph.csr import Graph
from paperlens.store.index import IndexStore
from paperlens.synth import generate_papers
from paperlens.vectors.embedders import HashEmbedder
from paperlens.vectors.late import (HashTokenEncoder, dequantize8, late_rerank, maxsim, quantize8)


def make(tmp_path, vectors=True, n=80):
    st = IndexStore(tmp_path / "ix", flush_docs=10**9, embedder_name="hash:32" if vectors else "", ivf_min_docs=0)
    emb = HashEmbedder(32)
    ps = generate_papers(n, 3)
    for p in ps:
        st.put(p, vector=emb.embed_paper(p) if vectors else None)
    st.commit(); st.flush()
    return st, ps, emb


def test_more_like_this_excludes_itself_and_equals_exact_neighbours(tmp_path):
    st, ps, emb = make(tmp_path)
    seed = ps[10]
    got = more_like_this(st, seed.arxiv_id, k=5)
    assert len(got) == 5 and all(h.arxiv_id != seed.arxiv_id for h, _ in got) and got[0][1] == "meaning vector"
    want = [h.arxiv_id for h in st.search_vector(st.get_vector(seed.arxiv_id), k=6, exact=True) if h.arxiv_id != seed.arxiv_id][:5]
    assert len({h.arxiv_id for h, _ in got} & set(want)) >= 4
    scores = [h.score for h, _ in got]
    assert scores == sorted(scores, reverse=True)
    with pytest.raises(KeyError):
        more_like_this(st, "9999.99999")
    st.close()


def test_more_like_this_falls_back_to_title_keywords_without_vectors(tmp_path):
    st, ps, _ = make(tmp_path, vectors=False)
    got = more_like_this(st, ps[3].arxiv_id, k=4)
    assert got and all(how == "title keywords" for _, how in got) and all(h.arxiv_id != ps[3].arxiv_id for h, _ in got)
    st.close()


def test_key_papers_ranks_by_pagerank_among_the_results(tmp_path):
    st, ps, _ = make(tmp_path, vectors=False, n=60)
    with pytest.raises(NoGraph):
        key_papers(st, "inverted postings")
    ids = [p.arxiv_id for p in ps]
    hub = ids[7]
    st.publish_graph(Graph.from_edges(ids, [(a, hub) for a in ids if a != hub] + [(ids[1], ids[2])]))
    hub_title = st.get(hub).title
    assert hub in {h.arxiv_id for h in st.search(hub_title, k=100)}
    got = key_papers(st, hub_title, k=3, depth=100)
    prs = [g["pagerank"] for g in got]
    assert len(got) == 3 and prs == sorted(prs, reverse=True) and prs[0] > prs[-1]
    assert got[0]["arxiv_id"] == hub                                           # everybody cites it
    assert got[0]["cited_by_results"] >= 2
    st.close()


# ------------------------------------------------------------------ late interaction
def test_token_encoder_is_deterministic_and_similar_words_are_close():
    e = HashTokenEncoder(dim=64)
    a, b = e.encode("indexing"), e.encode("indexing")
    assert np.array_equal(a, b)
    idx, indexes, zebra = (e.token_vector(w) for w in ("index", "indexes", "zebra"))
    assert float(idx @ indexes) > float(idx @ zebra)
    assert e.encode("").shape == (0, 64)
    assert e.encode("alpha beta alpha beta").shape[0] == 2                                      # distinct tokens only
    assert e.encode(" ".join(f"w{i}x" for i in range(500))).shape[0] == e.max_tokens


def test_8bit_quantization_is_small_and_nearly_lossless():
    rng = np.random.default_rng(0)
    m = rng.normal(size=(40, 32)).astype(np.float32)
    q, s = quantize8(m)
    assert q.dtype == np.int8 and q.nbytes + s.nbytes < m.nbytes / 3
    assert (np.abs(q).max(axis=1) == 127).all()                                # every row uses the full 8-bit range
    assert np.abs(dequantize8(q, s) - m).max() <= s.max() / 2 + 1e-6
    assert quantize8(np.zeros((0, 32), np.float32))[0].shape == (0, 32)


def test_maxsim_is_the_sum_of_best_matches():
    q = np.array([[1, 0], [0, 1]], dtype=np.float32)
    d = np.array([[1, 0], [0.5, 0.5]], dtype=np.float32)
    assert maxsim(q, d) == pytest.approx(1.0 + 0.5)
    assert maxsim(q, np.zeros((0, 2), np.float32)) == 0.0


def test_late_rerank_keeps_every_hit_and_lifts_the_best_token_match(tmp_path):
    st, ps, _ = make(tmp_path, vectors=False, n=120)
    hits = st.search("inverted postings segment", k=60)
    depth = len(hits) // 2
    assert depth >= 5 and len(hits) - depth >= 5
    out = late_rerank(st, "inverted postings segment", hits, depth=depth)
    assert sorted(a for a, *_ in out) == sorted(h.arxiv_id for h in hits)
    assert [a for a, *_ in out[depth:]] == [h.arxiv_id for h in hits[depth:]]     # the tail is untouched
    best = max(out[:depth], key=lambda t: t[2])
    assert [a for a, *_ in out[:depth]].index(best[0]) <= [h.arxiv_id for h in hits].index(best[0])
    a = late_rerank(st, "inverted postings segment", hits, compress=True)
    b = late_rerank(st, "inverted postings segment", hits, compress=False)
    assert len({x for x, *_ in a[:10]} & {x for x, *_ in b[:10]}) >= 9             # 8-bit barely changes the top 10
    st.close()
