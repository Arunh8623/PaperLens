"""Quantization, vector search, filters, hybrid fusion, fsck of vector files."""
import dataclasses
import json
import os
import zlib

import numpy as np
import pytest

from paperlens.store.filters import Filter, paper_year
from paperlens.store.fsck import fsck
from paperlens.store.index import IndexStore
from paperlens.store.reindex import reindex
from paperlens.synth import generate_papers, messy_stream
from paperlens.vectors import quant
from paperlens.vectors.embedders import HashEmbedder, make_embedder
from paperlens.vectors.fusion import rrf
from tests.crashlib import expected_vector

DIM = 24
E = HashEmbedder(DIM)


def exact_oracle(papers_live: dict, q, flt=None):
    """Independent brute force: cosine of half-precision unit vectors, filter applied by plain loops."""
    q = quant.unit(q)
    out = []
    for aid, p in papers_live.items():
        if flt is not None and not flt.empty and not flt.matches(paper_year(p.created, p.datestamp), p.categories):
            continue
        out.append((float(expected_vector(p, DIM) @ q), p.key))
    return sorted(out, key=lambda t: (-t[0], t[1]))


def build(path, papers, **kw):
    st = IndexStore(path, **kw)
    for p in papers:
        st.put(p, vector=E.embed_paper(p))
    st.commit()
    return st


# ------------------------------------------------------------------ quantizer
def clustered(n, dim, seed=0, clusters=20, spread=0.7):
    rng = np.random.default_rng(seed)
    c = rng.normal(size=(clusters, dim))
    x = c[rng.integers(0, clusters, n)] + spread * rng.normal(size=(n, dim))
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32), rng


def test_rotation_is_orthogonal_deterministic_and_seeded():
    r = quant.rotation_matrix(48)
    assert np.abs(r @ r.T - np.eye(48)).max() < 1e-5
    quant.rotation_matrix.cache_clear()
    assert np.array_equal(r, quant.rotation_matrix(48))
    assert not np.array_equal(r, quant.rotation_matrix(48, seed=1))
    assert quant.rotation_matrix(37).shape == (37, 37)                    # odd dimension works


@pytest.mark.parametrize("bits", [1, 2, 3, 4])
@pytest.mark.parametrize("dim", [8, 37, 64])
def test_pack_unpack_round_trip(bits, dim):
    rng = np.random.default_rng(bits * 100 + dim)
    codes = rng.integers(0, 1 << bits, size=(11, dim)).astype(np.uint8)
    packed = quant.pack_codes(codes, bits)
    assert packed.shape == (11, quant.code_bytes(dim, bits))
    levels = quant.unpack_levels(packed, dim, bits)
    assert np.array_equal(levels, (2 * codes.astype(np.int32) + 1 - (1 << bits)).astype(np.float32))


def test_more_bits_means_smaller_error_and_the_estimator_is_nearly_unbiased():
    x, rng = clustered(2000, 128)
    c = x.mean(axis=0)
    rot = quant.rotation_matrix(128)
    qs = x[:40] + 0.1 * rng.normal(size=(40, 128)).astype(np.float32)
    qs = (qs / np.linalg.norm(qs, axis=1, keepdims=True)).astype(np.float32)
    errs, bias = {}, {}
    for bits in (1, 2, 4):
        e = quant.encode(x, c, bits, rot)
        diffs = []
        for q in qs:
            est = quant.estimate(e["codes"], e["d_o"], e["inv"], e["oc"], quant.QueryPrep(q, c, rot), 128, bits)
            diffs.append(est - x @ q)
        d = np.concatenate(diffs)
        errs[bits], bias[bits] = float(np.abs(d).mean()), float(d.mean())
    assert errs[1] > errs[2] > errs[4]
    assert errs[1] < 0.06 and errs[4] < 0.012
    assert all(abs(b) < 0.006 for b in bias.values()), bias


def test_code_size_is_far_below_float32_for_one_bit():
    dim = 384
    per_vec = quant.code_bytes(dim, 1) + 12                              # codes + d_o, inv, oc
    assert quant.code_bytes(dim, 1) == 48                                # the brief's number
    assert per_vec <= dim * 4 / 8                                        # NFR7: <= 1/8 of float32 (60 vs 192 limit)
    assert quant.code_bytes(dim, 2) + 12 <= dim * 4 / 8                  # 2 bits still within the limit


def test_candidates_then_rescoring_reaches_high_recall_on_clustered_data():
    """SYNTHETIC clustered vectors (not real papers): shows the mechanism, not a quality claim."""
    x, rng = clustered(4000, 128, seed=3)
    c = x.mean(axis=0)
    rot = quant.rotation_matrix(128)
    qs = x[:60] + 0.15 * rng.normal(size=(60, 128)).astype(np.float32)
    qs = (qs / np.linalg.norm(qs, axis=1, keepdims=True)).astype(np.float32)
    for bits, floor in ((1, 0.95), (2, 0.98)):
        e = quant.encode(x, c, bits, rot)
        rec = []
        for q in qs:
            est = quant.estimate(e["codes"], e["d_o"], e["inv"], e["oc"], quant.QueryPrep(q, c, rot), 128, bits)
            exact = x @ q
            cand = np.argsort(-est)[:200]                                # pool of 200 out of 4000 (5%)
            final = cand[np.argsort(-exact[cand])[:10]]
            rec.append(len(set(final) & set(np.argsort(-exact)[:10])) / 10)
        assert np.mean(rec) >= floor, (bits, np.mean(rec))


def test_degenerate_inputs():
    with pytest.raises(ValueError):
        quant.unit(np.zeros(4))
    with pytest.raises(ValueError):
        quant.unit(np.array([1.0, np.nan]))
    with pytest.raises(ValueError):
        quant.encode(np.ones((2, 8), dtype=np.float32), np.zeros(8, dtype=np.float32), 5, quant.rotation_matrix(8))
    x = np.tile(np.eye(8, dtype=np.float32)[0], (3, 1))                  # every vector == centroid
    e = quant.encode(x, x.mean(axis=0), 1, quant.rotation_matrix(8))
    assert np.all(e["inv"] == 0) and np.all(np.isfinite(e["codes"]))


# ------------------------------------------------------------------ embedders
def test_hash_embedder_is_deterministic_unit_length_and_lexically_sensible():
    a, b = E.embed_query("graph partitioning heuristics"), E.embed_query("graph partitioning heuristics")
    assert np.array_equal(a, b) and abs(float(np.linalg.norm(a)) - 1) < 1e-5
    near = float(a @ E.embed_query("graph partitioning"))
    far = float(a @ E.embed_query("neural machine translation"))
    assert near > far
    with pytest.raises(ValueError):
        make_embedder("nonsense")
    with pytest.raises(ValueError):
        make_embedder("hash:2")
    assert make_embedder("hash:32").dim == 32


def test_sentence_transformers_spec_fails_clearly_when_not_installed():
    try:
        import sentence_transformers  # noqa: F401
        pytest.skip("sentence-transformers is installed here")
    except ImportError:
        with pytest.raises(RuntimeError, match="sentence-transformers is not installed"):
            make_embedder("st:sentence-transformers/all-MiniLM-L6-v2")


# ------------------------------------------------------------------ store: vector search vs oracle
def test_exact_search_equals_oracle_across_buffer_segments_replacements_and_deletes(tmp_path):
    papers = messy_stream(generate_papers(240, 5), seed=5)
    st = IndexStore(tmp_path / "ix", flush_docs=10 ** 9)
    live = {}
    for i, p in enumerate(papers):
        r = st.put(p, vector=E.embed_paper(p))
        if r.outcome in ("insert", "replace"):
            live[p.arxiv_id] = p
        if i % 60 == 59 and i < len(papers) - 20:
            st.commit()
            st.flush()                                                   # several segments + a buffer tail
    for aid in list(live)[:15]:                                          # deletes hit flushed segments too
        st.delete(aid)
        del live[aid]
    st.commit()
    assert len(st.segments) >= 3 and len(st.buf) > 0
    for query in ("checksum recovery", "graph partitioning", "neural"):
        q = E.embed_query(query)
        want = exact_oracle(live, q)[:10]
        got = [(h.score, h.key) for h in st.search_vector(q, k=10, exact=True)]
        assert [k for _, k in got] == [k for _, k in want]
        assert np.allclose([s for s, _ in got], [s for s, _ in want], atol=1e-6)
        every = st.search_vector(q, k=None)
        assert len(every) == len(live)                                   # k=None returns every live vector once
    st.close()


def test_quantized_search_with_a_big_pool_equals_exact_and_default_pool_has_high_overlap(tmp_path):
    papers = generate_papers(500, 6)
    st = build(tmp_path / "ix", papers, flush_docs=200)
    st.flush()
    q = E.embed_query("database transactions isolation")
    exact = [h.key for h in st.search_vector(q, k=10, exact=True)]
    big = [h.key for h in st.search_vector(q, k=10, pool=10 ** 6)]
    assert big == exact                                                  # pool covers everything -> rescoring decides
    default = [h.key for h in st.search_vector(q, k=10)]
    assert len(set(default) & set(exact)) >= 8
    st.close()


@pytest.mark.parametrize("bits", [1, 2, 3, 4])
def test_every_bit_width_round_trips_through_a_real_segment(tmp_path, bits):
    papers = generate_papers(120, 7)
    st = build(tmp_path / f"ix{bits}", papers, flush_docs=50, vec_bits=bits)
    st.flush()
    st.close()
    assert fsck(tmp_path / f"ix{bits}").ok
    with IndexStore(tmp_path / f"ix{bits}") as st2:
        q = E.embed_query("checksum")
        assert [h.key for h in st2.search_vector(q, k=5, pool=10 ** 6)] == \
               [k for _, k in exact_oracle({p.arxiv_id: p for p in papers}, q)[:5]]
        assert st2.segments[0].meta["vectors"]["bits"] == bits


def test_vectors_survive_reopen_from_segments_and_from_the_diary(tmp_path):
    papers = generate_papers(80, 8)
    st = build(tmp_path / "ix", papers[:50], flush_docs=10 ** 9)
    st.flush()
    for p in papers[50:]:
        st.put(p, vector=E.embed_paper(p))
    st.commit()
    # no close(): simulate an abrupt stop by releasing the lock only
    st._release_everything()
    with IndexStore(tmp_path / "ix") as st2:
        assert st2.recovery["replayed"] == 30                            # 30 papers came back from the diary
        for p in papers:
            assert np.array_equal(st2.get_vector(p.arxiv_id), expected_vector(p, DIM))
        assert st2.vec_dim == DIM


def test_replacing_and_deleting_a_paper_replaces_and_removes_its_vector(tmp_path):
    p = generate_papers(1, 9)[0]
    newer = dataclasses.replace(p, datestamp="2030-01-01", title="quantum annealing schedules",
                                abstract="adiabatic annealing for combinatorial optimisation")
    st = IndexStore(tmp_path / "ix", flush_docs=10 ** 9)
    st.put(p, vector=E.embed_paper(p))
    st.commit()
    st.flush()                                                           # old version lives in a segment
    st.put(newer, vector=E.embed_paper(newer))
    st.commit()
    hits = st.search_vector(E.embed_paper(p), k=5)
    assert [h.key for h in hits] == [newer.key]                          # the old version never comes back
    assert np.array_equal(st.get_vector(p.arxiv_id), expected_vector(newer, DIM))
    st.delete(p.arxiv_id)
    st.commit()
    assert st.search_vector(E.embed_paper(newer), k=5) == [] and st.get_vector(p.arxiv_id) is None
    st.flush()
    assert st.search_vector(E.embed_paper(newer), k=5) == []
    st.close()
    assert fsck(tmp_path / "ix").ok


def test_bad_vectors_are_refused_before_anything_is_logged(tmp_path):
    ps = generate_papers(3, 10)
    st = IndexStore(tmp_path / "ix")
    st.put(ps[0], vector=E.embed_paper(ps[0]))
    seq = st.stats()["last_seq"]
    for bad in (np.zeros(DIM), np.full(DIM, np.nan), np.ones(DIM + 1), np.ones((2, 2)), [1.0]):
        with pytest.raises(ValueError):
            st.put(ps[1], vector=bad)
    assert st.stats()["last_seq"] == seq and ps[1].arxiv_id not in st.live
    with pytest.raises(ValueError):
        st.search_vector(np.ones(DIM + 3))
    st.close()


def test_papers_without_vectors_and_phase1_segments_mix_with_vector_segments(tmp_path):
    a, b, c = generate_papers(3, 11)
    with IndexStore(tmp_path / "ix", flush_docs=10 ** 9) as st:
        st.put(a)                                                        # no vector at all (a "Phase 1" paper)
        st.commit()
        st.flush()
        assert not st.segments[0].has_vectors and st.search_vector(E.embed_paper(a)) == []
        st.put(b, vector=E.embed_paper(b))
        st.put(c)                                                        # no vector, inside a vector segment
        st.commit()
        st.flush()
        assert st.segments[1].has_vectors
        assert [h.key for h in st.search_vector(E.embed_paper(b), k=5)] == [b.key]
        assert st.get_vector(c.arxiv_id) is None and st.get_vector(a.arxiv_id) is None
        assert [h.key for h in st.search(a.title, k=1)] == [a.key]       # BM25 sees the vector-less paper
    assert fsck(tmp_path / "ix").ok


# ------------------------------------------------------------------ IVF
def test_ivf_probing_all_lists_equals_flat_scan_and_recall_grows_with_nprobe(tmp_path):
    papers = generate_papers(900, 12)
    st = build(tmp_path / "ix", papers, flush_docs=10 ** 9, ivf_min_docs=100)
    st.flush()
    seg = st.segments[0]
    nlist = seg.meta["vectors"]["ivf_nlist"]
    assert nlist >= 4 and seg._load_vectors()[6] is not None
    qs = [E.embed_query(t) for t in ("checksum recovery", "graph partitioning", "neural networks", "database index")]
    recalls = {1: [], 3: [], nlist: []}
    for q in qs:
        exact = {h.key for h in st.search_vector(q, k=10, exact=True)}
        flat = [h.key for h in st.search_vector(q, k=10, pool=10 ** 6)]
        for npb in recalls:
            got = {h.key for h in st.search_vector(q, k=10, nprobe=npb, pool=10 ** 6)}
            recalls[npb].append(len(got & exact) / 10)
        full = [h.key for h in st.search_vector(q, k=10, nprobe=nlist, pool=10 ** 6)]
        assert full == flat                                              # every list probed == no IVF
    mean = {k: float(np.mean(v)) for k, v in recalls.items()}
    assert mean[nlist] == 1.0 and mean[1] <= mean[3] <= mean[nlist]
    st.close()
    assert fsck(tmp_path / "ix").ok


def test_ivf_skipped_for_small_segments(tmp_path):
    st = build(tmp_path / "ix", generate_papers(40, 13), ivf_min_docs=100)
    st.flush()
    assert st.segments[0].meta["vectors"]["ivf_nlist"] == 0
    assert not os.path.exists(os.path.join(st.segments[0].path, "ivf.bin"))
    st.close()


# ------------------------------------------------------------------ fsck on vector files
def _segment_dir(path):
    return os.path.join(path, "segments", sorted(os.listdir(os.path.join(path, "segments")))[0])


def _set_crc(seg_dir, fname):
    """Pretend the byte flip was 'honest': recompute size+crc in meta.json (defeats the plain crc check)."""
    meta_path = os.path.join(seg_dir, "meta.json")
    meta = json.load(open(meta_path))
    data = open(os.path.join(seg_dir, fname), "rb").read()
    meta["files"][fname] = {"size": len(data), "crc32": zlib.crc32(data) & 0xFFFFFFFF}
    json.dump(meta, open(meta_path, "w"))


@pytest.mark.parametrize("fname", ["vectors.f16", "vectors.q", "ivf.bin", "attrs.json"])
def test_fsck_catches_a_flipped_byte_in_each_vector_file(tmp_path, fname):
    st = build(tmp_path / "ix", generate_papers(150, 14), flush_docs=10 ** 9, ivf_min_docs=50)
    st.flush()
    st.close()
    assert fsck(tmp_path / "ix").ok
    p = os.path.join(_segment_dir(tmp_path / "ix"), fname)
    data = bytearray(open(p, "rb").read())
    data[len(data) // 2] ^= 0xFF
    open(p, "wb").write(bytes(data))
    assert not fsck(tmp_path / "ix").ok


def test_fsck_cross_checks_codes_against_vectors_even_when_the_crc_was_fixed_up(tmp_path):
    """A corruption that also rewrites the checksum is only caught by re-deriving codes from the vectors."""
    st = build(tmp_path / "ix", generate_papers(150, 15), flush_docs=10 ** 9, ivf_min_docs=0)
    st.flush()
    st.close()
    seg = _segment_dir(tmp_path / "ix")
    p = os.path.join(seg, "vectors.f16")
    arr = np.frombuffer(open(p, "rb").read(), dtype="<f2").copy().reshape(150, DIM)
    arr[:, :] = arr[::-1, :]                                             # vectors silently swapped between papers
    open(p, "wb").write(arr.astype("<f2").tobytes())
    _set_crc(seg, "vectors.f16")
    rep = fsck(tmp_path / "ix")
    assert not rep.ok and any("quantized codes do not match" in e or "correction numbers" in e for e in rep.errors), rep.errors


def test_fsck_detects_attrs_that_disagree_with_papers(tmp_path):
    st = build(tmp_path / "ix", generate_papers(30, 16), flush_docs=10 ** 9)
    st.flush()
    st.close()
    seg = _segment_dir(tmp_path / "ix")
    a = json.load(open(os.path.join(seg, "attrs.json")))
    a["years"][3] += 1
    open(os.path.join(seg, "attrs.json"), "w").write(json.dumps(a))
    _set_crc(seg, "attrs.json")
    rep = fsck(tmp_path / "ix")
    assert any("attrs.json disagrees" in e for e in rep.errors)


# ------------------------------------------------------------------ filters
def test_filter_matching_rules_and_zone_map():
    f = Filter(categories=("cs.IR", "cs.LG"), year_from=2020, year_to=2022)
    assert f.matches(2021, ["cs.LG"]) and f.matches(2020, ["cs.DB", "cs.IR"]) and f.matches(2022, ["cs.IR"])
    assert not f.matches(2019, ["cs.IR"]) and not f.matches(2023, ["cs.IR"]) and not f.matches(2021, ["cs.DB"])
    assert Filter().empty and not Filter(year_from=1).empty
    assert Filter(year_from=2025).may_match_zone({"year_min": 2020, "year_max": 2026, "categories": []})
    assert not Filter(year_from=2025).may_match_zone({"year_min": 2020, "year_max": 2024, "categories": []})
    assert not Filter(year_to=2019).may_match_zone({"year_min": 2020, "year_max": 2024, "categories": []})
    assert not Filter(categories=("cs.DB",)).may_match_zone({"year_min": 1, "year_max": 3, "categories": ["cs.IR"]})
    assert paper_year("", "2021-05-06") == 2021 and paper_year("2019-01-01", "2021-05-06") == 2019


def test_filters_equal_brute_force_for_bm25_and_vectors_across_buffer_and_segments(tmp_path):
    papers = generate_papers(400, 17)
    st = IndexStore(tmp_path / "ix", flush_docs=10 ** 9)
    for i, p in enumerate(papers):
        st.put(p, vector=E.embed_paper(p))
        if i in (150, 300):
            st.commit()
            st.flush()
    st.commit()
    live = {p.arxiv_id: p for p in papers}
    years = sorted({paper_year(p.created, p.datestamp) for p in papers})
    cats = sorted({c for p in papers for c in p.categories})
    assert len(years) > 2 and len(cats) > 2, "synthetic data must vary for this test to mean anything"
    filters = [Filter(categories=(cats[0],)), Filter(year_from=years[len(years) // 2]),
               Filter(year_to=years[1]), Filter(categories=(cats[0], cats[1]), year_from=years[0], year_to=years[-2]),
               Filter(categories=("cs.NOPE",))]
    q = E.embed_query("checksum recovery index")
    for flt in filters:
        want_v = [k for _, k in exact_oracle(live, q, flt)[:10]]
        got_v = [h.key for h in st.search_vector(q, k=10, flt=flt, exact=True)]
        assert got_v == want_v, flt
        got_q = [h.key for h in st.search_vector(q, k=10, flt=flt, pool=10 ** 6)]
        assert got_q == want_v, flt
        allowed = {p.key for p in papers if flt.matches(paper_year(p.created, p.datestamp), p.categories)}
        bm_all = st.search("checksum recovery index", k=None)
        want_bm = [h.key for h in bm_all if h.key in allowed][:10]
        assert [h.key for h in st.search("checksum recovery index", k=10, flt=flt)] == want_bm, flt
    # a filter never shrinks the answer below what exists
    f = Filter(categories=(cats[0],))
    n_match = sum(1 for p in papers if cats[0] in p.categories)
    assert len(st.search_vector(q, k=n_match + 50, flt=f, exact=True)) == n_match
    st.close()


def test_whole_segment_is_skipped_by_the_zone_map(tmp_path):
    papers = [dataclasses.replace(p, created="2015-01-01") for p in generate_papers(30, 18)]
    st = build(tmp_path / "ix", papers, flush_docs=10 ** 9)
    st.flush()
    seg = st.segments[0]
    assert seg.meta["zonemap"]["year_min"] == seg.meta["zonemap"]["year_max"] == 2015
    mask = seg.filter_mask(Filter(year_from=2020))
    assert mask is not None and not mask.any()
    assert seg._attrs is None                                            # decided from meta.json alone, attrs never read
    assert st.search("checksum", k=5, flt=Filter(year_from=2020)) == []
    st.close()


def test_filters_work_on_phase1_segments_without_attrs(tmp_path):
    papers = generate_papers(60, 19)
    st = IndexStore(tmp_path / "ix", flush_docs=10 ** 9)
    for p in papers:
        st.put(p)
    st.commit()
    st.flush()
    st.close()
    seg = _segment_dir(tmp_path / "ix")
    meta = json.load(open(os.path.join(seg, "meta.json")))
    del meta["files"]["attrs.json"]
    del meta["zonemap"]
    json.dump(meta, open(os.path.join(seg, "meta.json"), "w"))
    os.remove(os.path.join(seg, "attrs.json"))                          # now it looks like a Phase 1 segment
    cat = papers[0].categories[0]
    query = papers[0].title + " " + papers[1].title                      # words that certainly occur
    with IndexStore(tmp_path / "ix") as st2:
        everything = st2.search(query, k=None)
        got = {h.key for h in st2.search(query, k=None, flt=Filter(categories=(cat,)))}
        want = {h.key for h in everything if cat in st2.get(h.arxiv_id).categories}
        assert got == want and got and len(got) < len(everything)         # the filter really removed something
    assert fsck(tmp_path / "ix").ok


# ------------------------------------------------------------------ RRF + hybrid
def test_rrf_hand_computed():
    out = rrf([["a", "b", "c"], ["b", "d", "a"]], k=60)
    assert [k for k, _, _ in out] == ["b", "a", "d", "c"]
    score = {k: s for k, s, _ in out}
    assert score["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert score["a"] == pytest.approx(1 / 61 + 1 / 63)
    assert score["d"] == pytest.approx(1 / 62) and score["c"] == pytest.approx(1 / 63)
    assert dict((k, r) for k, _, r in out)["d"] == [None, 2]


def test_rrf_edge_cases():
    assert rrf([]) == [] and rrf([[], []]) == []
    assert [k for k, _, _ in rrf([["x", "x", "y"]])] == ["x", "y"]       # a repeated key counts once, at its best rank
    tie = rrf([["b"], ["a"]])                                            # equal scores: deterministic by key
    assert [k for k, _, _ in tie] == ["a", "b"]
    assert rrf([["a"]], k=0)[0][1] == pytest.approx(1.0)


def test_hybrid_is_exactly_rrf_of_the_two_lists_and_explains_itself(tmp_path):
    papers = generate_papers(300, 20)
    st = build(tmp_path / "ix", papers, flush_docs=120)
    q_text = "checksum recovery index"
    qv = E.embed_query("recovery checksum journal")
    bm = st.search(q_text, k=100)
    vc = st.search_vector(qv, k=100)
    want = rrf([[h.key for h in bm], [h.key for h in vc]], k=60)[:10]
    got = st.search_hybrid(q_text, qv, k=10)
    assert [h.key for h in got] == [k for k, _, _ in want]
    for h, (k, s, ranks) in zip(got, want):
        assert h.score == pytest.approx(s) and [h.bm25_rank, h.vec_rank] == ranks
        assert (h.bm25_score is None) == (h.bm25_rank is None) and (h.vec_score is None) == (h.vec_rank is None)
    one_sided = [h for h in st.search_hybrid(q_text, qv, k=None) if h.bm25_rank is None or h.vec_rank is None]
    assert one_sided, "test data must contain papers found by only one side"
    assert all((h.bm25_score is None) == (h.bm25_rank is None) for h in one_sided)
    flt = Filter(year_from=2021)
    for h in st.search_hybrid(q_text, qv, k=20, flt=flt):
        p = st.get(h.arxiv_id)
        assert paper_year(p.created, p.datestamp) >= 2021
    only_kw = st.search_hybrid(q_text, None, k=5)                        # no query vector: keyword list alone
    assert [h.key for h in only_kw] == [h.key for h in st.search(q_text, k=5)]
    assert all(h.vec_rank is None and h.bm25_rank is not None for h in only_kw)
    st.close()


def test_hybrid_can_find_what_only_one_side_finds(tmp_path):
    """A paper with the right MEANING but different words is found by vectors only; RRF keeps it."""
    papers = generate_papers(100, 21)
    st = build(tmp_path / "ix", papers, flush_docs=10 ** 9)
    target = papers[7]
    qv = E.embed_paper(target)                                           # the query vector IS the target's vector
    hits = st.search_hybrid("zzzunseenword", qv, k=5)                    # keyword side finds nothing
    assert hits[0].key == target.key and hits[0].bm25_rank is None and hits[0].vec_rank == 1
    st.close()


# ------------------------------------------------------------------ reindex keeps vectors
def test_reindex_carries_vectors_over(tmp_path):
    papers = generate_papers(70, 22)
    st = build(tmp_path / "old", papers, flush_docs=30)
    st.flush()
    st.embedder_name = "hash:24"
    st.close()
    reindex(tmp_path / "old", tmp_path / "new", log=lambda *_: None)
    with IndexStore(tmp_path / "new") as new:
        for p in papers:
            assert np.array_equal(new.get_vector(p.arxiv_id), expected_vector(p, DIM))
    assert fsck(tmp_path / "new").ok


def test_embedder_name_is_recorded_in_segments_and_read_back(tmp_path):
    st = build(tmp_path / "ix", generate_papers(20, 23), embedder_name="hash:24")
    st.flush()
    st.close()
    with IndexStore(tmp_path / "ix") as st2:
        assert st2.embedder_name == "hash:24" and st2.stats()["embedder"] == "hash:24"


def test_default_pool_grows_with_the_index(tmp_path):
    papers = generate_papers(6000, 40)
    st = IndexStore(tmp_path / "ix", flush_docs=10 ** 9, ivf_min_docs=0)
    for p in papers[:60]:
        st.put(p, vector=E.embed_paper(p))
    assert st.default_pool(10) == 100 and st.default_pool(30) == 300      # small index: 10k, floor 100
    for p in papers[60:]:
        st.put(p, vector=E.embed_paper(p))
    assert st.default_pool(10) == 150                                     # 2.5% of 6000 beats the floor
    st.commit()
    st.flush()
    assert st.default_pool(10) == 150                                     # same after the buffer became a segment
    st.close()


def test_zone_map_boundaries_are_inclusive(tmp_path):
    papers = [dataclasses.replace(p, created="2015-06-01") for p in generate_papers(20, 41)]
    st = build(tmp_path / "ix", papers, flush_docs=10 ** 9)
    st.flush()
    seg = st.segments[0]
    for flt in (Filter(year_from=2015), Filter(year_to=2015), Filter(year_from=2015, year_to=2015)):
        m = seg.filter_mask(flt)
        assert m is not None and m.all(), flt                            # the edge year itself still matches
    for flt in (Filter(year_from=2016), Filter(year_to=2014)):
        assert not seg.filter_mask(flt).any(), flt
    assert len(st.search_vector(E.embed_query("checksum"), k=5, flt=Filter(year_from=2015), exact=True)) == 5
    st.close()
