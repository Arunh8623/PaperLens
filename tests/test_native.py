"""Differential tests: the C++ functions and their Python twins must agree on the same inputs.
Skipped (with the build command) when the extension is not built."""
import array
import importlib
import os
import random
import subprocess
import sys
import warnings

import numpy as np
import pytest

import paperlens.native as native
from paperlens.store.index import IndexStore
from paperlens.store.filters import Filter
from paperlens.synth import generate_papers
from paperlens.text.varint import VarintError, encode_postings, encode_uvarint
from paperlens.vectors import quant

needs_native = pytest.mark.skipif(not native.HAVE_NATIVE, reason="C++ extension not built: python scripts/build_native.py")
K1, B = 1.2, 0.75


def rand_postings(rng, n, max_gap=50, max_tf=30):
    doc, pairs = -1, []
    for _ in range(n):
        doc += rng.randint(1, max_gap)
        pairs.append((doc, rng.randint(1, max_tf)))
    return pairs


@needs_native
@pytest.mark.parametrize("seed", range(5))
def test_decode_postings_equal_on_random_lists(seed):
    rng = random.Random(seed)
    pairs = rand_postings(rng, rng.choice([0, 1, 2, 50, 3000]), max_gap=rng.choice([1, 5, 100, 70000]),
                          max_tf=rng.choice([1, 127, 128, 10**6]))
    buf = b"\xff\xfe" + encode_postings(pairs) + b"trailing junk"
    a = native.decode_postings_py(buf, 2, len(pairs))
    c = native._native.decode_postings(buf, 2, len(pairs))
    assert a == c and a[0] == pairs
    assert native.decode_postings(bytearray(buf), 2, len(pairs)) == a            # bytearray/memoryview inputs too
    assert native.decode_postings(memoryview(buf), 2, len(pairs)) == a


@needs_native
def test_huge_values_and_boundaries():
    for v in (0, 127, 128, 16383, 16384, 2**32 - 1, 2**32, 2**63 - 1, 2**63, 2**64 - 1):
        buf = encode_uvarint(1) + encode_uvarint(max(v, 1))
        assert native.decode_postings_py(buf, 0, 1) == native._native.decode_postings(buf, 0, 1)


BAD = {
    "truncated": b"\x81",
    "truncated second number": b"\x01\x80",
    "empty": b"",
    "too long": b"\x01" + b"\x80" * 10 + b"\x01",
    "exceeds 64 bits": b"\x01" + b"\xff" * 9 + b"\x7f",
    "zero gap": b"\x00\x01",
    "zero tf": b"\x01\x00",
}


@needs_native
@pytest.mark.parametrize("name", sorted(BAD))
def test_corrupt_input_raises_the_same_error_with_the_same_message(name):
    with pytest.raises(VarintError) as py_err:
        native.decode_postings_py(BAD[name], 0, 1)
    with pytest.raises(VarintError) as cpp_err:
        native.decode_postings(BAD[name], 0, 1)
    assert str(py_err.value) == str(cpp_err.value)


def make_scoring_case(rng, n_docs):
    pairs = rand_postings(rng, rng.randint(1, max(1, n_docs // 3)), max_gap=3, max_tf=20)
    pairs = [(d, tf) for d, tf in pairs if d < n_docs]
    doclens = array.array("I", [rng.randint(1, 500) for _ in range(n_docs)])
    dead = np.zeros(n_docs, dtype=np.uint8)
    dead[rng.sample(range(n_docs), n_docs // 5)] = 1
    mask = np.array([rng.random() < 0.7 for _ in range(n_docs)], dtype=bool)
    return pairs, doclens, dead, mask


@needs_native
@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("use_dead,use_mask", [(False, False), (True, False), (False, True), (True, True)])
def test_bm25_accumulate_is_bit_identical(seed, use_dead, use_mask):
    rng = random.Random(seed)
    n = rng.choice([1, 10, 400, 5000])
    pairs, doclens, dead, mask = make_scoring_case(rng, n)
    if not pairs:
        pairs = [(0, 1)]
    buf = encode_postings(pairs)
    idf, avgdl = rng.uniform(0.01, 12.0), rng.uniform(5.0, 400.0)
    base = np.array([rng.random() for _ in range(n)])                              # accumulate onto non-zero values
    out_py, out_cpp = base.copy(), base.copy()
    args = (buf, 0, len(pairs), doclens, dead if use_dead else None, mask if use_mask else None, idf, K1, B, avgdl)
    s_py = native.bm25_accumulate_py(*args, out_py)
    s_cpp = native._native.bm25_accumulate(*args, out_cpp)
    assert s_py == s_cpp
    assert np.array_equal(out_py, out_cpp), "doubles differ: the operation order must match Python exactly"
    assert not np.array_equal(out_py, base) or s_py == 0


@needs_native
def test_bm25_accumulate_errors_agree():
    rng = random.Random(1)
    doclens = array.array("I", [5] * 10)
    buf = encode_postings([(3, 1), (12, 2)])                                       # doc 12 does not exist
    for fn in (native.bm25_accumulate_py, native._native.bm25_accumulate):
        with pytest.raises(IndexError):
            fn(buf, 0, 2, doclens, None, None, 1.0, K1, B, 4.0, np.zeros(10))
    with pytest.raises(ValueError):                                                # C++ validates array shapes up front
        native._native.bm25_accumulate(buf[:2], 0, 1, doclens, None, None, 1.0, K1, B, 4.0, np.zeros(9))
    with pytest.raises(ValueError):
        native._native.bm25_accumulate(buf[:2], 0, 1, doclens, np.zeros(3, dtype=np.uint8), None, 1.0, K1, B, 4.0, np.zeros(10))


@needs_native
@pytest.mark.parametrize("bits", [1, 2, 3, 4])
@pytest.mark.parametrize("dim", [8, 13, 64, 384])
def test_level_dot_matches_the_numpy_reference(bits, dim):
    rng = np.random.default_rng(bits * 1000 + dim)
    n = 37
    vecs = rng.normal(size=(n, dim)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    centroid = vecs.mean(axis=0).astype(np.float32)
    enc = quant.encode(vecs, centroid, bits, quant.rotation_matrix(dim))
    q = rng.normal(size=dim).astype(np.float32)
    ref = native.level_dot_py(enc["codes"], q, dim, bits)
    got = native._native.level_dot(enc["codes"], q, dim, bits)
    assert got.shape == ref.shape == (n,) and got.dtype == np.float32
    assert np.allclose(got, ref, rtol=1e-5, atol=1e-4)
    with pytest.raises(ValueError):
        native._native.level_dot(enc["codes"], q, dim, 5)
    with pytest.raises(ValueError):                                                # rows shorter than dim*bits bits
        native._native.level_dot(np.zeros((2, 0), dtype=np.uint8), q, dim, bits)


@needs_native
def test_search_results_are_identical_with_and_without_the_extension(tmp_path):
    st = IndexStore(tmp_path / "ix", flush_docs=10**9)
    papers = generate_papers(300, 7)
    rng = random.Random(3)
    for i, p in enumerate(papers):
        st.put(p)
        if i % 60 == 59:
            st.commit()
            st.flush()
    for p in rng.sample(papers[:240], 30):                                         # dead docs inside flushed segments
        st.delete(p.arxiv_id)
    st.commit()                                                                    # leaves a buffer tail too
    queries = ["inverted postings", "crash recovery journal", "pagerank citation graph", "scheduler latency",
               "gradient embedding quantization", "zzzunknownterm", "the"]
    flts = [None, Filter(categories=(papers[0].categories[0],)), Filter(year_from=2015, year_to=2022)]
    try:
        for q in queries:
            for flt in flts:
                for k in (1, 5, 10, None):
                    native.set_enabled(True)
                    a = [(h.key, h.score) for h in st.search(q, k=k, flt=flt)]
                    native.set_enabled(False)
                    b = [(h.key, h.score) for h in st.search(q, k=k, flt=flt)]
                    assert a == b, (q, k, flt)
    finally:
        native.set_enabled(True)
    st.close()


@needs_native
def test_quantized_vector_search_agrees_with_and_without_the_extension(tmp_path):
    from paperlens.vectors.embedders import HashEmbedder
    emb = HashEmbedder(64)
    st = IndexStore(tmp_path / "ix", flush_docs=10**9, vec_bits=2, ivf_min_docs=0)
    for i, p in enumerate(generate_papers(200, 4)):
        st.put(p, vector=emb.embed_paper(p))
    st.commit(); st.flush()
    try:
        for q in ["inverted postings", "crash journal", "citation graph"]:
            qv = emb.embed_query(q)
            native.set_enabled(True)
            a = st.search_vector(qv, k=10, pool=60)
            native.set_enabled(False)
            b = st.search_vector(qv, k=10, pool=60)
            assert [h.key for h in a] == [h.key for h in b]
            assert np.allclose([h.score for h in a], [h.score for h in b], atol=1e-6)
    finally:
        native.set_enabled(True)
    st.close()


# ------------------------------------------------------------------ fallback behaviour (runs everywhere)
def test_fallback_when_the_extension_is_switched_off_by_environment():
    src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
    code = ("import warnings; warnings.simplefilter('error')\n"
            "import paperlens.native as n; print(n.status()['have_native'], n.status()['enabled'])\n"
            "from paperlens.text.varint import encode_postings\n"
            "print(n.decode_postings(encode_postings([(2,3),(9,1)]),0,2)[0])\n")
    r = subprocess.run([sys.executable, "-c", code], env=dict(os.environ, PYTHONPATH=src, PAPERLENS_NATIVE="0"),
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split("\n")[0] == "False False" and "[(2, 3), (9, 1)]" in r.stdout


def test_missing_extension_gives_a_clear_warning_and_python_still_works(monkeypatch):
    import paperlens
    monkeypatch.setenv("PAPERLENS_NATIVE", "1")
    monkeypatch.setitem(sys.modules, "paperlens._native", None)                    # makes `import` fail
    monkeypatch.delattr(paperlens, "_native", raising=False)                       # ... also the cached attribute
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mod = importlib.reload(native)
    try:
        assert mod.HAVE_NATIVE is False and mod.status()["import_error"]
        assert any("scripts/build_native.py" in str(x.message) for x in w)
        assert mod.decode_postings(encode_postings([(1, 2)]), 0, 1)[0] == [(1, 2)]
        out = np.zeros(4)
        assert mod.bm25_accumulate(encode_postings([(1, 2)]), 0, 1, array.array("I", [3] * 4), None, None, 1.0, K1, B, 3.0, out) == 1
        assert out[1] > 0
    finally:
        monkeypatch.undo()
        importlib.reload(native)


@needs_native
@pytest.mark.parametrize("name", sorted(BAD))
def test_bm25_accumulate_rejects_corrupt_postings_like_the_python_twin(name):
    doclens = array.array("I", [5] * 10)
    with pytest.raises(VarintError) as py_err:
        native.bm25_accumulate_py(BAD[name], 0, 1, doclens, None, None, 1.0, K1, B, 4.0, np.zeros(10))
    out = np.zeros(10)
    with pytest.raises(VarintError) as cpp_err:
        native.bm25_accumulate(BAD[name], 0, 1, doclens, None, None, 1.0, K1, B, 4.0, out)
    assert str(py_err.value) == str(cpp_err.value)
    assert not out.any()                                                            # nothing half-added before the error
