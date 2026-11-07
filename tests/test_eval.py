import json
import math
import os

import pytest

from paperlens.eval.beir import BeirError, load_beir, run_beir
from paperlens.eval.metrics import evaluate, mrr_at_k, ndcg_at_k, recall_at_k


# ---------------------------------------------------------------- metrics, by hand
def test_ndcg_graded_hand_computed():
    # ranked a,b,c with grades 0,2,1:  DCG = 0 + 2/log2(3) + 1/log2(4)
    # ideal order 2,1,0:               IDCG = 2/log2(2) + 1/log2(3)
    qrels = {"a": 0, "b": 2, "c": 1}
    expected = (2 / math.log2(3) + 1 / math.log2(4)) / (2 / math.log2(2) + 1 / math.log2(3))
    assert ndcg_at_k(["a", "b", "c"], qrels, 3) == pytest.approx(expected, abs=1e-12)
    assert ndcg_at_k(["a", "b", "c"], qrels, 3) == pytest.approx(0.66970, abs=1e-4)   # the number itself


def test_ndcg_perfect_zero_and_cutoff():
    qrels = {"a": 1, "b": 1}
    assert ndcg_at_k(["a", "b", "x"], qrels, 10) == pytest.approx(1.0)
    assert ndcg_at_k(["x", "y", "z"], qrels, 10) == 0.0
    assert ndcg_at_k([], qrels, 10) == 0.0
    assert ndcg_at_k(["a"], {}, 10) == 0.0                         # nothing relevant: defined as 0
    far = ["x"] * 10 + ["a"]                                        # relevant doc at rank 11
    assert ndcg_at_k(far, {"a": 1}, 10) == 0.0
    assert ndcg_at_k(far, {"a": 1}, 11) == pytest.approx(1 / math.log2(12))


def test_ndcg_ideal_ranking_is_cut_at_k_too():
    """3 relevant docs but only the top 2 count: returning 2 of them in order is a PERFECT top-2."""
    qrels = {"a": 1, "b": 1, "c": 1}
    assert ndcg_at_k(["a", "b"], qrels, 2) == pytest.approx(1.0)
    assert ndcg_at_k(["a", "b", "c"], qrels, 2) == pytest.approx(1.0)
    graded = {"a": 3, "b": 2, "c": 1}                      # best top-2 is a,b
    assert ndcg_at_k(["a", "b"], graded, 2) == pytest.approx(1.0)
    assert ndcg_at_k(["b", "a"], graded, 2) == pytest.approx((2 + 3 / math.log2(3)) / (3 + 2 / math.log2(3)))


def test_ndcg_ignores_negative_grades_and_unjudged_docs():
    assert ndcg_at_k(["u", "a"], {"a": 1, "bad": -1}, 2) == pytest.approx(1 / math.log2(3))


def test_recall():
    qrels = {"a": 1, "b": 2, "c": 1, "d": 1, "n": 0}
    assert recall_at_k(["a", "x", "b"], qrels, 3) == 2 / 4
    assert recall_at_k(["a", "x", "b"], qrels, 1) == 1 / 4
    assert recall_at_k(["n", "x"], qrels, 5) == 0.0                 # grade 0 is not relevant
    assert recall_at_k(["a"], {}, 5) == 0.0


def test_mrr():
    qrels = {"a": 1}
    assert mrr_at_k(["x", "y", "z", "a"], qrels, 10) == 0.25
    assert mrr_at_k(["x", "y", "z", "a"], qrels, 3) == 0.0
    assert mrr_at_k(["a"], qrels, 10) == 1.0
    assert mrr_at_k([], qrels, 10) == 0.0


def test_evaluate_means_and_missing_run():
    qrels = {"q1": {"a": 1}, "q2": {"b": 1}}
    out = evaluate({"q1": ["a"], "q2": ["x", "b"]}, qrels)
    assert out["ndcg@10"] == pytest.approx((1 + 1 / math.log2(3)) / 2)
    assert out["recall@100"] == 1.0 and out["mrr@10"] == pytest.approx(0.75) and out["queries"] == 2
    out2 = evaluate({"q1": ["a"]}, qrels)                            # q2 never answered: scores 0
    assert out2["recall@100"] == 0.5
    with pytest.raises(ValueError):
        evaluate({}, {})


# ---------------------------------------------------------------- BEIR loader / runner
def write_dataset(root, docs=None, queries=None, qrels=None, header=True):
    os.makedirs(root / "qrels", exist_ok=True)
    docs = docs if docs is not None else [
        ("d1", "alpha systems", "alpha beta gamma"), ("d2", "delta", "delta epsilon kappa"),
        ("d3", "zeta", "zeta eta theta"), ("d4", "iota", "iota lambda"), ("d5", "mu", "mu nu xi")]
    queries = queries if queries is not None else [("q1", "alpha"), ("q2", "delta"), ("q3", "zeta")]
    qrels = qrels if qrels is not None else [("q1", "d1", 1), ("q2", "d2", 1), ("q2", "d5", 1), ("q3", "d4", 1)]
    with open(root / "corpus.jsonl", "w", encoding="utf-8") as f:
        for i, t, x in docs:
            f.write(json.dumps({"_id": i, "title": t, "text": x, "metadata": {}}) + "\n")
    with open(root / "queries.jsonl", "w", encoding="utf-8") as f:
        for i, x in queries:
            f.write(json.dumps({"_id": i, "text": x}) + "\n")
    with open(root / "qrels" / "test.tsv", "w", encoding="utf-8", newline="") as f:
        if header:
            f.write("query-id\tcorpus-id\tscore\r\n")
        for q, d, s in qrels:
            f.write(f"{q}\t{d}\t{s}\r\n")                              # CRLF on purpose: Windows-made files
    return root


def test_loader(tmp_path):
    corpus, queries, qrels = load_beir(write_dataset(tmp_path / "ds"))
    assert corpus["d1"] == ("alpha systems", "alpha beta gamma") and queries["q2"] == "delta"
    assert qrels == {"q1": {"d1": 1}, "q2": {"d2": 1, "d5": 1}, "q3": {"d4": 1}}
    corpus2, _, qrels2 = load_beir(write_dataset(tmp_path / "ds2", header=False))
    assert qrels2 == qrels


def test_loader_errors(tmp_path):
    with pytest.raises(BeirError, match="missing file"):
        load_beir(tmp_path / "nope")
    with pytest.raises(BeirError, match="'@'"):
        load_beir(write_dataset(tmp_path / "a", docs=[("x@y", "t", "t")]))
    with pytest.raises(BeirError, match="missing from queries"):
        load_beir(write_dataset(tmp_path / "b", qrels=[("ghost", "d1", 1)]))
    with pytest.raises(BeirError, match="no judgements"):
        load_beir(write_dataset(tmp_path / "c", qrels=[]))
    with pytest.raises(BeirError, match="missing judgements"):
        load_beir(write_dataset(tmp_path / "d"), split="dev")


def test_run_beir_end_to_end_matches_hand_computed(tmp_path):
    res = run_beir(write_dataset(tmp_path / "ds"), log=lambda *_: None)
    m = res["metrics"]
    # q1: perfect. q2: d2 first (nDCG = 1 / (1 + 1/log2 3), recall 1/2, mrr 1). q3: d4 never retrieved.
    assert m["ndcg@10"] == pytest.approx((1 + 1 / (1 + 1 / math.log2(3)) + 0) / 3, abs=1e-9)
    assert m["recall@100"] == pytest.approx((1 + 0.5 + 0) / 3)
    assert m["mrr@10"] == pytest.approx(2 / 3)
    assert m["queries"] == 3 and res["documents"] == 5 and res["dataset"] == "ds"
    assert res["search_ms"]["p95"] >= res["search_ms"]["p50"] >= 0 and res["index_bytes"] > 0


def test_run_beir_refuses_to_reuse_a_nonempty_index_dir(tmp_path):
    ds = write_dataset(tmp_path / "ds")
    run_beir(ds, index_dir=str(tmp_path / "ix"), log=lambda *_: None)
    with pytest.raises(BeirError, match="already holds an index"):
        run_beir(ds, index_dir=str(tmp_path / "ix"), log=lambda *_: None)


def test_run_beir_cleans_up_its_temp_index(tmp_path, monkeypatch):
    import tempfile
    monkeypatch.setenv("TMP", str(tmp_path)); monkeypatch.setenv("TEMP", str(tmp_path))
    monkeypatch.setenv("TMPDIR", str(tmp_path)); monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    run_beir(write_dataset(tmp_path / "ds"), log=lambda *_: None)
    assert not [p for p in os.listdir(tmp_path) if p.startswith("paperlens-beir-")]


def test_run_beir_with_an_embedder_reports_three_modes(tmp_path):
    from paperlens.vectors.embedders import HashEmbedder
    res = run_beir(write_dataset(tmp_path / "ds"), log=lambda *_: None, embedder=HashEmbedder(16))
    assert set(res["modes"]) == {"bm25", "vector", "hybrid"} and res["embedder"] == "hash:16"
    assert res["metrics"] == res["modes"]["bm25"]["metrics"]                 # Phase 1 keys unchanged
    for name, m in res["modes"].items():
        assert 0.0 <= m["metrics"]["ndcg@10"] <= 1.0 and m["search_ms"]["p95"] >= 0
    # the keyword side is still the hand-computed one from the Phase 1 test
    assert res["modes"]["bm25"]["metrics"]["mrr@10"] == pytest.approx(2 / 3)
    # q1 asks for "alpha": the paper literally contains it, so hash vectors find it first too
    assert res["modes"]["vector"]["metrics"]["mrr@10"] > 0
    assert isinstance(res["nfr6_hybrid_at_least_best_single"], bool)
    assert res["index_bytes"] > run_beir(write_dataset(tmp_path / "ds2"), log=lambda *_: None)["index_bytes"]


def test_run_beir_without_embedder_is_unchanged(tmp_path):
    res = run_beir(write_dataset(tmp_path / "ds"), log=lambda *_: None)
    assert set(res["modes"]) == {"bm25"} and res["nfr6_hybrid_at_least_best_single"] is None
