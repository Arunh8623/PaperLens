import itertools
import os
import random
import subprocess
import sys

import numpy as np
import pytest

from paperlens.graph.minhash import (MinHasher, dedupe_ranked, jaccard, miss_probability,
                                     near_duplicate_clusters, near_duplicate_pairs, shingles)
from paperlens.graph.rerank import citation_rerank
from paperlens.graph.csr import Graph

VOCAB = [f"w{i}" for i in range(3000)]


def doc(rng, n=80):
    return " ".join(rng.choice(VOCAB) for _ in range(n))


def mutate(text, rng, frac):
    w = text.split()
    for i in rng.sample(range(len(w)), int(len(w) * frac)):
        w[i] = rng.choice(VOCAB)
    return " ".join(w)


def test_shingles_basic_and_short_texts():
    assert len(shingles("a b c d e")) == 3                       # abc bcd cde
    assert shingles("Hello, HELLO") == shingles("hello hello")   # case/punctuation insensitive
    assert len(shingles("one two")) == 1 and shingles("") == set()


def test_estimate_tracks_exact_jaccard():
    rng = random.Random(1)
    h = MinHasher(512, seed=3)
    for frac in (0.0, 0.05, 0.2, 0.5):
        a = doc(rng)
        b = mutate(a, rng, frac)
        sa, sb = shingles(a), shingles(b)
        est = MinHasher.estimate(h.signature(sa), h.signature(sb))
        assert abs(est - jaccard(sa, sb)) < 0.08, (frac, est, jaccard(sa, sb))


def test_signatures_are_identical_across_processes_with_different_hash_seeds(tmp_path):
    code = ("from paperlens.graph.minhash import MinHasher, shingles;"
            "print(MinHasher(16,5).signature(shingles('the quick brown fox jumps over the lazy dog')).tolist())")
    src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
    outs = set()
    for hs in ("1", "2", "random"):
        env = dict(os.environ, PYTHONHASHSEED=hs, PYTHONPATH=src)
        outs.add(subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True).stdout)
    assert len(outs) == 1


def test_lsh_finds_every_planted_duplicate_and_reports_no_false_pairs():
    rng = random.Random(7)
    texts, planted = {}, []
    for i in range(60):
        texts[f"p{i}"] = doc(rng)
    for i in range(0, 60, 6):                                    # 10 near-duplicates (about 3% of words changed)
        texts[f"d{i}"] = mutate(texts[f"p{i}"], rng, 0.03)
        planted.append((f"d{i}", f"p{i}"))
    got = {(a, b) for a, b, _ in near_duplicate_pairs(texts, threshold=0.8)}
    assert all(tuple(sorted(p)) in got for p in planted)
    # brute-force oracle: every reported pair really is >= threshold, and nothing >= 0.8 was missed
    shs = {k: shingles(t) for k, t in texts.items()}
    truth = {(a, b) for a, b in itertools.combinations(sorted(texts), 2) if jaccard(shs[a], shs[b]) >= 0.8}
    assert got == truth


def test_clusters_merge_chains_and_dedupe_keeps_the_best_ranked():
    rng = random.Random(9)
    base = doc(rng, 120)
    texts = {"a": base, "b": mutate(base, rng, 0.02), "c": mutate(base, rng, 0.02), "z": doc(rng, 120), "y": doc(rng, 120)}
    cl = near_duplicate_clusters(texts, threshold=0.7)
    assert cl == [["a", "b", "c"]]
    kept, dropped = dedupe_ranked(["z", "c", "y", "a", "b"], texts, threshold=0.7)
    assert kept == ["z", "c", "y"] and dropped == {"a": "c", "b": "c"}


def test_empty_texts_are_not_duplicates_and_threshold_is_respected():
    assert near_duplicate_clusters({"x": "", "y": ""}) == []
    rng = random.Random(2)
    a = doc(rng)
    t = {"a": a, "b": mutate(a, rng, 0.3)}
    assert near_duplicate_clusters(t, threshold=0.95) == []
    with pytest.raises(ValueError):
        near_duplicate_pairs(t, num_perm=100, bands=16)


def test_miss_probability_is_tiny_for_true_duplicates_and_large_for_unrelated():
    assert miss_probability(0.9) < 1e-3 and miss_probability(0.3) > 0.99


# ----------------------------------------------------------------------- rerank
def test_rerank_lifts_the_paper_in_the_seeds_citation_neighbourhood():
    ids = [f"p{i}" for i in range(12)]
    g = Graph.from_edges(ids, [("p0", "p9"), ("p1", "p9"), ("p9", "p2")])      # p9 sits among the top seeds
    ranked = ids[:]                                                            # p9 starts at rank 10
    out = citation_rerank(ranked, g, seeds=3)
    order = [a for a, *_ in out]
    assert order.index("p9") < ranked.index("p9")
    assert sorted(order) == sorted(ranked)
    rec = {a: (gr, o) for a, _s, gr, o in out}
    assert rec["p9"][1] == 10 and rec["p9"][0] is not None


def test_rerank_without_graph_or_seeds_keeps_the_original_order():
    ids = [f"p{i}" for i in range(5)]
    assert [a for a, *_ in citation_rerank(ids, None)] == ids
    g = Graph.from_edges(["q1", "q2"], [("q1", "q2")])                         # none of the results are in the graph
    assert [a for a, *_ in citation_rerank(ids, g)] == ids
    assert citation_rerank([], g) == []


def test_candidates_that_collide_but_are_not_similar_enough_are_rejected_by_the_exact_check():
    rng = random.Random(21)
    texts = {}
    for i in range(80):                                   # pairs with Jaccard about 0.75: most collide in some band
        a = doc(rng)
        texts[f"a{i}"], texts[f"b{i}"] = a, mutate(a, rng, 0.06)
    shs = {k: shingles(t) for k, t in texts.items()}
    truth = {(f"a{i}", f"b{i}") for i in range(80) if jaccard(shs[f"a{i}"], shs[f"b{i}"]) >= 0.8}
    below = 80 - len(truth)
    assert below >= 20, "test needs many near-miss pairs"
    everything = near_duplicate_pairs(texts, threshold=0.0)          # every LSH candidate, no filtering
    got = {(a, b) for a, b, _ in near_duplicate_pairs(texts, threshold=0.8)}
    assert len(everything) > len(truth) + 10                         # many collisions were NOT good enough ...
    assert got == truth                                              # ... and the exact check removed them
