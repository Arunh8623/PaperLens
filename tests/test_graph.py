import json
import math
import random

import numpy as np
import pytest

from paperlens.graph.csr import Graph
from paperlens.graph.fake_openalex import FakeOpenAlex
from paperlens.graph.openalex import (BudgetExhausted, OpenAlexClient, OpenAlexError, arxiv_to_doi,
                                      build_graph, doi_to_arxiv, fetch_citations)
from paperlens.graph.rank import label_propagation, pagerank, personalized_pagerank
from paperlens.harvest.limiter import MinIntervalLimiter
from paperlens.pipeline.shared_limiter import SharedBudget


# ------------------------------------------------------------------ independent oracle
def dense_pagerank(g: Graph, alpha=0.85, tele=None):
    """Solve (I - alpha * M) r = (1 - alpha) * tele directly with linear algebra (no iteration)."""
    n = g.n
    tele = np.full(n, 1.0 / n) if tele is None else np.asarray(tele, float) / np.sum(tele)
    M = np.zeros((n, n))
    for i in range(n):
        outs = g.out(i)
        if len(outs) == 0:
            M[:, i] = tele                       # dangling: jump by the teleport distribution
        else:
            for j in outs:
                M[j, i] += 1.0 / len(outs)
    return np.linalg.solve(np.eye(n) - alpha * M, (1 - alpha) * tele)


def random_graph(n, m, seed):
    rng = random.Random(seed)
    ids = [f"p{i:03d}" for i in range(n)]
    edges = {(ids[rng.randrange(n)], ids[rng.randrange(n)]) for _ in range(m)}
    return Graph.from_edges(ids, edges)


# ------------------------------------------------------------------ CSR
def test_from_edges_drops_unknown_self_and_duplicate_edges():
    g = Graph.from_edges(["a", "b", "c"], [("a", "b"), ("a", "b"), ("a", "a"), ("a", "zzz"), ("b", "c")])
    assert g.n == 3 and g.m == 2
    assert list(g.out(0)) == [1] and list(g.out(1)) == [2] and list(g.out(2)) == []
    assert list(g.out_degree()) == [1, 1, 0] and list(g.in_degree()) == [0, 1, 1]


def test_reverse_and_round_trip_bytes():
    g = random_graph(40, 120, 1)
    r = g.reverse()
    assert r.m == g.m
    fwd = {(i, int(j)) for i in range(g.n) for j in g.out(i)}
    rev = {(int(j), i) for i in range(r.n) for j in r.out(i)}
    assert fwd == rev
    g2 = Graph.from_bytes(g.to_bytes())
    assert g2.ids == g.ids and np.array_equal(g2.offsets, g.offsets) and np.array_equal(g2.targets, g.targets)
    with pytest.raises(ValueError):
        Graph.from_bytes(b"XXXX" + g.to_bytes()[4:])
    with pytest.raises(ValueError):
        Graph.from_bytes(g.to_bytes()[:-3])
    with pytest.raises(ValueError):
        Graph(["a"], [0, 1], [5])                                   # edge outside the graph


def test_empty_graph_and_nodes_without_edges():
    g = Graph.from_edges(["a", "b"], [])
    assert g.m == 0 and list(g.out(0)) == []
    r, _ = pagerank(g)
    assert np.allclose(r, [0.5, 0.5])
    assert Graph.from_bytes(g.to_bytes()).n == 2
    assert pagerank(Graph.from_edges([], []))[0].size == 0


# ------------------------------------------------------------------ PageRank
def test_pagerank_hand_computed_three_cycle_plus_pendant():
    # a->b->c->a is symmetric; d->c adds a pendant.  Verified against the linear-algebra solution below.
    g = Graph.from_edges(list("abcd"), [("a", "b"), ("b", "c"), ("c", "a"), ("d", "c")])
    r, iters = pagerank(g, tol=1e-12)
    assert r.sum() == pytest.approx(1.0, abs=1e-12) and iters < 200
    assert np.allclose(r, dense_pagerank(g), atol=1e-9)
    assert r[3] == pytest.approx(0.15 / 4)                           # nobody cites d: only teleport reaches it
    assert r[2] > r[0] > r[1] > r[3] or r[2] > r[1] > r[0] > r[3]    # c is cited twice


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_pagerank_equals_direct_solution_on_random_graphs_with_dangling_nodes(seed):
    g = random_graph(60, 140, seed)
    assert (g.out_degree() == 0).any(), "test needs dangling nodes"
    r, _ = pagerank(g, tol=1e-13, max_iter=500)
    assert np.allclose(r, dense_pagerank(g), atol=1e-9)
    assert r.sum() == pytest.approx(1.0, abs=1e-10)


def test_pagerank_other_alpha_and_validation():
    g = random_graph(30, 70, 4)
    assert np.allclose(pagerank(g, alpha=0.5, tol=1e-13)[0], dense_pagerank(g, alpha=0.5), atol=1e-9)
    for bad in (0.0, 1.0, -1):
        with pytest.raises(ValueError):
            pagerank(g, alpha=bad)
    with pytest.raises(ValueError):
        pagerank(g, personalization=np.zeros(g.n))
    with pytest.raises(ValueError):
        pagerank(g, personalization=np.ones(g.n - 1))


def test_personalized_pagerank_equals_direct_solution_and_favours_the_seed_neighbourhood():
    g = random_graph(50, 150, 5)
    seeds = {"p003": 2.0, "p010": 1.0, "unknown": 5.0}              # unknown ids are ignored
    tele = np.zeros(g.n)
    tele[g.index["p003"]], tele[g.index["p010"]] = 2.0, 1.0
    ppr = personalized_pagerank(g, seeds, tol=1e-13, max_iter=500)
    assert np.allclose(ppr, dense_pagerank(g, tele=tele), atol=1e-9)
    uniform, _ = pagerank(g)
    assert ppr[g.index["p003"]] > uniform[g.index["p003"]]
    with pytest.raises(ValueError):
        personalized_pagerank(g, {"nope": 1.0})


# ------------------------------------------------------------------ label propagation
def test_label_propagation_finds_two_cliques_joined_by_one_edge():
    a, b = [f"a{i}" for i in range(6)], [f"b{i}" for i in range(6)]
    edges = [(x, y) for x in a for y in a if x < y] + [(x, y) for x in b for y in b if x < y] + [("a0", "b0")]
    g = Graph.from_edges(a + b, edges)
    lab, rounds = label_propagation(g)
    assert rounds < 50
    assert len(set(lab[:6])) == 1 and len(set(lab[6:])) == 1 and lab[0] != lab[6]
    for seed in range(1, 6):                                          # robust to the visiting order
        l2, _ = label_propagation(g, seed=seed)
        assert len(set(l2[:6])) == 1 and len(set(l2[6:])) == 1 and l2[0] != l2[6]


def test_label_propagation_is_deterministic_and_isolated_nodes_keep_their_label():
    g = random_graph(40, 60, 6)
    l1, _ = label_propagation(g)
    l2, _ = label_propagation(g)
    assert np.array_equal(l1, l2)
    iso = Graph.from_edges(["x", "y", "z"], [("x", "y")])
    lab, _ = label_propagation(iso)
    assert lab[2] == 2 and lab[0] == lab[1]
    seeded, _ = label_propagation(iso, labels=[7, 7, 9])
    assert list(seeded) == [7, 7, 9]


# ------------------------------------------------------------------ OpenAlex client against the fake server
def synthetic_citations(n=70, seed=0):
    rng = random.Random(seed)
    ids = [f"2101.{i:05d}" for i in range(n)]
    refs = {a: sorted(rng.sample(ids[:i], min(i, rng.randint(0, 6)))) for i, a in enumerate(ids)}
    return ids, refs


def fast_client(url, **kw):
    kw.setdefault("limiter", MinIntervalLimiter(0.0))
    kw.setdefault("backoff_base", 0.01)
    return OpenAlexClient(base_url=url, **kw)


def test_doi_helpers():
    assert arxiv_to_doi("2101.00001") == "10.48550/arxiv.2101.00001"
    assert arxiv_to_doi("cs/0501001") == "10.48550/arxiv.cs/0501001"
    assert doi_to_arxiv("https://doi.org/10.48550/arXiv.2101.00001") == "2101.00001"
    assert doi_to_arxiv("10.1000/other") is None and doi_to_arxiv(None) is None


def test_fetch_and_build_graph_equals_the_ground_truth(tmp_path):
    ids, refs = synthetic_citations()
    with FakeOpenAlex(refs) as srv:
        c = fast_client(srv.url, batch=25, api_key="k")
        recs = fetch_citations(c, ids, tmp_path / "cites.jsonl")
        assert srv.count == math.ceil(len(ids) / 25) and c.requests_made == srv.count
        assert all(q.get("api_key") == "k" and q["per_page"] == "100" for _, q in srv.log)
    g = build_graph(ids, recs)
    truth = {(a, b) for a, cited in refs.items() for b in cited}
    got = {(g.ids[i], g.ids[int(j)]) for i in range(g.n) for j in g.out(i)}
    assert got == truth                                               # and the W999999 reference was dropped


def test_resume_does_not_ask_again_and_papers_unknown_to_openalex_are_remembered(tmp_path):
    ids, refs = synthetic_citations(40)
    known = {a: refs[a] for a in ids[:30]}                           # OpenAlex knows only the first 30
    path = tmp_path / "cites.jsonl"
    with FakeOpenAlex(known) as srv:
        fetch_citations(fast_client(srv.url, batch=10), ids, path)
        first = srv.count
    with FakeOpenAlex(known) as srv2:
        recs = fetch_citations(fast_client(srv2.url, batch=10), ids, path)
        assert srv2.count == 0 and first == 4                       # nothing re-fetched, even the unknown 10
    assert recs[ids[35]]["openalex_id"] is None and recs[ids[5]]["openalex_id"].startswith("https://openalex.org/W")


def test_torn_last_line_is_tolerated_on_resume(tmp_path):
    ids, refs = synthetic_citations(20)
    path = tmp_path / "cites.jsonl"
    with FakeOpenAlex(refs) as srv:
        fetch_citations(fast_client(srv.url, batch=10), ids[:10], path)
    with open(path, "ab") as f:
        f.write(b'{"arxiv_id":"2101.0001')                           # crash mid-write
    with FakeOpenAlex(refs) as srv:
        recs = fetch_citations(fast_client(srv.url, batch=10), ids, path)
    assert set(recs) == set(ids)


def test_429_with_retry_after_is_waited_out_and_penalises_the_shared_limiter(tmp_path):
    ids, refs = synthetic_citations(20)
    faults = {1: ("http", 429, "0", ""), 2: ("http", 503, None, ""), 3: ("drop",)}
    with FakeOpenAlex(refs, faults=faults) as srv:
        recs = fetch_citations(fast_client(srv.url, batch=20), ids, tmp_path / "c.jsonl")
        assert srv.count == 4 and len(recs) == 20


def test_own_budget_is_never_exceeded_and_the_run_can_continue_later(tmp_path):
    ids, refs = synthetic_citations(60)
    path = tmp_path / "c.jsonl"
    budget = SharedBudget(300)                                       # three requests of 100 units
    with FakeOpenAlex(refs) as srv:
        c = fast_client(srv.url, batch=10, budget=budget)
        with pytest.raises(BudgetExhausted):
            fetch_citations(c, ids, path)
        assert srv.count == 3 and budget.spent == 300                # not one request over budget
    assert sum(1 for _ in open(path)) == 30                          # what was fetched is saved
    with FakeOpenAlex(refs) as srv:
        recs = fetch_citations(fast_client(srv.url, batch=10), ids, path)
        assert srv.count == 3 and len(recs) == 60                    # only the missing 30 were asked for


def test_server_side_remaining_budget_stops_the_run(tmp_path):
    ids, refs = synthetic_citations(60)
    with FakeOpenAlex(refs, remaining=0.00015) as srv:               # after 2 requests the header goes <= 0
        c = fast_client(srv.url, batch=10)
        with pytest.raises(BudgetExhausted, match="OpenAlex says"):
            fetch_citations(c, ids, tmp_path / "c.jsonl")
        assert srv.count == 2


def test_inactive_key_is_reported_clearly_and_other_client_errors_do_not_retry(tmp_path):
    ids, refs = synthetic_citations(10)
    with FakeOpenAlex(refs, require_key="secret") as srv:
        with pytest.raises(OpenAlexError, match="not activated"):
            fetch_citations(fast_client(srv.url, api_key="wrong"), ids, tmp_path / "c.jsonl")
        assert srv.count == 1
    with FakeOpenAlex(refs, faults={1: ("http", 400, None, "bad")}) as srv:
        with pytest.raises(OpenAlexError, match="HTTP 400"):
            fetch_citations(fast_client(srv.url), ids, tmp_path / "d.jsonl")
        assert srv.count == 1


def test_gives_up_after_max_retries(tmp_path):
    ids, refs = synthetic_citations(5)
    with FakeOpenAlex(refs, faults={i: ("http", 503, None, "") for i in range(1, 20)}) as srv:
        with pytest.raises(OpenAlexError, match="giving up"):
            fetch_citations(fast_client(srv.url, max_retries=2), ids, tmp_path / "c.jsonl")
        assert srv.count == 3


def test_requests_respect_the_minimum_interval(tmp_path):
    ids, refs = synthetic_citations(30)
    with FakeOpenAlex(refs) as srv:
        fetch_citations(fast_client(srv.url, batch=5, limiter=MinIntervalLimiter(0.05)), ids, tmp_path / "c.jsonl")
        times = [t for t, _ in srv.log]
    # The limiter spaces the SENDS; the server stamps ARRIVAL. The arrival gap can be ~15 ms shorter on Windows
    # (timer resolution + thread scheduling) and under load, so allow slack. The strict real-clock spacing check is in
    # test_shared_limiter / the arXiv-spacing test.
    slack = 0.03                                    # also needed on a loaded Linux box (seen once)
    assert len(times) == 6 and min(b - a for a, b in zip(times, times[1:])) >= 0.05 - slack


def test_api_key_never_appears_in_error_messages(tmp_path):
    ids, refs = synthetic_citations(5)
    with FakeOpenAlex(refs, faults={1: ("http", 400, None, "bad")}) as srv:
        with pytest.raises(OpenAlexError) as ei:
            fetch_citations(fast_client(srv.url, api_key="TOPSECRETKEY"), ids, tmp_path / "c.jsonl")
    assert "TOPSECRETKEY" not in str(ei.value)
