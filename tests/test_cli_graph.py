import json
import os
import random
import subprocess
import sys

from paperlens.contracts.paper import write_jsonl
from paperlens.graph.fake_openalex import FakeOpenAlex
from paperlens.synth import generate_papers

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))


def cli(*args, env_extra=None):
    env = dict(os.environ, PYTHONPATH=SRC, **(env_extra or {}))
    return subprocess.run([sys.executable, "-m", "paperlens", *args], env=env,
                          capture_output=True, text=True, timeout=120)


def test_citations_graph_build_stats_search_rerank_and_dupes(tmp_path):
    papers = generate_papers(60, seed=4)
    # plant one near-duplicate: same text, new id
    dup = papers[3].__class__.from_dict({**papers[3].to_dict(), "arxiv_id": "2199.99999"})
    papers = papers + [dup]
    data, ix, cites = tmp_path / "p.jsonl", str(tmp_path / "ix"), str(tmp_path / "c.jsonl")
    write_jsonl(data, papers)
    assert cli("index", "--index", ix, "--input", str(data), "--flush-docs", "30").returncode == 0

    rng = random.Random(1)
    ids = [p.arxiv_id for p in papers]
    refs = {a: rng.sample(ids, 4) for a in ids}
    with FakeOpenAlex(refs) as srv:
        r = cli("citations", "--index", ix, "--out", cites, "--base-url", srv.url, "--batch", "25",
                "--contact", "someone@example.org", "--api-key", "SECRETKEY123")
        assert r.returncode == 0, r.stderr
        assert "SECRETKEY123" not in r.stdout + r.stderr
        assert srv.count == 3                                    # 61 papers / 25 per request
        r = cli("citations", "--index", ix, "--out", cites, "--base-url", srv.url)   # resume: nothing left to fetch
        assert r.returncode == 0 and srv.count == 3

    assert cli("search", "--index", ix, "--graph-rerank", "graph").returncode == 1      # no graph yet: clear error
    r = cli("graph-build", "--index", ix, "--citations", cites)
    assert r.returncode == 0 and "61 papers" in r.stdout, r.stdout + r.stderr
    r = cli("graph-stats", "--index", ix, "--top", "3")
    assert r.returncode == 0 and "papers 61" in r.stdout and r.stdout.count("cited") == 3
    r = cli("search", "--index", ix, "--graph-rerank", "-k", "8", "pagerank graph")
    assert r.returncode == 0 and "graph: was #" in r.stdout, r.stdout + r.stderr
    r = cli("dupes", "--index", ix)
    assert r.returncode == 0 and "2199.99999" in r.stdout and papers[3].arxiv_id in r.stdout
    assert cli("fsck", "--index", ix).returncode == 0


def test_citations_stops_cleanly_when_budget_is_zero(tmp_path):
    data, ix = tmp_path / "p.jsonl", str(tmp_path / "ix")
    write_jsonl(data, generate_papers(10, seed=1))
    assert cli("index", "--index", ix, "--input", str(data)).returncode == 0
    with FakeOpenAlex({}) as srv:
        r = cli("citations", "--index", ix, "--out", str(tmp_path / "c.jsonl"), "--base-url", srv.url,
                "--max-spend-usd", "0")
        assert r.returncode == 2 and "stopped" in r.stderr and srv.count == 0


def test_ablation_script_runs_and_applies_its_decision_rule(tmp_path):
    out = tmp_path / "ab.json"
    env = dict(os.environ, PYTHONPATH=SRC)
    r = subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "..", "bench", "graph_ablation.py"),
                        "--synthetic", "--out", str(out)], env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    d = json.loads(out.read_text())
    assert d["data"].startswith("SYNTHETIC") and len(d["settings"]) == 9
    # synthetic base is already perfect, so the rule must say "switch off" (no measured gain)
    assert d["base"]["ndcg@10"] == 1.0 and d["verdict"] == "switch off"


def test_similar_key_papers_and_late_commands(tmp_path):
    papers = generate_papers(50, seed=2)
    data, ix, cites = tmp_path / "p.jsonl", str(tmp_path / "ix"), str(tmp_path / "c.jsonl")
    write_jsonl(data, papers)
    assert cli("index", "--index", ix, "--input", str(data), "--embedder", "hash:32").returncode == 0
    r = cli("similar", "--index", ix, papers[5].arxiv_id, "-k", "4")
    assert r.returncode == 0 and r.stdout.count("(by meaning vector)") == 4 and r.stdout.count(papers[5].arxiv_id) == 1, r.stdout + r.stderr
    assert cli("similar", "--index", ix, "0000.00000").returncode == 1
    assert cli("key-papers", "--index", ix, "graph").returncode == 1                    # no graph yet
    rng = random.Random(1)
    ids = [p.arxiv_id for p in papers]
    with FakeOpenAlex({a: rng.sample(ids, 3) for a in ids}) as srv:
        assert cli("citations", "--index", ix, "--out", cites, "--base-url", srv.url).returncode == 0
    assert cli("graph-build", "--index", ix, "--citations", cites).returncode == 0
    r = cli("key-papers", "--index", ix, "-k", "3", "inverted postings")
    assert r.returncode == 0 and r.stdout.count("PageRank") == 3, r.stdout + r.stderr
    r = cli("search", "--index", ix, "--late", "-k", "5", "inverted postings")
    assert r.returncode == 0 and len([l for l in r.stdout.splitlines() if l[:3].strip().rstrip(".").isdigit()]) == 5, r.stdout + r.stderr
