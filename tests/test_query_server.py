import os
import threading
import time

import pytest

from paperlens.server.query_server import QueryServer
from paperlens.store.fsck import fsck
from paperlens.store.filters import Filter
from paperlens.store.index import IndexStore
from paperlens.store.reader import IndexReader
from paperlens.synth import generate_papers
from paperlens.vectors.embedders import HashEmbedder

QUERIES = ["inverted postings", "crash recovery journal", "pagerank citation graph", "scheduler latency"]


def make_index(tmp_path, n=80, vectors=False, flush=20):
    st = IndexStore(tmp_path / "ix", flush_docs=10**9, ivf_min_docs=0, embedder_name="hash:16" if vectors else "")
    emb = HashEmbedder(16)
    for i, p in enumerate(generate_papers(n, 1)):
        st.put(p, vector=emb.embed_paper(p) if vectors else None)
        if i % flush == flush - 1:
            st.commit()
            st.flush()
    st.commit()
    st.flush()
    return st


def test_server_answers_equal_an_in_process_reader_for_every_mode_and_filter(tmp_path):
    st = make_index(tmp_path, vectors=True)
    st.close()
    emb = HashEmbedder(16)
    with IndexReader(tmp_path / "ix") as rd, QueryServer(tmp_path / "ix", workers=2) as qs:
        assert rd.embedder_name == "hash:16"
        for q in QUERIES:
            got = qs.search(q, k=7)
            assert [h["arxiv_id"] for h in got["hits"]] == [h.arxiv_id for h in rd.search(q, k=7)]
            assert got["generation"] == rd.generation and got["hits"][0]["title"]
            hy = qs.search(q, k=7, mode="hybrid")
            want = rd.search_hybrid(q, emb.embed_query(q), k=7)
            assert [h["arxiv_id"] for h in hy["hits"]] == [h.arxiv_id for h in want]
            assert [h["bm25_rank"] for h in hy["hits"]] == [h.bm25_rank for h in want]
            vc = qs.search(q, k=5, mode="vector")
            assert [h["arxiv_id"] for h in vc["hits"]] == [h.arxiv_id for h in rd.search_vector(emb.embed_query(q), k=5)]
        cats = rd.get(sorted(rd.live)[0]).categories[:1]
        got = qs.search("inverted postings", k=20, cat=cats)
        want = rd.search("inverted postings", k=20, flt=Filter(categories=tuple(cats)))
        assert [h["arxiv_id"] for h in got["hits"]] == [h.arxiv_id for h in want] and got["hits"]
        assert all(set(h["categories"]) & set(cats) for h in got["hits"])


def test_a_killed_query_worker_is_replaced_and_the_request_repeated(tmp_path):
    make_index(tmp_path).close()
    with QueryServer(tmp_path / "ix", workers=2) as qs:
        base = qs.search("inverted postings", k=5)["hits"]
        os.kill(qs.pids()[0], 9)
        os.kill(qs.pids()[1], 9)
        time.sleep(0.1)
        for _ in range(6):
            assert qs.search("inverted postings", k=5)["hits"] == base
        assert qs.respawns >= 2
        assert qs.stats()["live_docs"] == 80
    from paperlens.store.pins import pinned_segments
    assert pinned_segments(tmp_path / "ix") == set()                             # killed workers' pins are stale ...
    assert os.listdir(tmp_path / "ix" / "pins") == []                            # ... and get cleaned up


def test_workers_pick_up_new_generations_by_themselves(tmp_path):
    st = make_index(tmp_path, n=40)
    with QueryServer(tmp_path / "ix", workers=1, refresh_interval=0.0) as qs:
        assert qs.stats()["live_docs"] == 40
        for p in generate_papers(80, 5)[40:60]:
            st.put(p)
        st.commit()
        st.flush()
        s = qs.stats()
        assert s["generation"] == st.manifest.generation and s["live_docs"] == st.stats()["live_docs"]
    st.close()


def test_queries_during_ingestion_and_compaction_always_see_one_consistent_snapshot(tmp_path):
    """NFR5. The writer flushes every 15 papers and a background thread merges segments while two reader
    processes are queried in a loop. Papers are unique, so in ANY consistent snapshot the number of live
    papers must equal the last flushed sequence number; a torn view (half a compaction, a segment missing
    or listed twice) breaks that equality or makes a search fail."""
    st = IndexStore(tmp_path / "ix", flush_docs=10**9)
    st.start_background_compaction(interval=0.01, fan_in=3, ratio=50)
    papers = generate_papers(450, 12)
    violations, seen_generations = [], {}
    stop = threading.Event()
    with QueryServer(tmp_path / "ix", workers=2, refresh_interval=0.0) as qs:
        def hammer():
            while not stop.is_set():
                s = qs.stats()
                if s["live_docs"] != s["last_flushed_seq"]:
                    violations.append(("count", s["generation"], s["live_docs"], s["last_flushed_seq"]))
                if s["physical_docs"] < s["live_docs"]:
                    violations.append(("physical", s))
                prev = seen_generations.get(s["worker"], 0)
                if s["generation"] < prev:
                    violations.append(("generation went backwards", s["worker"], prev, s["generation"]))
                seen_generations[s["worker"]] = max(prev, s["generation"])
                r = qs.search("inverted postings recovery", k=5)
                ids = [h["arxiv_id"] for h in r["hits"]]
                if len(ids) != len(set(ids)):
                    violations.append(("duplicate hit", ids))
        threads = [threading.Thread(target=hammer) for _ in range(2)]
        for t in threads:
            t.start()
        for i, p in enumerate(papers):
            st.put(p)
            if i % 15 == 14:
                st.commit()
                st.flush()
                time.sleep(0.01)
        st.commit()
        st.flush()
        deadline = time.time() + 20
        while st._bg.runs < 3 and time.time() < deadline:
            time.sleep(0.05)
        stop.set()
        for t in threads:
            t.join()
        final = qs.stats()
    st.stop_background_compaction()
    assert st._bg is None
    assert not violations, violations[:5]
    assert len(seen_generations) == 2 and max(seen_generations.values()) > 5     # both workers saw many generations
    assert final["live_docs"] == 450
    st.close()
    assert fsck(tmp_path / "ix").ok
    assert os.listdir(tmp_path / "ix" / "pins") == []


def test_cli_compact_ingest_with_background_compaction_and_serve_query(tmp_path):
    import subprocess, sys
    from paperlens.contracts.paper import write_jsonl
    src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
    env = dict(os.environ, PYTHONPATH=src)
    def cli(*a, inp=None):
        return subprocess.run([sys.executable, "-m", "paperlens", *a], env=env, capture_output=True, text=True,
                              input=inp, timeout=180)
    data = tmp_path / "p.jsonl"
    write_jsonl(data, generate_papers(200, 2))
    ix = str(tmp_path / "ix")
    r = cli("ingest", "--index", ix, "--input", str(data), "--readers", "2", "--flush-docs", "20", "--compact")
    assert r.returncode == 0, r.stdout + r.stderr
    r = cli("compact", "--index", ix, "--force")
    assert r.returncode == 0 and "live 200" in r.stdout, r.stdout + r.stderr
    r = cli("serve-query", "--index", ix, "--workers", "1", "-k", "3", inp="inverted postings\n\n")
    assert r.returncode == 0 and "generation" in r.stdout and r.stdout.count(". ") >= 3, r.stdout + r.stderr
    assert cli("fsck", "--index", ix).returncode == 0
