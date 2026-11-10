"""Honest speed comparison: pure-Python reference vs the C++ accelerator vs SQLite FTS5 (same data, same queries).

    python bench/native_speedup.py --docs 20000 --out docs/evidence/native-speedup-<machine>-SYNTHETIC.json

* Data is SYNTHETIC (tiny vocabulary: long postings lists), so absolute numbers say little about real arXiv text;
  the Python-vs-C++ RATIO on the same data is the meaningful part. Run it on your own machine for your own numbers.
* SQLite FTS5 is a baseline from the Python standard library (needs an sqlite3 built with FTS5). It uses its own
  tokenizer and ranking arithmetic, so its top-10 lists will differ from ours: we compare time only.
* Pyserini/Lucene needs Java and was NOT run here.
"""
import argparse
import json
import os
import platform
import sqlite3
import statistics
import tempfile
import time

import numpy as np

import paperlens.native as native
from paperlens.store.index import IndexStore
from paperlens.synth import generate_papers

QUERIES = ["inverted postings", "crash recovery journal", "pagerank citation graph", "scheduler latency",
           "gradient embedding quantization", "segment compaction merge", "vector search index", "attention transformer"]


def timeit(fn, rounds):
    ts = []
    for _ in range(rounds):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000)
    ts.sort()
    return {"p50_ms": round(statistics.median(ts), 3), "p95_ms": round(ts[int(0.95 * (len(ts) - 1))], 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=20000)
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--out")
    a = ap.parse_args()
    papers = generate_papers(a.docs, 5)
    res = {"data": "SYNTHETIC", "docs": a.docs, "python": platform.python_version(), "platform": platform.platform(),
           "cpu_count": os.cpu_count(), "native": native.status(), "date": time.strftime("%Y-%m-%d")}
    with tempfile.TemporaryDirectory() as d:
        st = IndexStore(os.path.join(d, "ix"), flush_docs=5000)
        for i, p in enumerate(papers):
            st.put(p)
            if i % 5000 == 4999:
                st.commit()
        st.commit(); st.flush()

        def run_all(k):
            for q in QUERIES:
                st.search(q, k=k)

        modes = [("python", False)] + ([("cpp", True)] if native.HAVE_NATIVE else [])
        res["search_all_queries_ms"] = {}
        for name, flag in modes:
            native.set_enabled(flag)
            run_all(10)
            res["search_all_queries_ms"][name] = timeit(lambda: run_all(10), a.rounds)
        if native.HAVE_NATIVE:
            def ranked(flag):
                native.set_enabled(flag)
                return [[(h.key, h.score) for h in st.search(q, k=10)] for q in QUERIES]
            same = ranked(True) == ranked(False)
            native.set_enabled(True)
            res["results_identical_in_this_run"] = bool(same)
            py, cp = (res["search_all_queries_ms"][m]["p50_ms"] for m in ("python", "cpp"))
            res["search_speedup_p50"] = round(py / cp, 2)
        st.close()

        # postings decode micro-benchmark on the biggest list
        from paperlens.store.reader import IndexReader
        rd = IndexReader(os.path.join(d, "ix"))
        seg = rd.segments[0]
        term = max(seg.terms, key=lambda t: seg.terms[t][1])
        data, df = seg._term_bytes(term)
        res["decode_postings"] = {"postings": df}
        for name, flag in modes:
            native.set_enabled(flag)
            res["decode_postings"][name] = timeit(lambda: native.decode_postings(data, 0, df), a.rounds)
        native.set_enabled(True)
        rd.close()

        # quantized vector scan
        from paperlens.vectors import quant
        rng = np.random.default_rng(1)
        dim, n, bits = 384, 50000, 1
        v = rng.normal(size=(n, dim)).astype(np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        enc = quant.encode(v, v.mean(axis=0), bits, quant.rotation_matrix(dim))
        q = rng.normal(size=dim).astype(np.float32)
        res["vector_scan"] = {"vectors": n, "dim": dim, "bits": bits}
        for name, flag in modes:
            native.set_enabled(flag)
            res["vector_scan"][name] = timeit(lambda: native.level_dot(enc["codes"], q, dim, bits), max(3, a.rounds // 3))
        native.set_enabled(True)

        # SQLite FTS5 baseline (own tokenizer/ranking; time only)
        con = sqlite3.connect(os.path.join(d, "fts.db"))
        try:
            con.execute("create virtual table t using fts5(title, abstract)")
            con.executemany("insert into t values (?,?)", ((p.title, p.abstract) for p in papers))
            con.commit()

            def fts_all():
                for q in QUERIES:
                    match = " OR ".join(q.split())
                    con.execute("select rowid from t where t match ? order by bm25(t) limit 10", (match,)).fetchall()
            fts_all()
            res["sqlite_fts5_all_queries_ms"] = timeit(fts_all, a.rounds)
            res["sqlite_version"] = sqlite3.sqlite_version
        except sqlite3.OperationalError as e:
            res["sqlite_fts5_all_queries_ms"] = f"not available: {e}"
    print(json.dumps(res, indent=1))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(res, f, indent=1)
            f.write("\n")


if __name__ == "__main__":
    main()
