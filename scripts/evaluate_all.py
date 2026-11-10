"""ONE command that reproduces the results table (NFR9, FR10).

    # real data (what the reviewers should see): BEIR SciFact and SCIDOCS you downloaded
    python scripts/evaluate_all.py --beir-dir data/beir/scifact --beir-dir data/beir/scidocs ^
           --embedder st:sentence-transformers/all-MiniLM-L6-v2 --out docs/evidence/final-mypc

    # plumbing check without any download (SYNTHETIC data, numbers are NOT quality evidence)
    python scripts/evaluate_all.py --synthetic --out docs/evidence/final-sandbox-SYNTHETIC

For each dataset it measures, on the same corpus and queries:
  * ours: BM25-only, vector-only, hybrid (RRF)  (nDCG@10, Recall@100, MRR, latency p50/p95, index size, build time)
  * baseline SQLite FTS5 (stdlib; its own tokenizer, OR query, bm25())
  * baseline FAISS IndexFlatIP exact search over the SAME embeddings (if an embedder is given and faiss is installed)
  * Lucene via Pyserini is NOT run here (needs Java); the table says so.
Writes <out>.json and <out>.md. Nothing in the table is typed by hand.
"""
import argparse
import json
import os
import platform
import shutil
import sqlite3
import sys
import tempfile
import time

import numpy as np

from paperlens.eval.beir import _pct, load_beir, run_beir
from paperlens.eval.metrics import evaluate
from paperlens.vectors.embedders import make_embedder, paper_input_text


def fts5_baseline(corpus, queries, qrels, k=100):
    from paperlens.text.tokenizer import tokenize
    d = tempfile.mkdtemp(prefix="pl-fts-")
    try:
        con = sqlite3.connect(os.path.join(d, "f.db"))
        con.execute("create virtual table t using fts5(did unindexed, title, text)")
        con.executemany("insert into t values (?,?,?)", ((i, t, x) for i, (t, x) in corpus.items()))
        con.commit()
        run, lat = {}, []
        for qid in sorted(qrels):
            words = [w for w in dict.fromkeys(tokenize(queries[qid])) if w.isalnum()]
            if not words:
                run[qid] = []
                continue
            t0 = time.perf_counter()
            rows = con.execute("select did from t where t match ? order by bm25(t) limit ?",
                               (" OR ".join(f'"{w}"' for w in words), k)).fetchall()
            lat.append((time.perf_counter() - t0) * 1000)
            run[qid] = [r[0] for r in rows]
        size = os.path.getsize(os.path.join(d, "f.db"))
        con.close()
        return {"metrics": evaluate(run, qrels), "search_ms": {"p50": round(_pct(lat, 50), 2), "p95": round(_pct(lat, 95), 2)},
                "index_bytes": size, "note": "SQLite %s FTS5, own tokenizer, OR query, bm25()" % sqlite3.sqlite_version}
    finally:
        shutil.rmtree(d, ignore_errors=True)


def faiss_baseline(corpus, queries, qrels, embedder, k=100):
    try:
        import faiss
    except ImportError:
        return {"skipped": "faiss not installed (python -m pip install faiss-cpu)"}
    ids = list(corpus)
    vecs = np.concatenate([embedder.embed_texts([paper_input_text(*corpus[i]) for i in ids[a:a + 256]])
                           for a in range(0, len(ids), 256)]).astype(np.float32)
    vecs /= np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    qids = sorted(qrels)
    qv = embedder.embed_texts([queries[q] for q in qids]).astype(np.float32)
    qv /= np.maximum(np.linalg.norm(qv, axis=1, keepdims=True), 1e-12)
    run, lat = {}, []
    for qid, v in zip(qids, qv):
        t0 = time.perf_counter()
        _, I = index.search(v[None, :], min(k, len(ids)))
        lat.append((time.perf_counter() - t0) * 1000)
        run[qid] = [ids[i] for i in I[0] if i >= 0]
    return {"metrics": evaluate(run, qrels), "search_ms": {"p50": round(_pct(lat, 50), 2), "p95": round(_pct(lat, 95), 2)},
            "index_bytes": int(vecs.nbytes), "note": "faiss %s IndexFlatIP, exact float32, same embeddings" % faiss.__version__}


def make_synthetic_beir(root, n=2000, seed=21):
    from paperlens.synth import TOPICS, generate_papers, topic_of
    papers = generate_papers(n, seed)
    os.makedirs(os.path.join(root, "qrels"), exist_ok=True)
    with open(os.path.join(root, "corpus.jsonl"), "w", encoding="utf-8") as f:
        for p in papers:
            f.write(json.dumps({"_id": p.arxiv_id.replace("@", "_"), "title": p.title, "text": p.abstract}) + "\n")
    by_topic = {}
    for p in papers:
        by_topic.setdefault(topic_of(p), []).append(p.arxiv_id)
    with open(os.path.join(root, "queries.jsonl"), "w", encoding="utf-8") as fq, \
            open(os.path.join(root, "qrels", "test.tsv"), "w", encoding="utf-8") as fr:
        fr.write("query-id\tcorpus-id\tscore\n")
        for t, words in TOPICS.items():
            fq.write(json.dumps({"_id": f"q-{t}", "text": " ".join(words[:3])}) + "\n")
            for a in by_topic.get(t, [])[:50]:
                fr.write(f"q-{t}\t{a}\t1\n")
    return root


def fmt(x):
    return f"{x:.4f}" if isinstance(x, float) else str(x)


def table(results):
    rows = ["| dataset | system | nDCG@10 | Recall@100 | MRR@10 | p50 ms | p95 ms | index MB |", "|---|---|---|---|---|---|---|---|"]
    for r in results:
        ds = f"{r['dataset']} ({r['documents']} docs)"
        for name, m in r["systems"].items():
            if "metrics" not in m:
                rows.append(f"| {ds} | {name} | not run: {m.get('skipped', '')} | | | | | |")
                continue
            mm = m["metrics"]
            rows.append(f"| {ds} | {name} | {fmt(mm['ndcg@10'])} | {fmt(mm['recall@100'])} | {fmt(mm['mrr@10'])} | "
                        f"{m['search_ms']['p50']} | {m['search_ms']['p95']} | {m.get('index_bytes', 0) / 1e6:.2f} |")
    rows.append("")
    rows.append("Lucene via Pyserini: NOT run (needs Java).")
    return "\n".join(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--beir-dir", action="append", default=[])
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--embedder", help="e.g. st:sentence-transformers/all-MiniLM-L6-v2 (real) or hash:64 (stand-in)")
    ap.add_argument("--bits", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    tmp = None
    dirs = list(a.beir_dir)
    if a.synthetic:
        tmp = tempfile.mkdtemp(prefix="pl-synth-")
        dirs.append(make_synthetic_beir(os.path.join(tmp, "synthetic-topics")))
        a.embedder = a.embedder or "hash:64"
    if not dirs:
        ap.error("give --beir-dir (one or more) or --synthetic")
    emb = make_embedder(a.embedder) if a.embedder else None
    results = []
    for d in dirs:
        print(f"== {d}", flush=True)
        ours = run_beir(d, embedder=emb, vec_bits=a.bits, log=lambda m: print("  " + m, flush=True))
        corpus, queries, qrels = load_beir(d)
        systems = {f"PaperLens {m}": {**v, "index_bytes": ours["index_bytes"]} for m, v in ours["modes"].items()}
        systems["SQLite FTS5"] = fts5_baseline(corpus, queries, qrels)
        if emb is not None:
            systems["FAISS flat (exact)"] = faiss_baseline(corpus, queries, qrels, emb)
        results.append({"dataset": ours["dataset"], "documents": ours["documents"], "embedder": ours["embedder"],
                        "nfr6_hybrid_at_least_best_single": ours["nfr6_hybrid_at_least_best_single"],
                        "build_seconds": ours["build_seconds"], "systems": systems})
    out = {"recorded": time.strftime("%Y-%m-%d %H:%M:%S"), "python": platform.python_version(), "platform": platform.platform(),
           "cpu_count": os.cpu_count(), "synthetic": a.synthetic, "embedder": a.embedder, "bits": a.bits, "results": results}
    try:
        import paperlens.native as native
        out["native_extension"] = native.status()["enabled"]
    except Exception:
        pass
    md = ("# Results table" + (" (SYNTHETIC DATA: plumbing check only, not quality evidence)" if a.synthetic else "") + "\n\n"
          + table(results) + "\n")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out + ".json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    with open(a.out + ".md", "w", encoding="utf-8", newline="\n") as f:
        f.write(md)
    print(md)
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
