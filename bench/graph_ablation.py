"""Does citation reranking earn its place? (brief: every extra ranking stage must show a measured gain or be switched off)

    python bench/graph_ablation.py --synthetic --out docs/evidence/graph-ablation-SYNTHETIC.json
    python bench/graph_ablation.py --index data/ix --queries data/my_queries.jsonl --out docs/evidence/graph-ablation-real.json

--queries is JSONL, one object per line: {"query": "text", "relevant": ["2101.00001", ...]}  (arXiv ids you judged relevant).
The index must already have a published citation graph (`citations` + `graph-build`).

Measured: nDCG@10, Recall@100 and MRR@10 of the base ranking (hybrid if the index has vectors and --embedder is given,
else BM25) versus the same ranking after `citation_rerank`, over a small grid of (seeds, weight). The verdict is
"keep" ONLY if the best setting beats the base on nDCG@10 by more than --min-gain (default 0.005); otherwise "switch off".

HONESTY: --synthetic plants citations inside synthetic topics, so it can only show the plumbing works and what a
helpful graph would do. It says nothing about real arXiv. The real verdict needs your judged queries + OpenAlex graph.
"""
import argparse
import json
import os
import platform
import random
import sys
import tempfile
import time

from paperlens.eval.metrics import evaluate
from paperlens.graph.csr import Graph
from paperlens.graph.rerank import citation_rerank
from paperlens.store.index import IndexStore
from paperlens.synth import TOPICS, generate_papers, topic_of

DEPTH = 100


def run_queries(st, queries, embedder, seeds_weights):
    runs = {"base": {}}
    for sw in seeds_weights:
        runs[f"graph(seeds={sw[0]},weight={sw[1]})"] = {}
    for qid, text in queries.items():
        if embedder is not None:
            hits = st.search_hybrid(text, embedder.embed_query(text), k=DEPTH)
        else:
            hits = st.search(text, k=DEPTH)
        ids = [h.arxiv_id for h in hits]
        runs["base"][qid] = ids
        for seeds, weight in seeds_weights:
            fused = citation_rerank(ids, st.graph, seeds=seeds, weight=weight)
            runs[f"graph(seeds={seeds},weight={weight})"][qid] = [a for a, *_ in fused]
    return runs


def build_synthetic(tmp):
    papers = generate_papers(400, seed=11)
    rng = random.Random(11)
    by_topic = {}
    for p in papers:
        by_topic.setdefault(topic_of(p), []).append(p.arxiv_id)
    edges = []
    for p in papers:
        t = topic_of(p)
        for _ in range(5):
            pool = by_topic[t] if rng.random() < 0.8 else [x.arxiv_id for x in papers]
            edges.append((p.arxiv_id, rng.choice(pool)))
    st = IndexStore(tmp, flush_docs=100)
    for p in papers:
        st.put(p)
    st.commit()
    st.publish_graph(Graph.from_edges([p.arxiv_id for p in papers], edges))
    queries, qrels = {}, {}
    for t, words in TOPICS.items():
        qid = f"q-{t}"
        queries[qid] = " ".join(words[:2])                       # two topic words: a weak, realistic query
        qrels[qid] = {a: 1 for a in by_topic.get(t, [])}
    return st, queries, qrels


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index")
    ap.add_argument("--queries")
    ap.add_argument("--embedder")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--min-gain", type=float, default=0.005)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    grid = [(s, w) for s in (3, 5, 10) for w in (0.5, 1.0, 2.0)]
    tmpdir = None
    embedder = None
    if a.synthetic:
        tmpdir = tempfile.TemporaryDirectory()
        st, queries, qrels = build_synthetic(os.path.join(tmpdir.name, "ix"))
        data = "SYNTHETIC (plumbing only)"
    else:
        if not (a.index and a.queries):
            ap.error("give --index and --queries, or --synthetic")
        st = IndexStore(a.index)
        if st.graph is None:
            sys.exit("error: this index has no citation graph (run `citations` and `graph-build`)")
        queries, qrels = {}, {}
        with open(a.queries, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if line.strip():
                    o = json.loads(line)
                    queries[f"q{i}"], qrels[f"q{i}"] = o["query"], {x: 1 for x in o["relevant"]}
        if a.embedder:
            from paperlens.vectors.embedders import make_embedder
            embedder = make_embedder(a.embedder)
        data = f"real: {a.queries}"
    with st:
        t0 = time.perf_counter()
        runs = run_queries(st, queries, embedder, grid)
        elapsed = time.perf_counter() - t0
    results = {name: evaluate(run, qrels) for name, run in runs.items()}
    base = results["base"]["ndcg@10"]
    best_name = max((n for n in results if n != "base"), key=lambda n: results[n]["ndcg@10"])
    gain = results[best_name]["ndcg@10"] - base
    out = {"data": data, "queries": len(queries), "base": results["base"], "settings": {n: r for n, r in results.items() if n != "base"},
           "best_setting": best_name, "best_ndcg10_gain": round(gain, 4),
           "verdict": "keep" if gain > a.min_gain else "switch off",
           "min_gain_required": a.min_gain, "seconds": round(elapsed, 2),
           "python": platform.python_version(), "platform": platform.platform(), "date": time.strftime("%Y-%m-%d")}
    print(json.dumps(out, indent=1))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(out, f, indent=1)
            f.write("\n")
    if tmpdir:
        tmpdir.cleanup()


if __name__ == "__main__":
    main()
