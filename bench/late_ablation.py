"""Does late-interaction reranking earn its place? Same rule as graph_ablation.py: keep only on a measured gain.

    python bench/late_ablation.py --synthetic --out docs/evidence/late-ablation-SYNTHETIC.json
    python bench/late_ablation.py --index data/ix --queries data/my_queries.jsonl --out docs/evidence/late-ablation-real.json

The token encoder is the hashed-trigram STAND-IN (see vectors/late.py), so even a real-data run only measures that
stand-in, not ColBERT. Synthetic data is already at the ceiling for the base ranking, so expect "switch off".
"""
import argparse
import json
import os
import platform
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from graph_ablation import build_synthetic                      # noqa: E402

from paperlens.eval.metrics import evaluate                     # noqa: E402
from paperlens.store.index import IndexStore                    # noqa: E402
from paperlens.vectors.late import late_rerank                  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index")
    ap.add_argument("--queries")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--min-gain", type=float, default=0.005)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    tmp = None
    if a.synthetic:
        tmp = tempfile.TemporaryDirectory()
        st, queries, qrels = build_synthetic(os.path.join(tmp.name, "ix"))
        data = "SYNTHETIC (plumbing only)"
    else:
        if not (a.index and a.queries):
            ap.error("give --index and --queries, or --synthetic")
        st = IndexStore(a.index)
        queries, qrels = {}, {}
        with open(a.queries, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if line.strip():
                    o = json.loads(line)
                    queries[f"q{i}"], qrels[f"q{i}"] = o["query"], {x: 1 for x in o["relevant"]}
        data = f"real: {a.queries}"
    grid = [(d, w, c) for d in (30, 100) for w in (0.5, 1.0, 2.0) for c in (True,)]
    runs = {"base": {}}
    with st:
        for qid, text in queries.items():
            hits = st.search(text, k=100)
            runs["base"][qid] = [h.arxiv_id for h in hits]
            for d, w, c in grid:
                name = f"late(depth={d},weight={w},int8={c})"
                runs.setdefault(name, {})[qid] = [x for x, *_ in late_rerank(st, text, hits, depth=d, weight=w, compress=c)]
    results = {n: evaluate(r, qrels) for n, r in runs.items()}
    base = results["base"]["ndcg@10"]
    best = max((n for n in results if n != "base"), key=lambda n: results[n]["ndcg@10"])
    gain = results[best]["ndcg@10"] - base
    out = {"data": data, "queries": len(queries), "token_encoder": "hash-trigram STAND-IN", "base": results["base"],
           "settings": {n: r for n, r in results.items() if n != "base"}, "best_setting": best,
           "best_ndcg10_gain": round(gain, 4), "verdict": "keep" if gain > a.min_gain else "switch off",
           "python": platform.python_version(), "date": time.strftime("%Y-%m-%d")}
    print(json.dumps(out, indent=1))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(out, f, indent=1)
            f.write("\n")
    if tmp:
        tmp.cleanup()


if __name__ == "__main__":
    main()
