"""BEIR-format loader and runner (SciFact, SCIDOCS, or our own judged queries).

What it does and why (plain language)
-------------------------------------
BEIR is a standard set of search-quality benchmarks. A dataset folder holds:

    corpus.jsonl         one {"_id", "title", "text"} per line
    queries.jsonl        one {"_id", "text"} per line
    qrels/test.tsv       tab-separated: query-id <TAB> corpus-id <TAB> score   (first line = header)

`run_beir` loads it, builds a fresh PaperLens index from the corpus, runs
every judged query, and reports nDCG@10, Recall@100, MRR@10, query latency
and index size. Same code path as real use: it goes through IndexStore.put /
commit / flush / search, nothing special-cased.

HONEST NOTES
* BEIR documents have no date, so every paper gets the placeholder datestamp
  2000-01-01 (the idempotency key stays unique because ids are unique).
* Our text = title repeated twice + body (decision D3) and no stemming; the
  published BM25 baselines use Lucene analyzers (stemming, different
  stopwords). Expect numbers in the same range, NOT identical.
* The same code evaluates our own ~50 hand-judged queries: put them in a
  folder with the same three files.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone

from ..contracts.paper import PaperRecord
from ..store.index import IndexStore
from .metrics import evaluate

PLACEHOLDER_DATE = "2000-01-01"


class BeirError(Exception):
    pass


def _read_jsonl(path):
    try:
        f = open(path, "r", encoding="utf-8")
    except FileNotFoundError:
        raise BeirError(f"missing file: {path}") from None
    with f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise BeirError(f"{path}:{n}: bad JSON ({e})") from None


def load_beir(data_dir, split: str = "test"):
    """Returns (corpus {id: (title, text)}, queries {id: text}, qrels {qid: {doc_id: rel}})."""
    data_dir = os.fspath(data_dir)
    corpus, queries, qrels = {}, {}, {}
    for d in _read_jsonl(os.path.join(data_dir, "corpus.jsonl")):
        did = str(d["_id"])
        if "@" in did:
            raise BeirError(f"document id {did!r} contains '@' (reserved in our keys)")
        corpus[did] = (d.get("title") or "", d.get("text") or "")
    for q in _read_jsonl(os.path.join(data_dir, "queries.jsonl")):
        queries[str(q["_id"])] = q.get("text") or ""
    path = os.path.join(data_dir, "qrels", f"{split}.tsv")
    try:
        f = open(path, "r", encoding="utf-8")
    except FileNotFoundError:
        raise BeirError(f"missing judgements file: {path}") from None
    with f:
        for n, line in enumerate(f, 1):
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 3 or not parts[0]:
                continue
            try:
                rel = int(float(parts[2]))
            except ValueError:
                if n == 1:
                    continue                      # header row
                raise BeirError(f"{path}:{n}: bad score {parts[2]!r}") from None
            qrels.setdefault(parts[0], {})[parts[1]] = rel
    if not qrels:
        raise BeirError(f"no judgements found in {path}")
    unknown = {q for q in qrels if q not in queries}
    if unknown:
        raise BeirError(f"{len(unknown)} judged query ids are missing from queries.jsonl, e.g. {sorted(unknown)[:3]}")
    return corpus, queries, qrels


def _pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]


def _dir_bytes(path) -> int:
    return sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(path) for f in fs)


def _evaluate_mode(st, name, queries, qrels, k, search):
    run, lat = {}, []
    for qid in sorted(qrels):
        t = time.perf_counter()
        hits = search(qid, queries[qid])
        lat.append((time.perf_counter() - t) * 1000)
        run[qid] = [h.arxiv_id for h in hits]
    return {"metrics": evaluate(run, qrels),
            "search_ms": {"p50": round(_pct(lat, 50), 2), "p95": round(_pct(lat, 95), 2),
                          "mean": round(sum(lat) / len(lat), 2)}}


def run_beir(data_dir, split: str = "test", index_dir=None, k: int = 100, flush_docs: int = 5000,
             log=print, embedder=None, vec_bits: int = 1, nprobe=None, ivf_min_docs: int = 4096) -> dict:
    """embedder=None: keyword (BM25) only, exactly as in Phase 1. embedder=<vectors.embedders.Embedder>:
    also store vectors and evaluate vector-only and hybrid (RRF) on the SAME index; the result then has
    "modes" with all three, and "nfr6_hybrid_at_least_best_single" (NFR6 check on nDCG@10)."""
    corpus, queries, qrels = load_beir(data_dir, split)
    own_tmp = index_dir is None
    index_dir = index_dir or tempfile.mkdtemp(prefix="paperlens-beir-")
    if not own_tmp and os.path.exists(os.path.join(index_dir, "MANIFEST")):
        raise BeirError(f"{index_dir} already holds an index; give an empty folder (or none for a temp one)")
    try:
        t0 = time.perf_counter()
        with IndexStore(index_dir, flush_docs=flush_docs, vec_bits=vec_bits, ivf_min_docs=ivf_min_docs,
                        embedder_name=embedder.name if embedder else "") as st:
            items = list(corpus.items())
            log(f"loaded {len(corpus)} documents, {len(qrels)} judged queries; indexing{' with embeddings (slow on CPU)' if embedder else ''}...")
            step = 256
            for a in range(0, len(items), step):
                chunk = items[a:a + step]
                vecs = None
                if embedder is not None:                                   # batch: models are much faster that way
                    from ..vectors.embedders import paper_input_text
                    vecs = embedder.embed_texts([paper_input_text(t, x) for _id, (t, x) in chunk])
                for j, (did, (title, text)) in enumerate(chunk):
                    st.put(PaperRecord(arxiv_id=did, datestamp=PLACEHOLDER_DATE, title=title, abstract=text),
                           vector=None if vecs is None else vecs[j])
                st.commit()
                log(f"  embedded+stored {min(a + step, len(items))}/{len(items)} documents, "
                    f"{time.perf_counter() - t0:.0f} s elapsed")
            st.flush()
            build_s = time.perf_counter() - t0
            log(f"indexed {len(corpus)} documents in {build_s:.1f} s")
            modes = {"bm25": _evaluate_mode(st, "bm25", queries, qrels, k,
                                            lambda qid, text: st.search(text, k=k))}
            if embedder is not None:
                qids = sorted(qrels)
                qv = dict(zip(qids, embedder.embed_texts([queries[q] for q in qids])))
                log("evaluating vector-only and hybrid...")
                modes["vector"] = _evaluate_mode(st, "vector", queries, qrels, k,
                                                 lambda qid, text: st.search_vector(qv[qid], k=k, nprobe=nprobe))
                modes["hybrid"] = _evaluate_mode(st, "hybrid", queries, qrels, k,
                                                 lambda qid, text: st.search_hybrid(text, qv[qid], k=k, nprobe=nprobe))
            index_bytes = _dir_bytes(index_dir)
        metrics, lat_summary = modes["bm25"]["metrics"], modes["bm25"]["search_ms"]
    finally:
        if own_tmp:
            shutil.rmtree(index_dir, ignore_errors=True)
    commit = ""
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
    except OSError:
        pass
    now = datetime.now(timezone.utc)
    return {
        "recorded_utc": f"{now:%Y-%m-%dT%H:%M:%SZ}", "git_commit": commit or None,
        "dataset": os.path.basename(os.path.abspath(os.fspath(data_dir))), "split": split,
        "documents": len(corpus), "k": k, "metrics": metrics,
        "build_seconds": round(build_s, 2), "index_bytes": index_bytes,
        "search_ms": lat_summary,
        "modes": modes,
        "embedder": embedder.name if embedder else None, "vec_bits": vec_bits if embedder else None,
        "nfr6_hybrid_at_least_best_single": (
            modes["hybrid"]["metrics"]["ndcg@10"] >= max(modes["bm25"]["metrics"]["ndcg@10"],
                                                         modes["vector"]["metrics"]["ndcg@10"]) - 1e-12
            if embedder else None),
        "python": platform.python_version(), "platform": platform.platform(), "cpu_count": os.cpu_count(),
        "notes": "BM25 k1=1.2 b=0.75, no stemming, title counted twice (D3); see eval/beir.py"
                 + ("; vector/hybrid use embedder " + embedder.name if embedder else ""),
    }
