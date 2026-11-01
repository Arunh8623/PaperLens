"""Quick profile of the Python reference index (NOT a benchmark of search quality).

    python scripts/profile_quick.py [N]      # default N = 20000 synthetic papers

Measures: ingest throughput (put + group commit + flush), search latency
(p50/p95 over random 1-3 word queries), and the top cProfile functions during
search. Writes docs/evidence/profile-<date>-<commit>.json.

HONEST LIMITS: synthetic text has ~60 distinct words, so postings lists are
extremely long compared with real arXiv text. Absolute numbers say nothing about
real performance; they only show WHICH CODE dominates. Re-run on real harvested
data and on your own machine before choosing a C++ target.
"""
import cProfile
import json
import os
import platform
import pstats
import random
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from paperlens.store.index import IndexStore          # noqa: E402
from paperlens.synth import COMMON, TOPICS, generate_papers  # noqa: E402


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]


def main(n: int) -> None:
    words = sorted({w for ws in TOPICS.values() for w in ws} | set(COMMON))
    papers = generate_papers(n, seed=1)
    td = tempfile.mkdtemp()
    try:
        t0 = time.perf_counter()
        with IndexStore(os.path.join(td, "ix"), flush_docs=5000) as st:
            for i, p in enumerate(papers):
                st.put(p)
                if i % 64 == 63:
                    st.commit()
            st.commit()
            st.flush()
            ingest_s = time.perf_counter() - t0
            stats = {k: v for k, v in st.stats().items() if k != "recovery"}
            rng = random.Random(2)
            queries = [" ".join(rng.sample(words, rng.randint(1, 3))) for _ in range(200)]
            lat = []
            for q in queries:
                t = time.perf_counter()
                st.search(q, k=10)
                lat.append((time.perf_counter() - t) * 1000)
            prof = cProfile.Profile()
            prof.enable()
            for q in queries[:60]:
                st.search(q, k=10)
            prof.disable()
        ps = pstats.Stats(prof)
        total = ps.total_tt or 1e-9
        top = sorted(ps.stats.items(), key=lambda kv: -kv[1][2])[:6]
        top_rows = [{"function": f"{os.path.basename(k[0])}:{k[2]}", "self_seconds": round(v[2], 4),
                     "share_pct": round(100 * v[2] / total, 1)} for k, v in top]
        ix_bytes = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(os.path.join(td, "ix"))
                       for f in fs)
    finally:
        shutil.rmtree(td, ignore_errors=True)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    now = datetime.now(timezone.utc)
    doc = {
        "recorded_utc": f"{now:%Y-%m-%dT%H:%M:%SZ}", "git_commit": commit,
        "python": platform.python_version(), "platform": platform.platform(), "cpu_count": os.cpu_count(),
        "WARNING": "synthetic data, tiny vocabulary: shows which code dominates, NOT real performance",
        "papers": n, "ingest_seconds": round(ingest_s, 2), "ingest_papers_per_second": round(n / ingest_s),
        "index_bytes": ix_bytes, "index": stats, "queries": len(queries),
        "search_ms": {"p50": round(pct(lat, 50), 2), "p95": round(pct(lat, 95), 2), "max": round(max(lat), 2)},
        "search_profile_top_self_time": top_rows,
    }
    out = os.path.join(ROOT, "docs", "evidence", f"profile-{now:%Y%m%d}-{commit}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    print(json.dumps(doc, indent=1))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20000)
