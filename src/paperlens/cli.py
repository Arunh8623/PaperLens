"""PaperLens command line (Phase 0: synth, harvest, stats).

    python -m paperlens synth   --n 1000 --seed 1 --out data/synth.jsonl
    python -m paperlens harvest --contact you@college.edu --out data/arxiv_cs.jsonl --max-records 10000
    python -m paperlens stats   data/arxiv_cs.jsonl

`harvest` refuses any --min-interval below arXiv's 3 s rule: compliance is
enforced by construction, not left to whoever runs it.
"""
from __future__ import annotations

import argparse
import os
import sys

from .harvest.arxiv_oai import (
    ARXIV_MIN_INTERVAL, ARXIV_RULE_SECONDS, DEFAULT_BASE_URL, HarvestError,
    OAIClient, harvest, summarize_jsonl,
)
from .harvest.limiter import MinIntervalLimiter


def _cmd_synth(a) -> int:
    from .contracts.paper import write_jsonl
    from .synth import generate_papers
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    n = write_jsonl(a.out, generate_papers(a.n, a.seed))
    print(f"wrote {n} synthetic papers to {a.out} (seed {a.seed})")
    return 0


def _cmd_harvest(a) -> int:
    contact = a.contact or os.environ.get("PAPERLENS_CONTACT", "")
    if not contact:
        print("error: give your contact email with --contact (or PAPERLENS_CONTACT); "
              "it goes in the User-Agent so arXiv can reach you.", file=sys.stderr)
        return 2
    if a.min_interval < ARXIV_RULE_SECONDS:
        print(f"error: --min-interval must be >= {ARXIV_RULE_SECONDS} s (arXiv rule: "
              "1 request per 3 s, one connection, all your machines combined).", file=sys.stderr)
        return 2
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    state_path = a.state or a.out + ".state.json"
    client = OAIClient(contact, base_url=a.base_url, limiter=MinIntervalLimiter(a.min_interval))
    try:
        state = harvest(client, set_spec=a.set, out_path=a.out, state_path=state_path,
                        from_date=a.from_date, max_records=a.max_records)
    except HarvestError as e:
        print(f"harvest stopped: {e}\n(re-run the same command to resume)", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted - re-run the same command to resume", file=sys.stderr)
        return 130
    print("state:", {k: state[k] for k in ("done", "pages", "lines_written", "restarts",
                                          "skipped_bad", "deleted_seen", "out_of_order",
                                          "newest_datestamp")})
    print("file :", summarize_jsonl(a.out))
    return 0


def _cmd_stats(a) -> int:
    print(summarize_jsonl(a.path))
    return 0


def _cmd_index(a) -> int:
    """JSONL -> index. Commits every --batch papers; flushes at the end."""
    from collections import Counter
    from .contracts.paper import read_jsonl
    from .store.index import IndexStore
    from .store.lock import IndexLocked
    outcomes = Counter()
    try:
        st = IndexStore(a.index, flush_docs=a.flush_docs)
    except IndexLocked as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    with st:
        n = 0
        for p in read_jsonl(a.input, skip_bad=a.skip_bad):
            outcomes[st.put(p).outcome] += 1
            n += 1
            if n % a.batch == 0:
                st.commit()
        st.commit()
        st.flush()
        print("outcomes:", dict(outcomes))
        print("index   :", {k: v for k, v in st.stats().items() if k != "recovery"})
    return 0


def _cmd_search(a) -> int:
    from .store.index import IndexStore
    from .store.lock import IndexLocked
    try:
        st = IndexStore(a.index)
    except IndexLocked as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    with st:
        hits = st.search(" ".join(a.query), k=a.k)
        if not hits:
            print("no results")
        for rank, h in enumerate(hits, 1):
            p = st.get(h.arxiv_id)
            print(f"{rank:2d}. {h.score:7.3f}  {h.arxiv_id}  {p.title[:90]}")
            print(f"            {', '.join(p.categories)}  {p.datestamp}")
    return 0


def _cmd_index_stats(a) -> int:
    from .store.index import IndexStore
    with IndexStore(a.index) as st:
        for k, v in st.stats().items():
            print(f"{k:22s} {v}")
    return 0


def _cmd_fsck(a) -> int:
    from .store.fsck import fsck
    rep = fsck(a.index)
    for w in rep.warnings:
        print("warning:", w)
    for e in rep.errors:
        print("ERROR  :", e)
    print("info   :", rep.info)
    print("RESULT :", "OK" if rep.ok else "DAMAGED")
    return 0 if rep.ok else 1


def _cmd_beir(a) -> int:
    import json
    from .eval.beir import BeirError, run_beir
    try:
        res = run_beir(a.data, split=a.split, index_dir=a.index, k=a.k)
    except BeirError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(json.dumps(res, indent=1))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(res, f, indent=1)
            f.write("\n")
        print(f"saved: {a.out}")
    return 0


def _cmd_reindex(a) -> int:
    from .store.index import IndexCorrupt
    from .store.lock import IndexLocked
    from .store.reindex import ReindexError, reindex
    try:
        reindex(a.src, a.dst, flush_docs=a.flush_docs)
    except (ReindexError, IndexLocked, IndexCorrupt) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="paperlens")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth", help="write seeded SYNTHETIC papers (plumbing tests only)")
    s.add_argument("--n", type=int, default=1000)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=_cmd_synth)

    h = sub.add_parser("harvest", help="polite, resumable arXiv OAI-PMH harvest")
    h.add_argument("--contact", help="your email (goes in the User-Agent)")
    h.add_argument("--set", default="cs", help="OAI set (default: cs = all of computer science)")
    h.add_argument("--out", required=True)
    h.add_argument("--state", help="state file (default: <out>.state.json)")
    h.add_argument("--from-date", help="YYYY-MM-DD: only records modified on/after this day")
    h.add_argument("--max-records", type=int, help="stop after about this many lines (resumable)")
    h.add_argument("--min-interval", type=float, default=ARXIV_MIN_INTERVAL,
                   help=f"seconds between requests; minimum {ARXIV_RULE_SECONDS}")
    h.add_argument("--base-url", default=DEFAULT_BASE_URL)
    h.set_defaults(fn=_cmd_harvest)

    t = sub.add_parser("stats", help="summarise a JSONL file of papers")
    t.add_argument("path")
    t.set_defaults(fn=_cmd_stats)

    i = sub.add_parser("index", help="build/extend a persistent index from a JSONL file")
    i.add_argument("--index", required=True, help="index directory (created if missing)")
    i.add_argument("--input", required=True, help="JSONL of papers (harvest or synth output)")
    i.add_argument("--batch", type=int, default=64, help="papers per commit (group commit)")
    i.add_argument("--flush-docs", type=int, default=1000, help="buffer size that triggers a segment")
    i.add_argument("--skip-bad", action="store_true", help="skip unparseable lines instead of stopping")
    i.set_defaults(fn=_cmd_index)

    s = sub.add_parser("search", help="BM25 keyword search")
    s.add_argument("--index", required=True)
    s.add_argument("-k", type=int, default=10)
    s.add_argument("query", nargs="+")
    s.set_defaults(fn=_cmd_search)

    x = sub.add_parser("index-stats", help="show index statistics (runs recovery on open)")
    x.add_argument("--index", required=True)
    x.set_defaults(fn=_cmd_index_stats)

    f = sub.add_parser("fsck", help="check an index for damage (read-only)")
    f.add_argument("--index", required=True)
    f.set_defaults(fn=_cmd_fsck)
    b = sub.add_parser("beir", help="evaluate on a BEIR-format dataset (nDCG@10, Recall@100, MRR@10)")
    b.add_argument("--data", required=True, help="folder with corpus.jsonl, queries.jsonl, qrels/<split>.tsv")
    b.add_argument("--split", default="test")
    b.add_argument("--index", help="empty folder to keep the index in (default: temporary, deleted)")
    b.add_argument("-k", type=int, default=100, help="results retrieved per query")
    b.add_argument("--out", help="also save the JSON result here (e.g. docs/evidence/beir-scifact.json)")
    b.set_defaults(fn=_cmd_beir)

    r = sub.add_parser("reindex", help="rebuild an index with the current tokenizer into a NEW folder")
    r.add_argument("--src", required=True)
    r.add_argument("--dst", required=True)
    r.add_argument("--flush-docs", type=int, default=5000)
    r.set_defaults(fn=_cmd_reindex)
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
