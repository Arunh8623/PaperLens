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
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
