r"""Harvest arXiv cs papers from SEVERAL YEARS (for example 2015 to 2026) in small date windows.

What it does and why (plain language)
-------------------------------------
arXiv's OAI-PMH interface returns records in order of their *modification date* (datestamp), oldest first, starting
from the date you give. Asking for "everything since 2015" would be enormous and the server is very slow at it.
So this script asks for a few SMALL WINDOWS in every year (by default one week in February, May, August and
November), caps each window, and merges everything into one JSONL file you can pass to `python -m paperlens index`.

HONEST NOTES
* The datestamp is the date a record was last changed, not the submission date. A 2015 paper that was revised
  in 2020 has a 2020 datestamp, so it will NOT appear in a 2015 window. What you get for 2015 is mostly papers
  that were submitted in 2015 and never revised. That is fine for a demo; it is not a complete 2015 archive.
* The year filter in the search page uses the paper's `created` year, so what you see is the real submission year.
* Polite: it runs `python -m paperlens harvest` once per window, one after another, so the rule "one request every
  3 seconds, one connection" is kept. Each window can be re-run safely: finished windows are skipped, unfinished
  ones resume.

Example:
    python scripts/harvest_years.py --contact you@example.com --first 2015 --last 2026
"""
import argparse
import datetime
import os
import subprocess
import sys


def windows_for(year, spec):
    for part in spec.split(","):
        a, b = part.split(":")
        yield f"{year}-{a}", f"{year}-{b}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--contact", required=True, help="your email (goes in the User-Agent, arXiv rule)")
    ap.add_argument("--first", type=int, default=2015)
    ap.add_argument("--last", type=int, default=datetime.date.today().year)
    ap.add_argument("--windows", default="02-01:02-07,05-01:05-07,08-01:08-07,11-01:11-07",
                    help="comma list of MM-DD:MM-DD windows inside each year")
    ap.add_argument("--per-window", type=int, default=300, help="stop a window after about this many records")
    ap.add_argument("--out-dir", default=os.path.join("data", "years"))
    ap.add_argument("--merged", default=os.path.join("data", "arxiv_cs_years.jsonl"))
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--base-url", help=argparse.SUPPRESS)           # tests only
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    today = datetime.date.today().isoformat()
    outs, failed = [], []
    for year in range(a.first, a.last + 1):
        for lo, hi in windows_for(year, a.windows):
            if lo > today:
                continue
            hi = min(hi, today)
            out = os.path.join(a.out_dir, f"w-{lo}_{hi}.jsonl")
            cmd = [sys.executable, "-m", "paperlens", "harvest", "--contact", a.contact, "--set", "cs",
                   "--from-date", lo, "--until-date", hi, "--max-records", str(a.per_window),
                   "--timeout", str(a.timeout), "--out", out]
            if a.base_url:
                cmd += ["--base-url", a.base_url]
            print(f"\n=== window {lo} .. {hi} ===", flush=True)
            r = subprocess.run(cmd)
            if r.returncode == 130:
                print("interrupted: re-run the same command to resume")
                return 130
            if r.returncode != 0:
                failed.append((lo, hi))
            if os.path.exists(out):
                outs.append(out)

    n = 0
    with open(a.merged, "w", encoding="utf-8", newline="") as w:
        for out in outs:
            with open(out, "r", encoding="utf-8", newline="") as f:
                for line in f:
                    if line.strip():
                        w.write(line if line.endswith("\n") else line + "\n")
                        n += 1
    print(f"\nmerged {n} lines from {len(outs)} windows into {a.merged}")
    if failed:
        print("windows that failed (re-run the same command to retry them):", failed)
    print(f"next: python -m paperlens index --index data\\ixs --input {a.merged} --embedder st:sentence-transformers/all-MiniLM-L6-v2")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
