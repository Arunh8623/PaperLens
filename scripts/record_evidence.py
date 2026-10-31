"""Run the test suite and write a dated, traceable evidence file.

    python scripts/record_evidence.py            # writes docs/evidence/tests-<UTC date>-<git hash>.json

The JSON records: UTC timestamp, git commit (+ whether the tree was dirty),
Python and OS versions, totals, and per-test-file counts. Reviewers can match
the commit hash against `git log`. Exit code = pytest's exit code, so CI fails
when tests fail.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timezone

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        junit = os.path.join(td, "junit.xml")
        rc = subprocess.run([sys.executable, "-m", "pytest", f"--junitxml={junit}", "-q"],
                            cwd=ROOT).returncode
        per_file = defaultdict(lambda: {"passed": 0, "failed": 0, "skipped": 0})
        for case in ET.parse(junit).getroot().iter("testcase"):
            f = case.get("classname", "").replace(".", "/") + ".py"
            if case.find("failure") is not None or case.find("error") is not None:
                per_file[f]["failed"] += 1
            elif case.find("skipped") is not None:
                per_file[f]["skipped"] += 1
            else:
                per_file[f]["passed"] += 1
    totals = {k: sum(v[k] for v in per_file.values()) for k in ("passed", "failed", "skipped")}
    now = datetime.now(timezone.utc)
    commit = git("rev-parse", "--short", "HEAD")
    dirty = bool(git("status", "--porcelain"))
    doc = {
        "recorded_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": commit or None,
        "git_tree_dirty": dirty,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "pytest_exit_code": rc,
        "totals": totals,
        "per_file": dict(sorted(per_file.items())),
    }
    out_dir = os.path.join(ROOT, "docs", "evidence")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"tests-{now.strftime('%Y%m%d')}-{commit or 'nogit'}"
                                f"{'-dirty' if dirty else ''}-py{platform.python_version()}-"
                                f"{platform.system().lower()}.json")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    print(f"evidence written: {os.path.relpath(out, ROOT)}  {totals}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
