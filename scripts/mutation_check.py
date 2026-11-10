"""Mutation check: break the code on purpose, confirm the tests notice.

    python scripts/mutation_check.py            # all mutants, writes docs/evidence/mutation-<date>-<commit>.json
    python scripts/mutation_check.py M03 M07    # only some

For each mutant: apply ONE small edit to the source, run the test suite with
`-x` (stop at the first failure), restore the file, record KILLED (a test
failed = good) or SURVIVED (no test failed = a gap in the tests, or an
equivalent change). A mutant whose edit does not match exactly once is
reported NOT_APPLIED (never silently counted as a survivor).

EXPECTED survivors are listed with the reason; they are limits of what a
process-kill test can see, not bugs hidden from you.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IDX = "src/paperlens/store/index.py"

# (id, file, old text (must occur exactly once), new text, what it breaks, expected_survivor_reason or None)
MUTANTS = [
    ("M01", IDX, 'if op == "put":\n                    self._apply_put(PaperRecord.from_dict(e["paper"]))',
     'if op == "put":\n                    pass',
     "recovery does not replay diary puts", None),
    ("M02", IDX, "                if seq <= flushed:\n                    continue                               # already inside a segment",
     "                if False:\n                    continue",
     "recovery re-applies entries already inside a segment", None),
    ("M03", IDX, "            elif name.startswith(\"seg-\") and name not in listed:\n                shutil.rmtree(os.path.join(self.seg_root, name))\n                stats[\"removed_unlisted\"] += 1",
     "            elif name.startswith(\"seg-\") and name not in listed:\n                pass",
     "recovery keeps unlisted (orphan) segments", None),
    ("M04", IDX, "res = recover_tail(path) if is_last else scan_file(path)",
     "res = scan_file(path)",
     "recovery never cuts the torn diary tail", None),
    ("M05", IDX, "        publish_segment(self.seg_root, tmp, name)\n        crashpoint(\"flush_after_segment_rename\")\n        write_manifest(self.path, self.manifest.next(\n            last_flushed_seq=self.seq, segments=self.manifest.segments + (name,)))",
     "        write_manifest(self.path, self.manifest.next(\n            last_flushed_seq=self.seq, segments=self.manifest.segments + (name,)))\n        crashpoint(\"flush_after_segment_rename\")\n        publish_segment(self.seg_root, tmp, name)",
     "flush publishes the MANIFEST before the segment is renamed into place", None),
    ("M06", IDX, "                if op == \"put\":", "                if op == \"put\" and False:\n                    pass\n                elif op == \"put\":",
     "(control) equivalent rewrite of the op test", "EQUIVALENT: behaves identically, a survivor here is correct"),
    ("M07", "src/paperlens/store/diary.py", "        if self._dirty:\n            os.fsync(self._fd)\n            self._dirty = False",
     "        if self._dirty:\n            self._dirty = False",
     "commit() no longer calls fsync", "process-kill tests cannot see a missing fsync: the OS still holds the bytes. Only power-loss tests on real hardware can."),
    ("M08", IDX, "        outcome = decide(split_key(cur)[1] if cur else None, paper.datestamp)\n        if outcome in (\"duplicate\", \"stale\"):\n            return PutResult(outcome, 0)",
     "        outcome = decide(None, paper.datestamp)",
     "idempotency rules disabled (duplicates/stale are stored)", None),
    ("M09", IDX, "                    if si < sj:", "                    if si <= sj:",
     "a segment's tombstone also kills documents in the SAME segment", None),
    ("M10", IDX, "        n_docs = sum(s.num_docs for s in self.segments) + len(self.buf)\n        if not terms",
     "        n_docs = len(self.live)\n        if not terms",
     "BM25 counts only live docs in N (instead of all physical docs)", None),
    ("M11", IDX, "K1, B = 1.2, 0.75", "K1, B = 1.2, 0.7", "BM25 b parameter wrong", None),
    ("M12", IDX, "total_len = sum(s.total_doclen for s in self.segments) + sum(d.dl for d in self.buf.values())",
     "total_len = sum(s.total_doclen for s in self.segments)",
     "avgdl ignores the buffer", None),
    ("M13", IDX, "last_flushed_seq=self.seq, segments=self.manifest.segments + (name,)))\n        self.manifest",
     "last_flushed_seq=self.seq - 1, segments=self.manifest.segments + (name,)))\n        self.manifest",
     "MANIFEST records last_flushed_seq one too low", None),
    ("M14", "src/paperlens/store/fsck.py", "        elif (zlib.crc32(data) & 0xFFFFFFFF) != info[\"crc32\"]:\n            err(f\"{fname}: checksum mismatch\")",
     "        elif False:\n            pass",
     "fsck stops verifying file checksums", None),
    ("M15", IDX, "        self._lock.acquire()\n        try:\n            self._recover()",
     "        try:\n            self._recover()",
     "writer lock is never taken", None),
]


def run_tests() -> tuple:
    p = subprocess.run([sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider"],
                       cwd=ROOT, capture_output=True, text=True)
    failed = [ln for ln in p.stdout.splitlines() if ln.startswith("FAILED")]
    return p.returncode, (failed[0][:110] if failed else "")


def main(argv) -> int:
    want = set(argv)
    results = []
    for mid, rel, old, new, what, expected in MUTANTS:
        if want and mid not in want:
            continue
        path = os.path.join(ROOT, rel)
        original = open(path, encoding="utf-8", newline="").read()
        if original.count(old) != 1:
            results.append({"id": mid, "status": "NOT_APPLIED", "what": what,
                            "detail": f"edit text found {original.count(old)} times"})
            print(f"{mid} NOT_APPLIED  {what}")
            continue
        try:
            open(path, "w", encoding="utf-8", newline="").write(original.replace(old, new))
            rc, first_fail = run_tests()
        finally:
            open(path, "w", encoding="utf-8", newline="").write(original)
        status = "KILLED" if rc != 0 else "SURVIVED"
        results.append({"id": mid, "status": status, "what": what, "caught_by": first_fail,
                        "expected_survivor_reason": expected})
        print(f"{mid} {status:9s} {what}\n      {first_fail or ('(expected) ' + expected if expected else '')}")
    killed = sum(r["status"] == "KILLED" for r in results)
    surv = [r["id"] for r in results if r["status"] == "SURVIVED"]
    napplied = [r["id"] for r in results if r["status"] == "NOT_APPLIED"]
    print(f"\nkilled {killed}/{len(results)}; survived {surv}; not applied {napplied}")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    now = datetime.now(timezone.utc)
    out = os.path.join(ROOT, "docs", "evidence", f"mutation-{now:%Y%m%d}-{commit}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"recorded_utc": f"{now:%Y-%m-%dT%H:%M:%SZ}", "git_commit": commit,
                   "results": results}, f, indent=1)
        f.write("\n")
    print("evidence:", os.path.relpath(out, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
