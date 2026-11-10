# Review-day demo script (about 10 minutes). Run it once tonight so nothing surprises you.

Open PowerShell in the project folder, activate the venv: `.venv\Scripts\Activate.ps1`.
Your real index is `data\ixs` (10,400 arXiv cs papers with vectors).

## 0. Before the review (5 minutes, do tonight)
    python -m paperlens index-stats --index data\ixs
    python -m paperlens fsck --index data\ixs                       # must say OK
    python -m paperlens search --index data\ixs "graph neural network for molecules" --mode hybrid
    python -m pytest -q                                              # keep the pass count; report any failure to me
    python scripts\demo_kill9.py --rounds 3 --docs 4000              # kill demo; NEVER run on Windows before: tell me if it fails

## 1. Show the system (what to say)
1. Search, three modes (same index):
       python -m paperlens search --index data\ixs "transformer attention efficiency" --mode bm25
       python -m paperlens search --index data\ixs "transformer attention efficiency" --mode vector
       python -m paperlens search --index data\ixs "transformer attention efficiency" --mode hybrid
   Say: keyword finds exact words, vector finds meaning, hybrid merges both lists (RRF).
2. Web page with a reason per result:
       python -m paperlens serve-web --index data\ixs     -> open http://127.0.0.1:8000
   (needs `python -m pip install -e ".[web]"` once)
3. Crash safety: `python scripts\demo_kill9.py --rounds 3 --docs 4000`
   Say: the writer is killed at random moments; on reopen it replays the diary, nothing acknowledged is lost, fsck passes.
   Honest line: this is process death, not power loss.
4. Pipeline fault tolerance: `python -m pytest -q tests\test_pipeline.py` (readers killed, writer killed, poison paper).
5. Numbers: show docs\RESULTS.md.

## 2. Questions you will get (honest answers)
- "Does it beat Lucene?" We never claimed that; Lucene via Pyserini was not run (needs Java).
- "Is hybrid always better?" On SciFact yes (0.694 vs 0.674/0.649). On SCIDOCS no (0.200 vs vector 0.217). Weighted fusion is planned.
- "Why 1-bit vectors?" 3.9% of float32 for scanning, recall 0.997 after rescoring; disk still keeps a half-precision copy.
- "Is the C++ part used?" Built and tested on Linux with a Python fallback; NOT built on this PC, so the runs here are pure Python.
- "Power loss?" Not tested. Only process kills.
- "50,000 papers?" Largest run: 25,657 docs on SCIDOCS (hybrid p95 191 ms). 50k is not measured.
- "Citation graph / late interaction?" Built and tested on synthetic data; no measured gain, so switched off by default. No real OpenAlex run.

## 3. If something breaks live
- Web page fails: use the CLI searches above.
- demo_kill9 fails on Windows: run `python -m pytest -q tests\test_crash.py` and show the passing crash-point tests, say Windows kill demo was untested.
