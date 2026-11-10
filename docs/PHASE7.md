# Phase 7: interface, evaluation, demo

Built: FastAPI app + one HTML page (no CDN) showing a reason per result (`web/`, `explain.py`, CLI `serve-web`);
`scripts/evaluate_all.py` (PaperLens bm25/vector/hybrid vs SQLite FTS5 vs FAISS flat; writes JSON + markdown table);
`scripts/demo_kill9.py` (kills the writer with SIGKILL several times, reopens, checks nothing promised is missing, runs fsck).
Verified by running (Linux, synthetic data): demo passed 3 kills, 0 missing, recovery in 5-9 ms.
NOT done: Lucene via Pyserini (no Java here), numbers on real SciFact/SCIDOCS/arXiv, Windows runs.
