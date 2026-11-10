# PaperLens: final report (state on 2026-11-09)

## What was built (all verified by running on Linux, Python 3.13, 2 cores)
- E2 persistent index: diary (WAL), immutable segments, atomic MANIFEST, recovery, fsck, single-writer lock, compaction, reader pins, snapshots.
- E1 pipeline: spawn-based supervisor, readers, one writer, private pipes, shared-memory vectors with crc, shared limiter/budget, chaos mode, dead letters.
- E3 retrieval: BM25 (compressed postings), 1-4 bit vectors + IVF + f16 rescoring, RRF hybrid, filters, citation graph (PageRank, PPR, label propagation), MinHash+LSH, late-interaction stand-in, more-like-this, key papers, explanations.
- Optional C++ accelerator with bit-identical Python fallback; FastAPI + one-page UI; evaluation and kill -9 demo scripts.

## Tests
481 passed, 0 failed (with C++); 410 passed + 71 skipped with `PAPERLENS_NATIVE=0`. Mutation scripts kill the injected bugs (counts in each PHASE doc; some mutants first survived and the tests were strengthened). Evidence JSON in `docs/evidence/`.

## What is NOT shown (be honest in the review)
- Nothing run on Windows (MSVC build, msvcrt locks, spawn query workers).
- No real arXiv harvest, OpenAlex call, BEIR numbers, or real embedding model.
- NFR3 (4-reader efficiency) and NFR4 (p95 at 50,000 papers) not measured.
- Graph rerank and late-interaction have no measured gain; by our own rule they stay OFF.
- On synthetic data hybrid was below vector-only, so NFR6 is not shown to hold. Synthetic data is plumbing evidence, not quality evidence.
- Block-Max WAND and SIMD popcount scan not built. Lucene/Pyserini baseline not run (no Java here).
- Power-loss safety not claimed (process kills only).
