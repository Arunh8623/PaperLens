# Results measured on real data (Windows 11, Python 3.14.3, 12 CPU threads, no GPU, C++ extension NOT built)

Date: 2026-11-10. Embedder: all-MiniLM-L6-v2 (384 dims), 1-bit codes + half-precision rescoring.
Raw numbers: `docs/evidence/beir-*-REAL-windows.json`, `docs/evidence/recall-bits-real.json` (if copied from the PC).
Nothing here is tuned on the test queries.

## 1. Search quality (BEIR, nDCG@10)
| Dataset | BM25 (ours) | Vector (ours) | Hybrid RRF k=60 (ours) | NFR6: hybrid >= best single? |
|---|---|---|---|---|
| SciFact (5,183 docs, 300 queries) | 0.674 | 0.649 | **0.694** | **YES** |
| SCIDOCS (25,657 docs, 1,000 queries) | 0.154 | **0.217** | 0.200 | **NO** (hybrid is 0.017 below vector) |

Recall@100: SciFact bm25 0.883 / vector 0.925 / hybrid 0.958. SCIDOCS bm25 0.355 / vector 0.509 / hybrid 0.479.
MRR@10: SciFact 0.639 / 0.607 / 0.661. SCIDOCS 0.274 / 0.361 / 0.339.

Reading it honestly:
- SciFact: fusing keyword and meaning search beats both. This is the intended behaviour.
- SCIDOCS: fusion is worse than vector alone. Plain RRF gives both lists equal weight, and BM25 is much weaker here.
  Hypothesis (NOT tested): SCIDOCS asks "which papers would this paper cite", where keyword overlap is a weak signal.
- Our BM25 and vector numbers are close to the BEIR numbers published for BM25 and MiniLM-L6 as far as we remember
  (about 0.665 / 0.158 for BM25); **check against the BEIR paper/leaderboard before quoting them**.
- Planned (not done): weighted fusion, with the weight chosen on a held-out half of the queries and scored on the other half.

## 2. Compression (our 10,400 real arXiv cs papers, 300 queries)
| Bits | size vs float32 | recall@10 codes only | recall@10 after rescoring |
|---|---|---|---|
| 1 | 3.9% | 0.676 | 0.997 |
| 2 | 7.0% | 0.790 | 1.000 |
| 3 | 10.2% | 0.888 | 1.000 |
| 4 | 13.3% | 0.937 | 1.000 |
NFR7 (<= 1/8 of float32 and recall >= 95%): met at 1, 2, 3 bits; 4 bits fails the size limit (13.3% > 12.5%).
The codes alone are not enough; rescoring a pool of 100 with the half-precision copy gives the recall.
Disk per vector = code + 768-byte half-precision copy: compression saves memory and scan time, not disk.

## 3. Latency (this PC, pure Python/numpy)
| Dataset | mode | p50 ms | p95 ms |
|---|---|---|---|
| SciFact | bm25 / vector / hybrid | 3.6 / 47.8 / 46.7 | 8.7 / 63.2 / 65.0 |
| SCIDOCS | bm25 / vector / hybrid | 9.3 / 165 / 158 | 23.6 / 325 / 191 |
NFR4 (p95 <= 500 ms at 50,000 papers) is NOT demonstrated: the largest run is 25,657 documents. Not extrapolated.

## 4. Index build cost
SciFact 236 s, SCIDOCS 1,736 s (mostly CPU embedding). Index size 15.5 MB and 68.7 MB.

## Still not measured
Windows pytest run, C++ build on Windows, 4-reader speed-up (NFR3), 50k-paper latency (NFR4), OpenAlex citation run, graph/late rerank gain, Lucene/Pyserini baseline.
