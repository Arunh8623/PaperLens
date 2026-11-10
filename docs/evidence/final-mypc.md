# Results table

| dataset | system | nDCG@10 | Recall@100 | MRR@10 | p50 ms | p95 ms | index MB |
|---|---|---|---|---|---|---|---|
| scifact (5183 docs) | PaperLens bm25 | 0.6740 | 0.8826 | 0.6395 | 2.98 | 7.04 | 15.54 |
| scifact (5183 docs) | PaperLens vector | 0.6494 | 0.9250 | 0.6072 | 29.7 | 37.65 | 15.54 |
| scifact (5183 docs) | PaperLens hybrid | 0.6941 | 0.9583 | 0.6606 | 73.89 | 94.28 | 15.54 |
| scifact (5183 docs) | SQLite FTS5 | 0.6670 | 0.8792 | 0.6317 | 8.61 | 16.59 | 14.91 |
| scifact (5183 docs) | FAISS flat (exact) | not run: faiss not installed (python -m pip install faiss-cpu) | | | | | |
| scidocs (25657 docs) | PaperLens bm25 | 0.1539 | 0.3547 | 0.2738 | 8.86 | 22.6 | 68.68 |
| scidocs (25657 docs) | PaperLens vector | 0.2170 | 0.5093 | 0.3610 | 141.04 | 161.49 | 68.68 |
| scidocs (25657 docs) | PaperLens hybrid | 0.2002 | 0.4789 | 0.3385 | 150.31 | 167.37 | 68.68 |
| scidocs (25657 docs) | SQLite FTS5 | 0.1543 | 0.3505 | 0.2751 | 9.34 | 17.89 | 58.84 |
| scidocs (25657 docs) | FAISS flat (exact) | not run: faiss not installed (python -m pip install faiss-cpu) | | | | | |

Lucene via Pyserini: NOT run (needs Java).
