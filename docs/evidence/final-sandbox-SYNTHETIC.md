# Results table (SYNTHETIC DATA: plumbing check only, not quality evidence)

| dataset | system | nDCG@10 | Recall@100 | MRR@10 | p50 ms | p95 ms | index MB |
|---|---|---|---|---|---|---|---|
| pl-synth-o49pp9v2 (2000 docs) | PaperLens bm25 | 0.1011 | 0.2360 | 0.2650 | 0.29 | 0.65 | 1.80 |
| pl-synth-o49pp9v2 (2000 docs) | PaperLens vector | 0.1529 | 0.2400 | 0.3333 | 2.17 | 2.51 | 1.80 |
| pl-synth-o49pp9v2 (2000 docs) | PaperLens hybrid | 0.1207 | 0.2520 | 0.3333 | 2.7 | 2.76 | 1.80 |
| pl-synth-o49pp9v2 (2000 docs) | SQLite FTS5 | 0.0662 | 0.2240 | 0.1202 | 0.63 | 0.83 | 1.52 |
| pl-synth-o49pp9v2 (2000 docs) | FAISS flat (exact) | 0.1529 | 0.2400 | 0.3333 | 0.05 | 0.15 | 0.51 |

Lucene via Pyserini: NOT run (needs Java).
