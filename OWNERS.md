# Engine map

MAJOR engines are E1, E2, E3. Anything marked SUPPORTING is not a graded engine.

The course split the work between four people on paper. In practice **Arun Hadimani builds and
maintains everything** in this repository, with AI assistance (see the README disclosure; commits from 2026-11-01
carry a `Co-Authored-By` trailer, the first 10 do not). This file records who is responsible for what in fact,
so the evidence is honest: the git history shows who did the work.

| Engine | Kind | Modules (src/paperlens/...) | Built by | Status |
|---|---|---|---|---|
| E1 OS Process & Ingestion | MAJOR | `pipeline/` | Arun (+ AI) | Phase 2 built (Linux-tested) |
| E1 harvester (arXiv/OpenAlex) | SUPPORTING | `harvest/` | Arun (+ AI) | arXiv + OpenAlex built (fake servers only) |
| E2 Persistent Hybrid Index | MAJOR | `store/` | Arun (+ AI) | keyword index + vector storage built (Phase 1a, 3); compaction later |
| E3 Retrieval, keyword side | MAJOR | `text/`, `eval/` | Arun (+ AI) | BM25 + toy tokenizer built; eval Phase 1b |
| E3 Retrieval, semantic + graph | MAJOR | `vectors/`, `graph/` | Arun (+ AI) | Phase 3 built (Linux-tested, stand-in embedder); Phase 4 built (CSR graph, PageRank/PPR/label propagation, MinHash+LSH, rerank; Linux-tested) |
| Shared contracts | shared | `contracts/` | Arun (+ AI) | built |
| Pre-trained embedding models | SUPPORTING | - | - | we do not train models |

Names on the course's paper split (from the brief, which listed them two different ways):
Aditya Gavimath, Kartik Hegde, Prajwal B. If the guide asks each member to explain an engine at a
review, the team has to decide among themselves who presents what; the "what and why" notes in
`docs/PHASE*.md` are written so any engine can be explained by someone who has read them.
