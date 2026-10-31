# Traceability: requirement -> module -> test

Status: **built** = code + passing tests for the whole requirement; **partial** = a named
slice is built, the rest is planned; **planned** = nothing built yet.
Test names are in `tests/`. Last updated: Phase 0.

## Functional

| Req | What | Engine | Module(s) | Tests | Status |
|---|---|---|---|---|---|
| FR1 | collect papers within source limits | E1 (supporting) | `harvest/arxiv_oai.py`, `harvest/limiter.py` | `test_harvest.py`, `test_limiter.py`, `test_cli.py` | **partial**: arXiv harvester built and tested against a local fake server; **real run on arXiv pending**; OpenAlex = Phase 4 |
| FR2 | parallel fault-tolerant ingestion | E1 | `pipeline/` | - | planned (Phase 2) |
| FR3 | durable index + crash recovery | E2 | `store/diary.py`, `store/manifest.py`, `store/crashpoints.py` | `test_diary.py`, `test_manifest.py` | **partial**: diary framing, torn-tail recovery, atomic MANIFEST. Segments, flush, replay: Phase 1 |
| FR4 | background compaction | E2 | - | - | planned (Phase 5) |
| FR5 | keyword search, BM25, science-aware tokenizer | E3 | `text/tokenizer.py` | `test_tokenizer.py` | **partial**: toy tokenizer only; postings + BM25 Phase 1 |
| FR6 | semantic search, compressed vectors | E3 | - | - | planned (Phase 3) |
| FR7 | hybrid RRF + category/year filters | E3 | - | - | planned (Phase 3) |
| FR8 | citation rerank + near-duplicate removal | E3 | - | - | planned (Phase 4) |
| FR9 | results with reasons: CLI + web | E3 | `cli.py` (Phase 0 commands only) | `test_cli.py` | planned (Phase 1 CLI search; Phase 7 web) |
| FR10 | one-command evaluation | E3 | - | - | planned (Phase 1+) |
| FR11 | late interaction, key papers (challenge) | E3 | - | - | planned (Phase 6) |

## Non-functional

| Req | Target | Engine | Evidence so far | Status |
|---|---|---|---|---|
| NFR1 | 0 lost / 0 duplicated acknowledged papers across kills at every write step | E2 | diary + MANIFEST crash tests only (2 of 6 crash points wired) | **partial** |
| NFR2 | ingestion fault tolerance (readers killed every ~0.25 s, writer twice) | E1 | - | planned (Phase 2) |
| NFR3 | >= 70% parallel efficiency at 4 readers | E1 | - (must be measured on a >= 4-core machine; our sandbox has 2) | planned |
| NFR4 | p95 <= 500 ms at 50,000 papers | E3 | - | planned |
| NFR5 | consistent read snapshots | E2 | - | planned (Phase 5) |
| NFR6 | hybrid nDCG@10 >= better of keyword/vector on SciFact + SCIDOCS | E3 | - | planned |
| NFR7 | codes <= 1/8 of float32, recall@10 >= 95% of exact | E3 + E2 | - | planned (Phase 3) |
| NFR8 | >= 3 s between arXiv requests, 0 calls over OpenAlex budget | E1 | spacing proven with a virtual clock **and** measured at the server with a real clock (`test_real_time_spacing_measured_by_the_server`); the CLI refuses < 3 s | **partial**: single process only; all-process limiter = Phase 2; OpenAlex budget = Phase 4 |
| NFR9 | Windows + Linux, no GPU/paid service, one-command results | all | tested on Linux, Python 3.10/3.11/3.12/3.13; **Windows untested** | **partial** |

## Phase 0 deliverables (from the brief)

| Item | Where | Status |
|---|---|---|
| repo layout + git | whole repo | built (dated commits) |
| contracts doc | `docs/CONTRACTS.md` | built; **awaiting 4 sign-offs** |
| PaperRecord | `contracts/paper.py` | built + tested |
| diary framing + torn-tail scan | `store/diary.py` | built + tested + mutation-checked |
| MANIFEST atomic write/read + checksum | `store/manifest.py` | built + tested + mutation-checked |
| science-aware toy tokenizer | `text/tokenizer.py` | built + tested + mutation-checked |
| polite OAI-PMH harvester | `harvest/` | built + tested vs fake server + mutation-checked; **real run pending** |
| synthetic data generator | `synth.py` | built + tested |
| CI-style test script | `scripts/`, `.github/workflows/ci.yml` | script built and run; **workflow written but never run** |
| ~10,000 real arXiv records | - | **not done** (needs your machine; see README) |
