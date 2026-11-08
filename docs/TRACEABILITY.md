# Traceability: requirement -> module -> test

Status: **built** = code + passing tests for the whole requirement; **partial** = a named
slice is built, the rest is planned; **planned** = nothing built yet.
Test names are in `tests/`. Last updated: Phase 0.

## Functional

| Req | What | Engine | Module(s) | Tests | Status |
|---|---|---|---|---|---|
| FR1 | collect papers within source limits | E1 (supporting) | `harvest/arxiv_oai.py`, `harvest/limiter.py` | `test_harvest.py`, `test_limiter.py`, `test_cli.py` | **partial**: arXiv harvester built and tested against a local fake server; **real run on arXiv pending**; OpenAlex = Phase 4 |
| FR2 | parallel fault-tolerant ingestion | E1 | `pipeline/` (`supervisor.py`, `workers.py`, `messages.py`, `shm.py`, `embed.py`) | `test_pipeline.py` (27 tests), `test_shared_limiter.py` (6) | **built, tested on Linux** (reader kills, writer kills at all 6 crash points, chaos, poison paper, hang, corrupt vector, re-run); **not run on Windows** |
| FR3 | durable index + crash recovery | E2 | `store/diary.py`, `manifest.py`, `segment.py`, `index.py`, `fsck.py`, `lock.py`, `crashpoints.py` | `test_diary.py`, `test_manifest.py`, `test_segment.py`, `test_index.py`, `test_crash.py`, `test_lock.py` | **built for process crashes** (all 6 crash points + random SIGKILL, Linux). **Not shown:** power-loss safety, Windows |
| FR4 | background compaction | E2 | - | - | planned (Phase 5) |
| FR5 | keyword search, BM25, science-aware tokenizer | E3 | `text/tokenizer.py`, `text/varint.py`, `store/index.py` (`search`) | `test_tokenizer.py`, `test_varint.py`, `test_index.py` (vs `oracle.py`) | **partial**: BM25 over varint postings equals a brute-force model to 1e-9 on synthetic data; tokenizer is still the toy; **quality on real data / BEIR not measured** |
| FR6 | semantic search, compressed vectors | E3 | `vectors/quant.py` (1-4 bit RaBitQ-style codes), `vectors/ivf.py`, `vectors/embedders.py`, `store/index.py` (`search_vector`), `store/segment.py` | `test_vectors.py` (49), `test_vector_crash.py` (24) | **built, tested on Linux** with a stand-in embedder; **real embedding model NOT tested here**; no real-data quality numbers yet |
| FR7 | hybrid RRF + category/year filters | E3 | `vectors/fusion.py`, `store/filters.py`, `IndexStore.search_hybrid` | `test_vectors.py` (RRF hand-computed, hybrid == RRF of the two lists, filters == brute force, zone map) | **built, tested on Linux** |
| FR8 | citation rerank + near-duplicate removal | E3 | - | - | planned (Phase 4) |
| FR9 | results with reasons: CLI + web | E3 | `cli.py` (`index`, `search --mode/--cat/--year-from`, `index-stats`, `fsck`) | `test_cli.py`, `test_cli_index.py` | **partial**: hybrid hits say which list found them and at which rank; richer plain-language reasons + web page = Phase 7 |
| FR10 | one-command evaluation | E3 | - | - | planned (Phase 1+) |
| FR11 | late interaction, key papers (challenge) | E3 | - | - | planned (Phase 6) |

## Non-functional

| Req | Target | Engine | Evidence so far | Status |
|---|---|---|---|---|
| NFR1 | 0 lost / 0 duplicated acknowledged papers across kills at every write step | E2 | 6 of 6 crash points x 2 occurrences x {plain, messy} streams = 24 runs, + 2 random-SIGKILL runs (6 rounds each), each followed by fsck and a full re-send; mutation check kills 13/15 recovery/ranking mutants (see PHASE1A.md for the 2 expected survivors) | **built for process kills on Linux**; "writer kill" under the pipeline = Phase 2; power loss untested |
| NFR2 | ingestion fault tolerance (readers killed every ~0.25 s, writer twice) | E1 | `test_chaos_readers_killed_every_quarter_second_and_writer_twice` (300-paper messy stream, >= 3 reader kills, 2 writer kills, index == ground truth, replacement < 5 s) | **built, Linux**; scale of the run is small (300 papers) |
| NFR3 | >= 70% parallel efficiency at 4 readers | E1 | `bench/pipeline_speedup.py`; sandbox (2 cores, SIMULATED 3 ms/paper CPU work) gave 1.79x at 2 readers = 0.89 efficiency (`docs/evidence/speedup-sandbox-2core-SIMULATED-burn3ms.json`) | **NOT demonstrated**: needs a >= 4-core run on your machine |
| NFR4 | p95 <= 500 ms at 50,000 papers | E3 | - | **not measured** (needs 50,000 real papers on your machine) |
| NFR5 | consistent read snapshots | E2 | - | planned (Phase 5) |
| NFR6 | hybrid nDCG@10 >= better of keyword/vector on SciFact + SCIDOCS | E3 | `eval/beir.py` evaluates bm25 / vector / hybrid on one index and sets `nfr6_hybrid_at_least_best_single` | mechanics tested on a tiny dataset | **NOT measured**: needs the datasets and a real embedder (`beir --embedder st:...`) on your machine |
| NFR7 | codes <= 1/8 of float32, recall@10 >= 95% of exact | E3 + E2 | `bench/recall_vs_bits.py`; `docs/evidence/recall-bits-sandbox-SYNTHETIC*.json` | `test_vectors.py` (size, recall on clustered data) | **partial**: size OK (1-bit = 3.9% of float32). Recall on SYNTHETIC data with 20,000 vectors, pool 100: 1-bit 0.81 (FAILS), 2-bit 0.95, 3-bit 0.999; with pool 500: 1-bit 0.997. Default pool now scales (2.5% of vectors). **Real-vector numbers pending** |
| NFR8 | >= 3 s between arXiv requests, 0 calls over OpenAlex budget | E1 | spacing proven with a virtual clock **and** measured at the server with a real clock (`test_real_time_spacing_measured_by_the_server`); the CLI refuses < 3 s | **partial**: all-process `SharedLimiter` + `SharedBudget` built and tested (4 racing processes; pipeline test with 3 readers); real arXiv fetching through it not wired yet; OpenAlex budget use = Phase 4 |
| NFR9 | Windows + Linux, no GPU/paid service, one-command results | all | tested on Linux, Python 3.10/3.11/3.12/3.13; Windows hardening written (`retry_fs`, timing slack) and `scripts/windows_check.py` ready; **never run on Windows** | **partial** |

## Phase 0 deliverables (from the brief)

| Item | Where | Status |
|---|---|---|
| repo layout + git | whole repo | built (dated commits; AI help disclosed in README, trailers from 2026-11-01) |
| contracts doc | `docs/CONTRACTS.md` | built; **awaiting 4 sign-offs** |
| PaperRecord | `contracts/paper.py` | built + tested |
| diary framing + torn-tail scan | `store/diary.py` | built + tested + mutation-checked |
| MANIFEST atomic write/read + checksum | `store/manifest.py` | built + tested + mutation-checked |
| science-aware toy tokenizer | `text/tokenizer.py` | built + tested + mutation-checked |
| polite OAI-PMH harvester | `harvest/` | built + tested vs fake server + mutation-checked; **real run pending** |
| synthetic data generator | `synth.py` | built + tested |
| CI-style test script | `scripts/`, `.github/workflows/ci.yml` | script built and run; **workflow written but never run** |
| ~10,000 real arXiv records | - | **not done** (needs your machine; see README) |
