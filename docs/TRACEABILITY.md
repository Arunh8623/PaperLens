# Traceability: requirement -> module -> test

Status: **built** = code + passing tests for the whole requirement; **partial** = a named
slice is built, the rest is planned; **planned** = nothing built yet.
Test names are in `tests/`. Last updated: Phase 7.

## Functional

| Req | What | Engine | Module(s) | Tests | Status |
|---|---|---|---|---|---|
| FR1 | collect papers within source limits | E1 (supporting) | `harvest/arxiv_oai.py`, `harvest/limiter.py`, `graph/openalex.py` (budget-capped, resumable) | `test_harvest.py`, `test_limiter.py`, `test_cli.py` | **partial**: arXiv harvester and OpenAlex client built and tested against local fake servers; **real runs on arXiv/OpenAlex pending** |
| FR2 | parallel fault-tolerant ingestion | E1 | `pipeline/` (`supervisor.py`, `workers.py`, `messages.py`, `shm.py`, `embed.py`) | `test_pipeline.py` (27 tests), `test_shared_limiter.py` (6) | **built, tested on Linux** (reader kills, writer kills at all 6 crash points, chaos, poison paper, hang, corrupt vector, re-run); **not run on Windows** |
| FR3 | durable index + crash recovery | E2 | `store/diary.py`, `manifest.py`, `segment.py`, `index.py`, `fsck.py`, `lock.py`, `crashpoints.py` | `test_diary.py`, `test_manifest.py`, `test_segment.py`, `test_index.py`, `test_crash.py`, `test_lock.py` | **built for process crashes** (all 6 crash points + random SIGKILL, Linux). **Not shown:** power-loss safety, Windows |
| FR4 | background compaction | E2 | `store/compaction.py`, `IndexStore.compact/start_background_compaction`, `store/pins.py`, CLI `compact`, `ingest --compact` | `test_compaction.py` (32 incl. 16 real-kill crash tests at the 4 compaction crash points, oracle equality, vectors, tombstones, delete-during-merge), `test_query_server.py` | **built, tested on Linux**; mutation check 14/14 killed. Never run on Windows; speed of merging at 50,000 papers not measured |
| FR5 | keyword search, BM25, science-aware tokenizer | E3 | `text/tokenizer.py`, `text/varint.py`, `store/index.py` (`search`) | `test_tokenizer.py`, `test_varint.py`, `test_index.py` (vs `oracle.py`) | **partial**: BM25 over varint postings equals a brute-force model to 1e-9 on synthetic data; tokenizer is still the toy; **quality on real data / BEIR not measured** |
| FR6 | semantic search, compressed vectors | E3 | `vectors/quant.py` (1-4 bit RaBitQ-style codes), `vectors/ivf.py`, `vectors/embedders.py`, `store/index.py` (`search_vector`), `store/segment.py` | `test_vectors.py` (49), `test_vector_crash.py` (24) | **built, tested on Linux** with a stand-in embedder; **real embedding model NOT tested here**; no real-data quality numbers yet |
| FR7 | hybrid RRF + category/year filters | E3 | `vectors/fusion.py`, `store/filters.py`, `IndexStore.search_hybrid` | `test_vectors.py` (RRF hand-computed, hybrid == RRF of the two lists, filters == brute force, zone map) | **built, tested on Linux** |
| FR8 | citation rerank + near-duplicate removal | E3 | `graph/{csr,rank,rerank,minhash}.py`, `store/graphfile.py`, `search --graph-rerank`, `dupes`, `bench/graph_ablation.py` | `test_graph.py`, `test_minhash.py`, `test_graph_publish.py`, `test_cli_graph.py` | **built, tested on Linux** (PageRank/PPR equal a direct linear-algebra solution; LSH pairs equal brute force). **Whether reranking HELPS is NOT measured** on real data: the synthetic ablation only proves the keep/switch-off rule works |
| FR9 | results with reasons: CLI + web | E3 | `explain.py`, `web/app.py`, `web/static/index.html`, `cli.py` (`search`, `serve-web`) | `test_web.py` (6), `test_cli_index.py` | **built, tested on Linux** (reason per hit: which list, rank, matched terms, duplicates). Browser look never checked in a real browser on your machine |
| FR10 | one-command evaluation | E3 | `eval/beir.py`, `eval/metrics.py`, `scripts/evaluate_all.py` | `test_eval.py`, `test_beir.py` | **built**; run on SYNTHETIC data only (`docs/evidence/final-sandbox-SYNTHETIC.md`); real SciFact/SCIDOCS numbers pending |
| FR11 | late interaction, key papers (challenge) | E3 | `vectors/late.py`, `discover.py`, CLI `similar`, `key-papers`, `search --late` | `test_discover.py`, `test_cli_graph.py`, `bench/late_ablation.py` | **built, Linux**; the encoder is a hashed-trigram STAND-IN, not ColBERT; no measured gain (synthetic at ceiling -> verdict: switch off) |

## Non-functional

| Req | Target | Engine | Evidence so far | Status |
|---|---|---|---|---|
| NFR1 | 0 lost / 0 duplicated acknowledged papers across kills at every write step | E2 | 6 of 6 crash points x 2 occurrences x {plain, messy} streams = 24 runs, + 2 random-SIGKILL runs (6 rounds each), each followed by fsck and a full re-send; mutation check kills 13/15 recovery/ranking mutants (see PHASE1A.md for the 2 expected survivors) | **built for process kills on Linux**; "writer kill" under the pipeline = Phase 2; power loss untested |
| NFR2 | ingestion fault tolerance (readers killed every ~0.25 s, writer twice) | E1 | `test_chaos_readers_killed_every_quarter_second_and_writer_twice` (300-paper messy stream, >= 3 reader kills, 2 writer kills, index == ground truth, replacement < 5 s) | **built, Linux**; scale of the run is small (300 papers) |
| NFR3 | >= 70% parallel efficiency at 4 readers | E1 | `bench/pipeline_speedup.py`; sandbox (2 cores, SIMULATED 3 ms/paper CPU work) gave 1.79x at 2 readers = 0.89 efficiency (`docs/evidence/speedup-sandbox-2core-SIMULATED-burn3ms.json`) | **NOT demonstrated**: needs a >= 4-core run on your machine |
| NFR4 | p95 <= 500 ms at 50,000 papers | E3 | - | **not measured** (needs 50,000 real papers on your machine) |
| NFR5 | consistent read snapshots | E2 | `store/reader.py`, `store/pins.py`, `server/query_server.py` | `test_query_server.py::test_queries_during_ingestion_and_compaction_always_see_one_consistent_snapshot` | **built, tested on Linux** (2 reader processes vs a writer flushing every 15 papers with background merging: invariant live papers == last flushed seq held in every sample). A snapshot shows what the writer has FLUSHED, not its in-memory buffer |
| NFR6 | hybrid nDCG@10 >= better of keyword/vector on SciFact + SCIDOCS | E3 | `eval/beir.py` evaluates bm25 / vector / hybrid on one index and sets `nfr6_hybrid_at_least_best_single` | mechanics tested on a tiny dataset | **NOT measured**: needs the datasets and a real embedder (`beir --embedder st:...`) on your machine |
| NFR7 | codes <= 1/8 of float32, recall@10 >= 95% of exact | E3 + E2 | `bench/recall_vs_bits.py`; `docs/evidence/recall-bits-sandbox-SYNTHETIC*.json` | `test_vectors.py` (size, recall on clustered data) | **partial**: size OK (1-bit = 3.9% of float32). Recall on SYNTHETIC data with 20,000 vectors, pool 100: 1-bit 0.81 (FAILS), 2-bit 0.95, 3-bit 0.999; with pool 500: 1-bit 0.997. Default pool now scales (2.5% of vectors). **Real-vector numbers pending** |
| NFR8 | >= 3 s between arXiv requests, 0 calls over OpenAlex budget | E1 | spacing proven with a virtual clock **and** measured at the server with a real clock (`test_real_time_spacing_measured_by_the_server`); the CLI refuses < 3 s | **partial**: all-process `SharedLimiter` + `SharedBudget` built and tested (4 racing processes; pipeline test with 3 readers); real arXiv fetching through it not wired yet; OpenAlex budget use built in Phase 4 (see FR1/FR8 rows) |
| (Phase 1b) | C++ accelerator with Python twin | E3 | `native/`, `src/paperlens/native.py`, `scripts/build_native.py` | `test_native.py` (66 differential tests; skipped when not built), whole suite also run with `PAPERLENS_NATIVE=0` | **built + tested on Linux/g++ only**; **Windows/MSVC build never run**; 7 C++ mutants killed (`scripts/native_mutation_check.sh`) |
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
