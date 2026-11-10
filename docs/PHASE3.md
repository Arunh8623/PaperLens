# Phase 3: semantic search, quantized vectors, hybrid ranking (Engine E3 semantic side + E2 storage)

Status: **built and tested on Linux with a stand-in embedder.** Not run on Windows. Real embedding
models, real arXiv data and BEIR numbers are NOT done (they need your machine). Date: 2026-11-08.

## What it does and why (for a classmate)

* **A meaning vector per paper.** An *embedder* (a pre-trained model, we do not train) turns
  "title. abstract" into e.g. 384 numbers; similar meaning = similar vector. Spec strings pick it:
  `st:<model>` (sentence-transformers) or `hash:<dim>` (a **meaningless** stand-in used by our tests).
* **Stored in the same crash-safe way as the paper.** The vector rides in the same diary record
  (half precision), and the segment gets `vectors.f16` (exact copy for re-scoring) and `vectors.q`
  (tiny codes for fast scanning) in the same atomic flush. After a crash a paper and its vector are
  both there or both absent (tested at all 6 crash points, 24 tests).
* **Quantization (RaBitQ-style, `vectors/quant.py`).** Keep 1-4 bits per number instead of 32.
  1-bit codes for 384 numbers are 48 bytes (+12 bytes of correction numbers) instead of 1536.
  Scanning codes is cheap but slightly wrong, so we take a pool of candidates by code and
  **re-score** them with the exact half-precision copy. Compression buys memory and speed, **not disk**:
  we keep the half-precision copy (768 bytes/vector at 384 dims), so disk per vector is about 828 bytes.
* **IVF (`vectors/ivf.py`).** For big segments, cluster the vectors; a query scans only the `nprobe`
  nearest clusters. Faster, slightly lossy; probing all clusters equals a full scan (tested).
* **Filters (`store/filters.py`).** Category (any-of) and year range, applied *inside* the search, using
  per-segment `attrs.json` and a zone map (min/max year, category set) that skips whole segments.
* **Hybrid (`vectors/fusion.py`).** Reciprocal Rank Fusion with k = 60 over the keyword list and the
  vector list; each hit records its rank in each list, which the CLI prints as the reason.
* **Pipeline.** Readers embed with the chosen embedder; the vector goes through shared memory with a
  crc; the writer verifies it and stores it with the paper (so shared memory is now carrying real
  payload). A wrong dimension or an embedder that disagrees with the config stops the run with a clear
  message instead of retrying forever.

## Files

`vectors/{embedders,quant,ivf,fusion}.py`, `store/filters.py`, changes in `store/{segment,index,fsck,reindex}.py`,
`pipeline/{workers,supervisor,messages}.py`, `eval/beir.py`, `cli.py`, `bench/recall_vs_bits.py`,
`scripts/vector_mutation_check.sh`.

## Verified by running (Linux)

* `tests/test_vectors.py` (49 tests): rotation orthogonal + deterministic; bit packing round trip for
  bits 1-4 and odd dims; estimator error shrinks with more bits and is nearly unbiased; exact search ==
  independent brute-force oracle across buffer + several segments + replacements + deletes; quantized
  search with a covering pool == exact; every bit width through a real segment; reopen from segments
  and from the diary; replace/delete remove the old vector; bad vectors refused before logging; mixing
  vector-less and vector segments; IVF; fsck catches a flipped byte in every vector file **and** a
  swapped-vectors corruption whose checksum was fixed up (only the re-derivation of codes catches it);
  filters == brute force for BM25 and vectors, zone-map skipping and inclusive boundaries, filters on
  Phase 1 segments; RRF hand-computed, hybrid == RRF of the two lists; reindex keeps vectors.
* `tests/test_vector_crash.py` (24): real killed child processes at the 6 crash points x 2 data kinds;
  afterwards every surviving paper has exactly its own vector, and a re-send finishes the job.
* `tests/test_pipeline.py` now also checks that **every** stored vector equals what the embedder makes
  for that paper version, including under chaos (readers killed every 0.25 s, writer killed twice).
* **Mutation check** (`scripts/vector_mutation_check.sh`): 13 deliberate bugs, all killed (vector not
  logged, dead docs in vector search, filter ignored in either search, estimator correction removed, RRF off by
  one, fsck vector check removed, vectors dropped on replay or on flush, writer forgets the vector, IVF list
  truncated, zone map off by one, dimension check removed). One mutant (zone-map boundary) first
  survived; a boundary test was added and it then died.

## Measured (SYNTHETIC data only, this sandbox)

`bench/recall_vs_bits.py`, 20,000 clustered synthetic 384-d vectors, 100 queries, recall@10 against
exact (JSON in `docs/evidence/`):

| bits | bytes/vec | % of float32 | recall (pool 100) | recall (pool 500) |
|---|---|---|---|---|
| 1 | 60 | 3.9% | **0.809** (below the 0.95 target) | 0.997 |
| 2 | 108 | 7.0% | 0.947 | 1.000 |
| 3 | 156 | 10.2% | 0.999 | 1.000 |
| 4 | 204 | 13.3% (above 1/8) | 1.000 | 1.000 |

Reading: with a small candidate pool the 1-bit codes are not enough; with a pool of 2.5% of the vectors
they are. The store's default pool therefore scales with the index (`default_pool`). This says nothing
about real embeddings: re-run with `--index` on your real vectors for the report.

## NOT done / NOT tested / honest limits

* **Windows**: not run. numpy + mmap + file deletion are new on Windows paths; run the tests first.
* **Real embedding models** (`st:...`, incl. the SPECTER2-vs-general comparison asked for in the brief):
  code path exists, **never executed here** (no model download in the sandbox); model names are from memory.
  SPECTER2 normally needs the `adapters` package; `st:allenai/specter2_base` loads only the base model.
* **Quality** (NFR6): no real dataset has been run. Hybrid >= best single is a target, not a result.
* **Latency** (NFR4): not measured at 50,000 papers. Scans are numpy (chunked); the C++ popcount scan
  from the brief is **not built** (Phase 1b not done; no profile yet says it is needed).
* **RaBitQ fidelity**: this is the idea written from memory, **not** the paper's algorithm: no error bound,
  no optimised multi-bit search ("Extended RaBitQ"), a single centroid per segment (also with IVF), and a
  simple uniform scalar quantizer for 2-4 bits. Check against the paper before claiming more.
* Vectors are not yet merged or dropped by compaction (Phase 5); deleted vectors stay on disk until then.
* A paper without a vector (put without one) is invisible to vector search; nothing warns about it.
* Not done: late interaction (Phase 6), citation rerank (Phase 4).

## Next step

Your Windows run, then real-data runs: harvest, index with `st:` embedder, `recall_vs_bits --index`,
BEIR with `--embedder`. After that Phase 4 (citation graph; needs an OpenAlex key) or Phase 5 (compaction).
