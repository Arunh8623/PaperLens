# Phase 1b: optional C++ accelerator (Engine E3, keyword + vector scan)

## What it does and why (for a classmate)
Profiling (`docs/evidence/profile-20261101-2ac9219.json`, synthetic data) showed **57% of search time inside the Python varint decoder**
(`decode_uvarint` 34%, `decode_postings` 23%). The brief says "move things to C++ only when a profile shows the need", so the first
targets are exactly those, plus one vector hot loop:

| C++ function (`native/src/paperlens_native.cpp`) | Python twin (`src/paperlens/native.py`) | used by |
|---|---|---|
| `decode_postings(buf, pos, count)` | `decode_postings_py` (= `text/varint.py`) | `Segment.postings` (not fsck: fsck forces the Python decoder as an independent check) |
| `bm25_accumulate(postings, ..., out)` decodes AND adds every document's BM25 contribution into a dense array, skipping deleted/filtered docs | `bm25_accumulate_py` | `IndexStore.search` (the search loop now works on dense per-segment arrays for BOTH paths) |
| `level_dot(packed, q, dim, bits)` unpacks 1-4 bit codes and dots them with the rotated query without building an (n x dim) float matrix | `level_dot_py` (= `unpack_levels @ q`) | `vectors/quant.estimate` |

**Rules kept:** every function has a pure-Python twin with the same arguments; if the extension is missing (or `PAPERLENS_NATIVE=0`) the Python
version runs and a clear warning is printed once; tests compare both on the same inputs; speed-ups are reported honestly.
Floating point: C++ is compiled with `-ffp-contract=off` (MSVC `/fp:precise`) and the BM25 expression has the same operation order as Python, so
BM25 scores are **bit-identical** (tests use `==`, not "close"). `level_dot` sums in double and differs from numpy's float32 matmul by ~1e-6, so
vector estimates are compared with a tolerance and could in theory reorder two near-tied candidates.

**Build:** `python scripts/build_native.py` (CMake + pybind11; see README for the exact Windows and Linux commands). It is deliberately NOT part of
`pip install -e .` so a missing compiler can never break the normal install.

## Verified by running (Linux, g++, Python 3.13): full suite 467 passed, 0 failed (`docs/evidence/tests-20261109-f63268d-py3.13.18-linux.json`)
* 66 differential tests: random postings lists (incl. 64-bit values, offsets, bytearray/memoryview inputs), 7 kinds of corrupt input (same exception
  type AND message), BM25 accumulation with/without dead and filter masks on 8 seeds x 4 combos (`np.array_equal` on the doubles), out-of-range docs,
  shape validation, `level_dot` for bits 1-4 and dims 8/13/64/384, and **end-to-end**: `IndexStore.search` results (keys and scores) identical with the
  extension on/off across 7 queries x 3 filters x 4 values of k, with deleted documents, several segments and an unflushed buffer.
* Whole suite also passes with `PAPERLENS_NATIVE=0` (the differential tests skip themselves).
* Missing-extension path: warning text and Python fallback tested by simulating a failed import.
* Mutation check on the C++ itself (`scripts/native_mutation_check.sh`, rebuilds per mutant): 7 mutants (operation order, dead/filter masks, gap off-by-one,
  bit order, overlong varint accepted, zero tf accepted). First run: 6 killed, 1 survived (zero-tf accepted inside `bm25_accumulate`); a test was
  added; second run: 7/7 killed (N7 killed by an abort of the test process, which still counts as a failure).
* Benchmark `docs/evidence/native-speedup-sandbox-2core-SYNTHETIC.json` (SYNTHETIC 20,000 papers, 2-core Linux sandbox, 8 queries x 15 rounds):
  search 49.1 ms -> 1.2 ms per batch of 8 queries (x40, p50; identical results in that run); postings decode of a 3,948-entry list 2.0 ms -> 0.13 ms;
  1-bit scan of 50,000 x 384-d codes 149 ms -> 31 ms (x4.8). SQLite FTS5 (stdlib, own tokenizer/ranking, different top-10 lists) took 15.8 ms for the same 8 queries.
  **Do not quote these as real-arXiv numbers**: synthetic text has a tiny vocabulary. Run `bench/native_speedup.py` on your machine.

## NOT done / NOT tested
* **Windows/MSVC build never run.** CMake flags for MSVC are written from knowledge, not executed. `.pyd` output placement is untested.
* **Block-Max WAND is not implemented.** Our postings carry no per-block upper bounds; WAND needs a format change (block metadata in `postings.bin`) and is only worth it
  after a profile on real data shows exhaustive scoring is the bottleneck. Current C++ scoring is exhaustive (term-at-a-time), which is why it is exact.
* **No popcount/SIMD scan with a quantised query**: it would change results relative to the Python reference and needs a recall experiment first. `level_dot` is a plain scalar loop.
* **Pyserini/Lucene not compared** (needs Java, not available here).
* MinHash stays in Python (no profile shows a need yet).
* Multi-thread or SIMD speed-ups: none; single-threaded C++ only.

## Next step
Phase 6 (late-interaction rerank, key papers, more-like-this), then Phase 7 (web UI, final evaluation, demo, report).
