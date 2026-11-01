# Phase 1a: keyword engine on disk, first half (report)

Date: 2026-11-01. Labels: **verified by running**, **from memory**, **needs checking**.
All runs on Linux, 2 cores, Python 3.13 (suite also on 3.10/3.11/3.12, see below).

## What was built

| Part | Engine | Module | One-line "what and why" (for the review) |
|---|---|---|---|
| Directory lock | E2 | `store/lock.py` | OS file lock on `<index>/LOCK`: only one writer, and the OS frees it if the writer dies, so no stale locks. |
| Varint postings codec | E3 | `text/varint.py` | Store doc ids as small gaps and each number in 1-2 bytes instead of 4-8. Pure-Python reference for the later C++ twin. |
| Segments | E2 | `store/segment.py` | A frozen folder of files for a batch of papers. Never edited, so it is crash-safe to publish and lock-free to read. `meta.json` (sizes + CRCs) is written last. |
| IndexStore | E2 | `store/index.py` | Diary + buffer + segments + MANIFEST. `put` logs, `commit` fsyncs (= acknowledged), `flush` freezes a segment and swaps the MANIFEST atomically. Opening always runs recovery. |
| Delete / replace | E2 | `store/index.py` | A newer version replaces an older one; a delete writes a tombstone into the next segment that applies to older segments only. |
| BM25 | E3 | `store/index.py` (`search`) | k1=1.2, b=0.75, Lucene idf; stats include deleted docs until compaction, like Lucene. |
| fsck | E2 | `store/fsck.py` | Independent auditor: re-checks every CRC and invariants (e.g. per document, sum of tf == doc length). |
| Crash harness | E2 | `store/crashpoints.py`, `testing/crash_child.py` | A child process that dies at a named step; it records what it was told is durable, so the test can check nothing acknowledged is lost. |
| CLI | shared | `cli.py` | `index`, `search`, `index-stats`, `fsck`. |
| Mutation check | shared | `scripts/mutation_check.py` | Breaks the code 15 ways; the tests must notice. |

## Tests: 184 passed, 0 failed (`docs/evidence/tests-20261101-3d3c60f-*.json`)

| file | tests | file | tests |
|---|---|---|---|
| test_crash | 34 | test_harvest | 24 |
| test_tokenizer | 19 | test_paper | 17 |
| test_varint | 17 | test_index | 15 |
| test_segment | 15 | test_diary | 11 |
| test_manifest | 10 | test_limiter | 6 |
| test_cli | 5 | test_cli_index | 4 |
| test_synth | 4 | test_lock | 3 |

Phase 1a added 88 tests (96 -> 184).

**"Done when" checks from the plan**

| Done when | Result |
|---|---|
| varint property tests round-trip random data | passed (5,000 random numbers; 300 random postings lists; every truncation point detected) |
| a flipped byte in a segment is detected | passed: fsck catches a one-bit flip in each of the 7 data files; opening refuses missing/short files |
| crash at each of the 6 points leaves 0 lost, 0 duplicated acknowledged papers | passed: 6 points x 2 occurrences x {plain, messy streams} = 24 real killed child processes. After each: fsck, no acknowledged paper missing, one live version per paper, then everything is re-sent and the final state equals the ground truth exactly. Every test asserts the child really died with the crash exit code, so an unwired point cannot pass silently |
| random SIGKILLs | passed: 2 runs x 6 rounds of `kill -9` at random times (>= 3 kills landed per run, asserted), resume with only unacknowledged papers, final full re-send equals ground truth |
| BM25 equals brute-force to 1e-9 across buffer + several segments | passed: fixed scenario (2 segments + buffer) and 4 random sequences of puts/duplicates/updates/stale versions/deletes/flushes/clean restarts, compared after every step. The oracle (`tests/oracle.py`) shares only the tokenizer with the store |
| delete/replace/duplicate semantics | passed (see `test_index.py`) |
| mutation tests of recovery | passed, see next section |

## Mutation check (`docs/evidence/mutation-20261101-2ac9219.json`): 13 of 15 killed

Killed (a test failed): recovery not replaying puts; recovery re-applying already-flushed
entries; orphan segments kept; torn tail not cut; MANIFEST published before the segment
rename; idempotency rules off; same-segment tombstone bug; BM25 with live-only N; wrong `b`;
avgdl ignoring the buffer; `last_flushed_seq` one too low; fsck skipping CRCs; writer lock
never taken.

Survivors, both expected:
* **M06** is a deliberately equivalent rewrite (a control): surviving is correct.
* **M07: removing `fsync` from `commit()` is NOT detected.** A killed process leaves its written
  bytes in the OS cache, so no process-kill test can tell. **Only a real power-loss test on
  real hardware can**, and we have not done one. Do not claim power-loss safety.

The check found one real gap in my own tests (M09: a document and a tombstone for the same key
in one segment), fixed with `test_document_and_tombstone_for_the_same_key_in_one_segment`, and
confirmed killed afterwards. Five test bugs of mine were found and fixed along the way (three
wrong expected counts, an over-strict "nothing acked" guard, and a SIGKILL test whose kills
landed in a no-op phase). None were store bugs.

## First profile (`docs/evidence/profile-20261101-2ac9219.json`) - READ THE WARNING

20,000 **synthetic** papers, 2 cores: ingest 6,264 papers/s (3.2 s), index 16 MB, search p50
12.6 ms / p95 24.7 ms (k=10). The synthetic text has ~60 distinct words, so postings lists are
unrealistically long; **these numbers say nothing about real performance and must not go in the
report as results.** What it does show: **about 57% of search time is varint decoding**
(`decode_uvarint` 34%, `decode_postings` 23%), then the scoring loop (15%) and the top-k heap (7%).
That supports the brief's first C++ candidate (postings codec + top-k), but the decision should be
re-made on real harvested data on your machine. NFR4 (p95 <= 500 ms at 50,000 papers) is not
measured on real data.

## NOT verified / not done (please read)

1. **No power-loss test.** (M07 above.) The word "crash-safe" currently means: survives a killed
   process on Linux, and the design follows the standard fsync ordering.
2. **Windows: nothing run.** Untested: `msvcrt.locking`, directory `rename` while mmaps are open,
   `os.replace` over an existing MANIFEST, `O_BINARY` diary, the SIGKILL-free crash tests. The
   SIGKILL and lock-holder-killed tests are skipped on Windows by design. Please run
   `python -m pytest` on a Windows machine and send me every failure.
3. **No quality numbers.** Nothing was measured on real arXiv data or BEIR (SciFact/SCIDOCS), so
   the "baseline nDCG recorded" item of Phase 1 is **not done**. It needs the real harvest and the
   BEIR download (my sandbox cannot reach either).
4. **Tokenizer is still the toy one.** No stemming; no evaluation of it.
5. **No `reindex` command, no `--explain` reasons**, no queries from a second process while a
   writer runs (read snapshots are Phase 5). The CLI `search` takes the writer lock.
6. **Memory use:** open loads every segment's `terms.idx` and key list into dicts and the
   `key -> location` map for all documents. Fine at 10^4-10^5 papers; untested beyond 20,000.
7. **A bad record in the middle of the newest diary file ends the log there** (everything after is
   cut). Fine for crashes (only the tail can tear) but bit-rot of acknowledged data is not
   recoverable; this is documented by `test_corrupt_acknowledged_diary_entry_in_the_middle...`.
8. **Not started from the Phase 1 list:** the BEIR runner and evaluation metrics (nDCG/Recall/MRR),
   and the rest of the profiling decision. The CI workflow is still never run (no repo yet).
9. Phase 0's open items (real arXiv harvest, OAI XML shape from memory) are unchanged.

## Questions for the team

* Q2 (tokenizer version) is built as proposed; E3 must bump `TOKENIZER_VERSION` when the tokenizer
  changes behaviour. OK?
* D7-D11 in CONTRACTS.md (decisions the brief did not specify): especially **D8** (a deleted
  paper may be re-added with the same key) and **D9** (repeated query words count once).
* Who owns E3's semantic+graph side and the harvester (still TO CONFIRM in OWNERS.md)?

## Next: second half of Phase 1

Evaluation metrics (nDCG@10, Recall@100, MRR) with tests against hand-computed values; BEIR
loader/runner for SciFact and SCIDOCS (you must download the datasets: my sandbox cannot);
`reindex`; a tokenizer v2 with measured effect; profile again on real data; then decide the
Phase 1b C++ target.
