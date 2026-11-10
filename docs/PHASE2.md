# Phase 2: multi-process ingestion pipeline (Engine E1)

Status: **built and tested on Linux (whole suite: 233 passed, evidence JSON in docs/evidence). Never run on Windows.** Date of the work: 2026-11-03.

## What it does and why (for a classmate)

Making a paper searchable has a slow part (tokenize, count words, make a vector) and a part that
must be done by exactly one process (writing the index safely). So:

* **Readers** (N processes) do the slow part in parallel.
* **One writer** owns the index; it batches up to 64 papers, writes the diary, does ONE fsync for the
  batch, and only then acknowledges them.
* **The supervisor** hands out jobs, forwards results, and keeps the books. A job is "done" only when
  the writer acknowledges it. If a process dies, the supervisor starts a replacement and re-sends the
  lost work. Re-sending is safe because storing the same `arxiv_id@datestamp` twice is ignored:
  *at-least-once delivery + idempotent writer = exactly-once storage*.

Key ideas you should be able to explain:

1. **Private pipes per worker incarnation.** Killing a process in the middle of writing to a shared
   queue can corrupt it. With private pipes a dead worker only breaks its own pipes.
2. **A sender thread per worker.** Pipe buffers are small (8 KB on Windows). If the supervisor
   blocked on sending to a busy worker while that worker blocked on sending to the supervisor, both
   would wait forever. The sender thread removes that risk.
3. **Blame only the head job.** When a reader dies, only the job it was working on (oldest unfinished)
   gets an attempt counted. Jobs queued behind it are innocent. A paper that keeps killing workers is
   parked in the dead-letter list after 3 attempts.
4. **Heartbeats mean progress, not just existence.** The heartbeat thread goes silent if the main
   thread has been stuck on one job longer than `job_timeout`; the supervisor then kills and replaces
   the worker.
5. **Shared-memory vectors with a checksum.** The supervisor picks the slot; the reader writes the
   vector there; the message carries only slot + crc32; the writer copies the bytes and verifies.
   A mismatch is rejected (`Ack -2`) and the job retried.
6. **One limiter and budget for all processes** (`shared_limiter.py`): reserve the next time slot under
   a lock, sleep outside it; the budget counter can never be overspent.
7. **Fault injection and chaos**: `Faults` (crash / raise / hang / corrupt vector / die at start),
   writer crash points injected through the same env vars as Phase 1, and a chaos mode that kills a
   random reader every 0.25 s and the writer twice.

## Files

| File | Role |
|---|---|
| `pipeline/messages.py` | message dataclasses (see CONTRACTS section 7) |
| `pipeline/supervisor.py` | scheduling, books, respawn, hang detection, chaos, shutdown |
| `pipeline/workers.py` | `reader_main`, `writer_main`, `Faults` |
| `pipeline/shm.py` | shared-memory arenas (resource-tracker handling) |
| `pipeline/shared_limiter.py` | `SharedLimiter`, `SharedBudget` |
| `pipeline/embed.py` | **stand-in** hash embedder (no real meaning; real model = Phase 3) |
| `testing/proc_helpers.py` | child functions for limiter tests |
| `bench/pipeline_speedup.py` | 1/2/4/8 readers benchmark |

## What was tested (verified by running, Linux, Python as in the sandbox)

* `tests/test_pipeline.py`: 22 tests. Plain run; result identical to single-process indexing (same
  scores, same order); messy stream (duplicates, newer and stale versions); re-run of the same input
  (all duplicates); reader killed once on a paper; poison paper parked after 3 attempts with exactly 3
  respawns; innocent neighbours not blamed; exception in reader does not kill it; paper with no text
  dead-lettered at once; hung reader detected and replaced; corrupt vector rejected then retried;
  start-up crash loop reported clearly; respawn limit; **writer killed at each of the 6 store crash
  points** (6 tests); shared limiter across 3 readers; **chaos** (readers killed every 0.25 s, writer
  killed twice, 300-paper messy stream, replacement time < 5 s); CLI `ingest`.
  Every end-to-end test finishes by comparing the index with the ground truth
  (`expected_final`) and running `fsck`.
* `tests/test_shared_limiter.py`: 6 tests, including 4 racing processes and a budget that is exactly
  spent, never overspent.
* **Mutation check** (8 deliberate bugs, all caught): ack before commit; skip crc check; blame every
  queued job; do not re-send to the new writer; no dead-lettering on retry path; limiter does not
  sleep; budget overspend; slots never freed. Two mutants first SURVIVED (a test aimed at the wrong
  code path; a limiter test that only looked at booked slots, not real sleeping). Both tests were
  fixed and the mutants then died. Script: `scripts/pipeline_mutation_check.sh` (POSIX shell; needs a clean git tree because it restores files with `git checkout -- src`).

## What is NOT tested / known limits

* **Windows**: nothing here has run on Windows. Risky spots: `connection.wait` on pipes plus process
  sentinels, `Process.kill`, shared memory lifetime, pipe buffer size. Run `python -m pytest tests\test_pipeline.py` first.
* **Parallel speed-up** (NFR3): the sandbox has 2 cores. A simulated-work run gave 0.89 efficiency at
  2 readers (JSON in `docs/evidence/`). The 4-reader >= 70% target must be measured on your machine
  with `bench/pipeline_speedup.py`. With the light real stand-in work, the single writer and fsync dominate and
  scaling will look poor; that is a property of the light workload.
* Scale: chaos test uses 300 papers; run `ingest --chaos` on thousands on your machine.
* Vectors are checked but **not persisted** (Phase 3). No query server yet (Phase 5).
* A worker stuck inside a C call that still heartbeats is not possible here, but a job that is merely
  slow (longer than `job_timeout`, default 10 s) will be treated as hung and retried.
* No power-loss test.

## Next step

Your Windows run (checklist in the hand-over message), then Phase 1b (C++ postings/BM25) once the
Windows report shows your compiler setup, or Phase 3 (semantic search) if you want to keep going on Python.
