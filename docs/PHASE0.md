# Phase 0: Foundations (report)

Date: 2026-10-31. Everything below is labelled **verified by running**, **from memory**, or
**needs checking**.

## What was built

| Part | Engine | Module | One-line "what and why" (for the review) |
|---|---|---|---|
| PaperRecord + key rules + JSONL | shared | `contracts/paper.py` | One validated paper shape for everyone; `arxiv_id@datestamp` makes re-delivery harmless. |
| Diary framing + torn-tail recovery | E2 | `store/diary.py` | Write the change to a log and fsync before acknowledging; after a crash, trust the log up to the first bad record and cut the rest. |
| Atomic MANIFEST | E2 | `store/manifest.py` | One tiny checksummed file says what the index is; swapping it with `os.replace` publishes a change all-or-nothing. |
| Crash points | E2 | `store/crashpoints.py` | Lets tests kill the process at an exact step, so crash safety is proven rather than hoped. |
| Toy science tokenizer | E3 | `text/tokenizer.py` | Cuts text into words the same way for papers and queries, keeping `C++`, `k-means`, `O(n log n)`. |
| Request limiter | E1 | `harvest/limiter.py` | Reserve the next free time slot under a lock, sleep **outside** it: callers never overlap and never block each other. |
| arXiv harvester | E1 (supporting) | `harvest/arxiv_oai.py` | Downloads metadata politely (3.2 s spacing, Retry-After, contact in User-Agent), resumable, records saved before progress is recorded. |
| Fake OAI server | E1 (test tool) | `harvest/fake_oai_server.py` | A local pretend arXiv that can fail on purpose (429, 503, dropped connection, garbled XML, expired token). |
| Synthetic data | shared | `synth.py` | Seeded fake papers incl. duplicates/updates/stale versions for plumbing tests. Says nothing about search quality. |
| CLI | shared | `cli.py` | `synth`, `harvest`, `stats`. Harvest refuses to run faster than arXiv's rule. |
| Evidence + CI | shared | `scripts/`, `.github/workflows/ci.yml` | Dated JSON of test results tied to a git commit. |

## Tests

**96 tests, 0 failed** (`docs/evidence/tests-20261031-*.json`): paper 17, diary 11,
manifest 10, tokenizer 19, synth 4, limiter 6, harvest 24, cli 5.

Verified by running, all on **Linux** (2 cores): full suite on Python **3.10, 3.11, 3.12, 3.13**;
a fresh `git clone` + new 3.10 venv + `pip install -e ".[dev]"` + `pytest` (96 passed); the
installed `paperlens` command.

**Mutation checks** (deliberately broke the code, confirmed a test fails, restored):

| Broken on purpose | Caught by |
|---|---|
| diary ignores CRC mismatch | `test_flip_every_byte_stops_at_that_record` |
| diary never truncates the torn tail | `test_recover_tail_truncates_and_appends_cleanly` |
| diary accepts short payloads | `test_flip_every_byte_stops_at_that_record` (the CRC check also guards this, so the truncation test alone would not notice) |
| MANIFEST skips its checksum | 2 tests (`..byte_flip..`, `..corrupt_file_on_disk..`) |
| tokenizer stops stripping accents / drops `C++` | the matching table rows |
| harvester ignores Retry-After | `test_429_retry_after_is_honoured` |
| limiter stops spacing requests | `test_full_harvest_pages_and_spacing` |
| restart ignores the newest datestamp | `test_expired_token_restarts_from_newest_datestamp` |
| state saved before the data | `test_state_never_claims_more_than_the_file_holds` |

## arXiv facts checked today (2026-10-31)

* Base URL `https://oaipmh.arxiv.org/oai`; the old `export.arxiv.org/oai2` was replaced in March 2025. ([OAI help](https://info.arxiv.org/help/oa/index.html))
* Terms of Use: no more than one request every three seconds, one connection at a time, across all your machines combined; they name OAI-PMH among the APIs covered; metadata is CC0. ([API terms](https://info.arxiv.org/help/api/tou.html))
* A User-Agent contact email and `Retry-After` handling are **not stated** by arXiv. We do both anyway (our policy).
* `ListSets` (fetched live) shows a bare `cs` set exists, plus `cs:cs` and 40 subject sets such as `cs:cs:AI`. So `set=cs` is valid.
* Datestamp = last modification time, not submission date; tokens expire daily; incremental harvest = `from`, no `until`. ([OAI help](https://info.arxiv.org/help/oa/index.html))
* Their bulk-data page mentions bursts of 4 requests/second for "custom programmatic harvesting". That conflicts with the 3 s rule; we follow the stricter one. ([bulk data](https://info.arxiv.org/help/bulk_data.html))

## NOT verified / not done (please read)

1. **No real arXiv harvest yet.** My sandbox's shell cannot reach arXiv (proxy refuses it), and my page-fetch tool dropped the query string of a single-record request. So **~10,000 real records is not done**: you run the command in the README.
2. **The `arXiv` metadata XML format is from memory.** The parser is namespace-agnostic and tolerant, but it has only been tested against my own fake server, which is built from the same memory. The first real run might expose a mismatch (skipped records are counted in `skipped_bad` and the first few are printed). If it does, send me the first page's raw XML and I will fix it.
3. **Windows: nothing was run.** The `O_BINARY` newline test passes trivially on Linux; its real value is when it runs on Windows. Also untested on Windows: `os.replace` atomicity assumptions, directory-fsync being ignored, the spawn tests (none yet).
4. **GitHub Actions workflow** is written, never run (no repo yet).
5. **Only 2 of the 6 crash points are wired** into code (`after_wal_write`, `after_wal_fsync`); the 4 flush points need the Phase 1 flush.
6. **Limiter covers threads in one process only.** The all-process version (lock + shared memory) is Phase 2. Known caveat: a caller that wakes late can end up closer to the next one, hence the 3.2 s margin.
7. **Power-loss safety is not claimed.** Our crash tests kill processes; they cannot simulate losing power.
8. **Mermaid diagrams are unrendered**; the full diagram set for the review is not yet produced.
9. **Datestamp order** from arXiv is assumed non-decreasing (Q3 in CONTRACTS.md); the real run will report `out_of_order`.

## Gate 0 checklist for the team

- [ ] All four read `docs/CONTRACTS.md`, answer Q1-Q3, tick the sign-off lines (separate commits)
- [ ] Confirm `OWNERS.md` (E3 semantic+graph owner; who owns the harvester)
- [ ] Create the GitHub repo, push, check the Actions run on Windows + Linux
- [ ] Run `python -m pytest` on a Windows machine; send me any failure
- [ ] Run the real harvest (README); send the printed summary + `.state.json` + the **first page's raw XML** if anything was skipped
- [ ] Each owner reads and can explain their modules without notes (see "what and why" above)

## Next: first half of Phase 1

Directory lock + the 4 remaining crash points; varint postings codec; segment writer/reader
with `meta.json` CRCs; atomic flush in the agreed order; recovery on open; brute-force BM25
oracle vs. real BM25 to 1e-9; delete/replace semantics; random-SIGKILL test (POSIX) and
mutation tests of recovery.
