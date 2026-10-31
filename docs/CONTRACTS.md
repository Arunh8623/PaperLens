# PaperLens contracts (v0.1, Phase 0)

This is the agreement between the three engines. If code and this file disagree,
that is a bug in one of them: fix it and note it in the commit message.

**Status labels:** `BUILT` = code exists and has passing tests. `PLANNED` = agreed
shape, no code yet. Nothing marked PLANNED has been run.

---

## 1. PaperRecord `BUILT` (`contracts/paper.py`)

| field | type | rule |
|---|---|---|
| arxiv_id | str | non-empty, must NOT contain `@` (old-style ids like `hep-th/9901001` are fine) |
| datestamp | str | OAI datestamp `YYYY-MM-DD` = when arXiv last modified the record (**not** the submission date) |
| title, abstract | str | strings (abstract may be empty; the harvester parser rejects an empty title) |
| authors, categories | list of str | stored as tuples; categories like `cs.IR` |
| created, updated | str | `YYYY-MM-DD` or empty |
| doi | str | may be empty |
| synthetic | bool | `true` only for generated test data |

Canonical JSON: sorted keys, no spaces, UTF-8 kept as-is. Unknown or missing fields are rejected.

**Idempotency key** = `arxiv_id@datestamp`. For one `arxiv_id`:

| incoming vs stored newest datestamp | outcome | stored result |
|---|---|---|
| no stored version | `insert` | new paper |
| equal | `duplicate` | ignored (Ack seq = -1) |
| newer | `replace` | old version tombstoned, new stored |
| older | `stale` | ignored |

`decide()` implements this table `BUILT`. The index applying it is Phase 1 `PLANNED`.

**JSONL**: one canonical JSON record per line, UTF-8, `\n` newlines on every OS (we write
with `newline="\n"` so Windows never makes `\r\n`) `BUILT`.

## 2. Diary (write-ahead log) `BUILT` framing, `PLANNED` payload

File name: `wal/wal-<first_seq, zero-padded to 20 digits>.log` (padding so names sort
correctly as text; see decision D1).

Record, little endian:

```
[u32 length][u32 crc32(payload)][payload bytes]
```

* `put()` appends. `commit()` = one `fsync` (group commit). A paper is **acknowledged
  only after commit**.
* A crash can only damage the **end** of the file. Scan stops at the first record that is
  short, has an absurd length (> 64 MiB), or fails its CRC; `recover_tail()` truncates
  there and fsyncs.
* Open with `O_BINARY` on Windows or `0x0A` becomes `0x0D0A` and the log is corrupted.
* **Payload `PLANNED` (Phase 1)**: UTF-8 JSON
  `{"seq": int, "op": "put" | "delete", "key": str, "paper": {...} | null}`.
  `seq` is positive, strictly increasing, assigned by the single writer.

## 3. MANIFEST `BUILT` (`store/manifest.py`)

```json
{"body": {"generation": 3, "last_flushed_seq": 120,
          "segments": ["seg-000001", "seg-000002"],
          "graph_version": 0, "schema": 1},
 "crc32": 123456789}
```

`crc32` is over the canonical JSON of `body` (sorted keys, no spaces). Publish order:
write `MANIFEST.tmp` -> fsync -> `os.replace` -> fsync directory. Missing file = new index.
Bad checksum = `ManifestError` (we refuse to guess). A leftover `MANIFEST.tmp` is junk.

## 4. Segment files `PLANNED` (Phase 1; vector files Phase 3)

Immutable directory `segments/seg-NNNNNN/`, built first as `segments/.tmp-seg-N/`, every
file fsynced, directory fsynced, renamed, parent fsynced, **then** listed in a new MANIFEST.

| file | content |
|---|---|
| postings.bin | per term: varint doc-gaps + term frequencies |
| terms.idx | sorted term -> offset, document frequency |
| keys.txt | idempotency key per local doc id |
| doclens.bin | token count per doc |
| docs.bin + docs.off | stored papers + offsets |
| deletes.json | tombstones from the batch |
| meta.json | counts + crc32 of every file, **written last** |
| *(later)* vectors.q, vectors.f16, IVF file, zonemap, token vectors | Phase 3-6 |

## 5. Recovery on every open `PLANNED` (Phase 1)

1. Read MANIFEST. 2. Delete `.tmp-*` and any `seg-*` not listed. 3. Replay diary entries
with `seq > last_flushed_seq`. 4. Cut a torn diary tail at the first bad record. 5. fsync.

Single writer: OS lock on `<index>/LOCK` (`fcntl.flock` / `msvcrt.locking`).

## 6. Crash points (fault injection) `BUILT` mechanism, partly wired

`PAPERLENS_CRASH_AT=<name>` (+ optional `PAPERLENS_CRASH_AFTER=N`) makes the process
`os._exit(77)` at that point, with no cleanup.

| name | wired into code today? |
|---|---|
| after_wal_write | yes (diary.put) |
| after_wal_fsync | yes (diary.commit) |
| flush_after_tmp_written, flush_after_segment_rename, flush_after_manifest, flush_after_wal_delete | **no**: flush does not exist until Phase 1 |

## 7. Pipeline messages `PLANNED` (Phase 2, E1)

`PaperJob(job_id, paper, slot)`, `PaperReady(job_id, reader_id, key, paper, term_freqs,
doc_len, shm_name, shm_slot, vec_dim, vec_crc)`, `Ack(job_id, key, seq)` with
`seq >= 1` stored, `-1` duplicate, `-2` rejected (retry).

## 8. Tokenizer `BUILT` toy (`text/tokenizer.py`)

`tokenize(text) -> list[str]` is the **only** way text becomes terms, at index time and at
query time. Keeps `c++`, `c#`, `k-means`, `o(nlogn)`; emits the parts of compounds
(`k-means` -> `k-means`, `means`); NFKC + lowercase + accent stripping; small stopword list;
no stemming. `paper_text(title, abstract)` repeats the title once. `term_freqs()` returns
`({term: count}, doc_len)` so readers pre-compute what the writer stores.

## 9. Harvester output `BUILT` (`harvest/arxiv_oai.py`)

* JSONL of PaperRecords. **At-least-once**: a crash can re-fetch a page, so the file may hold
  duplicate keys; consumers must rely on the idempotency key (they do).
* State file `<out>.state.json`: `set, from_date, token, newest_datestamp, lines_written,
  pages, restarts, skipped_bad, deleted_seen, out_of_order, done`.
* Rule: records are fsynced **before** the state file advances (tested).

## 10. Platform rules (learned the hard way)

Windows: `O_BINARY`; directory fsync unsupported (ignored); close files and mmaps before
deleting; `msvcrt.locking` for the lock. Everywhere: `spawn` start method, worker functions
importable from the package, `if __name__ == "__main__":` guards. SIGKILL tests are
POSIX-only. Power-loss safety is **not** claimed anywhere until tested on real hardware.

---

## Decisions the brief did not specify (please review at sign-off)

| # | Decision | Why |
|---|---|---|
| D1 | WAL file names zero-pad `first_seq` to 20 digits | text sort order = numeric order |
| D2 | Request spacing is **3.2 s**, not 3.0 s | arXiv's rule is 3 s; the margin covers a caller waking up late |
| D3 | `title` is repeated once in the searchable text | simple title boost; Phase 1 may revisit with evidence |
| D4 | Compound tokens emit the whole word **and** its parts | queries for either form match |
| D5 | Harvester rejects empty titles; `PaperRecord` itself does not | keeps the contract permissive, the parser strict |
| D6 | The contact email for the User-Agent has **no default** | arXiv does not require it (their pages are silent); it is our policy, and it must be a real address |

## Open questions needing a team decision

* **Q1 Deleted records.** OAI marks withdrawn papers `status="deleted"`. Today we count and
  skip them. Should they become `delete` ops (tombstones) in the index?
* **Q2 Tokenizer version.** Changing the tokenizer invalidates an existing index. Propose:
  store a `tokenizer_version` in each segment's `meta.json`. Needs E2 + E3 agreement.
* **Q3 Datestamp order.** The restart-after-expired-token rule assumes arXiv returns records in
  non-decreasing datestamp order. Unverified; the real run reports `out_of_order`.

## Sign-off

Each member ticks by committing this file with their line changed to `[x]` (the commit is
the dated evidence).

- [ ] Aditya Gavimath (E2)
- [ ] Kartik Hegde (E3 keyword side)
- [ ] Arun Hadimani (E1)
- [ ] Prajwal B (E1)  *(ownership of E3 semantic+graph side still to be confirmed, see OWNERS.md)*
