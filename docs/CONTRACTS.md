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

## 2. Diary (write-ahead log) `BUILT`

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
* **Payload `BUILT` (Phase 1)**: UTF-8 JSON
  `{"seq": int, "op": "put" | "delete", "key": str, "paper": {...} | null}`.
  `seq` is positive, strictly increasing, assigned by the single writer. Only ACCEPTED
  operations are logged: a duplicate or stale `put` never reaches the diary. Term counts are
  NOT logged; replay recomputes them with the tokenizer (so a tokenizer change needs a reindex, see Q2).
* A diary file is deleted only after a MANIFEST with `last_flushed_seq >=` all its entries
  is durable; the new diary file is created first, then the old ones removed.

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

### 3a. Citation-graph snapshot `BUILT` (Phase 4; `store/graphfile.py`, `graph/csr.py`)

`graphs/graph-<version:06d>.bin` = graph bytes + u32 crc32 trailer. Graph bytes: magic `PLG1`, u32 n, u32 m,
u32 offsets[n+1], u32 targets[m], u32 length + UTF-8 JSON list of the n arXiv ids (CSR: the papers a node cites are
`targets[offsets[i]:offsets[i+1]]`). Little endian. Publish order: write `.tmp-graph-N.bin` -> fsync -> rename ->
fsync dir -> MANIFEST `graph_version = N` -> delete other files in `graphs/`. Recovery deletes every file in `graphs/`
except the one the MANIFEST names. `graph_version = 0` means no graph.

OpenAlex fetch output (`citations --out`): JSONL `{"arxiv_id", "openalex_id" (null if unknown to OpenAlex), "refs": [OpenAlex work ids]}`;
appended and fsynced per request batch; a torn last line is ignored on resume.

## 4. Segment files `BUILT` (`store/segment.py`; vector files since Phase 3)

Immutable directory `segments/seg-NNNNNN/`, built first as `segments/.tmp-seg-NNNNNN/`, every
file fsynced, directory fsynced, renamed, parent fsynced, **then** listed in a new MANIFEST.

| file | content (exact format) |
|---|---|
| postings.bin | per term (sorted): `df` pairs of varint(doc gap), varint(tf); first gap measured from -1 |
| terms.idx | text lines `term<TAB>offset<TAB>df`, sorted by term (the byte length is implied by `df`: decode exactly `df` pairs) |
| keys.txt | one idempotency key per line; line n = local doc id n |
| doclens.bin | u32 little-endian token count per doc |
| docs.bin + docs.off | canonical-JSON papers concatenated; u64 little-endian offsets, n+1 entries |
| deletes.json | sorted list of keys killed by this batch; applies to **older** segments only |
| meta.json | format, `tokenizer_version`, counts, size + crc32 of every file above; **written last** |
| attrs.json | `{"years":[...],"cats":[[...],...]}`, one entry per doc (for filters). Always written since Phase 3; Phase 1 segments lack it and the store derives it from the stored papers |
| vectors.f16 | *(only if some doc has a vector)* little-endian float16 `[n x dim]`, unit-length rows, all-zero row = "no vector". The copy used to **re-score** candidates exactly |
| vectors.q | *(same condition)* `codes [n x ceil(dim*bits/8)] u8` (bit-packed, `bits` in 1..4) , `d_o f32[n]`, `inv f32[n]`, `oc f32[n]`, `valid u8[n]`, `centroid f32[dim]`. Meaning of the numbers: `vectors/quant.py` |
| ivf.bin | *(only if >= `ivf_min_docs` valid vectors)* `centroids f32[nlist x dim]`, `offsets u32[nlist+1]`, `order u32[n_valid]` |
| meta.json extras | `zonemap {year_min, year_max, categories}` and `vectors {dim, bits, qversion, rot_seed, embedder, valid, ivf_nlist}` |
| *(later)* token vectors for late interaction | Phase 6 |

Opening a segment checks every file size against meta.json; **fsck** also checks every CRC.
A segment built with a different `tokenizer_version` is refused (answers Q2 for now).

### 4a. Vectors in the diary `BUILT` (Phase 3)

A `put` entry may carry `"vec": base64(float16 bytes of the unit vector)`. A paper and its vector are
therefore one diary record: they are both durable or both absent after a crash, and replay restores
both. The first vector fixes the index's dimension; a later vector of another length is refused
*before* anything is logged. A paper without a vector is allowed (it is simply invisible to vector
search). Quantization parameters of a segment live in its meta.json, so segments of different bit
widths can coexist. The rotation matrix is regenerated from `(dim, rot_seed)` with Python's own
`random` module (stable stream) plus a QR step: a numpy upgrade does not silently change old codes
(fsck re-derives the codes and tolerates up to 0.1% of bits differing from float rounding).

## 5. Recovery on every open `BUILT` (`store/index.py`)

1. Take the writer lock. 2. Read MANIFEST (bad checksum = refuse, `IndexCorrupt`). 3. Delete
`MANIFEST.tmp`, `.tmp-*` and any `seg-*` not listed. 4. Open listed segments (a missing one is
`IndexCorrupt`). 5. Rebuild live/dead state from keys + tombstones. 6. Replay diary entries with
`seq > last_flushed_seq`; cut a torn tail only in the **newest** diary file (damage in an older
one is `IndexCorrupt`). 7. Drop diary files whose entries are all flushed; fsync.

Single writer: OS lock on `<index>/LOCK` (`fcntl.flock` / `msvcrt.locking`) `BUILT`, Linux-tested only.

Delete/replace semantics `BUILT`: killing a live document that sits in a segment writes its key
to the `deletes.json` of the NEXT segment. A key may legitimately reappear (delete then re-put);
the tombstone kills only copies in strictly older segments.

BM25 `BUILT`: k1=1.2, b=0.75, `idf = ln(1 + (N - df + 0.5)/(df + 0.5))`. `N`, `df` and
`avgdl` include deleted documents until compaction (Lucene behaviour); deleted documents are never
returned. Duplicate query words count once. Ties break by key. Verified against an independent
brute-force model to 1e-9.

## 6. Crash points (fault injection) `BUILT`, all 6 wired

`PAPERLENS_CRASH_AT=<name>` (+ optional `PAPERLENS_CRASH_AFTER=N`) makes the process
`os._exit(77)` at that point, with no cleanup.

| name | where |
|---|---|
| after_wal_write | `DiaryWriter.put` |
| after_wal_fsync | `DiaryWriter.commit` |
| flush_after_tmp_written | all segment files written + fsynced in `.tmp-seg-N` |
| flush_after_segment_rename | `.tmp-seg-N` renamed to `seg-N` (not yet in MANIFEST) |
| flush_after_manifest | new MANIFEST published |
| flush_after_wal_delete | new diary started, old diary files deleted |
| graph_after_file | (Phase 4, `GRAPH_POINTS`, not in `KNOWN_POINTS`) `graphs/graph-N.bin` renamed into place, MANIFEST not yet updated |
| graph_after_manifest | (Phase 4) MANIFEST now says `graph_version = N`, older graph files not yet deleted |

Each test asserts the child really exited with code 77, so an unwired point fails the test.

### 6a. Compaction, reader pins, snapshots `BUILT` (Phase 5)

Compaction crash points (`COMPACT_POINTS`, not in `KNOWN_POINTS`): `compact_after_tmp_written` (merged segment
written under `.tmp-seg-N`), `compact_after_segment_rename` (renamed to `seg-N`, MANIFEST still lists the old
run), `compact_after_manifest` (MANIFEST lists `seg-N` in place of the run; old folders not yet deleted),
`compact_after_old_deleted`. Recovery handles every case with the existing rules (delete `.tmp-*` and unlisted
`seg-*`), except that an unlisted `seg-*` is KEPT while a live reader pin lists it.
Only neighbouring segments are merged (a segment's tombstones apply to the segments before it). A tombstone in the
merged group survives only if its key exists in a segment before the group; a merge that includes the first segment
therefore drops all tombstones. Merged vectors are re-encoded from the half-precision copy.

`pins/pin-<pid>-<random>.json`: `{"pid", "generation", "segments": [names]}`, padded with spaces. The reader holds an OS
lock on the file (Windows: on a byte at offset 2^30, so the content stays readable). A pin whose lock can be taken belongs to
a dead process and is deleted. A reader writes its pin BEFORE re-reading the MANIFEST to confirm its snapshot.
Query-server messages (`server/query_server.py`), one private pipe per worker: `("search", {query, k, mode, cat, year_from,
year_to, graph_rerank})`, `("stats", {})`, `("refresh", {})`, `("ping", {})`, `("stop", {})`; answers `("ok", {generation,
last_flushed_seq, hits[], took_ms})`. Every answer names the generation it was computed from.

## 7. Pipeline messages `BUILT` (Phase 2, E1; `pipeline/messages.py`)

`PaperJob(job_id, paper, slot, attempt)`, `PaperReady(job_id, reader_id, key, paper, term_freqs,
doc_len, shm_name, shm_slot, vec_dim, vec_crc, attempt)`, `Ack(job_id, key, seq, outcome, vec_crc)`
with `seq >= 1` stored, `-1` duplicate or stale (already known), `-2` rejected (retry). Extra
fields beyond the brief: `attempt` (retry count), `outcome` (text: insert / replace / duplicate /
stale / `bad_vector: ...` / `permanent: ...`), `vec_crc` (what the writer actually read).
Also: `AckBatch`, `Heartbeat`, `Ready`, `JobFailed(job_id, error, permanent)`, `WriterDone`, `Stop`.

Rules (all tested, see PHASE2.md): one private pair of pipes per worker incarnation; the
supervisor picks the shared-memory slot; a slot is freed only when its job is done or re-queued;
the writer acknowledges only after `commit()`; a job is "done" only on an acknowledgement.
Vectors are verified (crc) but **not yet stored** by the writer: persisting them is Phase 3.

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
| D7 | Only accepted operations are written to the diary (duplicate/stale never are) | keeps the diary replay-deterministic and small |
| D8 | A deleted key may be re-put (outcome `insert`) | after a delete there is no live version, so the idempotency rule has nothing to compare with |
| D9 | Query words are de-duplicated before scoring | simplest well-defined behaviour; Lucene counts repeats |
| D10 | `terms.idx` is a text file loaded into a dict at open | fine at our scale; an FST/block index is a possible later improvement, not a requirement |
| D11 | `fsck` takes the writer lock | never reads a half-written state; reports "locked" instead of guessing |

## Open questions needing a team decision

* **Q1 Deleted records.** OAI marks withdrawn papers `status="deleted"`. Today we count and
  skip them. Should they become `delete` ops (tombstones) in the index?
* **Q2 Tokenizer version.** Changing the tokenizer invalidates an existing index. **Built as
  proposed**: `tokenizer_version` is in every `meta.json` and a mismatch is refused. Open: E3 must
  bump `TOKENIZER_VERSION` in `store/segment.py` whenever `text/tokenizer.py` changes behaviour, and
  we still need a `reindex` command (Phase 1 second half). Needs E2 + E3 agreement.
* **Q3 Datestamp order.** The restart-after-expired-token rule assumes arXiv returns records in
  non-decreasing datestamp order. Unverified; the real run reports `out_of_order`.

## Sign-off

Each member ticks by committing this file with their line changed to `[x]` (the commit is
the dated evidence).

- [ ] Aditya Gavimath (E2)
- [ ] Kartik Hegde (E3 keyword side)
- [ ] Arun Hadimani (E1)
- [ ] Prajwal B (E1)  *(ownership of E3 semantic+graph side still to be confirmed, see OWNERS.md)*
