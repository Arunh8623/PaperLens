# Diagrams (Phase 0)

Mermaid source. **Not rendered or checked by me** (no renderer in my sandbox): GitHub shows
them automatically; if one fails to render, tell me. Diagrams marked *planned* show design
that is not built yet. The full set for the review (engine-wise, ER of stored files, state
diagrams, query activity) is produced as each part is built.

## 1. What exists after Phase 0

```mermaid
flowchart LR
    A[arXiv OAI-PMH<br/>oaipmh.arxiv.org] -->|1 request / 3.2 s| H[harvest/arxiv_oai.py]
    L[limiter.py<br/>reserve slot, sleep outside lock] --> H
    H -->|fsync records, THEN state| J[(papers.jsonl<br/>+ state.json)]
    S[synth.py<br/>seeded fake papers] --> J2[(synth.jsonl)]
    J --> P[PaperRecord<br/>key = arxiv_id@datestamp]
    J2 --> P
    P -.planned Phase 1.-> W[store: diary -> segments -> MANIFEST]
    T[tokenizer.py] -.planned.-> W
```

## 2. Diary record on disk (built)

```mermaid
flowchart LR
    subgraph record
      A["u32 length<br/>(4 bytes, little endian)"] --- B["u32 crc32(payload)<br/>(4 bytes)"] --- C["payload<br/>(length bytes)"]
    end
    record --> record2[next record ...]
```

Scan rule: stop at the first record that is short, too long (> 64 MiB) or fails its CRC;
everything before it is trusted, everything from it onward is cut.

## 3. Publishing a MANIFEST (built) and a flush (planned)

```mermaid
sequenceDiagram
    participant W as Writer
    participant FS as Disk
    Note over W,FS: PLANNED flush: segment first
    W->>FS: write segments/.tmp-seg-N/* , fsync each, fsync dir
    W->>FS: rename .tmp-seg-N -> seg-N, fsync parent
    Note over W,FS: BUILT: atomic MANIFEST swap
    W->>FS: write MANIFEST.tmp
    W->>FS: fsync MANIFEST.tmp
    W->>FS: os.replace(MANIFEST.tmp, MANIFEST)   (atomic)
    W->>FS: fsync directory
    Note over W,FS: PLANNED: start new diary, delete old diary files
```

## 4. Harvester loop (built)

```mermaid
flowchart TD
    S([start or resume]) --> Q{state done?}
    Q -- yes --> E([stop])
    Q -- no --> R[wait for limiter slot]
    R --> G[GET ListRecords<br/>token, or set+from]
    G -->|429/503| B[wait Retry-After, or backoff] --> R
    G -->|network error / bad XML| B
    G -->|badResumptionToken| X[restart from newest datestamp] --> R
    G -->|OK| W[append records, fsync]
    W --> V[save state atomically]
    V --> D{more pages?}
    D -- yes --> R
    D -- no --> E
```

---

# Phase 1 additions (built)

Mermaid source, **not rendered by me**. If one fails to render on GitHub, tell me.

## 5. Life of a paper inside the store

```mermaid
stateDiagram-v2
    [*] --> Decided: put(paper)
    Decided --> Ignored: duplicate or stale (never logged)
    Decided --> Logged: insert or replace, record appended to diary
    Logged --> Acknowledged: commit() = fsync
    Acknowledged --> InBuffer: already searchable (buffer)
    InBuffer --> InSegment: flush()
    InBuffer --> Gone: delete() or replaced before flush
    InSegment --> Dead: delete() or replaced (tombstone in the NEXT segment)
    Dead --> [*]: removed physically only by compaction (Phase 5)
```

## 6. Flush and what a crash at each step leaves behind

```mermaid
sequenceDiagram
    participant S as IndexStore
    participant D as Disk
    S->>D: commit diary (fsync)
    S->>D: write all files into segments/.tmp-seg-N, fsync each + dir
    Note over S,D: crash here (flush_after_tmp_written): .tmp dir is junk, removed at open, diary replays
    S->>D: rename to seg-N, fsync parent
    Note over S,D: crash here (flush_after_segment_rename): seg-N unlisted = orphan, removed at open, diary replays
    S->>D: write MANIFEST (tmp, fsync, os.replace, fsync dir)
    Note over S,D: crash here (flush_after_manifest): segment is live, old diary entries skipped (seq <= last_flushed_seq)
    S->>D: create new diary file, delete old ones
    Note over S,D: crash here (flush_after_wal_delete): fully done
```

## 7. Opening an index (recovery)

```mermaid
flowchart TD
    A[open] --> L{writer lock free?}
    L -- no --> X1([IndexLocked])
    L -- yes --> M{MANIFEST checksum ok?}
    M -- no --> X2([IndexCorrupt: refuse, do not guess])
    M -- yes --> C[delete .tmp-* and unlisted seg-*]
    C --> O{all listed segments present and valid?}
    O -- no --> X2
    O -- yes --> R[rebuild live/dead state from keys + tombstones]
    R --> P[replay diary entries with seq > last_flushed_seq]
    P --> T[cut torn tail of the newest diary file]
    T --> W[append to diary, drop fully flushed diary files]
    W --> Z([ready])
```
