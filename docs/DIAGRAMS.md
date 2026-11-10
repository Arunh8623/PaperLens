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

# Phase 2 additions (built)

## 8. Life of a job in the supervisor's books

```mermaid
stateDiagram-v2
    [*] --> pending
    pending --> in_reader: dispatched (least-loaded reader, slot chosen)
    in_reader --> in_writer: PaperReady received, forwarded to writer
    in_reader --> pending: reader died (head job: attempt+1) / JobFailed / hang kill
    in_writer --> done: Ack seq>=1 (stored) or -1 (duplicate/stale)
    in_writer --> pending: Ack -2 (bad vector) -> attempt+1
    in_writer --> in_writer: writer died -> PaperReady re-sent to the new writer
    pending --> dead_letter: attempts >= max (or permanent error)
    in_reader --> dead_letter: attempts >= max
    done --> [*]
    dead_letter --> [*]
```

## 9. One paper through the pipeline

```mermaid
sequenceDiagram
    participant S as Supervisor
    participant R as Reader (own pipes)
    participant M as Shared memory arena
    participant W as Writer (own pipes)
    participant D as Diary / index
    S->>R: PaperJob(job, paper, slot)
    R->>R: tokenize, count terms, make vector
    R->>M: write vector into slot
    R->>S: PaperReady(slot, crc, term counts)
    S->>W: PaperReady (job now in_writer)
    W->>M: copy bytes out of slot, check crc
    W->>D: put() (batch of up to 64)
    W->>D: ONE commit (fsync)
    W->>S: AckBatch (only now durable)
    S->>S: job done, slot freed
```

## 10. Architecture of the whole system (Phases 0-7)

```mermaid
flowchart LR
    A[arXiv OAI-PMH harvester<br/>1 request / 3 s] --> SUP
    O[OpenAlex client<br/>credit budget] --> G
    subgraph E1[E1 Process and Ingestion]
      SUP[Supervisor] --> R1[Reader 1] & R2[Reader N]
      R1 & R2 -- vectors via shared memory --> W
      SUP --> W[Single writer]
    end
    subgraph E2[E2 Persistent Index]
      W --> D[(Diary / WAL)]
      W --> SEG[(Immutable segments)]
      SEG --> MAN[(MANIFEST)]
      C[Compaction thread] --> SEG
      G[(Graph snapshot PLG1)] --> MAN
    end
    subgraph E3[E3 Retrieval and Ranking]
      QS[Query servers / IndexReader snapshots] --> MAN
      QS --> BM[BM25] & VQ[1-4 bit vectors + IVF + f16 rescore]
      BM & VQ --> RRF[RRF k=60] --> RR[optional graph / late rerank] --> DD[MinHash dedupe] --> EX[explain reasons]
    end
    EX --> WEB[FastAPI + one-page UI / CLI]
```

## 11. Stored files (ER view)

```mermaid
erDiagram
    INDEX ||--|| MANIFEST : has
    INDEX ||--o{ DIARY_FILE : has
    INDEX ||--o{ SEGMENT : lists
    INDEX ||--o| GRAPH_SNAPSHOT : lists
    SEGMENT ||--|| POSTINGS_BIN : contains
    SEGMENT ||--|| TERMS_IDX : contains
    SEGMENT ||--|| DOCS_BIN : contains
    SEGMENT ||--o| VECTORS : contains
    SEGMENT ||--|| META_JSON : "checksums of all"
    MANIFEST { int generation int last_flushed_seq list segments int graph_version int crc32 }
    DIARY_FILE ||--o{ RECORD : frames
    RECORD { u32 length u32 crc32 json payload }
```

## 12. Segment life cycle

```mermaid
stateDiagram-v2
    [*] --> tmp: written in .tmp-seg-N, every file fsynced
    tmp --> renamed: rename to seg-N (not yet listed)
    renamed --> live: MANIFEST lists it
    live --> merged_away: compaction MANIFEST drops it
    merged_away --> deleted: no reader pin holds it
    tmp --> [*]: crash -> recovery deletes .tmp-*
    renamed --> [*]: crash before MANIFEST -> unlisted, deleted on open
```

## 13. Query processing (activity)

```mermaid
flowchart TD
    Q[query + filters] --> P[pin MANIFEST snapshot]
    P --> T[tokenize] --> B[BM25 over segments + buffer]
    P --> E[embed query] --> V[IVF probe, 1-4 bit scan, f16 rescore]
    B --> F[RRF merge]
    V --> F
    F --> FL[category / year filter via zone maps + doc check]
    FL --> GR{graph rerank on?}
    GR -- yes --> PPR[personalized PageRank + RRF] --> DU
    GR -- no --> DU[drop near-duplicates by MinHash]
    DU --> RS[reasons per result] --> OUT[results]
```
