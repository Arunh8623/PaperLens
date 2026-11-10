"""IndexStore: diary + buffer + immutable segments + MANIFEST + recovery + BM25.

What it does and why (plain language)
-------------------------------------
This is the engine's heart. Life of a paper:

  put()    -> decide (new? duplicate? newer version? older?) -> write to the
              diary -> add to the in-memory buffer
  commit() -> fsync the diary. From here on the paper is "acknowledged": it
              survives a crash. (A duplicate/stale paper is never logged.)
  flush()  -> freeze the buffer into an immutable segment, publish it with an
              atomic MANIFEST swap, then start a fresh diary.

Opening an index ALWAYS runs recovery (so a crash needs no special handling):
  1. take the single-writer lock          2. read MANIFEST (checksummed)
  3. delete half-built (.tmp-*) and unlisted segments
  4. open the listed segments             5. replay diary entries newer than the
  MANIFEST's last_flushed_seq, cutting a torn tail at the first bad record.

Flush order (each arrow is a crash-safe step; crash points in [brackets]):
  write all files into .tmp-seg-N, fsync each + dir  [flush_after_tmp_written]
  -> rename to seg-N, fsync parent                   [flush_after_segment_rename]
  -> write new MANIFEST (atomic)                     [flush_after_manifest]
  -> start new diary file, delete old ones           [flush_after_wal_delete]
A crash at any arrow leaves a state recovery understands: before the MANIFEST
swap the new segment is an unlisted orphan (deleted at next open, diary
replays); after it the segment is live and old diary entries are skipped
because seq <= last_flushed_seq.

Deletes: a delete (or the replacement of an old version) kills the live copy.
If the copy sits in a segment we cannot edit it, so the key is written to the
`deletes.json` of the NEXT flushed segment, and applies to all OLDER segments.

Vectors (Phase 3): put(paper, vector=...) also stores the paper's meaning vector.
It is logged in the diary as half-precision bytes (so a crash loses nothing), kept in
the buffer, and written into the segment (vectors.f16 + quantized vectors.q, optional
ivf.bin) in the SAME atomic flush. search_vector() scans the cheap quantized codes of
every segment to find candidates and then RE-SCORES them with the half-precision
copy; search_hybrid() fuses BM25 and vector ranks with Reciprocal Rank Fusion.

Ranking: BM25, k1=1.2, b=0.75, Lucene-style idf ln(1+(N-df+0.5)/(df+0.5)).
LIKE LUCENE, N, df and avgdl still count deleted documents until compaction
(Phase 5); deleted documents are never returned. Query terms are de-duplicated
(a repeated query word does not count twice).
"""
from __future__ import annotations

import base64
import heapq
import json
import math
import os
import shutil
import threading
from typing import NamedTuple

import numpy as np

from ..contracts.paper import PaperRecord, decide, split_key
from ..text.tokenizer import paper_text, term_freqs, tokenize
from ..vectors.fusion import rrf
from ..vectors.quant import unit
from .filters import paper_year
from .crashpoints import crashpoint
from .diary import DiaryWriter, list_wals, recover_tail, scan_file, wal_name
from .fsutil import fsync_dir, retry_fs
from .lock import DirLock
from .manifest import Manifest, ManifestError, read_manifest, write_manifest
from .compaction import merge_run, plan_run
from .graphfile import clean_graph_dir, load_graph, publish_graph
from .pins import pinned_segments
from .segment import (Segment, SegmentError, publish_segment, segment_name,
                      segment_number, write_segment_tmp)

K1, B = 1.2, 0.75


class IndexCorrupt(Exception):
    """The index is damaged in a way recovery must not guess about."""


class PutResult(NamedTuple):
    outcome: str      # insert | replace | duplicate | stale
    seq: int          # diary sequence number, 0 if nothing was logged


class Hit(NamedTuple):
    score: float
    key: str
    arxiv_id: str


class HybridHit(NamedTuple):
    score: float                 # fused RRF score
    key: str
    arxiv_id: str
    bm25_rank: object            # 1-based rank in the keyword list, or None
    vec_rank: object             # 1-based rank in the vector list, or None
    bm25_score: object
    vec_score: object


class _BufDoc(NamedTuple):
    paper: PaperRecord
    tf: dict
    dl: int
    vec: object = None           # half-precision bytes or None


class IndexStore:
    def __init__(self, path, flush_docs: int = 1000, allow_tokenizer_mismatch: bool = False,
                 vec_bits: int = 1, ivf_min_docs: int = 4096, embedder_name: str = ""):
        self.path = os.fspath(path)
        self.flush_docs = flush_docs
        self.vec_bits = vec_bits                 # quantization bits for NEW segments (each segment records its own)
        self.ivf_min_docs = ivf_min_docs         # segments with >= this many vectors get an IVF index (0 = never)
        self.embedder_name = embedder_name       # recorded in segment meta so search knows how to embed queries
        self.vec_dim = 0                         # 0 = this index holds no vectors (yet)
        self._buf_ver = 0
        self._buf_vec_cache = None
        self._mu = threading.RLock()     # guards in-memory state; compaction builds files OUTSIDE it
        self._reserved = 0               # highest segment number handed out (a compaction may be building one)
        self._compacting = False
        self._bg = None
        # True only for READING an old index so it can be reindexed; search results would be wrong.
        self.allow_tokenizer_mismatch = allow_tokenizer_mismatch
        self.seg_root = os.path.join(self.path, "segments")
        self.wal_dir = os.path.join(self.path, "wal")
        self.segments: list = []
        self.key_loc: dict = {}          # key -> [(segment index, local id)] oldest first
        self.dead: list = []             # per segment: set of dead local ids
        self.live: dict = {}             # arxiv_id -> key of its live version
        self.buf: dict = {}              # key -> _BufDoc (insertion ordered)
        self.buf_inv: dict = {}          # term -> {key: tf}
        self.pending_tomb: set = set()   # keys killed in segments since the last flush
        self.seq = 0
        self.recovery: dict = {}
        self._writer = None
        self._lock = DirLock(self._ensure_dir())
        self._lock.acquire()
        try:
            self._recover()
        except BaseException:
            self._release_everything()
            raise

    def _ensure_dir(self):
        os.makedirs(self.path, exist_ok=True)
        return self.path

    # ------------------------------------------------------------ recovery
    def _recover(self) -> None:
        os.makedirs(self.seg_root, exist_ok=True)
        os.makedirs(self.wal_dir, exist_ok=True)
        stats = {"removed_tmp": 0, "removed_unlisted": 0, "replayed": 0,
                 "torn_bytes_cut": 0, "old_wals_removed": 0}
        try:
            self.manifest = read_manifest(self.path)
        except ManifestError as e:
            raise IndexCorrupt(str(e)) from e
        stale = os.path.join(self.path, "MANIFEST.tmp")
        if os.path.exists(stale):
            retry_fs(os.remove, stale)
        listed = set(self.manifest.segments)
        pinned = pinned_segments(self.path)          # readers may still be using an obsolete segment
        for name in sorted(os.listdir(self.seg_root)):
            if name.startswith(".tmp-"):
                retry_fs(shutil.rmtree, os.path.join(self.seg_root, name))
                stats["removed_tmp"] += 1
            elif name.startswith("seg-") and name not in listed and pinned is not None and name not in pinned:
                retry_fs(shutil.rmtree, os.path.join(self.seg_root, name))
                stats["removed_unlisted"] += 1
        fsync_dir(self.seg_root)
        stats["removed_graph_files"] = clean_graph_dir(self.path, self.manifest.graph_version)
        for name in self.manifest.segments:
            path = os.path.join(self.seg_root, name)
            if not os.path.isdir(path):
                raise IndexCorrupt(f"MANIFEST lists {name} but it is missing")
            try:
                self.segments.append(Segment.open(path, check_tokenizer=not self.allow_tokenizer_mismatch))
            except SegmentError as e:
                raise IndexCorrupt(str(e)) from e
        for seg in self.segments:
            if seg.has_vectors:
                if self.vec_dim and seg.vec_dim != self.vec_dim:
                    raise IndexCorrupt(f"{seg.name}: vector dimension {seg.vec_dim} differs from {self.vec_dim}")
                self.vec_dim = seg.vec_dim
                self.embedder_name = self.embedder_name or seg.meta["vectors"].get("embedder", "")
        self._rebuild_from_segments()
        self.seq = self.manifest.last_flushed_seq
        self._replay_diary(stats)
        self.recovery = stats
        self.graph = load_graph(self.path)          # None until a graph has been published

    def _locate(self) -> None:
        """Recompute key_loc and dead from self.segments (+ kills not yet flushed as tombstones)."""
        self.key_loc = {}
        self.dead = [set() for _ in self.segments]
        for si, seg in enumerate(self.segments):
            for local, key in enumerate(seg.keys):
                self.key_loc.setdefault(key, []).append((si, local))
        for sj, seg in enumerate(self.segments):          # tombstones hit OLDER segments only
            for key in seg.tombstones:
                for si, local in self.key_loc.get(key, ()):
                    if si < sj:
                        self.dead[si].add(local)
        for key in self.pending_tomb:                     # killed since the last flush (tombstone not written yet)
            for si, local in reversed(self.key_loc.get(key, ())):
                if local not in self.dead[si]:
                    self.dead[si].add(local)
                    break

    def _rebuild_from_segments(self) -> None:
        self._locate()
        for si, seg in enumerate(self.segments):
            for local, key in enumerate(seg.keys):
                if local not in self.dead[si]:
                    self.live[split_key(key)[0]] = key

    def _replay_diary(self, stats) -> None:
        flushed = self.manifest.last_flushed_seq
        wals = list_wals(self.wal_dir)
        max_seq_in = {}
        for i, (_first, path) in enumerate(wals):
            is_last = i == len(wals) - 1
            res = recover_tail(path) if is_last else scan_file(path)
            if res.torn:
                if not is_last:
                    raise IndexCorrupt(f"damaged diary file that is not the newest: {path}")
                stats["torn_bytes_cut"] += res.file_bytes - res.good_bytes
            mx = 0
            for payload in res.payloads:
                try:
                    e = json.loads(payload.decode("utf-8"))
                    seq, op, key = e["seq"], e["op"], e["key"]
                except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as ex:
                    raise IndexCorrupt(f"{path}: bad diary entry ({ex})") from ex
                mx = max(mx, seq)
                if seq <= flushed:
                    continue                               # already inside a segment
                if seq <= self.seq:
                    raise IndexCorrupt(f"{path}: diary sequence goes backwards at {seq}")
                if op == "put":
                    vec = None
                    if e.get("vec"):
                        vec = base64.b64decode(e["vec"])
                        if len(vec) % 2 or (self.vec_dim and len(vec) // 2 != self.vec_dim):
                            raise IndexCorrupt(f"{path}: diary vector has a bad length")
                        self.vec_dim = self.vec_dim or len(vec) // 2
                    self._apply_put(PaperRecord.from_dict(e["paper"]), vec=vec)
                elif op == "delete":
                    self._apply_delete(key)
                else:
                    raise IndexCorrupt(f"{path}: unknown op {op!r}")
                self.seq = seq
                stats["replayed"] += 1
            max_seq_in[path] = mx
        # choose the diary file to append to; drop files whose entries are all flushed
        active = None
        if wals and max_seq_in[wals[-1][1]] > flushed:
            active = wals[-1][1]
        else:
            active = os.path.join(self.wal_dir, wal_name(self.seq + 1))
        for _first, path in wals:
            if path != active and max_seq_in[path] <= flushed:
                retry_fs(os.remove, path)
                stats["old_wals_removed"] += 1
        self._writer = DiaryWriter(active)
        fsync_dir(self.wal_dir)

    # ------------------------------------------------------------ state changes
    def _apply_put(self, paper: PaperRecord, precomputed=None, vec=None) -> None:
        old = self.live.get(paper.arxiv_id)
        if old is not None:
            self._kill(old)
        tf, dl = precomputed if precomputed is not None else term_freqs(paper_text(paper.title, paper.abstract))
        self._buf_ver += 1
        self.buf[paper.key] = _BufDoc(paper, tf, dl, vec)
        for term, c in tf.items():
            self.buf_inv.setdefault(term, {})[paper.key] = c
        self.live[paper.arxiv_id] = paper.key

    def _apply_delete(self, key: str) -> None:
        if self.live.get(split_key(key)[0]) == key:
            self._kill(key)
            del self.live[split_key(key)[0]]

    def _kill(self, key: str) -> None:
        """Remove the LIVE copy of `key` (buffer first, else newest live segment copy)."""
        doc = self.buf.pop(key, None)
        if doc is not None:
            self._buf_ver += 1
            for term in doc.tf:
                postings = self.buf_inv[term]
                del postings[key]
                if not postings:
                    del self.buf_inv[term]
            return
        for si, local in reversed(self.key_loc.get(key, ())):
            if local not in self.dead[si]:
                self.dead[si].add(local)
                self.pending_tomb.add(key)
                return

    # ------------------------------------------------------------ public write API
    def _vector_to_f16(self, vector) -> bytes:
        v = np.asarray(vector, dtype=np.float32)
        if v.ndim != 1 or v.size < 2:
            raise ValueError("vector must be a 1-D sequence of numbers")
        if self.vec_dim and v.size != self.vec_dim:
            raise ValueError(f"vector has {v.size} numbers but this index uses {self.vec_dim}")
        return unit(v).astype("<f2").tobytes()          # stored unit-length: cosine = dot product

    def put(self, paper: PaperRecord, precomputed=None, vector=None) -> PutResult:
        """precomputed=(term_freqs, doc_len) lets a reader process do the tokenizing so the
        single writer does not (pipeline, Phase 2). It is checked, never trusted blindly.
        Crash replay always recomputes from the paper text.
        vector=sequence of floats (the paper's meaning vector): normalised to unit length and
        stored as half precision; the first vector fixes this index's dimension."""
        vec = self._vector_to_f16(vector) if vector is not None else None
        if precomputed is not None:
            tf, dl = precomputed
            if (not tf or sum(tf.values()) != dl or any(c < 1 for c in tf.values())):
                raise ValueError("precomputed term counts are inconsistent (sum(tf) != doc_len)")
        cur = self.live.get(paper.arxiv_id)
        outcome = decide(split_key(cur)[1] if cur else None, paper.datestamp)
        if outcome in ("duplicate", "stale"):
            return PutResult(outcome, 0)
        seq = self.seq + 1
        entry = {"seq": seq, "op": "put", "key": paper.key, "paper": paper.to_dict()}
        if vec is not None:
            entry["vec"] = base64.b64encode(vec).decode("ascii")
        self._log(entry)
        self.seq = seq
        if vec is not None:
            self.vec_dim = self.vec_dim or len(vec) // 2
        self._apply_put(paper, precomputed, vec)
        return PutResult(outcome, seq)

    def delete(self, arxiv_id: str) -> bool:
        key = self.live.get(arxiv_id)
        if key is None:
            return False
        seq = self.seq + 1
        self._log({"seq": seq, "op": "delete", "key": key, "paper": None})
        self.seq = seq
        self._apply_delete(key)
        return True

    def _log(self, entry: dict) -> None:
        self._writer.put(json.dumps(entry, sort_keys=True, separators=(",", ":")).encode("utf-8"))

    def commit(self) -> None:
        """fsync the diary: everything put()/delete()d so far is now acknowledged."""
        self._writer.commit()
        if len(self.buf) >= self.flush_docs:
            self.flush()

    # ------------------------------------------------------------ flush
    def flush(self):
        """Freeze the buffer into a segment. Returns the segment name, or None if nothing to do."""
        if not self.buf and not self.pending_tomb:
            return None
        self._writer.commit()                              # nothing unlogged may be flushed
        name = segment_name(self._next_segment_number())
        docs = [(k, d.paper, d.tf, d.dl, d.vec) for k, d in self.buf.items()]
        tomb = sorted(self.pending_tomb)
        vec_cfg = None
        if self.vec_dim and any(d.vec is not None for d in self.buf.values()):
            vec_cfg = {"dim": self.vec_dim, "bits": self.vec_bits, "ivf_min_docs": self.ivf_min_docs,
                       "embedder": self.embedder_name}
        tmp = write_segment_tmp(self.seg_root, name, docs, tomb, vec_cfg)
        crashpoint("flush_after_tmp_written")
        publish_segment(self.seg_root, tmp, name)
        crashpoint("flush_after_segment_rename")
        write_manifest(self.path, self.manifest.next(
            last_flushed_seq=self.seq, segments=self.manifest.segments + (name,)))
        self.manifest = read_manifest(self.path)
        crashpoint("flush_after_manifest")
        # in-memory state follows the disk
        seg = Segment.open(os.path.join(self.seg_root, name))
        si = len(self.segments)
        self.segments.append(seg)
        self.dead.append(set())
        for local, key in enumerate(seg.keys):
            self.key_loc.setdefault(key, []).append((si, local))
        self.buf.clear()
        self.buf_inv.clear()
        self.pending_tomb.clear()
        # new diary file first, then delete the old ones
        self._writer.close()
        new_path = os.path.join(self.wal_dir, wal_name(self.seq + 1))
        self._writer = DiaryWriter(new_path)
        for _first, path in list_wals(self.wal_dir):
            if path != new_path:
                retry_fs(os.remove, path)
        fsync_dir(self.wal_dir)
        crashpoint("flush_after_wal_delete")
        return name

    def publish_graph(self, graph) -> int:
        """Publish a new citation-graph snapshot through the MANIFEST (see store/graphfile.py)."""
        version = publish_graph(self.path, graph)
        self.manifest = read_manifest(self.path)
        self.graph = graph
        return version

    # ------------------------------------------------------------ compaction
    def compact(self, force: bool = False, fan_in: int = 4, ratio: float = 4.0, dead_ratio: float = 0.5):
        """Merge neighbouring segments (see store/compaction.py). Returns a dict describing what was
        done, or None when the policy found nothing to do. Safe to call from a background thread while
        this process keeps writing and searching: the new segment is BUILT without holding the state
        lock; only the publish step (MANIFEST swap + in-memory update) holds it.

        Publish order (every step is a crash point, see CONTRACTS section 6):
          write .tmp-seg-N -> fsync -> rename to seg-N -> write MANIFEST (run replaced by seg-N)
          -> close + delete the obsolete folders (unless a reader process still pins them)."""
        with self._mu:
            if self._compacting or self._writer is None:
                return None
            run = plan_run([s.num_docs for s in self.segments], [len(d) for d in self.dead],
                           fan_in=fan_in, ratio=ratio, dead_ratio=dead_ratio, force=force)
            if run is None:
                if any(n.startswith("seg-") and n not in self.manifest.segments for n in os.listdir(self.seg_root)):
                    self._gc_segments()                       # obsolete folders a reader used to pin
                return None
            i, j = run
            self._compacting = True
            segs = self.segments[i:j + 1]
            old_names = [sg.name for sg in segs]
            dead_sets = [set(self.dead[k]) for k in range(i, j + 1)]
            older_keys = {key for sg in self.segments[:i] for key in sg.keys}
            name = segment_name(self._next_segment_number())
            physical_before = sum(sg.num_docs for sg in segs)
        try:
            docs, tomb, vec_cfg = merge_run(segs, dead_sets, older_keys, self.ivf_min_docs, self.vec_bits)
            new_name = None
            if docs or tomb:
                tmp = write_segment_tmp(self.seg_root, name, docs, tomb, vec_cfg)
                crashpoint("compact_after_tmp_written")
                publish_segment(self.seg_root, tmp, name)
                new_name = name
            else:
                crashpoint("compact_after_tmp_written")
            crashpoint("compact_after_segment_rename")
            with self._mu:
                cur = [sg.name for sg in self.segments]
                if cur[i:j + 1] != old_names:
                    raise IndexCorrupt("segments changed under compaction")        # cannot happen: only compaction reorders
                names = cur[:i] + ([new_name] if new_name else []) + cur[j + 1:]
                write_manifest(self.path, self.manifest.next(segments=names))
                self.manifest = read_manifest(self.path)
                crashpoint("compact_after_manifest")
                new_seg = Segment.open(os.path.join(self.seg_root, new_name)) if new_name else None
                self.segments = self.segments[:i] + ([new_seg] if new_seg else []) + self.segments[j + 1:]
                self._locate()
                for sg in segs:
                    sg.close()
                removed = self._gc_segments()
                crashpoint("compact_after_old_deleted")
            return {"merged": old_names, "into": new_name, "docs_before": physical_before, "docs_after": len(docs),
                    "dropped": physical_before - len(docs), "tombstones_kept": len(tomb), "folders_removed": removed}
        finally:
            with self._mu:
                self._compacting = False

    def _gc_segments(self) -> int:
        """Delete segment folders the MANIFEST no longer lists, except those a live reader pins."""
        pinned = pinned_segments(self.path)
        if pinned is None:
            return 0
        listed, removed = set(self.manifest.segments), 0
        for name in sorted(os.listdir(self.seg_root)):
            if name.startswith("seg-") and name not in listed and name not in pinned:
                try:
                    retry_fs(shutil.rmtree, os.path.join(self.seg_root, name))
                    removed += 1
                except OSError:
                    pass                                      # still in use (Windows): next compaction retries
        if removed:
            fsync_dir(self.seg_root)
        return removed

    def start_background_compaction(self, interval: float = 0.5, **policy) -> None:
        """A thread that calls compact(**policy) every `interval` seconds (FR4)."""
        if self._bg is not None:
            return
        self._bg = BackgroundCompactor(self, interval, policy)
        self._bg.start()

    def stop_background_compaction(self) -> None:
        if self._bg is not None:
            self._bg.stop()
            self._bg = None

    def _next_segment_number(self) -> int:
        nums = [segment_number(n) for n in self.manifest.segments]
        nums += [segment_number(n) for n in os.listdir(self.seg_root)]
        n = max([n for n in nums if n is not None] + [self._reserved, 0]) + 1
        self._reserved = n
        return n

    # ------------------------------------------------------------ reads
    def get(self, arxiv_id: str):
        key = self.live.get(arxiv_id)
        if key is None:
            return None
        if key in self.buf:
            return self.buf[key].paper
        for si, local in reversed(self.key_loc[key]):
            if local not in self.dead[si]:
                return self.segments[si].paper(local)
        raise IndexCorrupt(f"live key {key} has no live copy")

    def live_keys(self) -> set:
        return set(self.live.values())

    def get_vector(self, arxiv_id: str):
        """The stored (half-precision, unit) vector of the live version as float32, or None."""
        key = self.live.get(arxiv_id)
        if key is None:
            return None
        if key in self.buf:
            v = self.buf[key].vec
            return None if v is None else np.frombuffer(v, dtype="<f2").astype(np.float32)
        for si, local in reversed(self.key_loc[key]):
            if local not in self.dead[si]:
                seg = self.segments[si]
                if not seg.has_vectors or not seg.valid_vectors()[local]:
                    return None
                return seg.vector_rows([local])[0]
        return None

    def stats(self) -> dict:
        return {
            "live_docs": len(self.live),
            "physical_docs": sum(s.num_docs for s in self.segments) + len(self.buf),
            "segments": len(self.segments),
            "buffer_docs": len(self.buf),
            "pending_tombstones": len(self.pending_tomb),
            "last_seq": self.seq,
            "generation": self.manifest.generation,
            "last_flushed_seq": self.manifest.last_flushed_seq,
            "vector_dim": self.vec_dim,
            "embedder": self.embedder_name,
            "recovery": dict(self.recovery),
        }

    def _buf_ok(self, doc, flt) -> bool:
        return flt is None or flt.empty or flt.matches(paper_year(doc.paper.created, doc.paper.datestamp),
                                                       doc.paper.categories)

    def search(self, query: str, k=10, flt=None) -> list:
        """BM25 over buffer + all segments. k=None returns every matching live doc.
        flt (store.filters.Filter) keeps only papers passing the category/year filter."""
        terms = list(dict.fromkeys(tokenize(query)))
        n_docs = sum(s.num_docs for s in self.segments) + len(self.buf)
        if not terms or n_docs == 0:
            return []
        total_len = sum(s.total_doclen for s in self.segments) + sum(d.dl for d in self.buf.values())
        avgdl = total_len / n_docs
        masks = [s.filter_mask(flt) for s in self.segments]
        dead_masks = []                                   # uint8 flags per segment (None when nothing is dead)
        for si, seg in enumerate(self.segments):
            if self.dead[si]:
                dm = np.zeros(seg.num_docs, dtype=np.uint8)
                dm[list(self.dead[si])] = 1
                dead_masks.append(dm)
            else:
                dead_masks.append(None)
        accs: list = [None] * len(self.segments)          # dense float64 score array per segment, allocated on first hit
        scores: dict = {}
        for term in terms:
            df = sum(s.df(term) for s in self.segments) + len(self.buf_inv.get(term, ()))
            if df == 0:
                continue
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            for si, seg in enumerate(self.segments):
                if seg.df(term) == 0:
                    continue
                if accs[si] is None:
                    accs[si] = np.zeros(seg.num_docs, dtype=np.float64)
                seg.bm25_accumulate(term, idf, avgdl, accs[si], dead_masks[si], masks[si], K1, B)
            for key, tf in self.buf_inv.get(term, {}).items():
                if not self._buf_ok(self.buf[key], flt):
                    continue
                dl = self.buf[key].dl
                s = idf * tf * (K1 + 1.0) / (tf + K1 * (1.0 - B + B * dl / avgdl))
                scores[key] = scores.get(key, 0.0) + s
        for si, acc in enumerate(accs):
            if acc is None:
                continue
            seg = self.segments[si]
            idx = np.nonzero(acc > 0.0)[0]                # BM25 contributions are always > 0
            if k is not None and len(idx) > 4 * k + 16:   # keep this segment's k best (plus ties): enough for the global top k
                vals = acc[idx]
                thr = np.partition(vals, len(vals) - k)[len(vals) - k]
                idx = idx[vals >= thr]
            for local in idx.tolist():
                scores[seg.keys[local]] = float(acc[local])
        order = lambda kv: (-kv[1], kv[0])                 # score desc, then key: deterministic
        ranked = sorted(scores.items(), key=order) if k is None else heapq.nsmallest(k, scores.items(), key=order)
        return [Hit(s, key, split_key(key)[0]) for key, s in ranked]

    # ------------------------------------------------------------ vector + hybrid search
    def _buffer_vectors(self):
        """(keys, float32 matrix) of buffered docs that have a vector; cached until the buffer changes."""
        if self._buf_vec_cache is None or self._buf_vec_cache[0] != self._buf_ver:
            keys = [k for k, d in self.buf.items() if d.vec is not None]
            mat = (np.stack([np.frombuffer(self.buf[k].vec, dtype="<f2") for k in keys]).astype(np.float32)
                   if keys else np.zeros((0, self.vec_dim), dtype=np.float32))
            self._buf_vec_cache = (self._buf_ver, keys, mat)
        return self._buf_vec_cache[1], self._buf_vec_cache[2]

    def default_pool(self, k) -> int:
        """How many quantized candidates to re-score: at least 10k and 100, and 2.5% of all vectors.
        (Measured on synthetic data, bench/recall_vs_bits.py: with 1-bit codes a pool of 0.5% of 20,000
        vectors gave recall@10 = 0.81, a pool of 2.5% gave 0.997.)"""
        total = sum(s.meta["vectors"]["valid"] for s in self.segments if s.has_vectors)
        total += sum(1 for d in self.buf.values() if d.vec is not None)
        return max(10 * (k or 0), 100, math.ceil(0.025 * total))

    def search_vector(self, qvec, k=10, flt=None, pool=None, nprobe=None, exact=False) -> list:
        """Nearest papers by cosine similarity. Hit.score is the EXACT half-precision cosine.

        Normal mode: scan the quantized codes of each segment (all docs, or only the `nprobe`
        nearest IVF clusters when a segment has an IVF index), keep the best `pool` candidates
        (default: see default_pool), then re-score those with the half-precision vectors.
        exact=True skips the codes and scores every live vector exactly (the oracle for tests)."""
        if self.vec_dim == 0 or (k is not None and k <= 0):
            return []
        q = unit(qvec)
        if q.size != self.vec_dim:
            raise ValueError(f"query vector has {q.size} numbers but this index uses {self.vec_dim}")
        if k is None:
            exact = True
        pool = pool or self.default_pool(k)
        seg_cands = []                         # (estimated score, segment index, local id)
        for si, seg in enumerate(self.segments):
            if not seg.has_vectors:
                continue
            ok = seg.valid_vectors().copy()
            if self.dead[si]:
                ok[list(self.dead[si])] = False
            fm = seg.filter_mask(flt)
            if fm is not None:
                ok &= fm
            ids = None
            if nprobe and not exact:
                ids = seg.ivf_candidates(q, nprobe)
                if ids is not None:
                    ids = ids[ok[ids]]
            if ids is None:
                ids = np.nonzero(ok)[0]
            if len(ids) == 0:
                continue
            if exact:
                sc = seg.vector_rows(ids) @ q
            else:
                sc = seg.estimate_scores(q, ids)
            keep = np.argsort(-sc)[:pool] if len(sc) > pool and not exact else np.arange(len(sc))
            for i in keep:
                seg_cands.append((float(sc[i]), si, int(ids[i])))
        if not exact and len(seg_cands) > pool:
            seg_cands = heapq.nlargest(pool, seg_cands, key=lambda t: t[0])
        results = []                           # (exact score, key)
        by_seg: dict = {}
        for _s, si, local in seg_cands:
            by_seg.setdefault(si, []).append(local)
        for si, locals_ in by_seg.items():     # re-score candidates with the half-precision copy
            seg = self.segments[si]
            exact_scores = seg.vector_rows(locals_) @ q
            results += [(float(s), seg.keys[l]) for s, l in zip(exact_scores, locals_)]
        keys, mat = self._buffer_vectors()
        if keys:
            bs = mat @ q
            results += [(float(s), key) for s, key in zip(bs, keys)
                        if self._buf_ok(self.buf[key], flt)]
        results.sort(key=lambda t: (-t[0], t[1]))
        if k is not None:
            results = results[:k]
        return [Hit(s, key, split_key(key)[0]) for s, key in results]

    def search_hybrid(self, query: str, qvec, k=10, flt=None, depth=100, rrf_k=60,
                      nprobe=None, pool=None) -> list:
        """BM25 list + vector list, merged by Reciprocal Rank Fusion (default k=60).
        Each HybridHit says where the paper ranked in each list (its 'reason')."""
        bm = self.search(query, k=depth, flt=flt)
        vc = self.search_vector(qvec, k=depth, flt=flt, nprobe=nprobe, pool=pool) if qvec is not None else []
        bm_by = {h.key: (i, h.score) for i, h in enumerate(bm, 1)}
        vc_by = {h.key: (i, h.score) for i, h in enumerate(vc, 1)}
        fused = rrf([[h.key for h in bm], [h.key for h in vc]], k=rrf_k)
        out = []
        for key, score, (rb, rv) in fused[:k] if k is not None else fused:
            out.append(HybridHit(score, key, split_key(key)[0], rb, rv,
                                 bm_by[key][1] if key in bm_by else None,
                                 vc_by[key][1] if key in vc_by else None))
        return out

    # ------------------------------------------------------------ lifecycle
    def _release_everything(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        for s in self.segments:
            s.close()
        self.segments = []
        self._lock.release()

    def close(self) -> None:
        self.stop_background_compaction()
        with self._mu:
            if self._writer is not None:
                self._writer.commit()
            self._release_everything()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class BackgroundCompactor(threading.Thread):
    """Calls store.compact() on a timer until stopped. Errors are kept in .last_error (and stop the thread)
    instead of being swallowed; a crash point inside still kills the whole process, as intended."""

    def __init__(self, store, interval, policy):
        super().__init__(name="paperlens-compactor", daemon=True)
        self.store, self.interval, self.policy = store, interval, policy
        self._halt = threading.Event()
        self.runs = 0
        self.last_error = None

    def run(self):
        while not self._halt.wait(self.interval):
            try:
                if self.store.compact(**self.policy):
                    self.runs += 1
            except BaseException as e:             # noqa: BLE001 (reported, not hidden)
                self.last_error = e
                return

    def stop(self):
        self._halt.set()
        if self.is_alive() and threading.current_thread() is not self:
            self.join(timeout=60)


def _synchronized(fn):
    def wrapper(self, *a, **kw):
        with self._mu:
            return fn(self, *a, **kw)
    wrapper.__name__, wrapper.__doc__ = fn.__name__, fn.__doc__
    return wrapper


# Every public operation that touches the in-memory state takes the (re-entrant) state lock, so a
# background compaction can swap the segment list without a search seeing half of the swap.
for _name in ("put", "delete", "commit", "flush", "get", "live_keys", "get_vector", "stats", "search",
              "search_vector", "search_hybrid", "publish_graph"):
    setattr(IndexStore, _name, _synchronized(getattr(IndexStore, _name)))
