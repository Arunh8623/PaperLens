"""IndexReader: a read-only snapshot of an index, usable from any number of other processes.

What it does and why (plain language)
-------------------------------------
Queries should not wait for ingestion and should never see a half-finished state. A reader:
  1. pins the segments of the current MANIFEST (store/pins.py) so compaction cannot delete them,
  2. opens them (memory-mapped, no locks) and loads the matching citation graph,
  3. answers searches from that frozen snapshot, however much the writer changes meanwhile.
`refresh()` looks for a newer MANIFEST and moves to it as one step. What a snapshot contains is
everything the writer had FLUSHED into segments (the writer's in-memory buffer and unflushed diary
are not visible; that is the price of lock-free reads). Searching code is shared with IndexStore.
"""
from __future__ import annotations

import os
import threading

from .graphfile import GraphFileError, graph_path, read_graph_file
from .index import IndexCorrupt, IndexStore
from .manifest import ManifestError, read_manifest
from .pins import Pin
from .segment import Segment, SegmentError


class ReadOnlyIndex(Exception):
    pass


class IndexReader(IndexStore):
    def __init__(self, path, allow_tokenizer_mismatch: bool = False, attempts: int = 50):
        # deliberately NOT calling IndexStore.__init__: no writer lock, no diary, no recovery
        self.path = os.fspath(path)
        self.allow_tokenizer_mismatch = allow_tokenizer_mismatch
        self.seg_root = os.path.join(self.path, "segments")
        self._mu = threading.RLock()
        self._attempts = attempts
        self.buf, self.buf_inv, self.pending_tomb = {}, {}, set()
        self._buf_ver, self._buf_vec_cache = 0, None
        self._writer, self._bg, self._compacting, self._reserved = None, None, False, 0
        self.seq, self.recovery = 0, {}
        self.vec_bits, self.ivf_min_docs, self.embedder_name, self.vec_dim = 1, 0, "", 0
        self.segments, self.key_loc, self.dead, self.live, self.graph = [], {}, [], {}, None
        self.manifest = None
        if not os.path.isdir(self.path):
            raise FileNotFoundError(f"no such index: {self.path}")
        self._pin = Pin(self.path)
        try:
            self._install(*self._load_snapshot())
        except BaseException:
            self._pin.release()
            raise

    # ------------------------------------------------------------ snapshot loading
    def _load_snapshot(self):
        last_err = None
        for _ in range(self._attempts):
            m = read_manifest(self.path)
            self._pin.set(m.segments, m.generation)            # pin FIRST ...
            if read_manifest(self.path).generation != m.generation:
                continue                                       # ... then confirm nothing moved meanwhile
            segs = []
            try:
                for name in m.segments:
                    segs.append(Segment.open(os.path.join(self.seg_root, name),
                                             check_tokenizer=not self.allow_tokenizer_mismatch))
                graph = read_graph_file(graph_path(self.path, m.graph_version)) if m.graph_version else None
                return m, segs, graph
            except (SegmentError, GraphFileError, OSError) as e:
                last_err = e
                for sg in segs:
                    sg.close()
        raise IndexCorrupt(f"could not open a consistent snapshot ({last_err})")

    def _install(self, manifest, segments, graph) -> None:
        old = self.segments
        with self._mu:
            self.manifest, self.segments, self.graph = manifest, segments, graph
            self.live, self.vec_dim = {}, 0
            self._rebuild_from_segments()
            for seg in segments:
                if seg.has_vectors:
                    self.vec_dim = seg.vec_dim
                    self.embedder_name = self.embedder_name or seg.meta["vectors"].get("embedder", "")
            self.seq = manifest.last_flushed_seq
            self._buf_ver += 1
        for sg in old:
            sg.close()

    # ------------------------------------------------------------ public
    @property
    def generation(self) -> int:
        return self.manifest.generation

    def refresh(self) -> bool:
        """Move to the newest MANIFEST if there is one. True if the snapshot changed."""
        try:
            cur = read_manifest(self.path)
        except ManifestError:
            return False                                       # mid-replace on a odd filesystem: keep the snapshot
        if cur.generation == self.manifest.generation:
            return False
        self._install(*self._load_snapshot())
        return True

    def close(self) -> None:
        with self._mu:
            for sg in self.segments:
                sg.close()
            self.segments = []
            self._pin.release()

    def _release_everything(self) -> None:
        self.close()

    # ------------------------------------------------------------ writes are refused
    def _refuse(self, *a, **kw):
        raise ReadOnlyIndex("this is a read-only snapshot; open IndexStore in the writer process")

    put = delete = commit = flush = compact = publish_graph = _refuse
