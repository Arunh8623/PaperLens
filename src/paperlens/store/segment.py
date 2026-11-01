"""Immutable segments: the files an index is made of.

What it does and why (plain language)
-------------------------------------
New papers wait in a small in-memory buffer (protected by the diary). When it
is big enough we "flush": freeze the buffer into a SEGMENT, a folder of files
that is never modified again. Immutable files are easy to make crash-safe
(write it all, then publish with one atomic step), easy to read without
locks, and easy to explain.

Files in segments/seg-NNNNNN/ (see docs/CONTRACTS.md):

  postings.bin  per term (sorted): df pairs of varint (doc gap, tf)
  terms.idx     text lines  term<TAB>offset<TAB>df   (sorted by term)
  keys.txt      one idempotency key per local doc id (line n = doc n)
  doclens.bin   u32 little-endian token count per doc
  docs.bin      the stored papers (canonical JSON, concatenated)
  docs.off      u64 little-endian offsets into docs.bin, n+1 entries
  deletes.json  keys killed by this batch (apply to OLDER segments only)
  meta.json     counts + size + crc32 of every other file. WRITTEN LAST.

Local doc ids are 0..n-1 in the order the papers were accepted.

Reading: postings.bin and docs.bin are memory-mapped (read-only). terms.idx
is loaded into a dict (fine at our scale; a real engine would use an FST or
block index). Opening checks file SIZES against meta.json (fast); fsck checks
every CRC (slow, thorough).
"""
from __future__ import annotations

import array
import json
import mmap
import os
import shutil
import struct
import zlib
from typing import Iterable

from ..contracts.paper import PaperRecord
from ..text.varint import decode_postings, encode_postings
from .fsutil import fsync_dir, write_file_durable

FORMAT = 1
TOKENIZER_VERSION = 1          # bump when text/tokenizer.py changes behaviour (CONTRACTS Q2)
DATA_FILES = ("postings.bin", "terms.idx", "keys.txt", "doclens.bin",
              "docs.bin", "docs.off", "deletes.json")


class SegmentError(Exception):
    pass


def segment_name(n: int) -> str:
    return f"seg-{n:06d}"


def segment_number(name: str):
    if name.startswith("seg-"):
        try:
            return int(name[4:])
        except ValueError:
            return None
    return None


def _crc(b: bytes) -> int:
    return zlib.crc32(b) & 0xFFFFFFFF


def build_files(docs, tombstones: Iterable[str]) -> dict:
    """docs: list of (key, PaperRecord, tf_dict, doclen). Returns {filename: bytes}
    for every data file (meta.json is added by write_segment_tmp)."""
    postings: dict = {}
    for local, (_key, _paper, tf, _dl) in enumerate(docs):
        for term, c in tf.items():
            postings.setdefault(term, []).append((local, c))
    post, idx = bytearray(), []
    for term in sorted(postings):
        lst = postings[term]
        idx.append(f"{term}\t{len(post)}\t{len(lst)}\n")
        post += encode_postings(lst)
    blobs = [d[1].to_json().encode("utf-8") for d in docs]
    offs = array.array("Q", [0])
    for b in blobs:
        offs.append(offs[-1] + len(b))
    doclens = array.array("I", [d[3] for d in docs])
    assert offs.itemsize == 8 and doclens.itemsize == 4
    return {
        "postings.bin": bytes(post),
        "terms.idx": "".join(idx).encode("utf-8"),
        "keys.txt": "".join(d[0] + "\n" for d in docs).encode("utf-8"),
        "doclens.bin": _le(doclens),
        "docs.bin": b"".join(blobs),
        "docs.off": _le(offs),
        "deletes.json": json.dumps(sorted(set(tombstones))).encode("utf-8"),
    }


def _le(arr: array.array) -> bytes:
    if struct.pack("=H", 1) != struct.pack("<H", 1):     # big-endian host
        arr = array.array(arr.typecode, arr)
        arr.byteswap()
    return arr.tobytes()


def _from_le(data: bytes, typecode: str) -> array.array:
    arr = array.array(typecode)
    arr.frombytes(data)
    if struct.pack("=H", 1) != struct.pack("<H", 1):
        arr.byteswap()
    return arr


def write_segment_tmp(seg_root: str, name: str, docs, tombstones) -> str:
    """Write ALL files of a segment into segments/.tmp-<name>/ (each fsynced),
    meta.json last, then fsync the directory. Returns the tmp path.
    The caller renames it to its final name (the publish step)."""
    tmp = os.path.join(seg_root, ".tmp-" + name)
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    files = build_files(docs, tombstones)
    for fname, data in files.items():
        write_file_durable(os.path.join(tmp, fname), data)
    meta = {
        "format": FORMAT, "tokenizer_version": TOKENIZER_VERSION, "name": name,
        "num_docs": len(docs), "total_doclen": sum(d[3] for d in docs),
        "num_terms": files["terms.idx"].count(b"\n"),
        "num_tombstones": len(json.loads(files["deletes.json"])),
        "files": {f: {"size": len(b), "crc32": _crc(b)} for f, b in files.items()},
    }
    write_file_durable(os.path.join(tmp, "meta.json"),
                       json.dumps(meta, indent=1, sort_keys=True).encode("utf-8"))
    fsync_dir(tmp)
    return tmp


def publish_segment(seg_root: str, tmp: str, name: str) -> str:
    """Atomically make the segment visible: rename tmp -> final, fsync parent."""
    final = os.path.join(seg_root, name)
    os.rename(tmp, final)
    fsync_dir(seg_root)
    return final


class _Map:
    """Read-only mmap that tolerates empty files (mmap refuses size 0)."""

    def __init__(self, path):
        self._f = open(path, "rb")
        self._mm = None
        if os.fstat(self._f.fileno()).st_size > 0:
            self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)

    def __len__(self):
        return len(self._mm) if self._mm else 0

    def slice(self, a, b) -> bytes:
        return self._mm[a:b] if self._mm else b""

    def close(self):
        if self._mm:
            self._mm.close()
            self._mm = None
        if self._f:
            self._f.close()
            self._f = None


class Segment:
    """Read-only view of one segment directory."""

    def __init__(self):
        self.path = self.name = ""
        self.meta: dict = {}
        self.keys: list = []
        self.tombstones: list = []
        self.terms: dict = {}
        self.doclens = array.array("I")
        self.docs_off = array.array("Q")
        self._post = self._docs = None

    @property
    def num_docs(self) -> int:
        return self.meta["num_docs"]

    @property
    def total_doclen(self) -> int:
        return self.meta["total_doclen"]

    @classmethod
    def open(cls, path: str) -> "Segment":
        s = cls()
        s.path, s.name = path, os.path.basename(path)
        try:
            with open(os.path.join(path, "meta.json"), "rb") as f:
                s.meta = json.loads(f.read().decode("utf-8"))
            if s.meta["format"] != FORMAT:
                raise SegmentError(f"{s.name}: unsupported format {s.meta['format']}")
            if s.meta["tokenizer_version"] != TOKENIZER_VERSION:
                raise SegmentError(f"{s.name}: built with tokenizer v{s.meta['tokenizer_version']}, "
                                   f"this code is v{TOKENIZER_VERSION}; reindex")
            for fname, info in s.meta["files"].items():
                if os.path.getsize(os.path.join(path, fname)) != info["size"]:
                    raise SegmentError(f"{s.name}/{fname}: size differs from meta.json")
            s._read_small_files()
            s._post = _Map(os.path.join(path, "postings.bin"))
            s._docs = _Map(os.path.join(path, "docs.bin"))
        except SegmentError:
            s.close()
            raise
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
            s.close()
            raise SegmentError(f"{os.path.basename(path)}: cannot open ({type(e).__name__}: {e})") from e
        return s

    def _read(self, fname) -> bytes:
        with open(os.path.join(self.path, fname), "rb") as f:
            return f.read()

    def _read_small_files(self):
        n = self.meta["num_docs"]
        self.keys = self._read("keys.txt").decode("utf-8").splitlines()
        self.doclens = _from_le(self._read("doclens.bin"), "I")
        self.docs_off = _from_le(self._read("docs.off"), "Q")
        self.tombstones = json.loads(self._read("deletes.json").decode("utf-8"))
        if len(self.keys) != n or len(self.doclens) != n or len(self.docs_off) != n + 1:
            raise SegmentError(f"{self.name}: per-document files disagree with num_docs={n}")
        for line in self._read("terms.idx").decode("utf-8").splitlines():
            term, off, df = line.split("\t")
            self.terms[term] = (int(off), int(df))
        if len(self.terms) != self.meta["num_terms"]:
            raise SegmentError(f"{self.name}: terms.idx has {len(self.terms)} terms, "
                               f"meta says {self.meta['num_terms']}")

    def df(self, term: str) -> int:
        t = self.terms.get(term)
        return t[1] if t else 0

    def postings(self, term: str) -> list:
        """[(local_doc, tf)] for a term (includes docs that were later deleted)."""
        t = self.terms.get(term)
        if t is None:
            return []
        off, df = t
        data = self._post.slice(off, min(off + df * 20, len(self._post)))
        pairs, _ = decode_postings(data, 0, df)
        return pairs

    def paper(self, local: int) -> PaperRecord:
        a, b = self.docs_off[local], self.docs_off[local + 1]
        return PaperRecord.from_json(self._docs.slice(a, b).decode("utf-8"))

    def close(self) -> None:
        # Windows: mmaps/files must be closed before a segment folder can be deleted.
        for m in (self._post, self._docs):
            if m is not None:
                m.close()
        self._post = self._docs = None
