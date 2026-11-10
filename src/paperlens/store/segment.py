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
  attrs.json    per doc: year + categories (for filters); meta.json has a zone map
  vectors.f16   (only if the batch has vectors) half-precision unit vectors [n x dim], the
                copy used to RE-SCORE candidates exactly
  vectors.q     quantized codes + correction numbers + valid flags + centroid (vectors/quant.py)
  ivf.bin       (only for big enough segments) cluster index (vectors/ivf.py)
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
import math
import mmap
import os
import shutil
import struct
import zlib
from typing import Iterable

import numpy as np

from ..contracts.paper import PaperRecord
from .. import native
from ..text.varint import decode_postings, encode_postings
from ..vectors import ivf as ivf_mod
from ..vectors import quant
from .filters import paper_year
from .fsutil import fsync_dir, retry_fs, write_file_durable

FORMAT = 1
TOKENIZER_VERSION = 1          # bump when text/tokenizer.py changes behaviour (CONTRACTS Q2)
DATA_FILES = ("postings.bin", "terms.idx", "keys.txt", "doclens.bin",
              "docs.bin", "docs.off", "deletes.json")
OPTIONAL_FILES = ("attrs.json", "vectors.f16", "vectors.q", "ivf.bin")   # absent in Phase 1 segments


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


def _vector_files(docs, vec: dict):
    """Build vectors.f16 / vectors.q / ivf.bin. docs[i][4] is f16 bytes or None.
    Returns ({file: bytes}, meta dict) or ({}, None) when no doc has a vector."""
    dim, bits = vec["dim"], vec["bits"]
    n = len(docs)
    f16 = np.zeros((n, dim), dtype="<f2")
    valid = np.zeros(n, dtype=np.uint8)
    for i, d in enumerate(docs):
        if len(d) > 4 and d[4] is not None:
            if len(d[4]) != dim * 2:
                raise ValueError("stored vector has the wrong length")
            f16[i] = np.frombuffer(d[4], dtype="<f2")
            valid[i] = 1
    if not valid.any():
        return {}, None
    x = f16.astype(np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x = np.where(norms > 0, x / np.where(norms > 0, norms, 1.0), 0.0).astype(np.float32)
    vmask = valid.astype(bool)
    centroid = x[vmask].mean(axis=0).astype(np.float32)
    enc = quant.encode(x, centroid, bits, quant.rotation_matrix(dim))
    q_bytes = (enc["codes"].tobytes() + enc["d_o"].astype("<f4").tobytes() + enc["inv"].astype("<f4").tobytes()
               + enc["oc"].astype("<f4").tobytes() + valid.tobytes() + centroid.astype("<f4").tobytes())
    files = {"vectors.f16": f16.tobytes(), "vectors.q": q_bytes}
    n_valid = int(valid.sum())
    nlist = 0
    if vec.get("ivf_min_docs", 0) and n_valid >= vec["ivf_min_docs"]:
        nlist = max(4, int(math.sqrt(n_valid)))
        files["ivf.bin"] = ivf_mod.build(x[vmask], np.nonzero(vmask)[0], nlist)
    meta = {"dim": dim, "bits": bits, "qversion": quant.QVERSION, "rot_seed": quant.ROT_SEED,
            "embedder": vec.get("embedder", ""), "valid": n_valid, "ivf_nlist": nlist}
    return files, meta


def build_files(docs, tombstones: Iterable[str], vec: dict = None):
    """docs: list of (key, PaperRecord, tf_dict, doclen[, f16 vector bytes or None]).
    vec: {"dim", "bits", "ivf_min_docs", "embedder"} or None. Returns ({filename: bytes}, vector meta)
    for every data file (meta.json is added by write_segment_tmp)."""
    postings: dict = {}
    for local, d in enumerate(docs):
        tf = d[2]
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
    years = [paper_year(d[1].created, d[1].datestamp) for d in docs]
    attrs = {"years": years, "cats": [list(d[1].categories) for d in docs]}
    files = {
        "postings.bin": bytes(post),
        "terms.idx": "".join(idx).encode("utf-8"),
        "keys.txt": "".join(d[0] + "\n" for d in docs).encode("utf-8"),
        "doclens.bin": _le(doclens),
        "docs.bin": b"".join(blobs),
        "docs.off": _le(offs),
        "deletes.json": json.dumps(sorted(set(tombstones))).encode("utf-8"),
        "attrs.json": json.dumps(attrs, separators=(",", ":")).encode("utf-8"),
    }
    vmeta = None
    if vec is not None:
        vfiles, vmeta = _vector_files(docs, vec)
        files.update(vfiles)
    return files, vmeta


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


def write_segment_tmp(seg_root: str, name: str, docs, tombstones, vec: dict = None) -> str:
    """Write ALL files of a segment into segments/.tmp-<name>/ (each fsynced),
    meta.json last, then fsync the directory. Returns the tmp path.
    The caller renames it to its final name (the publish step)."""
    tmp = os.path.join(seg_root, ".tmp-" + name)
    if os.path.exists(tmp):
        retry_fs(shutil.rmtree, tmp)
    os.makedirs(tmp)
    files, vmeta = build_files(docs, tombstones, vec)
    for fname, data in files.items():
        write_file_durable(os.path.join(tmp, fname), data)
    meta = {
        "format": FORMAT, "tokenizer_version": TOKENIZER_VERSION, "name": name,
        "num_docs": len(docs), "total_doclen": sum(d[3] for d in docs),
        "num_terms": files["terms.idx"].count(b"\n"),
        "num_tombstones": len(json.loads(files["deletes.json"])),
        "files": {f: {"size": len(b), "crc32": _crc(b)} for f, b in files.items()},
        "zonemap": {"year_min": min(json.loads(files["attrs.json"])["years"], default=0),
                    "year_max": max(json.loads(files["attrs.json"])["years"], default=0),
                    "categories": sorted({c for d in docs for c in d[1].categories})},
    }
    if vmeta is not None:
        meta["vectors"] = vmeta
    write_file_durable(os.path.join(tmp, "meta.json"),
                       json.dumps(meta, indent=1, sort_keys=True).encode("utf-8"))
    fsync_dir(tmp)
    return tmp


def publish_segment(seg_root: str, tmp: str, name: str) -> str:
    """Atomically make the segment visible: rename tmp -> final, fsync parent."""
    final = os.path.join(seg_root, name)
    retry_fs(os.rename, tmp, final)
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
        self._f16 = None
        self._vq = None
        self._attrs = None
        self._masks: dict = {}

    @property
    def num_docs(self) -> int:
        return self.meta["num_docs"]

    @property
    def total_doclen(self) -> int:
        return self.meta["total_doclen"]

    @classmethod
    def open(cls, path: str, check_tokenizer: bool = True) -> "Segment":
        s = cls()
        s.path, s.name = path, os.path.basename(path)
        try:
            with open(os.path.join(path, "meta.json"), "rb") as f:
                s.meta = json.loads(f.read().decode("utf-8"))
            if s.meta["format"] != FORMAT:
                raise SegmentError(f"{s.name}: unsupported format {s.meta['format']}")
            if check_tokenizer and s.meta["tokenizer_version"] != TOKENIZER_VERSION:
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

    def _term_bytes(self, term: str):
        t = self.terms.get(term)
        if t is None:
            return None, 0
        off, df = t
        return self._post.slice(off, min(off + df * 20, len(self._post))), df

    def postings(self, term: str, reference: bool = False) -> list:
        """[(local_doc, tf)] for a term (includes docs that were later deleted).
        reference=True always uses the pure-Python decoder (fsck, as an independent check of the C++ one)."""
        data, df = self._term_bytes(term)
        if data is None:
            return []
        pairs, _ = (decode_postings if reference else native.decode_postings)(data, 0, df)
        return pairs

    def bm25_accumulate(self, term: str, idf: float, avgdl: float, out, dead_mask, mask, k1: float, b: float) -> int:
        """Add this segment's BM25 contribution of `term` into out[local] (float64, one entry per doc),
        skipping docs flagged in dead_mask (uint8 or None) or rejected by mask (bool or None)."""
        data, df = self._term_bytes(term)
        if data is None:
            return 0
        return native.bm25_accumulate(data, 0, df, self.doclens, dead_mask, mask, idf, k1, b, avgdl, out)

    def paper(self, local: int) -> PaperRecord:
        a, b = self.docs_off[local], self.docs_off[local + 1]
        return PaperRecord.from_json(self._docs.slice(a, b).decode("utf-8"))

    # ---------------------------------------------------------------- filters
    def _load_attrs(self):
        if self._attrs is None:
            try:
                a = json.loads(self._read("attrs.json").decode("utf-8"))
                self._attrs = (a["years"], a["cats"])
            except FileNotFoundError:                      # Phase 1 segment: derive from stored papers
                ps = [self.paper(i) for i in range(self.num_docs)]
                self._attrs = ([paper_year(p.created, p.datestamp) for p in ps],
                               [list(p.categories) for p in ps])
        return self._attrs

    def filter_mask(self, flt):
        """bool array (n,) of docs passing the filter, or None for 'everything passes'."""
        if flt is None or flt.empty:
            return None
        if flt in self._masks:
            return self._masks[flt]
        zone = self.meta.get("zonemap")
        if zone is not None and self.num_docs and not flt.may_match_zone(zone):
            mask = np.zeros(self.num_docs, dtype=bool)
        else:
            years, cats = self._load_attrs()
            mask = np.fromiter((flt.matches(y, c) for y, c in zip(years, cats)), dtype=bool, count=self.num_docs)
        self._masks[flt] = mask
        return mask

    # ---------------------------------------------------------------- vectors
    @property
    def has_vectors(self) -> bool:
        return "vectors" in self.meta

    @property
    def vec_dim(self) -> int:
        return self.meta["vectors"]["dim"] if self.has_vectors else 0

    def _load_vectors(self):
        if self._vq is not None:
            return self._vq
        vm = self.meta["vectors"]
        n, dim, bits = self.num_docs, vm["dim"], vm["bits"]
        cb = quant.code_bytes(dim, bits)
        raw = self._read("vectors.q")
        if len(raw) != n * cb + 3 * 4 * n + n + 4 * dim:
            raise SegmentError(f"{self.name}: vectors.q has the wrong size")
        o = 0
        codes = np.frombuffer(raw, dtype=np.uint8, count=n * cb, offset=o).reshape(n, cb); o += n * cb
        d_o = np.frombuffer(raw, dtype="<f4", count=n, offset=o); o += 4 * n
        inv = np.frombuffer(raw, dtype="<f4", count=n, offset=o); o += 4 * n
        oc = np.frombuffer(raw, dtype="<f4", count=n, offset=o); o += 4 * n
        valid = np.frombuffer(raw, dtype=np.uint8, count=n, offset=o).astype(bool); o += n
        centroid = np.frombuffer(raw, dtype="<f4", count=dim, offset=o).astype(np.float32)
        ivf = None
        if vm.get("ivf_nlist"):
            ivf = ivf_mod.Ivf(self._read("ivf.bin"), vm["ivf_nlist"], dim, vm["valid"])
        self._f16 = _Map(os.path.join(self.path, "vectors.f16"))
        self._vq = (codes, d_o, inv, oc, valid, centroid, ivf)
        return self._vq

    def valid_vectors(self):
        return self._load_vectors()[4]

    def vector_rows(self, locals_) -> np.ndarray:
        """Half-precision vectors of the given local doc ids as float32 (m, dim)."""
        self._load_vectors()
        dim = self.vec_dim
        row = dim * 2
        data = b"".join(self._f16.slice(int(i) * row, (int(i) + 1) * row) for i in locals_)
        return np.frombuffer(data, dtype="<f2").reshape(len(locals_), dim).astype(np.float32)

    def all_vectors(self) -> np.ndarray:
        self._load_vectors()
        return np.frombuffer(self._f16.slice(0, self.num_docs * self.vec_dim * 2),
                             dtype="<f2").reshape(self.num_docs, self.vec_dim).astype(np.float32)

    def estimate_scores(self, q: np.ndarray, candidates=None) -> np.ndarray:
        """Quantized estimate of <doc, q> for `candidates` (local ids) or for every doc."""
        codes, d_o, inv, oc, _valid, centroid, _ivf = self._load_vectors()
        vm = self.meta["vectors"]
        prep = quant.QueryPrep(q, centroid, quant.rotation_matrix(vm["dim"]))
        if candidates is None:
            return quant.estimate(codes, d_o, inv, oc, prep, vm["dim"], vm["bits"])
        c = np.asarray(candidates, dtype=np.int64)
        return quant.estimate(codes[c], d_o[c], inv[c], oc[c], prep, vm["dim"], vm["bits"])

    def ivf_candidates(self, q: np.ndarray, nprobe: int):
        ivf = self._load_vectors()[6]
        return None if ivf is None else ivf.candidates(q, nprobe)

    def close(self) -> None:
        # Windows: mmaps/files must be closed before a segment folder can be deleted.
        for m in (self._post, self._docs, self._f16):
            if m is not None:
                m.close()
        self._post = self._docs = self._f16 = None
        self._vq = None
