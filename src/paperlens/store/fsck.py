"""fsck: check an index directory for damage, without changing anything.

What it does and why (plain language)
-------------------------------------
Recovery repairs the ordinary leftovers of a crash. fsck is the independent
auditor: it re-reads every file, re-computes every checksum and checks the
invariants that must hold, then REPORTS. Tests run it after every simulated
crash, and it is a CLI command (`paperlens fsck`).

Checks
  MANIFEST        readable, checksum ok
  each segment    listed dir exists; meta.json; every file's size AND crc32;
                  keys / doclens / docs.off counts agree; terms sorted;
                  every postings list decodes, doc ids in range, df matches;
                  per document: sum of tf == doclen (a strong cross-check);
                  total doclen matches meta; docs.off monotonic and ends at
                  docs.bin size; every stored paper parses and matches its key
  vectors         (segments that have them) vectors.f16 / vectors.q / ivf.bin sizes agree
                  with n and dim; valid flags match the non-zero rows; the quantized codes
                  are re-computed from the half-precision vectors and must match (a tiny
                  fraction may differ because of float rounding); IVF lists are a permutation
                  of the valid docs; attrs.json agrees with the stored papers
  directory       unlisted seg-* / .tmp-* folders (WARNING: recovery removes them)
  diary           every file scanned; torn tail (WARNING: recovery cuts it);
                  entries well-formed, sequence strictly increasing

Returns FsckReport(errors, warnings, info). `ok` means no errors.
Takes the writer lock so it never reads a half-written state; if a writer is
running it reports that instead of guessing.
"""
from __future__ import annotations

import json
import os
import zlib
from dataclasses import dataclass, field

from ..contracts.paper import PaperError, PaperRecord
from .filters import paper_year
from ..text.varint import VarintError, decode_postings
from .diary import list_wals, scan_file
from .graphfile import GRAPH_DIR, GraphFileError, graph_path, read_graph_file
from .lock import DirLock, IndexLocked
from .manifest import ManifestError, read_manifest
from .segment import DATA_FILES, OPTIONAL_FILES, Segment, SegmentError


@dataclass
class FsckReport:
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    info: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


def fsck(index_dir) -> FsckReport:
    rep = FsckReport()
    index_dir = os.fspath(index_dir)
    lock = DirLock(index_dir)
    try:
        lock.acquire()
    except IndexLocked:
        rep.errors.append("index is locked by a running writer; cannot check safely")
        return rep
    try:
        _check(index_dir, rep)
    finally:
        lock.release()
    return rep


def _check(index_dir, rep: FsckReport) -> None:
    try:
        m = read_manifest(index_dir)
    except ManifestError as e:
        rep.errors.append(f"MANIFEST: {e}")
        return
    rep.info.update(generation=m.generation, last_flushed_seq=m.last_flushed_seq,
                    segments=list(m.segments))
    seg_root = os.path.join(index_dir, "segments")
    on_disk = sorted(os.listdir(seg_root)) if os.path.isdir(seg_root) else []
    for name in on_disk:
        if name.startswith(".tmp-") or (name.startswith("seg-") and name not in m.segments):
            rep.warnings.append(f"leftover {name} (recovery removes it)")
    total_docs = 0
    for name in m.segments:
        path = os.path.join(seg_root, name)
        if not os.path.isdir(path):
            rep.errors.append(f"{name}: listed in MANIFEST but missing")
            continue
        total_docs += _check_segment(path, name, rep)
    rep.info["physical_docs"] = total_docs
    _check_graph(index_dir, m.graph_version, rep)
    _check_diary(index_dir, m.last_flushed_seq, rep)


def _check_graph(index_dir, version, rep) -> None:
    gdir = os.path.join(index_dir, GRAPH_DIR)
    live = os.path.basename(graph_path(index_dir, version)) if version else None
    if os.path.isdir(gdir):
        for name in sorted(os.listdir(gdir)):
            if name != live:
                rep.warnings.append(f"leftover graphs/{name} (recovery removes it)")
    if not version:
        return
    try:
        g = read_graph_file(graph_path(index_dir, version))
        rep.info["graph"] = {"version": version, "nodes": g.n, "edges": g.m}
    except GraphFileError as e:
        rep.errors.append(f"graph v{version}: {e}")


def _check_segment(path, name, rep) -> int:
    err = lambda msg: rep.errors.append(f"{name}: {msg}")
    try:
        with open(os.path.join(path, "meta.json"), "rb") as f:
            meta = json.loads(f.read().decode("utf-8"))
        files = meta["files"]
    except (OSError, ValueError, KeyError) as e:
        err(f"meta.json unreadable ({e})")
        return 0
    if not set(DATA_FILES) <= set(files) or set(files) - set(DATA_FILES) - set(OPTIONAL_FILES):
        err(f"meta.json lists unexpected files {sorted(files)}")
    for fname, info in files.items():
        try:
            with open(os.path.join(path, fname), "rb") as f:
                data = f.read()
        except OSError as e:
            err(f"{fname}: cannot read ({e})")
            continue
        if len(data) != info["size"]:
            err(f"{fname}: size {len(data)} != {info['size']}")
        elif (zlib.crc32(data) & 0xFFFFFFFF) != info["crc32"]:
            err(f"{fname}: checksum mismatch")
    if rep.errors and any(e.startswith(name + ":") for e in rep.errors):
        return meta.get("num_docs", 0)                      # do not trust deeper checks on bad bytes
    try:
        seg = Segment.open(path)
    except SegmentError as e:
        err(str(e))
        return meta.get("num_docs", 0)
    try:
        n = seg.num_docs
        terms = list(seg.terms)
        if terms != sorted(terms):
            err("terms.idx is not sorted")
        sums = [0] * n
        for term, (_off, df) in seg.terms.items():
            try:
                pairs = seg.postings(term, reference=True)       # pure Python: independent of the C++ decoder
            except VarintError as e:
                err(f"postings for {term!r} do not decode ({e})")
                continue
            if len(pairs) != df:
                err(f"term {term!r}: df {df} but {len(pairs)} postings")
            for doc, tf in pairs:
                if doc >= n:
                    err(f"term {term!r}: doc id {doc} out of range")
                    break
                sums[doc] += tf
        if sums != list(seg.doclens):
            bad = next(i for i in range(n) if sums[i] != seg.doclens[i])
            err(f"doc {bad}: postings sum {sums[bad]} != doclen {seg.doclens[bad]}")
        if sum(seg.doclens) != seg.total_doclen:
            err("total doclen differs from meta.json")
        offs = list(seg.docs_off)
        if offs[0] != 0 or offs != sorted(offs) or offs[-1] != os.path.getsize(os.path.join(path, "docs.bin")):
            err("docs.off is inconsistent with docs.bin")
        else:
            for i in range(n):
                try:
                    p = seg.paper(i)
                except (PaperError, UnicodeDecodeError) as e:
                    err(f"doc {i}: stored paper does not parse ({e})")
                    break
                if p.key != seg.keys[i]:
                    err(f"doc {i}: stored paper key {p.key} != keys.txt {seg.keys[i]}")
                    break
            if "attrs.json" in files:
                years, cats = seg._load_attrs()
                if len(years) != n or len(cats) != n:
                    err("attrs.json has the wrong number of entries")
                else:
                    for i in range(n):
                        p = seg.paper(i)
                        if years[i] != paper_year(p.created, p.datestamp) or cats[i] != list(p.categories):
                            err(f"doc {i}: attrs.json disagrees with the stored paper")
                            break
        if seg.has_vectors:
            _check_vectors(seg, err)
        return n
    finally:
        seg.close()


def _check_vectors(seg, err) -> None:
    import numpy as np
    from ..vectors import quant
    try:
        vm = seg.meta["vectors"]
        n, dim, bits = seg.num_docs, vm["dim"], vm["bits"]
        if "vectors.f16" not in seg.meta["files"] or "vectors.q" not in seg.meta["files"]:
            err("meta.json has vectors info but not the vector files")
            return
        if seg.meta["files"]["vectors.f16"]["size"] != n * dim * 2:
            err("vectors.f16 size is not n * dim * 2")
            return
        codes, d_o, inv, oc, valid, centroid, ivf = seg._load_vectors()
        f16 = seg.all_vectors()
        norms = np.linalg.norm(f16, axis=1)
        if not np.array_equal(valid, norms > 0):
            err("vectors.q valid flags do not match the non-zero rows of vectors.f16")
        if int(valid.sum()) != vm["valid"]:
            err("meta.json vector count differs from the valid flags")
        x = np.where(norms[:, None] > 0, f16 / np.where(norms > 0, norms, 1.0)[:, None], 0.0).astype(np.float32)
        enc = quant.encode(x, centroid, bits, quant.rotation_matrix(dim))
        if valid.any():
            v = valid
            diff = np.unpackbits(enc["codes"][v] ^ codes[v]).sum() / max(1, v.sum() * dim * bits)
            if diff > 1e-3:
                err(f"quantized codes do not match vectors.f16 ({diff:.4%} of bits differ)")
            if not np.allclose(enc["d_o"][v], d_o[v], atol=1e-3) or not np.allclose(enc["oc"][v], oc[v], atol=1e-3):
                err("vectors.q correction numbers do not match vectors.f16")
        if ivf is not None:
            ids = np.sort(ivf.order)
            if not np.array_equal(ids, np.nonzero(valid)[0]) or ivf.offsets[0] != 0 \
                    or ivf.offsets[-1] != len(ivf.order) or np.any(np.diff(ivf.offsets.astype(np.int64)) < 0):
                err("ivf.bin lists are not a consistent partition of the valid documents")
    except (SegmentError, ValueError, KeyError, OSError) as e:
        err(f"vectors unreadable ({e})")


def _check_diary(index_dir, flushed, rep) -> None:
    wal_dir = os.path.join(index_dir, "wal")
    if not os.path.isdir(wal_dir):
        return
    wals = list_wals(wal_dir)
    prev = 0
    for i, (_first, path) in enumerate(wals):
        res = scan_file(path)
        base = os.path.basename(path)
        if res.torn:
            msg = f"{base}: torn tail ({res.reason}, {res.file_bytes - res.good_bytes} bytes; recovery cuts it)"
            (rep.warnings if i == len(wals) - 1 else rep.errors).append(msg)
        for payload in res.payloads:
            try:
                e = json.loads(payload.decode("utf-8"))
                seq = e["seq"]
                if e["op"] == "put":
                    p = PaperRecord.from_dict(e["paper"])
                    if p.key != e["key"]:
                        raise ValueError("entry key does not match its paper")
                    if e.get("vec"):
                        import base64
                        if len(base64.b64decode(e["vec"])) % 2:
                            raise ValueError("diary vector has an odd byte count")
                elif e["op"] != "delete":
                    raise ValueError(f"unknown op {e['op']!r}")
            except (ValueError, KeyError, TypeError, PaperError) as ex:
                rep.errors.append(f"{base}: bad entry ({ex})")
                break
            if seq <= flushed:
                continue
            if seq <= prev:
                rep.errors.append(f"{base}: sequence {seq} not increasing")
                break
            prev = seq
    rep.info["diary_files"] = len(wals)
    rep.info["unflushed_diary_entries_last_seq"] = prev
