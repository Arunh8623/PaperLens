"""Citation-graph snapshot files, published through the MANIFEST (field graph_version).

What it does and why (plain language)
-------------------------------------
The citation graph is computed OUTSIDE the diary (it comes from OpenAlex, not from papers
being indexed), so it gets its own tiny publish protocol, the same idea as a segment:

    1. write graphs/.tmp-graph-N.bin   (graph bytes + 4-byte crc32 trailer), fsync
    2. rename to graphs/graph-N.bin, fsync the directory            [graph_after_file]
    3. write a new MANIFEST with graph_version = N (atomic)         [graph_after_manifest]
    4. delete older graph files

Crash between 2 and 3: the MANIFEST still says N-1, so graph-N.bin is an unlisted leftover that
recovery deletes (readers keep seeing graph N-1). Crash after 3: graph N is live, old files are
leftovers that recovery deletes. Readers load exactly the version their MANIFEST names.
"""
from __future__ import annotations

import os
import struct
import zlib

from ..graph.csr import Graph
from .crashpoints import crashpoint
from .fsutil import fsync_dir, retry_fs, write_file_durable
from .manifest import read_manifest, write_manifest

GRAPH_DIR = "graphs"


class GraphFileError(Exception):
    pass


def graph_path(index_dir, version: int) -> str:
    return os.path.join(index_dir, GRAPH_DIR, f"graph-{version:06d}.bin")


def encode_graph(g: Graph) -> bytes:
    body = g.to_bytes()
    return body + struct.pack("<I", zlib.crc32(body) & 0xFFFFFFFF)


def read_graph_file(path) -> Graph:
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError as e:
        raise GraphFileError(f"cannot read {path}: {e}") from e
    if len(raw) < 8:
        raise GraphFileError("graph file too short")
    body, (crc,) = raw[:-4], struct.unpack("<I", raw[-4:])
    if zlib.crc32(body) & 0xFFFFFFFF != crc:
        raise GraphFileError("graph file checksum mismatch")
    try:
        return Graph.from_bytes(body)
    except ValueError as e:
        raise GraphFileError(str(e)) from e


def load_graph(index_dir):
    """The graph named by the MANIFEST, or None when graph_version is 0."""
    m = read_manifest(index_dir)
    if m.graph_version == 0:
        return None
    return read_graph_file(graph_path(index_dir, m.graph_version))


def publish_graph(index_dir, g: Graph) -> int:
    """Caller must hold the writer lock (IndexStore.publish_graph does). Returns the new version."""
    m = read_manifest(index_dir)
    version = m.graph_version + 1
    gdir = os.path.join(index_dir, GRAPH_DIR)
    os.makedirs(gdir, exist_ok=True)
    tmp = os.path.join(gdir, f".tmp-graph-{version:06d}.bin")
    write_file_durable(tmp, encode_graph(g))
    retry_fs(os.replace, tmp, graph_path(index_dir, version))
    fsync_dir(gdir)
    crashpoint("graph_after_file")
    write_manifest(index_dir, m.next(graph_version=version))
    crashpoint("graph_after_manifest")
    clean_graph_dir(index_dir, version)
    return version


def clean_graph_dir(index_dir, keep_version: int) -> int:
    """Delete .tmp-* and every graph file except the live one. Returns how many were removed."""
    gdir = os.path.join(index_dir, GRAPH_DIR)
    if not os.path.isdir(gdir):
        return 0
    keep = os.path.basename(graph_path(index_dir, keep_version)) if keep_version else None
    removed = 0
    for name in sorted(os.listdir(gdir)):
        if name != keep:
            retry_fs(os.remove, os.path.join(gdir, name))
            removed += 1
    if removed:
        fsync_dir(gdir)
    return removed
