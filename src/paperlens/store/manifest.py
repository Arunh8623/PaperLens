"""MANIFEST: the single file that says what the index IS.

What it does and why (plain language)
-------------------------------------
The index is a folder of immutable segment directories. The MANIFEST lists
which segments are "live" and how far the diary has been absorbed. Swapping
the MANIFEST is the ONE atomic step that publishes a change: before the swap
readers see the old index, after it they see the new one, never half of each.

File format (JSON):

    {"body": {"generation": 3, "last_flushed_seq": 120,
              "segments": ["seg-000001", "seg-000002"],
              "graph_version": 0, "schema": 1},
     "crc32": <crc32 of the canonical JSON text of body>}

Atomic write recipe (the order matters):
    1. write MANIFEST.tmp            2. fsync MANIFEST.tmp
    3. os.replace(tmp, MANIFEST)     4. fsync the directory
os.replace is atomic on POSIX and on Windows (same volume): a crash leaves
either the old file or the new one, never a mix.

If MANIFEST is missing -> a brand new index (generation 0, no segments).
If MANIFEST exists but fails its checksum -> ManifestError (we refuse to
guess; fsck/operator decides). A stale MANIFEST.tmp is just leftover junk.
"""
from __future__ import annotations

import json
import os
import zlib
from dataclasses import dataclass, field, replace

from .crashpoints import crashpoint  # noqa: F401  (flush uses it; kept for symmetry)
from .fsutil import fsync_dir, retry_fs, write_file_durable

MANIFEST_NAME = "MANIFEST"
TMP_NAME = "MANIFEST.tmp"
SCHEMA = 1


class ManifestError(Exception):
    pass


@dataclass(frozen=True)
class Manifest:
    generation: int = 0
    last_flushed_seq: int = 0
    segments: tuple = ()
    graph_version: int = 0
    schema: int = SCHEMA

    def body(self) -> dict:
        return {
            "generation": self.generation,
            "last_flushed_seq": self.last_flushed_seq,
            "segments": list(self.segments),
            "graph_version": self.graph_version,
            "schema": self.schema,
        }

    def next(self, **changes) -> "Manifest":
        """New manifest with generation+1 and the given fields changed."""
        changes.setdefault("generation", self.generation + 1)
        if "segments" in changes:
            changes["segments"] = tuple(changes["segments"])
        return replace(self, **changes)


def _canon(body: dict) -> bytes:
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")


def encode(m: Manifest) -> bytes:
    body = m.body()
    crc = zlib.crc32(_canon(body)) & 0xFFFFFFFF
    return json.dumps({"body": body, "crc32": crc}, sort_keys=True, indent=1).encode("utf-8")


def decode(raw: bytes) -> Manifest:
    try:
        doc = json.loads(raw.decode("utf-8"))
        body, crc = doc["body"], doc["crc32"]
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as e:
        raise ManifestError(f"MANIFEST unreadable: {e}") from e
    if not isinstance(body, dict) or (zlib.crc32(_canon(body)) & 0xFFFFFFFF) != crc:
        raise ManifestError("MANIFEST checksum mismatch")
    try:
        if body["schema"] != SCHEMA:
            raise ManifestError(f"unsupported schema {body['schema']}")
        segs = body["segments"]
        if not isinstance(segs, list) or not all(isinstance(s, str) for s in segs):
            raise ManifestError("segments must be a list of names")
        return Manifest(int(body["generation"]), int(body["last_flushed_seq"]),
                        tuple(segs), int(body["graph_version"]), int(body["schema"]))
    except (KeyError, TypeError, ValueError) as e:
        raise ManifestError(f"MANIFEST malformed: {e}") from e


def read_manifest(index_dir) -> Manifest:
    path = os.path.join(index_dir, MANIFEST_NAME)
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except FileNotFoundError:
        return Manifest()  # brand-new index
    return decode(raw)


def write_manifest(index_dir, m: Manifest) -> None:
    tmp = os.path.join(index_dir, TMP_NAME)
    final = os.path.join(index_dir, MANIFEST_NAME)
    write_file_durable(tmp, encode(m))      # steps 1-2
    retry_fs(os.replace, tmp, final)        # step 3 (atomic; retried on Windows if briefly locked)
    fsync_dir(index_dir)                    # step 4
