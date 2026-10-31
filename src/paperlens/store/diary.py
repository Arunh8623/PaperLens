"""The diary (write-ahead log): framing + torn-tail scan.

What it does and why (plain language)
-------------------------------------
Before we change the index we first write the change in a diary, then force
it to disk (fsync). If the machine or process dies, on restart we re-read the
diary and redo whatever the index had not yet absorbed. A paper counts as
"acknowledged" only after commit() has fsynced it.

Record framing (little endian):

    [u32 length][u32 crc32(payload)][payload bytes]

A crash can leave a half-written record at the END of the file (a "torn
tail"). scan() walks records from the start and stops at the first one that
is short or fails its CRC; everything before is good. recover_tail() then
truncates the file there so new appends start on a clean boundary.

Important: only the tail can be torn by a crash. We treat the first bad
record as the end of the log on purpose - anything after it was never
acknowledged, because acknowledgement happens after fsync of earlier bytes.

Payload here is opaque bytes; the store puts a JSON entry in it
(seq, op, key, paper) - see docs/CONTRACTS.md.
"""
from __future__ import annotations

import os
import struct
import zlib
from dataclasses import dataclass
from typing import Iterator

from .crashpoints import crashpoint
from .fsutil import O_BINARY, fsync_dir

HEADER = struct.Struct("<II")          # length, crc32
MAX_RECORD = 64 * 1024 * 1024          # sanity cap: a bigger length means garbage


def frame(payload: bytes) -> bytes:
    """Wrap payload into one diary record."""
    if len(payload) > MAX_RECORD:
        raise ValueError("record too large")
    return HEADER.pack(len(payload), zlib.crc32(payload) & 0xFFFFFFFF) + payload


@dataclass(frozen=True)
class ScanResult:
    payloads: list          # good payloads, in order
    good_bytes: int         # offset where the good prefix ends
    file_bytes: int         # total file size
    reason: str             # "clean" | "short_header" | "short_payload" | "bad_crc" | "bad_length"

    @property
    def torn(self) -> bool:
        return self.good_bytes != self.file_bytes


def scan_bytes(data: bytes) -> ScanResult:
    """Pure function: parse a diary image. Easy to test at every byte offset."""
    pos, out = 0, []
    n = len(data)
    reason = "clean"
    while pos < n:
        if n - pos < HEADER.size:
            reason = "short_header"
            break
        length, crc = HEADER.unpack_from(data, pos)
        if length > MAX_RECORD:
            reason = "bad_length"
            break
        end = pos + HEADER.size + length
        if end > n:
            reason = "short_payload"
            break
        payload = data[pos + HEADER.size:end]
        if (zlib.crc32(payload) & 0xFFFFFFFF) != crc:
            reason = "bad_crc"
            break
        out.append(payload)
        pos = end
    return ScanResult(out, pos, n, reason)


def scan_file(path) -> ScanResult:
    with open(path, "rb") as f:
        return scan_bytes(f.read())


def recover_tail(path) -> ScanResult:
    """Scan, then cut the torn tail (if any) and fsync. Returns the scan."""
    res = scan_file(path)
    if res.torn:
        fd = os.open(path, os.O_RDWR | O_BINARY)
        try:
            os.ftruncate(fd, res.good_bytes)
            os.fsync(fd)
        finally:
            os.close(fd)
    return res


class DiaryWriter:
    """Append-only writer for ONE diary file.

    put(payload)  -> appends a record (NOT yet durable)
    commit()      -> fsync: everything put() so far becomes durable (group commit)
    """

    def __init__(self, path):
        self.path = os.fspath(path)
        existed = os.path.exists(self.path)
        # O_BINARY is essential on Windows (see fsutil).
        self._fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | O_BINARY, 0o644)
        if not existed:
            fsync_dir(os.path.dirname(self.path) or ".")
        self._dirty = False

    def put(self, payload: bytes) -> None:
        data = frame(payload)
        view = memoryview(data)
        while view:
            n = os.write(self._fd, view)
            view = view[n:]
        self._dirty = True
        crashpoint("after_wal_write")

    def commit(self) -> None:
        if self._dirty:
            os.fsync(self._fd)
            self._dirty = False
        crashpoint("after_wal_fsync")

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def wal_name(first_seq: int) -> str:
    return f"wal-{first_seq:020d}.log"


def parse_wal_name(name: str) -> int | None:
    if name.startswith("wal-") and name.endswith(".log"):
        try:
            return int(name[4:-4])
        except ValueError:
            return None
    return None


def list_wals(wal_dir) -> list:
    """(first_seq, path) for every diary file, oldest first."""
    out = []
    for name in os.listdir(wal_dir):
        s = parse_wal_name(name)
        if s is not None:
            out.append((s, os.path.join(wal_dir, name)))
    return sorted(out)
