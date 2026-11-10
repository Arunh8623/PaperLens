"""Shared-memory arenas for vectors: one arena per reader, `slots` x `dim` float32.

What it does and why: sending a 384-float vector through a pipe means copying
it twice. Instead each reader owns an arena in shared memory; the SUPERVISOR
decides which slot a job may use, the reader writes the vector there, and only
the slot number + a checksum travel in the message. The writer copies the bytes
out and verifies the checksum, so a stale or half-written slot is detected
(Ack -2 -> the job is retried).

Resource-tracker note (POSIX): a process that merely ATTACHES to an existing
block must not make Python delete the block when it exits. Python 3.13+ has
track=False; for older versions we unregister the block from the tracker
ourselves. On Windows there is no tracker and the block lives until the last
handle closes (the supervisor keeps one open).
"""
from __future__ import annotations

import os
import sys
from multiprocessing import shared_memory

FLOAT_BYTES = 4


def arena_bytes(slots: int, dim: int) -> int:
    return slots * dim * FLOAT_BYTES


def create_arena(slots: int, dim: int) -> shared_memory.SharedMemory:
    return shared_memory.SharedMemory(create=True, size=max(arena_bytes(slots, dim), 1))


def attach_arena(name: str) -> shared_memory.SharedMemory:
    if sys.version_info >= (3, 13):
        return shared_memory.SharedMemory(name=name, track=False)
    shm = shared_memory.SharedMemory(name=name)
    if os.name == "posix":
        try:
            from multiprocessing import resource_tracker
            resource_tracker.unregister(shm._name, "shared_memory")
        except Exception:                                     # best effort; never fail an attach
            pass
    return shm


def slot_range(slot: int, dim: int) -> tuple:
    start = slot * dim * FLOAT_BYTES
    return start, start + dim * FLOAT_BYTES


def write_slot(shm, slot: int, dim: int, raw: bytes) -> None:
    a, b = slot_range(slot, dim)
    if len(raw) != b - a or b > shm.size:
        raise ValueError("vector does not fit its slot")
    shm.buf[a:b] = raw


def read_slot(shm, slot: int, dim: int) -> bytes:
    a, b = slot_range(slot, dim)
    if b > shm.size:
        raise ValueError("slot outside the arena")
    return bytes(shm.buf[a:b])                                # COPY out: the reader may reuse the slot later
