"""Small, careful file helpers shared by the store.

Why this file exists: crash safety depends on a few OS details that differ
between Windows and Linux. We keep all of them in ONE place.

  * O_BINARY: on Windows, os.open() defaults to TEXT mode, which rewrites
    byte 0x0A as 0x0D 0x0A and silently corrupts binary logs. os.O_BINARY
    exists only on Windows, so we use getattr(..., 0).
  * fsync_dir: Linux needs the *directory* fsynced to make a create/rename
    durable. Windows cannot open a directory like that, so errors are ignored.
"""
from __future__ import annotations

import os

O_BINARY = getattr(os, "O_BINARY", 0)


def fsync_dir(path) -> None:
    """Best effort: make directory entries (create/rename/delete) durable."""
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return  # Windows: directories cannot be opened this way
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def write_file_durable(path, data: bytes) -> None:
    """Create/overwrite `path` with `data` and fsync it (not the directory)."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | O_BINARY, 0o644)
    try:
        view = memoryview(data)
        while view:
            n = os.write(fd, view)
            view = view[n:]
        os.fsync(fd)
    finally:
        os.close(fd)
