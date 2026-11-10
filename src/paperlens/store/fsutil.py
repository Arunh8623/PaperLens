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


def retry_fs(fn, *args, attempts: int = 8, delay: float = 0.05, sleep=None,
             on_windows: bool = os.name == "nt"):
    """Call fn(*args); on Windows retry a few times if it raises PermissionError.

    Why: on Windows, antivirus scanners and the search indexer can hold a file
    open for a few milliseconds right after we wrote it. Renaming, replacing or
    deleting it in that window fails with PermissionError although nothing is
    wrong. Waiting a moment and retrying is the standard fix. On Linux/macOS
    this never retries (a PermissionError there is a real problem).
    Backoff: delay, 2*delay, 4*delay ... capped at 1 s per wait.
    """
    import time
    sleep = sleep or time.sleep
    for i in range(attempts):
        try:
            return fn(*args)
        except PermissionError:
            if not on_windows or i == attempts - 1:
                raise
            sleep(min(delay * (2 ** i), 1.0))
