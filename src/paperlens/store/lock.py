"""Single-writer lock on <index>/LOCK.

What it does and why (plain language)
-------------------------------------
Two writers on one index would corrupt it. The OS gives us a lock that
belongs to an open file and is released automatically if the process dies
(even kill -9), so a crashed writer never leaves a stale lock behind.

  POSIX   : fcntl.flock(LOCK_EX | LOCK_NB)
  Windows : msvcrt.locking(LK_NBLCK) on byte 0  (UNTESTED here: Linux only sandbox)

A second opener gets IndexLocked immediately (non-blocking).
"""
from __future__ import annotations

import os

from .fsutil import O_BINARY


class IndexLocked(Exception):
    """Another process (or another store object) already holds the writer lock."""


class DirLock:
    _offset = 0                                      # byte that is locked on Windows (POSIX locks the whole file)

    def __init__(self, index_dir):
        self.path = os.path.join(os.fspath(index_dir), "LOCK")
        self._fd = None

    def acquire(self) -> None:
        if self._fd is not None:
            raise RuntimeError("lock already held by this object")
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT | O_BINARY, 0o644)
        try:
            if os.name == "nt":                      # pragma: no cover (Windows)
                import msvcrt
                os.lseek(fd, self._offset, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            os.close(fd)
            raise IndexLocked(f"index is locked by another writer: {self.path}") from e
        self._fd = fd

    def release(self) -> None:
        if self._fd is None:
            return
        fd, self._fd = self._fd, None
        try:
            if os.name == "nt":                      # pragma: no cover (Windows)
                import msvcrt
                os.lseek(fd, self._offset, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        finally:
            os.close(fd)                             # POSIX: closing drops the flock

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()


class FileLock(DirLock):
    """Same OS lock on any file path (used by reader pins). On Windows the locked byte is far past the
    end of the file, so other processes can still READ the file's contents while it is locked."""
    _offset = 1 << 30

    def __init__(self, path):
        self.path = os.fspath(path)
        self._fd = None

    @property
    def fd(self):
        return self._fd
