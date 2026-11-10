"""Reader pins: how a reader process tells the writer "I am still using these segments".

What it does and why (plain language)
-------------------------------------
Compaction replaces several old segments by one new segment and then wants to delete the old folders.
A query process (reader) may still be reading them (it opened a snapshot a moment ago). Deleting files
under a reader is wrong (on Windows it is even impossible while they are mapped). So:

  * a reader creates pins/pin-<pid>-<random>.json listing the segment names it uses, and HOLDS an OS
    lock on that file for as long as it lives (the lock vanishes automatically if the reader dies);
  * after publishing a new MANIFEST, the writer deletes an obsolete segment folder only if no live
    pin lists it. A pin whose lock can be taken is a dead reader's leftover and is removed.

Race closed by ordering: a reader writes its pin BEFORE it re-reads the MANIFEST to confirm. If the
writer published a newer MANIFEST in between, the reader notices and retries; if not, the writer's later
cleanup will see the pin. (Same protocol as "hazard pointers".)
"""
from __future__ import annotations

import json
import os
import uuid

from .fsutil import retry_fs
from .lock import FileLock, IndexLocked

PIN_DIR = "pins"


class Pin:
    def __init__(self, index_dir):
        d = os.path.join(os.fspath(index_dir), PIN_DIR)
        os.makedirs(d, exist_ok=True)
        self.path = os.path.join(d, f"pin-{os.getpid()}-{uuid.uuid4().hex[:8]}.json")
        self._lock = FileLock(self.path)
        self._lock.acquire()                       # creates the file and holds the lock until release()
        self.set([])

    def set(self, segments, generation: int = 0) -> None:
        data = json.dumps({"pid": os.getpid(), "generation": generation, "segments": list(segments)}).encode("utf-8")
        fd = self._lock.fd
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, data + b" " * 64)             # padding: a shorter rewrite never leaves a stale tail
        os.ftruncate(fd, len(data) + 64)           # pins are not durable state; no fsync on purpose

    def release(self) -> None:
        self._lock.release()
        try:
            retry_fs(os.remove, self.path)
        except OSError:
            pass                                   # the writer removes it later (its lock is free)


def pinned_segments(index_dir):
    """Names of segments listed by LIVE pins, or None if a live pin could not be read (caller must not delete).
    Dead readers' pins are deleted on the way."""
    d = os.path.join(os.fspath(index_dir), PIN_DIR)
    if not os.path.isdir(d):
        return set()
    names: set = set()
    for fn in sorted(os.listdir(d)):
        path = os.path.join(d, fn)
        probe = FileLock(path)
        try:
            probe.acquire()                        # we could lock it -> the owner is gone
        except IndexLocked:
            try:
                with open(path, "rb") as f:
                    doc = json.loads(f.read().decode("utf-8").strip())
                names.update(doc["segments"])
            except (OSError, ValueError, KeyError, TypeError):
                return None                        # alive but mid-write or unreadable: be conservative
            continue
        except OSError:
            continue
        probe.release()
        try:
            retry_fs(os.remove, path)
        except OSError:
            pass
    return names
