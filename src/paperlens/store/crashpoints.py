"""Named crash points (fault injection), built in from day one.

What it does and why
--------------------
To PROVE the index survives crashes, tests must be able to kill the process at
an exact step (for example "after the diary was written but before fsync").
Store code calls `crashpoint("name")` at those steps. Normally it does
nothing. If the environment variable PAPERLENS_CRASH_AT equals that name, the
process dies immediately with os._exit (no cleanup, no flush, like a crash).

Optional PAPERLENS_CRASH_AFTER=N means "die on the N-th time this point is hit"
(default 1), so a test can crash in the middle of a long run.

Known points (grows with the store): after_wal_write, after_wal_fsync,
flush_after_tmp_written, flush_after_segment_rename, flush_after_manifest,
flush_after_wal_delete, graph_after_file, graph_after_manifest.
"""
from __future__ import annotations

import os

CRASH_EXIT_CODE = 77  # distinctive, so tests can tell "injected crash" from a bug

KNOWN_POINTS = (
    "after_wal_write",
    "after_wal_fsync",
    "flush_after_tmp_written",
    "flush_after_segment_rename",
    "flush_after_manifest",
    "flush_after_wal_delete",
)

# Graph snapshot publication (store/graphfile.py); separate because only publish_graph reaches them.
GRAPH_POINTS = ("graph_after_file", "graph_after_manifest")

# Compaction (store/index.py IndexStore.compact)
COMPACT_POINTS = ("compact_after_tmp_written", "compact_after_segment_rename",
                  "compact_after_manifest", "compact_after_old_deleted")

_hits: dict = {}


def crashpoint(name: str) -> None:
    target = os.environ.get("PAPERLENS_CRASH_AT")
    if not target or target != name:
        return
    _hits[name] = _hits.get(name, 0) + 1
    after = int(os.environ.get("PAPERLENS_CRASH_AFTER", "1"))
    if _hits[name] >= after:
        os._exit(CRASH_EXIT_CODE)
