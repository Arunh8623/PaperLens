"""Reindex: rebuild an index with the CURRENT tokenizer.

What it does and why (plain language)
-------------------------------------
Terms in an index are produced by the tokenizer. If the tokenizer changes, old
segments would give wrong answers, so opening them is refused (see
TOKENIZER_VERSION in segment.py). The papers themselves are stored in full in
the segments, so nothing is lost: reindex reads every LIVE paper out of the old
index and puts it into a brand-new index built with the current tokenizer.

The old index is not modified (only opened to read, which also replays its diary
as any open does). Afterwards you swap the folders yourself, e.g.
    ren ix ix.old ; ren ix.new ix        (PowerShell: Rename-Item)
"""
from __future__ import annotations

import os

from .index import IndexStore


class ReindexError(Exception):
    pass


def _embedder_of(src) -> str:
    import json
    import os
    try:
        with open(os.path.join(src, "MANIFEST"), "rb"):
            pass
        seg_root = os.path.join(src, "segments")
        for name in sorted(os.listdir(seg_root), reverse=True):
            meta = os.path.join(seg_root, name, "meta.json")
            if os.path.exists(meta):
                with open(meta, "rb") as f:
                    v = json.loads(f.read().decode("utf-8")).get("vectors")
                if v:
                    return v.get("embedder", "")
    except OSError:
        pass
    return ""


def reindex(src, dst, flush_docs: int = 5000, batch: int = 256, log=print) -> dict:
    src, dst = os.fspath(src), os.fspath(dst)
    if os.path.abspath(src) == os.path.abspath(dst):
        raise ReindexError("source and destination must be different folders")
    if os.path.exists(os.path.join(dst, "MANIFEST")) or os.path.isdir(os.path.join(dst, "wal")):
        raise ReindexError(f"{dst} already holds an index; choose a new folder")
    if not os.path.exists(os.path.join(src, "MANIFEST")) and not os.path.isdir(os.path.join(src, "wal")):
        raise ReindexError(f"{src} does not look like an index")
    with IndexStore(src, allow_tokenizer_mismatch=True) as old, \
            IndexStore(dst, flush_docs=flush_docs, embedder_name=_embedder_of(src)) as new:
        ids = sorted(old.live)
        for n, arxiv_id in enumerate(ids, 1):
            new.put(old.get(arxiv_id), vector=old.get_vector(arxiv_id))        # vectors are kept, not recomputed
            if n % batch == 0:
                new.commit()
        new.commit()
        new.flush()
        if len(new.live) != len(ids):
            raise ReindexError(f"expected {len(ids)} live papers, new index has {len(new.live)}")
        out = {"papers": len(ids), "segments": new.stats()["segments"]}
    log(f"reindexed {out['papers']} papers into {dst}")
    return out
