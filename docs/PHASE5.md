# Phase 5: compaction, read snapshots, query server (Engine E2 + E1)

## What it does and why (for a classmate)
* **Why compaction.** Every flush adds a segment and a query must visit all of them; deleted papers still sit inside old
  segments and still count in BM25 statistics. *Compaction* (like an LSM-tree merge) rewrites neighbouring segments as one,
  dropping dead papers for good.
* **Policy** (`store/compaction.py::plan_run`): merge the first 4 neighbouring segments whose sizes are within 4x of each other
  (size-tiered); otherwise rewrite one segment that is >= 50% dead. Only NEIGHBOURS merge because a segment's tombstones apply to
  the segments before it. A tombstone survives a merge only if the paper it hides lives in an older segment outside the merge.
* **Safe publishing** (`IndexStore.compact`): build `.tmp-seg-N` (no lock held, so searches and writes continue) -> rename -> new MANIFEST
  (old run replaced by the new segment) -> delete old folders. Four crash points; recovery needed no new rule except "do not
  delete a segment a reader still pins".
* **Vectors** are carried over from their half-precision copies and re-quantized; a merged segment big enough gets a fresh IVF index.
* **Readers and pins.** A reader process (`IndexReader`) pins the segments of its snapshot with a lock-held file in `pins/`; the writer only
  deletes obsolete folders nobody pins; a dead reader's pin is recognised because its lock is free. (On Windows deleting files that
  are still mapped is impossible, which is exactly why the pins exist.)
* **Query server** (`server/query_server.py`): N worker processes, one private pipe each, each on its own snapshot, refreshing by themselves; a
  killed worker is replaced and the (read-only) request repeated. Every answer says which generation it came from.
* **Thread safety.** Public store operations take one re-entrant lock; compaction holds it only for planning and for the swap.
* Zone maps (min/max year, categories per segment) were added in Phase 3 and are used by filtered searches; they are carried through merges.

## Verified by running (Linux, Python 3.13): full suite 394 passed, 0 failed (`docs/evidence/tests-20261109-79da7e2-py3.13.18-linux.json`)
* Merge without deletions changes **no BM25 score** (to 1e-12), also after restart.
* Full compaction on messy streams (replacements + deletes, 3 seeds) equals the independent brute-force oracle exactly, before and after restart.
* Partial merge keeps exactly the tombstones that still hide older papers; unflushed deletes survive; a delete that lands **while the merge is being
  built** (with and without a flush in between) is not lost.
* Vectors: exact top-15 identical before and after compaction, stored vectors bit-identical, IVF rebuilt, fsck clean.
* **16 crash tests with a real killed process** at each of the 4 compaction crash points (two occurrences each, plain and messy streams):
  fsck clean, no acknowledged paper lost or duplicated, and re-sending everything ends exactly at ground truth.
* Pins: pinned segments survive compaction and writer restart, vanish once released; a SIGKILLed reader's pin is ignored and removed.
* NFR5 stress: 2 reader processes queried in a loop while the writer flushes every 15 papers and a background thread merges: the snapshot
  invariant (live papers == last flushed seq), non-decreasing generations and duplicate-free hits held throughout; fsck clean; no pins left.
* Mutation check `scripts/compaction_mutation_check.sh`: 14 mutants. First run 13 killed, 1 survived (pending deletes during a merge); a test was
  added that injects deletes into the merge window; then 14/14 killed.

## NOT done / NOT tested / honest limits
* **Windows never run.** The pin lock (a byte at offset 2^30 via `msvcrt.locking`), deleting merged folders while another process has them open,
  and spawn-based query workers are written for Windows but unproven there.
* The reader pins BEFORE confirming its MANIFEST (closes a race), but no test provokes that race; it is argued, not demonstrated.
* A reader sees flushed data only (not the writer's buffer/diary). Search latency at 50,000 papers (NFR4) and merge throughput are **not measured**.
* Compaction of the citation graph is not needed (graphs are separate snapshots). The pipeline writer's `--compact` was exercised by one CLI test only.
* Merge of very large segments reads whole segments into memory (fine for tens of thousands of papers; not streaming).

## Next step
Phase 1b: C++ core (postings codec + BM25 top-k, popcount scan) with Python fallback.
