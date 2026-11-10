# Phase 4: citation graph, near-duplicates, OpenAlex (Engine E3 graph side + E1 supporting fetcher)

## What it does and why (for a classmate)
* **Citation graph.** Papers that cite each other are about the same agenda, not just the same topic words.
  We fetch each paper's reference list from OpenAlex, keep only citations between papers *we hold*, and store the
  result as a compact CSR graph (`graph/csr.py`: offsets + targets arrays, like a sparse matrix).
* **PageRank** (`graph/rank.py`): power iteration; a paper is important if important papers cite it. Papers that cite
  nobody ("dangling") spread their score by the teleport distribution so the total stays 1.
* **Personalized PageRank**: the same walk but teleporting only back to chosen seed papers = "what is close to these?".
* **Label propagation**: every paper starts as its own community; each round it adopts the most common label among
  its neighbours. Nodes are visited in a seeded random order (a first version updated all nodes at once and two linked
  papers swapped labels forever; a test found that).
* **Citation rerank** (`graph/rerank.py`): take the top results as seeds, run Personalized PageRank, and merge that ordering with the
  original one by Reciprocal Rank Fusion. It is optional and **must earn its place**: `bench/graph_ablation.py` measures nDCG@10 with
  and without it and prints "keep" or "switch off".
* **Near-duplicates** (`graph/minhash.py`): word 3-grams -> MinHash signatures (128 numbers) -> LSH banding (16 bands x 8 rows) finds
  candidate pairs fast -> exact Jaccard confirms them -> union-find groups them. Reported pairs are real; the only error is a
  missed pair (`miss_probability`: at Jaccard 0.9 it is below 0.1%).
* **OpenAlex client** (`graph/openalex.py`): 50 DOIs per request (arXiv DOIs are `10.48550/arXiv.<id>`), every request first
  reserves its cost in the shared budget, 429/5xx are retried with `Retry-After`, a 403 `key_not_activated` stops with a clear
  message, results are appended + fsynced per batch so a stopped run resumes.
* **Graph snapshot through the MANIFEST** (`store/graphfile.py`): same idea as segments: write file, fsync, rename, THEN switch the
  MANIFEST's `graph_version`. Two new crash points; recovery deletes leftovers; fsck verifies the file checksum.

## Files
`graph/{csr,rank,rerank,minhash,openalex,fake_openalex}.py`, `store/graphfile.py` (+ edits in `index.py`, `fsck.py`, `crashpoints.py`),
CLI `citations`, `graph-build`, `graph-stats`, `dupes`, `search --graph-rerank`, `bench/graph_ablation.py`,
`scripts/graph_mutation_check.sh`, tests `test_graph.py`, `test_minhash.py`, `test_graph_publish.py`, `test_cli_graph.py`.

## Verified by running (Linux, Python 3.13): full suite 355 passed, 0 failed (`docs/evidence/tests-20261108-9dc5320-py3.13.18-linux.json`)
* PageRank and Personalized PageRank equal an independent direct solution `(I - alpha M) r = (1 - alpha) t` (dense linear algebra)
  to 1e-9 on random graphs with dangling nodes; hand-checked 4-node case.
* LSH+exact-check pairs equal a brute-force all-pairs Jaccard oracle; near-miss collisions are rejected.
* MinHash estimates track exact Jaccard (within 0.08 at 512 permutations); signatures identical across processes with different `PYTHONHASHSEED`.
* OpenAlex client against a local fake server: batching, resume, torn last line, 429/503/dropped connection retries, own-budget cap never
  exceeded (exact request count), server-side remaining-budget stop, inactive-key message, API key never in output.
* Graph publish: reopen, corrupt file detected by fsck and open, **crash at both graph crash points** leaves exactly one consistent version,
  recovery removes the leftover, fsck clean.
* CLI end to end (`test_cli_graph.py`): index -> citations -> graph-build -> graph-stats -> search --graph-rerank -> dupes -> fsck.
* Mutation check: 15 mutants of Phase 4 code, run `bash scripts/graph_mutation_check.sh` (commit first). First run: 12 killed, 3 survived
  (a meaningless tie-break mutant, an LSH verification mutant, a checksum mutant), tests were strengthened, then those 3 were killed.

## NOT done / NOT tested / honest limits
* **Never run against the real OpenAlex.** The fake server follows the documentation I read on 2026-11-08, not a live response. **Needs checking:**
  the response field names, the OpenAlex id format in `referenced_works`, the daily-budget numbers, and whether a key is *required*
  (my sources disagreed: the auth page says optional, one announcement title says required since Feb 13). Run `citations` on 100 papers first.
* **Whether citation rerank improves results is unknown.** `docs/evidence/graph-ablation-SYNTHETIC.json` is at the ceiling (nDCG 1.0 before
  reranking), so it only shows that the keep/switch-off rule works. The real verdict needs ~50 judged queries and an OpenAlex graph.
* The graph is rebuilt from scratch each time (no incremental update); fine for tens of thousands of papers, untested beyond ~400 nodes.
* Compaction does not carry the graph (it is independent of segments).
* MinHash is Python/numpy; a C++ version is only worth it if profiling on real data says so (Phase 1b).
* Window/Linux note: `msvcrt`/Windows behaviour of the new `graphs/` folder deletion uses `retry_fs`, untested on Windows.

## Next step
Phase 5: compaction (including vectors), snapshots/pinning, query server process.
