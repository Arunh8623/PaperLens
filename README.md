# PaperLens

A crash-safe hybrid (keyword + semantic) search engine for computer-science research
papers (arXiv `cs.*`), with a persistent on-disk index we design ourselves, fed by a
multi-process pipeline. KLE Technological University, CSE mini project
("Multi-Process Search Engine with Persistent Index"), guide Ms. Umadevi F M.

**Status: Phase 0 (foundations) and the first half of Phase 1 (keyword index on disk, Python reference) are built. Everything else is planned.** See
[docs/PHASE0.md](docs/PHASE0.md) and [docs/PHASE1A.md](docs/PHASE1A.md) for exactly what is and is not done,
[docs/CONTRACTS.md](docs/CONTRACTS.md) for the agreements between engines,
[OWNERS.md](OWNERS.md) for who owns what, and
[docs/TRACEABILITY.md](docs/TRACEABILITY.md) for requirement -> code -> test.

## AI-assistance disclosure

The code, tests and docs in this repository were written by Claude (an AI assistant from
Anthropic) at Arun's request, in a sandbox, and committed under Arun's git identity for
convenience. Commits from 2026-11-01 onward carry a `Co-Authored-By: Claude` trailer; the
earlier Phase 0 commits (10 of them) do not, so this paragraph is the record for those. Check your college's policy
on declaring AI assistance and declare it. Whoever presents the project must read, run and be able
to explain the code. The tests were run by the assistant on Linux; re-run them on your own machine
(see "Check your Windows machine").

## Setup

Needs Python 3.10 or newer. Phase 0 has **no third-party runtime dependencies** (pytest
is only for tests).

**Windows (PowerShell)**
```powershell
git clone <your-repo-url> paperlens; cd paperlens
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope Process Bypass
python -m pip install -U pip
python -m pip install -e ".[dev]"
python -m pytest
```

**Linux / macOS**
```bash
git clone <your-repo-url> paperlens && cd paperlens
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
python -m pytest
```

## Optional C++ accelerator (Phase 1b)

The project runs without it (pure Python, with a one-time warning). It makes search several times faster on the same data.

**Windows (PowerShell)** - needs the C++ compiler once:
```powershell
winget install Microsoft.VisualStudio.2022.BuildTools --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
python -m pip install pybind11 cmake
python scripts\build_native.py
python -c "import paperlens.native as n; print(n.status())"     # have_native should be True
python -m pytest tests\test_native.py                           # C++ must equal the Python twins
python bench\native_speedup.py --docs 20000 --out docs\evidence\native-speedup-mypc-SYNTHETIC.json
```
**Linux**: `sudo apt install g++ cmake` (or your distro's equivalent), then the same `pip install` / `build_native.py` lines.
Switch it off any time with `$env:PAPERLENS_NATIVE = "0"` (PowerShell) or `PAPERLENS_NATIVE=0` (bash).
**Never built on Windows by me** - if the build fails, send me the last 30 lines of its output.

## Check your Windows machine (please do this first)

The code was developed and tested on Linux. Windows differs in exactly the places crash-safety
depends on (renaming folders that hold open files, memory-mapped files, locking, newlines). After
the setup above, run one command and send back the report it writes:

```powershell
python scripts\windows_check.py
```

It runs small probes of how your filesystem behaves, then the whole test suite, and writes
`windows-check-report.txt`. Tests that use `kill -9` are skipped on Windows by design; the crash
tests that exit the process at named steps do run.

The C++ extension is **not part of Phase 0**. It arrives in Phase 1b, behind a pure-Python
fallback, with exact Visual Studio Build Tools / Linux compiler steps added here then.

## Run the tests and record evidence

```
python -m pytest                      # quick
python scripts/record_evidence.py     # tests + dated JSON in docs/evidence/ (commit hash, counts)
```

## Try it

```
python -m paperlens synth --n 1000 --seed 1 --out data/synth.jsonl   # SYNTHETIC data, plumbing only
python -m paperlens stats data/synth.jsonl
```

## Build and search a persistent index

```
python -m paperlens synth --n 2000 --seed 1 --out data/synth.jsonl
python -m paperlens index --index data/ix --input data/synth.jsonl       # diary -> segments -> MANIFEST
python -m paperlens search --index data/ix -k 5 fsync recovery           # BM25
python -m paperlens index-stats --index data/ix
python -m paperlens fsck --index data/ix                                 # read-only audit, exit 1 if damaged
```

Try the crash demo by hand (Linux/macOS): start an `index` of a big file, `kill -9` it, then run
`fsck` (it will only warn about leftovers) and `index-stats` (recovery runs on open).
Re-running `index` with the same file is safe: duplicates are ignored.

## Parallel ingestion (Phase 2)

```powershell
python -m paperlens synth --n 5000 --seed 1 --out data\synth.jsonl
python -m paperlens ingest --index data\ix --input data\synth.jsonl --readers 4
python -m paperlens ingest --index data\ix2 --input data\synth.jsonl --readers 4 --chaos   # kills workers on purpose
python -m paperlens fsck --index data\ix2
python bench\pipeline_speedup.py --papers 4000 --burn-ms 5 --out docs\evidence\speedup-mypc.json
```

`--chaos` kills a random reader every 0.25 s and the writer twice; the final index must still hold
every paper exactly once (`fsck` plus a search to check). The benchmark header explains why
`--burn-ms` exists (simulated model work) and what the numbers do and do not mean.

## Semantic and hybrid search (Phase 3)

```powershell
python -m pip install sentence-transformers          # real meaning vectors (downloads a model on first use)
python -m paperlens index --index data\ixs --input data\arxiv_cs.jsonl --embedder st:sentence-transformers/all-MiniLM-L6-v2
python -m paperlens search --index data\ixs "robust learning with noisy labels"                 # hybrid (default)
python -m paperlens search --index data\ixs --mode vector --cat cs.LG --year-from 2020 "noisy labels"
python -m paperlens search --index data\ixs --mode bm25 "noisy labels"
python bench\recall_vs_bits.py --index data\ixs --queries 300 --out docs\evidence\recall-bits-real.json
python -m paperlens beir --data data\scifact --embedder st:sentence-transformers/all-MiniLM-L6-v2 --out docs\evidence\beir-scifact-hybrid.json
```
Without `sentence-transformers`, try everything with the meaningless stand-in `--embedder hash:64`
(the plumbing works, the results mean nothing). Each hybrid result prints *why*: its keyword rank and
its meaning rank.

## Citation graph and near-duplicates (Phase 4)

```powershell
$env:OPENALEX_API_KEY = "your-key"        # optional but recommended: a free key gives a bigger daily budget (check openalex.org docs)
python -m paperlens citations --index data\ixs --out data\cites.jsonl --contact you@example.org --max-spend-usd 0.50
python -m paperlens graph-build --index data\ixs --citations data\cites.jsonl     # publishes a graph snapshot into the index
python -m paperlens graph-stats --index data\ixs                                   # PageRank leaders, communities
python -m paperlens search --index data\ixs --graph-rerank "noisy labels"          # re-order using the citation graph
python -m paperlens dupes --index data\ixs --threshold 0.8                         # near-duplicate papers
python bench\graph_ablation.py --index data\ixs --queries data\my_queries.jsonl --out docs\evidence\graph-ablation-real.json
```
`citations` can be stopped and re-run: it continues where it stopped and never spends more than `--max-spend-usd`.
Whether `--graph-rerank` helps is decided by the ablation on YOUR judged queries (it prints "keep" or "switch off").

## Compaction and query workers (Phase 5)

```powershell
python -m paperlens ingest --index data\ixs --input data\papers.jsonl --compact    # writer merges segments in the background
python -m paperlens compact --index data\ixs --force                              # merge everything into one segment now
python -m paperlens serve-query --index data\ixs --workers 2 --mode hybrid        # read-only query processes; type queries
```
Query workers read a frozen snapshot of the flushed data and never block the writer; they switch to a newer snapshot by themselves.

## Harvest real arXiv metadata (do this on your own machine)

arXiv rules (checked 2026-10-31): 1 request per 3 s, one connection, all your machines
combined. The tool enforces this itself and will not go faster. Use your real email: it goes
in the User-Agent so arXiv can contact you if something misbehaves.

```
python -m paperlens harvest --contact you@college.edu --set cs --out data/arxiv_cs.jsonl --max-records 10000
```

Each request costs at least 3.2 s. How many records arrive per request is decided by
arXiv (we have not measured it yet), so the first run will tell us the real time for 10,000
records: note it down. If it stops (network, Ctrl+C, laptop sleeps), **run the same command again**:
it resumes from `data/arxiv_cs.jsonl.state.json`. Send the final printed summary and the
state file to the team: they are the evidence for Gate 0.

## Layout

```
src/paperlens/
  contracts/   PaperRecord, key rules, JSONL            (shared)
  store/       diary, MANIFEST, segments, IndexStore, fsck, lock, crash points  (E2)
  text/        science-aware tokenizer, varint postings codec   (E3 keyword)
  testing/     helpers that crash-test child processes import
  pipeline/    supervisor, reader/writer workers, shared memory, shared limiter  (E1)
  vectors/     embedders, 1-4 bit quantization, IVF, rank fusion (RRF)  (E3 semantic)
  graph/       CSR citation graph, PageRank/PPR/label propagation, MinHash+LSH, OpenAlex client, rerank  (E3 graph)
  native/      C++17 source of the optional accelerator (pybind11 + CMake); Python twins in src/paperlens/native.py
  server/      query worker processes on read-only snapshots (Phase 5)
  harvest/     arXiv OAI-PMH harvester, limiter, fake server   (E1, supporting)
  synth.py     seeded synthetic data (plumbing tests only)
  cli.py       python -m paperlens ...
native/        C++17 + pybind11 (Phase 1b, empty for now)
tests/  scripts/  docs/
```

## Web page, evaluation and demo (Phase 7)

    python -m pip install -e ".[web]"
    python -m paperlens serve-web --index <index-dir>          # open http://127.0.0.1:8000
    python scripts/evaluate_all.py --synthetic --out docs/evidence/final-sandbox-SYNTHETIC              # table: PaperLens vs SQLite FTS5 vs FAISS
    python scripts/evaluate_all.py --beir-dir <scifact-dir> --beir-dir <scidocs-dir> --embedder st:<model> --out docs/evidence/final-mypc
    python scripts/demo_kill9.py --rounds 3 --docs 4000         # kill -9 the writer, recover instantly (POSIX only)
    python -m paperlens similar <id> --index <dir>              # "more like this"
    python -m paperlens key-papers --index <dir>                # highest PageRank
