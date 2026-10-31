# PaperLens

A crash-safe hybrid (keyword + semantic) search engine for computer-science research
papers (arXiv `cs.*`), with a persistent on-disk index we design ourselves, fed by a
multi-process pipeline. KLE Technological University, CSE mini project
("Multi-Process Search Engine with Persistent Index"), guide Ms. Umadevi F M.

**Status: Phase 0 (foundations) built. Everything else is planned.** See
[docs/PHASE0.md](docs/PHASE0.md) for exactly what is and is not done,
[docs/CONTRACTS.md](docs/CONTRACTS.md) for the agreements between engines,
[OWNERS.md](OWNERS.md) for who owns what, and
[docs/TRACEABILITY.md](docs/TRACEABILITY.md) for requirement -> code -> test.

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
  store/       diary, MANIFEST, crash points, fs utils  (E2)
  text/        science-aware tokenizer                  (E3 keyword)
  harvest/     arXiv OAI-PMH harvester, limiter, fake server   (E1, supporting)
  synth.py     seeded synthetic data (plumbing tests only)
  cli.py       python -m paperlens ...
native/        C++17 + pybind11 (Phase 1b, empty for now)
tests/  scripts/  docs/
```
