# Windows checklist (do these in order; nothing here has been run on Windows by me)

1. Python 3.10+; open PowerShell in the unzipped folder.
2. `python -m venv .venv ; .venv\Scripts\Activate.ps1`
3. `python -m pip install -e ".[dev]"`
4. `python scripts\windows_check.py`   (checks fsync, locking, spawn, mmap)
5. `python -m pytest`   (expect a few POSIX-only SIGKILL tests skipped; **report any failure**)
6. Optional C++: install "Visual Studio Build Tools" (C++ workload), then `python -m pip install pybind11 cmake` and `python scripts\build_native.py`; re-run step 5. Without it everything falls back to Python with a warning.
7. Web: `python -m pip install -e ".[web]"`, then `python -m paperlens serve-web --index <dir>`.
8. Real data: `python -m paperlens harvest ...` (see README, 1 request / 3 s, put your email in the User-Agent option); then `ingest`.
9. Real quality: download BEIR SciFact + SCIDOCS, `python -m pip install -e ".[semantic]"`, run `scripts\evaluate_all.py` with `--beir-dir ... --embedder st:... --out ...`.
10. OpenAlex: get a key from their site (rules may have changed: needs checking), pass it as documented in README.
11. Speed-up (NFR3) and 50k latency (NFR4): run `bench/pipeline_speedup.py` on a >= 4-core machine.

NOT tested by me: Windows of any kind, real arXiv/OpenAlex/BEIR, a real embedding model, power loss, NFR3, NFR4.
