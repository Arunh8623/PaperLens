"""scripts/harvest_years.py end to end against the local fake arXiv server (real subprocesses, real 3 s spacing)."""
import os
import subprocess
import sys

import crashlib as cl
from paperlens.contracts.paper import read_jsonl
from paperlens.harvest.fake_oai_server import FakeOAIServer
from paperlens.synth import generate_papers

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "scripts", "harvest_years.py")


def test_script_harvests_each_year_window_and_merges(tmp_path):
    papers = generate_papers(120, seed=5)                      # created/datestamp years 2020..2024
    merged = tmp_path / "merged.jsonl"
    with FakeOAIServer(papers, page_size=1000) as srv:
        r = subprocess.run([sys.executable, SCRIPT, "--contact", "t@example.com", "--first", "2021", "--last", "2023",
                            "--windows", "01-01:12-31", "--per-window", "1000", "--out-dir", str(tmp_path / "w"),
                            "--merged", str(merged), "--base-url", srv.url],
                           env=cl.child_env(), capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    got = {p.key for p in read_jsonl(str(merged))}
    assert {p.key for p in papers if p.datestamp[:4] in ("2021", "2022", "2023")} == got   # 3 windows, one per year
    assert "merged" in r.stdout and "next: python -m paperlens index" in r.stdout
