import os
import subprocess
import sys

import pytest

import paperlens.cli as cli
from paperlens.contracts.paper import read_jsonl
from paperlens.harvest.fake_oai_server import FakeOAIServer
from paperlens.synth import generate_papers


def run_cli(*args):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
    return subprocess.run([sys.executable, "-m", "paperlens", *args], env=env,
                          capture_output=True, text=True, timeout=60)


def test_synth_and_stats_via_real_subprocess(tmp_path):
    out = tmp_path / "d" / "s.jsonl"
    r = run_cli("synth", "--n", "40", "--seed", "3", "--out", str(out))
    assert r.returncode == 0 and "wrote 40" in r.stdout
    assert len(list(read_jsonl(out))) == 40
    r = run_cli("stats", str(out))
    assert r.returncode == 0 and "'unique_keys': 40" in r.stdout


def test_harvest_requires_contact(tmp_path, monkeypatch):
    monkeypatch.delenv("PAPERLENS_CONTACT", raising=False)
    assert cli.main(["harvest", "--out", str(tmp_path / "o.jsonl")]) == 2


def test_harvest_refuses_to_go_faster_than_the_rule(tmp_path, capsys):
    rc = cli.main(["harvest", "--contact", "a@b.org", "--min-interval", "1.0",
                   "--out", str(tmp_path / "o.jsonl")])
    assert rc == 2 and "must be >=" in capsys.readouterr().err


def test_harvest_end_to_end_against_fake_server(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "ARXIV_RULE_SECONDS", 0.0)       # test-only: skip the 3 s floor
    papers = generate_papers(120, seed=4)
    out = str(tmp_path / "o.jsonl")
    with FakeOAIServer(papers, page_size=50) as srv:
        rc = cli.main(["harvest", "--contact", "a@b.org", "--min-interval", "0.01",
                       "--base-url", srv.url, "--out", out])
    assert rc == 0
    assert "'unique_keys': 120" in capsys.readouterr().out
    assert os.path.exists(out + ".state.json")


def test_harvest_failure_exits_nonzero_and_says_how_to_resume(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "ARXIV_RULE_SECONDS", 0.0)
    papers = generate_papers(10, seed=4)
    with FakeOAIServer(papers, faults={1: ("http", 404, None)}) as srv:
        rc = cli.main(["harvest", "--contact", "a@b.org", "--min-interval", "0.01",
                       "--base-url", srv.url, "--out", str(tmp_path / "o.jsonl")])
    assert rc == 1 and "re-run the same command" in capsys.readouterr().err
