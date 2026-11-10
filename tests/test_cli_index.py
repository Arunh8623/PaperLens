import os
import subprocess
import sys

from paperlens.contracts.paper import write_jsonl
from paperlens.synth import generate_papers, messy_stream, TOPICS

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))


def cli(*args):
    env = dict(os.environ, PYTHONPATH=SRC)
    return subprocess.run([sys.executable, "-m", "paperlens", *args], env=env,
                          capture_output=True, text=True, timeout=120)


def test_index_search_stats_fsck_roundtrip(tmp_path):
    data = tmp_path / "p.jsonl"
    stream = messy_stream(generate_papers(120, seed=5), seed=5)
    write_jsonl(data, stream)
    ix = str(tmp_path / "ix")
    r = cli("index", "--index", ix, "--input", str(data), "--flush-docs", "50")
    assert r.returncode == 0, r.stderr
    # replace/stale/duplicate split depends on the shuffled arrival order, so assert the invariants:
    import ast
    outcomes = ast.literal_eval(r.stdout.split("outcomes:")[1].split("\n")[0].strip())
    assert outcomes["insert"] == 120                       # exactly one insert per paper (no deletes here)
    assert sum(outcomes.values()) == len(stream) == 150    # every item got an outcome
    assert "'live_docs': 120" in r.stdout

    r = cli("search", "--index", ix, "-k", "3", TOPICS["storage"][1])        # "fsync"
    assert r.returncode == 0 and r.stdout.count("\n") >= 4
    assert " 1. " in r.stdout and "synth." in r.stdout

    assert cli("search", "--index", ix, "zzzznotaword").stdout.strip() == "no results"
    r = cli("index-stats", "--index", ix)
    assert "live_docs" in r.stdout and "120" in r.stdout
    r = cli("fsck", "--index", ix)
    assert r.returncode == 0 and "RESULT : OK" in r.stdout


def test_reindexing_same_file_changes_nothing(tmp_path):
    data = tmp_path / "p.jsonl"
    write_jsonl(data, generate_papers(60, seed=6))
    ix = str(tmp_path / "ix")
    cli("index", "--index", ix, "--input", str(data))
    r = cli("index", "--index", ix, "--input", str(data))
    assert "'already_stored': 60" in r.stdout and "'live_docs': 60" in r.stdout


def test_fsck_exit_code_on_damage(tmp_path):
    data = tmp_path / "p.jsonl"
    write_jsonl(data, generate_papers(30, seed=7))
    ix = tmp_path / "ix"
    cli("index", "--index", str(ix), "--input", str(data))
    f = ix / "segments" / "seg-000001" / "docs.bin"
    b = bytearray(f.read_bytes()); b[10] ^= 1; f.write_bytes(bytes(b))
    r = cli("fsck", "--index", str(ix))
    assert r.returncode == 1 and "DAMAGED" in r.stdout and "checksum" in r.stdout


def test_cli_reports_a_locked_index(tmp_path):
    from paperlens.store.index import IndexStore
    ix = tmp_path / "ix"
    with IndexStore(ix):
        r = cli("search", "--index", str(ix), "graph")
        assert r.returncode == 1 and "locked" in r.stderr


def test_cli_semantic_index_search_modes_and_filters(tmp_path, capsys):
    from paperlens.cli import main
    src = tmp_path / "p.jsonl"
    assert main(["synth", "--n", "150", "--seed", "4", "--out", str(src)]) == 0
    ix = str(tmp_path / "ix")
    assert main(["index", "--index", ix, "--input", str(src), "--embedder", "hash:32", "--flush-docs", "60"]) == 0
    capsys.readouterr()
    assert main(["index-stats", "--index", ix]) == 0
    stats = capsys.readouterr().out
    assert "vector_dim" in stats and "hash:32" in stats
    # default mode on an index with vectors is hybrid and explains each hit
    assert main(["search", "--index", ix, "-k", "3", "checksum", "recovery"]) == 0
    out = capsys.readouterr().out
    assert "why: keyword rank" in out or "why: not found by keywords" in out
    for mode in ("bm25", "vector", "hybrid"):
        assert main(["search", "--index", ix, "--mode", mode, "-k", "3", "checksum", "recovery"]) == 0
        assert capsys.readouterr().out.strip()
    # a filter that nothing satisfies
    assert main(["search", "--index", ix, "--cat", "cs.NOPE", "checksum"]) == 0
    assert "no results" in capsys.readouterr().out
    # a mismatching embedder is refused clearly
    assert main(["search", "--index", ix, "--embedder", "hash:16", "checksum"]) == 1
    assert "makes 16-number vectors, the index has 32" in capsys.readouterr().err


def test_cli_vector_mode_on_an_index_without_vectors_is_a_clear_error(tmp_path, capsys):
    from paperlens.cli import main
    src = tmp_path / "p.jsonl"
    main(["synth", "--n", "20", "--seed", "4", "--out", str(src)])
    ix = str(tmp_path / "ix")
    main(["index", "--index", ix, "--input", str(src)])
    capsys.readouterr()
    assert main(["search", "--index", ix, "--mode", "vector", "checksum"]) == 1
    assert "no vectors" in capsys.readouterr().err
    assert main(["search", "--index", ix, "checksum"]) == 0              # default falls back to bm25
