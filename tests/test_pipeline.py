"""End-to-end tests of the multi-process pipeline (real spawned processes, real kills)."""
import dataclasses

import pytest

from paperlens.contracts.paper import PaperRecord
from paperlens.pipeline.shared_limiter import SharedLimiter
from paperlens.pipeline.supervisor import PipelineConfig, PipelineError, run_pipeline
from paperlens.pipeline.workers import Faults
from paperlens.store.crashpoints import KNOWN_POINTS
from paperlens.store.fsck import fsck
from paperlens.store.index import IndexStore
from paperlens.synth import expected_final, generate_papers, messy_stream
from paperlens.vectors.embedders import HashEmbedder
from tests.crashlib import expected_vector
import numpy as np


def cfg(tmp_path, **kw):
    base = dict(index_path=str(tmp_path / "ix"), n_readers=2, dim=16, flush_docs=50,
                heartbeat_interval=0.1, job_timeout=1.0, hang_timeout=1.5, timeout=90)
    base.update(kw)
    return PipelineConfig(**base)


def check_index(path, stream, dead=()):
    """The index must hold EXACTLY the newest version of every paper (minus dead letters)."""
    want = {a: p for a, p in expected_final(stream).items() if a not in dead}
    with IndexStore(path) as st:
        assert set(st.live) == set(want)
        for a, p in want.items():
            assert st.live[a] == p.key, a
            assert st.get(a) == p
            # the vector that travelled through shared memory arrived intact and belongs to THIS version
            assert np.array_equal(st.get_vector(a), expected_vector(p, 16)), a
    rep = fsck(path)
    assert rep.ok, rep.errors


def test_plain_run_stores_everything_once(tmp_path):
    papers = generate_papers(300, 1)
    rep = run_pipeline(papers, cfg(tmp_path))
    assert rep.jobs == 300 and rep.outcomes == {"insert": 300} and not rep.dead_letters
    assert rep.reader_respawns == rep.writer_respawns == 0
    assert rep.writer_stats["commits"] < 300                       # group commit: far fewer fsyncs than papers
    check_index(tmp_path / "ix", papers)


def test_result_equals_single_process_indexing(tmp_path):
    # Plain stream on purpose: BM25 statistics count replaced (dead) documents until compaction,
    # so with replacements the arrival ORDER would legitimately change scores slightly.
    papers = generate_papers(150, 4)
    run_pipeline(papers, cfg(tmp_path))
    with IndexStore(tmp_path / "single") as st:
        for p in papers:
            st.put(p)
        st.commit()
        single = {q: st.search(q, k=None) for q in ("checksum", "recovery", "index")}
    with IndexStore(tmp_path / "ix") as st:
        assert {q: st.search(q, k=None) for q in single} == single      # same scores, same order


def test_messy_stream_duplicates_and_versions(tmp_path):
    papers = messy_stream(generate_papers(200, 2), seed=2)
    rep = run_pipeline(papers, cfg(tmp_path))
    assert rep.jobs == len(papers) > 200
    check_index(tmp_path / "ix", papers)


def test_rerun_of_same_input_changes_nothing(tmp_path):
    papers = generate_papers(120, 3)
    run_pipeline(papers, cfg(tmp_path))
    rep = run_pipeline(papers, cfg(tmp_path))
    assert rep.outcomes == {"duplicate": 120}
    check_index(tmp_path / "ix", papers)


def test_reader_killed_on_a_paper_once_is_retried(tmp_path):
    papers = generate_papers(100, 5)
    f = Faults(crash_once_on_key=frozenset({papers[10].arxiv_id, papers[50].arxiv_id}))
    rep = run_pipeline(papers, cfg(tmp_path, faults=f))
    assert rep.reader_respawns >= 2 and not rep.dead_letters
    check_index(tmp_path / "ix", papers)


def test_poison_paper_is_parked_everything_else_finishes(tmp_path):
    papers = generate_papers(80, 6)
    bad = papers[20]
    rep = run_pipeline(papers, cfg(tmp_path, faults=Faults(crash_on_key=frozenset({bad.arxiv_id}))))
    assert [d[0] for d in rep.dead_letters] == [bad.key]
    assert rep.dead_letters[0][2] == 3                              # max_attempts
    assert rep.reader_respawns == 3
    check_index(tmp_path / "ix", papers, dead={bad.arxiv_id})


def test_innocent_neighbours_are_not_blamed(tmp_path):
    """Jobs queued behind the poison paper must not use up their own attempts."""
    papers = generate_papers(60, 7)
    bad = papers[0]
    rep = run_pipeline(papers, cfg(tmp_path, n_readers=1, max_inflight_per_reader=8,
                                   faults=Faults(crash_on_key=frozenset({bad.arxiv_id}))))
    assert len(rep.dead_letters) == 1 and rep.dead_letters[0][0] == bad.key
    check_index(tmp_path / "ix", papers, dead={bad.arxiv_id})


def test_exception_in_reader_does_not_kill_the_worker(tmp_path):
    papers = generate_papers(60, 8)
    bad = papers[5]
    rep = run_pipeline(papers, cfg(tmp_path, faults=Faults(raise_on_key=frozenset({bad.arxiv_id}))))
    assert [d[0] for d in rep.dead_letters] == [bad.key]
    assert rep.reader_respawns == 0
    check_index(tmp_path / "ix", papers, dead={bad.arxiv_id})


def test_paper_without_text_is_dead_lettered_at_once(tmp_path):
    papers = generate_papers(30, 9)
    empty = dataclasses.replace(papers[3], title="---", abstract="...")
    stream = papers[:3] + [empty] + papers[4:]
    rep = run_pipeline(stream, cfg(tmp_path))
    assert len(rep.dead_letters) == 1 and rep.dead_letters[0][2] == 1     # no wasted retries
    check_index(tmp_path / "ix", stream, dead={empty.arxiv_id})


def test_hung_reader_is_detected_killed_and_replaced(tmp_path):
    papers = generate_papers(60, 10)
    f = Faults(hang_once_on_key=frozenset({papers[7].arxiv_id}))
    rep = run_pipeline(papers, cfg(tmp_path, faults=f, job_timeout=0.4, hang_timeout=0.8))
    assert rep.hang_kills >= 1 and rep.reader_respawns >= 1 and not rep.dead_letters
    check_index(tmp_path / "ix", papers)


def test_corrupt_vector_is_rejected_then_retried(tmp_path):
    papers = generate_papers(60, 11)
    f = Faults(corrupt_vector_once_on_key=frozenset({papers[9].arxiv_id, papers[30].arxiv_id}))
    rep = run_pipeline(papers, cfg(tmp_path, faults=f))
    assert rep.rejected_retries == 2 and not rep.dead_letters
    check_index(tmp_path / "ix", papers)


def test_startup_crash_loop_is_reported_clearly(tmp_path):
    with pytest.raises(PipelineError, match="before they are ready"):
        run_pipeline(generate_papers(10, 1), cfg(tmp_path, faults=Faults(die_at_start_readers=True),
                                                 startup_fail_limit=3))


def test_respawn_limit_stops_endless_loops(tmp_path):
    papers = generate_papers(40, 12)
    f = Faults(crash_on_key=frozenset(p.arxiv_id for p in papers))     # everything is poison
    with pytest.raises(PipelineError, match="respawn limit"):
        run_pipeline(papers, cfg(tmp_path, faults=f, max_respawns=4, max_attempts=50))


@pytest.mark.parametrize("point", KNOWN_POINTS)
def test_writer_killed_at_every_crash_point(tmp_path, point):
    papers = generate_papers(150, 13)
    rep = run_pipeline(papers, cfg(tmp_path, flush_docs=40, writer_batch=16,
                                   writer_crash_at=point, writer_crash_after=2))
    assert rep.writer_respawns == 1, rep
    assert not rep.dead_letters
    check_index(tmp_path / "ix", papers)


def test_shared_limiter_spaces_all_readers_together(tmp_path):
    import multiprocessing as mp
    lim = SharedLimiter(0.04, ctx=mp.get_context("spawn"))
    papers = generate_papers(40, 14)
    rep = run_pipeline(papers, cfg(tmp_path, n_readers=3, reader_limiter=lim))
    assert rep.limiter_min_gap >= 0.04 - 1e-6
    assert len(lim.history()) == 40
    check_index(tmp_path / "ix", papers)


def test_chaos_readers_killed_every_quarter_second_and_writer_twice(tmp_path):
    papers = messy_stream(generate_papers(300, 15), seed=15)
    rep = run_pipeline(papers, cfg(tmp_path, n_readers=3, flush_docs=60, writer_batch=16,
                                   faults=Faults(slow_seconds=0.03), chaos_reader_interval=0.25,
                                   chaos_writer_kills=2, chaos_writer_interval=0.8, chaos_seed=1,
                                   max_attempts=6, timeout=120))
    assert rep.chaos_reader_kills >= 3 and rep.chaos_writer_kills == 2
    assert rep.writer_respawns >= 2 and not rep.dead_letters
    assert rep.replace_seconds_max < 5.0                              # NFR2: replaced within 5 s
    check_index(tmp_path / "ix", papers)


def test_cli_ingest_then_search(tmp_path, capsys):
    from paperlens.cli import main
    src = tmp_path / "p.jsonl"
    assert main(["synth", "--n", "120", "--seed", "3", "--out", str(src)]) == 0
    assert main(["ingest", "--index", str(tmp_path / "ix"), "--input", str(src), "--readers", "2",
                 "--flush-docs", "50"]) == 0
    out = capsys.readouterr().out
    assert "{'insert': 120}" in out and "dead letters : []" in out
    assert main(["fsck", "--index", str(tmp_path / "ix")]) == 0
    assert main(["ingest", "--index", str(tmp_path / "ix"), "--input", str(src)]) == 0   # re-run: all duplicates
    assert "{'duplicate': 120}" in capsys.readouterr().out


def test_vectors_are_searchable_after_a_pipeline_run(tmp_path):
    papers = generate_papers(200, 30)
    run_pipeline(papers, cfg(tmp_path, flush_docs=60, ivf_min_docs=40))
    E = HashEmbedder(16)
    with IndexStore(tmp_path / "ix") as st:
        assert st.vec_dim == 16 and st.stats()["embedder"] == "hash:16"
        for p in papers[:10]:
            assert st.search_vector(E.embed_paper(p), k=1, exact=True)[0].arxiv_id == p.arxiv_id
            assert st.search_vector(E.embed_paper(p), k=1, nprobe=4)[0].score > 0.99
    assert fsck(tmp_path / "ix").ok


def test_store_vectors_can_be_switched_off(tmp_path):
    papers = generate_papers(40, 31)
    rep = run_pipeline(papers, cfg(tmp_path, store_vectors=False))
    assert not rep.dead_letters
    with IndexStore(tmp_path / "ix") as st:
        assert st.vec_dim == 0 and len(st.live) == 40


def test_index_with_other_vector_dimension_is_refused_clearly(tmp_path):
    with IndexStore(tmp_path / "ix") as st:
        p = generate_papers(1, 32)[0]
        st.put(p, vector=HashEmbedder(24).embed_paper(p))
    with pytest.raises(PipelineError, match="stores 24-number vectors"):
        run_pipeline(generate_papers(10, 33), cfg(tmp_path))


def test_embedder_that_disagrees_with_dim_is_reported_not_retried_forever(tmp_path):
    with pytest.raises(PipelineError, match="reader cannot start"):
        run_pipeline(generate_papers(10, 34), cfg(tmp_path, embedder="hash:32"))


def test_cli_ingest_with_embedder_then_vector_search(tmp_path, capsys):
    from paperlens.cli import main
    src = tmp_path / "p.jsonl"
    assert main(["synth", "--n", "100", "--seed", "5", "--out", str(src)]) == 0
    ix = str(tmp_path / "ix")
    assert main(["ingest", "--index", ix, "--input", str(src), "--embedder", "hash:32", "--readers", "2",
                 "--flush-docs", "40"]) == 0
    capsys.readouterr()
    assert main(["search", "--index", ix, "--mode", "vector", "-k", "3", "checksum", "recovery"]) == 0
    assert capsys.readouterr().out.count("synth.") >= 3
    assert main(["fsck", "--index", ix]) == 0
