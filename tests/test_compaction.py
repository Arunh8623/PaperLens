import os
import random
import threading
import time

import numpy as np
import pytest

from paperlens.store.compaction import plan_run
from paperlens.store.crashpoints import COMPACT_POINTS, CRASH_EXIT_CODE
from paperlens.store.fsck import fsck
from paperlens.store.index import IndexStore
from paperlens.store.pins import pinned_segments
from paperlens.store.reader import IndexReader, ReadOnlyIndex
from paperlens.synth import TOPICS, generate_papers, messy_stream
from paperlens.vectors.embedders import HashEmbedder
from tests import crashlib as cl
from tests.oracle import Model

QUERIES = ["inverted postings", "crash recovery journal", "pagerank citation graph", "scheduler latency", "gradient embedding"]


def scores(st, q):
    return {h.key: h.score for h in st.search(q, k=None)}


def assert_scores_equal(a: dict, b: dict):
    assert set(a) == set(b)
    for k in a:
        assert a[k] == pytest.approx(b[k], rel=1e-12, abs=1e-12)


def build(tmp_path, n=120, flush=20, name="ix", **kw):
    st = IndexStore(tmp_path / name, flush_docs=flush, **kw)
    for i, p in enumerate(generate_papers(n, 1)):
        st.put(p)
        if i % flush == flush - 1:
            st.commit()
            st.flush()
    st.commit()
    st.flush()
    return st


# ------------------------------------------------------------------ policy
def test_plan_run_picks_similar_sized_neighbours_then_mostly_dead_segments():
    assert plan_run([], []) is None
    assert plan_run([10, 10, 10], [0, 0, 0]) is None                          # fewer than fan_in
    assert plan_run([10, 10, 10, 10, 10], [0] * 5) == (0, 3)                  # leftmost window
    assert plan_run([1000, 50, 50, 50, 50], [0] * 5) == (1, 4)                # the big one is left alone
    assert plan_run([1000, 10, 500, 20], [0] * 4) is None                     # sizes too different
    assert plan_run([100, 100], [0, 60]) == (1, 1)                            # mostly dead: rewrite alone
    assert plan_run([100, 100], [0, 49]) is None
    assert plan_run([100, 200], [0, 0], force=True) == (0, 1)
    assert plan_run([100], [0], force=True) is None and plan_run([100], [3], force=True) == (0, 0)


# ------------------------------------------------------------------ correctness
def test_merge_without_deletes_changes_no_score_and_no_result(tmp_path):
    st = build(tmp_path)
    assert st.stats()["segments"] == 6
    before = {q: scores(st, q) for q in QUERIES}
    r = st.compact(fan_in=3, ratio=10)
    assert r["merged"] == ["seg-000001", "seg-000002", "seg-000003"] and r["dropped"] == 0
    assert st.stats()["segments"] == 4
    st.compact(force=True)
    assert st.stats()["segments"] == 1
    for q in QUERIES:
        assert_scores_equal(before[q], scores(st, q))                          # nothing was dead: identical BM25
    st.close()
    assert fsck(tmp_path / "ix").ok
    with IndexStore(tmp_path / "ix") as st2:
        assert st2.stats()["segments"] == 1 and st2.stats()["live_docs"] == 120
        for q in QUERIES:
            assert_scores_equal(before[q], scores(st2, q))


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_full_compaction_equals_the_independent_oracle_on_a_messy_stream(tmp_path, seed):
    stream = messy_stream(generate_papers(150, seed), seed)
    st, model = IndexStore(tmp_path / "ix", flush_docs=10**9), Model()
    rng = random.Random(seed)
    for i, p in enumerate(stream):
        st.put(p)
        model.put(p)
        if i % 25 == 24:
            st.commit()
            st.flush()
            model.flush()
        if i % 40 == 39:                                                       # some explicit deletes too
            victim = rng.choice(sorted(model.live))
            assert st.delete(victim) == model.delete(victim)
    st.commit()
    st.flush()
    model.flush()
    for q in QUERIES:                                                          # sanity: equal BEFORE compaction too
        assert_scores_equal({k: v for k, v in model.bm25(q).items()}, scores(st, q))
    physical_before = st.stats()["physical_docs"]
    r = st.compact(force=True)
    model.compact()
    assert st.stats()["segments"] == 1 and st.stats()["physical_docs"] == len(model.live) < physical_before
    assert r["dropped"] == physical_before - len(model.live)
    assert st.live_keys() == model.live_keys()
    for q in QUERIES:
        assert_scores_equal(model.bm25(q), scores(st, q))
    st.close()
    with IndexStore(tmp_path / "ix") as st2:                                   # and after a restart
        assert st2.live_keys() == model.live_keys()
        for q in QUERIES:
            assert_scores_equal(model.bm25(q), scores(st2, q))
    assert fsck(tmp_path / "ix").ok


def test_partial_merge_keeps_tombstones_that_still_hide_older_documents(tmp_path):
    st = IndexStore(tmp_path / "ix", flush_docs=10**9)
    papers = generate_papers(60, 4)
    for p in papers[:20]:
        st.put(p)
    st.commit(); st.flush()                                                    # seg1: papers 0-19 (will NOT be merged)
    for p in papers[20:30]:
        st.put(p)
    st.delete(papers[3].arxiv_id)                                              # tombstone for seg1 lands in seg2
    st.commit(); st.flush()
    for p in papers[30:40]:
        st.put(p)
    st.commit(); st.flush()
    assert st.stats()["segments"] == 3
    r = st.compact(fan_in=2, ratio=1.5)                                        # 20 vs 10 is too different: merges the two NEWER segments
    assert r["merged"] == ["seg-000002", "seg-000003"] and r["tombstones_kept"] == 1
    assert papers[3].arxiv_id not in st.live and st.get(papers[3].arxiv_id) is None
    assert papers[3].key not in {h.key for h in st.search(papers[3].title, k=None)}
    st.close()
    with IndexStore(tmp_path / "ix") as st2:
        assert st2.get(papers[3].arxiv_id) is None and st2.stats()["live_docs"] == 39
    assert fsck(tmp_path / "ix").ok
    with IndexStore(tmp_path / "ix") as st3:                                   # a merge that includes seg1 drops it for good
        r = st3.compact(force=True)
        assert r["tombstones_kept"] == 0 and r["docs_after"] == 39


def test_pending_unflushed_deletes_survive_a_compaction(tmp_path):
    st = build(tmp_path, n=60, flush=15)
    victim = generate_papers(60, 1)[7]
    st.delete(victim.arxiv_id)                                                 # not flushed yet
    st.compact(force=True)
    assert st.get(victim.arxiv_id) is None
    assert victim.key not in {h.key for h in st.search(victim.title, k=None)}
    st.commit(); st.flush()
    assert st.stats()["live_docs"] == 59
    st.close()
    with IndexStore(tmp_path / "ix") as st2:
        assert st2.get(victim.arxiv_id) is None and st2.stats()["live_docs"] == 59


def test_everything_dead_in_a_run_just_removes_the_run(tmp_path):
    st = IndexStore(tmp_path / "ix", flush_docs=10**9)
    ps = generate_papers(10, 2)
    for p in ps:
        st.put(p)
    st.commit(); st.flush()
    for p in ps:
        st.delete(p.arxiv_id)
    st.commit(); st.flush()                                                    # seg2 holds only tombstones
    r = st.compact(force=True)
    assert r["into"] is None and st.stats()["segments"] == 0 and st.stats()["live_docs"] == 0
    st.close()
    with IndexStore(tmp_path / "ix") as st2:
        assert st2.search("inverted postings") == []
    assert fsck(tmp_path / "ix").ok


def test_vectors_survive_compaction(tmp_path):
    emb = HashEmbedder(16)
    st = IndexStore(tmp_path / "ix", flush_docs=10**9, ivf_min_docs=30, vec_bits=2)
    papers = generate_papers(90, 3)
    for i, p in enumerate(papers):
        st.put(p, vector=emb.embed_paper(p))
        if i % 30 == 29:
            st.commit(); st.flush()
    st.delete(papers[5].arxiv_id)
    st.commit(); st.flush()
    qv = emb.embed_query("inverted postings segment")
    before = [(h.key, round(h.score, 6)) for h in st.search_vector(qv, k=15, exact=True)]
    vecs_before = {p.arxiv_id: st.get_vector(p.arxiv_id) for p in papers if p.arxiv_id in st.live}
    r = st.compact(force=True)
    assert r["into"] and st.stats()["segments"] == 1
    assert [(h.key, round(h.score, 6)) for h in st.search_vector(qv, k=15, exact=True)] == before
    for aid, v in vecs_before.items():
        assert np.array_equal(st.get_vector(aid), v)
    approx = {h.key for h in st.search_vector(qv, k=15)}
    assert len(approx & {k for k, _ in before}) >= 14                          # codes still find the neighbours
    assert st.segments[0].meta["vectors"]["ivf_nlist"] > 0 and st.segments[0].meta["vectors"]["bits"] == 2
    st.close()
    assert fsck(tmp_path / "ix").ok


def test_dead_ratio_triggers_a_single_segment_rewrite(tmp_path):
    st = IndexStore(tmp_path / "ix", flush_docs=10**9)
    ps = generate_papers(40, 6)
    for p in ps[:20]:
        st.put(p)
    st.commit(); st.flush()
    for p in ps[20:]:
        st.put(p)
    st.commit(); st.flush()
    for p in ps[:12]:
        st.delete(p.arxiv_id)
    st.commit()
    st.flush()
    assert st.compact(dead_ratio=0.7) is None                                  # 60% dead is below this bar
    r = st.compact()                                                           # default bar 50% -> seg1 rewritten alone
    assert r["merged"] == ["seg-000001"] and r["docs_after"] == 8 and st.stats()["live_docs"] == 28


# ------------------------------------------------------------------ readers, pins, snapshots
def test_reader_is_a_frozen_snapshot_and_refresh_moves_it_in_one_step(tmp_path):
    st = build(tmp_path, n=60, flush=20)
    rd = IndexReader(tmp_path / "ix")
    g0 = rd.generation
    snap = scores(rd, "inverted postings")
    for p in generate_papers(60, 9)[:40]:
        st.put(p)
    st.commit(); st.flush()
    st.compact(force=True)
    assert rd.generation == g0 and scores(rd, "inverted postings") == snap       # unchanged until refresh
    assert rd.refresh() and rd.generation == st.manifest.generation
    assert set(rd.live_keys()) == st.live_keys()
    assert not rd.refresh()
    with pytest.raises(ReadOnlyIndex):
        rd.put(generate_papers(1, 1)[0])
    with pytest.raises(ReadOnlyIndex):
        rd.compact()
    rd.close()
    st.close()


def test_obsolete_segments_stay_while_a_reader_pins_them_and_go_afterwards(tmp_path):
    st = build(tmp_path, n=60, flush=20)
    rd = IndexReader(tmp_path / "ix")
    old = set(rd.manifest.segments)
    st.compact(force=True)
    segdir = tmp_path / "ix" / "segments"
    assert old <= set(os.listdir(segdir))                                       # pinned: not deleted
    assert pinned_segments(tmp_path / "ix") == old
    assert scores(rd, "inverted postings")                                      # reader still works on its files
    rd.refresh()
    st.compact()                                                                # nothing to merge, but collects garbage
    assert set(os.listdir(segdir)) == set(st.manifest.segments)
    rd.close()
    assert os.listdir(tmp_path / "ix" / "pins") == []
    st.close()


def test_writer_restart_does_not_delete_a_segment_a_live_reader_pins(tmp_path):
    st = build(tmp_path, n=60, flush=20)
    rd = IndexReader(tmp_path / "ix")
    old = set(rd.manifest.segments)
    st.compact(force=True)
    st.close()
    with IndexStore(tmp_path / "ix") as st2:
        assert old <= set(os.listdir(tmp_path / "ix" / "segments"))
        assert st2.recovery["removed_unlisted"] == 0
        assert scores(rd, "inverted postings")
    rd.close()
    with IndexStore(tmp_path / "ix") as st3:                                    # reader gone: now they are garbage
        assert st3.recovery["removed_unlisted"] == len(old)


def test_a_dead_readers_pin_is_ignored_and_removed(tmp_path):
    import subprocess, sys
    st = build(tmp_path, n=40, flush=20)
    code = ("import sys,time\nfrom paperlens.store.reader import IndexReader\n"
            "r=IndexReader(sys.argv[1]);print('up',flush=True);time.sleep(60)\n")
    p = subprocess.Popen([sys.executable, "-c", code, str(tmp_path / "ix")], env=cl.child_env(),
                         stdout=subprocess.PIPE, text=True)
    assert p.stdout.readline().strip() == "up"
    assert pinned_segments(tmp_path / "ix") == set(st.manifest.segments)         # alive in ANOTHER process
    p.kill()
    p.wait()
    # Windows releases a dead process's byte-range lock a little LATER than Linux (Microsoft documents that the
    # unlock time depends on system resources), so poll briefly instead of checking once. Safe direction:
    # until the lock is gone the writer simply keeps the segments.
    import time
    deadline = time.time() + 5
    while pinned_segments(tmp_path / "ix") != set() and time.time() < deadline:
        time.sleep(0.1)
    assert pinned_segments(tmp_path / "ix") == set()                             # lock gone with the process
    assert os.listdir(tmp_path / "ix" / "pins") == []
    st.close()


# ------------------------------------------------------------------ background compaction
def test_background_compaction_while_writing_and_searching(tmp_path):
    st = IndexStore(tmp_path / "ix", flush_docs=15)
    st.start_background_compaction(interval=0.01, fan_in=3, ratio=50)
    papers = generate_papers(400, 8)
    errors = []
    stop = threading.Event()

    def searcher():
        try:
            while not stop.is_set():
                for q in QUERIES:
                    for h in st.search(q, k=5):
                        assert h.arxiv_id in st.live or True                    # must simply not raise or crash
                st.stats()
        except BaseException as e:                                              # noqa: BLE001
            errors.append(e)

    t = threading.Thread(target=searcher)
    t.start()
    for i, p in enumerate(papers):
        st.put(p)
        if i % 10 == 9:
            st.commit()
            time.sleep(0.002)
    st.commit()
    deadline = time.time() + 20
    while st._bg.runs == 0 and time.time() < deadline:
        time.sleep(0.05)
    stop.set()
    t.join()
    assert not errors, errors
    assert st._bg.last_error is None and st._bg.runs > 0
    st.stop_background_compaction()
    assert st.live_keys() == {p.key for p in papers}
    n_seg = st.stats()["segments"]
    assert n_seg < 400 // 15                                                    # many fewer than one per flush
    st.close()
    assert fsck(tmp_path / "ix").ok
    with IndexStore(tmp_path / "ix") as st2:
        assert st2.live_keys() == {p.key for p in papers}


# ------------------------------------------------------------------ crash safety (real killed processes)
def test_compact_points_are_registered():
    assert COMPACT_POINTS == ("compact_after_tmp_written", "compact_after_segment_rename",
                              "compact_after_manifest", "compact_after_old_deleted")


@pytest.mark.parametrize("point", COMPACT_POINTS)
@pytest.mark.parametrize("after", [1, 3])
@pytest.mark.parametrize("kind", ["plain", "messy"])
def test_crash_during_compaction_loses_and_duplicates_nothing(tmp_path, point, after, kind):
    ix, ack = tmp_path / "ix", tmp_path / "ack.txt"
    n, seed = 150, 3
    r = cl.run_child(ix, ack, n, seed, kind, batch=7, flush_every=20, crash_at=point, crash_after=after, compact=True)
    assert r.returncode == CRASH_EXIT_CODE, f"{point} never fired (exit {r.returncode}): {r.stderr[-500:]}"
    if kind == "plain":
        assert cl.read_acked(ack), "nothing acknowledged before the crash: weak test"
    cl.check_after_crash(ix, ack, kind)
    cl.finish_and_verify(ix, ack, n, seed, kind, flush_every=20)


@pytest.mark.parametrize("flush_in_gap", [False, True])
def test_a_delete_that_arrives_while_the_merge_is_being_built_is_not_lost(tmp_path, monkeypatch, flush_in_gap):
    """compact() builds the new segment WITHOUT the state lock, so writes can land in that window."""
    import paperlens.store.index as index_mod
    st = build(tmp_path, n=60, flush=15)
    papers = generate_papers(60, 1)
    victim, other = papers[4], papers[40]
    real = index_mod.merge_run

    def merge_then_delete(*a, **kw):
        out = real(*a, **kw)                                                   # the merged docs still contain the victim
        st.delete(victim.arxiv_id)
        if flush_in_gap:
            st.commit()
            st.flush()                                                         # its tombstone now sits in a NEWER segment
        else:
            st.delete(other.arxiv_id)                                          # stays pending (unflushed)
        return out

    monkeypatch.setattr(index_mod, "merge_run", merge_then_delete)
    r = st.compact(force=True)
    assert r["into"]
    for p in ([victim] if flush_in_gap else [victim, other]):
        assert st.get(p.arxiv_id) is None
        assert p.key not in {h.key for h in st.search(p.title, k=None)}
    st.commit()
    st.flush()
    want = 59 if flush_in_gap else 58
    assert st.stats()["live_docs"] == want
    st.close()
    with IndexStore(tmp_path / "ix") as st2:
        assert st2.get(victim.arxiv_id) is None and st2.stats()["live_docs"] == want
    assert fsck(tmp_path / "ix").ok
