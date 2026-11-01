import os
import random

import pytest

from paperlens.contracts.paper import PaperRecord
from paperlens.store.index import IndexStore, IndexCorrupt
from paperlens.store.lock import IndexLocked
from paperlens.store.manifest import MANIFEST_NAME
from paperlens.synth import generate_papers, messy_stream, TOPICS
from tests.oracle import Model

QUERIES = ["graph", "recovery crash", "k-means embedding", "scheduler queue latency",
           "o(nlogn) pagerank", "novel approach", "inverted postings bm25", "nosuchword",
           "citation community spectral sketch", "the"]


def assert_same(store, model, queries=QUERIES, tol=1e-9):
    assert store.live_keys() == model.live_keys()
    for q in queries:
        got = {h.key: h.score for h in store.search(q, k=None)}
        want = model.bm25(q)
        assert set(got) == set(want), f"different documents for {q!r}"
        for key, s in want.items():
            assert abs(got[key] - s) <= tol, f"{q!r} {key}: {got[key]} vs {s}"
        ranked = [h.score for h in store.search(q, k=None)]
        assert ranked == sorted(ranked, reverse=True)


def mk(i, ds="2024-01-01", title=None, abstract=None):
    return PaperRecord(arxiv_id=f"t.{i}", datestamp=ds, title=title or f"title {i}",
                       abstract=abstract or f"abstract about graph theory {i}")


# ---------------------------------------------------------------- basics
def test_put_commit_search_and_reopen(tmp_path):
    papers = generate_papers(40, seed=1)
    with IndexStore(tmp_path / "ix") as st:
        for p in papers:
            assert st.put(p).outcome == "insert"
        st.commit()
        assert st.stats()["buffer_docs"] == 40 and st.stats()["segments"] == 0
        hits = st.search("recovery", k=5)
        assert 0 < len(hits) <= 5 and all(isinstance(h.score, float) for h in hits)
    with IndexStore(tmp_path / "ix") as st:                    # buffer rebuilt from the diary
        assert st.stats()["recovery"]["replayed"] == 40 and len(st.live_keys()) == 40
        assert st.get(papers[3].arxiv_id) == papers[3]
        assert st.get("missing") is None


def test_flush_creates_segment_manifest_and_new_diary(tmp_path):
    with IndexStore(tmp_path / "ix") as st:
        for p in generate_papers(30, seed=2):
            st.put(p)
        assert st.flush() == "seg-000001"
        s = st.stats()
        assert s["segments"] == 1 and s["buffer_docs"] == 0 and s["generation"] == 1
        assert s["last_flushed_seq"] == 30
        wals = os.listdir(tmp_path / "ix" / "wal")
        assert wals == ["wal-00000000000000000031.log"]        # old diary gone, new one named by next seq
        assert st.flush() is None                              # nothing to flush
    with IndexStore(tmp_path / "ix") as st:
        assert st.stats()["recovery"]["replayed"] == 0 and len(st.live_keys()) == 30


def test_auto_flush_threshold(tmp_path):
    with IndexStore(tmp_path / "ix", flush_docs=10) as st:
        for i, p in enumerate(generate_papers(35, seed=3)):
            st.put(p)
            if i % 5 == 4:
                st.commit()
        assert st.stats()["segments"] == 3                      # flushed at 10, 20, 30


def test_single_writer_lock(tmp_path):
    with IndexStore(tmp_path / "ix"):
        with pytest.raises(IndexLocked):
            IndexStore(tmp_path / "ix")
    IndexStore(tmp_path / "ix").close()                         # free again after close


def test_failed_open_releases_lock(tmp_path):
    ix = tmp_path / "ix"
    with IndexStore(ix) as st:
        st.put(mk(1)); st.flush()
    mp = ix / MANIFEST_NAME
    mp.write_bytes(mp.read_bytes().replace(b'"generation": 1', b'"generation": 7'))
    with pytest.raises(IndexCorrupt):
        IndexStore(ix)
    mp.write_bytes(mp.read_bytes().replace(b'"generation": 7', b'"generation": 1'))
    IndexStore(ix).close()                                      # lock was not leaked


# ---------------------------------------------------- put / replace / delete rules
def test_idempotency_rules(tmp_path):
    with IndexStore(tmp_path / "ix") as st:
        assert st.put(mk(1, "2024-01-01")) == ("insert", 1)
        assert st.put(mk(1, "2024-01-01")) == ("duplicate", 0)
        assert st.put(mk(1, "2023-06-01")) == ("stale", 0)
        r = st.put(mk(1, "2024-05-01", title="newer title"))
        assert r.outcome == "replace" and r.seq == 2
        assert st.get("t.1").title == "newer title" and len(st.live_keys()) == 1
        assert st.stats()["last_seq"] == 2                      # dup/stale were never logged


def test_replace_across_a_flush_hides_old_version_but_keeps_stats_until_compaction(tmp_path):
    with IndexStore(tmp_path / "ix") as st:
        st.put(mk(1, "2024-01-01", title="zebra", abstract="zebra zebra"))
        st.flush()
        st.put(mk(1, "2024-02-01", title="giraffe", abstract="giraffe"))
        assert st.search("zebra") == []                         # old version never returned
        assert [h.key for h in st.search("giraffe")] == ["t.1@2024-02-01"]
        assert st.stats()["physical_docs"] == 2 and st.stats()["live_docs"] == 1
        assert st.stats()["pending_tombstones"] == 1
        st.flush()
        assert st.search("zebra") == []
    with IndexStore(tmp_path / "ix") as st:                     # tombstone survives reopen
        assert st.search("zebra") == [] and len(st.search("giraffe")) == 1


def test_delete_in_buffer_and_in_segment_and_reput(tmp_path):
    with IndexStore(tmp_path / "ix") as st:
        st.put(mk(1)); st.put(mk(2))
        assert st.delete("t.1") and not st.delete("t.1") and not st.delete("nope")
        assert st.stats()["buffer_docs"] == 1                   # buffer doc vanished completely
        st.flush()
        assert st.delete("t.2")
        assert st.search("graph") == []
        st.flush()                                              # segment holding only a tombstone
        assert st.stats()["segments"] == 2 and st.search("graph") == []
        assert st.put(mk(2)).outcome == "insert"                # same key may come back after a delete
        assert [h.key for h in st.search("graph")] == ["t.2@2024-01-01"]
        st.flush()
    with IndexStore(tmp_path / "ix") as st:
        assert [h.key for h in st.search("graph")] == ["t.2@2024-01-01"]


def test_document_and_tombstone_for_the_same_key_in_one_segment(tmp_path):
    """K lives in seg-1; it is deleted and put back BEFORE the next flush, so seg-2
    holds both the new copy of K and a tombstone for K. The tombstone must kill
    only the OLD copy (older segments), never the new one beside it."""
    ix = tmp_path / "ix"
    model = Model()
    with IndexStore(ix) as st:
        p = mk(1, abstract="unicorn")
        st.put(p); model.put(p); st.flush(); model.flush()
        assert st.delete("t.1") and model.delete("t.1")
        assert st.put(p).outcome == "insert" and model.put(p) == "insert"
        assert st.stats()["pending_tombstones"] == 1
        st.flush(); model.flush()
        assert [h.key for h in st.search("unicorn")] == ["t.1@2024-01-01"]
        assert_same(st, model, ["unicorn", "graph"])
    with IndexStore(ix) as st:                                  # the tombstone logic is re-derived on open
        assert [h.key for h in st.search("unicorn")] == ["t.1@2024-01-01"]
        assert st.get("t.1") == mk(1, abstract="unicorn")
        assert_same(st, model, ["unicorn", "graph"])


def test_delete_replayed_from_diary(tmp_path):
    ix = tmp_path / "ix"
    with IndexStore(ix) as st:
        st.put(mk(1)); st.flush()
        st.delete("t.1"); st.put(mk(2)); st.commit()
    with IndexStore(ix) as st:                                  # nothing flushed since: all from diary
        assert st.stats()["recovery"]["replayed"] == 2
        assert [h.key for h in st.search("graph")] == ["t.2@2024-01-01"]
        assert st.stats()["pending_tombstones"] == 1


# ------------------------------------------------------ BM25 == brute force (1e-9)
def test_bm25_equals_brute_force_across_buffer_and_segments(tmp_path):
    base = generate_papers(150, seed=7)
    model = Model()
    with IndexStore(tmp_path / "ix") as st:
        for i, p in enumerate(base):
            assert st.put(p).outcome == model.put(p)
            if i in (40, 90):
                st.flush(); model.flush()                       # two segments, rest stays in buffer
        assert st.stats()["segments"] == 2 and st.stats()["buffer_docs"] == 150 - 91
        assert_same(st, model)
        assert any(model.bm25(q) for q in QUERIES[:7])         # the queries really match things


def test_bm25_random_operation_sequences_with_reopen(tmp_path):
    """Random puts (incl. duplicates/updates/stale), deletes, flushes and reopens;
    after every step the store must equal the brute-force model."""
    for seed in range(4):
        rng = random.Random(seed)
        stream = messy_stream(generate_papers(80, seed=seed), seed=seed)
        ix = tmp_path / f"ix{seed}"
        model = Model()
        st = IndexStore(ix, flush_docs=10**9)
        ids = [p.arxiv_id for p in stream]
        for i, p in enumerate(stream):
            assert st.put(p).outcome == model.put(p)
            r = rng.random()
            if r < 0.06:
                victim = rng.choice(ids)
                assert st.delete(victim) == model.delete(victim)
            elif r < 0.12:
                st.flush(); model.flush()
            elif r < 0.16:
                st.commit(); st.close()                         # clean restart: buffer replays from diary
                st = IndexStore(ix, flush_docs=10**9)
            if i % 25 == 0:
                assert_same(st, model, QUERIES[:6])
        assert_same(st, model)
        st.close()
        st = IndexStore(ix)
        assert_same(st, model)
        st.close()


def test_search_edge_cases(tmp_path):
    with IndexStore(tmp_path / "ix") as st:
        assert st.search("anything") == []                      # empty index
        st.put(mk(1)); st.put(mk(2, abstract="graph graph graph"))
        assert st.search("") == [] and st.search("the of and") == []
        assert st.search("graph graph graph") == st.search("graph")   # repeated words count once
        assert len(st.search("graph", k=1)) == 1
        hits = st.search("graph", k=None)
        assert len(hits) == 2 and hits[0].key == "t.2@2024-01-01"     # higher tf ranks first


def test_ties_break_by_key(tmp_path):
    with IndexStore(tmp_path / "ix") as st:
        for i in (3, 1, 2):
            st.put(PaperRecord(arxiv_id=f"x.{i}", datestamp="2024-01-01", title="same", abstract="same"))
        assert [h.arxiv_id for h in st.search("same")] == ["x.1", "x.2", "x.3"]


def test_topic_words_retrieve_their_topic(tmp_path):
    """Plumbing sanity only (synthetic data says nothing about real quality)."""
    papers = generate_papers(300, seed=9)
    with IndexStore(tmp_path / "ix") as st:
        for p in papers:
            st.put(p)
        st.flush()
        word = TOPICS["storage"][1]                             # "fsync"
        top = st.search(word, k=10)
        assert top and all(word in st.get(h.arxiv_id).abstract.lower() for h in top)
