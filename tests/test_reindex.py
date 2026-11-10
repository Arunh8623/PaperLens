import pytest

import paperlens.store.segment as seg_mod
from paperlens.store.index import IndexCorrupt, IndexStore
from paperlens.store.reindex import ReindexError, reindex
from paperlens.store.fsck import fsck
from paperlens.synth import generate_papers, messy_stream


def test_old_tokenizer_index_is_refused_then_reindex_rescues_it(tmp_path, monkeypatch):
    src, dst = tmp_path / "old", tmp_path / "new"
    stream = messy_stream(generate_papers(120, seed=3), seed=3)
    with IndexStore(src, flush_docs=40) as st:
        for p in stream:
            st.put(p)
        st.delete(stream[0].arxiv_id)
        st.commit()
        st.put(generate_papers(1, seed=99)[0]); st.commit()           # leaves a diary tail too
        before = {a: st.get(a) for a in st.live}
        hits_before = [h.key for h in st.search("checksum", k=None)]
    monkeypatch.setattr(seg_mod, "TOKENIZER_VERSION", 2)              # "the tokenizer changed"
    with pytest.raises(IndexCorrupt, match="tokenizer"):
        IndexStore(src)
    import os
    seg_before = {n: sorted(os.listdir(src / 'segments' / n)) for n in os.listdir(src / 'segments')}
    sizes_before = {(n, f): (src / 'segments' / n / f).stat().st_size for n, fs in seg_before.items() for f in fs}
    out = reindex(src, dst, log=lambda *_: None)
    assert out["papers"] == len(before)
    with IndexStore(dst) as st:                                       # opens fine under the new version
        assert {a: st.get(a) for a in st.live} == before
        assert [h.key for h in st.search("checksum", k=None)] == hits_before   # same words -> same answers
    assert fsck(dst).ok
    assert {n: sorted(os.listdir(src / 'segments' / n)) for n in os.listdir(src / 'segments')} == seg_before
    assert {(n, f): (src / 'segments' / n / f).stat().st_size for n, fs in seg_before.items() for f in fs} == sizes_before


def test_reindex_refuses_bad_arguments(tmp_path):
    with IndexStore(tmp_path / "a") as st:
        st.put(generate_papers(1, 1)[0])
    with pytest.raises(ReindexError, match="different"):
        reindex(tmp_path / "a", tmp_path / "a")
    with pytest.raises(ReindexError, match="does not look like"):
        reindex(tmp_path / "empty", tmp_path / "x")
    reindex(tmp_path / "a", tmp_path / "b", log=lambda *_: None)
    with pytest.raises(ReindexError, match="already holds"):
        reindex(tmp_path / "a", tmp_path / "b")


def test_put_precomputed_is_validated_and_equivalent(tmp_path):
    from paperlens.text.tokenizer import paper_text, term_freqs
    p = generate_papers(1, 5)[0]
    tf, dl = term_freqs(paper_text(p.title, p.abstract))
    with IndexStore(tmp_path / "a") as a, IndexStore(tmp_path / "b") as b:
        a.put(p); b.put(p, precomputed=(tf, dl))
        assert a.search("recovery", k=None) == b.search("recovery", k=None)
        for bad in [({}, 0), ({"x": 2}, 3), ({"x": 0}, 0), (dict(tf), dl + 1)]:
            q = generate_papers(2, 6)[1]
            with pytest.raises(ValueError):
                b.put(q, precomputed=bad)
        assert b.stats()["last_seq"] == 1                              # nothing bad reached the diary
