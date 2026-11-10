import json
import os
import shutil

import pytest

from paperlens.contracts.paper import PaperRecord
from paperlens.store import segment as sg
from paperlens.synth import generate_papers
from paperlens.text.tokenizer import paper_text, term_freqs


def make_docs(n, seed=1):
    out = []
    for p in generate_papers(n, seed):
        tf, dl = term_freqs(paper_text(p.title, p.abstract))
        out.append((p.key, p, tf, dl))
    return out


def build(tmp_path, docs, tombs=(), name="seg-000001"):
    root = str(tmp_path / "segments")
    os.makedirs(root, exist_ok=True)
    tmp = sg.write_segment_tmp(root, name, docs, tombs)
    return sg.publish_segment(root, tmp, name)


def test_roundtrip_everything(tmp_path):
    docs = make_docs(60)
    path = build(tmp_path, docs, tombs=["old@2020-01-01", "gone@2019-05-05"])
    s = sg.Segment.open(path)
    try:
        assert s.num_docs == 60 and s.keys == [d[0] for d in docs]
        assert s.tombstones == ["gone@2019-05-05", "old@2020-01-01"]
        assert list(s.doclens) == [d[3] for d in docs]
        assert s.total_doclen == sum(d[3] for d in docs)
        assert all(s.paper(i) == docs[i][1] for i in range(60))
        # every posting equals what we put in, for every term
        expect = {}
        for local, (_k, _p, tf, _dl) in enumerate(docs):
            for t, c in tf.items():
                expect.setdefault(t, []).append((local, c))
        assert set(s.terms) == set(expect)
        for t, lst in expect.items():
            assert s.postings(t) == lst and s.df(t) == len(lst)
        assert s.postings("no-such-term") == [] and s.df("no-such-term") == 0
        # per-doc invariant: sum of tf == doc length
        sums = [0] * 60
        for t in expect:
            for d, c in s.postings(t):
                sums[d] += c
        assert sums == list(s.doclens)
    finally:
        s.close()


def test_empty_segment_with_only_tombstones(tmp_path):
    s = sg.Segment.open(build(tmp_path, [], tombs=["k@2020-01-01"]))
    try:
        assert s.num_docs == 0 and s.terms == {} and s.tombstones == ["k@2020-01-01"]
    finally:
        s.close()


def test_tmp_dir_is_not_final_and_publish_moves_it(tmp_path):
    root = str(tmp_path / "segments"); os.makedirs(root)
    tmp = sg.write_segment_tmp(root, "seg-000001", make_docs(3), [])
    assert os.path.basename(tmp) == ".tmp-seg-000001" and os.path.exists(tmp)
    assert not os.path.exists(os.path.join(root, "seg-000001"))
    final = sg.publish_segment(root, tmp, "seg-000001")
    assert os.path.exists(final) and not os.path.exists(tmp)


def test_meta_is_written_last_and_covers_all_files(tmp_path):
    path = build(tmp_path, make_docs(5))
    meta = json.load(open(os.path.join(path, "meta.json")))
    assert set(meta["files"]) == set(sg.DATA_FILES) | {"attrs.json"}      # Phase 3: attrs.json is always written
    for f, info in meta["files"].items():
        assert os.path.getsize(os.path.join(path, f)) == info["size"]
    mtimes = {f: os.stat(os.path.join(path, f)).st_mtime_ns for f in os.listdir(path)}
    assert mtimes["meta.json"] >= max(v for k, v in mtimes.items() if k != "meta.json")


@pytest.mark.parametrize("victim", sg.DATA_FILES + ("meta.json",))
def test_missing_or_truncated_file_is_refused(tmp_path, victim):
    path = build(tmp_path, make_docs(8), tombs=["x@2020-01-01"])
    f = os.path.join(path, victim)
    data = open(f, "rb").read()
    os.remove(f)
    with pytest.raises(sg.SegmentError):
        sg.Segment.open(path)
    if victim != "meta.json" and len(data) > 1:
        open(f, "wb").write(data[:-1])                     # one byte short
        with pytest.raises(sg.SegmentError):
            sg.Segment.open(path)


def test_wrong_tokenizer_version_refused(tmp_path):
    path = build(tmp_path, make_docs(3))
    mp = os.path.join(path, "meta.json")
    m = json.load(open(mp)); m["tokenizer_version"] = 99
    json.dump(m, open(mp, "w"))
    with pytest.raises(sg.SegmentError, match="tokenizer"):
        sg.Segment.open(path)


def test_failed_open_does_not_leak_open_files(tmp_path):
    """Windows needs this: a half-opened segment must not keep files locked."""
    path = build(tmp_path, make_docs(4))
    os.remove(os.path.join(path, "deletes.json"))
    for _ in range(50):
        with pytest.raises(sg.SegmentError):
            sg.Segment.open(path)
    shutil.rmtree(path)                                    # would fail on Windows if leaked


def test_names():
    assert sg.segment_name(7) == "seg-000007" and sg.segment_number("seg-000007") == 7
    assert sg.segment_number("seg-x") is None and sg.segment_number("wal-1") is None
