import pytest

from paperlens.contracts.paper import (
    PaperRecord, PaperError, make_key, split_key, decide, write_jsonl, read_jsonl,
)


def mk(**kw):
    base = dict(arxiv_id="2401.00001", datestamp="2024-01-05", title="T", abstract="A")
    base.update(kw)
    return PaperRecord(**base)


def test_key_format():
    assert mk().key == "2401.00001@2024-01-05"
    assert split_key("hep-th/9901001@2020-02-03") == ("hep-th/9901001", "2020-02-03")


def test_roundtrip_json_and_unicode():
    p = mk(title="Café — k-means ≥ 2", authors=["Ana Ünal", "B. Rao"],
           categories=["cs.IR", "cs.LG"], doi="10.1/x", synthetic=True)
    assert PaperRecord.from_json(p.to_json()) == p


def test_canonical_json_is_stable():
    a = mk(authors=["x"]).to_json()
    b = PaperRecord.from_json(a).to_json()
    assert a == b and "\n" not in a


@pytest.mark.parametrize("bad", [
    dict(arxiv_id=""), dict(arxiv_id="a@b"), dict(datestamp="2024-1-5"),
    dict(datestamp="20240105"), dict(title=3), dict(authors="Bob"),
    dict(authors=[1]), dict(created="yesterday"), dict(synthetic="yes"),
])
def test_bad_records_rejected(bad):
    with pytest.raises(PaperError):
        mk(**bad)


def test_from_dict_rejects_unknown_and_missing():
    with pytest.raises(PaperError):
        PaperRecord.from_dict({"arxiv_id": "1", "datestamp": "2024-01-01"})
    d = mk().to_dict(); d["extra"] = 1
    with pytest.raises(PaperError):
        PaperRecord.from_dict(d)
    with pytest.raises(PaperError):
        PaperRecord.from_json("{not json")
    with pytest.raises(PaperError):
        PaperRecord.from_json("[1,2]")


def test_split_key_bad():
    for k in ("noatsign", "@2020-01-01", "x@"):
        with pytest.raises(PaperError):
            split_key(k)


def test_decide_rules():
    assert decide(None, "2024-01-01") == "insert"
    assert decide("2024-01-01", "2024-01-01") == "duplicate"
    assert decide("2024-01-01", "2024-02-01") == "replace"
    assert decide("2024-02-01", "2024-01-01") == "stale"


def test_jsonl_roundtrip_and_no_crlf(tmp_path):
    papers = [mk(arxiv_id=f"2401.{i:05d}", title=f"t{i}") for i in range(5)]
    f = tmp_path / "p.jsonl"
    assert write_jsonl(f, papers) == 5
    raw = f.read_bytes()
    assert b"\r\n" not in raw and raw.count(b"\n") == 5
    assert list(read_jsonl(f)) == papers


def test_jsonl_bad_line(tmp_path):
    f = tmp_path / "p.jsonl"
    f.write_text(mk().to_json() + "\nGARBAGE\n", encoding="utf-8")
    with pytest.raises(PaperError, match=":2:"):
        list(read_jsonl(f))
    assert len(list(read_jsonl(f, skip_bad=True))) == 1
