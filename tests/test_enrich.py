"""Phase 8 tests: [PDF]/[HTML] links, publication info + citation counts from OpenAlex (against a LOCAL fake).

The fake follows the field names in OpenAlex's documentation (read 2026-11-17). Nothing here talks to the real
service, so a green test proves our code is self-consistent, not that it matches OpenAlex byte for byte.
"""
import json
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from paperlens.enrich.links import html_url, links_for, pdf_url
from paperlens.enrich.openalex_meta import (MetaStore, arxiv_ids_of, fetch_meta, merge_view, normalise)
from paperlens.graph.openalex import OpenAlexClient
from paperlens.harvest.limiter import MinIntervalLimiter


# ---------------------------------------------------------------- fake OpenAlex with MERGED works
def _journal_loc(landing="https://publisher.example/x"):
    return {"is_published": True, "version": "publishedVersion", "landing_page_url": landing,
            "source": {"display_name": "Journal of Testing", "type": "journal"}}


def _arxiv_loc(aid):
    return {"is_published": False, "version": "submittedVersion", "landing_page_url": f"https://arxiv.org/abs/{aid}",
            "pdf_url": f"https://arxiv.org/pdf/{aid}", "id": f"pmh:oai:arXiv.org:{aid}",
            "source": {"display_name": "arXiv (Cornell University)", "type": "repository"}}


WORKS = [
    # A: published. OpenAlex filed it under the PUBLISHER doi, so filter=doi:<arXiv doi> will NOT find it.
    {"id": "https://openalex.org/W1", "doi": "https://doi.org/10.1000/jt.2021.7", "type": "article", "cited_by_count": 42,
     "publication_year": 2021, "is_retracted": False, "biblio": {"volume": "7", "issue": "2", "first_page": "100", "last_page": "120"},
     "primary_location": _journal_loc(), "best_oa_location": _arxiv_loc("2101.00001"),
     "locations": [_journal_loc(), _arxiv_loc("2101.00001")]},
    # B: preprint only; its doi IS the arXiv doi
    {"id": "https://openalex.org/W2", "doi": "https://doi.org/10.48550/arxiv.2101.00002", "type": "preprint",
     "cited_by_count": 3, "publication_year": 2021, "is_retracted": False, "biblio": {},
     "primary_location": _arxiv_loc("2101.00002"), "best_oa_location": _arxiv_loc("2101.00002"),
     "locations": [_arxiv_loc("2101.00002")]},
    # D: old-style id, published, merged
    {"id": "https://openalex.org/W4", "doi": "https://doi.org/10.1000/old.1", "type": "article", "cited_by_count": 900,
     "publication_year": 2006, "is_retracted": True, "biblio": {"first_page": "55", "last_page": "55"},
     "primary_location": _journal_loc(), "locations": [_journal_loc(), _arxiv_loc("cs/0601001")]},
]


class FakeOA:
    def __init__(self, reject_location_filter=False):
        self.reject, self.queries = reject_location_filter, []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass

            def do_GET(self):
                q = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(self.path).query))
                outer.queries.append(q)
                flt = q.get("filter", "")
                name, _, vals = flt.partition(":")
                vals = vals.split("|")
                if name == "locations.landing_page_url" and outer.reject:
                    self.send_response(400); self.end_headers(); self.wfile.write(b'{"error":"unknown filter"}'); return
                res = []
                for w in WORKS:
                    if name == "doi" and w["doi"].replace("https://doi.org/", "").lower() in [v.lower() for v in vals]:
                        res.append(w)                              # matches work.doi ONLY, like the real filter may
                    elif name == "locations.landing_page_url" and any(
                            (l or {}).get("landing_page_url") in vals for l in w["locations"]):
                        res.append(w)
                body = json.dumps({"results": res}).encode()
                self.send_response(200); self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()

    def close(self):
        self.server.shutdown(); self.server.server_close()


def _client(url):
    return OpenAlexClient(base_url=url, limiter=MinIntervalLimiter(0.0), batch=50, max_retries=0)


# ---------------------------------------------------------------- links
def test_links_pdf_always_html_guessed_by_date():
    assert pdf_url("2401.00001") == "https://arxiv.org/pdf/2401.00001"
    assert pdf_url("cs/0601001") == "https://arxiv.org/pdf/cs/0601001"
    assert html_url("2401.00001") == "https://arxiv.org/html/2401.00001"          # live HTML channel (Dec 2023 on)
    assert html_url("2312.00001") == "https://arxiv.org/html/2312.00001"
    assert html_url("2311.99999") == "https://ar5iv.labs.arxiv.org/html/2311.99999"  # historical corpus
    assert html_url("cs/0601001") == "https://ar5iv.labs.arxiv.org/html/cs/0601001"
    assert links_for("2401.00001", "https://p.example")["publisher"] == "https://p.example"
    assert "publisher" not in links_for("2401.00001")


# ---------------------------------------------------------------- normalise / matching
def test_arxiv_ids_found_from_every_place_a_work_mentions_them():
    assert arxiv_ids_of(WORKS[0]) == {"2101.00001"}          # work.doi is the publisher's; arXiv only in a location
    assert arxiv_ids_of(WORKS[1]) == {"2101.00002"}
    assert arxiv_ids_of(WORKS[2]) == {"cs/0601001"}


def test_published_rule():
    a = normalise("2101.00001", WORKS[0], "location", today="2026-11-17")
    assert a["published"] and a["basis"] == "journal-location" and a["doi"] == "10.1000/jt.2021.7"
    assert a["venue"] == "Journal of Testing" and a["pages"] == "100-120" and a["cited_by_count"] == 42
    b = normalise("2101.00002", WORKS[1], "doi")
    assert not b["published"] and b["doi"] == "" and b["cited_by_count"] == 3
    d = normalise("cs/0601001", WORKS[2], "location")
    assert d["pages"] == "55" and d["retracted"]
    # no journal location, but a publisher DOI on an "article": counted, with its own basis
    w = {**WORKS[0], "locations": [_arxiv_loc("2101.00001")], "primary_location": _arxiv_loc("2101.00001")}
    assert normalise("2101.00001", w, "doi")["basis"] == "publisher-doi"
    # a preprint on another server (non-arXiv DOI, type preprint) is NOT "published"
    assert not normalise("x", {**WORKS[1], "doi": "https://doi.org/10.36227/techrxiv.1"}, "doi")["published"]


def test_merge_view_uses_the_authors_doi_when_openalex_has_nothing():
    v = merge_view("10.1145/3456.789", None)
    assert v["published"] and v["basis"] == "arxiv-doi" and v["doi_url"] == "https://doi.org/10.1145/3456.789"
    assert not merge_view("10.48550/arXiv.2101.00001", None)["published"]       # arXiv's own DOI is not a journal
    assert not merge_view("", None)["published"] and merge_view("", None)["cited_by"] is None
    v = merge_view("", normalise("2101.00001", WORKS[0], "doi", today="2026-11-17"))
    assert v["pub_info"] == "Journal of Testing, 7(2), pp. 100-120, 2021" and v["cited_by"] == 42


# ---------------------------------------------------------------- fetch_meta against the fake
def test_fetch_meta_finds_published_papers_filed_under_the_publisher_doi(tmp_path):
    fake = FakeOA()
    try:
        out = tmp_path / "meta.jsonl"
        ids = ["2101.00001", "2101.00002", "2101.00003", "cs/0601001"]
        stats = fetch_meta(_client(fake.url), ids, out)
        assert stats["by_doi"] == 1 and stats["by_location"] == 2 and stats["unknown"] == 1 and stats["fallback"] == "ok"
        recs = {json.loads(l)["arxiv_id"]: json.loads(l) for l in out.read_text().splitlines()}
        assert recs["2101.00001"]["published"] and recs["2101.00001"]["matched_by"] == "location"
        assert recs["2101.00002"]["matched_by"] == "doi" and not recs["2101.00002"]["published"]
        assert recs["2101.00003"]["openalex_id"] is None
        assert recs["cs/0601001"]["cited_by_count"] == 900
        n = len(fake.queries)
        again = fetch_meta(_client(fake.url), ids, out)                     # resumable: nothing left to ask
        assert again["asked"] == 0 and len(fake.queries) == n
    finally:
        fake.close()


def test_fetch_meta_survives_a_rejected_location_filter(tmp_path):
    fake = FakeOA(reject_location_filter=True)
    msgs = []
    try:
        stats = fetch_meta(_client(fake.url), ["2101.00001", "2101.00002"], tmp_path / "m.jsonl", log=msgs.append)
        assert stats["fallback"] == "unavailable" and stats["by_doi"] == 1 and stats["unknown"] == 1
        assert any("rejected the location filter" in m for m in msgs)
    finally:
        fake.close()


def test_metastore_reloads_when_the_file_grows_and_ignores_a_torn_line(tmp_path):
    p = tmp_path / "m.jsonl"
    ms = MetaStore(p)
    assert ms.get("2101.00001") is None and len(ms) == 0                     # file missing = feature off, no crash
    p.write_text(json.dumps({"arxiv_id": "2101.00001", "cited_by_count": 5}) + "\n" + '{"arxiv_id": "21')
    assert ms.get("2101.00001")["cited_by_count"] == 5 and len(ms) == 1
    with open(p, "a") as f:
        f.write('01.00002", "cited_by_count": 6}\n')
    assert ms.get("2101.00002")["cited_by_count"] == 6                       # the torn line was completed: now it counts
    with open(p, "a") as f:
        f.write(json.dumps({"arxiv_id": "2101.00009", "cited_by_count": 9}) + "\n")
    assert ms.get("2101.00009")["cited_by_count"] == 9
