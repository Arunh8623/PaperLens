"""Phase 8 through the real web app: published filter, citation counts, [PDF]/[HTML] links, reader page, Quick read."""
import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient                                    # noqa: E402

from paperlens.enrich.openalex_meta import normalise                          # noqa: E402
from paperlens.enrich.quickread import QuickRead                              # noqa: E402
from paperlens.store.index import IndexStore                                  # noqa: E402
from paperlens.synth import generate_papers                                   # noqa: E402
from paperlens.vectors.embedders import HashEmbedder                          # noqa: E402
from paperlens.web.app import create_app                                      # noqa: E402
from tests.test_enrich import WORKS                                           # noqa: E402
from tests.test_quickread import GOOD, FakeGemini                             # noqa: E402


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("p8")
    d = root / "ix"
    st = IndexStore(d, flush_docs=10**9, embedder_name="hash:32", ivf_min_docs=0)
    emb = HashEmbedder(32)
    papers = generate_papers(60, 7)
    # the three papers we label are chosen among those the query "evaluate" matches, so the tests can assert hard
    hit_ix = [i for i, p in enumerate(papers) if "evaluate" in (p.title + " " + p.abstract).lower().split()]
    assert len(hit_ix) >= 6
    A, B, C = hit_ix[0], hit_ix[1], hit_ix[2]        # A published (journal), B preprint with citations, C author-typed DOI
    papers[C] = papers[C].__class__.from_dict({**papers[C].to_dict(), "doi": "10.1145/author.supplied"})
    for p in papers:
        st.put(p, vector=emb.embed_paper(p))
    st.commit(); st.flush(); st.close()
    # OpenAlex side file: papers[0] published (journal), papers[2] preprint with a citation count
    meta = root / "ix.openalex_meta.jsonl"
    rows = [normalise(papers[A].arxiv_id, WORKS[0], "location", today="2026-11-17"),
            normalise(papers[B].arxiv_id, WORKS[1], "doi", today="2026-11-17")]
    meta.write_text("".join(json.dumps(r) + "\n" for r in rows))
    gem = FakeGemini([("ok", GOOD)])
    qr = QuickRead(root / "ix.quickread", api_key="SECRET-KEY", base_url=gem.url, pdf_fetch=lambda i: b"%PDF-fake")
    with TestClient(create_app(d, workers=1, meta_path=meta, quickread=qr)) as c:
        c.papers, c.gem, c.qr, c.root, c.A, c.B, c.C = papers, gem, qr, root, A, B, C
        yield c
    gem.close()


def _by_id(hits):
    return {h["arxiv_id"]: h for h in hits}


def test_features_endpoint_says_what_is_on(env):
    f = env.get("/api/features").json()
    assert f["publication_data_papers"] == 2 and f["quickread"] and f["quickread_model"] == "gemini-3-flash-preview"


def test_every_result_carries_links_citations_and_publication_info(env):
    hits = env.get("/api/search", params={"q": "evaluate", "k": 60, "mode": "bm25"}).json()["hits"]
    byid = _by_id(hits)
    for h in hits:
        assert h["links"]["pdf"] == f"https://arxiv.org/pdf/{h['arxiv_id']}" and h["links"]["html"] and h["pub"]
    pa, pb, pc = (env.papers[i].arxiv_id for i in (env.A, env.B, env.C))
    assert {pa, pb, pc} <= set(byid), "the three labelled papers must be among the hits"
    pub = byid[pa]["pub"]
    assert pub["published"] and pub["cited_by"] == 42 and pub["venue"] == "Journal of Testing"
    assert pub["doi"] == "10.1000/jt.2021.7" and pub["pub_info"].startswith("Journal of Testing, 7(2)")
    assert byid[pc]["pub"]["published"] and byid[pc]["pub"]["basis"] == "arxiv-doi" and byid[pc]["pub"]["cited_by"] is None
    assert not byid[pb]["pub"]["published"] and byid[pb]["pub"]["cited_by"] == 3


def test_published_only_keeps_exactly_the_published_papers_in_rank_order(env):
    q = {"q": "evaluate", "mode": "bm25"}
    allhits = env.get("/api/search", params={**q, "k": 100}).json()["hits"]
    want = [h["arxiv_id"] for h in allhits if h["pub"]["published"]]
    assert set(want) == {env.papers[env.A].arxiv_id, env.papers[env.C].arxiv_id}
    r = env.get("/api/search", params={**q, "k": 10, "published": True}).json()
    assert [h["arxiv_id"] for h in r["hits"]] == want[:10]            # same order as the ranker, nothing else
    assert all(h["pub"]["published"] for h in r["hits"]) and [h["rank"] for h in r["hits"]] == list(range(1, len(r["hits"]) + 1))
    assert r["published_filter"]["considered"] >= len(r["hits"]) and r["published_filter"]["have_openalex_data"]


def test_paper_endpoint_and_reader_page(env):
    pid = env.papers[env.A].arxiv_id
    j = env.get(f"/api/paper/{pid}").json()["paper"]
    assert j["pub"]["cited_by"] == 42 and j["links"]["pdf"].endswith(pid)
    page = env.get(f"/paper/{pid}")
    assert page.status_code == 200 and "Quick read" in page.text and "[PDF]" in page.text and "[HTML]" in page.text
    assert env.get("/paper/cs/0601001").status_code == 200           # old-style ids contain a slash
    assert env.get("/paper/" + "x" * 60).status_code == 400          # not an arXiv id
    assert env.get("/api/paper/9999.99999").status_code == 404


def test_search_page_has_the_new_controls(env):
    t = env.get("/").text
    assert 'id="published"' in t and "[PDF]" in t and "data-quick" in t and "Cited by" in t and "/paper/" in t


def test_quickread_needs_post_and_the_custom_header_then_caches(env):
    pid = env.papers[3].arxiv_id
    assert env.get(f"/api/quickread/{pid}").json()["result"] is None                # nothing saved yet, no API call
    assert env.post(f"/api/quickread/{pid}").status_code == 403                      # no header: refused
    assert env.post(f"/api/quickread/{pid}", headers={"Origin": "https://evil.example"}).status_code == 403
    assert not env.gem.calls
    r = env.post(f"/api/quickread/{pid}", headers={"X-PaperLens": "1"})
    assert r.status_code == 200 and r.json()["summary"]["tldr"] == "T" and r.json()["source"] == "pdf"
    n = len(env.gem.calls)
    r2 = env.post(f"/api/quickread/{pid}", headers={"X-PaperLens": "1"}).json()
    assert r2["cached"] and len(env.gem.calls) == n
    assert env.get(f"/api/quickread/{pid}").json()["result"]["summary"]["method"] == "M"
    sent = json.dumps(env.gem.calls[0]["body"])
    assert env.papers[3].title in sent                                                # the paper's own title went to Gemini
    assert env.post("/api/quickread/9999.99999", headers={"X-PaperLens": "1"}).status_code == 404


def test_quickread_without_a_key_is_a_clear_503(env, tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False); monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    off = QuickRead(tmp_path / "q", api_key=None)
    d = env.root / "ix"
    with TestClient(create_app(d, workers=1, meta_path=tmp_path / "none.jsonl", quickread=off)) as c:
        assert c.get("/api/features").json() == {"publication_data_papers": 0, "quickread": False,
                                                 "quickread_model": "gemini-3-flash-preview"}
        r = c.post(f"/api/quickread/{env.papers[4].arxiv_id}", headers={"X-PaperLens": "1"})
        assert r.status_code == 503 and "GEMINI_API_KEY" in r.json()["detail"]
        # with no OpenAlex file, the filter still works for author-supplied DOIs and says why it is thin
        r = c.get("/api/search", params={"q": "evaluate", "k": 60, "published": True}).json()
        assert not r["published_filter"]["have_openalex_data"]
        assert all(h["pub"]["basis"] == "arxiv-doi" for h in r["hits"])
