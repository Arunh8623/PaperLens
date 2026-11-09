import random

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient                                    # noqa: E402

from paperlens.graph.csr import Graph                                          # noqa: E402
from paperlens.store.index import IndexStore                                   # noqa: E402
from paperlens.synth import generate_papers                                    # noqa: E402
from paperlens.vectors.embedders import HashEmbedder                           # noqa: E402
from paperlens.web.app import create_app                                       # noqa: E402


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    d = tmp_path_factory.mktemp("web") / "ix"
    st = IndexStore(d, flush_docs=10**9, embedder_name="hash:32", ivf_min_docs=0)
    emb = HashEmbedder(32)
    papers = generate_papers(80, 5)
    dup = papers[2].__class__.from_dict({**papers[2].to_dict(), "arxiv_id": "2199.00001"})      # a planted near-duplicate
    for p in papers + [dup]:
        st.put(p, vector=emb.embed_paper(p))
    st.commit(); st.flush()
    ids = [p.arxiv_id for p in papers]
    rng = random.Random(1)
    st.publish_graph(Graph.from_edges(ids, [(a, rng.choice(ids)) for a in ids for _ in range(2)]))
    st.close()
    with TestClient(create_app(d, workers=2)) as c:
        c.papers = papers
        yield c


def test_page_and_health(client):
    r = client.get("/")
    assert r.status_code == 200 and "PaperLens" in r.text and "More like this" in r.text and "<script>" in r.text
    h = client.get("/api/health").json()
    assert h["ok"] and h["generation"] >= 1


def test_search_modes_return_reasons_and_the_snapshot_generation(client):
    for mode in ("bm25", "vector", "hybrid"):
        r = client.get("/api/search", params={"q": "inverted postings segment", "mode": mode, "k": 5})
        assert r.status_code == 200, r.text
        j = r.json()
        assert j["hits"] and j["generation"] >= 1 and j["took_ms"] >= 0
        assert all(h["why"] for h in j["hits"]), "every result must explain itself"
    hy = client.get("/api/search", params={"q": "inverted postings segment", "mode": "hybrid"}).json()["hits"]
    assert any("keyword search" in w or "meaning search" in w for h in hy for w in h["why"])
    bm = client.get("/api/search", params={"q": "inverted postings", "mode": "bm25"}).json()["hits"]
    assert any(w.startswith("Title contains") or w.startswith("Abstract contains") for w in bm[0]["why"])


def test_filters_graph_late_and_dedupe(client):
    p0 = client.papers[0]
    r = client.get("/api/search", params={"q": "inverted postings", "k": 30, "cat": p0.categories[0]}).json()
    assert r["hits"] and all(p0.categories[0] in h["categories"] for h in r["hits"])
    r = client.get("/api/search", params={"q": "inverted postings", "k": 20, "graph": True}).json()
    assert any(any("citation graph" in w for w in h["why"]) for h in r["hits"]) or len(r["hits"]) < 3
    r = client.get("/api/search", params={"q": "inverted postings", "k": 20, "late": True}).json()
    assert r["hits"]
    t = client.papers[2].title
    plain = client.get("/api/search", params={"q": t, "k": 10}).json()
    ded = client.get("/api/search", params={"q": t, "k": 10, "dedupe": True}).json()
    ids_plain = [h["arxiv_id"] for h in plain["hits"]]
    assert "2199.00001" in ids_plain and client.papers[2].arxiv_id in ids_plain
    assert len(ded["hits"]) == len(plain["hits"]) - 1 and ded["merged_duplicates"]


def test_similar_key_paper_endpoints_and_errors(client):
    a = client.papers[7].arxiv_id
    r = client.get(f"/api/similar/{a}", params={"k": 4}).json()
    assert len(r["hits"]) == 4 and all(h["arxiv_id"] != a for h in r["hits"])
    assert client.get("/api/similar/0000.00000").status_code == 404
    k = client.get("/api/key", params={"q": "inverted postings", "k": 3}).json()
    assert 1 <= len(k["key"]) <= 3 and "pagerank" in k["key"][0]
    p = client.get(f"/api/paper/{a}").json()["paper"]
    assert p["arxiv_id"] == a and p["title"]
    assert client.get("/api/paper/0000.00000").status_code == 404
    assert client.get("/api/search", params={"q": "x", "mode": "nonsense"}).status_code == 422
    assert client.get("/api/search", params={"q": ""}).status_code == 422
    assert client.get("/api/search", params={"q": "x", "k": 10000}).status_code == 422


def test_html_in_titles_is_escaped_by_the_page_not_injected(client):
    assert "esc(h.title)" in client.get("/").text and "innerHTML = r.hits" in client.get("/").text


def test_key_papers_without_a_graph_is_a_clear_409(tmp_path):
    st = IndexStore(tmp_path / "ix")
    for p in generate_papers(10, 1):
        st.put(p)
    st.commit(); st.flush(); st.close()
    with TestClient(create_app(tmp_path / "ix", workers=1)) as c:
        r = c.get("/api/key", params={"q": "inverted"})
        assert r.status_code == 409 and "citation graph" in r.json()["detail"]
        assert c.get("/api/search", params={"q": "inverted postings"}).status_code == 200
