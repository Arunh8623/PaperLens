"""FastAPI app: JSON API and self-contained HTML pages (no CDN, works offline).

What it does and why (plain language)
-------------------------------------
The web process owns no index data. It starts the query-worker processes (server/query_server.py), which read frozen
snapshots, and forwards requests to them. So the page keeps answering while another process ingests or compacts, and
every answer shows which snapshot ("generation") it came from.

Phase 8 adds three optional things on top, all read from SIDE FILES (never the index):
  * publication info + citation counts (enrich/openalex_meta.py)   -> "published only" filter, "Cited by N"
  * [PDF] / [HTML] links and a reader page that shows the full paper (enrich/links.py, static/reader.html)
  * AI "Quick read" through Gemini (enrich/quickread.py)
Each one switches itself off (with a visible hint, never a crash) when its data or key is missing.
"""
from __future__ import annotations

import os
import re
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse

from ..enrich.links import links_for
from ..enrich.openalex_meta import MetaStore, merge_view
from ..enrich.quickread import QuickRead, QuickReadError
from ..server.query_server import QueryServer, QueryServerError

STATIC = os.path.join(os.path.dirname(__file__), "static")
PAGE = os.path.join(STATIC, "index.html")
READER = os.path.join(STATIC, "reader.html")
_ID_OK = re.compile(r"^[A-Za-z0-9./_-]{1,40}$")          # arXiv ids: 2401.00001, 2401.00001v2, cs/0601001


def default_meta_path(index_path) -> str:
    return os.fspath(index_path).rstrip("/\\") + ".openalex_meta.jsonl"


def default_quickread_dir(index_path) -> str:
    return os.fspath(index_path).rstrip("/\\") + ".quickread"


def create_app(index_path, workers: int = 2, meta_path=None, quickread: QuickRead = None) -> FastAPI:
    holder = {}
    meta = MetaStore(meta_path if meta_path is not None else default_meta_path(index_path))
    qr = quickread if quickread is not None else QuickRead(default_quickread_dir(index_path))

    @asynccontextmanager
    async def lifespan(app):
        qs = QueryServer(index_path, workers=workers)
        await run_in_threadpool(qs.start)
        holder["qs"] = qs
        try:
            yield
        finally:
            await run_in_threadpool(qs.close)

    app = FastAPI(title="PaperLens", lifespan=lifespan)

    def qs() -> QueryServer:
        return holder["qs"]

    async def call(fn, *a, **kw):
        try:
            return await run_in_threadpool(fn, *a, **kw)
        except QueryServerError as e:
            raise HTTPException(503, f"query workers unavailable: {e}")

    def check_id(arxiv_id: str) -> None:
        if not _ID_OK.match(arxiv_id):
            raise HTTPException(400, "that does not look like an arXiv id")

    def decorate(h: dict) -> dict:
        """Add publication info, citation count and [PDF]/[HTML] links to one hit or paper dict."""
        view = merge_view(h.get("doi", ""), meta.get(h["arxiv_id"]))
        h["pub"] = view
        h["links"] = links_for(h["arxiv_id"], view["publisher_url"])
        return h

    def read_page(path: str) -> str:
        with open(path, encoding="utf-8") as f:
            return f.read()

    @app.get("/", response_class=HTMLResponse)
    def home():
        return read_page(PAGE)

    @app.get("/paper/{arxiv_id:path}", response_class=HTMLResponse)
    def reader(arxiv_id: str):
        check_id(arxiv_id)
        return read_page(READER)                  # the page fetches /api/paper/<id> itself

    @app.get("/api/health")
    async def health():
        s = await call(qs().stats)
        return {"ok": True, "generation": s["generation"], "workers": workers}

    @app.get("/api/stats")
    async def stats():
        return await call(qs().stats)

    @app.get("/api/features")
    async def features():
        """What is switched on, so the page can explain a missing feature instead of hiding it silently."""
        return {"publication_data_papers": len(meta), "quickread": qr.available, "quickread_model": qr.model}

    @app.get("/api/search")
    async def search(q: str = Query(..., min_length=1, max_length=500), k: int = Query(10, ge=1, le=100),
                     mode: str = Query("bm25", pattern="^(bm25|vector|hybrid)$"), cat: list[str] = Query(default=[]),
                     year_from: int = Query(0, ge=0, le=3000), year_to: int = Query(0, ge=0, le=3000),
                     graph: bool = False, late: bool = False, dedupe: bool = False, published: bool = False):
        # "Published only" is applied AFTER ranking, on a longer candidate list, so the order stays the ranker's.
        want = min(100, max(k * 5, 50)) if published else k
        r = await call(qs().search, q, k=want, mode=mode, cat=cat, year_from=year_from, year_to=year_to,
                       graph_rerank=graph, late=late, dedupe=dedupe)
        hits = [decorate(h) for h in r["hits"]]
        if published:
            r["published_filter"] = {"considered": len(hits), "have_openalex_data": len(meta) > 0}
            hits = [h for h in hits if h["pub"]["published"]][:k]
        for i, h in enumerate(hits, 1):
            h["rank"] = i
        r["hits"] = hits
        return r

    @app.get("/api/similar/{arxiv_id:path}")
    async def similar(arxiv_id: str, k: int = Query(10, ge=1, le=50)):
        r = await call(qs().similar, arxiv_id, k=k)
        if r.get("error") == "not_found":
            raise HTTPException(404, "no such paper")
        r["hits"] = [decorate(h) for h in r.get("hits", [])]
        return r

    @app.get("/api/key")
    async def key(q: str = Query(..., min_length=1, max_length=500), k: int = Query(5, ge=1, le=20)):
        r = await call(qs().key_papers, q, k=k)
        if r.get("error") == "no_graph":
            raise HTTPException(409, "this index has no citation graph yet (run `citations` and `graph-build`)")
        return r

    @app.get("/api/paper/{arxiv_id:path}")
    async def paper(arxiv_id: str):
        r = await call(qs().paper, arxiv_id)
        if r.get("error") == "not_found":
            raise HTTPException(404, "no such paper")
        decorate(r["paper"])
        return r

    @app.get("/api/quickread/{arxiv_id:path}")
    async def quickread_cached(arxiv_id: str):
        """Free to call: returns the saved summary if this paper was explained before, never contacts Gemini."""
        check_id(arxiv_id)
        return {"available": qr.available, "model": qr.model, "result": await run_in_threadpool(qr.cached, arxiv_id)}

    @app.post("/api/quickread/{arxiv_id:path}")
    async def quickread_make(arxiv_id: str, x_paperlens: str = Header(default="")):
        """Spends Gemini quota, so: POST only, and a custom header that another website cannot add to a request
        to this server without our permission (browsers refuse that cross-site), which blocks 'click-jacking' of the key."""
        check_id(arxiv_id)
        if x_paperlens != "1":
            raise HTTPException(403, "missing X-PaperLens header")
        r = await call(qs().paper, arxiv_id)
        if r.get("error") == "not_found":
            raise HTTPException(404, "no such paper")
        p = r["paper"]
        try:
            return await run_in_threadpool(qr.get, arxiv_id, p["title"], p["abstract"])
        except QuickReadError as e:
            raise HTTPException(e.status, str(e))

    return app
