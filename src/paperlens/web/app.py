"""FastAPI app: JSON API and a single self-contained HTML page (no CDN, works offline).

What it does and why (plain language)
-------------------------------------
The web process owns no index data. It starts the query-worker processes (server/query_server.py), which read frozen
snapshots, and forwards requests to them. So the page keeps answering while another process ingests or compacts, and
every answer shows which snapshot ("generation") it came from.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse

from ..server.query_server import QueryServer, QueryServerError

PAGE = os.path.join(os.path.dirname(__file__), "static", "index.html")


def create_app(index_path, workers: int = 2) -> FastAPI:
    holder = {}

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

    @app.get("/", response_class=HTMLResponse)
    def home():
        with open(PAGE, encoding="utf-8") as f:
            return f.read()

    @app.get("/api/health")
    async def health():
        s = await call(qs().stats)
        return {"ok": True, "generation": s["generation"], "workers": workers}

    @app.get("/api/stats")
    async def stats():
        return await call(qs().stats)

    @app.get("/api/search")
    async def search(q: str = Query(..., min_length=1, max_length=500), k: int = Query(10, ge=1, le=100),
                     mode: str = Query("bm25", pattern="^(bm25|vector|hybrid)$"), cat: list[str] = Query(default=[]),
                     year_from: int = Query(0, ge=0, le=3000), year_to: int = Query(0, ge=0, le=3000),
                     graph: bool = False, late: bool = False, dedupe: bool = False):
        r = await call(qs().search, q, k=k, mode=mode, cat=cat, year_from=year_from, year_to=year_to,
                       graph_rerank=graph, late=late, dedupe=dedupe)
        return r

    @app.get("/api/similar/{arxiv_id:path}")
    async def similar(arxiv_id: str, k: int = Query(10, ge=1, le=50)):
        r = await call(qs().similar, arxiv_id, k=k)
        if r.get("error") == "not_found":
            raise HTTPException(404, "no such paper")
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
        return r

    return app
