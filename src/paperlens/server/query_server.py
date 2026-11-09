"""QueryServer: N query worker processes, each with a PRIVATE channel, serving read-only snapshots.

What it does and why (plain language)
-------------------------------------
Search must keep working while the writer ingests and compacts. Each worker process opens an
`IndexReader` (a pinned, lock-free snapshot) and answers requests sent over its own pipe. Before it
answers it checks whether the writer published a newer MANIFEST and, if so, switches snapshots (at most
every `refresh_interval` seconds). Every answer says which generation it came from, so a test (or the UI)
can prove which consistent snapshot it saw. As in the ingestion pipeline there is one private channel per
worker incarnation: a killed worker can never corrupt anybody else's queue; a dead worker is replaced and
the (read-only, repeatable) request is retried once.
"""
from __future__ import annotations

import multiprocessing as mp
import threading
import time


# ----------------------------------------------------------------------------- worker side
def _hit_dict(reader, rank, h, mode):
    p = reader.get(h.arxiv_id)
    d = {"rank": rank, "arxiv_id": h.arxiv_id, "score": float(h.score), "title": p.title if p else "",
         "categories": list(p.categories) if p else [], "datestamp": p.datestamp if p else ""}
    if mode == "hybrid":
        d["bm25_rank"], d["vec_rank"] = h.bm25_rank, h.vec_rank
    return d


def _serve(conn, index_path, refresh_interval):
    """Worker main loop (module level so 'spawn' can import it)."""
    from ..store.filters import Filter
    from ..store.reader import IndexReader
    reader = None
    try:
        reader = IndexReader(index_path)
        conn.send(("ready", reader.generation))
        last = time.monotonic()
        embedder = None
        while True:
            op, args = conn.recv()
            if op == "stop":
                break
            if op == "refresh" or time.monotonic() - last >= refresh_interval:
                reader.refresh()
                last = time.monotonic()
            t0 = time.perf_counter()
            if op == "search":
                text, k, mode = args["query"], args["k"], args["mode"]
                flt = Filter(categories=tuple(args.get("cat") or ()), year_from=args.get("year_from") or 0,
                             year_to=args.get("year_to") or 0)
                if mode in ("vector", "hybrid"):
                    if embedder is None:
                        from ..vectors.embedders import make_embedder
                        embedder = make_embedder(reader.embedder_name)
                    qv = embedder.embed_query(text)
                if mode == "bm25":
                    hits = reader.search(text, k=k, flt=flt)
                elif mode == "vector":
                    hits = reader.search_vector(qv, k=k, flt=flt)
                else:
                    hits = reader.search_hybrid(text, qv, k=k, flt=flt)
                if args.get("graph_rerank") and reader.graph is not None and hits:
                    from ..graph.rerank import citation_rerank
                    by_id = {h.arxiv_id: h for h in hits}
                    fused = citation_rerank([h.arxiv_id for h in hits], reader.graph)
                    hits = [by_id[a] for a, *_ in fused]
                out = {"generation": reader.generation, "last_flushed_seq": reader.manifest.last_flushed_seq,
                       "hits": [_hit_dict(reader, i, h, mode) for i, h in enumerate(hits, 1)]}
            elif op == "stats":
                out = {"generation": reader.generation, **reader.stats()}
            elif op == "refresh":
                out = {"generation": reader.generation}
            elif op == "ping":
                out = {"generation": reader.generation, "pid": __import__("os").getpid()}
            else:
                out = {"error": f"unknown op {op!r}"}
            out["took_ms"] = round((time.perf_counter() - t0) * 1000, 3)
            conn.send(("ok", out))
    except (EOFError, KeyboardInterrupt):
        pass
    except BaseException as e:                           # report, then die: the parent will replace us
        try:
            conn.send(("error", f"{type(e).__name__}: {e}"))
        except Exception:
            pass
    finally:
        if reader is not None:
            reader.close()


# ----------------------------------------------------------------------------- parent side
class QueryServerError(Exception):
    pass


class _Worker:
    def __init__(self, ctx, index_path, refresh_interval, start_timeout):
        self.parent, child = ctx.Pipe(duplex=True)
        self.proc = ctx.Process(target=_serve, args=(child, index_path, refresh_interval), daemon=True)
        self.proc.start()
        child.close()
        self.lock = threading.Lock()
        if not self.parent.poll(start_timeout):
            self.kill()
            raise QueryServerError("query worker did not start in time")
        kind, val = self.parent.recv()
        if kind != "ready":
            self.kill()
            raise QueryServerError(f"query worker failed to start: {val}")
        self.first_generation = val

    def kill(self):
        if self.proc.is_alive():
            self.proc.kill()
        self.proc.join(5)
        self.parent.close()


class QueryServer:
    def __init__(self, index_path, workers: int = 1, refresh_interval: float = 0.05,
                 call_timeout: float = 120.0, start_timeout: float = 60.0):
        self.index_path, self.n = str(index_path), workers
        self.refresh_interval, self.call_timeout, self.start_timeout = refresh_interval, call_timeout, start_timeout
        self._ctx = mp.get_context("spawn")
        self._workers: list = []
        self._rr = 0
        self._rr_lock = threading.Lock()
        self.respawns = 0

    def start(self) -> "QueryServer":
        self._workers = [_Worker(self._ctx, self.index_path, self.refresh_interval, self.start_timeout)
                         for _ in range(self.n)]
        return self

    def _pick(self) -> int:
        with self._rr_lock:
            self._rr = (self._rr + 1) % len(self._workers)
            return self._rr

    def _call(self, op, args=None, worker=None) -> dict:
        idx = self._pick() if worker is None else worker
        for attempt in (0, 1):
            w = self._workers[idx]
            with w.lock:
                try:
                    w.parent.send((op, args or {}))
                    if not w.parent.poll(self.call_timeout):
                        raise TimeoutError("query worker did not answer in time")
                    kind, val = w.parent.recv()
                except (EOFError, OSError, BrokenPipeError, TimeoutError) as e:
                    kind, val = "dead", str(e)
                if kind == "ok":
                    val["worker"] = idx
                    return val
                if attempt == 0:                          # replace the worker, repeat the (read-only) request once
                    w.kill()
                    self._workers[idx] = _Worker(self._ctx, self.index_path, self.refresh_interval, self.start_timeout)
                    self.respawns += 1
                    continue
                raise QueryServerError(f"query worker {idx} failed twice: {val}")
        raise AssertionError("unreachable")

    def search(self, query: str, k: int = 10, mode: str = "bm25", cat=None, year_from: int = 0, year_to: int = 0,
               graph_rerank: bool = False) -> dict:
        return self._call("search", {"query": query, "k": k, "mode": mode, "cat": list(cat or ()),
                                     "year_from": year_from, "year_to": year_to, "graph_rerank": graph_rerank})

    def stats(self) -> dict:
        return self._call("stats")

    def refresh_all(self) -> list:
        return [self._call("refresh", worker=i)["generation"] for i in range(len(self._workers))]

    def pids(self) -> list:
        return [w.proc.pid for w in self._workers]

    def close(self) -> None:
        for w in self._workers:
            try:
                w.parent.send(("stop", {}))
            except (OSError, BrokenPipeError):
                pass
            w.proc.join(5)
            w.kill()
        self._workers = []

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.close()
