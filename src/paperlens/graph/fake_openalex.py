"""A tiny local stand-in for api.openalex.org/works (tests only; never contacts the real service).

Shape follows what the real API documents (as read on 2026-11-08): GET /works?filter=doi:a|b&select=...&per_page=...
returns {"results":[{"id","doi","referenced_works"}]}. NOT checked against a live response, so a green test proves
the client is self-consistent, not that it matches OpenAlex byte for byte.

Faults map a 1-based request number to: ("http", code, retry_after_or_None, body_text) | ("drop",).
`remaining` starts the X-RateLimit-Remaining header (dollars) and falls by 0.0001 per request.
`require_key=True` answers 403 key_not_activated unless api_key=<key>.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeOpenAlex:
    def __init__(self, papers_refs: dict, faults=None, remaining: float = 1.0, require_key=None):
        """papers_refs: {arxiv_id: [arxiv_ids it cites]}. OpenAlex ids are 'https://openalex.org/W<n>'."""
        self.ids = {a: f"https://openalex.org/W{1000 + i}" for i, a in enumerate(sorted(papers_refs))}
        self.refs = papers_refs
        self.faults = faults or {}
        self.remaining = remaining
        self.require_key = require_key
        self.log = []                 # (monotonic time, query dict)
        self.count = 0
        self._lock = threading.Lock()
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                with outer._lock:
                    outer.count += 1
                    n = outer.count
                    q = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(self.path).query))
                    outer.log.append((time.monotonic(), q))
                    fault = outer.faults.get(n)
                    outer.remaining -= 0.0001
                    rem = outer.remaining
                if fault and fault[0] == "drop":
                    self.close_connection = True
                    return
                if fault and fault[0] == "http":
                    self.send_response(fault[1])
                    if fault[2] is not None:
                        self.send_header("Retry-After", str(fault[2]))
                    self.end_headers()
                    self.wfile.write((fault[3] if len(fault) > 3 else "").encode())
                    return
                if outer.require_key is not None and q.get("api_key") != outer.require_key:
                    self.send_response(403)
                    self.end_headers()
                    self.wfile.write(b'{"error":"key_not_activated"}')
                    return
                flt = q.get("filter", "")
                dois = flt[len("doi:"):].split("|") if flt.startswith("doi:") else []
                by_doi = {"10.48550/arxiv." + a.lower(): a for a in outer.ids}
                results = []
                for d in dois[:100]:
                    a = by_doi.get(d.lower())
                    if a:
                        results.append({"id": outer.ids[a], "doi": "https://doi.org/" + d.lower(),
                                        "referenced_works": [outer.ids[r] for r in outer.refs[a] if r in outer.ids]
                                        + ["https://openalex.org/W999999"]})        # one cited paper we do not hold
                body = json.dumps({"results": results}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("X-RateLimit-Remaining", f"{rem:.6f}")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self._t = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *a):
        self.server.shutdown()
        self.server.server_close()
