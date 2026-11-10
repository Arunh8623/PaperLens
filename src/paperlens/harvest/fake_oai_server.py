"""A tiny local OAI-PMH server for testing the harvester (no network needed).

What it does and why
--------------------
Tests must never hammer the real arXiv. This serves synthetic papers in the
same shape as arXiv's `arXiv` metadata format (as remembered - NOT yet
checked against a live response, so a green test here proves the harvester is
self-consistent, not that it matches arXiv byte for byte).

It can misbehave on purpose. `faults` maps a 1-based REQUEST NUMBER to:
    ("http", 429, "2")    reply HTTP 429 with Retry-After: 2   (None = no header)
    ("drop",)             close the connection without replying
    ("garbage",)          reply 200 with a truncated XML body
    ("expire",)           reply OAI error badResumptionToken
`log` records (monotonic_time, query_dict) for every request so tests can
check spacing and which arguments the client used.
"""
from __future__ import annotations

import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from xml.sax.saxutils import escape

OAI_NS = "http://www.openarchives.org/OAI/2.0/"
ARXIV_NS = "http://arxiv.org/OAI/arXiv/"


def _author_xml(name: str) -> str:
    forenames, _, keyname = name.rpartition(" ")
    fn = f"<forenames>{escape(forenames)}</forenames>" if forenames else ""
    return f"<author><keyname>{escape(keyname)}</keyname>{fn}</author>"


def record_xml(p) -> str:
    authors = "".join(_author_xml(a) for a in p.authors)
    upd = f"<updated>{p.updated}</updated>" if p.updated else ""
    doi = f"<doi>{escape(p.doi)}</doi>" if p.doi else ""
    return (
        f"<record><header><identifier>oai:arXiv.org:{p.arxiv_id}</identifier>"
        f"<datestamp>{p.datestamp}</datestamp><setSpec>cs</setSpec></header>"
        f'<metadata><arXiv xmlns="{ARXIV_NS}"><id>{p.arxiv_id}</id>'
        f"<created>{p.created or p.datestamp}</created>{upd}"
        f"<authors>{authors}</authors><title>{escape(p.title)}</title>"
        f"<categories>{' '.join(p.categories)}</categories>"
        f"<abstract>\n  {escape(p.abstract)}\n</abstract>{doi}</arXiv></metadata></record>"
    )


def deleted_xml(arxiv_id: str, datestamp: str) -> str:
    return (f'<record><header status="deleted"><identifier>oai:arXiv.org:{arxiv_id}</identifier>'
            f"<datestamp>{datestamp}</datestamp></header></record>")


class FakeOAIServer:
    def __init__(self, papers, page_size=50, faults=None, deleted=()):
        self.papers = sorted(papers, key=lambda p: (p.datestamp, p.arxiv_id))
        self.page_size = page_size
        self.faults = dict(faults or {})
        self.deleted = list(deleted)              # [(arxiv_id, datestamp)] put on page 1
        self.log: list = []
        self.user_agents: list = []
        self.generation = 0                      # bump to invalidate all old tokens
        self._n = 0
        self._lock = threading.Lock()
        self._httpd = None
        self._thread = None

    # ---- lifecycle ----
    def __enter__(self):
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                outer._handle(self)

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self._httpd.daemon_threads = True
        # poll_interval: shutdown() waits for one poll; the 0.5 s default made every test slow
        self._thread = threading.Thread(
            target=lambda: self._httpd.serve_forever(poll_interval=0.01), daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_address[1]}/oai"

    def invalidate_tokens(self):
        self.generation += 1

    # ---- protocol ----
    def _wrap(self, inner: str, verb="ListRecords") -> bytes:
        return (f'<?xml version="1.0" encoding="UTF-8"?><OAI-PMH xmlns="{OAI_NS}">'
                f"<responseDate>2026-01-01T00:00:00Z</responseDate>"
                f'<request verb="{verb}">http://fake/oai</request>{inner}</OAI-PMH>').encode("utf-8")

    def _error(self, code, msg) -> bytes:
        return self._wrap(f'<error code="{code}">{escape(msg)}</error>')

    def _handle(self, h):
        q = {k: v[0] for k, v in urllib.parse.parse_qs(urllib.parse.urlparse(h.path).query).items()}
        with self._lock:
            self._n += 1
            n = self._n
            self.log.append((time.monotonic(), q))
            self.user_agents.append(h.headers.get("User-Agent"))
        fault = self.faults.get(n)

        def reply(status, body=b"", headers=()):
            h.send_response(status)
            for k, v in headers:
                h.send_header(k, v)
            h.send_header("Content-Type", "text/xml; charset=utf-8")
            h.send_header("Content-Length", str(len(body)))
            h.end_headers()
            h.wfile.write(body)

        if fault and fault[0] == "http":
            _, code, retry_after = fault
            return reply(code, b"", [("Retry-After", retry_after)] if retry_after else [])
        if fault and fault[0] == "drop":
            h.close_connection = True
            return
        if fault and fault[0] == "expire":
            return reply(200, self._error("badResumptionToken", "token expired"))

        if q.get("verb") != "ListRecords":
            return reply(200, self._error("badVerb", "Invalid verb provided."))
        if "resumptionToken" in q:
            try:
                gen, frm, off = q["resumptionToken"].split("|")
                gen, off = int(gen), int(off)
            except ValueError:
                return reply(200, self._error("badResumptionToken", "malformed"))
            if gen != self.generation:
                return reply(200, self._error("badResumptionToken", "token expired"))
        else:
            if "metadataPrefix" not in q:
                return reply(200, self._error("badArgument", "metadataPrefix required"))
            if q.get("set") not in (None, "cs"):
                return reply(200, self._error("noRecordsMatch", "no such set"))
            frm, off = q.get("from", ""), 0

        sel = [p for p in self.papers if p.datestamp >= frm] if frm else self.papers
        if not sel:
            return reply(200, self._error("noRecordsMatch", "no records"))
        chunk = sel[off:off + self.page_size]
        recs = "".join(record_xml(p) for p in chunk)
        if off == 0 and not q.get("resumptionToken"):
            recs = "".join(deleted_xml(a, d) for a, d in self.deleted) + recs
        more = off + self.page_size < len(sel)
        tok = f"<resumptionToken>{self.generation}|{frm}|{off + self.page_size}</resumptionToken>" \
            if more else "<resumptionToken/>"
        body = self._wrap(f"<ListRecords>{recs}{tok}</ListRecords>")
        if fault and fault[0] == "garbage":
            body = body[: len(body) // 2]
        return reply(200, body)
