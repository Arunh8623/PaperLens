"""OpenAlex client: who cites whom, within a spending budget (SUPPORTING part of E1/E3).

What it does and why (plain language)
-------------------------------------
For each arXiv paper we ask OpenAlex for its `referenced_works` (the papers it cites).
arXiv papers have DOIs of the form 10.48550/arXiv.<id>, so we look papers up by DOI,
up to `batch` DOIs per request. The OpenAlex ids that come back are then mapped back
to arXiv ids, and citations to papers we do not hold are dropped (graph.csr).

Rules this client follows (read from OpenAlex's own documentation on 2026-11-08; the
brief said a key is REQUIRED since Feb 2026, the pages I read say it is optional but
recommended: $1/day with a free key vs $0.10/day without. NEEDS CHECKING by you):
  * the key goes in the `api_key` query parameter (never logged)
  * a list request costs $0.0001; one request may filter up to 100 DOIs; per_page max 100
  * rate limit 10 requests/s without key, 30/s with a free key; we use far less
  * 429 = over a limit -> wait (Retry-After if given) and retry; 403 key_not_activated
    = the key has not been activated by the e-mail link -> stop with a clear message
  * response headers X-RateLimit-Remaining / -Credits-Used tell the real budget

Budget safety (NFR8 "0 calls over the budget"): before EVERY request we reserve its
cost in a SharedBudget (counted in millionths of a dollar); if the reservation fails
the request is not made. We also stop when the server's own remaining-budget header
says we cannot afford another request.

Resumable: results are appended to a JSONL file (fsync before moving on); papers
already in the file are not asked for again, so a stopped run continues where it left off.
"""
from __future__ import annotations

import http.client
import json
import os
import socket
import urllib.error
import urllib.parse
import urllib.request

from ..harvest.arxiv_oai import parse_retry_after
from ..harvest.limiter import MinIntervalLimiter
from ..store.fsutil import fsync_dir

DEFAULT_BASE_URL = "https://api.openalex.org"
MICRO = 1_000_000                      # budget unit: 1 = one millionth of a dollar
LIST_COST = 100                        # $0.0001 per list request (as documented 2026-11-08)
KEYLESS_DAILY = 100_000                # $0.10
KEY_DAILY = 1_000_000                  # $1.00
DOI_PREFIX = "10.48550/arxiv."


class OpenAlexError(Exception):
    pass


class BudgetExhausted(OpenAlexError):
    pass


def arxiv_to_doi(arxiv_id: str) -> str:
    return DOI_PREFIX + arxiv_id.lower()


def doi_to_arxiv(doi: str):
    """'https://doi.org/10.48550/arxiv.2101.00001' -> '2101.00001' (lower case) or None."""
    d = (doi or "").lower()
    for p in ("https://doi.org/", "http://doi.org/", "doi:"):
        if d.startswith(p):
            d = d[len(p):]
    return d[len(DOI_PREFIX):] if d.startswith(DOI_PREFIX) else None


class OpenAlexClient:
    def __init__(self, *, api_key: str = None, contact: str = None, base_url: str = DEFAULT_BASE_URL,
                 limiter=None, budget=None, batch: int = 50, cost_per_request: int = LIST_COST,
                 max_retries: int = 6, backoff_base: float = 2.0, max_backoff: float = 120.0,
                 timeout: float = 60.0):
        if not 1 <= batch <= 100:
            raise ValueError("batch must be 1..100 (OpenAlex allows 100 OR values per filter)")
        self.api_key, self.base_url, self.batch = api_key, base_url.rstrip("/"), batch
        # 0.2 s between requests = 5/s: half the keyless limit of 10/s
        self.limiter = limiter or MinIntervalLimiter(0.2)
        self.budget = budget
        self.cost = cost_per_request
        self.max_retries, self.backoff_base, self.max_backoff, self.timeout = max_retries, backoff_base, max_backoff, timeout
        self.user_agent = "PaperLens/0.0.1 (KLE Tech student project" + (f"; contact: {contact})" if contact else ")")
        self.requests_made = 0
        self.server_remaining = None          # last X-RateLimit-Remaining seen (dollars as the server reports)
        self.spent_units = 0

    def _afford(self) -> None:
        if self.budget is not None and not self.budget.try_spend(self.cost):
            raise BudgetExhausted("our own budget for this run is used up; nothing more was requested")
        if self.server_remaining is not None and self.server_remaining <= 0:
            raise BudgetExhausted("OpenAlex says today's budget is used up (resets at midnight UTC)")

    def _get_json(self, params: dict) -> dict:
        q = dict(params)
        if self.api_key:
            q["api_key"] = self.api_key
        url = f"{self.base_url}/works?{urllib.parse.urlencode(q)}"
        failures = 0
        while True:
            self._afford()
            self.limiter.wait()
            self.requests_made += 1
            self.spent_units += self.cost
            req = urllib.request.Request(url, headers={"User-Agent": self.user_agent, "Accept": "application/json"})
            delay, why = None, ""
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    rem = resp.headers.get("X-RateLimit-Remaining")
                    if rem is not None:
                        try:
                            self.server_remaining = float(rem)
                        except ValueError:
                            pass
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                body = ""
                try:
                    body = e.read().decode("utf-8", "replace")[:200]
                except Exception:
                    pass
                if e.code == 403:
                    raise OpenAlexError("HTTP 403 from OpenAlex: " + (
                        "the API key is not activated yet (click the link in the e-mail OpenAlex sent)"
                        if "key_not_activated" in body else f"forbidden ({body})")) from e
                if e.code == 429:
                    delay = parse_retry_after(e.headers.get("Retry-After") if e.headers else None)
                    why = "HTTP 429 (rate or daily limit)"
                elif 500 <= e.code < 600:
                    why = f"HTTP {e.code}"
                else:
                    raise OpenAlexError(f"HTTP {e.code} from OpenAlex: {body}") from e
            except (urllib.error.URLError, http.client.HTTPException, socket.timeout,
                    ConnectionError, OSError, json.JSONDecodeError) as e:
                why = f"{type(e).__name__}: {e}"
            failures += 1
            if failures > self.max_retries:
                raise OpenAlexError(f"giving up after {failures} failed attempts ({why})")
            if delay is None:
                delay = min(self.backoff_base * 2 ** (failures - 1), self.max_backoff)
            self.limiter.penalize(delay)

    def works_by_arxiv_ids(self, arxiv_ids: list) -> list:
        """One request. Returns [{"arxiv_id", "openalex_id", "refs": [W-ids]}] for the papers OpenAlex knows."""
        if not arxiv_ids:
            return []
        dois = "|".join(arxiv_to_doi(a) for a in arxiv_ids)
        data = self._get_json({"filter": f"doi:{dois}", "per_page": 100, "select": "id,doi,referenced_works"})
        wanted = {a.lower(): a for a in arxiv_ids}
        out = []
        for w in data.get("results", []):
            key = doi_to_arxiv(w.get("doi"))
            if key in wanted:
                out.append({"arxiv_id": wanted[key], "openalex_id": w["id"],
                            "refs": list(w.get("referenced_works") or [])})
        return out


def _load_done(path) -> dict:
    done = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue                      # a torn last line from a crash: that batch is simply asked again
                done[r["arxiv_id"]] = r
    return done


def fetch_citations(client: OpenAlexClient, arxiv_ids: list, out_path, log=None) -> dict:
    """Fetch refs for every id not already in out_path. Returns {arxiv_id: record} of EVERYTHING in the file.
    Papers OpenAlex does not know are stored with openalex_id null so they are not asked for again.
    Raises BudgetExhausted/OpenAlexError after saving what it has."""
    out_path = os.fspath(out_path)
    done = _load_done(out_path)
    todo = [a for a in arxiv_ids if a not in done]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "ab") as f:
        for i in range(0, len(todo), client.batch):
            chunk = todo[i:i + client.batch]
            found = {r["arxiv_id"]: r for r in client.works_by_arxiv_ids(chunk)}
            lines = []
            for a in chunk:
                rec = found.get(a) or {"arxiv_id": a, "openalex_id": None, "refs": []}
                done[a] = rec
                lines.append(json.dumps(rec, separators=(",", ":")) + "\n")
            f.write("".join(lines).encode("utf-8"))
            f.flush()
            os.fsync(f.fileno())                  # durable BEFORE we ask for the next batch
            if log:
                log(f"{min(i + client.batch, len(todo))}/{len(todo)} fetched, "
                    f"{client.requests_made} requests, {client.spent_units / MICRO:.4f} USD")
    return done


def build_graph(ids: list, records: dict):
    """CSR graph over `ids` from fetched records (citations outside `ids` are dropped)."""
    from .csr import Graph
    w2a = {r["openalex_id"]: a for a, r in records.items() if r.get("openalex_id") and a in set(ids)}
    edges = []
    for a, r in records.items():
        if a in set(ids):
            edges += [(a, w2a[w]) for w in r["refs"] if w in w2a]
    return Graph.from_edges(list(ids), edges)
