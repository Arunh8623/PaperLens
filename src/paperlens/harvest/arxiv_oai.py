"""Polite arXiv OAI-PMH harvester (SUPPORTING part of E1).

What it does and why (plain language)
-------------------------------------
Downloads paper metadata (title, abstract, authors, categories ...) from
arXiv's official bulk interface, OAI-PMH, and appends it to a JSONL file.
"Polite" means it obeys arXiv's rules:
  * at most 1 request / 3 s, one connection at a time  (Terms of Use)
  * identifies itself with a contact email in the User-Agent
    (OUR policy: arXiv's pages do not require it, but it is good manners and
    lets them reach us). The email is never defaulted - you must supply it.
  * on HTTP 429/503 waits for the Retry-After the server asks for
  * resumable: a state file remembers the resumptionToken; if the token has
    expired (arXiv says tokens expire daily) it restarts from the newest
    datestamp it has seen.

Delivery guarantee: AT LEAST once. Records are written + fsynced BEFORE the
state file is updated, so a crash can only cause a page to be fetched again
(duplicate keys in the JSONL), never a gap. Duplicates are harmless: the index
ignores a key it already has (see contracts/paper.py).

Facts checked on 2026-10-31 (info.arxiv.org): base URL
https://oaipmh.arxiv.org/oai ; sets `cs`, `cs:cs`, `cs:cs:AI` ... exist
(ListSets); datestamp = last-modified time, NOT submission date; tokens expire
daily; incremental harvests use `from` and no `until`.

NOT yet verified against live data (needs the first real run): the exact XML
of metadataPrefix=arXiv (parser is written from memory and is namespace
agnostic), and whether records arrive in non-decreasing datestamp order (the
restart-from-newest rule assumes so; `out_of_order` in the state file counts
violations so the real run settles it).
"""
from __future__ import annotations

import email.utils
import http.client
import json
import os
import socket
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone

from ..contracts.paper import PaperError, PaperRecord
from ..store.fsutil import fsync_dir, write_file_durable
from .limiter import MinIntervalLimiter

DEFAULT_BASE_URL = "https://oaipmh.arxiv.org/oai"
# arXiv's rule is 1 request / 3 s. We use 3.2 s: a caller that wakes up a little
# late must not end up closer than 3 s to the next request (scheduling jitter).
ARXIV_RULE_SECONDS = 3.0
ARXIV_MIN_INTERVAL = 3.2


class HarvestError(Exception):
    pass


class TokenExpired(HarvestError):
    pass


class OAIError(HarvestError):
    def __init__(self, code, message):
        super().__init__(f"OAI error {code}: {message}")
        self.code = code


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------
def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child(elem, name):
    for c in elem:
        if _local(c.tag) == name:
            return c
    return None


def _text(elem, name) -> str:
    c = _child(elem, name)
    return " ".join((c.text or "").split()) if c is not None else ""


@dataclass
class Page:
    records: list = field(default_factory=list)
    deleted: list = field(default_factory=list)      # arxiv ids marked deleted
    token: str | None = None
    skipped_bad: int = 0
    bad_examples: list = field(default_factory=list)


def _author_name(a) -> str:
    parts = [_text(a, "forenames"), _text(a, "keyname"), _text(a, "suffix")]
    return " ".join(p for p in parts if p)


def parse_response(xml_bytes: bytes) -> Page:
    """Parse one OAI-PMH ListRecords response. Raises ET.ParseError on broken XML."""
    root = ET.fromstring(xml_bytes)
    err = next((e for e in root if _local(e.tag) == "error"), None)
    if err is not None:
        code, msg = err.get("code", "?"), (err.text or "").strip()
        if code == "noRecordsMatch":
            return Page()
        if code == "badResumptionToken":
            raise TokenExpired(msg or "badResumptionToken")
        raise OAIError(code, msg)
    lr = next((e for e in root if _local(e.tag) == "ListRecords"), None)
    if lr is None:
        raise HarvestError("response has neither <error> nor <ListRecords>")
    page = Page()
    for rec in lr:
        name = _local(rec.tag)
        if name == "resumptionToken":
            page.token = (rec.text or "").strip() or None
            continue
        if name != "record":
            continue
        header = _child(rec, "header")
        ident = _text(header, "identifier") if header is not None else ""
        if header is not None and header.get("status") == "deleted":
            page.deleted.append(ident.split(":")[-1])
            continue
        try:
            meta = _child(rec, "metadata")
            ax = meta[0] if meta is not None and len(meta) else None
            if ax is None:
                raise PaperError("record has no metadata")
            authors_el = _child(ax, "authors")
            authors = [n for n in (_author_name(a) for a in authors_el if _local(a.tag) == "author") if n] \
                if authors_el is not None else []
            title = _text(ax, "title")
            if not title:
                raise PaperError("record has no title")
            page.records.append(PaperRecord(
                arxiv_id=_text(ax, "id") or ident.split(":")[-1],
                datestamp=_text(header, "datestamp")[:10],
                title=title,
                abstract=_text(ax, "abstract"),
                authors=authors,
                categories=_text(ax, "categories").split(),
                created=_text(ax, "created")[:10],
                updated=_text(ax, "updated")[:10],
                doi=_text(ax, "doi"),
            ))
        except PaperError as e:
            page.skipped_bad += 1
            if len(page.bad_examples) < 5:
                page.bad_examples.append(f"{ident}: {e}")
    return page


def parse_retry_after(value, now: datetime | None = None) -> float | None:
    """Retry-After is either delta-seconds or an HTTP-date. None if unusable."""
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    try:
        secs = float(value)
        return secs if secs >= 0 else None
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    now = now or datetime.now(timezone.utc)
    return max((when - now).total_seconds(), 0.0)


# --------------------------------------------------------------------------
# HTTP client
# --------------------------------------------------------------------------
class OAIClient:
    """One polite HTTP client. All waiting goes through the limiter, so the
    3-second rule also covers retries and Retry-After pauses."""

    def __init__(self, contact: str, *, base_url: str = DEFAULT_BASE_URL,
                 limiter: MinIntervalLimiter | None = None, max_retries: int = 8,
                 backoff_base: float = 5.0, max_backoff: float = 300.0,
                 max_retry_after: float = 3600.0, timeout: float = 60.0, log=None):
        if not contact or "@" not in contact:
            raise ValueError("a contact email is required (put it in the User-Agent)")
        self.base_url = base_url
        self.limiter = limiter or MinIntervalLimiter(ARXIV_MIN_INTERVAL)
        self.user_agent = f"PaperLens/0.0.1 (KLE Tech student project; contact: {contact})"
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.max_backoff = max_backoff
        self.max_retry_after = max_retry_after
        self.timeout = timeout
        self.requests_made = 0
        self.log = log or (lambda m: None)        # progress / retry messages (CLI passes print)

    def _get(self, params: dict) -> bytes:
        url = f"{self.base_url}?{urllib.parse.urlencode(params)}"
        failures = 0
        while True:
            self.limiter.wait()
            self.requests_made += 1
            self.log(f"request {self.requests_made}: asking arXiv (waits up to {self.timeout:.0f}s for the answer)...")
            req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
            delay, why = None, ""
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = resp.read()
                return body
            except urllib.error.HTTPError as e:
                if e.code in (429, 503):
                    ra = parse_retry_after(e.headers.get("Retry-After") if e.headers else None)
                    if ra is not None and ra > self.max_retry_after:
                        raise HarvestError(f"server asked to wait {ra:.0f}s; try again later") from e
                    delay, why = ra, f"HTTP {e.code}"
                elif 500 <= e.code < 600:
                    why = f"HTTP {e.code}"
                else:
                    raise HarvestError(f"HTTP {e.code} for {url}") from e
            except (urllib.error.URLError, http.client.HTTPException, socket.timeout,
                    ConnectionError, OSError) as e:
                why = f"{type(e).__name__}: {e}"
            failures += 1
            if failures > self.max_retries:
                raise HarvestError(f"giving up after {failures} failed attempts ({why})")
            if delay is None:                     # no Retry-After: exponential backoff
                delay = min(self.backoff_base * 2 ** (failures - 1), self.max_backoff)
            self.log(f"attempt {failures} failed ({why}); waiting {delay:.0f}s before retrying")
            self.limiter.penalize(delay)

    def list_records(self, *, set_spec="cs", from_date=None, token=None,
                     metadata_prefix="arXiv") -> Page:
        if token:
            params = {"verb": "ListRecords", "resumptionToken": token}
        else:
            params = {"verb": "ListRecords", "metadataPrefix": metadata_prefix}
            if set_spec:
                params["set"] = set_spec
            if from_date:
                params["from"] = from_date
        failures = 0
        while True:
            body = self._get(params)
            try:
                return parse_response(body)
            except ET.ParseError as e:            # truncated / garbled body -> try again
                failures += 1
                if failures > self.max_retries:
                    raise HarvestError(f"unparseable response {failures} times: {e}") from e
                self.limiter.penalize(min(self.backoff_base * 2 ** (failures - 1), self.max_backoff))


# --------------------------------------------------------------------------
# resumable harvest loop
# --------------------------------------------------------------------------
def _load_state(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def _save_state(path, state):
    tmp = path + ".tmp"
    write_file_durable(tmp, json.dumps(state, indent=1, sort_keys=True).encode("utf-8"))
    os.replace(tmp, path)
    fsync_dir(os.path.dirname(os.path.abspath(path)))


def harvest(client: OAIClient, *, set_spec: str = "cs", out_path: str, state_path: str,
            from_date: str | None = None, max_records: int | None = None,
            max_restarts: int = 5, log=print) -> dict:
    """Run (or resume) a harvest. Returns the final state dict.

    max_records: stop after the page that reaches this many LINES written
    (whole pages only, so it can overshoot by one page). Resume later by
    calling again with the same state_path.
    """
    state = _load_state(state_path) or {
        "version": 1, "set": set_spec, "from_date": from_date, "token": None,
        "newest_datestamp": None, "lines_written": 0, "pages": 0, "restarts": 0,
        "skipped_bad": 0, "deleted_seen": 0, "out_of_order": 0, "done": False,
    }
    if state["set"] != set_spec:
        raise HarvestError(f"state file is for set {state['set']!r}, not {set_spec!r}")
    if state["done"]:
        log("harvest already complete (delete the state file to run again)")
        return state
    restarts_now = 0
    while True:
        try:
            page = client.list_records(set_spec=set_spec, from_date=state["from_date"],
                                       token=state["token"])
        except TokenExpired:
            restarts_now += 1
            if restarts_now > max_restarts:
                raise HarvestError("resumption token keeps expiring; giving up")
            # Restart from the newest datestamp seen (inclusive -> small overlap).
            state["from_date"] = state["newest_datestamp"] or state["from_date"]
            state["token"] = None
            state["restarts"] += 1
            log(f"token expired; restarting from {state['from_date']}")
            _save_state(state_path, state)
            continue

        newest = state["newest_datestamp"]
        with open(out_path, "a", encoding="utf-8", newline="\n") as f:
            for p in page.records:
                if newest is not None and p.datestamp < newest:
                    state["out_of_order"] += 1
                if newest is None or p.datestamp > newest:
                    newest = p.datestamp
                f.write(p.to_json() + "\n")
            f.flush()
            os.fsync(f.fileno())                 # records durable BEFORE state moves on
        state["newest_datestamp"] = newest
        state["lines_written"] += len(page.records)
        state["pages"] += 1
        state["skipped_bad"] += page.skipped_bad
        state["deleted_seen"] += len(page.deleted)
        state["token"] = page.token
        state["done"] = page.token is None
        _save_state(state_path, state)
        for ex in page.bad_examples:
            log(f"skipped bad record: {ex}")
        log(f"page {state['pages']}: +{len(page.records)} records "
            f"(total lines {state['lines_written']}, newest {newest})")
        if state["done"]:
            return state
        if max_records is not None and state["lines_written"] >= max_records:
            log(f"stopping at {state['lines_written']} lines (resumable)")
            return state


def summarize_jsonl(path) -> dict:
    """Lines, unique keys, distinct papers, date range of a harvested file."""
    from ..contracts.paper import read_jsonl
    lines, keys, ids, dmin, dmax = 0, set(), set(), None, None
    for p in read_jsonl(path):
        lines += 1
        keys.add(p.key)
        ids.add(p.arxiv_id)
        dmin = p.datestamp if dmin is None or p.datestamp < dmin else dmin
        dmax = p.datestamp if dmax is None or p.datestamp > dmax else dmax
    return {"lines": lines, "unique_keys": len(keys), "distinct_papers": len(ids),
            "datestamp_min": dmin, "datestamp_max": dmax}
