"""Publication info + citation counts from OpenAlex, kept in a side file (FR12).

What it does and why (plain language)
-------------------------------------
arXiv only tells us "this preprint exists". To show "is it published, where, what is its DOI, how many papers cite it"
we ask OpenAlex (a free scholarly database) about each paper and keep the answer in a JSONL side file, one line per
paper. The index itself is never touched.

Two things that are easy to get wrong, so they are spelled out:

1. OpenAlex merges a preprint and its journal version into ONE "work". The work's `doi` is the PUBLISHER's DOI when
   there is one (OpenAlex documents: "a published version's DOI beats a preprint's"). So looking a paper up only by
   its arXiv DOI (10.48550/arXiv.<id>) can miss exactly the papers that were published. We therefore
     pass 1: ask by arXiv DOI;   pass 2: for ids still missing, ask by the arXiv landing page of any location;
   and we map every returned work back to our ids using ALL the places it mentions (doi, location urls, oai ids),
   not only work.doi. Pass 2 uses a filter name read from OpenAlex's docs but NEVER TESTED against the live API; if the
   API rejects it we say so once and carry on with pass 1 only.
2. "Published" is our rule, written down so it can be argued with:
     published = type is not "preprint" AND
        ( some location is a publishedVersion in a journal/conference source      -> basis "journal-location"
          OR the work's DOI is a non-arXiv DOI and type is article/book-chapter/review -> basis "publisher-doi" ).
   A DOI that the AUTHOR typed into the arXiv record counts too, at the web layer (basis "arxiv-doi"), because arXiv's
   `doi` field is meant for the journal version. That one works even before OpenAlex has been asked anything.

Citation counts are OpenAlex's ("successful reference matches"). They are NOT Google Scholar's numbers and are
usually lower; the UI says "OpenAlex" next to them.

Side-file line (see docs/CONTRACTS.md): {"arxiv_id","openalex_id","matched_by","cited_by_count","type","published",
"basis","doi","venue","venue_type","year","volume","issue","pages","publisher_url","retracted","fetched"}.
Papers OpenAlex does not know are stored with openalex_id null so they are not asked for again.
Resumable and fsync'ed per batch, exactly like the citation downloader.
"""
from __future__ import annotations

import datetime
import json
import os
import re
import threading

from ..graph.openalex import MICRO, OpenAlexError, arxiv_to_doi, doi_to_arxiv

SELECT = ("id,doi,cited_by_count,type,publication_year,is_retracted,biblio,"
          "primary_location,best_oa_location,locations")
PUBLISHED_SOURCE_TYPES = {"journal", "conference", "book series"}
ARTICLE_TYPES = {"article", "book-chapter", "review"}

_ABS = re.compile(r"arxiv\.org/(?:abs|pdf)/(.+?)(?:v\d+)?(?:\.pdf)?(?:[?#].*)?$", re.I)
_OAI = re.compile(r"oai:arxiv\.org:(.+?)(?:v\d+)?$", re.I)


def bare_doi(doi: str) -> str:
    d = (doi or "").strip()
    for p in ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "doi:"):
        if d.lower().startswith(p):
            d = d[len(p):]
    return d


def is_arxiv_doi(doi: str) -> bool:
    return bare_doi(doi).lower().startswith("10.48550/arxiv.")


def arxiv_ids_of(work: dict) -> set:
    """Every arXiv id (lower case, no version) that this OpenAlex work claims, from ALL its places."""
    found = set()
    a = doi_to_arxiv(work.get("doi"))
    if a:
        found.add(a)
    for loc in [work.get("primary_location"), work.get("best_oa_location")] + list(work.get("locations") or []):
        if not loc:
            continue
        for field in ("landing_page_url", "pdf_url"):
            m = _ABS.search(loc.get(field) or "")
            if m:
                found.add(m.group(1).lower())
        m = _OAI.search(loc.get("id") or "")
        if m:
            found.add(m.group(1).lower())
        a = doi_to_arxiv(loc.get("doi"))
        if a:
            found.add(a)
    return found


def normalise(arxiv_id: str, work: dict, matched_by: str, today: str = None) -> dict:
    """Turn one OpenAlex work into our side-file record. Pure function (tested without a network)."""
    locs = [l for l in (work.get("locations") or []) if l]
    pub_loc = next((l for l in locs if (l.get("is_published") or l.get("version") == "publishedVersion")
                    and (l.get("source") or {}).get("type") in PUBLISHED_SOURCE_TYPES), None)
    wtype = work.get("type") or ""
    doi = bare_doi(work.get("doi") or "")
    published, basis = False, ""
    if wtype != "preprint":
        if pub_loc is not None:
            published, basis = True, "journal-location"
        elif doi and not is_arxiv_doi(doi) and wtype in ARTICLE_TYPES:
            published, basis = True, "publisher-doi"
    src = ((pub_loc or work.get("primary_location") or {}).get("source") or {}) if published else {}
    bib = work.get("biblio") or {}
    pages = "-".join(x for x in (bib.get("first_page"), bib.get("last_page")) if x)
    if bib.get("first_page") and bib.get("first_page") == bib.get("last_page"):
        pages = bib["first_page"]
    return {
        "arxiv_id": arxiv_id, "openalex_id": work.get("id"), "matched_by": matched_by,
        "cited_by_count": work.get("cited_by_count"), "type": wtype,
        "published": published, "basis": basis,
        "doi": doi if (published and not is_arxiv_doi(doi)) else "",
        "venue": src.get("display_name") or "", "venue_type": src.get("type") or "",
        "year": work.get("publication_year"), "volume": bib.get("volume") or "", "issue": bib.get("issue") or "",
        "pages": pages, "publisher_url": (pub_loc or {}).get("landing_page_url") or "" if published else "",
        "retracted": bool(work.get("is_retracted")),
        "fetched": today or datetime.date.today().isoformat(),
    }


def unknown_record(arxiv_id: str, today: str = None) -> dict:
    return {"arxiv_id": arxiv_id, "openalex_id": None, "matched_by": None, "cited_by_count": None, "type": "",
            "published": False, "basis": "", "doi": "", "venue": "", "venue_type": "", "year": None, "volume": "",
            "issue": "", "pages": "", "publisher_url": "", "retracted": False,
            "fetched": today or datetime.date.today().isoformat()}


def _pass(client, ids: list, flt: str) -> dict:
    """One request. Returns {our_id: work} for every id this request answered."""
    data = client.get_works({"filter": flt, "per_page": 100, "select": SELECT})
    wanted = {a.lower(): a for a in ids}
    out = {}
    for w in data.get("results", []):
        for key in arxiv_ids_of(w):
            if key in wanted and wanted[key] not in out:
                out[wanted[key]] = w
    return out


def _load_done(path) -> dict:
    done = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done[r["arxiv_id"]] = r
                except (json.JSONDecodeError, KeyError):
                    continue                       # a torn last line from a crash: that batch is asked again
    return done


def fetch_meta(client, arxiv_ids: list, out_path, log=None) -> dict:
    """Fill `out_path` with a record for every id not already in it. Returns the stats of THIS run:
    {"asked","by_doi","by_location","unknown","fallback":"ok"|"unavailable"|"unused"}.
    Raises BudgetExhausted/OpenAlexError after saving what it has."""
    out_path = os.fspath(out_path)
    done = _load_done(out_path)
    todo = [a for a in arxiv_ids if a not in done]
    stats = {"asked": len(todo), "by_doi": 0, "by_location": 0, "unknown": 0, "fallback": "unused"}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fallback_ok = True
    with open(out_path, "ab") as f:
        for i in range(0, len(todo), client.batch):
            chunk = todo[i:i + client.batch]
            got = {a: ("doi", w) for a, w in _pass(
                client, chunk, "doi:" + "|".join(arxiv_to_doi(a) for a in chunk)).items()}
            missing = [a for a in chunk if a not in got]
            if missing and fallback_ok:
                try:
                    got.update({a: ("location", w) for a, w in _pass(
                        client, missing, "locations.landing_page_url:" + "|".join(
                            f"https://arxiv.org/abs/{a}" for a in missing)).items()})
                    stats["fallback"] = "ok"
                except OpenAlexError as e:
                    if "HTTP 400" in str(e):       # the filter name is not accepted: stop trying, say so once
                        fallback_ok = False
                        stats["fallback"] = "unavailable"
                        if log:
                            log("note: OpenAlex rejected the location filter; published papers whose OpenAlex "
                                "record is filed under the publisher DOI may be missing from this run")
                    else:
                        raise
            lines = []
            for a in chunk:
                if a in got:
                    how, w = got[a]
                    rec = normalise(a, w, how)
                    stats["by_doi" if how == "doi" else "by_location"] += 1
                else:
                    rec = unknown_record(a)
                    stats["unknown"] += 1
                lines.append(json.dumps(rec, separators=(",", ":")) + "\n")
            f.write("".join(lines).encode("utf-8"))
            f.flush()
            os.fsync(f.fileno())                   # durable BEFORE the next request
            if log:
                log(f"{min(i + client.batch, len(todo))}/{len(todo)} done, {client.requests_made} requests, "
                    f"{client.spent_units / MICRO:.4f} USD")
    return stats


class MetaStore:
    """Read side: the side file as a dict, re-read automatically when `enrich` has appended to it."""

    def __init__(self, path):
        self.path = os.fspath(path) if path else None
        self._sig = None
        self._data = {}
        self._lock = threading.Lock()

    def _refresh(self) -> None:
        if not self.path or not os.path.exists(self.path):
            self._data, self._sig = {}, None
            return
        st = os.stat(self.path)
        sig = (st.st_size, st.st_mtime_ns)
        if sig != self._sig:
            self._data, self._sig = _load_done(self.path), sig

    def get(self, arxiv_id: str):
        with self._lock:
            self._refresh()
            return self._data.get(arxiv_id)

    def __len__(self) -> int:
        with self._lock:
            self._refresh()
            return len(self._data)


def merge_view(paper_doi: str, rec) -> dict:
    """What the UI shows for one paper: OpenAlex record (if any) + the DOI the author typed on arXiv (if any).
    Returns {"cited_by", "published", "basis", "doi", "doi_url", "venue", "year", "pub_info", "retracted", "have_meta"}."""
    rec = rec or {}
    published, basis, doi = bool(rec.get("published")), rec.get("basis", ""), rec.get("doi", "")
    if not published and paper_doi and not is_arxiv_doi(paper_doi):
        published, basis, doi = True, "arxiv-doi", bare_doi(paper_doi)
    parts = []
    if rec.get("venue") and published and basis != "arxiv-doi":
        parts.append(rec["venue"])
        vi = rec.get("volume", "")
        if rec.get("issue"):
            vi += f"({rec['issue']})"
        if vi:
            parts.append(vi)
        if rec.get("pages"):
            parts.append("pp. " + rec["pages"])
        if rec.get("year"):
            parts.append(str(rec["year"]))
    return {"cited_by": rec.get("cited_by_count"), "published": published, "basis": basis, "doi": doi,
            "doi_url": f"https://doi.org/{doi}" if doi else "", "venue": rec.get("venue", "") if published else "",
            "year": rec.get("year"), "pub_info": ", ".join(parts), "retracted": bool(rec.get("retracted")),
            "publisher_url": rec.get("publisher_url", "") if published else "", "have_meta": bool(rec.get("openalex_id"))}
