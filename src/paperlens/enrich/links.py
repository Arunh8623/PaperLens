"""Where to read a paper in full: [PDF] and [HTML] links (like the right-hand column of Google Scholar).

What it does and why (plain language)
-------------------------------------
Every arXiv paper has a PDF at arxiv.org/pdf/<id> -- that link always exists, so it is always shown.
HTML versions are different, and this is what the arXiv paper "Scaling Accessible Mathematics on arXiv" (2026) says:
  * papers submitted since late 2023 get an HTML version made automatically -> arxiv.org/html/<id>
    (about 97% produce some HTML, but only ~75% convert without errors, so a few pages may look rough or be missing)
  * older papers are NOT in that live channel; historical ones are on the ar5iv site -> ar5iv.labs.arxiv.org/html/<id>
    (covers roughly the 90% of papers that have TeX source)
We do NOT check each link over the network (that would be one request per result and arXiv asks for politeness),
so the [HTML] link is a best guess: it can occasionally open a "not available" page. The PDF link never guesses.
"""
from __future__ import annotations

import re

_NEW = re.compile(r"^(\d{2})(\d{2})\.\d{4,5}$")          # 2401.00001  (since April 2007)
_HTML_LIVE_FROM = (23, 12)                                # Dec 2023: first month with automatic HTML


def abs_url(arxiv_id: str) -> str:
    return f"https://arxiv.org/abs/{arxiv_id}"


def pdf_url(arxiv_id: str) -> str:
    return f"https://arxiv.org/pdf/{arxiv_id}"


def html_url(arxiv_id: str) -> str:
    """Best-guess HTML location (see module docstring). Old-style ids like 'cs/0601001' go to ar5iv."""
    m = _NEW.match(arxiv_id)
    if m and (int(m.group(1)), int(m.group(2))) >= _HTML_LIVE_FROM:
        return f"https://arxiv.org/html/{arxiv_id}"
    return f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"


def links_for(arxiv_id: str, publisher_url: str = "") -> dict:
    out = {"abs": abs_url(arxiv_id), "pdf": pdf_url(arxiv_id), "html": html_url(arxiv_id)}
    if publisher_url:
        out["publisher"] = publisher_url
    return out
