"""PaperRecord: the one data shape every engine agrees on.

What it does and why (plain language)
-------------------------------------
A paper is a small, flat record. Every engine (harvester, pipeline, index,
ranking) passes this same record around, so we define it once, validate it
once, and give it one canonical JSON form.

Idempotency key = "arxiv_id@datestamp".
  * same key seen twice            -> ignored (duplicate)
  * newer datestamp, same arxiv_id -> replaces the old version
  * older datestamp, same arxiv_id -> ignored (stale)
The store applies those rules; this module only builds and compares keys.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from typing import Iterable, Iterator

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class PaperError(ValueError):
    """Raised when a record is malformed."""


@dataclass(frozen=True)
class PaperRecord:
    arxiv_id: str
    datestamp: str                       # OAI datestamp, YYYY-MM-DD
    title: str
    abstract: str
    authors: tuple = ()                  # tuple of str (immutable on purpose)
    categories: tuple = ()               # e.g. ("cs.IR", "cs.LG")
    created: str = ""                    # YYYY-MM-DD of version 1, "" if unknown
    updated: str = ""                    # YYYY-MM-DD of last update, "" if none
    doi: str = ""
    synthetic: bool = False              # True for generated test data

    def __post_init__(self):
        if not isinstance(self.arxiv_id, str) or not self.arxiv_id.strip():
            raise PaperError("arxiv_id must be a non-empty string")
        if "@" in self.arxiv_id:
            raise PaperError("arxiv_id must not contain '@' (used in the key)")
        if not isinstance(self.datestamp, str) or not _DATE_RE.match(self.datestamp):
            raise PaperError(f"datestamp must be YYYY-MM-DD, got {self.datestamp!r}")
        for name in ("title", "abstract", "doi", "created", "updated"):
            if not isinstance(getattr(self, name), str):
                raise PaperError(f"{name} must be a string")
        for name in ("created", "updated"):
            v = getattr(self, name)
            if v and not _DATE_RE.match(v):
                raise PaperError(f"{name} must be YYYY-MM-DD or empty, got {v!r}")
        # normalise list-likes to tuples of str (frozen dataclass -> object.__setattr__)
        for name in ("authors", "categories"):
            v = getattr(self, name)
            if isinstance(v, str) or not isinstance(v, Iterable):
                raise PaperError(f"{name} must be a list of strings")
            v = tuple(v)
            if not all(isinstance(x, str) for x in v):
                raise PaperError(f"{name} must contain only strings")
            object.__setattr__(self, name, v)
        if not isinstance(self.synthetic, bool):
            raise PaperError("synthetic must be a bool")

    # ---- identity -------------------------------------------------------
    @property
    def key(self) -> str:
        return make_key(self.arxiv_id, self.datestamp)

    # ---- (de)serialisation ---------------------------------------------
    def to_dict(self) -> dict:
        d = asdict(self)
        d["authors"] = list(self.authors)
        d["categories"] = list(self.categories)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PaperRecord":
        if not isinstance(d, dict):
            raise PaperError("record must be a JSON object")
        allowed = {"arxiv_id", "datestamp", "title", "abstract", "authors",
                   "categories", "created", "updated", "doi", "synthetic"}
        extra = set(d) - allowed
        if extra:
            raise PaperError(f"unknown fields: {sorted(extra)}")
        missing = {"arxiv_id", "datestamp", "title", "abstract"} - set(d)
        if missing:
            raise PaperError(f"missing fields: {sorted(missing)}")
        return cls(**d)

    def to_json(self) -> str:
        # canonical form: sorted keys, no spaces, UTF-8 kept as-is
        return json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False,
                          separators=(",", ":"))

    @classmethod
    def from_json(cls, s: str) -> "PaperRecord":
        try:
            d = json.loads(s)
        except json.JSONDecodeError as e:
            raise PaperError(f"bad JSON: {e}") from e
        return cls.from_dict(d)


def make_key(arxiv_id: str, datestamp: str) -> str:
    return f"{arxiv_id}@{datestamp}"


def split_key(key: str) -> tuple:
    arxiv_id, sep, datestamp = key.rpartition("@")
    if not sep or not arxiv_id or not datestamp:
        raise PaperError(f"bad key {key!r}")
    return arxiv_id, datestamp


def decide(existing_datestamp: str | None, incoming_datestamp: str) -> str:
    """Apply the idempotency rules for ONE arxiv_id.

    existing_datestamp: newest datestamp already stored for this arxiv_id, or None.
    Returns "insert" | "duplicate" | "replace" | "stale".
    (ISO dates compare correctly as plain strings.)
    """
    if existing_datestamp is None:
        return "insert"
    if incoming_datestamp == existing_datestamp:
        return "duplicate"
    if incoming_datestamp > existing_datestamp:
        return "replace"
    return "stale"


# ---- JSONL helpers (one record per line, UTF-8, '\n' newlines) -------------
def write_jsonl(path, papers: Iterable[PaperRecord]) -> int:
    n = 0
    # newline="\n": never let Windows turn '\n' into '\r\n'
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for p in papers:
            f.write(p.to_json() + "\n")
            n += 1
    return n


def read_jsonl(path, *, skip_bad: bool = False) -> Iterator[PaperRecord]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        for lineno, line in enumerate(f, 1):
            line = line.rstrip("\r\n")
            if not line.strip():
                continue
            try:
                yield PaperRecord.from_json(line)
            except PaperError as e:
                if skip_bad:
                    continue
                raise PaperError(f"{path}:{lineno}: {e}") from e
