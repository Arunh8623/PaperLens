"""Seeded synthetic paper generator, for plumbing and speed tests ONLY.

What it does and why
--------------------
Crash tests, pipeline tests and speed tests need thousands of papers, quickly
and reproducibly, without touching the network. This makes them from a small
made-up vocabulary, grouped into topics so a "query" has known relevant
papers.

HONEST LIMIT: the vocabulary is tiny and artificial, so this data says
NOTHING about search quality. Quality is measured on real arXiv data and BEIR
(Phase 1+). Every record has synthetic=True so it can never be mistaken for
real data.

Same seed -> byte-identical output, on every OS.
"""
from __future__ import annotations

import random

from .contracts.paper import PaperRecord

CATEGORIES = ["cs.IR", "cs.LG", "cs.CL", "cs.DB", "cs.OS", "cs.DS", "cs.CV", "cs.DC"]

# topic -> distinctive words. A paper draws mostly from ONE topic.
TOPICS = {
    "indexing": ["inverted", "postings", "segment", "compaction", "merge", "skiplist", "wand", "bm25"],
    "learning": ["gradient", "transformer", "embedding", "pretraining", "attention", "loss", "k-means", "bert-base"],
    "storage": ["journal", "fsync", "checksum", "crash", "recovery", "durability", "snapshot", "manifest"],
    "systems": ["scheduler", "process", "shared-memory", "mutex", "throughput", "latency", "queue", "cache"],
    "graphs": ["pagerank", "citation", "community", "spectral", "minhash", "sketch", "shortest-path", "o(nlogn)"],
}
COMMON = ["we", "propose", "method", "results", "show", "novel", "approach", "evaluate",
          "dataset", "baseline", "improve", "analysis", "framework", "efficient", "scalable"]
_FIRST = ["Ana", "Bo", "Chen", "Dev", "Eli", "Farah", "Gita", "Hao", "Ivan", "Jin", "Kavya", "Lars"]
_LAST = ["Rao", "Silva", "Kim", "Novak", "Khan", "Ortiz", "Patel", "Li", "Berg", "Sato"]


def _words(rng, topic_words, n, topic_share=0.6):
    return [rng.choice(topic_words) if rng.random() < topic_share else rng.choice(COMMON)
            for _ in range(n)]


def make_paper(i: int, rng: random.Random) -> PaperRecord:
    topic = rng.choice(sorted(TOPICS))
    tw = TOPICS[topic]
    title = " ".join(_words(rng, tw, rng.randint(4, 8), 0.7)).capitalize()
    abstract = " ".join(_words(rng, tw, rng.randint(30, 80))) + "."
    year = rng.randint(2020, 2024)
    created = f"{year}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
    cats = sorted(rng.sample(CATEGORIES, rng.randint(1, 3)))
    authors = [f"{rng.choice(_FIRST)} {rng.choice(_LAST)}" for _ in range(rng.randint(1, 4))]
    return PaperRecord(
        arxiv_id=f"synth.{i:06d}", datestamp=created, title=title, abstract=abstract,
        authors=authors, categories=cats, created=created, updated="", doi="", synthetic=True,
    )


def generate_papers(n: int, seed: int = 0) -> list:
    rng = random.Random(seed)
    return [make_paper(i, rng) for i in range(n)]


def topic_of(paper: PaperRecord) -> str:
    """Best-guess topic (most topic words in the abstract). For plumbing tests."""
    words = paper.abstract.split()
    best = max(sorted(TOPICS), key=lambda t: sum(w.strip(".") in TOPICS[t] for w in words))
    return best


def messy_stream(papers, seed: int = 0, dup_frac=0.10, update_frac=0.10, stale_frac=0.05) -> list:
    """Papers + exact duplicates + newer versions + older (stale) versions,
    shuffled. Used to test idempotency: the correct final store holds exactly
    ONE version per arxiv_id, the newest.

    Returns the list of PaperRecord to feed in order.
    """
    rng = random.Random(seed)
    out = list(papers)
    n = len(papers)
    for p in rng.sample(papers, int(n * dup_frac)):
        out.append(p)                                    # exact duplicate (same key)
    for p in rng.sample(papers, int(n * update_frac)):
        y, m, d = p.datestamp.split("-")
        newer = PaperRecord(**{**p.to_dict(), "datestamp": f"{int(y) + 1}-{m}-{d}",
                               "title": p.title + " revised", "updated": f"{int(y) + 1}-{m}-{d}"})
        out.append(newer)
    for p in rng.sample(papers, int(n * stale_frac)):
        y, m, d = p.datestamp.split("-")
        older = PaperRecord(**{**p.to_dict(), "datestamp": f"{int(y) - 1}-{m}-{d}",
                               "title": p.title + " draft"})
        out.append(older)
    rng.shuffle(out)
    return out


def expected_final(stream) -> dict:
    """The ground truth: arxiv_id -> newest PaperRecord (what a correct store holds)."""
    best: dict = {}
    for p in stream:
        cur = best.get(p.arxiv_id)
        if cur is None or p.datestamp > cur.datestamp:
            best[p.arxiv_id] = p
    return best
