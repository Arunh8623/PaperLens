"""Science-aware TOY tokenizer (Phase 0 version; E3 keyword side improves it).

What it does and why (plain language)
-------------------------------------
Search only works if the text of a paper and the text of a query are cut into
words the SAME way. Generic tokenizers mangle computer-science text: they turn
"C++" into "c", "k-means" into two unrelated words, and "O(n log n)" into
junk. This one keeps those intact.

Rules, in order:
 1. Unicode NFKC (full-width letters -> normal), casefold (lowercase),
    then strip accents (café -> cafe).
 2. Possessive 's is dropped ("scikit-learn's" -> "scikit-learn");
    other apostrophes are removed ("don't" -> "dont").
 3. Big-O expressions become one token with spaces removed: "O(n log n)" ->
    "o(nlogn)".
 4. A word is letters/digits, optionally joined by - . / _ inside
    ("k-means", "bert-base", "3.5", "tcp/ip"), optionally ending in ++ or #
    ("c++", "c#").
 5. A joined word ALSO emits its parts (len >= 2): "k-means" -> "k-means",
    "means". So a query for "means" still finds "k-means", and a query for
    "k-means" matches the exact compound.
 6. A small English stopword list is removed.

Known limits (toy on purpose): no stemming; no CJK word segmentation (a run of
CJK characters is one token); "c++11" becomes "c++" + "11".

The function `tokenize` is the single source of truth: index time and query
time must both call it.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

STOPWORDS = frozenset("""
a an and are as at be but by for from has have in is it its of on or that the
their this to was were which with we our us can may also such these those
than then there been being into over under using use used
""".split())

_BIGO = r"[oO]\((?:[^()]|\([^()]*\))*\)"
_WORD = r"[^\W_]+(?:[-./_][^\W_]+)*(?:\+\+|#)?"
_TOKEN_RE = re.compile(f"(?P<bigo>{_BIGO})|(?P<word>{_WORD})")
_SPLIT_RE = re.compile(r"[-./_]")
_POSSESSIVE_RE = re.compile(r"['’]s\b")
MAX_TOKEN_LEN = 64


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = _POSSESSIVE_RE.sub("", text)
    return text.replace("'", "").replace("’", "")


def tokenize(text: str) -> list:
    """Text -> list of tokens (with repeats, in order)."""
    out = []
    for m in _TOKEN_RE.finditer(_normalise(text)):
        if m.group("bigo"):
            tok = re.sub(r"\s+", "", m.group("bigo"))
            out.append(tok[:MAX_TOKEN_LEN])
            continue
        word = m.group("word")
        if word in STOPWORDS:
            continue
        out.append(word[:MAX_TOKEN_LEN])
        if _SPLIT_RE.search(word):
            for part in _SPLIT_RE.split(word):
                if len(part) >= 2 and part not in STOPWORDS:
                    out.append(part[:MAX_TOKEN_LEN])
    return out


def term_freqs(text: str) -> tuple:
    """Text -> ({term: count}, doc_len). doc_len = number of tokens emitted.

    Readers (E1) call this so the writer never has to re-tokenize.
    """
    toks = tokenize(text)
    return dict(Counter(toks)), len(toks)


def paper_text(title: str, abstract: str) -> str:
    """The searchable text of a paper. Title is repeated once so title words
    count a bit more (a documented, simple boost; Phase 1 may revisit)."""
    return f"{title}\n{title}\n{abstract}"
