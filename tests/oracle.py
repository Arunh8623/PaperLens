"""Independent brute-force model of the index, used as a test oracle.

It shares NOTHING with the store except the tokenizer (which is the contract):
no segments, no postings, no varints, no diary. It just keeps lists of
documents and recomputes BM25 from scratch with plain loops.

It mirrors the documented semantics:
  * a put applies decide() rules (insert / duplicate / replace / stale),
  * a document deleted or replaced while still in the buffer disappears
    completely; one that was already flushed stays "physical" (counted in
    N, df, avgdl) but is never returned (Lucene behaviour until compaction),
  * flush() moves the buffer into the flushed list.
"""
import math
from collections import Counter

from paperlens.text.tokenizer import paper_text, tokenize

K1, B = 1.2, 0.75


class Model:
    def __init__(self):
        self.live = {}        # arxiv_id -> PaperRecord
        self.buffer = {}      # key -> tokens
        self.flushed = []     # [key, tokens, alive]

    def _kill(self, paper):
        key = paper.key
        if key in self.buffer:
            del self.buffer[key]
            return
        for entry in reversed(self.flushed):
            if entry[0] == key and entry[2]:
                entry[2] = False
                return

    def put(self, paper):
        cur = self.live.get(paper.arxiv_id)
        if cur is not None and paper.datestamp == cur.datestamp:
            return "duplicate"
        if cur is not None and paper.datestamp < cur.datestamp:
            return "stale"
        if cur is not None:
            self._kill(cur)
        self.buffer[paper.key] = tokenize(paper_text(paper.title, paper.abstract))
        self.live[paper.arxiv_id] = paper
        return "replace" if cur is not None else "insert"

    def delete(self, arxiv_id):
        cur = self.live.pop(arxiv_id, None)
        if cur is None:
            return False
        self._kill(cur)
        return True

    def flush(self):
        for key, toks in self.buffer.items():
            self.flushed.append([key, toks, True])
        self.buffer = {}

    def compact(self):
        """Full merge: documents killed after their flush are finally forgotten (N, df, avgdl shrink)."""
        self.flushed = [e for e in self.flushed if e[2]]

    def bm25(self, query):
        """{key: score} for every LIVE doc matching at least one query term."""
        terms = list(dict.fromkeys(tokenize(query)))
        physical = [(k, t, a) for k, t, a in self.flushed] + [(k, t, True) for k, t in self.buffer.items()]
        n = len(physical)
        if n == 0:
            return {}
        avgdl = sum(len(t) for _, t, _ in physical) / n
        counts = [(k, Counter(t), len(t), a) for k, t, a in physical]
        out = {}
        for term in terms:
            df = sum(1 for _, c, _, _ in counts if term in c)
            if df == 0:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for key, c, dl, alive in counts:
                if alive and term in c:
                    tf = c[term]
                    out[key] = out.get(key, 0.0) + idf * tf * (K1 + 1) / (tf + K1 * (1 - B + B * dl / avgdl))
        return out

    def live_keys(self):
        return {p.key for p in self.live.values()}
