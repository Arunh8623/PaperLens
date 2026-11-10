"""Embedders: turn a paper (or a query) into a unit vector.

What it does and why (plain language)
-------------------------------------
Semantic search needs "meaning vectors". We do NOT train a model; we use a
pre-trained one. Which one is chosen by a short spec string, so the choice is
recorded and can be compared in experiments:

  hash:64                 STAND-IN. Hashes words into 64 slots. Deterministic, needs
                          nothing installed, has NO real meaning (it behaves like a
                          fuzzy bag-of-words). Used by tests and plumbing demos only.
  st:<model name>         a sentence-transformers model, for example
                          st:sentence-transformers/all-MiniLM-L6-v2 (small, general)
                          st:allenai/specter2_base            (made for papers)
                          Needs `pip install sentence-transformers` and a download
                          of the model on first use. NOT tested in our sandbox
                          (no model download possible there); spec names are from memory.

Input text for a paper = title + ". " + abstract (the usual way SPECTER-style
models are fed). Queries are embedded as plain text.
"""
from __future__ import annotations

import numpy as np

from ..text.tokenizer import paper_text, term_freqs
from ..pipeline.embed import embed_terms


def paper_input_text(title: str, abstract: str) -> str:
    return f"{title}. {abstract}".strip()


class Embedder:
    name = ""
    dim = 0

    def embed_texts(self, texts: list) -> np.ndarray:
        raise NotImplementedError

    def embed_paper(self, paper, tf=None) -> np.ndarray:
        return self.embed_texts([paper_input_text(paper.title, paper.abstract)])[0]

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]


class HashEmbedder(Embedder):
    def __init__(self, dim: int = 64):
        if dim < 8:
            raise ValueError("hash embedder dim must be >= 8")
        self.dim, self.name = dim, f"hash:{dim}"

    def embed_texts(self, texts: list) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            tf, _ = term_freqs(t)
            out[i] = np.frombuffer(embed_terms(tf, self.dim).tobytes(), dtype=np.float32)
        return out

    def embed_paper(self, paper, tf=None) -> np.ndarray:
        if tf is None:
            tf, _ = term_freqs(paper_text(paper.title, paper.abstract))
        return np.frombuffer(embed_terms(tf, self.dim).tobytes(), dtype=np.float32).copy()


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str, batch_size: int = 32):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:                      # pragma: no cover - depends on the machine
            raise RuntimeError("sentence-transformers is not installed: "
                               "python -m pip install sentence-transformers") from e
        self._model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        get_dim = getattr(self._model, "get_embedding_dimension", None) or self._model.get_sentence_embedding_dimension
        self.dim = int(get_dim())          # newer sentence-transformers renamed this method
        self.name = f"st:{model_name}"

    def embed_texts(self, texts: list) -> np.ndarray:
        v = self._model.encode(texts, batch_size=self.batch_size, convert_to_numpy=True,
                               normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(v, dtype=np.float32)


def make_embedder(spec: str) -> Embedder:
    kind, _, arg = spec.partition(":")
    if kind == "hash":
        return HashEmbedder(int(arg) if arg else 64)
    if kind == "st" and arg:
        return SentenceTransformerEmbedder(arg)
    raise ValueError(f"unknown embedder spec {spec!r} (use hash:<dim> or st:<model name>)")
