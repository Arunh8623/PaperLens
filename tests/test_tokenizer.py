import pytest

from paperlens.text.tokenizer import tokenize, term_freqs, paper_text


@pytest.mark.parametrize("text,expected", [
    ("k-means clustering", ["k-means", "means", "clustering"]),
    ("C++ and C# programming", ["c++", "c#", "programming"]),
    ("runs in O(n log n) time", ["runs", "o(nlogn)", "time"]),
    ("sorting is O(n^2)", ["sorting", "o(n^2)"]),
    ("BERT-base vs. RoBERTa", ["bert-base", "bert", "base", "vs", "roberta"]),
    ("Café résumé naïve", ["cafe", "resume", "naive"]),
    ("ＢＥＲＴ", ["bert"]),
    ("The α-divergence of 3.5 GPUs", ["α-divergence", "divergence", "3.5", "gpus"]),
    ("don't use scikit-learn's API", ["dont", "scikit-learn", "scikit", "learn", "api"]),
    ("TCP/IP stack", ["tcp/ip", "tcp", "ip", "stack"]),
    ("", []),
    ("   \n\t ", []),
    ("the of and", []),
    ("end. Next", ["end", "next"]),
])
def test_table(text, expected):
    assert tokenize(text) == expected


def test_deterministic_and_case_insensitive():
    assert tokenize("Hello, World!") == tokenize("hello world")


def test_term_freqs_counts_and_len():
    tf, n = term_freqs("graph graph theory")
    assert tf == {"graph": 2, "theory": 1} and n == 3
    tf, n = term_freqs("k-means")
    assert tf == {"k-means": 1, "means": 1} and n == 2


def test_long_token_is_capped():
    assert max(len(t) for t in tokenize("a" * 500)) == 64


def test_paper_text_repeats_title():
    toks = tokenize(paper_text("Graph Cuts", "We study graphs"))
    assert toks.count("cuts") == 2 and toks.count("study") == 1


def test_query_matches_document_for_compound_terms():
    doc = set(tokenize("We improve k-means initialisation"))
    assert "k-means" in doc and "means" in doc          # both query styles hit
    assert set(tokenize("k-means")) <= doc
