"""
tests/test_preprocess.py: Unit tests for text preprocessing & tokenization.
"""

import pytest
from src.preprocess import tokenize, preprocess_corpus, preprocess_corpus_mp


def test_tokenize_basic():
    text = "The quick brown fox jumps over the lazy dogs."
    tokens = tokenize(text)
    # 'the', 'over' are stop words; 'jumps' -> 'jump', 'dogs' -> 'dog'
    assert "the" not in tokens
    assert "over" not in tokens
    assert "quick" in tokens
    assert "brown" in tokens
    assert "fox" in tokens
    assert "jump" in tokens
    assert "lazi" in tokens
    assert "dog" in tokens


def test_tokenize_stemming():
    tokens1 = tokenize("running")
    tokens2 = tokenize("runs")
    assert tokens1 == ["run"]
    assert tokens2 == ["run"]


def test_tokenize_edge_cases():
    assert tokenize("") == []
    assert tokenize("   \n\t  ") == []
    assert tokenize("a an the in on at") == []  # all stopwords or length <= 1
    # Single letters filtered out
    assert tokenize("x y z") == []
    # Alphanumeric support
    assert "4060" in tokenize("RTX 4060 GPU")


def test_preprocess_corpus_equivalence():
    sample_corpus = {
        10: "Rabbits are very clean animals.",
        20: "Brake fluids must be replaced regularly.",
        30: "How to train your pet dog to sit?"
    }
    serial_res = preprocess_corpus(sample_corpus)
    mp_res = preprocess_corpus_mp(sample_corpus, num_workers=2)

    assert set(serial_res.keys()) == set(mp_res.keys())
    for pid in sample_corpus:
        assert serial_res[pid] == mp_res[pid], f"Mismatch for pid {pid}"
