"""
tests/test_rag.py: Unit tests for RAG generation and citation extraction.
"""

import pytest
from src.rag import (
    format_context,
    extract_cited_indices,
    _fallback_generate,
    generate_rag_answer
)


def test_format_context():
    passages = [
        {"pid": 100, "text": "First passage text."},
        {"pid": 200, "text": "Second passage text."}
    ]
    ctx = format_context(passages)
    assert "[1] (PID 100): First passage text." in ctx
    assert "[2] (PID 200): Second passage text." in ctx


def test_extract_cited_indices():
    ans = "Zebra loaches eat shrimp [1] and are active at night [3]. However [1] they are peaceful."
    indices = extract_cited_indices(ans)
    assert indices == [1, 3]


def test_fallback_generate_match():
    query = "feed english mastiff"
    passages = [
        {"pid": 1, "text": "I feed my english mastiff twice a day with high protein kibble."},
        {"pid": 2, "text": "Bicycle brake levers are essential for road safety."}
    ]
    ans = _fallback_generate(query, passages)
    assert "[1]" in ans
    assert "english mastiff" in ans.lower()
    assert ans != "not found"


def test_fallback_generate_not_found():
    query = "quantum mechanical tunneling in semiconductors"
    passages = [
        {"pid": 1, "text": "Rabbits like to eat fresh timothy hay and leafy vegetables."},
        {"pid": 2, "text": "Dogs should be walked at least once a day."}
    ]
    ans = _fallback_generate(query, passages)
    assert ans == "not found"


def test_generate_rag_answer_structure():
    query = "how to feed mastiff"
    passages = [
        {"pid": 10, "text": "Feed a growing mastiff three meals daily."}
    ]
    res = generate_rag_answer(query, passages)
    assert "answer" in res
    assert "citations" in res
    assert "cited_indices" in res
    assert len(res["citations"]) == 1
    assert res["citations"][0]["pid"] == 10
