"""
tests/test_sparse.py: Unit tests for Inverted Index, TF-IDF, and BM25 retrievers.
"""

import pytest
from rank_bm25 import BM25Okapi
from src.sparse import InvertedIndex, TFIDFRetriever, BM25Retriever


@pytest.fixture
def toy_corpus():
    return {
        101: ["python", "program", "code", "comput"],
        102: ["data", "retriev", "search", "engin", "python"],
        103: ["garden", "plant", "flower", "soil"],
        104: ["comput", "hardwar", "processor", "gpu", "retriev"]
    }


def test_inverted_index(toy_corpus):
    idx = InvertedIndex().fit(toy_corpus)
    assert idx.N == 4
    assert idx.doc_lens[101] == 4
    assert idx.doc_lens[102] == 5
    assert idx.avg_dl == (4 + 5 + 4 + 5) / 4.0
    # 'python' appears in 101 and 102
    assert set(idx.postings["python"].keys()) == {101, 102}
    assert idx.postings["python"][101] == 1


def test_tfidf_retriever(toy_corpus):
    idx = InvertedIndex().fit(toy_corpus)
    tfidf = TFIDFRetriever(idx)

    # Search for term present in corpus
    res = tfidf.search(["python"], top_k=2)
    assert len(res) == 2
    pids = [pid for pid, _ in res]
    assert 101 in pids
    assert 102 in pids
    # Scores must be sorted descending
    assert res[0][1] >= res[1][1]

    # Unknown term
    assert tfidf.search(["nonexistent"], top_k=5) == []
    # Empty query
    assert tfidf.search([], top_k=5) == []


def test_bm25_matches_reference(toy_corpus):
    idx = InvertedIndex().fit(toy_corpus)
    custom_bm25 = BM25Retriever(idx, k1=1.5, b=0.75)

    pids = list(toy_corpus.keys())
    tokenized_docs = [toy_corpus[pid] for pid in pids]
    ref_bm25 = BM25Okapi(tokenized_docs, k1=1.5, b=0.75)

    test_queries = [
        ["python", "retriev"],
        ["comput", "gpu"],
        ["flower", "garden"],
        ["python"]
    ]

    for q in test_queries:
        custom_scores = dict(custom_bm25.search(q, top_k=len(pids)))
        ref_scores_list = ref_bm25.get_scores(q)
        ref_scores = {pids[i]: ref_scores_list[i] for i in range(len(pids))}

        for pid in pids:
            c_score = custom_scores.get(pid, 0.0)
            r_score = ref_scores.get(pid, 0.0)
            assert abs(c_score - r_score) < 1e-4, f"Mismatch for query {q} on doc {pid}: {c_score} vs {r_score}"


def test_bm25_edge_cases(toy_corpus):
    idx = InvertedIndex().fit(toy_corpus)
    bm25 = BM25Retriever(idx, k1=1.5, b=0.75)

    assert bm25.search([], top_k=10) == []
    assert bm25.search(["alien_word_not_in_vocab"], top_k=10) == []
    res = bm25.search(["python"], top_k=1)
    assert len(res) == 1
