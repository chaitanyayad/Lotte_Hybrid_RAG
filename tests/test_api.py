"""
tests/test_api.py: Integration tests for FastAPI backend service endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from api import app, STATE


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["ready"] is True
    assert data["corpus_size"] == 268893


@pytest.mark.parametrize("ranker", ["tfidf", "bm25", "dense", "rrf"])
def test_search_endpoints(client, ranker):
    query = "feed english mastiff"
    response = client.get(f"/search?q={query}&ranker={ranker}&k=5")
    assert response.status_code == 200
    results = response.json()

    assert len(results) == 5
    for item in results:
        assert "rank" in item
        assert "pid" in item
        assert "score" in item
        assert "text" in item
        assert isinstance(item["rank"], int)
        assert isinstance(item["pid"], int)
        assert isinstance(item["score"], float)
        assert len(item["text"]) > 0

    # Ensure rank ordering 1 to 5
    ranks = [item["rank"] for item in results]
    assert ranks == [1, 2, 3, 4, 5]


def test_search_invalid_ranker(client):
    response = client.get("/search?q=test&ranker=unknown_ranker")
    assert response.status_code == 400


def test_search_empty_query(client):
    response = client.get("/search?q=   &ranker=bm25")
    assert response.status_code == 400


def test_rag_endpoint(client):
    payload = {
        "q": "how much to feed mastiff",
        "ranker": "rrf",
        "k": 3
    }
    response = client.post("/rag", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "citations" in data
    assert len(data["citations"]) == 3
    assert data["ranker"] == "rrf"
    for cit in data["citations"]:
        assert "n" in cit
        assert "pid" in cit
        assert "text" in cit
