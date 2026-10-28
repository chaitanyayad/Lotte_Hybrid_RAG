"""
tests/test_dense.py: Unit tests for Dense retrieval, FAISS indexing, and embeddings cache.
"""

from pathlib import Path
import numpy as np
import pytest
import faiss

from src.dense import DenseRetriever, get_device


def test_device():
    device = get_device()
    assert device in ("cuda", "cpu")


def test_cached_embeddings_file():
    cache_path = Path(__file__).resolve().parent.parent / "results" / "dense_embeddings.npz"
    assert cache_path.exists(), "Embeddings cache file does not exist!"

    data = np.load(cache_path)
    assert "embeddings" in data
    assert "doc_ids" in data

    embeddings = data["embeddings"]
    doc_ids = data["doc_ids"]

    assert len(doc_ids) == 268893
    assert embeddings.shape == (268893, 384)
    assert embeddings.dtype == np.float32

    # Check L2 normalization on sample vectors
    sample = embeddings[:100]
    norms = np.linalg.norm(sample, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-3, err_msg="Embeddings must be L2 normalized!")


def test_dense_retriever_toy():
    # 3 mock unit vectors in 4D space
    v1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    v3 = np.array([0.7071, 0.7071, 0.0, 0.0], dtype=np.float32)

    embeddings = np.vstack([v1, v2, v3])
    doc_ids = np.array([10, 20, 30], dtype=np.int64)

    retriever = DenseRetriever(embeddings, doc_ids)
    assert retriever.index.ntotal == 3

    # Direct FAISS search with query parallel to v1
    q_emb = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    scores, indices = retriever.index.search(q_emb, 3)

    assert indices[0][0] == 0  # doc 10 is exact match
    assert abs(scores[0][0] - 1.0) < 1e-4

    assert indices[0][1] == 2  # doc 30 is 45-degree angle (cos ~ 0.7071)
    assert abs(scores[0][1] - 0.7071) < 1e-3

    assert indices[0][2] == 1  # doc 20 is orthogonal (cos = 0)
    assert abs(scores[0][2] - 0.0) < 1e-4
