"""
tests/test_fusion.py: Unit tests for Reciprocal Rank Fusion (RRF).
"""

import pytest
from src.fusion import reciprocal_rank_fusion


def test_rrf_scoring():
    # System 1: A (rank 1), B (rank 2), C (rank 3)
    list1 = [(1, 0.9), (2, 0.8), (3, 0.7)]
    # System 2: B (rank 1), A (rank 2), D (rank 3)
    list2 = [(2, 0.95), (1, 0.85), (4, 0.65)]

    # k = 60
    # Doc 1 (A): 1/(60+1) + 1/(60+2) = 1/61 + 1/62 = 0.0163934 + 0.0161290 = 0.0325224
    # Doc 2 (B): 1/(60+2) + 1/(60+1) = 1/62 + 1/61 = 0.0325224 (identical!)
    # Doc 3 (C): 1/(60+3) = 1/63 = 0.0158730
    # Doc 4 (D): 1/(60+3) = 1/63 = 0.0158730

    res = reciprocal_rank_fusion([list1, list2], k=60, top_k=4)
    res_dict = dict(res)

    expected_1 = 1.0 / 61.0 + 1.0 / 62.0
    expected_3 = 1.0 / 63.0

    assert abs(res_dict[1] - expected_1) < 1e-6
    assert abs(res_dict[2] - expected_1) < 1e-6
    assert abs(res_dict[3] - expected_3) < 1e-6
    assert abs(res_dict[4] - expected_3) < 1e-6


def test_rrf_top_k():
    list1 = [(i, 1.0 / i) for i in range(1, 20)]
    list2 = [(i + 5, 1.0 / i) for i in range(1, 20)]

    res = reciprocal_rank_fusion([list1, list2], k=60, top_k=5)
    assert len(res) == 5
    # Scores must be descending
    for i in range(len(res) - 1):
        assert res[i][1] >= res[i + 1][1]


def test_rrf_empty():
    assert reciprocal_rank_fusion([], k=60, top_k=10) == []
    assert reciprocal_rank_fusion([[]], k=60, top_k=10) == []
