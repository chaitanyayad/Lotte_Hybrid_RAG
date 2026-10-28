"""
tests/test_evaluate.py: Unit tests for evaluation metrics (P@10, Recall@100, MAP, nDCG@10)
and 11-point interpolated precision.
"""

import math
import numpy as np
import pytest
from src.evaluate import compute_query_metrics, compute_interpolated_pr


def test_compute_metrics_perfect():
    gold = {1, 2}
    ranked = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    metrics = compute_query_metrics(ranked, gold)

    # 2 hits out of 10
    assert abs(metrics["P@10"] - 0.2) < 1e-6
    # 2 out of 2 gold recalled
    assert abs(metrics["Recall@100"] - 1.0) < 1e-6
    # Both at ranks 1 and 2 -> MAP = (1/1 + 2/2)/2 = 1.0
    assert abs(metrics["MAP"] - 1.0) < 1e-6
    # nDCG@10 = 1.0
    assert abs(metrics["nDCG@10"] - 1.0) < 1e-6


def test_compute_metrics_partial():
    gold = {10, 20}
    # Doc 10 at rank 1, Doc 20 at rank 3
    ranked = [10, 99, 20, 98, 97, 96, 95, 94, 93, 92]

    metrics = compute_query_metrics(ranked, gold)

    assert abs(metrics["P@10"] - 0.2) < 1e-6
    assert abs(metrics["Recall@100"] - 1.0) < 1e-6

    # AP: rank 1 (hit 1, prec=1/1=1.0), rank 3 (hit 2, prec=2/3=0.666667)
    # MAP = (1.0 + 2/3) / 2 = 5/6 = 0.833333
    assert abs(metrics["MAP"] - (5.0 / 6.0)) < 1e-5

    # DCG = 1/log2(2) + 1/log2(4) = 1.0 + 0.5 = 1.5
    # IDCG = 1/log2(2) + 1/log2(3) = 1.0 + 1/1.5849625 = 1.6309297
    # nDCG = 1.5 / 1.6309297 = 0.91972
    expected_ndcg = (1.0 + 0.5) / (1.0 + 1.0 / math.log2(3))
    assert abs(metrics["nDCG@10"] - expected_ndcg) < 1e-4


def test_compute_metrics_zero_hits():
    gold = {1, 2}
    ranked = [10, 20, 30]

    metrics = compute_query_metrics(ranked, gold)
    assert metrics["P@10"] == 0.0
    assert metrics["Recall@100"] == 0.0
    assert metrics["MAP"] == 0.0
    assert metrics["nDCG@10"] == 0.0


def test_compute_interpolated_pr_monotonic():
    gold = {1, 2, 3}
    ranked = [1, 10, 2, 20, 30, 3]
    recall_levels = np.linspace(0.0, 1.0, 11)

    interp_p = compute_interpolated_pr(ranked, gold, recall_levels)

    # Interpolated precision MUST be non-increasing with respect to recall
    for i in range(len(interp_p) - 1):
        assert interp_p[i] >= interp_p[i + 1] - 1e-7, (
            f"PR curve not monotonically non-increasing at index {i}: {interp_p[i]} vs {interp_p[i+1]}"
        )
