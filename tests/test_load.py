"""
tests/test_load.py: Unit tests for dataset loading, schema verification, and sanity checks.
"""

import pytest
from src.load import load_lotte, compute_and_print_stats


def test_dataset_loading_and_sanity():
    passages, queries, qrels = load_lotte()

    # Verify counts
    assert len(passages) == 268893, f"Expected 268,893 passages, got {len(passages)}"
    assert len(queries) == 417, f"Expected 417 queries, got {len(queries)}"
    assert len(qrels) == 417, f"Expected 417 qrels entries, got {len(qrels)}"

    # Check types
    sample_pid = next(iter(passages))
    assert isinstance(sample_pid, int)
    assert isinstance(passages[sample_pid], str)
    assert len(passages[sample_pid]) > 0

    sample_qid = next(iter(queries))
    assert isinstance(sample_qid, int)
    assert isinstance(queries[sample_qid], str)

    # Sanity check: every single qrels pid MUST exist in passages
    missing_pids = set()
    for qid, pids in qrels.items():
        assert len(pids) > 0, f"Query {qid} has empty qrels!"
        for pid in pids:
            if pid not in passages:
                missing_pids.add(pid)

    assert len(missing_pids) == 0, f"Sanity check failed! Found missing PIDs: {missing_pids}"


def test_compute_stats():
    passages, queries, qrels = load_lotte()
    stats = compute_and_print_stats(passages, queries, qrels)

    assert stats["corpus_size"] == 268893
    assert stats["query_count"] == 417
    assert stats["avg_passage_length_words"] > 100.0
    assert stats["avg_rel_per_query"] > 1.0
    assert stats["sanity_check_passed"] is True
