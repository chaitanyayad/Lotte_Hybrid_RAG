"""
src/fusion.py: Reciprocal Rank Fusion (RRF) for hybrid retrieval.
Fuses BM25 and Dense retrieval rankings with k=60.
Formula: score(d) = Σ 1 / (k + rank_i(d)), where rank is 1-indexed.
"""

from collections import defaultdict
from typing import List, Tuple, Dict


def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[int, float]]],
    k: int = 60,
    top_k: int = 100
) -> List[Tuple[int, float]]:
    """
    Applies Reciprocal Rank Fusion over multiple ranked candidate lists.

    Args:
        ranked_lists: List of rankings from individual systems,
                      each list containing (pid, score) sorted descending by score.
        k: Smoothing constant (default 60 as per spec).
        top_k: Maximum number of combined documents to return.

    Returns:
        List of (pid, rrf_score) sorted descending by rrf_score.
    """
    rrf_scores: Dict[int, float] = defaultdict(float)

    for ranked_list in ranked_lists:
        for rank, (pid, _) in enumerate(ranked_list, start=1):
            rrf_scores[pid] += 1.0 / (k + rank)

    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:top_k]


class HybridRRFRetriever:
    """
    Wrapper for fusing BM25 and Dense retrievers.
    """
    def __init__(self, bm25_retriever, dense_retriever, k: int = 60):
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        self.k = k

    def search(
        self,
        query_text: str,
        query_tokens: List[str],
        candidate_pool_size: int = 100,
        top_k: int = 100
    ) -> List[Tuple[int, float]]:
        bm25_res = self.bm25_retriever.search(query_tokens, top_k=candidate_pool_size)
        dense_res = self.dense_retriever.search(query_text, top_k=candidate_pool_size)

        return reciprocal_rank_fusion([bm25_res, dense_res], k=self.k, top_k=top_k)
