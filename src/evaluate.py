"""
src/evaluate.py: Evaluation suite computing P@10, Recall@100, MAP, and nDCG@10.
Generates an overlaid 11-point interpolated Precision-Recall curve and extracts
qualitative BM25 vs Dense failure/success case studies.
"""

import math
import json
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any
import numpy as np
import matplotlib.pyplot as plt


def compute_query_metrics(
    ranked_pids: List[int],
    gold_pids: Set[int]
) -> Dict[str, float]:
    """
    Computes P@10, Recall@100, MAP, and nDCG@10 for a single query.
    """
    if not gold_pids:
        return {"P@10": 0.0, "Recall@100": 0.0, "MAP": 0.0, "nDCG@10": 0.0}

    num_gold = len(gold_pids)

    # 1. Precision @ 10
    top_10 = ranked_pids[:10]
    hits_10 = sum(1 for pid in top_10 if pid in gold_pids)
    p_at_10 = hits_10 / 10.0

    # 2. Recall @ 100
    top_100 = ranked_pids[:100]
    hits_100 = sum(1 for pid in top_100 if pid in gold_pids)
    recall_at_100 = hits_100 / num_gold

    # 3. Average Precision (MAP)
    hits = 0
    prec_sum = 0.0
    for rank, pid in enumerate(ranked_pids, start=1):
        if pid in gold_pids:
            hits += 1
            prec_sum += hits / rank
    ap = prec_sum / num_gold

    # 4. nDCG @ 10
    dcg = 0.0
    for rank, pid in enumerate(top_10, start=1):
        if pid in gold_pids:
            dcg += 1.0 / math.log2(rank + 1)

    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(10, num_gold) + 1))
    ndcg_at_10 = (dcg / idcg) if idcg > 0 else 0.0

    return {
        "P@10": p_at_10,
        "Recall@100": recall_at_100,
        "MAP": ap,
        "nDCG@10": ndcg_at_10
    }


def evaluate_system(
    system_rankings: Dict[int, List[int]],
    qrels: Dict[int, Set[int]]
) -> Dict[str, Any]:
    """
    Evaluates rankings for all queries.
    Returns mean metrics and per-query metric records.
    """
    qids = [qid for qid in system_rankings.keys() if qid in qrels and len(qrels[qid]) > 0]
    per_query = {}
    metric_sums = {"P@10": 0.0, "Recall@100": 0.0, "MAP": 0.0, "nDCG@10": 0.0}

    for qid in qids:
        ranked = system_rankings[qid]
        gold = qrels[qid]
        q_metrics = compute_query_metrics(ranked, gold)
        per_query[qid] = q_metrics
        for m in metric_sums:
            metric_sums[m] += q_metrics[m]

    n = len(qids) if qids else 1
    mean_metrics = {m: metric_sums[m] / n for m in metric_sums}
    return {
        "mean": mean_metrics,
        "per_query": per_query
    }


def compute_interpolated_pr(
    ranked_pids: List[int],
    gold_pids: Set[int],
    recall_levels: np.ndarray
) -> np.ndarray:
    """
    Computes 11-point interpolated precision values for a single query.
    """
    if not gold_pids:
        return np.zeros_like(recall_levels)

    num_gold = len(gold_pids)
    hits = 0
    recalls = []
    precisions = []

    for rank, pid in enumerate(ranked_pids, start=1):
        if pid in gold_pids:
            hits += 1
        recalls.append(hits / num_gold)
        precisions.append(hits / rank)

    recalls = np.array(recalls)
    precisions = np.array(precisions)

    # Interpolated precision: P_interp(r) = max_{r' >= r} P(r')
    interp_precisions = []
    for r in recall_levels:
        eligible = precisions[recalls >= r]
        if len(eligible) > 0:
            interp_precisions.append(np.max(eligible))
        else:
            interp_precisions.append(0.0)

    return np.array(interp_precisions)


def plot_overlaid_pr_curves(
    all_system_rankings: Dict[str, Dict[int, List[int]]],
    qrels: Dict[int, Set[int]],
    output_path: Path
):
    """
    Plots an 11-point interpolated Precision-Recall curve overlaying all systems.
    """
    recall_levels = np.linspace(0.0, 1.0, 11)
    plt.figure(figsize=(8, 6), dpi=150)

    colors = {
        "TF-IDF": "#2b5c8f",
        "BM25": "#d95f02",
        "Dense (MiniLM)": "#7570b3",
        "Hybrid RRF": "#1b9e77"
    }

    for sys_name, rankings in all_system_rankings.items():
        qids = [qid for qid in rankings.keys() if qid in qrels and len(qrels[qid]) > 0]
        all_interp = []
        for qid in qids:
            interp = compute_interpolated_pr(rankings[qid], qrels[qid], recall_levels)
            all_interp.append(interp)

        mean_pr = np.mean(all_interp, axis=0) if all_interp else np.zeros_like(recall_levels)
        color = colors.get(sys_name, None)
        plt.plot(recall_levels, mean_pr, marker='o', label=sys_name, color=color, linewidth=2)

    plt.title("LoTTE Lifestyle Dev (Search) — Precision-Recall Curves", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision (Interpolated)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(fontsize=11, loc="upper right")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Precision-Recall curve saved to {output_path}")


def extract_failure_success_cases(
    bm25_eval: Dict[str, Any],
    dense_eval: Dict[str, Any],
    queries: Dict[int, str],
    passages: Dict[int, str],
    qrels: Dict[int, Set[int]],
    bm25_rankings: Dict[int, List[int]],
    dense_rankings: Dict[int, List[int]],
    output_path: Path
) -> List[Dict[str, Any]]:
    """
    Extracts 3-5 failure/success cases comparing BM25 vs Dense.
    - BM25 wins (keyword precision, vocabulary match)
    - Dense wins (semantic paraphrase, vocabulary mismatch)
    """
    bm25_pq = bm25_eval["per_query"]
    dense_pq = dense_eval["per_query"]

    # Difference in nDCG@10
    qids = list(bm25_pq.keys())

    # Sort by (BM25 - Dense)
    bm25_favored = sorted(qids, key=lambda q: bm25_pq[q]["nDCG@10"] - dense_pq[q]["nDCG@10"], reverse=True)
    dense_favored = sorted(qids, key=lambda q: dense_pq[q]["nDCG@10"] - bm25_pq[q]["nDCG@10"], reverse=True)

    cases = []

    # Pick top 2 BM25 wins
    for qid in bm25_favored[:2]:
        diff = bm25_pq[qid]["nDCG@10"] - dense_pq[qid]["nDCG@10"]
        if diff > 0.2:
            gold_pids = list(qrels[qid])
            cases.append({
                "type": "BM25 Win / Dense Loss",
                "qid": qid,
                "query": queries.get(qid, ""),
                "bm25_ndcg10": round(bm25_pq[qid]["nDCG@10"], 4),
                "dense_ndcg10": round(dense_pq[qid]["nDCG@10"], 4),
                "gold_pids": gold_pids,
                "bm25_top3": [
                    {"pid": p, "text": passages.get(p, "")[:200]}
                    for p in bm25_rankings[qid][:3]
                ],
                "dense_top3": [
                    {"pid": p, "text": passages.get(p, "")[:200]}
                    for p in dense_rankings[qid][:3]
                ],
                "analysis": "BM25 capitalizes on exact term matching, whereas dense representation drifted to generic semantic neighbors."
            })

    # Pick top 2 Dense wins
    for qid in dense_favored[:2]:
        diff = dense_pq[qid]["nDCG@10"] - bm25_pq[qid]["nDCG@10"]
        if diff > 0.2:
            gold_pids = list(qrels[qid])
            cases.append({
                "type": "Dense Win / BM25 Loss",
                "qid": qid,
                "query": queries.get(qid, ""),
                "bm25_ndcg10": round(bm25_pq[qid]["nDCG@10"], 4),
                "dense_ndcg10": round(dense_pq[qid]["nDCG@10"], 4),
                "gold_pids": gold_pids,
                "bm25_top3": [
                    {"pid": p, "text": passages.get(p, "")[:200]}
                    for p in bm25_rankings[qid][:3]
                ],
                "dense_top3": [
                    {"pid": p, "text": passages.get(p, "")[:200]}
                    for p in dense_rankings[qid][:3]
                ],
                "analysis": "Dense model matches semantic meaning despite vocabulary mismatch where lexical BM25 failed."
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2)
    print(f"Extracted {len(cases)} failure/success cases to {output_path}")
    return cases
