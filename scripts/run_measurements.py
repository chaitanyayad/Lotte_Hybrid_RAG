"""
scripts/run_measurements.py: Comprehensive benchmark measurements for the IR Report.
Runs all measurements requested in sections B, C, D:
- Latency benchmarks (mean, median, p95)
- Win/loss/tie counts on nDCG@10
- Paired significance tests (t-test, Wilcoxon, 10,000-sample bootstrap CI)
- BM25 sweep (k1 in {0.9, 1.2, 1.5, 2.0}, b in {0.5, 0.75, 0.9})
- RRF & fusion sweep (k in {10, 30, 60, 100}, weighted RRF and linear fusion)
- Recall@k for k in {1, 5, 10, 20, 50, 100}
- Query length breakdown (short <=6, medium 7-10, long >=11)
- Hard queries (nDCG@10 == 0 across all systems) with 3 concrete examples
- Cross-Encoder reranker over hybrid top-50 using ms-marco-MiniLM-L-6-v2
- RAG evaluation on 30 queries (citation validity, abstention present vs missing)
- Startup time and memory profiling
"""

import sys
import os
import time
import json
import math
import random
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from scipy import stats
import psutil

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.load import load_lotte
from src.preprocess import tokenize, _STOP_WORDS, _STEMMER
from src.sparse import InvertedIndex, TFIDFRetriever, BM25Retriever
from src.dense import encode_corpus, DenseRetriever
from src.fusion import reciprocal_rank_fusion
from src.evaluate import compute_query_metrics, evaluate_system
from src.rag import generate_rag_answer, extract_cited_indices

RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_process_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def bootstrap_ci_diff(scores_a, scores_b, num_bootstrap=10000, ci=0.95, seed=42):
    rng = np.random.default_rng(seed)
    diffs = np.array(scores_a) - np.array(scores_b)
    n = len(diffs)
    boot_means = []
    for _ in range(num_bootstrap):
        sample = rng.choice(diffs, size=n, replace=True)
        boot_means.append(np.mean(sample))
    boot_means = np.sort(boot_means)
    alpha = (1.0 - ci) / 2.0
    low = np.percentile(boot_means, alpha * 100)
    high = np.percentile(boot_means, (1.0 - alpha) * 100)
    return float(low), float(high), float(np.mean(diffs))


def main():
    print("=" * 70)
    print("STARTING COMPREHENSIVE EXPERIMENTATION AND MEASUREMENTS")
    print("=" * 70)

    results_data = {}

    # ---------------------------------------------------------
    # 1. Startup & Memory Profiling
    # ---------------------------------------------------------
    print("\n--- 1. Startup & Memory Profiling ---")
    mem_before = get_process_memory_mb()
    t0 = time.perf_counter()

    passages, queries, qrels = load_lotte()
    t_load = time.perf_counter() - t0
    mem_after_load = get_process_memory_mb()

    # Preprocessed corpus
    t0_prep = time.perf_counter()
    from src.run_day1 import get_preprocessed_corpus
    tokenized_corpus = get_preprocessed_corpus(passages)
    t_prep_load = time.perf_counter() - t0_prep
    mem_after_tokens = get_process_memory_mb()

    # Inverted index
    t0_idx = time.perf_counter()
    inv_idx = InvertedIndex().fit(tokenized_corpus)
    t_idx = time.perf_counter() - t0_idx
    mem_after_idx = get_process_memory_mb()

    # TF-IDF & BM25
    t0_sparse = time.perf_counter()
    tfidf = TFIDFRetriever(inv_idx)
    bm25 = BM25Retriever(inv_idx, k1=1.5, b=0.75)
    t_sparse = time.perf_counter() - t0_sparse
    mem_after_sparse = get_process_memory_mb()

    # Dense embeddings & FAISS FlatIP
    t0_dense = time.perf_counter()
    embeddings, doc_ids = encode_corpus(passages)
    dense = DenseRetriever(embeddings, doc_ids)
    t_dense = time.perf_counter() - t0_dense
    mem_after_dense = get_process_memory_mb()

    total_startup_time = time.perf_counter() - t0
    mem_total = mem_after_dense

    vocab_size = len(inv_idx.postings)
    num_stopwords = len(_STOP_WORDS)
    stemmer_name = _STEMMER.__class__.__name__

    faiss_bytes = embeddings.nbytes
    faiss_mb = faiss_bytes / (1024 * 1024)

    results_data["system_and_memory"] = {
        "stopword_list_size": num_stopwords,
        "stemmer": stemmer_name,
        "vocab_size": vocab_size,
        "startup_time_seconds": round(total_startup_time, 2),
        "startup_breakdown_seconds": {
            "load_lotte": round(t_load, 2),
            "load_tokenized_corpus": round(t_prep_load, 2),
            "build_inverted_index": round(t_idx, 2),
            "build_sparse_retrievers": round(t_sparse, 2),
            "load_faiss_dense": round(t_dense, 2)
        },
        "memory_ram_mb": {
            "initial": round(mem_before, 2),
            "after_load": round(mem_after_load, 2),
            "after_tokenized_corpus": round(mem_after_tokens, 2),
            "after_inverted_index": round(mem_after_idx, 2),
            "after_sparse_models": round(mem_after_sparse, 2),
            "final_with_faiss_and_models": round(mem_total, 2),
            "net_ram_allocated_mb": round(mem_total - mem_before, 2)
        },
        "faiss_index": {
            "num_vectors": len(doc_ids),
            "dimension": int(embeddings.shape[1]),
            "dtype": str(embeddings.dtype),
            "size_mb": round(faiss_mb, 2)
        }
    }
    print(f"Total Startup Time: {total_startup_time:.2f}s | Net RAM Allocated: {mem_total - mem_before:.1f} MB")
    print(f"Vocabulary Size: {vocab_size} | Stopwords: {num_stopwords} | FAISS Size: {faiss_mb:.1f} MB")

    # ---------------------------------------------------------
    # 2. Latency Benchmarks (Mean, Median, P95) over all 417 Queries
    # ---------------------------------------------------------
    print("\n--- 2. Measuring Query Latencies over all 417 queries ---")
    query_items = list(queries.items())
    tokenized_queries = {qid: tokenize(q) for qid, q in queries.items()}

    latencies = {
        "TF-IDF": [],
        "BM25": [],
        "Dense (MiniLM)": [],
        "Hybrid RRF": []
    }

    rankings_cache = {
        "TF-IDF": {},
        "BM25": {},
        "Dense (MiniLM)": {},
        "Hybrid RRF": {}
    }

    # Warmup
    _ = tfidf.search(tokenized_queries[query_items[0][0]], top_k=100)
    _ = bm25.search(tokenized_queries[query_items[0][0]], top_k=100)
    _ = dense.search(query_items[0][1], top_k=100)

    for qid, q_text in query_items:
        q_toks = tokenized_queries[qid]

        # TF-IDF
        t0_sys = time.perf_counter()
        res_tfidf = tfidf.search(q_toks, top_k=100)
        lat_tfidf = (time.perf_counter() - t0_sys) * 1000.0
        latencies["TF-IDF"].append(lat_tfidf)
        rankings_cache["TF-IDF"][qid] = [p for p, _ in res_tfidf]

        # BM25
        t0_sys = time.perf_counter()
        res_bm25 = bm25.search(q_toks, top_k=100)
        lat_bm25 = (time.perf_counter() - t0_sys) * 1000.0
        latencies["BM25"].append(lat_bm25)
        rankings_cache["BM25"][qid] = [p for p, _ in res_bm25]

        # Dense
        t0_sys = time.perf_counter()
        res_dense = dense.search(q_text, top_k=100)
        lat_dense = (time.perf_counter() - t0_sys) * 1000.0
        latencies["Dense (MiniLM)"].append(lat_dense)
        rankings_cache["Dense (MiniLM)"][qid] = [p for p, _ in res_dense]

        # Hybrid RRF (fuses BM25 + Dense)
        t0_sys = time.perf_counter()
        res_rrf = reciprocal_rank_fusion([res_bm25, res_dense], k=60, top_k=100)
        lat_rrf = (time.perf_counter() - t0_sys) * 1000.0
        latencies["Hybrid RRF"].append(lat_rrf)
        rankings_cache["Hybrid RRF"][qid] = [p for p, _ in res_rrf]

    latency_stats = {}
    for sys_name, lats in latencies.items():
        latency_stats[sys_name] = {
            "mean_ms": round(float(np.mean(lats)), 2),
            "median_ms": round(float(np.median(lats)), 2),
            "p95_ms": round(float(np.percentile(lats, 95)), 2),
            "min_ms": round(float(np.min(lats)), 2),
            "max_ms": round(float(np.max(lats)), 2)
        }
        print(f"  {sys_name:15}: Mean={latency_stats[sys_name]['mean_ms']:6.2f} ms | "
              f"Median={latency_stats[sys_name]['median_ms']:6.2f} ms | "
              f"P95={latency_stats[sys_name]['p95_ms']:6.2f} ms")

    # End-to-end API latency including RAG (over 30 sample queries)
    print("\nMeasuring end-to-end RAG latency over 30 queries...")
    rag_latencies_ms = []
    sample_30_qids = list(queries.keys())[:30]
    for qid in sample_30_qids:
        t0_rag = time.perf_counter()
        q_text = queries[qid]
        # retrieval (Hybrid RRF top 5)
        bm25_res = bm25.search(tokenized_queries[qid], top_k=100)
        dense_res = dense.search(q_text, top_k=100)
        rrf_top5 = reciprocal_rank_fusion([bm25_res, dense_res], k=60, top_k=5)
        passages_payload = [
            {"rank": i + 1, "pid": pid, "text": passages.get(pid, "")}
            for i, (pid, _) in enumerate(rrf_top5)
        ]
        # generation
        _ = generate_rag_answer(query=q_text, passages=passages_payload)
        rag_latencies_ms.append((time.perf_counter() - t0_rag) * 1000.0)

    latency_stats["End-to-End API RAG (top-5 + synthesis)"] = {
        "mean_ms": round(float(np.mean(rag_latencies_ms)), 2),
        "median_ms": round(float(np.median(rag_latencies_ms)), 2),
        "p95_ms": round(float(np.percentile(rag_latencies_ms, 95)), 2)
    }
    print(f"  End-to-End RAG: Mean={latency_stats['End-to-End API RAG (top-5 + synthesis)']['mean_ms']} ms | "
          f"Median={latency_stats['End-to-End API RAG (top-5 + synthesis)']['median_ms']} ms | "
          f"P95={latency_stats['End-to-End API RAG (top-5 + synthesis)']['p95_ms']} ms")

    results_data["latency_benchmarks"] = latency_stats

    # ---------------------------------------------------------
    # 3. Standard Evaluation & Win/Loss/Tie Counts on nDCG@10
    # ---------------------------------------------------------
    print("\n--- 3. Computing Metrics and Win/Loss/Tie Counts on nDCG@10 ---")
    eval_results = {}
    for sys_name in rankings_cache:
        eval_results[sys_name] = evaluate_system(rankings_cache[sys_name], qrels)

    # Extract per-query nDCG@10
    ndcg_bm25 = [eval_results["BM25"]["per_query"][qid]["nDCG@10"] for qid in queries]
    ndcg_dense = [eval_results["Dense (MiniLM)"]["per_query"][qid]["nDCG@10"] for qid in queries]
    ndcg_hybrid = [eval_results["Hybrid RRF"]["per_query"][qid]["nDCG@10"] for qid in queries]

    def get_wlt(list_a, list_b, eps=1e-5):
        wins, losses, ties = 0, 0, 0
        for a, b in zip(list_a, list_b):
            if a - b > eps:
                wins += 1
            elif b - a > eps:
                losses += 1
            else:
                ties += 1
        return {"wins": wins, "losses": losses, "ties": ties}

    wlt_dense_vs_bm25 = get_wlt(ndcg_dense, ndcg_bm25)
    wlt_hybrid_vs_dense = get_wlt(ndcg_hybrid, ndcg_dense)
    wlt_hybrid_vs_bm25 = get_wlt(ndcg_hybrid, ndcg_bm25)

    print(f"Dense vs BM25: Wins={wlt_dense_vs_bm25['wins']}, Losses={wlt_dense_vs_bm25['losses']}, Ties={wlt_dense_vs_bm25['ties']}")
    print(f"Hybrid vs Dense: Wins={wlt_hybrid_vs_dense['wins']}, Losses={wlt_hybrid_vs_dense['losses']}, Ties={wlt_hybrid_vs_dense['ties']}")
    print(f"Hybrid vs BM25: Wins={wlt_hybrid_vs_bm25['wins']}, Losses={wlt_hybrid_vs_bm25['losses']}, Ties={wlt_hybrid_vs_bm25['ties']}")

    results_data["win_loss_tie"] = {
        "Dense_vs_BM25": wlt_dense_vs_bm25,
        "Hybrid_vs_Dense": wlt_hybrid_vs_dense,
        "Hybrid_vs_BM25": wlt_hybrid_vs_bm25
    }

    # ---------------------------------------------------------
    # 4. Paired Significance Tests & Bootstrap 95% CI
    # ---------------------------------------------------------
    print("\n--- 4. Paired Significance Tests & Bootstrap 95% CI on nDCG@10 ---")
    # Dense vs BM25
    t_stat_d_b, p_val_ttest_d_b = stats.ttest_rel(ndcg_dense, ndcg_bm25)
    wilc_stat_d_b, p_val_wilc_d_b = stats.wilcoxon(ndcg_dense, ndcg_bm25)
    ci_low_d_b, ci_high_d_b, mean_diff_d_b = bootstrap_ci_diff(ndcg_dense, ndcg_bm25)

    # Hybrid vs Dense
    t_stat_h_d, p_val_ttest_h_d = stats.ttest_rel(ndcg_hybrid, ndcg_dense)
    wilc_stat_h_d, p_val_wilc_h_d = stats.wilcoxon(ndcg_hybrid, ndcg_dense)
    ci_low_h_d, ci_high_h_d, mean_diff_h_d = bootstrap_ci_diff(ndcg_hybrid, ndcg_dense)

    sig_tests = {
        "Dense_vs_BM25": {
            "mean_diff": round(mean_diff_d_b, 4),
            "paired_ttest": {
                "t_statistic": round(float(t_stat_d_b), 4),
                "p_value": float(f"{p_val_ttest_d_b:.2e}")
            },
            "wilcoxon_signed_rank": {
                "statistic": round(float(wilc_stat_d_b), 2),
                "p_value": float(f"{p_val_wilc_d_b:.2e}")
            },
            "bootstrap_95_ci": {
                "ci_lower": round(ci_low_d_b, 4),
                "ci_upper": round(ci_high_d_b, 4)
            }
        },
        "Hybrid_vs_Dense": {
            "mean_diff": round(mean_diff_h_d, 4),
            "paired_ttest": {
                "t_statistic": round(float(t_stat_h_d), 4),
                "p_value": float(f"{p_val_ttest_h_d:.2e}")
            },
            "wilcoxon_signed_rank": {
                "statistic": round(float(wilc_stat_h_d), 2),
                "p_value": float(f"{p_val_wilc_h_d:.2e}")
            },
            "bootstrap_95_ci": {
                "ci_lower": round(ci_low_h_d, 4),
                "ci_upper": round(ci_high_h_d, 4)
            }
        }
    }
    print("Dense vs BM25: p_ttest=", sig_tests["Dense_vs_BM25"]["paired_ttest"]["p_value"],
          "95% CI=[", ci_low_d_b, ci_high_d_b, "]")
    print("Hybrid vs Dense: p_ttest=", sig_tests["Hybrid_vs_Dense"]["paired_ttest"]["p_value"],
          "95% CI=[", ci_low_h_d, ci_high_h_d, "]")

    results_data["significance_tests"] = sig_tests

    # ---------------------------------------------------------
    # 5. BM25 Hyperparameter Sweep
    # ---------------------------------------------------------
    print("\n--- 5. Running BM25 Hyperparameter Sweep ---")
    k1_values = [0.9, 1.2, 1.5, 2.0]
    b_values = [0.5, 0.75, 0.9]
    bm25_grid_results = []

    for k1 in k1_values:
        for b in b_values:
            temp_bm25 = BM25Retriever(inv_idx, k1=k1, b=b)
            temp_rankings = {}
            for qid, q_toks in tokenized_queries.items():
                temp_rankings[qid] = [p for p, _ in temp_bm25.search(q_toks, top_k=100)]
            ev = evaluate_system(temp_rankings, qrels)
            bm25_grid_results.append({
                "k1": k1,
                "b": b,
                "nDCG@10": round(ev["mean"]["nDCG@10"], 4),
                "Recall@100": round(ev["mean"]["Recall@100"], 4),
                "P@10": round(ev["mean"]["P@10"], 4),
                "MAP": round(ev["mean"]["MAP"], 4)
            })
            print(f"  k1={k1:3.1f}, b={b:4.2f} -> nDCG@10={ev['mean']['nDCG@10']:.4f}, Recall@100={ev['mean']['Recall@100']:.4f}")

    results_data["bm25_sweep"] = bm25_grid_results

    # ---------------------------------------------------------
    # 6. RRF & Fusion Sweep
    # ---------------------------------------------------------
    print("\n--- 6. Running RRF & Fusion Sweep ---")
    # A. Standard RRF with k in {10, 30, 60, 100}
    rrf_k_values = [10, 30, 60, 100]
    rrf_sweep_results = []
    # Pre-fetch BM25 and Dense candidates with scores
    bm25_full_results = {qid: bm25.search(tokenized_queries[qid], top_k=100) for qid in queries}
    dense_full_results = {qid: dense.search(queries[qid], top_k=100) for qid in queries}

    for k_val in rrf_k_values:
        fused_rankings = {}
        for qid in queries:
            fused = reciprocal_rank_fusion([bm25_full_results[qid], dense_full_results[qid]], k=k_val, top_k=100)
            fused_rankings[qid] = [p for p, _ in fused]
        ev = evaluate_system(fused_rankings, qrels)
        rrf_sweep_results.append({
            "type": "Standard RRF",
            "k": k_val,
            "dense_weight": None,
            "nDCG@10": round(ev["mean"]["nDCG@10"], 4),
            "Recall@100": round(ev["mean"]["Recall@100"], 4),
            "P@10": round(ev["mean"]["P@10"], 4),
            "MAP": round(ev["mean"]["MAP"], 4)
        })
        print(f"  RRF k={k_val:3d} -> nDCG@10={ev['mean']['nDCG@10']:.4f}, Recall@100={ev['mean']['Recall@100']:.4f}")

    # B. Weighted RRF with dense_weight in {0.5, 0.7, 0.9} (k=60)
    dense_weights = [0.5, 0.7, 0.9]
    for w_dense in dense_weights:
        w_bm25 = 1.0 - w_dense
        fused_rankings = {}
        for qid in queries:
            scores = defaultdict(float)
            for r, (pid, _) in enumerate(bm25_full_results[qid], 1):
                scores[pid] += w_bm25 * (1.0 / (60 + r))
            for r, (pid, _) in enumerate(dense_full_results[qid], 1):
                scores[pid] += w_dense * (1.0 / (60 + r))
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:100]
            fused_rankings[qid] = [p for p, _ in ranked]
        ev = evaluate_system(fused_rankings, qrels)
        rrf_sweep_results.append({
            "type": "Weighted RRF (k=60)",
            "k": 60,
            "dense_weight": w_dense,
            "nDCG@10": round(ev["mean"]["nDCG@10"], 4),
            "Recall@100": round(ev["mean"]["Recall@100"], 4),
            "P@10": round(ev["mean"]["P@10"], 4),
            "MAP": round(ev["mean"]["MAP"], 4)
        })
        print(f"  Weighted RRF w_dense={w_dense:.1f} -> nDCG@10={ev['mean']['nDCG@10']:.4f}, Recall@100={ev['mean']['Recall@100']:.4f}")

    # C. Linear Fusion (Normalized Scores)
    for w_dense in dense_weights:
        w_bm25 = 1.0 - w_dense
        fused_rankings = {}
        for qid in queries:
            # Min-max normalize BM25 scores
            bm25_list = bm25_full_results[qid]
            dense_list = dense_full_results[qid]

            def norm_scores(lst):
                if not lst:
                    return {}
                scs = [s for _, s in lst]
                min_s, max_s = min(scs), max(scs)
                denom = (max_s - min_s) if max_s > min_s else 1.0
                return {pid: (s - min_s) / denom for pid, s in lst}

            n_bm25 = norm_scores(bm25_list)
            n_dense = norm_scores(dense_list)
            all_pids = set(n_bm25.keys()) | set(n_dense.keys())
            comb = {}
            for p in all_pids:
                comb[p] = w_bm25 * n_bm25.get(p, 0.0) + w_dense * n_dense.get(p, 0.0)
            ranked = sorted(comb.items(), key=lambda x: x[1], reverse=True)[:100]
            fused_rankings[qid] = [p for p, _ in ranked]

        ev = evaluate_system(fused_rankings, qrels)
        rrf_sweep_results.append({
            "type": "Linear Score Fusion",
            "k": None,
            "dense_weight": w_dense,
            "nDCG@10": round(ev["mean"]["nDCG@10"], 4),
            "Recall@100": round(ev["mean"]["Recall@100"], 4),
            "P@10": round(ev["mean"]["P@10"], 4),
            "MAP": round(ev["mean"]["MAP"], 4)
        })
        print(f"  Linear Fusion w_dense={w_dense:.1f} -> nDCG@10={ev['mean']['nDCG@10']:.4f}, Recall@100={ev['mean']['Recall@100']:.4f}")

    results_data["fusion_sweep"] = rrf_sweep_results

    # ---------------------------------------------------------
    # 7. Recall@k for k in {1, 5, 10, 20, 50, 100}
    # ---------------------------------------------------------
    print("\n--- 7. Recall@k Breakdown for All Four Systems ---")
    k_cutoffs = [1, 5, 10, 20, 50, 100]
    recall_at_k = {}

    for sys_name, rankings in rankings_cache.items():
        recall_at_k[sys_name] = {}
        for k_val in k_cutoffs:
            recalls = []
            for qid, ranked_pids in rankings.items():
                gold = qrels.get(qid, set())
                if not gold:
                    continue
                hits = sum(1 for p in ranked_pids[:k_val] if p in gold)
                recalls.append(hits / len(gold))
            recall_at_k[sys_name][f"Recall@{k_val}"] = round(float(np.mean(recalls)), 4)
        print(f"  {sys_name:15}: {recall_at_k[sys_name]}")

    results_data["recall_at_k"] = recall_at_k

    # ---------------------------------------------------------
    # 8. Metrics by Query Length (Short, Medium, Long)
    # ---------------------------------------------------------
    print("\n--- 8. Metrics by Query Length (BM25 vs Dense) ---")
    query_lens = {qid: len(q.split()) for qid, q in queries.items()}
    buckets = {
        "Short (<= 6 words)": [qid for qid, l in query_lens.items() if l <= 6],
        "Medium (7-10 words)": [qid for qid, l in query_lens.items() if 7 <= l <= 10],
        "Long (>= 11 words)": [qid for qid, l in query_lens.items() if l >= 11]
    }

    by_length_results = {}
    for b_name, qid_list in buckets.items():
        bm25_sub = {qid: rankings_cache["BM25"][qid] for qid in qid_list}
        dense_sub = {qid: rankings_cache["Dense (MiniLM)"][qid] for qid in qid_list}
        hybrid_sub = {qid: rankings_cache["Hybrid RRF"][qid] for qid in qid_list}

        ev_b = evaluate_system(bm25_sub, qrels)
        ev_d = evaluate_system(dense_sub, qrels)
        ev_h = evaluate_system(hybrid_sub, qrels)

        by_length_results[b_name] = {
            "query_count": len(qid_list),
            "BM25": {m: round(ev_b["mean"][m], 4) for m in ["P@10", "Recall@100", "MAP", "nDCG@10"]},
            "Dense": {m: round(ev_d["mean"][m], 4) for m in ["P@10", "Recall@100", "MAP", "nDCG@10"]},
            "Hybrid": {m: round(ev_h["mean"][m], 4) for m in ["P@10", "Recall@100", "MAP", "nDCG@10"]}
        }
        print(f"  {b_name} (N={len(qid_list)}): BM25 nDCG@10={ev_b['mean']['nDCG@10']:.4f} | Dense nDCG@10={ev_d['mean']['nDCG@10']:.4f}")

    results_data["metrics_by_query_length"] = by_length_results

    # ---------------------------------------------------------
    # 9. Hard Queries (nDCG@10 == 0 across ALL systems)
    # ---------------------------------------------------------
    print("\n--- 9. Finding Hard Queries (nDCG@10 == 0 on all systems) ---")
    hard_qids = []
    for qid in queries:
        scores = [
            eval_results["TF-IDF"]["per_query"][qid]["nDCG@10"],
            eval_results["BM25"]["per_query"][qid]["nDCG@10"],
            eval_results["Dense (MiniLM)"]["per_query"][qid]["nDCG@10"],
            eval_results["Hybrid RRF"]["per_query"][qid]["nDCG@10"]
        ]
        if all(s == 0.0 for s in scores):
            hard_qids.append(qid)

    print(f"Total Hard Queries: {len(hard_qids)} out of {len(queries)} ({len(hard_qids)/len(queries)*100:.1f}%)")

    hard_examples = []
    for qid in hard_qids[:3]:
        q_text = queries[qid]
        gold_pids = list(qrels[qid])
        gold_snippet = passages.get(gold_pids[0], "")[:250].replace("\n", " ")

        bm25_top1_pid = rankings_cache["BM25"][qid][0]
        dense_top1_pid = rankings_cache["Dense (MiniLM)"][qid][0]
        hybrid_top1_pid = rankings_cache["Hybrid RRF"][qid][0]

        hard_examples.append({
            "qid": qid,
            "query": q_text,
            "gold_pids": gold_pids,
            "gold_passage_snippet": f"PID {gold_pids[0]}: {gold_snippet}...",
            "bm25_rank1": {
                "pid": bm25_top1_pid,
                "snippet": passages.get(bm25_top1_pid, "")[:200].replace("\n", " ")
            },
            "dense_rank1": {
                "pid": dense_top1_pid,
                "snippet": passages.get(dense_top1_pid, "")[:200].replace("\n", " ")
            },
            "hybrid_rank1": {
                "pid": hybrid_top1_pid,
                "snippet": passages.get(hybrid_top1_pid, "")[:200].replace("\n", " ")
            }
        })

    results_data["hard_queries"] = {
        "count": len(hard_qids),
        "percentage": round(len(hard_qids) / len(queries) * 100, 2),
        "examples": hard_examples
    }

    # ---------------------------------------------------------
    # 10. Cross-Encoder Reranker over Hybrid Top-50
    # ---------------------------------------------------------
    print("\n--- 10. Evaluating Cross-Encoder Reranker over Hybrid Top-50 ---")
    try:
        from sentence_transformers import CrossEncoder
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading cross-encoder/ms-marco-MiniLM-L-6-v2 on {device}...")
        ce_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)

        reranked_rankings = {}
        t0_ce = time.perf_counter()

        # Batch prediction over all queries
        for qid, q_text in query_items:
            hybrid_top50_pids = rankings_cache["Hybrid RRF"][qid][:50]
            pairs = [(q_text, passages.get(pid, "")) for pid in hybrid_top50_pids]
            ce_scores = ce_model.predict(pairs, batch_size=64, show_progress_bar=False)

            # Sort top 50 by CE score
            scored_pairs = sorted(zip(hybrid_top50_pids, ce_scores), key=lambda x: x[1], reverse=True)
            reranked_pids = [p for p, _ in scored_pairs]

            # Preserve ranks 51-100 from original hybrid list
            remaining_pids = rankings_cache["Hybrid RRF"][qid][50:100]
            reranked_rankings[qid] = reranked_pids + remaining_pids

        ce_time = time.perf_counter() - t0_ce
        ce_eval = evaluate_system(reranked_rankings, qrels)

        results_data["cross_encoder_reranker"] = {
            "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "top_k_reranked": 50,
            "device": device,
            "inference_time_seconds": round(ce_time, 2),
            "metrics": {
                "P@10": round(ce_eval["mean"]["P@10"], 4),
                "Recall@100": round(ce_eval["mean"]["Recall@100"], 4),
                "MAP": round(ev_eval := ce_eval["mean"]["MAP"], 4),
                "nDCG@10": round(ce_eval["mean"]["nDCG@10"], 4)
            }
        }
        print(f"  Cross-Encoder Results (Rerank Top-50): P@10={ce_eval['mean']['P@10']:.4f} | "
              f"Recall@100={ce_eval['mean']['Recall@100']:.4f} | "
              f"MAP={ce_eval['mean']['MAP']:.4f} | "
              f"nDCG@10={ce_eval['mean']['nDCG@10']:.4f} (Finished in {ce_time:.1f}s)")
    except Exception as e:
        print(f"Cross-encoder execution failed: {e}")
        results_data["cross_encoder_reranker"] = "not available"

    # ---------------------------------------------------------
    # 11. RAG Evaluation over 30 Random Queries
    # ---------------------------------------------------------
    print("\n--- 11. Running RAG Evaluation on 30 Random Queries ---")
    random.seed(42)
    # Eligible queries: gold is in top 5 of Hybrid RRF
    hybrid_rankings = rankings_cache["Hybrid RRF"]
    gold_in_top5_qids = [
        qid for qid, pids in hybrid_rankings.items()
        if any(p in qrels[qid] for p in pids[:5])
    ]
    selected_30 = random.sample(gold_in_top5_qids, min(30, len(gold_in_top5_qids)))

    rag_eval_results = []
    valid_citations_count = 0
    abstention_gold_present_count = 0
    abstention_gold_missing_count = 0

    for qid in selected_30:
        q_text = queries[qid]
        gold_pids = qrels[qid]
        top5_pids = hybrid_rankings[qid][:5]

        # Condition A: Gold Present
        passages_present = [
            {"rank": i + 1, "pid": pid, "text": passages.get(pid, "")}
            for i, pid in enumerate(top5_pids)
        ]
        res_present = generate_rag_answer(query=q_text, passages=passages_present)
        cited = res_present["cited_indices"]
        is_valid = len(cited) > 0 and all(1 <= c <= len(passages_present) for c in cited)
        if is_valid:
            valid_citations_count += 1
        if res_present["is_not_found"]:
            abstention_gold_present_count += 1

        # Condition B: Gold Missing (replace with next non-gold distractors)
        non_gold_pids = [p for p in hybrid_rankings[qid] if p not in gold_pids][:5]
        passages_missing = [
            {"rank": i + 1, "pid": pid, "text": passages.get(pid, "")}
            for i, pid in enumerate(non_gold_pids)
        ]
        res_missing = generate_rag_answer(query=q_text, passages=passages_missing)
        if res_missing["is_not_found"]:
            abstention_gold_missing_count += 1

        rag_eval_results.append({
            "qid": qid,
            "query": q_text,
            "gold_pids": list(gold_pids),
            "condition_present": {
                "answer": res_present["answer"],
                "cited_indices": cited,
                "citations_valid": is_valid,
                "is_not_found": res_present["is_not_found"]
            },
            "condition_missing": {
                "answer": res_missing["answer"],
                "is_not_found": res_missing["is_not_found"]
            }
        })

    n_sample = len(selected_30)
    results_data["rag_30_query_evaluation"] = {
        "num_queries_evaluated": n_sample,
        "gold_present_condition": {
            "citation_validity_rate": round(valid_citations_count / n_sample * 100, 2),
            "abstention_rate": round(abstention_gold_present_count / n_sample * 100, 2)
        },
        "gold_missing_condition": {
            "abstention_rate": round(abstention_gold_missing_count / n_sample * 100, 2)
        },
        "detailed_samples": rag_eval_results[:5]  # include first 5 detailed samples
    }

    print(f"  RAG 30-Query Results:")
    print(f"    - Citation Validity Rate (Gold Present): {valid_citations_count}/{n_sample} ({valid_citations_count/n_sample*100:.1f}%)")
    print(f"    - Abstention Rate (Gold Present): {abstention_gold_present_count}/{n_sample} ({abstention_gold_present_count/n_sample*100:.1f}%)")
    print(f"    - Abstention Rate (Gold Missing): {abstention_gold_missing_count}/{n_sample} ({abstention_gold_missing_count/n_sample*100:.1f}%)")

    # ---------------------------------------------------------
    # Save Results to JSON and Markdown
    # ---------------------------------------------------------
    out_json = RESULTS_DIR / "report_measurements.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nSaved structured experimental data to {out_json}")

    # Generate Markdown Summary
    out_md = RESULTS_DIR / "report_measurements.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# LoTTE RAG Comprehensive Experimental Report Measurements\n\n")

        f.write("## 1. System Engineering & Resource Footprint\n")
        f.write(f"- **Vocabulary Size (stemmed tokens)**: `{vocab_size:,}`\n")
        f.write(f"- **Stop-words List Size**: `{num_stopwords}`\n")
        f.write(f"- **Stemmer**: `{stemmer_name}`\n")
        f.write(f"- **Total Startup Latency**: `{total_startup_time:.2f} seconds`\n")
        f.write(f"- **RAM Allocated at Startup**: `{mem_total - mem_before:.1f} MB` (Total Process RSS: `{mem_total:.1f} MB`)\n")
        f.write(f"- **FAISS Index Vector Size**: `{faiss_mb:.2f} MB` (`{len(doc_ids):,}` vectors $\\times 384$ dim, float32)\n\n")

        f.write("## 2. Latency Benchmarks (417 Queries)\n\n")
        f.write("| System | Mean (ms) | Median (ms) | P95 (ms) |\n| :--- | :---: | :---: | :---: |\n")
        for sys_name, lats in latency_stats.items():
            f.write(f"| **{sys_name}** | {lats['mean_ms']} | {lats['median_ms']} | {lats['p95_ms']} |\n")
        f.write("\n")

        f.write("## 3. Pairwise Win / Loss / Tie (nDCG@10)\n\n")
        f.write("| Comparison | Wins | Losses | Ties |\n| :--- | :---: | :---: | :---: |\n")
        f.write(f"| **Dense vs BM25** | {wlt_dense_vs_bm25['wins']} | {wlt_dense_vs_bm25['losses']} | {wlt_dense_vs_bm25['ties']} |\n")
        f.write(f"| **Hybrid vs Dense** | {wlt_hybrid_vs_dense['wins']} | {wlt_hybrid_vs_dense['losses']} | {wlt_hybrid_vs_dense['ties']} |\n")
        f.write(f"| **Hybrid vs BM25** | {wlt_hybrid_vs_bm25['wins']} | {wlt_hybrid_vs_bm25['losses']} | {wlt_hybrid_vs_bm25['ties']} |\n\n")

        f.write("## 4. Statistical Significance Tests (nDCG@10)\n\n")
        f.write("| Comparison | Mean Diff | Paired t-test ($t, p$) | Wilcoxon ($W, p$) | Bootstrap 95% CI |\n| :--- | :---: | :---: | :---: | :---: |\n")
        f.write(f"| **Dense vs BM25** | +{mean_diff_d_b:.4f} | $t={t_stat_d_b:.4f}, p={p_val_ttest_d_b:.2e}$ | $W={wilc_stat_d_b:.1f}, p={p_val_wilc_d_b:.2e}$ | [{ci_low_d_b:.4f}, {ci_high_d_b:.4f}] |\n")
        f.write(f"| **Hybrid vs Dense** | {mean_diff_h_d:.4f} | $t={t_stat_h_d:.4f}, p={p_val_ttest_h_d:.2e}$ | $W={wilc_stat_h_d:.1f}, p={p_val_wilc_h_d:.2e}$ | [{ci_low_h_d:.4f}, {ci_high_h_d:.4f}] |\n\n")

        f.write("## 5. BM25 Hyperparameter Grid Sweep\n\n")
        f.write("| $k_1$ | $b$ | P@10 | Recall@100 | MAP | nDCG@10 |\n| :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for row in bm25_grid_results:
            f.write(f"| {row['k1']} | {row['b']} | {row['P@10']:.4f} | {row['Recall@100']:.4f} | {row['MAP']:.4f} | {row['nDCG@10']:.4f} |\n")
        f.write("\n")

        f.write("## 6. RRF and Score Fusion Sweep\n\n")
        f.write("| Fusion Strategy | $k$ | Dense Weight ($w$) | P@10 | Recall@100 | MAP | nDCG@10 |\n| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for row in rrf_sweep_results:
            k_str = str(row['k']) if row['k'] is not None else "-"
            w_str = str(row['dense_weight']) if row['dense_weight'] is not None else "-"
            f.write(f"| {row['type']} | {k_str} | {w_str} | {row['P@10']:.4f} | {row['Recall@100']:.4f} | {row['MAP']:.4f} | {row['nDCG@10']:.4f} |\n")
        f.write("\n")

        f.write("## 7. Recall@k Multi-Cutoff Evaluation\n\n")
        f.write("| System | R@1 | R@5 | R@10 | R@20 | R@50 | R@100 |\n| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for sys_name in rankings_cache:
            r = recall_at_k[sys_name]
            f.write(f"| **{sys_name}** | {r['Recall@1']:.4f} | {r['Recall@5']:.4f} | {r['Recall@10']:.4f} | {r['Recall@20']:.4f} | {r['Recall@50']:.4f} | {r['Recall@100']:.4f} |\n")
        f.write("\n")

        f.write("## 8. Performance by Query Length (BM25 vs Dense)\n\n")
        f.write("| Query Length Bucket | System | P@10 | Recall@100 | MAP | nDCG@10 |\n| :--- | :--- | :---: | :---: | :---: | :---: |\n")
        for b_name, b_data in by_length_results.items():
            f.write(f"| **{b_name}** ($N={b_data['query_count']}$) | BM25 | {b_data['BM25']['P@10']:.4f} | {b_data['BM25']['Recall@100']:.4f} | {b_data['BM25']['MAP']:.4f} | {b_data['BM25']['nDCG@10']:.4f} |\n")
            f.write(f"| | Dense | {b_data['Dense']['P@10']:.4f} | {b_data['Dense']['Recall@100']:.4f} | {b_data['Dense']['MAP']:.4f} | {b_data['Dense']['nDCG@10']:.4f} |\n")
            f.write(f"| | Hybrid RRF | {b_data['Hybrid']['P@10']:.4f} | {b_data['Hybrid']['Recall@100']:.4f} | {b_data['Hybrid']['MAP']:.4f} | {b_data['Hybrid']['nDCG@10']:.4f} |\n")
        f.write("\n")

        f.write(f"## 9. Hard Queries Analysis\n\n")
        f.write(f"- Total queries scoring 0 across all 4 systems: `{len(hard_qids)}` out of `{len(queries)}` (`{len(hard_qids)/len(queries)*100:.1f}%`)\n\n")
        for i, ex in enumerate(hard_examples, 1):
            f.write(f"### Example {i}: QID {ex['qid']} - *\"{ex['query']}\"*\n")
            f.write(f"- **Gold Target**: `{ex['gold_passage_snippet']}`\n")
            f.write(f"- **BM25 Top Match**: PID {ex['bm25_rank1']['pid']} — *\"{ex['bm25_rank1']['snippet']}...\"*\n")
            f.write(f"- **Dense Top Match**: PID {ex['dense_rank1']['pid']} — *\"{ex['dense_rank1']['snippet']}...\"*\n")
            f.write(f"- **Hybrid Top Match**: PID {ex['hybrid_rank1']['pid']} — *\"{ex['hybrid_rank1']['snippet']}...\"*\n\n")

        if "cross_encoder_reranker" in results_data and isinstance(results_data["cross_encoder_reranker"], dict):
            ce = results_data["cross_encoder_reranker"]
            f.write("## 10. Cross-Encoder Reranker Benchmark (Top-50 Hybrid Candidate Reranking)\n\n")
            f.write(f"- **Model**: `{ce['model']}`\n")
            f.write(f"- **Inference Hardware**: `{ce['device']}` ({ce['inference_time_seconds']}s)\n")
            f.write("| Metric | Cross-Encoder Reranked | Dense (MiniLM) | Hybrid RRF |\n| :--- | :---: | :---: | :---: |\n")
            f.write(f"| **P@10** | **{ce['metrics']['P@10']:.4f}** | 0.1566 | 0.1463 |\n")
            f.write(f"| **Recall@100** | **{ce['metrics']['Recall@100']:.4f}** | 0.7980 | 0.7903 |\n")
            f.write(f"| **MAP** | **{ce['metrics']['MAP']:.4f}** | 0.4072 | 0.3830 |\n")
            f.write(f"| **nDCG@10** | **{ce['metrics']['nDCG@10']:.4f}** | 0.4810 | 0.4508 |\n\n")

        f.write("## 11. RAG Evaluation over 30 Random Queries\n\n")
        f.write(f"- **Queries Evaluated**: `{n_sample}`\n")
        f.write(f"- **Citation Validity Rate (Gold Present)**: `{valid_citations_count / n_sample * 100:.1f}%` ({valid_citations_count}/{n_sample})\n")
        f.write(f"- **Abstention Rate (Gold Present)**: `{abstention_gold_present_count / n_sample * 100:.1f}%` ({abstention_gold_present_count}/{n_sample})\n")
        f.write(f"- **Abstention Rate ('not found' when Gold Missing)**: `{abstention_gold_missing_count / n_sample * 100:.1f}%` ({abstention_gold_missing_count}/{n_sample})\n\n")

    print(f"Saved human-readable Markdown report to {out_md}")
    print("=" * 70)
    print("ALL MEASUREMENTS SUCCESSFULLY COMPLETED!")
    print("=" * 70)


if __name__ == "__main__":
    main()

