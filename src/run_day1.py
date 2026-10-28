"""
src/run_day1.py: Master pipeline for Day 1.
Executes:
1. Load dataset & print stats (LoTTE lifestyle dev search)
2. Preprocess corpus for sparse retrieval (with disk caching)
3. Build InvertedIndex, TF-IDF, and BM25 rankers
4. Validate BM25 against rank_bm25
5. Dense encoding on GPU with all-MiniLM-L6-v2 (with disk caching) & FAISS FlatIP
6. Hybrid Reciprocal Rank Fusion (BM25 + Dense, k=60)
7. Full evaluation: P@10, Recall@100, MAP, nDCG@10 across all 4 systems
8. Generate overlaid Precision-Recall curve plot
9. Extract and analyze 3-5 failure/success cases (BM25 vs Dense)
"""

import time
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

from src.load import load_lotte, compute_and_print_stats
from src.preprocess import tokenize
from src.sparse import InvertedIndex, TFIDFRetriever, BM25Retriever, validate_bm25
from src.dense import encode_corpus, DenseRetriever
from src.fusion import reciprocal_rank_fusion
from src.evaluate import (
    evaluate_system,
    plot_overlaid_pr_curves,
    extract_failure_success_cases
)


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_preprocessed_corpus(passages: Dict[int, str]) -> Dict[int, List[str]]:
    """
    Tokenizes and stems the corpus for sparse retrieval.
    Caches tokenized representation to disk to avoid repeated work.
    """
    cache_path = RESULTS_DIR / "tokenized_corpus.pkl"
    if cache_path.exists():
        print(f"Loading preprocessed sparse tokens from {cache_path}...")
        with open(cache_path, "rb") as f:
            tokenized = pickle.load(f)
        print(f"Loaded {len(tokenized):,} preprocessed documents.")
        return tokenized

    print(f"Preprocessing {len(passages):,} passages for sparse retrieval (multiprocessing)...")
    start_time = time.time()
    from src.preprocess import preprocess_corpus_mp
    tokenized = preprocess_corpus_mp(passages, num_workers=8)

    print(f"Tokenization completed in {time.time() - start_time:.2f}s. Caching to {cache_path}...")
    with open(cache_path, "wb") as f:
        pickle.dump(tokenized, f, protocol=pickle.HIGHEST_PROTOCOL)
    return tokenized


def main():
    print("=" * 70)
    print("STARTING DAY 1 PIPELINE: LoTTE RAG IR")
    print("=" * 70)

    # 1. Load dataset & print stats
    passages, queries, qrels = load_lotte()
    stats = compute_and_print_stats(passages, queries, qrels)

    # 2. Preprocess corpus for sparse models
    tokenized_corpus = get_preprocessed_corpus(passages)

    # Preprocess queries
    print("Preprocessing queries...")
    tokenized_queries = {qid: tokenize(q) for qid, q in queries.items()}

    # 3. Build Sparse Inverted Index
    print("Building Inverted Index for sparse models...")
    idx_start = time.time()
    inverted_index = InvertedIndex().fit(tokenized_corpus)
    print(f"Inverted Index built in {time.time() - idx_start:.2f}s. Vocab size: {len(inverted_index.postings):,}")

    # 4. Validate BM25
    print("\nValidating hand-rolled BM25 against rank_bm25...")
    validate_bm25()

    # Initialize Sparse Retrievers
    tfidf_retriever = TFIDFRetriever(inverted_index)
    bm25_retriever = BM25Retriever(inverted_index, k1=1.5, b=0.75)

    # 5. Dense Encoding & FAISS Setup
    print("\nSetting up Dense Retriever (sentence-transformers/all-MiniLM-L6-v2 + FAISS)...")
    embeddings, doc_ids = encode_corpus(passages)
    dense_retriever = DenseRetriever(embeddings, doc_ids)

    # 6. Retrieve for all queries across all 4 systems
    print("\nRetrieving Top-100 results for all 417 queries across systems...")

    all_rankings: Dict[str, Dict[int, List[int]]] = {
        "TF-IDF": {},
        "BM25": {},
        "Dense (MiniLM)": {},
        "Hybrid RRF": {}
    }

    # Also keep raw scored lists for BM25 and Dense to perform RRF
    bm25_scored: Dict[int, List[Tuple[int, float]]] = {}
    dense_scored: Dict[int, List[Tuple[int, float]]] = {}

    print("Running TF-IDF retrieval...")
    for qid, q_toks in tqdm(tokenized_queries.items(), desc="TF-IDF"):
        res = tfidf_retriever.search(q_toks, top_k=100)
        all_rankings["TF-IDF"][qid] = [pid for pid, _ in res]

    print("Running BM25 retrieval...")
    for qid, q_toks in tqdm(tokenized_queries.items(), desc="BM25"):
        res = bm25_retriever.search(q_toks, top_k=100)
        bm25_scored[qid] = res
        all_rankings["BM25"][qid] = [pid for pid, _ in res]

    print("Running Dense retrieval...")
    for qid, q_text in tqdm(queries.items(), desc="Dense"):
        res = dense_retriever.search(q_text, top_k=100)
        dense_scored[qid] = res
        all_rankings["Dense (MiniLM)"][qid] = [pid for pid, _ in res]

    print("Running Hybrid RRF (BM25 + Dense, k=60)...")
    for qid in tqdm(queries.keys(), desc="Hybrid RRF"):
        rrf_res = reciprocal_rank_fusion(
            [bm25_scored[qid], dense_scored[qid]],
            k=60,
            top_k=100
        )
        all_rankings["Hybrid RRF"][qid] = [pid for pid, _ in rrf_res]

    # Save rankings to disk
    rankings_file = RESULTS_DIR / "all_rankings.json"
    with open(rankings_file, "w", encoding="utf-8") as f:
        json.dump(all_rankings, f)
    print(f"\nAll system rankings cached to {rankings_file}")

    # 7. Evaluate all four systems
    print("\nEvaluating all systems on P@10, Recall@100, MAP, nDCG@10...")
    eval_results = {}
    for sys_name, rankings in all_rankings.items():
        eval_results[sys_name] = evaluate_system(rankings, qrels)

    # 8. Print Comparison Table
    print("\n" + "=" * 70)
    print(f"{'System':<20} | {'P@10':<10} | {'Recall@100':<12} | {'MAP':<10} | {'nDCG@10':<10}")
    print("-" * 70)
    comparison_data = []
    for sys_name in ["TF-IDF", "BM25", "Dense (MiniLM)", "Hybrid RRF"]:
        m = eval_results[sys_name]["mean"]
        print(f"{sys_name:<20} | {m['P@10']:<10.4f} | {m['Recall@100']:<12.4f} | {m['MAP']:<10.4f} | {m['nDCG@10']:<10.4f}")
        comparison_data.append({
            "System": sys_name,
            "P@10": round(m["P@10"], 4),
            "Recall@100": round(m["Recall@100"], 4),
            "MAP": round(m["MAP"], 4),
            "nDCG@10": round(m["nDCG@10"], 4)
        })
    print("=" * 70)

    # Save evaluation table
    table_file = RESULTS_DIR / "metrics_table.json"
    with open(table_file, "w", encoding="utf-8") as f:
        json.dump(comparison_data, f, indent=2)

    # Save Markdown table for report
    md_table_file = RESULTS_DIR / "metrics_table.md"
    with open(md_table_file, "w", encoding="utf-8") as f:
        f.write("| System | P@10 | Recall@100 | MAP | nDCG@10 |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        for row in comparison_data:
            f.write(f"| **{row['System']}** | {row['P@10']:.4f} | {row['Recall@100']:.4f} | {row['MAP']:.4f} | {row['nDCG@10']:.4f} |\n")
    print(f"Metrics table saved to {table_file} and {md_table_file}")

    # 9. Plot Overlaid Precision-Recall Curves
    pr_plot_file = RESULTS_DIR / "precision_recall_curves.png"
    plot_overlaid_pr_curves(all_rankings, qrels, pr_plot_file)

    # 10. Extract Failure / Success Cases
    cases_file = RESULTS_DIR / "case_studies.json"
    extract_failure_success_cases(
        eval_results["BM25"],
        eval_results["Dense (MiniLM)"],
        queries,
        passages,
        qrels,
        all_rankings["BM25"],
        all_rankings["Dense (MiniLM)"],
        cases_file
    )

    print("\n" + "=" * 70)
    print("DAY 1 PIPELINE COMPLETE SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
