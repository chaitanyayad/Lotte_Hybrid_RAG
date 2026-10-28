"""
src/load.py: Data loader and dataset statistics for LoTTE (lifestyle domain, dev split, search queries).
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, Set, Tuple


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def ensure_dataset(data_dir: Path = DATA_DIR) -> Tuple[Path, Path]:
    """
    Ensures that LoTTE lifestyle dev collection and search QAs are present locally in data_dir.
    Downloads them using huggingface_hub if not already present.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    collection_file = data_dir / "lifestyle_dev_collection.jsonl"
    qas_file = data_dir / "lifestyle_dev_qas.search.jsonl"

    if not collection_file.exists():
        print(f"Downloading lifestyle dev collection to {collection_file}...")
        try:
            from huggingface_hub import hf_hub_download
            cached_path = hf_hub_download(
                repo_id="colbertv2/lotte_passages",
                filename="lifestyle/dev_collection.jsonl",
                repo_type="dataset"
            )
            shutil.copyfile(cached_path, collection_file)
            print(f"Collection saved to {collection_file}")
        except Exception as e:
            raise RuntimeError(f"Failed to download collection: {e}")

    if not qas_file.exists():
        print(f"Downloading lifestyle dev search QAs to {qas_file}...")
        try:
            from huggingface_hub import hf_hub_download
            cached_path = hf_hub_download(
                repo_id="colbertv2/lotte",
                filename="lifestyle/dev_qas.search.jsonl",
                repo_type="dataset"
            )
            shutil.copyfile(cached_path, qas_file)
            print(f"QAs saved to {qas_file}")
        except Exception as e:
            raise RuntimeError(f"Failed to download QAs: {e}")

    return collection_file, qas_file


def load_lotte(data_dir: Path = DATA_DIR) -> Tuple[Dict[int, str], Dict[int, str], Dict[int, Set[int]]]:
    """
    Loads the LoTTE lifestyle dev search dataset.

    Returns:
        passages: dict[pid -> text]
        queries:  dict[qid -> text]
        qrels:    dict[qid -> set(pid)]
    """
    collection_file, qas_file = ensure_dataset(data_dir)

    passages: Dict[int, str] = {}
    with open(collection_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            # Verify actual schema: 'doc_id' as pid, 'text' as content
            pid = int(item.get("doc_id", item.get("id", item.get("pid"))))
            text = item.get("text", item.get("passage", ""))
            passages[pid] = text

    queries: Dict[int, str] = {}
    qrels: Dict[int, Set[int]] = {}
    with open(qas_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            qid = int(item["qid"])
            query = item["query"]
            answer_pids = set(int(p) for p in item.get("answer_pids", []))

            queries[qid] = query
            qrels[qid] = answer_pids

    return passages, queries, qrels


def compute_and_print_stats(
    passages: Dict[int, str],
    queries: Dict[int, str],
    qrels: Dict[int, Set[int]]
) -> Dict[str, float]:
    """
    Computes and prints required statistics for the dataset report,
    and runs the sanity check on qrels.
    """
    corpus_size = len(passages)
    query_count = len(queries)

    # Passage lengths (word count)
    total_words = sum(len(text.split()) for text in passages.values())
    avg_passage_length = total_words / corpus_size if corpus_size > 0 else 0.0

    # Relevant passages per query
    rel_counts = [len(pids) for pids in qrels.values()]
    avg_rel_per_query = sum(rel_counts) / len(rel_counts) if rel_counts else 0.0
    min_rel = min(rel_counts) if rel_counts else 0
    max_rel = max(rel_counts) if rel_counts else 0

    # Sanity check: every qrels pid must exist in the corpus
    missing_pids = set()
    for qid, pids in qrels.items():
        for pid in pids:
            if pid not in passages:
                missing_pids.add(pid)

    sanity_check_passed = (len(missing_pids) == 0)

    print("=" * 60)
    print("LoTTE Dataset Statistics (Lifestyle / Dev / Search)")
    print("=" * 60)
    print(f"Corpus size (passages)       : {corpus_size:,}")
    print(f"Query count                  : {query_count:,}")
    print(f"Avg passage length (words)   : {avg_passage_length:.2f}")
    print(f"Relevant passages / query    : avg {avg_rel_per_query:.2f} (min {min_rel}, max {max_rel})")
    print(f"Sanity Check (all qrels in corpus): {'PASSED (0 missing)' if sanity_check_passed else f'FAILED ({len(missing_pids)} missing)'}")
    print("=" * 60)

    if not sanity_check_passed:
        print(f"WARNING: Missing passage IDs in corpus: {list(missing_pids)[:10]}")

    stats = {
        "corpus_size": corpus_size,
        "query_count": query_count,
        "avg_passage_length_words": avg_passage_length,
        "avg_rel_per_query": avg_rel_per_query,
        "min_rel_per_query": min_rel,
        "max_rel_per_query": max_rel,
        "sanity_check_passed": sanity_check_passed,
    }

    # Save to results/dataset_stats.json for report generation
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    stats_file = results_dir / "dataset_stats.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Dataset stats saved to {stats_file}")

    return stats


if __name__ == "__main__":
    passages, queries, qrels = load_lotte()
    compute_and_print_stats(passages, queries, qrels)
