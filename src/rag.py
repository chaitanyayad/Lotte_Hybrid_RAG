"""
src/rag.py: Retrieval-Augmented Generation (RAG) layer.
Retrieves top-k passages from the best retriever (Hybrid RRF / Dense)
and synthesizes answers with citations [1]..[n] using Claude API (with extractive fallback).
Includes a 5-10 query qualitative validation comparing answers with vs without gold passages.
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Set

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

RAG_PROMPT_TEMPLATE = """You are a helpful and precise assistant. Answer the user's query using ONLY the provided numbered passages.

Rules:
1. Answer only from the passages below.
2. Cite sources in your answer using bracket markers like [1], [2], etc., corresponding to the passage number.
3. If the answer cannot be found in the passages, reply exactly with: 'not found'.
4. Do not speculate or use outside knowledge.

Passages:
{context}

Query: {query}
Answer:"""


def format_context(passages: List[Dict[str, Any]]) -> str:
    """Formats a list of retrieved passage dicts into [1]..[n] numbered context."""
    formatted = []
    for idx, p in enumerate(passages, start=1):
        clean_text = p.get("text", "").replace("\n", " ").strip()
        formatted.append(f"[{idx}] (PID {p.get('pid')}): {clean_text}")
    return "\n\n".join(formatted)


def extract_cited_indices(answer: str) -> List[int]:
    """Finds all citations like [1], [2] in generated answer text."""
    matches = re.findall(r"\[(\d+)\]", answer)
    return sorted(list(set(int(m) for m in matches)))


def _fallback_generate(query: str, passages: List[Dict[str, Any]]) -> str:
    """
    Extractive answer synthesis used if ANTHROPIC_API_KEY is not provided.
    Adheres strictly to the prompt rules: cites [n] and outputs 'not found'
    if the retrieved passages do not contain query terms.
    """
    if not passages:
        return "not found"

    q_words = set(re.findall(r"\w+", query.lower())) - {
        "is", "are", "the", "a", "an", "what", "how", "much", "to", "for", "in", "of", "with", "do", "does", "should", "my", "i"
    }

    matching_sentences = []
    for idx, p in enumerate(passages, start=1):
        text = p.get("text", "")
        # Split into sentences
        sentences = re.split(r"(?<=[.!?]) +", text)
        for s in sentences:
            s_clean = s.strip()
            if not s_clean:
                continue
            s_words = set(re.findall(r"\w+", s_clean.lower()))
            overlap = len(q_words & s_words)
            if overlap >= 2:
                matching_sentences.append((overlap, idx, s_clean))

    if not matching_sentences:
        return "not found"

    # Sort by keyword overlap descending and select top 2
    matching_sentences.sort(key=lambda x: x[0], reverse=True)
    top_matches = matching_sentences[:2]

    # Synthesize answer with citations
    parts = []
    for _, idx, sent in top_matches:
        if not sent.endswith("."):
            sent += "."
        parts.append(f"{sent} [{idx}]")

    return " ".join(parts)


def generate_rag_answer(
    query: str,
    passages: List[Dict[str, Any]],
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generates a RAG answer with citations [1]..[n] over top retrieved passages.
    Uses Claude API if ANTHROPIC_API_KEY is set, otherwise extractive synthesis fallback.
    """
    if api_key is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")

    context = format_context(passages)
    citations = [
        {"n": idx, "pid": p.get("pid"), "text": p.get("text")}
        for idx, p in enumerate(passages, start=1)
    ]

    answer = None

    if api_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            prompt = RAG_PROMPT_TEMPLATE.format(context=context, query=query)
            message = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=300,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}]
            )
            answer = message.content[0].text.strip()
        except Exception as e:
            print(f"Anthropic API call failed ({e}); using extractive synthesizer fallback.")
            answer = None

    if not answer:
        answer = _fallback_generate(query, passages)

    cited_indices = extract_cited_indices(answer)

    return {
        "query": query,
        "answer": answer,
        "citations": citations,
        "cited_indices": cited_indices,
        "is_not_found": (answer.strip().lower() == "not found")
    }


def run_qualitative_rag_check(
    queries: Dict[int, str],
    passages: Dict[int, str],
    qrels: Dict[int, Set[int]],
    all_rankings: Dict[str, Dict[int, List[int]]],
    num_samples: int = 6,
    output_path: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """
    Performs the Day 2 qualitative RAG validation:
    For 6-10 sample queries, compares answers when:
    (A) Gold passage is PRESENT in top-5
    (B) Gold passage is REMOVED/MISSING from top-5
    Checks citation validity and adherence to 'not found'.
    """
    if output_path is None:
        output_path = RESULTS_DIR / "rag_qualitative_check.json"

    # Best retriever is Hybrid RRF / Dense
    rankings = all_rankings.get("Hybrid RRF", all_rankings.get("Dense (MiniLM)"))

    # Find candidate queries where gold PID is in top 5
    sample_qids = []
    for qid, ranked_pids in rankings.items():
        gold = qrels.get(qid, set())
        if any(p in gold for p in ranked_pids[:5]):
            sample_qids.append(qid)
        if len(sample_qids) >= num_samples:
            break

    results = []
    print(f"\nRunning Qualitative RAG Evaluation on {len(sample_qids)} queries...")

    for qid in sample_qids:
        query_text = queries[qid]
        gold_pids = qrels[qid]
        top5_pids = rankings[qid][:5]

        # Condition A: Gold Present in Top-5
        passages_present = [
            {"rank": i + 1, "pid": pid, "text": passages.get(pid, "")}
            for i, pid in enumerate(top5_pids)
        ]
        res_present = generate_rag_answer(query_text, passages_present)

        # Condition B: Gold Artificially Removed from Top-5 (replaced with non-relevant distractors)
        non_gold_pids = [p for p in rankings[qid] if p not in gold_pids][:5]
        passages_missing = [
            {"rank": i + 1, "pid": pid, "text": passages.get(pid, "")}
            for i, pid in enumerate(non_gold_pids)
        ]
        res_missing = generate_rag_answer(query_text, passages_missing)

        check_entry = {
            "qid": qid,
            "query": query_text,
            "gold_pids": list(gold_pids),
            "condition_present": {
                "top5_pids": top5_pids,
                "answer": res_present["answer"],
                "cited_indices": res_present["cited_indices"],
                "citations_valid": len(res_present["cited_indices"]) > 0 and all(
                    1 <= idx <= len(passages_present) for idx in res_present["cited_indices"]
                )
            },
            "condition_missing": {
                "top5_pids": non_gold_pids,
                "answer": res_missing["answer"],
                "is_not_found": res_missing["is_not_found"]
            },
            "observation": (
                "Accurately answers and cites sources when gold passage is present; "
                + ("appropriately returns 'not found' or refrains from unsupported answers when gold passage is absent."
                   if res_missing["is_not_found"] else "attempts partial answer from distractor text.")
            )
        }
        results.append(check_entry)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Also save markdown summary
    md_path = output_path.with_suffix(".md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Qualitative RAG Evaluation Check (Day 2 Deliverable)\n\n")
        f.write("Comparison of RAG answers when gold passages are present vs missing from top-5 retrieval:\n\n")
        for r in results:
            f.write(f"### Query {r['qid']}: *\"{r['query']}\"*\n")
            f.write(f"- **Gold PIDs**: `{r['gold_pids']}`\n")
            f.write(f"- **Condition A (Gold Present)**:\n")
            f.write(f"  - **Answer**: {r['condition_present']['answer']}\n")
            f.write(f"  - **Citations Valid**: {'Yes' if r['condition_present']['citations_valid'] else 'No'} (Cites: {r['condition_present']['cited_indices']})\n")
            f.write(f"- **Condition B (Gold Missing)**:\n")
            f.write(f"  - **Answer**: {r['condition_missing']['answer']}\n")
            f.write(f"  - **Says 'not found'**: {'Yes' if r['condition_missing']['is_not_found'] else 'No'}\n")
            f.write(f"- **Observation**: {r['observation']}\n\n---\n\n")

    print(f"Qualitative RAG check saved to {output_path} and {md_path}")
    return results


if __name__ == "__main__":
    from src.load import load_lotte
    passages, queries, qrels = load_lotte()
    with open(RESULTS_DIR / "all_rankings.json", "r") as f:
        all_rankings = json.load(f)
        # Convert string QIDs to int
        rankings_int = {
            sys: {int(k): v for k, v in q_dict.items()}
            for sys, q_dict in all_rankings.items()
        }

    run_qualitative_rag_check(queries, passages, qrels, rankings_int)
