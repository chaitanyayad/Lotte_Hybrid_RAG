"""
src/sparse.py: Inverted Index, TF-IDF, and BM25 implementations for sparse retrieval.
Includes validation against rank_bm25.BM25Okapi.
"""

import math
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set, Optional


class InvertedIndex:
    """
    Lightweight inverted index mapping term -> {pid: term_frequency}.
    Also stores document lengths and average document length.
    """
    def __init__(self):
        # term -> {pid: tf}
        self.postings: Dict[str, Dict[int, int]] = defaultdict(dict)
        # pid -> doc_length (number of tokens)
        self.doc_lens: Dict[int, int] = {}
        # total documents
        self.N: int = 0
        self.avg_dl: float = 0.0

    def fit(self, tokenized_corpus: Dict[int, List[str]]):
        self.N = len(tokenized_corpus)
        total_len = 0
        for pid, tokens in tokenized_corpus.items():
            doc_len = len(tokens)
            self.doc_lens[pid] = doc_len
            total_len += doc_len

            counts = Counter(tokens)
            for term, count in counts.items():
                self.postings[term][pid] = count

        self.avg_dl = (total_len / self.N) if self.N > 0 else 0.0
        return self

    def get_df(self, term: str) -> int:
        return len(self.postings.get(term, {}))


class TFIDFRetriever:
    """
    TF-IDF retriever based on the inverted index.
    tf(t, d) = count(t, d) / len(d)
    idf(t) = ln((N + 1) / (df(t) + 1)) + 1
    """
    def __init__(self, index: InvertedIndex):
        self.index = index
        self.idf_cache: Dict[str, float] = {}
        for term, posting in self.index.postings.items():
            df = len(posting)
            self.idf_cache[term] = math.log((self.index.N + 1) / (df + 1)) + 1.0

    def search(self, query_tokens: List[str], top_k: int = 100) -> List[Tuple[int, float]]:
        scores: Dict[int, float] = defaultdict(float)
        query_counts = Counter(query_tokens)

        for term, q_tf in query_counts.items():
            if term not in self.index.postings:
                continue
            idf = self.idf_cache[term]
            postings = self.index.postings[term]

            for pid, tf in postings.items():
                doc_len = self.index.doc_lens[pid]
                if doc_len > 0:
                    tf_norm = tf / doc_len
                    scores[pid] += q_tf * tf_norm * idf

        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]


class BM25Retriever:
    """
    BM25 implementation (~40 lines core) with default k1=1.5, b=0.75.
    Uses Okapi BM25 formulation identical to rank_bm25.BM25Okapi.
    """
    def __init__(self, index: InvertedIndex, k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25):
        self.index = index
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.idf_cache: Dict[str, float] = {}

        idf_sum = 0.0
        negative_idfs = []
        for term, posting in self.index.postings.items():
            df = len(posting)
            # Robertson-Spärck Jones IDF
            idf = math.log(self.index.N - df + 0.5) - math.log(df + 0.5)
            self.idf_cache[term] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(term)

        avg_idf = idf_sum / len(self.idf_cache) if self.idf_cache else 0.0
        eps = self.epsilon * avg_idf
        for term in negative_idfs:
            self.idf_cache[term] = eps

    def search(self, query_tokens: List[str], top_k: int = 100) -> List[Tuple[int, float]]:
        scores: Dict[int, float] = defaultdict(float)

        for term in query_tokens:
            if term not in self.index.postings:
                continue
            idf = self.idf_cache[term]
            postings = self.index.postings[term]

            for pid, tf in postings.items():
                dl = self.index.doc_lens[pid]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / self.index.avg_dl))
                numerator = tf * (self.k1 + 1.0)
                scores[pid] += idf * (numerator / denom)

        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]



def validate_bm25():
    """
    Validates our BM25 implementation against rank_bm25.BM25Okapi on a toy corpus.
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        print("rank_bm25 not installed; skipping external library validation comparison.")
        return True

    corpus = {
        1: ["dog", "barks", "loudly", "in", "park"],
        2: ["cat", "meows", "softly", "at", "night"],
        3: ["dog", "and", "cat", "play", "in", "the", "garden"],
        4: ["rabbit", "eats", "fresh", "carrots"],
    }
    query = ["dog", "in", "garden"]

    # Hand-rolled
    idx = InvertedIndex().fit(corpus)
    custom_bm25 = BM25Retriever(idx, k1=1.5, b=0.75)
    custom_scores = dict(custom_bm25.search(query, top_k=4))

    # rank_bm25 reference
    pids = list(corpus.keys())
    tokenized_docs = [corpus[pid] for pid in pids]
    ref_bm25 = BM25Okapi(tokenized_docs, k1=1.5, b=0.75)
    ref_scores_list = ref_bm25.get_scores(query)
    ref_scores = {pids[i]: ref_scores_list[i] for i in range(len(pids))}

    print("BM25 Validation on Toy Corpus:")
    all_close = True
    for pid in pids:
        c_score = custom_scores.get(pid, 0.0)
        r_score = ref_scores.get(pid, 0.0)
        diff = abs(c_score - r_score)
        print(f"Doc {pid}: Custom = {c_score:.4f}, Reference = {r_score:.4f}, diff = {diff:.6f}")
        if diff > 1e-4:
            all_close = False

    if all_close:
        print("PASSED: Custom BM25 matches rank_bm25 within 1e-4 tolerance!")
    else:
        print("WARNING: Custom BM25 has discrepancies with rank_bm25.")
    return all_close


if __name__ == "__main__":
    validate_bm25()
