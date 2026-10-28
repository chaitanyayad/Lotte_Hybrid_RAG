"""
src/preprocess.py: Text preprocessing specifically for sparse retrieval models (TF-IDF, BM25).
Note: Dense retrieval models (e.g., all-MiniLM-L6-v2) process raw text directly.
"""

import re
from typing import List, Dict
import nltk
from nltk.stem.porter import PorterStemmer

# Ensure NLTK stopwords are downloaded
try:
    from nltk.corpus import stopwords
    _STOP_WORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords
    _STOP_WORDS = set(stopwords.words("english"))

_STEMMER = PorterStemmer()
_TOKEN_PATTERN = re.compile(r"\b[a-zA-Z0-9]+\b")


def tokenize(text: str) -> List[str]:
    """
    Standard sparse pipeline:
    1. Lowercase
    2. Alphanumeric regex tokenization
    3. Stopword removal
    4. Porter stemming
    """
    tokens = _TOKEN_PATTERN.findall(text.lower())
    filtered = [
        _STEMMER.stem(tok)
        for tok in tokens
        if tok not in _STOP_WORDS and len(tok) > 1
    ]
    return filtered


def _tokenize_pair(pair):
    pid, text = pair
    return pid, tokenize(text)


def preprocess_corpus(passages: Dict[int, str]) -> Dict[int, List[str]]:
    """
    Preprocesses the whole passage collection for sparse retrieval indexing.
    """
    return {pid: tokenize(text) for pid, text in passages.items()}


def preprocess_corpus_mp(passages: Dict[int, str], num_workers: int = 8) -> Dict[int, List[str]]:
    """
    Multiprocessing accelerated corpus preprocessing.
    """
    import multiprocessing
    from tqdm import tqdm
    items = list(passages.items())
    results = {}
    with multiprocessing.Pool(processes=num_workers) as pool:
        for pid, toks in tqdm(pool.imap_unordered(_tokenize_pair, items, chunksize=1000), total=len(items), desc="Tokenizing corpus (MP)"):
            results[pid] = toks
    return results


if __name__ == "__main__":
    sample = "How much should I feed my 1-year-old English Mastiff?"
    print(f"Original: {sample}")
    print(f"Processed: {tokenize(sample)}")

