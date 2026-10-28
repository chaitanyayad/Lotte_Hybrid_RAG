"""
src/dense.py: Dense retrieval pipeline using sentence-transformers/all-MiniLM-L6-v2 and FAISS FlatIP.
Encodes passages on GPU (fp16), caches embeddings to disk, and performs cosine search.
"""

import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

# Lazy imports for heavy packages to allow flexible execution
_MODEL = None
_INDEX = None
_DOC_IDS = None


def get_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def load_encoder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: Optional[str] = None):
    from sentence_transformers import SentenceTransformer
    import torch

    if device is None:
        device = get_device()

    print(f"Loading {model_name} on device: {device}...")
    model = SentenceTransformer(model_name, device=device)
    if device == "cuda":
        model = model.half()  # fp16 for fast inference on RTX 4060
        print("Model cast to half-precision (fp16) for GPU acceleration.")
    return model


def encode_corpus(
    passages: Dict[int, str],
    cache_path: Optional[Path] = None,
    batch_size: int = 128,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encodes the corpus using all-MiniLM-L6-v2 with L2 normalization.
    Caches to disk so it is never recomputed.

    Returns:
        embeddings: np.ndarray [N, D] (float32 for FAISS)
        doc_ids: np.ndarray [N] (doc_id mapping)
    """
    if cache_path is None:
        cache_path = Path(__file__).resolve().parent.parent / "results" / "dense_embeddings.npz"

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        print(f"Loading cached embeddings from {cache_path}...")
        data = np.load(cache_path)
        embeddings = data["embeddings"]
        doc_ids = data["doc_ids"]
        print(f"Loaded {len(doc_ids)} embeddings of dimension {embeddings.shape[1]}.")
        return embeddings, doc_ids

    print(f"No cached embeddings found at {cache_path}. Starting encoding...")
    start_time = time.time()

    model = load_encoder(model_name=model_name)
    doc_ids = np.array(list(passages.keys()), dtype=np.int64)
    texts = [passages[pid] for pid in doc_ids]

    # Normalize embeddings for cosine similarity via dot product
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    # FAISS expects float32
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)

    print(f"Encoding complete in {time.time() - start_time:.2f}s. Caching to {cache_path}...")
    np.savez_compressed(cache_path, embeddings=embeddings, doc_ids=doc_ids)
    print("Cached embeddings saved successfully.")

    return embeddings, doc_ids


class DenseRetriever:
    """
    FAISS FlatIP (inner product on L2-normalized embeddings = cosine similarity).
    """
    def __init__(
        self,
        embeddings: np.ndarray,
        doc_ids: np.ndarray,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        import faiss

        self.doc_ids = doc_ids
        self.dimension = embeddings.shape[1]
        self.model_name = model_name
        self.model = None

        print(f"Building FAISS FlatIP index with {len(doc_ids)} vectors of dim {self.dimension}...")
        # L2-normalized vectors -> Inner Product is exact Cosine Similarity
        self.index = faiss.IndexFlatIP(self.dimension)
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        self.index.add(embeddings)
        print(f"FAISS index built. Total indexed vectors: {self.index.ntotal}")

    def _ensure_model(self):
        if self.model is None:
            self.model = load_encoder(model_name=self.model_name)

    def search(self, query_text: str, top_k: int = 100) -> List[Tuple[int, float]]:
        self._ensure_model()
        q_emb = self.model.encode(
            [query_text],
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype(np.float32)

        scores, indices = self.index.search(q_emb, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(self.doc_ids):
                results.append((int(self.doc_ids[idx]), float(score)))
        return results

    def batch_search(self, query_texts: List[str], top_k: int = 100) -> List[List[Tuple[int, float]]]:
        self._ensure_model()
        q_embs = self.model.encode(
            query_texts,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype(np.float32)

        scores, indices = self.index.search(q_embs, top_k)
        all_results = []
        for q_scores, q_indices in zip(scores, indices):
            res = [
                (int(self.doc_ids[idx]), float(s))
                for s, idx in zip(q_scores, q_indices)
                if idx != -1 and idx < len(self.doc_ids)
            ]
            all_results.append(res)
        return all_results
