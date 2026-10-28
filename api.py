"""
api.py: FastAPI backend service for LoTTE Search & RAG.
Serves:
- GET /search?q=<query>&ranker=<tfidf|bm25|dense|rrf>&k=10
- POST /rag with body {q, ranker, k}
- GET /stats for dataset & model metadata
"""

import os
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.load import load_lotte
from src.preprocess import tokenize
from src.sparse import InvertedIndex, TFIDFRetriever, BM25Retriever
from src.dense import encode_corpus, DenseRetriever
from contextlib import asynccontextmanager
from src.fusion import reciprocal_rank_fusion


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up LoTTE IR service...")
    try:
        passages, queries, qrels = load_lotte()
        STATE["passages"] = passages
        STATE["queries"] = queries
        STATE["qrels"] = qrels

        from src.run_day1 import get_preprocessed_corpus
        tokenized_corpus = get_preprocessed_corpus(passages)
        idx = InvertedIndex().fit(tokenized_corpus)
        STATE["tfidf"] = TFIDFRetriever(idx)
        STATE["bm25"] = BM25Retriever(idx, k1=1.5, b=0.75)

        # Dense retriever
        embeddings, doc_ids = encode_corpus(passages)
        STATE["dense"] = DenseRetriever(embeddings, doc_ids)

        STATE["ready"] = True
        print("LoTTE IR service startup complete and models loaded in memory.")
    except Exception as e:
        print(f"Warning during startup (models will be loaded on demand): {e}")
    yield
    print("Shutting down LoTTE IR service...")


app = FastAPI(title="LoTTE RAG IR Service", version="1.0.0", lifespan=lifespan)

# Enable CORS for local Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage loaded once at startup
STATE = {
    "passages": {},
    "queries": {},
    "qrels": {},
    "tfidf": None,
    "bm25": None,
    "dense": None,
    "ready": False
}


class SearchResult(BaseModel):
    rank: int
    pid: int
    score: float
    text: str


class RAGRequest(BaseModel):
    q: str
    ranker: str = "rrf"
    k: int = 5


class Citation(BaseModel):
    n: int
    pid: int
    text: str


class RAGResponse(BaseModel):
    answer: str
    citations: List[Citation]
    ranker: str



@app.get("/health")
def health():
    return {"status": "ok", "ready": STATE["ready"], "corpus_size": len(STATE["passages"])}


@app.get("/search", response_model=List[SearchResult])
def search(
    q: str = Query(..., description="Query text"),
    ranker: str = Query("rrf", description="Ranker algorithm: tfidf, bm25, dense, rrf"),
    k: int = Query(10, description="Top-k passages to return")
):
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    ranker = ranker.lower()
    q_tokens = tokenize(q)
    results_raw = []

    if ranker == "tfidf":
        if STATE["tfidf"] is None:
            raise HTTPException(status_code=503, detail="TF-IDF ranker not initialized.")
        results_raw = STATE["tfidf"].search(q_tokens, top_k=k)

    elif ranker == "bm25":
        if STATE["bm25"] is None:
            raise HTTPException(status_code=503, detail="BM25 ranker not initialized.")
        results_raw = STATE["bm25"].search(q_tokens, top_k=k)

    elif ranker == "dense":
        if STATE["dense"] is None:
            raise HTTPException(status_code=503, detail="Dense ranker not initialized.")
        results_raw = STATE["dense"].search(q, top_k=k)

    elif ranker == "rrf":
        if STATE["bm25"] is None or STATE["dense"] is None:
            raise HTTPException(status_code=503, detail="BM25 or Dense ranker not initialized.")
        bm25_res = STATE["bm25"].search(q_tokens, top_k=max(k, 100))
        dense_res = STATE["dense"].search(q, top_k=max(k, 100))
        results_raw = reciprocal_rank_fusion([bm25_res, dense_res], k=60, top_k=k)

    else:
        raise HTTPException(status_code=400, detail=f"Unknown ranker: {ranker}. Supported: tfidf, bm25, dense, rrf")

    passages = STATE["passages"]
    return [
        SearchResult(
            rank=idx + 1,
            pid=pid,
            score=round(float(score), 6),
            text=passages.get(pid, "")
        )
        for idx, (pid, score) in enumerate(results_raw)
    ]


@app.post("/rag", response_model=RAGResponse)
def rag_generate(req: RAGRequest):
    """
    RAG endpoint (configured for Claude API in Day 2).
    Retrieves top-k passages and formats context citations.
    """
    search_res = search(q=req.q, ranker=req.ranker, k=req.k)

    citations = [
        Citation(n=item.rank, pid=item.pid, text=item.text)
        for item in search_res
    ]

    # Placeholder logic until Day 2 Claude API integration
    answer = f"[Answer synthesized from top-{len(citations)} passages retrieved by {req.ranker}]"

    return RAGResponse(
        answer=answer,
        citations=citations,
        ranker=req.ranker
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
