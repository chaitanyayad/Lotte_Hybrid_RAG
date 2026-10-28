# LoTTE RAG — Information Retrieval Course Project

## Goal
Build an IR system comparing four retrieval methods on the LoTTE benchmark, plus a RAG layer on top of the best retriever. Deliverables: working code, a FastAPI backend with a Vite + React frontend, a demo notebook, and a report. Timeline: 2 days total. Hardware: local Windows laptop, RTX 4060 Laptop GPU (8GB VRAM), run everything locally — no Colab.

## Dataset
- **LoTTE** (from the ColBERTv2 repo), **lifestyle** domain, **dev** split, **search** queries.
- Files to expect: a passage collection, a queries file, and a qrels file (JSONL, answer pids per query) — **verify actual field names and format after download; don't assume.**
- First task: write a loader and print corpus size, query count, avg passage length, and relevant-passages-per-query. Put these in the report.
- Sanity check: every qrels pid must exist in the corpus.

## Required outputs of the loader
```
passages: dict[pid -> text]
queries:  dict[qid -> text]
qrels:    dict[qid -> set(pid)]
```

## Systems to implement (all four required by the course template)
1. **TF-IDF** (scikit-learn or hand-rolled)
2. **BM25** (own inverted index, ~40 lines, validated against `rank_bm25`; defaults k1=1.5, b=0.75 — no grid search, time-boxed)
3. **Dense retrieval** — `sentence-transformers/all-MiniLM-L6-v2`, encode locally on GPU (`batch_size=128`, `model.half()`, `normalize_embeddings=True`), FAISS flat index (CPU is fine at this corpus size), cosine similarity. Cache embeddings to disk (`np.save`) so they're never recomputed.
4. **Hybrid RRF** — `score(d) = Σ 1/(k + rank_i(d))`, k=60. No ablation: RRF fuses BM25 + Dense only (time-boxed).

Preprocessing (clean, tokenize, stop-words, stemming) applies **only** to the sparse models. Dense retrieval gets raw text. State this explicitly in the report.

## Evaluation
- Metrics: **P@10, Recall@100, MAP, nDCG@10** for every system.
- One precision-recall plot with all systems overlaid.
- One comparison table.
- 3–5 manually noted failure/success cases (BM25 wins vs dense wins) — capture these while running eval, don't reconstruct later.
- Use `ir_measures` or `pytrec_eval`.

## RAG layer (keep small)
- Take top-5 passages from the best retriever (expected: RRF).
- Prompt: "Answer only from the passages below; cite sources as [1]..[n]; say 'not found' if the answer isn't in the passages."
- Use the Claude API for generation.
- Small qualitative check only: 5 to 10 sample queries, compare answers when the gold passage is present vs missing from top-5. Note citation validity by eye. No formal groundedness metric — out of scope for a 2-day timeline.

## Demo interface: Vite + React over FastAPI
**Backend: `api.py`** (FastAPI + uvicorn, port 8000). Load indexes and cached embeddings once at startup and keep them in memory.
- `GET /search?q=<query>&ranker=<tfidf|bm25|dense|rrf>&k=10` returns `[{rank, pid, score, text}]`
- `POST /rag` with body `{q, ranker, k}` (k=5 default) returns `{answer, citations: [{n, pid, text}]}`

**Frontend: `frontend/`** (Vite + React, plain JS or TS, plain CSS, no UI framework, no router, no state library).
- Scaffold: `npm create vite@latest frontend -- --template react`
- Dev server on 5173 with a Vite proxy: `/api` -> `http://localhost:8000` (avoids CORS).
- Single page, three components: `SearchBar` (query input + ranker dropdown), `ResultsList` (rank, score, snippet with expand), `AnswerPanel` (RAG answer with [n] markers that jump to the cited passage).
- Include loading and error states. Nothing else.

**`demo.ipynb`:** runs the fixed demo queries, shows the four rankers' top-5 for one query side by side, the results table, and the P-R plot. This is also the fallback demo if the frontend is not ready.

## Explicitly cut (do not build)
- Cross-encoder reranking
- Query-type breakdown analysis
- Formal groundedness/entailment checker
- k1/b grid search
- Significance testing (paired t-test)
- RRF ablations and the TF-IDF-in-RRF variant
- Streamlit or Gradio (the frontend is Vite + React only)
- Routing, auth, state libraries, or UI component frameworks in the frontend
- Second dense model (bge-small) — only add if everything else finishes early

## Repo layout
```
ir-lotte-rag/
├── data/            # LoTTE lifestyle/dev (gitignored)
├── src/
│   ├── load.py
│   ├── preprocess.py
│   ├── sparse.py       # inverted index, TF-IDF, BM25
│   ├── dense.py         # encode + FAISS
│   ├── fusion.py         # RRF
│   ├── evaluate.py       # metrics + P-R plot
│   └── rag.py
├── api.py            # FastAPI backend
├── frontend/         # Vite + React app
├── demo.ipynb        # demo notebook
├── results/          # tables, plots, cached embeddings
└── report/
```

## 2-Day Schedule
Work in parallel: let the agent build while the encoding job and evaluation run.

**Day 1: core pipeline and evaluation (must finish today)**
1. Download LoTTE lifestyle/dev, inspect the real file format, write `load.py`, print dataset stats.
2. Write `preprocess.py` (sparse only), then start the local dense encoding job immediately (standalone script, GPU in Performance mode).
3. While it runs: `sparse.py` (small own inverted index, TF-IDF, BM25 with default k1/b).
4. When embeddings finish: `dense.py` (FAISS), `fusion.py` (RRF = BM25 + Dense, k=60).
5. `evaluate.py`: P@10, Recall@100, MAP, nDCG@10 for all four systems, comparison table, P-R plot. Save 3 failure/success cases.
6. Commit. At the end of Day 1 the four systems and all metrics must be done.

**Day 2: RAG, demo, report**
1. Morning: `rag.py`, `api.py`, minimal `frontend/`, `demo.ipynb`. Qualitative RAG check on 5 to 10 queries only.
2. Afternoon: write the report from the actual results (intro, related work, dataset, method, setup, results, analysis, limitations, conclusion). Run the similarity check.
3. Evening: pick the demo queries, one full dry run, record a backup screen capture.

## Non-negotiables
- Start local embedding encoding as soon as `load.py` and the corpus text are ready. It is the longest wait.
- Cache every expensive computation (embeddings, rankings) to disk.
- Cut order if time runs out: (1) `frontend/` and `api.py`, keeping `demo.ipynb`; (2) RAG qualitative check, keeping a single static RAG example; (3) the second-system failure cases. Never cut the four required systems, the four required metrics, the comparison table, or the P-R plot.

## Environment
- Windows, RTX 4060 Laptop GPU, 8GB VRAM — set GPU to Performance mode (not Quiet) before encoding runs; thermal throttling has been an issue before.
- `pip install sentence-transformers faiss-cpu rank_bm25 nltk ir_measures fastapi uvicorn jupyter`
- PyTorch: CUDA build matching the installed driver. Verify with `torch.cuda.is_available() == True`.
- Node 18+ and npm are required for the frontend.
- Run encoding as a standalone script, not inside a notebook cell.

## Demo script (~4 min)
1. Dataset stats (notebook).
2. One query through all four rankers (notebook).
3. Results table + P-R plot.
4. RAG answer with citations (web UI live).
5. One failure case, explained.
