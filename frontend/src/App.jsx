import React, { useState } from 'react';
import SearchBar from './components/SearchBar';
import ResultsList from './components/ResultsList';
import AnswerPanel from './components/AnswerPanel';
import './App.css';

export default function App() {
  const [query, setQuery] = useState('how much should i feed my 1 year old english mastiff?');
  const [ranker, setRanker] = useState('rrf');
  const [k, setK] = useState(10);
  const [enableRag, setEnableRag] = useState(true);

  const [results, setResults] = useState([]);
  const [ragData, setRagData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSearch = async () => {
    if (!query.trim()) return;

    setLoading(true);
    setError(null);
    setRagData(null);

    try {
      // 1. Fetch retrieval results from FastAPI
      const searchUrl = `/api/search?q=${encodeURIComponent(query.trim())}&ranker=${ranker}&k=${k}`;
      const searchRes = await fetch(searchUrl);

      if (!searchRes.ok) {
        const errDetail = await searchRes.json().catch(() => ({}));
        throw new Error(errDetail.detail || `Search failed with status ${searchRes.status}`);
      }

      const searchJson = await searchRes.json();
      setResults(searchJson);

      // 2. If RAG is enabled, trigger synthesis over top-5
      if (enableRag) {
        const ragRes = await fetch('/api/rag', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            q: query.trim(),
            ranker: ranker,
            k: 5
          })
        });

        if (ragRes.ok) {
          const ragJson = await ragRes.json();
          setRagData(ragJson);
        } else {
          console.warn('RAG synthesis endpoint failed, proceeding with search results.');
        }
      }
    } catch (err) {
      console.error('Retrieval error:', err);
      setError(err.message || 'Failed to connect to retrieval backend. Ensure api.py is running on port 8000.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-layout">
      {/* Top Banner / Navigation */}
      <header className="app-header">
        <div className="header-inner">
          <div className="brand-group">
            <span className="brand-badge">CS IR PROJECT</span>
            <h1 className="brand-title">LoTTE RAG IR Explorer</h1>
            <p className="brand-subtitle">
              Evaluating TF-IDF, BM25, Dense (all-MiniLM-L6-v2), and Hybrid RRF on 268,893 Lifestyle passages.
            </p>
          </div>
          <div className="corpus-meta-pill">
            <span className="meta-dot"></span>
            <span>268,893 Passages Indexed</span>
          </div>
        </div>
      </header>

      {/* Main Container */}
      <main className="main-content">
        <SearchBar
          query={query}
          setQuery={setQuery}
          ranker={ranker}
          setRanker={setRanker}
          k={k}
          setK={setK}
          onSearch={handleSearch}
          loading={loading}
          enableRag={enableRag}
          setEnableRag={setEnableRag}
        />

        {error && (
          <div className="error-alert">
            <span className="error-icon">⚠️</span>
            <div className="error-body">
              <strong>Retrieval Error</strong>
              <p>{error}</p>
            </div>
            <button className="dismiss-btn" onClick={() => setError(null)}>✕</button>
          </div>
        )}

        {enableRag && (
          <AnswerPanel ragData={ragData} loading={loading} />
        )}

        <ResultsList results={results} ranker={ranker} loading={loading} />
      </main>

      {/* Footer */}
      <footer className="app-footer">
        <div className="footer-inner">
          <p>LoTTE RAG IR System — Information Retrieval Course Benchmark Deliverable</p>
          <div className="footer-tags">
            <span className="tech-tag">FastAPI (port 8000)</span>
            <span className="tech-tag">Vite + React (port 5173)</span>
            <span className="tech-tag">FAISS FlatIP</span>
            <span className="tech-tag">RTX 4060 GPU</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
