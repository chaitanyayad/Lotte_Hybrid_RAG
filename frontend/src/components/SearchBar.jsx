import React from 'react';

const SAMPLE_QUERIES = [
  "how much should i feed my 1 year old english mastiff?",
  "are zebra loaches safe with shrimp?",
  "are serpae tetras fin nippers?",
  "are christmas trees poisonous to rats?",
  "is brake fluid water soluble?"
];

export default function SearchBar({
  query,
  setQuery,
  ranker,
  setRanker,
  k,
  setK,
  onSearch,
  loading,
  enableRag,
  setEnableRag
}) {
  const handleSubmit = (e) => {
    e.preventDefault();
    if (query.trim()) {
      onSearch();
    }
  };

  return (
    <div className="search-bar-container">
      <form onSubmit={handleSubmit} className="search-form">
        <div className="input-group">
          <span className="search-icon">🔍</span>
          <input
            id="query-input"
            type="text"
            className="search-input"
            placeholder="Search 268,893 LoTTE passages (e.g., 'how to feed english mastiff?')"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          {query && (
            <button
              type="button"
              className="clear-btn"
              onClick={() => setQuery('')}
              title="Clear input"
            >
              ✕
            </button>
          )}
        </div>

        <div className="controls-row">
          <div className="control-item">
            <label htmlFor="ranker-select">Algorithm:</label>
            <select
              id="ranker-select"
              value={ranker}
              onChange={(e) => setRanker(e.target.value)}
              className="ranker-select"
            >
              <option value="rrf">⚡ Hybrid RRF (BM25 + Dense, k=60)</option>
              <option value="dense">🧠 Dense (all-MiniLM-L6-v2)</option>
              <option value="bm25">📄 Okapi BM25 (k1=1.5, b=0.75)</option>
              <option value="tfidf">📊 Classic TF-IDF</option>
            </select>
          </div>

          <div className="control-item">
            <label htmlFor="topk-select">Top-K:</label>
            <select
              id="topk-select"
              value={k}
              onChange={(e) => setK(Number(e.target.value))}
              className="topk-select"
            >
              <option value={5}>5 passages</option>
              <option value={10}>10 passages</option>
              <option value={20}>20 passages</option>
            </select>
          </div>

          <label className="rag-toggle-label">
            <input
              id="rag-toggle"
              type="checkbox"
              checked={enableRag}
              onChange={(e) => setEnableRag(e.target.checked)}
            />
            <span className="rag-toggle-text">✨ Generate with RAG</span>
          </label>

          <button
            id="search-button"
            type="submit"
            className="search-submit-btn"
            disabled={loading || !query.trim()}
          >
            {loading ? <span className="spinner"></span> : 'Search & Retrieve'}
          </button>
        </div>
      </form>

      <div className="samples-row">
        <span className="sample-label">Sample queries:</span>
        {SAMPLE_QUERIES.map((sq, i) => (
          <button
            key={i}
            type="button"
            className="sample-btn"
            onClick={() => {
              setQuery(sq);
            }}
          >
            {sq}
          </button>
        ))}
      </div>
    </div>
  );
}
