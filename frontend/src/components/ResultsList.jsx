import React, { useState } from 'react';

function PassageCard({ item }) {
  const [expanded, setExpanded] = useState(false);
  const isLong = item.text.length > 250;
  const displayText = expanded || !isLong ? item.text : item.text.slice(0, 250) + '...';

  return (
    <div id={`passage-${item.rank}`} className="passage-card">
      <div className="passage-header">
        <div className="badge-group">
          <span className="rank-badge">#{item.rank}</span>
          <span className="pid-badge">PID: {item.pid}</span>
        </div>
        <span className="score-badge">Score: {item.score.toFixed(4)}</span>
      </div>

      <div className="passage-body">
        <p className="passage-text">{displayText}</p>
        {isLong && (
          <button
            type="button"
            className="expand-btn"
            onClick={() => setExpanded(!expanded)}
          >
            {expanded ? '▲ Collapse' : '▼ Read more'}
          </button>
        )}
      </div>
    </div>
  );
}

export default function ResultsList({ results, ranker, loading }) {
  if (loading) {
    return (
      <div className="results-loading-state">
        <div className="loading-card pulse"></div>
        <div className="loading-card pulse"></div>
        <div className="loading-card pulse"></div>
      </div>
    );
  }

  if (!results || results.length === 0) {
    return (
      <div className="empty-results-state">
        <p>No passages retrieved yet. Enter a query above or click a sample query to search.</p>
      </div>
    );
  }

  return (
    <div className="results-list-container">
      <div className="results-list-header">
        <h3>Retrieved Passages ({results.length})</h3>
        <span className="ranker-indicator">Ranker: {ranker.toUpperCase()}</span>
      </div>
      <div className="passages-grid">
        {results.map((item) => (
          <PassageCard key={item.pid} item={item} />
        ))}
      </div>
    </div>
  );
}
