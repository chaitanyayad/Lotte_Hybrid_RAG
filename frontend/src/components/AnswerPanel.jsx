import React from 'react';

export default function AnswerPanel({ ragData, loading }) {
  if (loading) {
    return (
      <div className="answer-panel-loading">
        <div className="answer-header-skeleton pulse"></div>
        <div className="answer-text-skeleton pulse"></div>
      </div>
    );
  }

  if (!ragData) return null;

  const scrollToPassage = (num) => {
    const el = document.getElementById(`passage-${num}`);
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'center' });
      el.classList.add('highlight-passage');
      setTimeout(() => {
        el.classList.remove('highlight-passage');
      }, 2000);
    }
  };

  // Render text with clickable [n] citation buttons
  const renderFormattedAnswer = (text) => {
    const parts = text.split(/(\[\d+\])/g);
    return parts.map((part, index) => {
      const match = part.match(/^\[(\d+)\]$/);
      if (match) {
        const num = match[1];
        return (
          <button
            key={index}
            type="button"
            className="citation-pill"
            onClick={() => scrollToPassage(num)}
            title={`Jump to retrieved passage #${num}`}
          >
            [{num}]
          </button>
        );
      }
      return <span key={index}>{part}</span>;
    });
  };

  const isNotFound = ragData.answer.trim().toLowerCase() === 'not found';

  return (
    <div className={`answer-panel-card ${isNotFound ? 'not-found-style' : ''}`}>
      <div className="answer-panel-header">
        <div className="title-row">
          <span className="sparkle-icon">✨</span>
          <h4>Grounded RAG Answer</h4>
          <span className="rag-badge">Synthesized via {ragData.ranker.toUpperCase()}</span>
        </div>
        <span className="citation-hint">Click any [n] marker to jump to cited evidence</span>
      </div>

      <div className="answer-content">
        <p className="answer-text">
          {renderFormattedAnswer(ragData.answer)}
        </p>
      </div>

      {ragData.citations && ragData.citations.length > 0 && (
        <div className="citations-summary-footer">
          <span className="footer-label">Sources Referenced:</span>
          <div className="sources-pills">
            {ragData.citations.map((c) => (
              <button
                key={c.n}
                type="button"
                className="source-quick-btn"
                onClick={() => scrollToPassage(c.n)}
              >
                [{c.n}] PID {c.pid}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
