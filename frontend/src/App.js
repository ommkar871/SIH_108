import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
  // State management
  const [query, setQuery] = useState('');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [topK, setTopK] = useState(3);
  const [selectedDomain, setSelectedDomain] = useState('');
  const [domains, setDomains] = useState([]);
  const [stats, setStats] = useState(null);

  // Load domains and stats on mount
  useEffect(() => {
    const fetchDomains = async () => {
      try {
        const response = await fetch('http://localhost:8000/domains');
        const data = await response.json();
        setDomains(data.domains_list || []);
      } catch (err) {
        console.log('Could not load domains');
      }
    };

    const fetchStats = async () => {
      try {
        const response = await fetch('http://localhost:8000/stats');
        const data = await response.json();
        setStats(data);
      } catch (err) {
        console.log('Could not load stats');
      }
    };

    fetchDomains();
    fetchStats();
  }, []);

  const handleSearch = async (e) => {
    e.preventDefault();

    if (!query.trim()) {
      setError('Please enter a search query');
      return;
    }

    setLoading(true);
    setError(null);
    setResults(null);

    try {
      let url = `http://localhost:8000/search?query=${encodeURIComponent(
        query
      )}&top_k=${topK}`;

      if (selectedDomain) {
        url = `http://localhost:8000/search-by-domain?query=${encodeURIComponent(
          query
        )}&domain=${encodeURIComponent(selectedDomain)}&top_k=${topK}`;
      }

      const response = await fetch(url);

      if (!response.ok) throw new Error('Search failed');

      const data = await response.json();
      setResults(data);
    } catch (err) {
      setError(
        'Error connecting to backend. Make sure it\'s running on port 8000.'
      );
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleExportPDF = (result) => {
    const content = `
===========================================
BIS STANDARD MATCH REPORT
===========================================

MATCHED STANDARD:
${result.is_code} - ${result.title}

MATCH SCORE: ${result.match_score}%
CONFIDENCE: ${result.match_confidence}

DESCRIPTION:
${result.description}

DOMAIN: ${result.domain}

EXPLANATION:
${result.explanation}

KEYWORDS:
${result.keywords}

GENERATED: ${new Date().toLocaleString()}
===========================================
    `;

    const element = document.createElement('a');
    element.setAttribute(
      'href',
      'data:text/plain;charset=utf-8,' + encodeURIComponent(content)
    );
    element.setAttribute('download', `${result.is_code}_Match_Report.txt`);
    element.style.display = 'none';
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  const handleCopyCode = (isCode) => {
    navigator.clipboard.writeText(isCode);
    alert(`Copied: ${isCode}`);
  };

  return (
    <div className="app">
      <header className="header">
        <div className="header-content">
          <h1>🔍 BIS-Sync</h1>
          <p>Smart Indian Standards Finder</p>
          <span className="subtitle">
            Find the right standard in seconds, not days
          </span>
        </div>
      </header>

      <main className="main">
        {/* Statistics Banner */}
        {stats && (
          <div className="stats-banner">
            <div className="stat-item">
              <span className="stat-number">{stats.total_standards}</span>
              <span className="stat-label">Standards Available</span>
            </div>
            <div className="stat-item">
              <span className="stat-number">{stats.domains}</span>
              <span className="stat-label">Domains Covered</span>
            </div>
            <div className="stat-item">
              <span className="stat-number">{stats.model_dimensions}</span>
              <span className="stat-label">Semantic Dimensions</span>
            </div>
          </div>
        )}

        {/* Search Form */}
        <form onSubmit={handleSearch} className="search-form">
          <div className="search-input-container">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Describe what you need... (e.g., 'galvanized steel pipe for drinking water, 100mm')"
              className="search-input"
              disabled={loading}
            />
            <button
              type="submit"
              disabled={loading}
              className="search-button"
            >
              {loading ? (
                <>
                  <span className="spinner"></span> Searching...
                </>
              ) : (
                '🔍 Search'
              )}
            </button>
          </div>

          {/* Filters */}
          <div className="filters">
            <div className="filter-group">
              <label>Results to show:</label>
              <select
                value={topK}
                onChange={(e) => setTopK(parseInt(e.target.value))}
                disabled={loading}
              >
                <option value={1}>1 Result</option>
                <option value={3}>3 Results</option>
                <option value={5}>5 Results</option>
                <option value={10}>10 Results</option>
              </select>
            </div>

            {domains.length > 0 && (
              <div className="filter-group">
                <label>Filter by domain:</label>
                <select
                  value={selectedDomain}
                  onChange={(e) => setSelectedDomain(e.target.value)}
                  disabled={loading}
                >
                  <option value="">All Domains</option>
                  {domains.map((domain) => (
                    <option key={domain} value={domain}>
                      {domain}
                    </option>
                  ))}
                </select>
              </div>
            )}
          </div>
        </form>

        {/* Error Message */}
        {error && (
          <div className="error-message">
            <span>⚠️</span>
            {error}
          </div>
        )}

        {/* Results */}
        {results && (
          <div className="results-section">
            <h2 className="results-title">
              Results for: <em>"{results.query}"</em>
            </h2>

            {results.results && results.results.length > 0 ? (
              <div className="results-list">
                {results.results.map((result, idx) => (
                  <div key={idx} className="result-card">
                    <div className="result-header">
                      <div className="result-left">
                        <span
                          className="is-code"
                          onClick={() => handleCopyCode(result.is_code)}
                          title="Click to copy"
                        >
                          {result.is_code}
                        </span>
                        <span className="result-rank">Rank {result.rank}</span>
                      </div>
                      <div className="result-right">
                        <span
                          className={`match-score match-${
                            result.match_score >= 80
                              ? 'high'
                              : result.match_score >= 60
                              ? 'medium'
                              : 'low'
                          }`}
                        >
                          {result.match_score}% Match
                        </span>
                        <span className="confidence">
                          {result.match_confidence}
                        </span>
                      </div>
                    </div>

                    <h3 className="result-title">{result.title}</h3>

                    <p className="result-description">{result.description}</p>

                    <div className="result-tags">
                      <span className="domain-tag">{result.domain}</span>
                    </div>

                    <div className="result-explanation">
                      <strong>Why it matched:</strong> {result.explanation}
                    </div>

                    <div className="result-actions">
                      <button
                        className="action-btn export-btn"
                        onClick={() => handleExportPDF(result)}
                      >
                        📥 Export Report
                      </button>
                      <button
                        className="action-btn view-btn"
                        onClick={() =>
                          alert(
                            'In production, this would link to the full BIS PDF.'
                          )
                        }
                      >
                        📖 View Full Standard
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="no-results">
                <p>No standards found matching your query.</p>
                <p className="hint">
                  Try using more specific technical terms or different keywords.
                </p>
              </div>
            )}
          </div>
        )}

        {/* Welcome / Empty State */}
        {!results && !loading && !error && (
          <div className="welcome-section">
            <div className="welcome-card">
              <h2>How BIS-Sync Works</h2>
              <ol className="welcome-steps">
                <li>
                  <strong>📝 Describe</strong> — Tell us what you need in plain
                  language
                </li>
                <li>
                  <strong>🧠 Understand</strong> — AI grasps the meaning, not
                  just keywords
                </li>
                <li>
                  <strong>⚡ Find</strong> — Get the exact Indian Standard
                  instantly
                </li>
                <li>
                  <strong>✅ Export</strong> — Download compliance report to
                  submit
                </li>
              </ol>
            </div>

            <div className="example-queries">
              <h3>Example Searches:</h3>
              <div className="example-list">
                <button
                  className="example-btn"
                  onClick={() => {
                    setQuery(
                      'Galvanized steel pipes for underground drinking water supply, 100mm diameter'
                    );
                  }}
                >
                  🔧 Water Pipe
                </button>
                <button
                  className="example-btn"
                  onClick={() => {
                    setQuery(
                      'Steel reinforcement bars for concrete construction, 12mm'
                    );
                  }}
                >
                  🏗️ Reinforcement Bar
                </button>
                <button
                  className="example-btn"
                  onClick={() => {
                    setQuery(
                      'Copper conductor wires for building electrical wiring, 2.5mm'
                    );
                  }}
                >
                  ⚡ Copper Wire
                </button>
              </div>
            </div>

            {domains.length > 0 && (
              <div className="domains-info">
                <h3>Available Domains:</h3>
                <div className="domain-pills">
                  {domains.map((domain) => (
                    <span key={domain} className="domain-pill">
                      {domain}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </main>

      <footer className="footer">
        <p>
          BIS-Sync • All rights reserved @2026
        </p>
        <p className="footer-note">
          💡 Tip: Use specific technical terms for better results
        </p>
      </footer>
    </div>
  );
}

export default App;
