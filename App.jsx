import React, { useState } from 'react';
import { AlertCircle, CheckCircle, Search, Loader, BarChart3, Zap } from 'lucide-react';
import './App.css';

export default function App() {
  const [inputText, setInputText] = useState('');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [batchMode, setBatchMode] = useState(false);
  const [batchInput, setBatchInput] = useState('');

  const analyzeNews = async (text) => {
    if (!text.trim()) {
      setError('Please enter some text to analyze');
      return;
    }

    setLoading(true);
    setError('');
    try {
      const response = await fetch('http://localhost:5000/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ texts: [text] })
      });

      if (!response.ok) throw new Error('API request failed');
      const data = await response.json();
      setResults(data.results[0]);
    } catch (err) {
      setError('Failed to connect to API. Make sure Flask is running on port 5000.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const analyzeBatch = async () => {
    const texts = batchInput.split('\n').filter(t => t.trim());
    if (texts.length === 0) {
      setError('Please enter at least one headline');
      return;
    }

    setLoading(true);
    setError('');
    try {
      const response = await fetch('http://localhost:5000/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ texts })
      });

      if (!response.ok) throw new Error('API request failed');
      const data = await response.json();
      setResults(data.results);
    } catch (err) {
      setError('Failed to connect to API. Make sure Flask is running on port 5000.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (batchMode) {
      analyzeBatch();
    } else {
      analyzeNews(inputText);
    }
  };

  return (
    <div className="app-container">
      {/* Animated Background */}
      <div className="animated-bg"></div>
      <div className="animated-bg-2"></div>

      {/* Header */}
      <header className="header">
        <div className="header-content">
          <div className="header-top">
            <div className="logo-section">
              <div className="logo-icon">
                <Search size={32} />
              </div>
              <div>
                <h1 className="main-title">Fake News Detector</h1>
                <p className="subtitle">AI-Powered Misinformation Detection</p>
              </div>
            </div>
            <div className="badge">v1.0 • ML Powered</div>
          </div>
        </div>
      </header>

      <main className="main-content">
        {/* Features Grid */}
        <div className="features-grid">
          <div className="feature-card">
            <BarChart3 size={24} />
            <h3>92% Accuracy</h3>
            <p>Advanced ML ensemble</p>
          </div>
          <div className="feature-card">
            <Zap size={24} />
            <h3>Instant Results</h3>
            <p>Real-time analysis</p>
          </div>
          <div className="feature-card">
            <CheckCircle size={24} />
            <h3>Reliable</h3>
            <p>Multi-model validation</p>
          </div>
        </div>

        {/* Mode Toggle */}
        <div className="mode-toggle-container">
          <div className="mode-toggle">
            <button
              onClick={() => { setBatchMode(false); setResults(null); setError(''); }}
              className={`toggle-btn ${!batchMode ? 'active' : ''}`}
            >
              <Search size={18} />
              Single Analysis
            </button>
            <button
              onClick={() => { setBatchMode(true); setResults(null); setError(''); }}
              className={`toggle-btn ${batchMode ? 'active' : ''}`}
            >
              <BarChart3 size={18} />
              Batch Check
            </button>
          </div>
        </div>

        {/* Input Section */}
        <form onSubmit={handleSubmit} className="form-container">
          <div className="form-card">
            <div className="form-header">
              <label className="form-label">
                {batchMode ? '📋 Enter Multiple Headlines' : '📰 Enter News Headline'}
              </label>
              <span className="char-count">
                {batchMode ? batchInput.length : inputText.length} characters
              </span>
            </div>
            <textarea
              value={batchMode ? batchInput : inputText}
              onChange={(e) => batchMode ? setBatchInput(e.target.value) : setInputText(e.target.value)}
              placeholder={batchMode 
                ? "Paste multiple headlines here...\nOne headline per line\nExample:\nTrump wins election\nNew vaccine approved" 
                : "Enter a news headline or article text..."}
              className="textarea-input"
            />
            <button
              type="submit"
              disabled={loading}
              className="submit-btn"
            >
              {loading ? (
                <>
                  <Loader size={20} className="spinner" />
                  Analyzing...
                </>
              ) : (
                <>
                  <Search size={20} />
                  Analyze News
                </>
              )}
            </button>
          </div>
        </form>

        {/* Error Message */}
        {error && (
          <div className="error-box">
            <AlertCircle size={20} />
            <p>{error}</p>
          </div>
        )}

        {/* Results Section */}
        {results && !loading && (
          <div className="results-section">
            {Array.isArray(results) ? (
              <>
                {/* Stats Cards */}
                <div className="stats-container">
                  <div className="stat-card total">
                    <div className="stat-number">{results.length}</div>
                    <div className="stat-label">Total Analyzed</div>
                  </div>
                  <div className="stat-card fake">
                    <div className="stat-number">{results.filter(r => r.is_fake).length}</div>
                    <div className="stat-label">Likely Fake</div>
                  </div>
                  <div className="stat-card real">
                    <div className="stat-number">{results.filter(r => !r.is_fake).length}</div>
                    <div className="stat-label">Likely Real</div>
                  </div>
                </div>

                {/* Results List */}
                <div className="results-list">
                  {results.map((result, idx) => (
                    <ResultCard key={idx} result={result} index={idx} />
                  ))}
                </div>
              </>
            ) : (
              <ResultCard result={results} index={0} />
            )}
          </div>
        )}

        {/* Empty State */}
        {!results && !loading && !error && (
          <div className="empty-state">
            <div className="empty-icon">
              <Search size={64} />
            </div>
            <h2>Ready to Detect Fake News?</h2>
            <p>Enter a headline or article text above to get started</p>
            <div className="empty-tips">
              <div className="tip">💡 Try: "Trump wins election"</div>
              <div className="tip">💡 Try: "Scientists discover cure"</div>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="footer">
        <p>🔒 Your data is processed locally | ML Powered Fake News Detection</p>
      </footer>
    </div>
  );
}

function ResultCard({ result, index }) {
  const isFake = result.is_fake;
  const confidence = Math.round(result.confidence * 100);

  return (
    <div className={`result-card ${isFake ? 'fake' : 'real'}`}>
      <div className="result-header">
        <div className="result-badge">
          {isFake ? (
            <>
              <AlertCircle size={24} />
              <span>LIKELY FAKE</span>
            </>
          ) : (
            <>
              <CheckCircle size={24} />
              <span>LIKELY REAL</span>
            </>
          )}
        </div>
        <span className="result-index">#{index + 1}</span>
      </div>

      <p className="result-text">{result.text.substring(0, 250)}...</p>

      <div className="result-footer">
        <div className="confidence-section">
          <label>Confidence Score</label>
          <div className="confidence-bar">
            <div className="confidence-fill" style={{ width: `${confidence}%` }}></div>
          </div>
          <span className="confidence-value">{confidence}%</span>
        </div>

        <div className="recommendation-section">
          <label>Status</label>
          <p className="recommendation">
            {isFake ? '⚠️ Verify with trusted sources' : '✅ Likely credible'}
          </p>
        </div>
      </div>
    </div>
  );
}