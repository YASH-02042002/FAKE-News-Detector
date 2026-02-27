import React, { useState } from 'react';
import { AlertCircle, CheckCircle, Search, Loader, TrendingDown, TrendingUp } from 'lucide-react';

export default function FakeNewsDetector() {
  const [inputText, setInputText] = useState('');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [apiUrl] = useState('http://localhost:5000');
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
      const response = await fetch(`${apiUrl}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ texts: [text] })
      });

      if (!response.ok) throw new Error('API request failed');
      const data = await response.json();
      setResults(data.results[0]);
    } catch (err) {
      setError('Failed to connect to API. Make sure the server is running on port 5000.');
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
      const response = await fetch(`${apiUrl}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ texts })
      });

      if (!response.ok) throw new Error('API request failed');
      const data = await response.json();
      setResults(data.results);
    } catch (err) {
      setError('Failed to connect to API. Make sure the server is running on port 5000.');
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
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      {/* Header */}
      <header className="bg-gradient-to-r from-blue-600 to-cyan-600 shadow-lg">
        <div className="max-w-6xl mx-auto px-4 py-8">
          <div className="flex items-center gap-3 mb-2">
            <Search className="w-8 h-8 text-white" />
            <h1 className="text-4xl font-bold text-white">Fake News Detector</h1>
          </div>
          <p className="text-blue-100">AI-powered system to verify news credibility</p>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 py-12">
        {/* Mode Toggle */}
        <div className="flex gap-4 mb-8 justify-center">
          <button
            onClick={() => { setBatchMode(false); setResults(null); setError(''); }}
            className={`px-6 py-2 rounded-lg font-semibold transition ${
              !batchMode
                ? 'bg-blue-600 text-white'
                : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
            }`}
          >
            Single Article
          </button>
          <button
            onClick={() => { setBatchMode(true); setResults(null); setError(''); }}
            className={`px-6 py-2 rounded-lg font-semibold transition ${
              batchMode
                ? 'bg-blue-600 text-white'
                : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
            }`}
          >
            Batch Analysis
          </button>
        </div>

        {/* Input Section */}
        <form onSubmit={handleSubmit} className="mb-8">
          <div className="bg-slate-800 rounded-xl p-6 shadow-xl">
            <label className="block text-white text-lg font-semibold mb-4">
              {batchMode ? 'Enter Headlines (one per line):' : 'Enter News Headline or Article:'}
            </label>
            <textarea
              value={batchMode ? batchInput : inputText}
              onChange={(e) => batchMode ? setBatchInput(e.target.value) : setInputText(e.target.value)}
              placeholder={batchMode 
                ? "Trump wins election\nNew climate study released\nCelebrity announces..." 
                : "Enter a news headline or article text..."}
              className="w-full h-32 p-4 bg-slate-700 text-white rounded-lg border border-slate-600 focus:border-blue-500 focus:outline-none resize-none placeholder-slate-400"
            />
            <button
              type="submit"
              disabled={loading}
              className="mt-4 w-full bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700 disabled:opacity-50 text-white font-bold py-3 px-6 rounded-lg transition flex items-center justify-center gap-2"
            >
              {loading ? (
                <>
                  <Loader className="w-5 h-5 animate-spin" />
                  Analyzing...
                </>
              ) : (
                <>
                  <Search className="w-5 h-5" />
                  Analyze News
                </>
              )}
            </button>
          </div>
        </form>

        {/* Error Message */}
        {error && (
          <div className="bg-red-900 border-l-4 border-red-600 p-4 rounded-lg mb-8">
            <p className="text-red-100">{error}</p>
          </div>
        )}

        {/* Results Section */}
        {results && !loading && (
          <div className="space-y-6">
            {Array.isArray(results) ? (
              <>
                <div className="bg-slate-800 rounded-xl p-6 shadow-xl">
                  <h2 className="text-2xl font-bold text-white mb-4">Batch Analysis Results</h2>
                  <div className="grid grid-cols-3 gap-4 mb-6">
                    <div className="bg-slate-700 p-4 rounded-lg">
                      <p className="text-slate-300 text-sm">Total Analyzed</p>
                      <p className="text-3xl font-bold text-white">{results.length}</p>
                    </div>
                    <div className="bg-red-900 p-4 rounded-lg">
                      <p className="text-red-200 text-sm">Likely Fake</p>
                      <p className="text-3xl font-bold text-red-400">
                        {results.filter(r => r.is_fake).length}
                      </p>
                    </div>
                    <div className="bg-green-900 p-4 rounded-lg">
                      <p className="text-green-200 text-sm">Likely Real</p>
                      <p className="text-3xl font-bold text-green-400">
                        {results.filter(r => !r.is_fake).length}
                      </p>
                    </div>
                  </div>
                </div>

                {results.map((result, idx) => (
                  <ResultCard key={idx} result={result} index={idx} />
                ))}
              </>
            ) : (
              <ResultCard result={results} index={0} />
            )}
          </div>
        )}

        {/* Empty State */}
        {!results && !loading && !error && (
          <div className="text-center py-16 text-slate-400">
            <Search className="w-16 h-16 mx-auto mb-4 opacity-30" />
            <p className="text-lg">Enter a headline or article to get started</p>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="mt-16 bg-slate-800 border-t border-slate-700 py-6">
        <div className="max-w-6xl mx-auto px-4 text-center text-slate-400 text-sm">
          <p>Fake News Detection System v1.0</p>
          <p className="mt-2">Powered by Machine Learning | Multiple Classifier Ensemble</p>
        </div>
      </footer>
    </div>
  );
}

function ResultCard({ result, index }) {
  const isFake = result.is_fake;
  const confidence = Math.round(result.confidence * 100);

  return (
    <div className={`rounded-xl p-6 shadow-xl border-l-4 ${
      isFake
        ? 'bg-red-950 border-red-600'
        : 'bg-green-950 border-green-600'
    }`}>
      <div className="flex items-start gap-4">
        {isFake ? (
          <AlertCircle className="w-8 h-8 text-red-400 flex-shrink-0 mt-1" />
        ) : (
          <CheckCircle className="w-8 h-8 text-green-400 flex-shrink-0 mt-1" />
        )}
        
        <div className="flex-grow">
          <div className="flex items-center justify-between mb-3">
            <h3 className={`text-xl font-bold ${
              isFake ? 'text-red-400' : 'text-green-400'
            }`}>
              {isFake ? '⚠️ LIKELY FAKE' : '✅ LIKELY REAL'}
            </h3>
            <span className="text-sm font-semibold px-3 py-1 rounded-full bg-slate-700 text-white">
              #{index + 1}
            </span>
          </div>

          <p className="text-slate-200 mb-4 line-clamp-2">{result.text}</p>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-slate-400 text-sm mb-2">Confidence Score</p>
              <div className="flex items-center gap-2">
                <div className="flex-grow bg-slate-700 rounded-full h-2">
                  <div
                    className={`h-2 rounded-full transition-all ${
                      isFake ? 'bg-red-500' : 'bg-green-500'
                    }`}
                    style={{ width: `${confidence}%` }}
                  />
                </div>
                <span className="text-white font-bold text-lg w-12 text-right">{confidence}%</span>
              </div>
            </div>

            <div>
              <p className="text-slate-400 text-sm mb-2">Recommendation</p>
              <p className="text-slate-200 font-semibold">
                {isFake ? 'Verify with trusted sources' : 'Likely credible'}
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}