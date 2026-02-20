import { useState } from 'react'
import { predict } from './api'
import type { PredictRequest, PredictResponse } from './types'
import { sampleTransaction, getRandomLegitimateSample, getRandomFraudSample } from './sampleData'
import './App.css'

const FEATURE_ORDER = [
  'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
  'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
  'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount',
] as const

function App() {
  const [form, setForm] = useState<PredictRequest>(() => sampleTransaction)
  const [result, setResult] = useState<PredictResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showAdvanced, setShowAdvanced] = useState(false)

  const update = (key: keyof PredictRequest, value: number) => {
    setForm((f) => ({ ...f, [key]: value }))
    setError(null)
  }

  const loadSample = (sample: PredictRequest) => {
    setForm(sample)
    setResult(null)
    setError(null)
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const res = await predict(form)
      setResult(res)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Request failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <header className="header">
        <h1>Fraud Detection</h1>
        <p>Transaction risk score (0–100) from ensemble model</p>
      </header>

      <main className="main">
        <section className="card form-card">
          <h2>Transaction input</h2>
          <div className="sample-buttons">
            <button type="button" className="btn btn-secondary" onClick={() => loadSample(getRandomLegitimateSample())}>
              Load legitimate sample
            </button>
            <button type="button" className="btn btn-secondary" onClick={() => loadSample(getRandomFraudSample())}>
              Load fraud sample
            </button>
          </div>

          <form onSubmit={handleSubmit}>
            <div className="form-row">
              <label>
                <span>Time (sec)</span>
                <input
                  type="number"
                  step="any"
                  value={form.Time}
                  onChange={(e) => update('Time', Number(e.target.value))}
                />
              </label>
              <label>
                <span>Amount</span>
                <input
                  type="number"
                  step="0.01"
                  value={form.Amount}
                  onChange={(e) => update('Amount', Number(e.target.value))}
                />
              </label>
            </div>

            <button type="button" className="link" onClick={() => setShowAdvanced((s) => !s)}>
              {showAdvanced ? 'Hide' : 'Show'} V1–V28 (PCA features)
            </button>

            {showAdvanced && (
              <div className="v-grid">
                {FEATURE_ORDER.filter((k) => k !== 'Time' && k !== 'Amount').map((key) => (
                  <label key={key} className="v-label">
                    <span>{key}</span>
                    <input
                      type="number"
                      step="any"
                      value={form[key]}
                      onChange={(e) => update(key, Number(e.target.value))}
                    />
                  </label>
                ))}
              </div>
            )}

            <button type="submit" className="btn btn-primary" disabled={loading}>
              {loading ? 'Checking…' : 'Check risk'}
            </button>
          </form>

          {error && <p className="error">{error}</p>}
        </section>

        {result && (
          <section className="card result-card">
            <h2>Result</h2>
            <div className={`badge badge-${result.final_decision.toLowerCase()}`}>
              {result.final_decision}
            </div>
            <dl className="result-details">
              <dt>Fraud probability</dt>
              <dd>{(result.fraud_probability * 100).toFixed(2)}%</dd>
              <dt>Threshold used</dt>
              <dd>{result.threshold_used.toFixed(3)}</dd>
              <dt>Model</dt>
              <dd>{result.model}</dd>
            </dl>
          </section>
        )}
      </main>

      <footer className="footer">
        <p>API: POST /predict with Time, V1–V28, Amount. Backend must be running on port 5000.</p>
      </footer>
    </div>
  )
}

export default App
