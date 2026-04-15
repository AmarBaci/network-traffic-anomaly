import { useMemo, useState } from 'react'
import './App.css'

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'

function formatNumber(value, digits = 4) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return 'N/A'
  }
  return Number(value).toFixed(digits)
}

function formatCount(value) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return 'N/A'
  }
  return Number(value).toLocaleString('en-US')
}

function App() {
  const [payload, setPayload] = useState(null)
  const [loading, setLoading] = useState(false)
  const [refreshing, setRefreshing] = useState(false)
  const [error, setError] = useState('')

  const bestModel = payload?.summary?.best_model_by_f1 || 'N/A'
  const totalRuns = payload?.source?.total_runs || 0
  const latestModels = payload?.latest_models || []
  const recentRuns = payload?.recent_runs || []
  const quality = payload?.data_quality || {}

  function toReadableFetchError(err, method, endpoint) {
    if (err instanceof TypeError) {
      return `${method} ${endpoint} failed: could not reach API at ${API_BASE}. Make sure the backend server is running.`
    }
    if (err instanceof Error) {
      return err.message
    }
    return `${method} ${endpoint} failed with an unknown error.`
  }

  const metricCards = useMemo(() => {
    if (!latestModels.length) {
      return []
    }

    const best = [...latestModels].sort((a, b) => (b.f1 || -1) - (a.f1 || -1))[0]
    return [
      { label: 'Best Model (F1)', value: best?.model || 'N/A' },
      { label: 'Best F1', value: formatNumber(best?.f1) },
      { label: 'Best ROC-AUC', value: formatNumber(best?.roc_auc) },
      { label: 'Best PR-AUC', value: formatNumber(best?.pr_auc) },
      { label: 'Total Runs Logged', value: formatCount(totalRuns) },
      { label: 'Models Tracked', value: formatCount(payload?.summary?.model_count) },
    ]
  }, [latestModels, payload, totalRuns])

  async function fetchDashboardData() {
    setLoading(true)
    setError('')
    try {
      const response = await fetch(`${API_BASE}/dashboard/data`)
      if (!response.ok) {
        throw new Error(`GET /dashboard/data failed with status ${response.status}`)
      }
      const data = await response.json()
      setPayload(data)
    } catch (err) {
      setError(toReadableFetchError(err, 'GET', '/dashboard/data'))
    } finally {
      setLoading(false)
    }
  }

  async function refreshBackend(skipTrain) {
    setRefreshing(true)
    setError('')
    try {
      const response = await fetch(
        `${API_BASE}/dashboard/refresh?skip_train=${skipTrain ? 'true' : 'false'}`,
        { method: 'POST' },
      )
      if (!response.ok) {
        throw new Error(`POST /dashboard/refresh failed with status ${response.status}`)
      }
      await fetchDashboardData()
    } catch (err) {
      setError(toReadableFetchError(err, 'POST', '/dashboard/refresh'))
    } finally {
      setRefreshing(false)
    }
  }

  const preprocess = quality?.preprocess_report || {}
  const schema = quality?.schema_report || {}
  const qualitySummary = quality?.quality_summary || {}

  return (
    <div className="page">
      <header className="hero">
        <div>
          <p className="eyebrow">Network Traffic Dashboard</p>
          <h1>Baseline Monitoring Console</h1>
          <p className="subtitle">
            Live metrics, quality checks, and run history from your anomaly detection pipeline.
          </p>
        </div>
        <div className="hero-actions">
          <button onClick={fetchDashboardData} disabled={loading || refreshing}>
            {loading ? 'Loading...' : 'Load Data'}
          </button>
          <button onClick={() => refreshBackend(true)} disabled={loading || refreshing}>
            {refreshing ? 'Refreshing...' : 'Quick Refresh'}
          </button>
          <button className="danger" onClick={() => refreshBackend(false)} disabled={loading || refreshing}>
            Full Refresh
          </button>
        </div>
      </header>

      <section className="summary-grid">
        {metricCards.map((card) => (
          <article key={card.label} className="card">
            <h3>{card.label}</h3>
            <p>{card.value}</p>
          </article>
        ))}
        {!metricCards.length && (
          <article className="card empty">
            <h3>No data loaded</h3>
            <p>Click Load Data to pull results from the backend.</p>
          </article>
        )}
      </section>

      {error && <section className="error">{error}</section>}

      <section className="panel">
        <h2>Latest Models</h2>
        <table>
          <thead>
            <tr>
              <th>Model</th>
              <th>Run ID</th>
              <th>ROC-AUC</th>
              <th>PR-AUC</th>
              <th>F1</th>
              <th>FPR</th>
              <th>Samples</th>
              <th>Anomalies</th>
            </tr>
          </thead>
          <tbody>
            {latestModels.map((row) => (
              <tr key={row.run_id}>
                <td>{row.model}</td>
                <td>{row.run_id}</td>
                <td>{formatNumber(row.roc_auc)}</td>
                <td>{formatNumber(row.pr_auc)}</td>
                <td>{formatNumber(row.f1)}</td>
                <td>{formatNumber(row.fpr)}</td>
                <td>{formatCount(row.n_samples)}</td>
                <td>{formatCount(row.n_anomalies)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <section className="panel split">
        <article>
          <h2>Leaderboard</h2>
          <ol>
            {(payload?.leaderboard || []).map((row) => (
              <li key={row.run_id}>
                <strong>{row.model}</strong> ({row.run_id}) - F1 {formatNumber(row.f1)}
              </li>
            ))}
          </ol>
        </article>
        <article>
          <h2>Recent Runs</h2>
          <ul>
            {recentRuns.map((row) => (
              <li key={row.run_id}>
                {row.model} | {row.run_id} | F1 {formatNumber(row.f1)} | ROC-AUC {formatNumber(row.roc_auc)}
              </li>
            ))}
          </ul>
        </article>
      </section>

      <section className="panel split">
        <article>
          <h2>Schema Report</h2>
          <ul>
            <li>Rows: {formatCount(schema.n_rows)}</li>
            <li>Columns: {formatCount(schema.n_columns)}</li>
            <li>Duplicate Rows: {formatCount(schema.duplicate_rows)}</li>
            <li>Missing Ratio: {formatNumber(schema.global_missing_ratio, 6)}</li>
            <li>Best Model by F1: {bestModel}</li>
          </ul>
        </article>
        <article>
          <h2>Preprocess Report</h2>
          <ul>
            <li>Original Features: {formatCount(preprocess.original_feature_columns)}</li>
            <li>Final Features: {formatCount(preprocess.final_feature_columns)}</li>
            <li>
              Dropped Constants:{' '}
              {(preprocess.dropped_constant_columns || []).length
                ? preprocess.dropped_constant_columns.join(', ')
                : 'None'}
            </li>
            <li>
              Top Missing Columns:{' '}
              {qualitySummary.top_missing_columns
                ? JSON.stringify(qualitySummary.top_missing_columns)
                : 'None'}
            </li>
          </ul>
        </article>
      </section>
    </div>
  )
}

export default App
