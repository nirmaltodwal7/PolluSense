/**
 * Dashboard.js
 * ------------
 * Main PolluSense dashboard component.
 *
 * Features:
 *  - City selector (dropdown from /cities API)
 *  - Date picker
 *  - Days slider (1–7)
 *  - Predict button (calls POST /predict)
 *  - AQI stat cards
 *  - Health alert banner
 *  - 7-day forecast chart
 */

import React, { useState, useEffect } from 'react';
import axios from 'axios';
import AQIChart from './AQIChart.jsx';
import './Dashboard.css';

const API_BASE = 'http://localhost:5000';

/* ── AQI helpers (mirrored from backend) ───────────────────── */
const AQI_CATEGORY_CLASS = {
  'Good':           'aqi-good',
  'Moderate':       'aqi-moderate',
  'Unhealthy':      'aqi-unhealthy',
  'Very Unhealthy': 'aqi-very-unhealthy',
  'Hazardous':      'aqi-hazardous',
};

const BADGE_CLASS = {
  'Good':           'badge-good',
  'Moderate':       'badge-moderate',
  'Unhealthy':      'badge-unhealthy',
  'Very Unhealthy': 'badge-very-unhealthy',
  'Hazardous':      'badge-hazardous',
};

const CATEGORY_ICON = {
  'Good':           '🌿',
  'Moderate':       '🌤️',
  'Unhealthy':      '😷',
  'Very Unhealthy': '⚠️',
  'Hazardous':      '☠️',
};

const todayStr = () => new Date().toISOString().split('T')[0];

/* ── Skeleton component ─────────────────────────────────────── */
const Skeleton = ({ className }) => (
  <div className={`skeleton ${className}`} />
);

/* ── Dashboard ──────────────────────────────────────────────── */
const Dashboard = () => {
  // Form state
  const [city, setCity]   = useState('Delhi');
  const [date, setDate]   = useState(todayStr());
  const [days, setDays]   = useState(7);
  const [cities, setCities] = useState([]);

  // Result state
  const [result, setResult]     = useState(null);
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState(null);
  const [hasFetched, setHasFetched] = useState(false);

  // Fetch city list on mount
  useEffect(() => {
    axios.get(`${API_BASE}/cities`)
      .then((res) => setCities(res.data.cities || []))
      .catch(() => {
        // Fallback list if backend not yet running
        setCities([
          'Delhi','Mumbai','Kolkata','Chennai','Bangalore',
          'Hyderabad','Pune','Ahmedabad','Jaipur','Lucknow'
        ]);
      });
  }, []);

  /* ── Predict handler ─────────────────────────────────────── */
  const handlePredict = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    setHasFetched(true);

    try {
      const payload = { city, days: parseInt(days, 10) };
      const res = await axios.post(`${API_BASE}/predict`, payload);
      setResult(res.data);
    } catch (err) {
      if (err.response?.data?.error) {
        setError(err.response.data.error);
      } else if (err.code === 'ERR_NETWORK') {
        setError(
          'Cannot connect to the PolluSense API. Make sure the Flask backend ' +
          'is running on http://localhost:5000.'
        );
      } else {
        setError('An unexpected error occurred. Please try again.');
      }
    } finally {
      setLoading(false);
    }
  };

  /* ── Derived display values ──────────────────────────────── */
  const peakAqi      = result ? Math.max(...result.predictions) : null;
  const peakCategory = peakAqi != null ? result.categories[result.predictions.indexOf(peakAqi)] : null;
  const avgAqi       = result
    ? Math.round(result.predictions.reduce((a, b) => a + b, 0) / result.predictions.length)
    : null;
  const avgCategory  = result ? result.categories[Math.floor(result.categories.length / 2)] : null;

  /* ── Render ──────────────────────────────────────────────── */
  return (
    <div className="dashboard-page">

      {/* ── Header ── */}
      <header className="header">
        <span className="header-icon">🌫️</span>
        <div>
          <div className="header-title">PolluSense</div>
          <div className="header-subtitle">
            Hybrid AI Air Quality Prediction Dashboard
          </div>
        </div>
      </header>

      <div className="main-content">

        {/* ── Control Panel ── */}
        <div className="glass-card control-panel">
          <h2>Configure Prediction</h2>

          {/* City */}
          <div className="field-group">
            <label htmlFor="city-select">City</label>
            <select
              id="city-select"
              value={city}
              onChange={(e) => setCity(e.target.value)}
            >
              {cities.map((c) => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
          </div>

          {/* Date */}
          <div className="field-group">
            <label htmlFor="date-picker">Target Date</label>
            <input
              id="date-picker"
              type="date"
              value={date}
              onChange={(e) => setDate(e.target.value)}
              min="2021-01-01"
              max="2030-12-31"
            />
          </div>

          {/* Days */}
          <div className="field-group">
            <label htmlFor="days-select">
              Forecast Days
              <span className="days-value">{days}</span>
            </label>
            <select
              id="days-select"
              value={days}
              onChange={(e) => setDays(Number(e.target.value))}
            >
              {[1,2,3,4,5,6,7].map((d) => (
                <option key={d} value={d}>{d} {d === 1 ? 'Day' : 'Days'}</option>
              ))}
            </select>
          </div>

          {/* Submit */}
          <button
            id="predict-btn"
            className={`predict-btn${loading ? ' loading' : ''}`}
            onClick={handlePredict}
            disabled={loading}
          >
            {loading ? '⏳ Predicting…' : '🔮 Predict AQI'}
          </button>
        </div>

        {/* ── Error ── */}
        {error && (
          <div className="error-banner" role="alert">
            <span>❌</span>
            <span>{error}</span>
          </div>
        )}

        {/* ── Skeleton while loading ── */}
        {loading && hasFetched && (
          <div className="results-section">
            <div className="glass-card stat-card"><Skeleton className="skeleton-stat" /></div>
            <div className="glass-card stat-card"><Skeleton className="skeleton-stat" /></div>
            <div className="glass-card chart-panel" style={{ gridColumn: '1 / -1' }}>
              <Skeleton className="skeleton-chart" />
            </div>
          </div>
        )}

        {/* ── Results ── */}
        {result && !loading && (
          <>
            {/* City tag */}
            <div className="city-tag">📍 {result.city} · {days}-Day Forecast</div>

            {/* Alert */}
            {result.alert && (
              <div className="alert-card" role="alert">
                {result.alert}
              </div>
            )}

            {/* Stats */}
            <div className="results-section">

              {/* Peak AQI */}
              <div className="glass-card stat-card">
                <span className="stat-label">Peak AQI</span>
                <span className={`stat-value ${AQI_CATEGORY_CLASS[peakCategory] || ''}`}>
                  {peakAqi}
                </span>
                <span
                  className={`category-badge ${BADGE_CLASS[peakCategory] || ''}`}
                >
                  {CATEGORY_ICON[peakCategory]} {peakCategory}
                </span>
              </div>

              {/* Average AQI */}
              <div className="glass-card stat-card">
                <span className="stat-label">Avg AQI ({days} days)</span>
                <span className={`stat-value ${AQI_CATEGORY_CLASS[avgCategory] || ''}`}>
                  {avgAqi}
                </span>
                <span
                  className={`category-badge ${BADGE_CLASS[avgCategory] || ''}`}
                >
                  {CATEGORY_ICON[avgCategory]} {avgCategory}
                </span>
              </div>

              {/* Day 1 AQI */}
              <div className="glass-card stat-card">
                <span className="stat-label">Tomorrow's AQI</span>
                <span className={`stat-value ${AQI_CATEGORY_CLASS[result.categories[0]] || ''}`}>
                  {Math.round(result.predictions[0])}
                </span>
                <span
                  className={`category-badge ${BADGE_CLASS[result.categories[0]] || ''}`}
                >
                  {CATEGORY_ICON[result.categories[0]]} {result.categories[0]}
                </span>
              </div>

            </div>

            {/* Forecast Table */}
            <div className="glass-card" style={{ padding: '20px 24px', overflowX: 'auto' }}>
              <p style={{ fontSize: '0.72rem', fontWeight: 600, letterSpacing: '1.2px',
                textTransform: 'uppercase', color: 'var(--text-secondary)', marginBottom: 14 }}>
                Daily Breakdown
              </p>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.88rem' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid rgba(99,170,255,0.12)' }}>
                    {['Day', 'Date', 'AQI', 'Category'].map((h) => (
                      <th key={h} style={{ padding: '8px 12px', textAlign: 'left',
                        color: 'var(--text-secondary)', fontWeight: 600,
                        fontSize: '0.72rem', letterSpacing: '0.8px', textTransform: 'uppercase' }}>
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {result.forecast.map((f) => (
                    <tr key={f.day}
                      style={{ borderBottom: '1px solid rgba(99,170,255,0.06)',
                        transition: 'background 0.15s' }}
                      onMouseEnter={(e) => e.currentTarget.style.background = 'rgba(79,142,247,0.05)'}
                      onMouseLeave={(e) => e.currentTarget.style.background = 'transparent'}
                    >
                      <td style={{ padding: '10px 12px', color: 'var(--text-secondary)', fontWeight: 600 }}>
                        Day {f.day}
                      </td>
                      <td style={{ padding: '10px 12px', color: 'var(--text-primary)' }}>
                        {f.date}
                      </td>
                      <td style={{ padding: '10px 12px', fontWeight: 700,
                        color: BADGE_COLOR(f.category), fontFamily: 'Space Grotesk, sans-serif',
                        fontSize: '1.05rem' }}>
                        {Math.round(f.aqi)}
                      </td>
                      <td style={{ padding: '10px 12px' }}>
                        <span className={`category-badge ${BADGE_CLASS[f.category] || ''}`}>
                          {CATEGORY_ICON[f.category]} {f.category}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Chart */}
            <div className="glass-card chart-panel">
              <h3>AQI Forecast Trend</h3>
              <AQIChart data={result.forecast} />
            </div>
          </>
        )}
      </div>
    </div>
  );
};

/* Helper for table text color */
function BADGE_COLOR(category) {
  const map = {
    'Good':           '#2dd77b',
    'Moderate':       '#f9c846',
    'Unhealthy':      '#f97316',
    'Very Unhealthy': '#e53e3e',
    'Hazardous':      '#9f3fbf',
  };
  return map[category] || '#4f8ef7';
}

export default Dashboard;
