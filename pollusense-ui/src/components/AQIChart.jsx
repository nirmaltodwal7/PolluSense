/**
 * AQIChart.js
 * -----------
 * Recharts AreaChart for visualising the 7-day AQI forecast.
 * Color zone references lines are drawn at each AQI threshold.
 */

import React from 'react';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ReferenceLine, ResponsiveContainer, Dot
} from 'recharts';

/* ── AQI zone colors ────────────────────────────────────────── */
const AQI_COLORS = {
  Good:           '#2dd77b',
  Moderate:       '#f9c846',
  Unhealthy:      '#f97316',
  'Very Unhealthy': '#e53e3e',
  Hazardous:      '#9f3fbf',
};

const getColor = (category) => AQI_COLORS[category] || '#4f8ef7';

/**
 * CustomTooltip — rich hover tooltip with AQI value + category badge.
 */
const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload || !payload.length) return null;
  const { aqi, category, date } = payload[0].payload;
  const color = getColor(category);

  return (
    <div style={{
      background: 'rgba(7, 20, 40, 0.92)',
      border: `1px solid ${color}55`,
      borderRadius: '10px',
      padding: '12px 16px',
      backdropFilter: 'blur(12px)',
      boxShadow: `0 8px 32px rgba(0,0,0,0.4)`,
      fontSize: '0.82rem',
    }}>
      <p style={{ color: '#7fa3d4', marginBottom: 6, fontWeight: 600 }}>
        {label} — {date}
      </p>
      <p style={{ color: '#fff', fontSize: '1.4rem', fontWeight: 700, lineHeight: 1 }}>
        AQI <span style={{ color }}>{aqi}</span>
      </p>
      <span style={{
        display: 'inline-block',
        marginTop: 8,
        padding: '3px 10px',
        borderRadius: '20px',
        backgroundColor: `${color}22`,
        border: `1px solid ${color}66`,
        color,
        fontWeight: 600,
        fontSize: '0.75rem',
      }}>
        {category}
      </span>
    </div>
  );
};

/**
 * CustomDot — render a colored dot based on AQI category.
 */
const CustomDot = (props) => {
  const { cx, cy, payload, index } = props;
  const color = getColor(payload.category);
  return (
    <g key={`dot-${index}`}>
      {/* Glow ring */}
      <circle cx={cx} cy={cy} r={10} fill={`${color}22`} />
      <circle cx={cx} cy={cy} r={5} fill={color} stroke="#fff" strokeWidth={1.5} />
    </g>
  );
};

/**
 * AQIChart
 * Props:
 *   data — array of { day, date, aqi, category }
 */
const AQIChart = ({ data }) => {
  if (!data || data.length === 0) return null;

  // Build chart data with a nice x-label
  const chartData = data.map((d) => ({
    ...d,
    label: `Day ${d.day}`,
    aqi: Math.round(d.aqi),
  }));

  // Determine gradient stops based on AQI range
  const maxAqi = Math.max(...chartData.map((d) => d.aqi));
  const primaryColor = getColor(chartData[0]?.category || 'Moderate');

  return (
    <ResponsiveContainer width="100%" height={300}>
      <AreaChart
        data={chartData}
        margin={{ top: 10, right: 20, left: -10, bottom: 0 }}
      >
        {/* Gradient fill */}
        <defs>
          <linearGradient id="aqiGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%"  stopColor={primaryColor} stopOpacity={0.35} />
            <stop offset="95%" stopColor={primaryColor} stopOpacity={0.02} />
          </linearGradient>
        </defs>

        <CartesianGrid
          strokeDasharray="3 3"
          stroke="rgba(99,170,255,0.08)"
          vertical={false}
        />

        <XAxis
          dataKey="label"
          tick={{ fill: '#7fa3d4', fontSize: 12, fontFamily: 'Inter' }}
          axisLine={{ stroke: 'rgba(99,170,255,0.15)' }}
          tickLine={false}
        />

        <YAxis
          domain={[0, Math.max(500, maxAqi + 50)]}
          tick={{ fill: '#7fa3d4', fontSize: 11, fontFamily: 'Inter' }}
          axisLine={false}
          tickLine={false}
          tickFormatter={(v) => v}
        />

        <Tooltip content={<CustomTooltip />} />

        {/* AQI threshold reference lines */}
        <ReferenceLine y={50}  stroke="#2dd77b" strokeDasharray="4 4" strokeOpacity={0.5}
          label={{ value: 'Good 50',       fill: '#2dd77b', fontSize: 10, position: 'insideTopRight' }} />
        <ReferenceLine y={100} stroke="#f9c846" strokeDasharray="4 4" strokeOpacity={0.5}
          label={{ value: 'Moderate 100',  fill: '#f9c846', fontSize: 10, position: 'insideTopRight' }} />
        <ReferenceLine y={150} stroke="#f97316" strokeDasharray="4 4" strokeOpacity={0.5}
          label={{ value: 'Unhealthy 150', fill: '#f97316', fontSize: 10, position: 'insideTopRight' }} />
        <ReferenceLine y={200} stroke="#e53e3e" strokeDasharray="4 4" strokeOpacity={0.5}
          label={{ value: 'Very Unhealthy 200', fill: '#e53e3e', fontSize: 10, position: 'insideTopRight' }} />

        <Area
          type="monotone"
          dataKey="aqi"
          stroke={primaryColor}
          strokeWidth={2.5}
          fill="url(#aqiGradient)"
          dot={<CustomDot />}
          activeDot={false}
          animationDuration={900}
          animationEasing="ease-out"
        />
      </AreaChart>
    </ResponsiveContainer>
  );
};

export default AQIChart;
