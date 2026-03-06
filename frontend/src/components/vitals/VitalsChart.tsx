import React from 'react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts'

interface VitalPoint {
  step: number
  heart_rate: number
  map: number
  spo2: number
  lactate: number
}

interface VitalsChartProps {
  data: VitalPoint[]
}

export function VitalsChart({ data }: VitalsChartProps) {
  return (
    <div className="bg-icu-card border border-icu-border rounded-lg p-4">
      <h3 className="text-sm font-semibold text-slate-300 mb-3">Vitals Trend (last 60 steps)</h3>
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={data} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis dataKey="step" stroke="#64748b" tick={{ fontSize: 11 }} />
          <YAxis stroke="#64748b" tick={{ fontSize: 11 }} />
          <Tooltip
            contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: 6 }}
            labelStyle={{ color: '#94a3b8' }}
          />
          <Legend />
          <Line type="monotone" dataKey="map" name="MAP" stroke="#38bdf8" dot={false} strokeWidth={2} />
          <Line type="monotone" dataKey="spo2" name="SpO₂" stroke="#22c55e" dot={false} strokeWidth={2} />
          <Line type="monotone" dataKey="heart_rate" name="HR" stroke="#f59e0b" dot={false} strokeWidth={1.5} />
          <Line type="monotone" dataKey="lactate" name="Lactate" stroke="#f87171" dot={false} strokeWidth={1.5} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
