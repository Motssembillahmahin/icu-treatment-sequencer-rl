import React from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import type { MetricPoint } from '../../types'

interface LossCurveProps {
  data: MetricPoint[]
}

export function LossCurve({ data }: LossCurveProps) {
  const filtered = data.filter((d) => d.loss !== null)
  return (
    <div className="bg-icu-card border border-icu-border rounded-lg p-4">
      <h3 className="text-sm font-semibold text-slate-300 mb-3">Training Loss</h3>
      <ResponsiveContainer width="100%" height={180}>
        <LineChart data={filtered}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis dataKey="timestep" stroke="#64748b" tick={{ fontSize: 10 }} tickFormatter={(v) => `${(v / 1000).toFixed(0)}k`} />
          <YAxis stroke="#64748b" tick={{ fontSize: 10 }} />
          <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }} />
          <Line type="monotone" dataKey="loss" name="Loss" stroke="#f87171" dot={false} strokeWidth={2} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
