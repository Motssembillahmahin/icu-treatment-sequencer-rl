import React from 'react'

interface VitalsGaugeProps {
  label: string
  value: number
  unit: string
  min: number
  max: number
  normalMin: number
  normalMax: number
  decimals?: number
}

function getStatus(value: number, normalMin: number, normalMax: number): 'normal' | 'warning' | 'critical' {
  const range = normalMax - normalMin
  if (value >= normalMin && value <= normalMax) return 'normal'
  if (value < normalMin - range * 0.3 || value > normalMax + range * 0.3) return 'critical'
  return 'warning'
}

const STATUS_COLORS = {
  normal: 'text-green-400 border-green-500/30 bg-green-500/5',
  warning: 'text-yellow-400 border-yellow-500/30 bg-yellow-500/5',
  critical: 'text-red-400 border-red-500/30 bg-red-500/5',
}

export function VitalsGauge({ label, value, unit, min, max, normalMin, normalMax, decimals = 0 }: VitalsGaugeProps) {
  const status = getStatus(value, normalMin, normalMax)
  const pct = Math.max(0, Math.min(100, ((value - min) / (max - min)) * 100))

  return (
    <div className={`rounded-lg border p-3 ${STATUS_COLORS[status]}`}>
      <div className="text-xs text-slate-400 mb-1">{label}</div>
      <div className="text-2xl font-mono font-bold">
        {value.toFixed(decimals)}
        <span className="text-sm font-normal ml-1 text-slate-400">{unit}</span>
      </div>
      <div className="mt-2 h-1 bg-slate-700 rounded">
        <div
          className={`h-1 rounded transition-all duration-300 ${
            status === 'normal' ? 'bg-green-400' : status === 'warning' ? 'bg-yellow-400' : 'bg-red-400'
          }`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <div className="text-xs text-slate-500 mt-1">
        Normal: {normalMin}–{normalMax}
      </div>
    </div>
  )
}
