import React from 'react'
import type { VitalsReading } from '../../types'
import { VitalsGauge } from './VitalsGauge'

interface VitalsPanelProps {
  vitals: VitalsReading
}

const VITALS_CONFIG = [
  { key: 'heart_rate', label: 'Heart Rate', unit: 'bpm', min: 20, max: 250, normalMin: 60, normalMax: 100, decimals: 0 },
  { key: 'map', label: 'MAP', unit: 'mmHg', min: 20, max: 160, normalMin: 70, normalMax: 105, decimals: 0 },
  { key: 'spo2', label: 'SpO₂', unit: '%', min: 50, max: 100, normalMin: 95, normalMax: 100, decimals: 1 },
  { key: 'gcs', label: 'GCS', unit: '', min: 3, max: 15, normalMin: 13, normalMax: 15, decimals: 0 },
  { key: 'lactate', label: 'Lactate', unit: 'mmol/L', min: 0.5, max: 10, normalMin: 0.5, normalMax: 2.0, decimals: 1 },
  { key: 'rr', label: 'Resp Rate', unit: '/min', min: 4, max: 40, normalMin: 12, normalMax: 20, decimals: 0 },
] as const

export function VitalsPanel({ vitals }: VitalsPanelProps) {
  return (
    <div className="grid grid-cols-2 gap-3">
      {VITALS_CONFIG.map((cfg) => (
        <VitalsGauge
          key={cfg.key}
          label={cfg.label}
          value={vitals[cfg.key]}
          unit={cfg.unit}
          min={cfg.min}
          max={cfg.max}
          normalMin={cfg.normalMin}
          normalMax={cfg.normalMax}
          decimals={cfg.decimals}
        />
      ))}
    </div>
  )
}
