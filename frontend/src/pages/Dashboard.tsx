import React, { useEffect, useRef, useState } from 'react'
import { useVitalsPolling } from '../hooks/useVitalsPolling'
import { usePatientStore } from '../stores/patientStore'
import { VitalsPanel } from '../components/vitals/VitalsPanel'
import { VitalsChart } from '../components/vitals/VitalsChart'
import { ActionCard } from '../components/actions/ActionCard'
import { ActionHistory } from '../components/actions/ActionHistory'
import type { VitalsReading } from '../types'

// Simulated rolling vitals history for chart (in real use, this would come from polling)
const MOCK_INITIAL_VITALS: VitalsReading = {
  heart_rate: 95,
  map: 72,
  spo2: 94,
  gcs: 13,
  lactate: 2.5,
  rr: 22,
  temperature: 38.2,
  fio2: 0.40,
  peep: 5,
  vasopressor: 0.1,
  fluid_balance: 200,
}

export function Dashboard() {
  useVitalsPolling()
  const { health, lastAction, actionHistory, error } = usePatientStore()
  const [vitals, setVitals] = useState<VitalsReading>(MOCK_INITIAL_VITALS)
  const [chartData, setChartData] = useState<Array<{ step: number } & Record<string, number>>>([])
  const stepRef = useRef(0)

  // Simulate vitals updates for demo purposes
  useEffect(() => {
    const timer = setInterval(() => {
      setVitals((prev) => ({
        ...prev,
        heart_rate: Math.max(40, Math.min(150, prev.heart_rate + (Math.random() - 0.5) * 5)),
        map: Math.max(40, Math.min(130, prev.map + (Math.random() - 0.5) * 4)),
        spo2: Math.max(80, Math.min(100, prev.spo2 + (Math.random() - 0.5) * 1)),
        lactate: Math.max(0.5, Math.min(8, prev.lactate + (Math.random() - 0.5) * 0.3)),
      }))
      stepRef.current += 1
    }, 2000)
    return () => clearInterval(timer)
  }, [])

  useEffect(() => {
    setChartData((prev) => [
      ...prev.slice(-59),
      {
        step: stepRef.current,
        heart_rate: vitals.heart_rate,
        map: vitals.map,
        spo2: vitals.spo2,
        lactate: vitals.lactate,
      },
    ])
  }, [vitals])

  return (
    <div className="space-y-4">
      {/* Status bar */}
      <div className="flex items-center justify-between bg-icu-card border border-icu-border rounded-lg px-4 py-2">
        <span className="text-sm text-slate-400">ICU Treatment Sequencer</span>
        <div className="flex items-center gap-2">
          {error ? (
            <span className="text-red-400 text-xs">● Backend unreachable</span>
          ) : (
            <span className="text-green-400 text-xs">● {health?.model_loaded ? 'Model Live' : 'Connected'}</span>
          )}
        </div>
      </div>

      {/* Main grid */}
      <div className="grid grid-cols-3 gap-4">
        {/* Left: chart */}
        <div className="col-span-2 space-y-4">
          <VitalsChart data={chartData} />
          <ActionHistory history={actionHistory} />
        </div>

        {/* Right: gauges + action */}
        <div className="space-y-4">
          <VitalsPanel vitals={vitals} />
          <ActionCard action={lastAction} />
        </div>
      </div>
    </div>
  )
}
