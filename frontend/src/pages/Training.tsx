import React, { useState } from 'react'
import { useTrainingMetrics } from '../hooks/useTrainingMetrics'
import { usePatientStore } from '../stores/patientStore'
import { RewardCurve } from '../components/metrics/RewardCurve'
import { LossCurve } from '../components/metrics/LossCurve'
import { trainingApi } from '../api/client'

export function Training() {
  useTrainingMetrics()
  const { trainingMetrics, trainingStatus, setTrainingStatus } = usePatientStore()
  const [error, setError] = useState<string | null>(null)

  const handleStart = async () => {
    try {
      setError(null)
      await trainingApi.start(undefined, 10000, 1) // debug: 10k steps
      setTrainingStatus('running')
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to start training')
    }
  }

  const handleStop = async () => {
    try {
      await trainingApi.stop()
      setTrainingStatus('stopped')
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to stop training')
    }
  }

  const points = trainingMetrics?.points ?? []

  return (
    <div className="space-y-4">
      <div className="bg-icu-card border border-icu-border rounded-lg p-4">
        <h2 className="text-lg font-semibold text-slate-200 mb-3">Training Control</h2>
        <div className="flex gap-3">
          <button
            onClick={handleStart}
            disabled={trainingStatus === 'running'}
            className="px-4 py-2 bg-green-600 hover:bg-green-500 disabled:opacity-40 rounded text-sm font-medium"
          >
            Start Training (10k debug)
          </button>
          <button
            onClick={handleStop}
            disabled={trainingStatus !== 'running'}
            className="px-4 py-2 bg-red-600 hover:bg-red-500 disabled:opacity-40 rounded text-sm font-medium"
          >
            Stop
          </button>
          <span className="self-center text-sm text-slate-400">
            Status: <span className="text-slate-200">{trainingStatus}</span>
          </span>
        </div>
        {error && <p className="mt-2 text-red-400 text-sm">{error}</p>}
      </div>

      <div className="grid grid-cols-2 gap-4">
        <RewardCurve data={points} />
        <LossCurve data={points} />
      </div>

      {trainingMetrics && (
        <div className="bg-icu-card border border-icu-border rounded-lg p-4 text-sm text-slate-400">
          <span>Total episodes: {trainingMetrics.total_episodes}</span>
          {' · '}
          <span>Latest timestep: {trainingMetrics.latest_timestep.toLocaleString()}</span>
        </div>
      )}
    </div>
  )
}
