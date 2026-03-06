import React, { useState } from 'react'
import type { EpisodeReplay } from '../../types'
import { ACTION_LABELS } from '../../types'
import { VitalsChart } from '../vitals/VitalsChart'

interface EpisodeReplayViewerProps {
  episode: EpisodeReplay
}

export function EpisodeReplayViewer({ episode }: EpisodeReplayViewerProps) {
  const [currentStep, setCurrentStep] = useState(0)
  const step = episode.steps[currentStep]

  const chartData = episode.steps.slice(0, currentStep + 1).map((s) => ({
    step: s.step,
    heart_rate: s.vitals.heart_rate ?? 0,
    map: s.vitals.map ?? 0,
    spo2: s.vitals.spo2 ?? 0,
    lactate: s.vitals.lactate ?? 0,
  }))

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-4 bg-icu-card border border-icu-border rounded-lg p-3">
        <div className="text-sm text-slate-400">
          <span className="font-semibold text-slate-200">Episode {episode.id}</span>
          {' · '}{episode.archetype}
          {' · '}{episode.survived ? <span className="text-green-400">Survived</span> : <span className="text-red-400">Died</span>}
          {' · '}Reward: <span className="text-icu-accent">{episode.total_reward.toFixed(1)}</span>
        </div>
      </div>

      <VitalsChart data={chartData} />

      {/* Step controls */}
      <div className="bg-icu-card border border-icu-border rounded-lg p-4">
        <div className="flex items-center gap-3 mb-3">
          <button
            onClick={() => setCurrentStep(Math.max(0, currentStep - 1))}
            disabled={currentStep === 0}
            className="px-3 py-1 text-sm bg-slate-700 hover:bg-slate-600 disabled:opacity-40 rounded"
          >
            ← Prev
          </button>
          <input
            type="range"
            min={0}
            max={Math.max(0, episode.steps.length - 1)}
            value={currentStep}
            onChange={(e) => setCurrentStep(Number(e.target.value))}
            className="flex-1"
          />
          <button
            onClick={() => setCurrentStep(Math.min(episode.steps.length - 1, currentStep + 1))}
            disabled={currentStep >= episode.steps.length - 1}
            className="px-3 py-1 text-sm bg-slate-700 hover:bg-slate-600 disabled:opacity-40 rounded"
          >
            Next →
          </button>
          <span className="text-sm text-slate-400">
            Step {currentStep + 1} / {episode.steps.length}
          </span>
        </div>

        {step && (
          <div className="grid grid-cols-2 gap-3 text-sm">
            <div>
              <div className="text-slate-400">Action</div>
              <div className="text-icu-accent font-semibold">{ACTION_LABELS[step.action_id] ?? step.action_name}</div>
            </div>
            <div>
              <div className="text-slate-400">Reward</div>
              <div className={step.reward >= 0 ? 'text-green-400' : 'text-red-400'}>
                {step.reward.toFixed(3)}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
