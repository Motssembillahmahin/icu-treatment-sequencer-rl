import React from 'react'
import type { ActionResponse } from '../../types'
import { ACTION_LABELS } from '../../types'

interface ActionCardProps {
  action: ActionResponse | null
}

export function ActionCard({ action }: ActionCardProps) {
  if (!action) {
    return (
      <div className="bg-icu-card border border-icu-border rounded-lg p-4">
        <h3 className="text-sm font-semibold text-slate-300 mb-2">Recommended Action</h3>
        <p className="text-slate-500 text-sm">No model loaded — awaiting inference</p>
      </div>
    )
  }

  const label = ACTION_LABELS[action.action_id] ?? action.action_name
  const confidencePct = (action.confidence * 100).toFixed(1)

  return (
    <div className="bg-icu-card border border-icu-accent/40 rounded-lg p-4">
      <h3 className="text-sm font-semibold text-slate-300 mb-2">Recommended Action</h3>
      <div className="text-xl font-bold text-icu-accent">{label}</div>
      <div className="mt-1 text-sm text-slate-400">
        Confidence: <span className="text-white font-medium">{confidencePct}%</span>
      </div>

      {/* Action probability bars */}
      <div className="mt-3 space-y-1">
        {action.all_action_probs.slice(0, 5).map((prob, idx) => (
          <div key={idx} className="flex items-center gap-2">
            <span className="text-xs text-slate-500 w-28 truncate">{ACTION_LABELS[idx]}</span>
            <div className="flex-1 h-1.5 bg-slate-700 rounded">
              <div
                className={`h-1.5 rounded ${idx === action.action_id ? 'bg-icu-accent' : 'bg-slate-500'}`}
                style={{ width: `${(prob * 100).toFixed(1)}%` }}
              />
            </div>
            <span className="text-xs text-slate-500 w-10 text-right">{(prob * 100).toFixed(0)}%</span>
          </div>
        ))}
      </div>

      {action.reasoning_tags.length > 0 && (
        <div className="mt-3">
          {action.reasoning_tags.map((tag, i) => (
            <span
              key={i}
              className="inline-block text-xs bg-slate-700 text-slate-300 rounded px-2 py-0.5 mr-1 mb-1"
            >
              {tag}
            </span>
          ))}
        </div>
      )}
    </div>
  )
}
