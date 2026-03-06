import React from 'react'
import type { ActionResponse } from '../../types'
import { ACTION_LABELS } from '../../types'

interface ActionHistoryProps {
  history: ActionResponse[]
}

export function ActionHistory({ history }: ActionHistoryProps) {
  return (
    <div className="bg-icu-card border border-icu-border rounded-lg p-4">
      <h3 className="text-sm font-semibold text-slate-300 mb-3">Action History (last 10)</h3>
      {history.length === 0 ? (
        <p className="text-slate-500 text-sm">No actions yet</p>
      ) : (
        <div className="space-y-1">
          {history.map((a, i) => (
            <div
              key={i}
              className={`flex items-center justify-between text-sm rounded px-2 py-1 ${
                i === 0 ? 'bg-slate-700/50' : ''
              }`}
            >
              <span className="text-slate-300">{ACTION_LABELS[a.action_id] ?? a.action_name}</span>
              <span className="text-slate-500 text-xs">{(a.confidence * 100).toFixed(0)}%</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
