import React, { useEffect, useState } from 'react'
import type { EpisodeReplay, EpisodeSummary } from '../types'
import { episodesApi } from '../api/client'
import { EpisodeReplayViewer } from '../components/replay/EpisodeReplayViewer'

export function EpisodeViewer() {
  const [summaries, setSummaries] = useState<EpisodeSummary[]>([])
  const [selected, setSelected] = useState<EpisodeReplay | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    episodesApi
      .list(20, 0)
      .then(setSummaries)
      .catch(() => setError('Failed to load episodes'))
  }, [])

  const loadEpisode = async (id: number) => {
    setLoading(true)
    try {
      const ep = await episodesApi.get(id)
      setSelected(ep)
    } catch {
      setError(`Failed to load episode ${id}`)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="grid grid-cols-3 gap-4">
      {/* Episode list */}
      <div className="col-span-1 bg-icu-card border border-icu-border rounded-lg p-4">
        <h2 className="text-sm font-semibold text-slate-300 mb-3">Episodes</h2>
        {error && <p className="text-red-400 text-xs mb-2">{error}</p>}
        {summaries.length === 0 ? (
          <p className="text-slate-500 text-sm">No episodes recorded yet. Run training first.</p>
        ) : (
          <div className="space-y-2">
            {summaries.map((ep) => (
              <button
                key={ep.id}
                onClick={() => loadEpisode(ep.id)}
                className={`w-full text-left p-2 rounded border text-sm transition-colors ${
                  selected?.id === ep.id
                    ? 'border-icu-accent bg-icu-accent/10'
                    : 'border-icu-border hover:border-slate-400'
                }`}
              >
                <div className="flex justify-between">
                  <span className="text-slate-300">#{ep.id} {ep.archetype}</span>
                  <span className={ep.survived ? 'text-green-400' : 'text-red-400'}>
                    {ep.survived ? '✓' : '✗'}
                  </span>
                </div>
                <div className="text-xs text-slate-500">
                  {ep.total_steps} steps · reward {ep.total_reward.toFixed(1)}
                </div>
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Episode replay */}
      <div className="col-span-2">
        {loading && <p className="text-slate-400">Loading episode...</p>}
        {!loading && !selected && (
          <div className="bg-icu-card border border-icu-border rounded-lg p-8 text-center text-slate-500">
            Select an episode to replay
          </div>
        )}
        {!loading && selected && <EpisodeReplayViewer episode={selected} />}
      </div>
    </div>
  )
}
