import axios from 'axios'
import type {
  ActionRequest,
  ActionResponse,
  EpisodeReplay,
  EpisodeSummary,
  HealthStatus,
  TrainingJobStatus,
  TrainingMetrics,
} from '../types'

const BASE_URL = import.meta.env.VITE_API_BASE_URL || ''

const api = axios.create({
  baseURL: `${BASE_URL}/api/v1`,
  timeout: 10_000,
  headers: { 'Content-Type': 'application/json' },
})

export const healthApi = {
  check: () => api.get<HealthStatus>('/health').then((r) => r.data),
}

export const inferenceApi = {
  predict: (req: ActionRequest) =>
    api.post<ActionResponse>('/inference', req).then((r) => r.data),
}

export const metricsApi = {
  get: (limit = 500) =>
    api.get<TrainingMetrics>('/metrics', { params: { limit } }).then((r) => r.data),
}

export const episodesApi = {
  list: (limit = 20, offset = 0) =>
    api.get<EpisodeSummary[]>('/episodes', { params: { limit, offset } }).then((r) => r.data),
  get: (id: number) => api.get<EpisodeReplay>(`/episodes/${id}`).then((r) => r.data),
}

export const trainingApi = {
  start: (configPath?: string, totalTimesteps?: number, nEnvs = 4) =>
    api
      .post<TrainingJobStatus>('/training/start', {
        config_path: configPath ?? 'configs/hyperparams/ppo_default.yaml',
        total_timesteps: totalTimesteps,
        n_envs: nEnvs,
      })
      .then((r) => r.data),
  stop: () => api.post('/training/stop').then((r) => r.data),
  status: () => api.get<TrainingJobStatus[]>('/training/status').then((r) => r.data),
}
