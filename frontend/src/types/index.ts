// TypeScript types mirroring backend Pydantic schemas

export interface VitalsReading {
  heart_rate: number
  map: number
  spo2: number
  gcs: number
  lactate: number
  rr: number
  temperature: number
  fio2: number
  peep: number
  vasopressor: number
  fluid_balance: number
}

export interface PatientStateResponse extends VitalsReading {
  time_in_icu: number
  episode_step_pct: number
  archetype: string
}

export interface ActionRequest {
  vitals: VitalsReading
  time_in_icu: number
  episode_step_pct: number
  deterministic: boolean
}

export interface ActionResponse {
  action_id: number
  action_name: string
  confidence: number
  all_action_probs: number[]
  reasoning_tags: string[]
}

export interface MetricPoint {
  timestep: number
  episode: number
  mean_reward: number | null
  mean_ep_length: number | null
  loss: number | null
  logged_at: string
}

export interface TrainingMetrics {
  points: MetricPoint[]
  total_episodes: number
  latest_timestep: number
}

export interface StepRecord {
  step: number
  action_id: number
  action_name: string
  reward: number
  vitals: Record<string, number>
  terminated: boolean
}

export interface EpisodeSummary {
  id: number
  archetype: string
  total_steps: number
  total_reward: number
  survived: boolean
  started_at: string
  ended_at: string
}

export interface EpisodeReplay extends EpisodeSummary {
  steps: StepRecord[]
}

export interface HealthStatus {
  status: string
  model_loaded: boolean
}

export interface TrainingJobStatus {
  job_id: string
  status: string
  config_path: string
  started_at: string | null
  ended_at: string | null
  error: string | null
}

export const ACTION_NAMES: Record<number, string> = {
  0: 'NOOP',
  1: 'BOLUS_250',
  2: 'BOLUS_500',
  3: 'VASOPRESSOR_UP',
  4: 'VASOPRESSOR_DOWN',
  5: 'FIO2_UP',
  6: 'FIO2_DOWN',
  7: 'PEEP_UP',
  8: 'PEEP_DOWN',
  9: 'SEDATION',
  10: 'EXTUBATE',
}

export const ACTION_LABELS: Record<number, string> = {
  0: 'No Intervention',
  1: 'IV Bolus 250 mL',
  2: 'IV Bolus 500 mL',
  3: 'Vasopressor ↑',
  4: 'Vasopressor ↓',
  5: 'FiO₂ ↑',
  6: 'FiO₂ ↓',
  7: 'PEEP ↑',
  8: 'PEEP ↓',
  9: 'Sedation',
  10: 'Extubate',
}
