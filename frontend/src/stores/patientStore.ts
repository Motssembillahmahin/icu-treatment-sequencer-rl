import { create } from 'zustand'
import type { ActionResponse, HealthStatus, PatientStateResponse, TrainingMetrics } from '../types'

interface PatientStore {
  // Health
  health: HealthStatus | null
  setHealth: (h: HealthStatus) => void

  // Current patient state (simulated / live)
  patientState: PatientStateResponse | null
  setPatientState: (s: PatientStateResponse) => void

  // Last inference result
  lastAction: ActionResponse | null
  setLastAction: (a: ActionResponse) => void

  // Action history (last 10 steps)
  actionHistory: ActionResponse[]
  pushAction: (a: ActionResponse) => void

  // Training metrics
  trainingMetrics: TrainingMetrics | null
  setTrainingMetrics: (m: TrainingMetrics) => void

  // Training job status
  trainingStatus: string
  setTrainingStatus: (s: string) => void

  // Errors
  error: string | null
  setError: (e: string | null) => void
}

export const usePatientStore = create<PatientStore>((set) => ({
  health: null,
  setHealth: (h) => set({ health: h }),

  patientState: null,
  setPatientState: (s) => set({ patientState: s }),

  lastAction: null,
  setLastAction: (a) => set({ lastAction: a }),

  actionHistory: [],
  pushAction: (a) =>
    set((state) => ({
      actionHistory: [a, ...state.actionHistory].slice(0, 10),
    })),

  trainingMetrics: null,
  setTrainingMetrics: (m) => set({ trainingMetrics: m }),

  trainingStatus: 'idle',
  setTrainingStatus: (s) => set({ trainingStatus: s }),

  error: null,
  setError: (e) => set({ error: e }),
}))
