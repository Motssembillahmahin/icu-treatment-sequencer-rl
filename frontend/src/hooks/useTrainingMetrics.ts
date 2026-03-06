import { useEffect, useRef } from 'react'
import { metricsApi } from '../api/client'
import { usePatientStore } from '../stores/patientStore'

const POLL_INTERVAL_MS = 5000

export function useTrainingMetrics() {
  const { setTrainingMetrics } = usePatientStore()
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    const poll = async () => {
      try {
        const metrics = await metricsApi.get()
        setTrainingMetrics(metrics)
      } catch {
        // silently skip — metrics may not be available yet
      }
    }

    poll()
    timerRef.current = setInterval(poll, POLL_INTERVAL_MS)

    return () => {
      if (timerRef.current) clearInterval(timerRef.current)
    }
  }, [setTrainingMetrics])
}
