import { useEffect, useRef } from 'react'
import { healthApi } from '../api/client'
import { usePatientStore } from '../stores/patientStore'

const POLL_INTERVAL_MS = 2000

export function useVitalsPolling() {
  const { setHealth, setError } = usePatientStore()
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    const poll = async () => {
      try {
        const health = await healthApi.check()
        setHealth(health)
        setError(null)
      } catch {
        setError('Cannot reach backend')
      }
    }

    poll()
    timerRef.current = setInterval(poll, POLL_INTERVAL_MS)

    return () => {
      if (timerRef.current) clearInterval(timerRef.current)
    }
  }, [setHealth, setError])
}
