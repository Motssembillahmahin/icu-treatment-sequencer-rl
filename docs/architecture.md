# Architecture

## System Overview

```
┌──────────────────────────────────────────────────────────────────┐
│  React Frontend (Vite + TypeScript + Tailwind + Zustand)         │
│  Dashboard | Training | EpisodeViewer                            │
└────────────────────────┬─────────────────────────────────────────┘
                         │ HTTP REST (polling 2s)
┌────────────────────────▼─────────────────────────────────────────┐
│  FastAPI Backend                                                  │
│  /api/v1/health  /api/v1/inference  /api/v1/metrics              │
│  /api/v1/episodes  /api/v1/training/start|stop                   │
├──────────────────────────────────────────────────────────────────┤
│  Agent Singleton (PPO / DQN via stable-baselines3)               │
├──────────────────────────────────────────────────────────────────┤
│  ICUPatientEnv (Gymnasium)                                       │
│  PhysiologyModel → RewardFunction → PatientState                 │
├──────────────────────────────────────────────────────────────────┤
│  SQLite (aiosqlite)  │  TensorBoard logs  │  Model checkpoints   │
└──────────────────────────────────────────────────────────────────┘
```

## Directory Layout

- `backend/env/`      — Gymnasium ICU environment
- `backend/agent/`    — RL agent wrappers (PPO, DQN)
- `backend/training/` — Training CLI, callbacks, replay buffer
- `backend/api/`      — FastAPI app, routers, schemas
- `backend/config/`   — Pydantic settings + hyperparameter models
- `frontend/src/`     — React application
- `configs/`          — YAML hyperparameter configs + Dockerfiles
- `models/`           — Trained model artifacts (gitignored)
- `runs/`             — TensorBoard logs (gitignored)
- `data/episodes/`    — SQLite episode database (gitignored)
