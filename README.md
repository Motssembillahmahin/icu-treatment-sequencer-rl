# ICU Treatment Sequencer RL

> Teaching an AI to make life-or-death treatment decisions — by letting it fail safely in simulation, millions of times.

A reinforcement learning system that trains an agent to learn optimal intervention sequences for critically ill ICU patients. The agent observes patient vitals every simulated hour and decides which of 11 clinical actions to take. It gets rewarded when the patient stabilises and penalised when they deteriorate — learning effective treatment strategies entirely through trial and error, with no patient ever at risk.

---

## What It Does

A virtual ICU patient arrives in one of four critical states — septic shock, respiratory failure, cardiogenic shock, or post-surgical instability. Their vitals (heart rate, blood pressure, oxygen saturation, lactate, GCS, and more) evolve each hour based on a physiological model that captures real clinical coupling effects: fluids raise blood pressure, PEEP improves oxygenation but drops blood pressure, vasopressors at high doses damage tissues and raise lactate.

The RL agent sees these 13 vital signs and picks one action per hour from:

| Action | Clinical meaning |
|---|---|
| `NOOP` | Do nothing — watch and wait |
| `BOLUS_250 / BOLUS_500` | IV fluid resuscitation (250 or 500 mL) |
| `VASOPRESSOR_UP/DOWN` | Titrate vasopressor (blood pressure support) |
| `FIO2_UP/DOWN` | Adjust supplemental oxygen |
| `PEEP_UP/DOWN` | Adjust ventilator pressure |
| `SEDATION` | Administer sedation |
| `EXTUBATE` | Remove breathing tube |

Over millions of simulated episodes the agent learns when to give fluids vs. vasopressors, how to balance oxygenation against haemodynamics, and when doing nothing is the right call.

Once trained, a **FastAPI server** exposes the policy as a REST endpoint: send patient vitals, receive a recommended action with confidence score and clinical reasoning tags. A **React dashboard** visualises vitals in real time, shows training curves, and lets you replay any past episode step by step.

---

## Use Cases

**Clinical AI researchers** — benchmark RL algorithms on a medically meaningful environment; swap in new physiology models or reward functions without touching the training infrastructure.

**ML engineers learning RL** — a non-trivial domain problem with a correct Gymnasium implementation, SB3 integration, production API serving, and full test suite.

**Medical students** — explore physiological trade-offs interactively; watch the agent reason through a septic shock episode and understand why it chooses fluids over vasopressors in the first hour.

**Health tech product teams** — reference architecture for clinical decision support APIs: structured inference endpoint, Pydantic schemas shared between Python and TypeScript, episode audit trail, Docker deployment.

**Sepsis protocol researchers** — train exclusively on the septic shock archetype and extract the agent's learned fluid/vasopressor policy for comparison against the Surviving Sepsis Campaign guidelines.

---

## Quick Start

### Prerequisites
- Python 3.11+, [uv](https://docs.astral.sh/uv/)
- Node.js 18+, npm 9+

### Install
```bash
git clone https://github.com/Motssembillahmahin/icu-treatment-sequencer-rl.git
cd icu-treatment-sequencer-rl
make setup
```

### Run tests
```bash
make test
# 30 tests — all should pass, including gymnasium check_env compliance
```

### Train the agent
```bash
make train-debug       # 2048 steps (~30s) — smoke test
make train             # 2M steps — proper training run
```

### Start the API
```bash
make serve
# http://localhost:8000/api/v1/health
# http://localhost:8000/docs  ← Swagger UI
```

### Start the frontend
```bash
cd frontend && npm install && npm run dev
# http://localhost:5173
```

### Docker (both services)
```bash
make docker-up
# Backend → http://localhost:8000
# Frontend → http://localhost:5173
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v1/health` | Liveness check + model load status |
| `POST` | `/api/v1/inference` | Patient vitals → recommended action |
| `GET` | `/api/v1/metrics` | Training reward/loss curves |
| `GET` | `/api/v1/episodes` | List recorded episodes |
| `GET` | `/api/v1/episodes/{id}` | Full step-by-step episode replay |
| `POST` | `/api/v1/training/start` | Launch background training job |
| `POST` | `/api/v1/training/stop` | Stop running job |

**Example inference request:**
```bash
curl -s -X POST http://localhost:8000/api/v1/inference \
  -H "Content-Type: application/json" \
  -d '{
    "vitals": {
      "heart_rate": 118, "map": 54, "spo2": 90, "gcs": 12,
      "lactate": 5.2, "rr": 26, "temperature": 39.1,
      "fio2": 0.45, "peep": 5.0, "vasopressor": 0.0, "fluid_balance": -200
    },
    "time_in_icu": 3.0, "episode_step_pct": 0.04, "deterministic": true
  }'
```

---

## Project Structure

```
├── backend/
│   ├── env/          ← Gymnasium ICUPatientEnv (13-dim obs, 11 actions)
│   ├── agent/        ← PPO and DQN wrappers (stable-baselines3)
│   ├── training/     ← CLI, callbacks, SQLite episode storage
│   ├── api/          ← FastAPI routers, schemas, dependencies
│   └── config/       ← Pydantic settings + YAML hyperparameters
├── frontend/src/
│   ├── pages/        ← Dashboard, Training, EpisodeViewer
│   ├── components/   ← VitalsChart, VitalsGauge, ActionCard, RewardCurve
│   ├── hooks/        ← useVitalsPolling, useTrainingMetrics
│   └── stores/       ← Zustand patient store
├── configs/
│   └── hyperparams/  ← ppo_default.yaml, dqn_default.yaml, env_config.yaml
├── docs/             ← Local analysis docs (not tracked in git)
├── models/           ← Trained artifacts (gitignored)
├── runs/             ← TensorBoard logs (gitignored)
└── data/episodes/    ← SQLite episode database (gitignored)
```

---

## Makefile Reference

```bash
make setup            # install all dependencies
make train            # full PPO training (2M steps, 4 envs)
make train-debug      # quick smoke run (2048 steps)
make serve            # FastAPI on :8000 with hot reload
make serve-frontend   # Vite dev server on :5173
make test             # run all 30 backend tests
make lint             # ruff + mypy
make fmt              # auto-format
make tensorboard      # TensorBoard at :6006
make docker-up        # start both services
make docker-down      # stop services
```

---

## Tech Stack

**Backend:** Python 3.11+, uv, Gymnasium, stable-baselines3, PyTorch, FastAPI, Pydantic v2, aiosqlite, TensorBoard

**Frontend:** React 18, TypeScript, Vite, Tailwind CSS, Recharts, Zustand, Axios

**Tooling:** ruff, mypy, pytest, pytest-asyncio, httpx, pre-commit, Docker Compose

---

> **Disclaimer:** This is a research and educational tool. The physiological model is a deliberate simplification and is not validated on real patient data. It is not intended for use in actual clinical care.
