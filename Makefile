.PHONY: setup train train-debug serve serve-frontend test test-env test-api lint fmt tensorboard docker-up docker-down

# ── Setup ──────────────────────────────────────────────────────────────────────
setup:
	uv sync --all-extras
	cp -n .env.example .env || true
	@echo "Setup complete. Run 'make train' to start training."

# ── Training ───────────────────────────────────────────────────────────────────
train:
	uv run train --config configs/hyperparams/ppo_default.yaml

train-debug:
	uv run train --config configs/hyperparams/ppo_default.yaml --total-timesteps 10000 --n-envs 1

# ── Server ─────────────────────────────────────────────────────────────────────
serve:
	uv run uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload

serve-frontend:
	cd frontend && npm run dev

# ── Tests ──────────────────────────────────────────────────────────────────────
test:
	uv run pytest backend/tests/ -v

test-env:
	uv run pytest backend/tests/test_env/ -v

test-api:
	uv run pytest backend/tests/test_api/ -v

# ── Code Quality ───────────────────────────────────────────────────────────────
lint:
	uv run ruff check backend/
	uv run mypy backend/

fmt:
	uv run ruff format backend/
	uv run ruff check --fix backend/

# ── TensorBoard ────────────────────────────────────────────────────────────────
tensorboard:
	uv run tensorboard --logdir runs/

# ── Docker ─────────────────────────────────────────────────────────────────────
docker-up:
	docker compose up --build -d

docker-down:
	docker compose down
