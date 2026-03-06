"""POST /api/v1/training/start|stop — background training job control."""

from __future__ import annotations

import asyncio
import threading
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.config.hyperparams import TrainingConfig
from backend.config.settings import get_settings

router = APIRouter()

# In-memory job registry: job_id → status dict
_job_registry: dict[str, dict[str, Any]] = {}


class TrainingStartRequest(BaseModel):
    config_path: str = "configs/hyperparams/ppo_default.yaml"
    total_timesteps: int | None = None
    n_envs: int = 4


class TrainingJobStatus(BaseModel):
    job_id: str
    status: str  # "pending" | "running" | "completed" | "failed"
    config_path: str
    started_at: str | None = None
    ended_at: str | None = None
    error: str | None = None


@router.post("/training/start", response_model=TrainingJobStatus)
async def start_training(req: TrainingStartRequest) -> TrainingJobStatus:
    # Check for already-running job
    for job in _job_registry.values():
        if job["status"] == "running":
            raise HTTPException(
                status_code=409,
                detail=f"Training job {job['job_id']} is already running",
            )

    job_id = str(uuid.uuid4())[:8]
    import datetime

    job: dict[str, Any] = {
        "job_id": job_id,
        "status": "pending",
        "config_path": req.config_path,
        "started_at": datetime.datetime.utcnow().isoformat(),
        "ended_at": None,
        "error": None,
    }
    _job_registry[job_id] = job

    # Launch training in a background thread
    thread = threading.Thread(
        target=_run_training,
        args=(job_id, req.config_path, req.total_timesteps, req.n_envs),
        daemon=True,
    )
    thread.start()
    job["status"] = "running"

    return TrainingJobStatus(**job)


@router.post("/training/stop")
async def stop_training() -> dict[str, str]:
    running = [j for j in _job_registry.values() if j["status"] == "running"]
    if not running:
        raise HTTPException(status_code=404, detail="No running training job")

    # Signal stop (best-effort via flag; SB3 doesn't support mid-epoch kill natively)
    for job in running:
        job["status"] = "stopping"

    return {"message": "Stop signal sent", "jobs": [j["job_id"] for j in running]}


@router.get("/training/status", response_model=list[TrainingJobStatus])
async def training_status() -> list[TrainingJobStatus]:
    return [TrainingJobStatus(**j) for j in _job_registry.values()]


def _run_training(
    job_id: str,
    config_path: str,
    total_timesteps: int | None,
    n_envs: int,
) -> None:
    import datetime
    job = _job_registry[job_id]
    try:
        from backend.training.train import train
        config = TrainingConfig.from_yaml(Path(config_path))
        train(config, total_timesteps=total_timesteps, n_envs=n_envs)
        job["status"] = "completed"
    except Exception as exc:
        job["status"] = "failed"
        job["error"] = str(exc)
    finally:
        job["ended_at"] = datetime.datetime.utcnow().isoformat()
