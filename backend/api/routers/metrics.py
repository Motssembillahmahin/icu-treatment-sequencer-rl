"""GET /api/v1/metrics — training loss/reward curves from SQLite."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.api.dependencies import get_episode_db
from backend.api.schemas.metrics import MetricPoint, TrainingMetrics
from backend.training.replay_buffer import EpisodeDB

router = APIRouter()


@router.get("/metrics", response_model=TrainingMetrics)
async def get_metrics(
    limit: int = 500,
    db: EpisodeDB = Depends(get_episode_db),
) -> TrainingMetrics:
    raw = await db.get_metrics(limit=limit)
    points = [MetricPoint(**r) for r in raw]
    return TrainingMetrics(
        points=points,
        total_episodes=len(points),
        latest_timestep=points[-1].timestep if points else 0,
    )
