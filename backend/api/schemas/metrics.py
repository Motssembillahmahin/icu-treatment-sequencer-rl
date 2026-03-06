"""Training metrics Pydantic schemas."""

from __future__ import annotations

from pydantic import BaseModel


class MetricPoint(BaseModel):
    timestep: int
    episode: int
    mean_reward: float | None
    mean_ep_length: float | None
    loss: float | None
    logged_at: str


class TrainingMetrics(BaseModel):
    points: list[MetricPoint]
    total_episodes: int
    latest_timestep: int
