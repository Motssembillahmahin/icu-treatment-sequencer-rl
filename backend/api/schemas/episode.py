"""Episode replay Pydantic schemas."""

from __future__ import annotations

from pydantic import BaseModel


class StepRecord(BaseModel):
    step: int
    action_id: int
    action_name: str
    reward: float
    vitals: dict[str, float]
    terminated: bool


class EpisodeSummary(BaseModel):
    id: int
    archetype: str
    total_steps: int
    total_reward: float
    survived: bool
    started_at: str
    ended_at: str


class EpisodeReplay(EpisodeSummary):
    steps: list[StepRecord]
