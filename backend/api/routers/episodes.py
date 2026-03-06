"""GET /api/v1/episodes — episode replay viewer."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from backend.api.dependencies import get_episode_db
from backend.api.schemas.episode import EpisodeReplay, EpisodeSummary, StepRecord
from backend.training.replay_buffer import EpisodeDB

router = APIRouter()


@router.get("/episodes", response_model=list[EpisodeSummary])
async def list_episodes(
    limit: int = 20,
    offset: int = 0,
    db: EpisodeDB = Depends(get_episode_db),
) -> list[EpisodeSummary]:
    rows = await db.list_episodes(limit=limit, offset=offset)
    return [EpisodeSummary(**r) for r in rows]


@router.get("/episodes/{episode_id}", response_model=EpisodeReplay)
async def get_episode(
    episode_id: int,
    db: EpisodeDB = Depends(get_episode_db),
) -> EpisodeReplay:
    data = await db.get_episode(episode_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Episode {episode_id} not found")

    steps = [
        StepRecord(
            step=s["step"],
            action_id=s["action_id"],
            action_name=s["action_name"],
            reward=s["reward"],
            vitals=s["vitals"],
            terminated=bool(s["terminated"]),
        )
        for s in data.get("steps", [])
    ]
    return EpisodeReplay(**{k: v for k, v in data.items() if k != "steps"}, steps=steps)
