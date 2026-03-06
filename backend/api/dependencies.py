"""FastAPI dependency injection: agent singleton, env factory, DB."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import Depends

from backend.config.settings import Settings, get_settings
from backend.training.replay_buffer import EpisodeDB

# ── Agent singleton ────────────────────────────────────────────────────────────
_agent_singleton = None


def get_agent():
    """Return the loaded agent, or None if no model is available."""
    return _agent_singleton


def load_agent_from_path(path: Path, agent_name: str = "ppo") -> None:
    """Load agent into the singleton (call during lifespan or training completion)."""
    global _agent_singleton
    from backend.agent.agent_factory import load_agent
    _agent_singleton = load_agent(agent_name, path, env=None)


def clear_agent() -> None:
    global _agent_singleton
    _agent_singleton = None


# ── DB dependency (lazily initialized singleton keyed by db path) ─────────────
_db_instances: dict[str, EpisodeDB] = {}


async def get_episode_db(settings: Settings = Depends(get_settings)) -> EpisodeDB:
    key = str(settings.episodes_db)
    if key not in _db_instances:
        db = EpisodeDB(settings.episodes_db)
        await db.initialize()
        _db_instances[key] = db
    return _db_instances[key]


# ── Settings dependency ────────────────────────────────────────────────────────
SettingsDep = Annotated[Settings, Depends(get_settings)]
