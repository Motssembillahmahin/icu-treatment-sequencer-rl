"""SQLite-backed episode storage using aiosqlite."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import aiosqlite


CREATE_EPISODES_TABLE = """
CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    archetype TEXT NOT NULL,
    total_steps INTEGER NOT NULL,
    total_reward REAL NOT NULL,
    survived INTEGER NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT NOT NULL
)
"""

CREATE_STEPS_TABLE = """
CREATE TABLE IF NOT EXISTS episode_steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id INTEGER NOT NULL REFERENCES episodes(id),
    step INTEGER NOT NULL,
    action_id INTEGER NOT NULL,
    action_name TEXT NOT NULL,
    reward REAL NOT NULL,
    vitals TEXT NOT NULL,  -- JSON blob
    terminated INTEGER NOT NULL DEFAULT 0
)
"""

CREATE_METRICS_TABLE = """
CREATE TABLE IF NOT EXISTS training_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestep INTEGER NOT NULL,
    episode INTEGER NOT NULL,
    mean_reward REAL,
    mean_ep_length REAL,
    loss REAL,
    logged_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""


class EpisodeDB:
    """Async SQLite wrapper for episode replay storage."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    async def initialize(self) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(CREATE_EPISODES_TABLE)
            await db.execute(CREATE_STEPS_TABLE)
            await db.execute(CREATE_METRICS_TABLE)
            await db.commit()

    async def save_episode(
        self,
        archetype: str,
        total_steps: int,
        total_reward: float,
        survived: bool,
        started_at: str,
        ended_at: str,
        steps: list[dict[str, Any]],
    ) -> int:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """INSERT INTO episodes
                   (archetype, total_steps, total_reward, survived, started_at, ended_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (archetype, total_steps, total_reward, int(survived), started_at, ended_at),
            )
            episode_id = cursor.lastrowid
            for step in steps:
                await db.execute(
                    """INSERT INTO episode_steps
                       (episode_id, step, action_id, action_name, reward, vitals, terminated)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        episode_id,
                        step["step"],
                        step["action_id"],
                        step["action_name"],
                        step["reward"],
                        json.dumps(step["vitals"]),
                        int(step.get("terminated", False)),
                    ),
                )
            await db.commit()
            assert episode_id is not None
            return episode_id

    async def get_episode(self, episode_id: int) -> dict[str, Any] | None:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM episodes WHERE id = ?", (episode_id,)
            )
            row = await cursor.fetchone()
            if row is None:
                return None
            episode = dict(row)

            cursor = await db.execute(
                "SELECT * FROM episode_steps WHERE episode_id = ? ORDER BY step",
                (episode_id,),
            )
            step_rows = await cursor.fetchall()
            episode["steps"] = [
                {**dict(s), "vitals": json.loads(s["vitals"])} for s in step_rows
            ]
            return episode

    async def list_episodes(self, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM episodes ORDER BY id DESC LIMIT ? OFFSET ?",
                (limit, offset),
            )
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]

    async def log_metrics(
        self,
        timestep: int,
        episode: int,
        mean_reward: float | None = None,
        mean_ep_length: float | None = None,
        loss: float | None = None,
    ) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """INSERT INTO training_metrics
                   (timestep, episode, mean_reward, mean_ep_length, loss)
                   VALUES (?, ?, ?, ?, ?)""",
                (timestep, episode, mean_reward, mean_ep_length, loss),
            )
            await db.commit()

    async def get_metrics(self, limit: int = 500) -> list[dict[str, Any]]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM training_metrics ORDER BY timestep ASC LIMIT ?",
                (limit,),
            )
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]
