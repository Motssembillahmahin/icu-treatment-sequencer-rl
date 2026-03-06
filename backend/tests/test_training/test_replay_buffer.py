"""Tests for SQLite episode storage."""

from __future__ import annotations

import pytest


class TestEpisodeDB:
    async def test_save_and_retrieve_episode(self, episode_db):
        steps = [
            {
                "step": i,
                "action_id": 0,
                "action_name": "NOOP",
                "reward": 1.5,
                "vitals": {"heart_rate": 80.0, "map": 75.0},
                "terminated": i == 9,
            }
            for i in range(10)
        ]
        ep_id = await episode_db.save_episode(
            archetype="septic_shock",
            total_steps=10,
            total_reward=15.0,
            survived=False,
            started_at="2024-01-01T00:00:00",
            ended_at="2024-01-01T00:10:00",
            steps=steps,
        )
        assert ep_id is not None and ep_id > 0

        episode = await episode_db.get_episode(ep_id)
        assert episode is not None
        assert episode["archetype"] == "septic_shock"
        assert episode["total_steps"] == 10
        assert len(episode["steps"]) == 10
        assert episode["steps"][0]["action_name"] == "NOOP"

    async def test_get_nonexistent_episode_returns_none(self, episode_db):
        result = await episode_db.get_episode(99999)
        assert result is None

    async def test_list_episodes_pagination(self, episode_db):
        for i in range(5):
            await episode_db.save_episode(
                archetype="post_surgical",
                total_steps=5,
                total_reward=float(i),
                survived=True,
                started_at="2024-01-01T00:00:00",
                ended_at="2024-01-01T00:05:00",
                steps=[],
            )
        all_eps = await episode_db.list_episodes(limit=10)
        assert len(all_eps) == 5

        first_page = await episode_db.list_episodes(limit=3, offset=0)
        assert len(first_page) == 3

    async def test_log_and_retrieve_metrics(self, episode_db):
        await episode_db.log_metrics(
            timestep=1000,
            episode=1,
            mean_reward=5.2,
            mean_ep_length=20.0,
            loss=0.5,
        )
        metrics = await episode_db.get_metrics()
        assert len(metrics) == 1
        assert metrics[0]["timestep"] == 1000
        assert metrics[0]["mean_reward"] == pytest.approx(5.2)
