"""Training callbacks: TensorBoard metrics, checkpointing, SQLite episode logging."""

from __future__ import annotations

import asyncio
import datetime
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback

from backend.training.replay_buffer import EpisodeDB


class EpisodeMetricsCallback(BaseCallback):
    """
    Logs episode metrics to SQLite and TensorBoard after each episode completion.
    Works with SubprocVecEnv by tracking the `dones` signals.
    """

    def __init__(
        self,
        db: EpisodeDB,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.db = db
        self._episode_count = 0
        self._episode_rewards: list[float] = []
        self._step_buffer: list[dict[str, Any]] = []
        self._current_start: str = datetime.datetime.utcnow().isoformat()
        self._loop: asyncio.AbstractEventLoop | None = None

    def _on_training_start(self) -> None:
        try:
            self._loop = asyncio.get_event_loop()
        except RuntimeError:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
        # Initialize DB tables
        self._run_async(self.db.initialize())

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        rewards = self.locals.get("rewards", [])

        for i, (done, info) in enumerate(zip(dones, infos)):
            reward = float(rewards[i]) if i < len(rewards) else 0.0
            self._episode_rewards.append(reward)

            step_record: dict[str, Any] = {
                "step": info.get("step", 0),
                "action_id": info.get("action", {}).get("id", 0),
                "action_name": info.get("action", {}).get("name", "NOOP"),
                "reward": reward,
                "vitals": info.get("vitals", {}),
                "terminated": done,
            }
            self._step_buffer.append(step_record)

            if done:
                self._flush_episode(info)

        return True

    def _flush_episode(self, final_info: dict[str, Any]) -> None:
        self._episode_count += 1
        total_reward = sum(self._episode_rewards)
        now = datetime.datetime.utcnow().isoformat()

        self._run_async(
            self.db.save_episode(
                archetype=final_info.get("archetype", "unknown"),
                total_steps=len(self._step_buffer),
                total_reward=total_reward,
                survived=final_info.get("survived", False),
                started_at=self._current_start,
                ended_at=now,
                steps=self._step_buffer,
            )
        )

        # Log to TensorBoard via SB3 logger
        if self.logger:
            self.logger.record("episode/total_reward", total_reward)
            self.logger.record("episode/length", len(self._step_buffer))
            self.logger.record("episode/survived", int(final_info.get("survived", False)))
            self.logger.dump(self.num_timesteps)

        # Reset buffers
        self._episode_rewards = []
        self._step_buffer = []
        self._current_start = datetime.datetime.utcnow().isoformat()

    def _run_async(self, coro: Any) -> None:
        if self._loop and self._loop.is_running():
            asyncio.ensure_future(coro, loop=self._loop)
        else:
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(coro)
            finally:
                loop.close()


def make_checkpoint_callback(save_path: Path, save_freq: int = 50_000) -> CheckpointCallback:
    save_path.mkdir(parents=True, exist_ok=True)
    return CheckpointCallback(
        save_freq=save_freq,
        save_path=str(save_path),
        name_prefix="icu_agent",
        verbose=1,
    )
