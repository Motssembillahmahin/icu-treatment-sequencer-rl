"""Stable-Baselines3 DQN wrapper — swap-in replacement for PPOAgent."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecEnv


class DQNAgent:
    """Delegates to SB3 DQN without subclassing."""

    def __init__(self, env: VecEnv | Any, **kwargs: Any) -> None:
        self._model = DQN(policy="MlpPolicy", env=env, **kwargs)

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> tuple[np.ndarray, Any | None]:
        action, state = self._model.predict(observation, deterministic=deterministic)
        return action, state

    def learn(
        self,
        total_timesteps: int,
        callback: Any | None = None,
        **kwargs: Any,
    ) -> "DQNAgent":
        self._model.learn(total_timesteps=total_timesteps, callback=callback, **kwargs)
        return self

    def save(self, path: Path | str) -> None:
        self._model.save(str(path))

    @classmethod
    def load(cls, path: Path | str, env: Any | None = None) -> "DQNAgent":
        instance = cls.__new__(cls)
        instance._model = DQN.load(str(path), env=env)
        return instance
