"""Stable-Baselines3 PPO wrapper implementing the BaseAgent protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv


class PPOAgent:
    """Delegates to SB3 PPO without subclassing."""

    def __init__(self, env: VecEnv | Any, **kwargs: Any) -> None:
        self._model = PPO(policy="MlpPolicy", env=env, **kwargs)

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
    ) -> "PPOAgent":
        self._model.learn(total_timesteps=total_timesteps, callback=callback, **kwargs)
        return self

    def save(self, path: Path | str) -> None:
        self._model.save(str(path))

    @classmethod
    def load(cls, path: Path | str, env: Any | None = None) -> "PPOAgent":
        instance = cls.__new__(cls)
        instance._model = PPO.load(str(path), env=env)
        return instance

    @property
    def policy(self) -> Any:
        return self._model.policy

    def get_action_probs(self, observation: np.ndarray) -> np.ndarray:
        """Return softmax action probabilities for a single observation."""
        import torch
        obs_tensor = self._model.policy.obs_to_tensor(observation)[0]
        with torch.no_grad():
            distribution = self._model.policy.get_distribution(obs_tensor)
            probs = distribution.distribution.probs
        return probs.cpu().numpy().flatten()
