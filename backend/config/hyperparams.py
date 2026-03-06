"""Pydantic models for YAML hyperparameter configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class PPOHyperparams(BaseModel):
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5


class DQNHyperparams(BaseModel):
    learning_rate: float = 1e-4
    buffer_size: int = 100_000
    learning_starts: int = 1000
    batch_size: int = 32
    tau: float = 1.0
    gamma: float = 0.99
    train_freq: int = 4
    gradient_steps: int = 1
    target_update_interval: int = 1000
    exploration_fraction: float = 0.1
    exploration_final_eps: float = 0.05


class EnvConfig(BaseModel):
    max_steps: int = 72
    n_envs: int = 4
    seed: int | None = None


class TrainingConfig(BaseModel):
    agent: str = "ppo"
    total_timesteps: int = 2_000_000
    n_envs: int = 4
    checkpoint_freq: int = 50_000
    eval_freq: int = 10_000
    eval_episodes: int = 10
    log_interval: int = 1
    verbose: int = 1
    seed: int | None = None

    ppo: PPOHyperparams = Field(default_factory=PPOHyperparams)
    dqn: DQNHyperparams = Field(default_factory=DQNHyperparams)
    env: EnvConfig = Field(default_factory=EnvConfig)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "TrainingConfig":
        with open(path) as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}
        return cls(**data)

    def agent_hyperparams(self) -> dict[str, Any]:
        """Return hyperparameters for the configured agent as a plain dict."""
        if self.agent == "ppo":
            return self.ppo.model_dump()
        elif self.agent == "dqn":
            return self.dqn.model_dump()
        return {}
