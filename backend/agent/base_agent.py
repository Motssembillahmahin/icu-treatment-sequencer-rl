"""Protocol / ABC interface for RL agents."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class BaseAgent(Protocol):
    """Minimal interface every RL agent must satisfy."""

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> tuple[np.ndarray, Any | None]:
        """
        Given an observation, return (action, state).
        `state` is None for non-recurrent policies.
        """
        ...

    def learn(
        self,
        total_timesteps: int,
        callback: Any | None = None,
        **kwargs: Any,
    ) -> "BaseAgent":
        """Train the agent for `total_timesteps` environment steps."""
        ...

    def save(self, path: Path | str) -> None:
        """Persist the agent to disk."""
        ...

    @classmethod
    def load(cls, path: Path | str, env: Any | None = None) -> "BaseAgent":
        """Load a saved agent from disk."""
        ...
