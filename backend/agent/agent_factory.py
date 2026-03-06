"""Agent registry and factory function."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.agent.dqn_agent import DQNAgent
from backend.agent.ppo_agent import PPOAgent

_REGISTRY: dict[str, type] = {
    "ppo": PPOAgent,
    "dqn": DQNAgent,
}


def create_agent(name: str, env: Any, config: dict[str, Any] | None = None) -> PPOAgent | DQNAgent:
    """
    Instantiate an agent by name.

    Args:
        name:   Agent key (e.g. "ppo", "dqn")
        env:    Gymnasium / VecEnv environment
        config: Keyword arguments forwarded to the agent constructor

    Returns:
        Agent instance implementing the BaseAgent protocol
    """
    name_lower = name.lower()
    if name_lower not in _REGISTRY:
        raise ValueError(f"Unknown agent '{name}'. Available: {list(_REGISTRY)}")
    cls = _REGISTRY[name_lower]
    return cls(env, **(config or {}))


def load_agent(
    name: str,
    path: Path | str,
    env: Any | None = None,
) -> PPOAgent | DQNAgent:
    """Load a persisted agent from disk."""
    name_lower = name.lower()
    if name_lower not in _REGISTRY:
        raise ValueError(f"Unknown agent '{name}'")
    cls = _REGISTRY[name_lower]
    return cls.load(path, env=env)


def available_agents() -> list[str]:
    return list(_REGISTRY.keys())
