"""Evaluation script: rollout trained agent and report metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

from backend.agent.agent_factory import load_agent
from backend.env.icu_env import ICUPatientEnv


console = Console()


def evaluate(
    agent_name: str,
    model_path: Path,
    n_episodes: int = 20,
    render: bool = False,
    seed: int | None = None,
) -> dict[str, float]:
    """
    Run the agent for n_episodes and return aggregate metrics.
    """
    env = ICUPatientEnv(render_mode="human" if render else None, seed=seed)
    agent = load_agent(agent_name, model_path, env=None)

    rewards: list[float] = []
    lengths: list[int] = []
    survivals: list[bool] = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        ep_reward = 0.0
        ep_steps = 0
        survived = False

        while True:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_steps += 1

            if terminated or truncated:
                survived = info.get("survived", False)
                break

        rewards.append(ep_reward)
        lengths.append(ep_steps)
        survivals.append(survived)
        console.print(
            f"Episode {ep + 1:3d}: reward={ep_reward:8.2f}, "
            f"steps={ep_steps:3d}, survived={survived}"
        )

    metrics = {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_length": float(np.mean(lengths)),
        "survival_rate": float(np.mean(survivals)),
    }

    table = Table(title="Evaluation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    for k, v in metrics.items():
        table.add_row(k, f"{v:.3f}")
    console.print(table)

    env.close()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained ICU RL agent")
    parser.add_argument("--agent", default="ppo", choices=["ppo", "dqn"])
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    evaluate(args.agent, args.model_path, args.episodes, args.render, args.seed)


if __name__ == "__main__":
    main()
