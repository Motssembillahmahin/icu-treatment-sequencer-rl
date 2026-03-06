"""CLI entry point for training: uv run train --config configs/hyperparams/ppo_default.yaml"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CallbackList

from backend.agent.agent_factory import create_agent
from backend.config.hyperparams import TrainingConfig
from backend.config.settings import get_settings
from backend.env.icu_env import ICUPatientEnv
from backend.training.callbacks import EpisodeMetricsCallback, make_checkpoint_callback
from backend.training.replay_buffer import EpisodeDB

console = Console()


def build_vec_env(n_envs: int, seed: int | None = None) -> VecNormalize:
    vec_env = make_vec_env(
        ICUPatientEnv,
        n_envs=n_envs,
        seed=seed,
        vec_env_cls=SubprocVecEnv,
    )
    return VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)


def train(config: TrainingConfig, total_timesteps: int | None = None, n_envs: int | None = None) -> None:
    settings = get_settings()
    settings.ensure_dirs()

    effective_timesteps = total_timesteps or config.total_timesteps
    effective_n_envs = n_envs or config.n_envs

    console.rule("[bold blue]ICU Treatment Sequencer RL — Training")
    console.print(f"  Agent:      [cyan]{config.agent}[/]")
    console.print(f"  Timesteps:  [cyan]{effective_timesteps:,}[/]")
    console.print(f"  Envs:       [cyan]{effective_n_envs}[/]")
    console.print(f"  Models dir: [cyan]{settings.models_dir}[/]")

    vec_env = build_vec_env(effective_n_envs, seed=config.seed)

    agent_kwargs = config.agent_hyperparams()
    agent_kwargs["verbose"] = config.verbose
    agent_kwargs["tensorboard_log"] = str(settings.runs_dir)

    agent = create_agent(config.agent, vec_env, agent_kwargs)

    db = EpisodeDB(settings.episodes_db)
    metrics_cb = EpisodeMetricsCallback(db=db, verbose=1)
    checkpoint_cb = make_checkpoint_callback(
        save_path=settings.models_dir / config.agent,
        save_freq=max(config.checkpoint_freq // effective_n_envs, 1),
    )
    callback = CallbackList([metrics_cb, checkpoint_cb])

    console.print("\n[bold green]Starting training...[/]\n")
    agent.learn(total_timesteps=effective_timesteps, callback=callback)

    final_path = settings.models_dir / f"{config.agent}_final"
    agent.save(final_path)
    console.print(f"\n[bold green]Training complete. Model saved to {final_path}[/]")

    vec_env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an ICU RL agent")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/hyperparams/ppo_default.yaml"),
        help="Path to YAML hyperparameter config",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Override total training timesteps",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=None,
        help="Number of parallel environments",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default=None,
        choices=["ppo", "dqn"],
        help="Override agent type from config",
    )
    args = parser.parse_args()

    config = TrainingConfig.from_yaml(args.config)
    if args.agent:
        config.agent = args.agent

    try:
        train(config, total_timesteps=args.total_timesteps, n_envs=args.n_envs)
    except KeyboardInterrupt:
        console.print("\n[yellow]Training interrupted by user.[/]")
        sys.exit(0)


if __name__ == "__main__":
    main()
