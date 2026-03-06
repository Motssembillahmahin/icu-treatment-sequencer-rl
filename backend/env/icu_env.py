"""Core Gymnasium ICU treatment sequencing environment."""

from __future__ import annotations

from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from backend.env.actions import Action
from backend.env.patient_state import PatientState
from backend.env.physiology import apply_action_effects, apply_physiology_step
from backend.env.reward import compute_reward
from backend.env.spaces import make_observation_space, make_action_space


MAX_STEPS = 72           # 72 ICU hours per episode
SURVIVAL_CONSECUTIVE = 12  # steps all vitals normal for survival

# Death conditions
DEATH_MAP_THRESHOLD = 30.0      # mmHg
DEATH_MAP_STEPS = 3             # consecutive steps below threshold
DEATH_SPO2_THRESHOLD = 70.0     # %
DEATH_SPO2_STEPS = 2
DEATH_LACTATE_THRESHOLD = 15.0  # mmol/L (single step)


class ICUPatientEnv(gym.Env):
    """
    Gymnasium environment simulating an ICU patient responding to clinical interventions.

    Observation:  Box(0, 1, shape=(13,), dtype=float32)  — normalized vitals
    Action:       Discrete(11)  — clinical interventions (see Action enum)
    Reward:       Composite (see reward.py)
    Termination:  Patient survives (all vitals normal ≥12 steps) or dies
    Truncation:   Episode step limit (72 steps = 72 ICU hours)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode: str | None = None, seed: int | None = None):
        super().__init__()
        self.render_mode = render_mode

        self.observation_space: spaces.Box = make_observation_space()
        self.action_space: spaces.Discrete = make_action_space()

        self._rng = np.random.default_rng(seed)
        self._state: PatientState | None = None
        self._step_count: int = 0

        # Death tracking counters
        self._low_map_steps: int = 0
        self._low_spo2_steps: int = 0
        # Survival tracking
        self._normal_steps: int = 0

        # Info tracking
        self._episode_reward: float = 0.0
        self._prev_state: PatientState | None = None

    # ── Gymnasium interface ────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._state = PatientState.sample_initial(self._rng)
        self._step_count = 0
        self._low_map_steps = 0
        self._low_spo2_steps = 0
        self._normal_steps = 0
        self._episode_reward = 0.0
        self._prev_state = self._state.copy()

        obs = self._state.to_obs()
        info = self._make_info()
        return obs, info

    def step(
        self, action: int | np.integer
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert self._state is not None, "Call reset() before step()"
        act = Action(int(action))

        self._prev_state = self._state.copy()

        # Apply action effects, then physiology dynamics
        state_after_action = apply_action_effects(self._state, act)
        self._state = apply_physiology_step(
            state_after_action, self._rng, self._step_count, MAX_STEPS
        )
        self._step_count += 1

        # Update survival / death counters
        self._update_counters()

        terminated, survived = self._check_termination()
        truncated = self._step_count >= MAX_STEPS

        reward_breakdown = compute_reward(
            prev_state=self._prev_state,
            new_state=self._state,
            action=act,
            terminated=terminated,
            survived=survived,
        )
        reward = reward_breakdown.total
        self._episode_reward += reward

        obs = self._state.to_obs()
        info = self._make_info(reward_breakdown=reward_breakdown, action=act, survived=survived)

        return obs, float(reward), terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode == "human":
            self._render_human()
        return None

    def close(self) -> None:
        pass

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _update_counters(self) -> None:
        s = self._state
        assert s is not None

        # Death counters
        self._low_map_steps = (self._low_map_steps + 1) if s.map < DEATH_MAP_THRESHOLD else 0
        self._low_spo2_steps = (self._low_spo2_steps + 1) if s.spo2 < DEATH_SPO2_THRESHOLD else 0

        # Survival: all monitored vitals in normal range
        from backend.env.spaces import is_normal
        monitored = ["map", "spo2", "lactate", "gcs", "heart_rate", "rr"]
        all_normal = all(is_normal(getattr(s, v), v) for v in monitored)
        self._normal_steps = (self._normal_steps + 1) if all_normal else 0

    def _check_termination(self) -> tuple[bool, bool]:
        """Return (terminated, survived)."""
        s = self._state
        assert s is not None

        # Death conditions
        died = (
            self._low_map_steps >= DEATH_MAP_STEPS
            or self._low_spo2_steps >= DEATH_SPO2_STEPS
            or s.lactate >= DEATH_LACTATE_THRESHOLD
        )
        if died:
            return True, False

        # Survival condition
        survived = self._normal_steps >= SURVIVAL_CONSECUTIVE
        if survived:
            return True, True

        return False, False

    def _make_info(
        self,
        reward_breakdown=None,
        action: Action | None = None,
        survived: bool | None = None,
    ) -> dict[str, Any]:
        s = self._state
        assert s is not None
        info: dict[str, Any] = {
            "step": self._step_count,
            "archetype": s.archetype.value,
            "vitals": {
                "heart_rate": round(s.heart_rate, 1),
                "map": round(s.map, 1),
                "spo2": round(s.spo2, 1),
                "gcs": int(s.gcs),
                "lactate": round(s.lactate, 2),
                "rr": round(s.rr, 1),
                "temperature": round(s.temperature, 1),
                "vasopressor": round(s.vasopressor, 2),
                "fio2": round(s.fio2, 2),
                "peep": round(s.peep, 1),
            },
            "normal_steps_streak": self._normal_steps,
        }
        if reward_breakdown is not None:
            info["reward"] = {
                "vitals": round(reward_breakdown.vitals_reward, 3),
                "action_cost": round(reward_breakdown.action_cost, 3),
                "terminal": round(reward_breakdown.terminal_reward, 3),
                "stability": round(reward_breakdown.stability_bonus, 3),
                "total": round(reward_breakdown.total, 3),
            }
        if action is not None:
            info["action"] = {"id": int(action), "name": action.name}
        if survived is not None:
            info["survived"] = survived
        return info

    def _render_human(self) -> None:
        s = self._state
        if s is None:
            return
        print(
            f"Step {self._step_count:3d} | "
            f"HR={s.heart_rate:5.1f} | MAP={s.map:5.1f} | "
            f"SpO2={s.spo2:5.1f}% | GCS={s.gcs:4.0f} | "
            f"Lac={s.lactate:5.2f} | Vaso={s.vasopressor:.2f}"
        )
