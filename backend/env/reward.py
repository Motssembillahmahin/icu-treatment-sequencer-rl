"""Composable reward function for ICU treatment sequencing."""

from __future__ import annotations

from dataclasses import dataclass

from backend.env.actions import Action
from backend.env.patient_state import PatientState


# ── Vital reward weights ───────────────────────────────────────────────────────
VITAL_WEIGHTS = {
    "map": 2.0,
    "spo2": 2.0,
    "lactate": 1.5,
    "gcs": 1.5,
    "heart_rate": 1.0,
    "rr": 0.5,
    "temperature": 0.5,
}

# Normal / warning / critical thresholds (raw values)
THRESHOLDS = {
    "map":         {"normal": (70, 105), "warning": (55, 120), "critical": (0, 30)},
    "spo2":        {"normal": (95, 100), "warning": (88, 94),  "critical": (0, 70)},
    "lactate":     {"normal": (0, 2.0),  "warning": (2.0, 4.0), "critical": (10, 999)},
    "gcs":         {"normal": (13, 15),  "warning": (9, 12),   "critical": (0, 8)},
    "heart_rate":  {"normal": (60, 100), "warning": (50, 130), "critical": (0, 40)},
    "rr":          {"normal": (12, 20),  "warning": (8, 30),   "critical": (0, 6)},
    "temperature": {"normal": (36.5, 37.5), "warning": (35, 39), "critical": (0, 34)},
}

ACTION_COST = -0.1          # cost for any non-NOOP action
TERMINAL_SURVIVAL = 50.0
TERMINAL_DEATH = -50.0
STABILITY_BONUS_SCALE = 0.2  # per-vital-improvement-delta bonus


@dataclass
class RewardBreakdown:
    vitals_reward: float
    action_cost: float
    terminal_reward: float
    stability_bonus: float

    @property
    def total(self) -> float:
        return self.vitals_reward + self.action_cost + self.terminal_reward + self.stability_bonus


def _vital_reward(value: float, name: str) -> float:
    """Return reward contribution for a single vital sign."""
    weight = VITAL_WEIGHTS.get(name, 0.5)
    thresh = THRESHOLDS.get(name)
    if thresh is None:
        return 0.0

    norm_lo, norm_hi = thresh["normal"]
    warn_lo, warn_hi = thresh["warning"]
    crit_lo, crit_hi = thresh["critical"]

    if norm_lo <= value <= norm_hi:
        return weight
    elif warn_lo <= value <= warn_hi:
        return -0.5 * weight
    elif crit_lo <= value <= crit_hi:
        return -2.0 * weight
    else:
        # Outside warning but not critical: mild penalty
        return -0.5 * weight


def compute_vitals_reward(state: PatientState) -> float:
    """Sum of per-vital rewards."""
    total = 0.0
    for name, _ in VITAL_WEIGHTS.items():
        value = getattr(state, name)
        total += _vital_reward(value, name)
    return total


def compute_action_cost(action: Action) -> float:
    """Penalize non-NOOP actions to encourage parsimony."""
    return 0.0 if action == Action.NOOP else ACTION_COST


def compute_terminal_reward(survived: bool) -> float:
    """Large reward/penalty for episode outcome."""
    return TERMINAL_SURVIVAL if survived else TERMINAL_DEATH


def compute_stability_bonus(prev_state: PatientState, new_state: PatientState) -> float:
    """Dense stability signal: reward improvement in each vital."""
    bonus = 0.0
    for name in VITAL_WEIGHTS:
        prev_val = getattr(prev_state, name)
        new_val = getattr(new_state, name)
        thresh = THRESHOLDS.get(name)
        if thresh is None:
            continue
        norm_lo, norm_hi = thresh["normal"]
        norm_center = (norm_lo + norm_hi) / 2.0
        # Improvement = moved closer to normal center
        prev_dist = abs(prev_val - norm_center)
        new_dist = abs(new_val - norm_center)
        delta = prev_dist - new_dist  # positive = improvement
        bonus += STABILITY_BONUS_SCALE * delta
    return bonus


def compute_reward(
    prev_state: PatientState,
    new_state: PatientState,
    action: Action,
    terminated: bool,
    survived: bool,
) -> RewardBreakdown:
    """Compute full reward breakdown for one environment step."""
    vitals = compute_vitals_reward(new_state)
    cost = compute_action_cost(action)
    terminal = compute_terminal_reward(survived) if terminated else 0.0
    stability = compute_stability_bonus(prev_state, new_state)
    return RewardBreakdown(
        vitals_reward=vitals,
        action_cost=cost,
        terminal_reward=terminal,
        stability_bonus=stability,
    )
