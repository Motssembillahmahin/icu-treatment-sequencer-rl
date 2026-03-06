"""Observation and action space definitions for the ICU environment."""

import numpy as np
from gymnasium.spaces import Box, Discrete

# ── Observation indices ────────────────────────────────────────────────────────
OBS_HEART_RATE = 0
OBS_MAP = 1
OBS_SPO2 = 2
OBS_GCS = 3
OBS_LACTATE = 4
OBS_RR = 5
OBS_TEMPERATURE = 6
OBS_FIO2 = 7
OBS_PEEP = 8
OBS_VASOPRESSOR = 9
OBS_FLUID_BALANCE = 10
OBS_TIME_IN_ICU = 11
OBS_EPISODE_STEP_PCT = 12

OBS_DIM = 13
NUM_ACTIONS = 11

# ── Raw vital ranges (for de-normalization / display) ─────────────────────────
VITAL_RANGES = {
    "heart_rate":      (20.0, 250.0),   # bpm
    "map":             (20.0, 160.0),   # mmHg
    "spo2":            (50.0, 100.0),   # %
    "gcs":             (3.0, 15.0),     # score
    "lactate":         (0.5, 20.0),     # mmol/L
    "rr":              (4.0, 60.0),     # breaths/min
    "temperature":     (34.0, 42.0),   # °C
    "fio2":            (0.21, 1.0),    # fraction
    "peep":            (0.0, 20.0),    # cmH2O
    "vasopressor":     (0.0, 1.0),     # normalized dose
    "fluid_balance":   (-5000.0, 5000.0),  # mL
    "time_in_icu":     (0.0, 168.0),   # hours
    "episode_step_pct": (0.0, 1.0),    # fraction
}

# Normal ranges (raw values) for each vital
NORMAL_RANGES = {
    "heart_rate":   (60.0, 100.0),
    "map":          (70.0, 105.0),
    "spo2":         (95.0, 100.0),
    "gcs":          (14.0, 15.0),
    "lactate":      (0.5, 2.0),
    "rr":           (12.0, 20.0),
    "temperature":  (36.5, 37.5),
    "fio2":         (0.21, 0.40),
    "peep":         (0.0, 5.0),
    "vasopressor":  (0.0, 0.1),
    "fluid_balance": (-500.0, 500.0),
    "time_in_icu":  (0.0, 168.0),
    "episode_step_pct": (0.0, 1.0),
}

VITAL_NAMES = list(VITAL_RANGES.keys())


def make_observation_space() -> Box:
    """Create the normalized [0, 1] observation space."""
    return Box(
        low=np.zeros(OBS_DIM, dtype=np.float32),
        high=np.ones(OBS_DIM, dtype=np.float32),
        shape=(OBS_DIM,),
        dtype=np.float32,
    )


def make_action_space() -> Discrete:
    """Create the discrete action space with 11 clinical interventions."""
    return Discrete(NUM_ACTIONS)


def normalize(value: float, name: str) -> float:
    """Normalize a raw vital value to [0, 1]."""
    lo, hi = VITAL_RANGES[name]
    return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))


def denormalize(norm_value: float, name: str) -> float:
    """Convert a normalized [0, 1] value back to raw scale."""
    lo, hi = VITAL_RANGES[name]
    return lo + norm_value * (hi - lo)


def is_normal(raw_value: float, name: str) -> bool:
    """Return True if the raw vital value is within normal range."""
    lo, hi = NORMAL_RANGES[name]
    return lo <= raw_value <= hi
