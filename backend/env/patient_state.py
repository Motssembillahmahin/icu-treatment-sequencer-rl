"""PatientState dataclass representing raw ICU vital signs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from backend.env.spaces import (
    VITAL_NAMES,
    normalize,
    denormalize,
)


class PatientArchetype(Enum):
    SEPTIC_SHOCK = "septic_shock"
    RESPIRATORY_FAILURE = "respiratory_failure"
    CARDIOGENIC_SHOCK = "cardiogenic_shock"
    POST_SURGICAL = "post_surgical"


# Probabilities must sum to 1.0
ARCHETYPE_WEIGHTS = np.array([0.35, 0.25, 0.20, 0.20])
ARCHETYPES = list(PatientArchetype)


@dataclass
class PatientState:
    """Raw (de-normalized) ICU patient vital signs."""

    heart_rate: float = 80.0        # bpm
    map: float = 80.0               # mmHg
    spo2: float = 97.0              # %
    gcs: float = 15.0               # 3–15
    lactate: float = 1.0            # mmol/L
    rr: float = 16.0                # breaths/min
    temperature: float = 37.0       # °C
    fio2: float = 0.21              # fraction (room air)
    peep: float = 0.0               # cmH2O
    vasopressor: float = 0.0        # normalized 0–1
    fluid_balance: float = 0.0      # mL (+ = net fluid gain)
    time_in_icu: float = 0.0        # hours
    episode_step_pct: float = 0.0   # 0–1

    # Metadata (not part of obs vector)
    archetype: PatientArchetype = PatientArchetype.SEPTIC_SHOCK

    def to_obs(self) -> np.ndarray:
        """Convert state to normalized [0,1] observation vector."""
        raw_values = [
            self.heart_rate,
            self.map,
            self.spo2,
            self.gcs,
            self.lactate,
            self.rr,
            self.temperature,
            self.fio2,
            self.peep,
            self.vasopressor,
            self.fluid_balance,
            self.time_in_icu,
            self.episode_step_pct,
        ]
        return np.array(
            [normalize(v, name) for v, name in zip(raw_values, VITAL_NAMES)],
            dtype=np.float32,
        )

    @classmethod
    def from_obs(cls, obs: np.ndarray) -> "PatientState":
        """Reconstruct PatientState from normalized obs vector."""
        raw = {name: denormalize(float(obs[i]), name) for i, name in enumerate(VITAL_NAMES)}
        return cls(**raw)

    def copy(self) -> "PatientState":
        import copy
        return copy.copy(self)

    @classmethod
    def sample_initial(cls, rng: np.random.Generator | None = None) -> "PatientState":
        """Sample a realistic initial state for a random patient archetype."""
        if rng is None:
            rng = np.random.default_rng()
        archetype = ARCHETYPES[rng.choice(len(ARCHETYPES), p=ARCHETYPE_WEIGHTS)]
        return cls._archetype_initial(archetype, rng)

    @classmethod
    def _archetype_initial(cls, archetype: PatientArchetype, rng: np.random.Generator | None = None) -> "PatientState":
        """Return typical initial state for each patient archetype."""
        if rng is None:
            rng = np.random.default_rng()
        np_rng = rng

        if archetype == PatientArchetype.SEPTIC_SHOCK:
            return cls(
                heart_rate=float(np_rng.normal(115, 10)),
                map=float(np_rng.normal(58, 5)),
                spo2=float(np_rng.normal(92, 2)),
                gcs=float(np_rng.normal(12, 2)),
                lactate=float(np_rng.normal(4.5, 0.8)),
                rr=float(np_rng.normal(24, 3)),
                temperature=float(np_rng.normal(38.8, 0.4)),
                fio2=0.40,
                peep=5.0,
                vasopressor=0.0,
                fluid_balance=0.0,
                archetype=archetype,
            )
        elif archetype == PatientArchetype.RESPIRATORY_FAILURE:
            return cls(
                heart_rate=float(np_rng.normal(95, 8)),
                map=float(np_rng.normal(72, 8)),
                spo2=float(np_rng.normal(84, 3)),
                gcs=float(np_rng.normal(13, 1)),
                lactate=float(np_rng.normal(2.0, 0.4)),
                rr=float(np_rng.normal(30, 4)),
                temperature=float(np_rng.normal(37.5, 0.3)),
                fio2=0.60,
                peep=8.0,
                vasopressor=0.0,
                fluid_balance=-200.0,
                archetype=archetype,
            )
        elif archetype == PatientArchetype.CARDIOGENIC_SHOCK:
            return cls(
                heart_rate=float(np_rng.normal(105, 10)),
                map=float(np_rng.normal(52, 7)),
                spo2=float(np_rng.normal(88, 3)),
                gcs=float(np_rng.normal(13, 1)),
                lactate=float(np_rng.normal(3.5, 0.6)),
                rr=float(np_rng.normal(22, 3)),
                temperature=float(np_rng.normal(36.8, 0.3)),
                fio2=0.50,
                peep=5.0,
                vasopressor=0.1,
                fluid_balance=500.0,
                archetype=archetype,
            )
        else:  # POST_SURGICAL
            return cls(
                heart_rate=float(np_rng.normal(88, 8)),
                map=float(np_rng.normal(70, 6)),
                spo2=float(np_rng.normal(93, 2)),
                gcs=float(np_rng.normal(14, 1)),
                lactate=float(np_rng.normal(1.8, 0.4)),
                rr=float(np_rng.normal(18, 2)),
                temperature=float(np_rng.normal(37.2, 0.4)),
                fio2=0.35,
                peep=3.0,
                vasopressor=0.0,
                fluid_balance=-300.0,
                archetype=archetype,
            )

    def __repr__(self) -> str:
        return (
            f"PatientState(HR={self.heart_rate:.0f}, MAP={self.map:.0f}, "
            f"SpO2={self.spo2:.1f}%, GCS={self.gcs:.0f}, Lac={self.lactate:.1f}, "
            f"archetype={self.archetype.value})"
        )
