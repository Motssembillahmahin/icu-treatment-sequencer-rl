"""Simplified compartmental physiological model for ICU patient dynamics."""

from __future__ import annotations

import numpy as np

from backend.env.actions import Action
from backend.env.patient_state import PatientState


# ── Physiological constants ────────────────────────────────────────────────────
HR_MIN, HR_MAX = 20.0, 250.0
MAP_MIN, MAP_MAX = 20.0, 160.0
SPO2_MIN, SPO2_MAX = 50.0, 100.0
GCS_MIN, GCS_MAX = 3.0, 15.0
LACTATE_MIN, LACTATE_MAX = 0.5, 20.0
RR_MIN, RR_MAX = 4.0, 60.0
TEMP_MIN, TEMP_MAX = 34.0, 42.0
FIO2_MIN, FIO2_MAX = 0.21, 1.0
PEEP_MIN, PEEP_MAX = 0.0, 20.0
VASO_MIN, VASO_MAX = 0.0, 1.0
FB_MIN, FB_MAX = -5000.0, 5000.0

# Noise standard deviations per vital per step
NOISE_STD = {
    "heart_rate": 2.5,
    "map": 2.0,
    "spo2": 0.5,
    "gcs": 0.0,   # GCS changes deterministically
    "lactate": 0.15,
    "rr": 1.0,
    "temperature": 0.05,
    "fio2": 0.0,  # fio2 is a controlled variable
    "peep": 0.0,  # peep is a controlled variable
    "vasopressor": 0.0,  # controlled variable
    "fluid_balance": 50.0,  # insensible losses variance
}


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def apply_action_effects(state: PatientState, action: Action) -> PatientState:
    """Apply immediate action effects to a copy of the state."""
    s = state.copy()

    if action == Action.BOLUS_250:
        s.map += 4.0
        s.fluid_balance += 250.0
        s.heart_rate -= 4.0

    elif action == Action.BOLUS_500:
        s.map += 7.0
        s.fluid_balance += 500.0
        s.heart_rate -= 6.0

    elif action == Action.VASOPRESSOR_UP:
        s.vasopressor = _clip(s.vasopressor + 0.1, VASO_MIN, VASO_MAX)

    elif action == Action.VASOPRESSOR_DOWN:
        s.vasopressor = _clip(s.vasopressor - 0.1, VASO_MIN, VASO_MAX)

    elif action == Action.FIO2_UP:
        s.fio2 = _clip(s.fio2 + 0.1, FIO2_MIN, FIO2_MAX)

    elif action == Action.FIO2_DOWN:
        s.fio2 = _clip(s.fio2 - 0.1, FIO2_MIN, FIO2_MAX)

    elif action == Action.PEEP_UP:
        s.peep = _clip(s.peep + 2.0, PEEP_MIN, PEEP_MAX)

    elif action == Action.PEEP_DOWN:
        s.peep = _clip(s.peep - 2.0, PEEP_MIN, PEEP_MAX)

    elif action == Action.SEDATION:
        s.heart_rate -= 8.0
        s.rr -= 3.0
        s.gcs = _clip(s.gcs - 2.0, GCS_MIN, GCS_MAX)

    elif action == Action.EXTUBATE:
        # Remove ventilator — risky if SpO2/RR not stable
        s.fio2 = 0.21
        s.peep = 0.0

    # NOOP: no changes

    return s


def apply_physiology_step(
    state: PatientState,
    rng: np.random.Generator,
    step: int,
    max_steps: int,
) -> PatientState:
    """Apply one step of physiological dynamics (coupling effects + noise)."""
    s = state.copy()
    s.time_in_icu += 1.0
    s.episode_step_pct = step / max_steps

    # ── Vasopressor coupling ───────────────────────────────────────────────────
    vaso = s.vasopressor
    if vaso > 0:
        map_effect = vaso * 20.0  # vasopressor raises MAP
        hr_effect = vaso * 15.0   # also raises HR
        # High-dose vasopressor → ischemia → rising lactate
        lactate_effect = max(0.0, (vaso - 0.6) * 1.5)
        s.map += map_effect
        s.heart_rate += hr_effect
        s.lactate += lactate_effect
    else:
        # No vasopressor → gradual MAP drift down (if MAP was low to begin with)
        if s.map < 65.0:
            s.map -= 2.0

    # ── PEEP coupling (venous return / oxygenation) ────────────────────────────
    if s.peep > 0:
        spo2_improvement = s.peep * 0.8  # PEEP improves SpO2
        map_reduction = s.peep * 0.3     # PEEP reduces venous return → lower MAP
        s.spo2 += spo2_improvement
        s.map -= map_reduction

    # ── FiO2 coupling ─────────────────────────────────────────────────────────
    fio2_above_room_air = max(0.0, s.fio2 - 0.21)
    s.spo2 += fio2_above_room_air * 15.0  # supplemental O2 raises SpO2

    # ── Low MAP → lactate cascade (tissue ischemia) ───────────────────────────
    if s.map < 55.0:
        ischemia_severity = (55.0 - s.map) / 55.0
        s.lactate += ischemia_severity * 0.8
        s.gcs = _clip(s.gcs - ischemia_severity * 0.5, GCS_MIN, GCS_MAX)

    # ── High lactate → compensatory tachycardia ───────────────────────────────
    if s.lactate > 2.0:
        s.heart_rate += (s.lactate - 2.0) * 2.0
        s.rr += (s.lactate - 2.0) * 0.5

    # ── Natural mean-reversion (mild homeostasis) ────────────────────────────
    s.heart_rate += (75.0 - s.heart_rate) * 0.02
    s.rr += (16.0 - s.rr) * 0.02
    s.temperature += (37.0 - s.temperature) * 0.03

    # ── Lactate clearance when MAP is adequate ────────────────────────────────
    if s.map >= 65.0:
        clearance = 0.05 * s.lactate * (s.map - 64.0) / 100.0
        s.lactate = max(0.5, s.lactate - clearance)

    # ── Fluid balance: insensible losses ─────────────────────────────────────
    s.fluid_balance -= 50.0  # ~50 mL/hr insensible loss

    # ── Sepsis/infection dynamics: temperature drives tachycardia ─────────────
    if s.temperature > 38.0:
        s.heart_rate += (s.temperature - 38.0) * 5.0
    elif s.temperature < 36.0:
        s.heart_rate -= (36.0 - s.temperature) * 4.0

    # ── Add stochastic noise ──────────────────────────────────────────────────
    s.heart_rate += float(rng.normal(0, NOISE_STD["heart_rate"]))
    s.map += float(rng.normal(0, NOISE_STD["map"]))
    s.spo2 += float(rng.normal(0, NOISE_STD["spo2"]))
    s.lactate += float(rng.normal(0, NOISE_STD["lactate"]))
    s.rr += float(rng.normal(0, NOISE_STD["rr"]))
    s.temperature += float(rng.normal(0, NOISE_STD["temperature"]))
    s.fluid_balance += float(rng.normal(0, NOISE_STD["fluid_balance"]))

    # ── Hard clamp all values to physiological bounds ─────────────────────────
    s.heart_rate = _clip(s.heart_rate, HR_MIN, HR_MAX)
    s.map = _clip(s.map, MAP_MIN, MAP_MAX)
    s.spo2 = _clip(s.spo2, SPO2_MIN, SPO2_MAX)
    s.gcs = _clip(round(s.gcs), GCS_MIN, GCS_MAX)
    s.lactate = _clip(s.lactate, LACTATE_MIN, LACTATE_MAX)
    s.rr = _clip(s.rr, RR_MIN, RR_MAX)
    s.temperature = _clip(s.temperature, TEMP_MIN, TEMP_MAX)
    s.fio2 = _clip(s.fio2, FIO2_MIN, FIO2_MAX)
    s.peep = _clip(s.peep, PEEP_MIN, PEEP_MAX)
    s.vasopressor = _clip(s.vasopressor, VASO_MIN, VASO_MAX)
    s.fluid_balance = _clip(s.fluid_balance, FB_MIN, FB_MAX)

    return s
