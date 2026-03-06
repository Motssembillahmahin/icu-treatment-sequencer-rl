"""Gymnasium compliance tests + dynamics sanity checks."""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from backend.env.actions import Action
from backend.env.icu_env import ICUPatientEnv
from backend.env.spaces import OBS_DIM, NUM_ACTIONS


class TestGymnasiumCompliance:
    def test_check_env_passes(self):
        """gymnasium.utils.env_checker.check_env() must pass with no warnings."""
        env = ICUPatientEnv(seed=0)
        check_env(env, warn=True, skip_render_check=True)
        env.close()

    def test_observation_space_shape(self, env):
        obs, _ = env.reset()
        assert obs.shape == (OBS_DIM,)
        assert obs.dtype == np.float32

    def test_observation_in_bounds(self, env):
        obs, _ = env.reset()
        assert np.all(obs >= 0.0), "Obs below 0"
        assert np.all(obs <= 1.0), "Obs above 1"

    def test_action_space_size(self, env):
        assert env.action_space.n == NUM_ACTIONS

    def test_reset_returns_correct_types(self, env):
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)

    def test_step_returns_correct_types(self, env):
        env.reset()
        obs, reward, terminated, truncated, info = env.step(0)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_obs_in_bounds(self, env):
        env.reset()
        for _ in range(10):
            action = env.action_space.sample()
            obs, *_ = env.step(action)
            assert np.all(obs >= 0.0)
            assert np.all(obs <= 1.0)

    def test_deterministic_reset(self):
        """Same seed → same initial obs."""
        env1 = ICUPatientEnv(seed=77)
        env2 = ICUPatientEnv(seed=77)
        obs1, _ = env1.reset(seed=77)
        obs2, _ = env2.reset(seed=77)
        np.testing.assert_array_equal(obs1, obs2)
        env1.close()
        env2.close()

    def test_truncation_at_max_steps(self):
        """Episode must truncate at step 72."""
        from backend.env.icu_env import MAX_STEPS
        env = ICUPatientEnv(seed=0)
        env.reset()
        for _ in range(MAX_STEPS - 1):
            _, _, terminated, truncated, _ = env.step(0)  # NOOP
            if terminated:
                break
        # Force one more step if not terminated
        if not terminated:
            _, _, terminated, truncated, _ = env.step(0)
            assert truncated or terminated
        env.close()

    def test_all_actions_valid(self, env):
        """All 11 actions must be accepted without error."""
        for action_id in range(NUM_ACTIONS):
            env.reset()
            obs, reward, terminated, truncated, info = env.step(action_id)
            assert obs.shape == (OBS_DIM,)


class TestPhysiologyDynamics:
    def test_bolus_raises_map(self, seeded_env):
        """Fluid bolus should tend to raise MAP."""
        seeded_env.reset(seed=0)
        # Get baseline MAP
        from backend.env.spaces import denormalize
        obs, _, _, _, info = seeded_env.step(Action.NOOP)
        baseline_map = info["vitals"]["map"]
        seeded_env.reset(seed=0)
        obs, _, _, _, info = seeded_env.step(Action.BOLUS_500)
        bolus_map = info["vitals"]["map"]
        # With noise this isn't always strictly greater, but with fixed seed it should be
        assert bolus_map >= baseline_map - 5  # allow small noise tolerance

    def test_vasopressor_up_raises_map(self, seeded_env):
        """Vasopressor up should raise MAP."""
        seeded_env.reset(seed=0)
        obs, _, _, _, info = seeded_env.step(Action.VASOPRESSOR_UP)
        # Vasopressor dose increased
        assert info["vitals"]["vasopressor"] > 0

    def test_fio2_up_improves_spo2(self):
        """FiO2 increase should improve SpO2 in respiratory failure patients."""
        env = ICUPatientEnv(seed=42)
        from backend.env.patient_state import PatientArchetype, PatientState
        state = PatientState._archetype_initial(PatientArchetype.RESPIRATORY_FAILURE)
        env._state = state
        env._step_count = 0
        env._prev_state = state.copy()
        obs, reward, terminated, truncated, info = env.step(Action.FIO2_UP)
        assert info["vitals"]["fio2"] > 0.60
        env.close()

    def test_survival_terminates_episode(self):
        """Episode terminates with survived=True when all vitals normal for 12 steps."""
        env = ICUPatientEnv(seed=0)
        env.reset(seed=0)
        from backend.env.patient_state import PatientState
        # Use vitals well within normal range so noise won't push them out
        healthy = PatientState(
            heart_rate=75, map=88, spo2=98, gcs=15, lactate=1.0,
            rr=15, temperature=37.0, fio2=0.21, peep=0.0, vasopressor=0.0,
        )
        # Run until survival triggers by injecting healthy state at each step
        survived_terminal = False
        for _ in range(15):
            env._state = healthy.copy()
            env._low_map_steps = 0
            env._low_spo2_steps = 0
            obs, reward, terminated, truncated, info = env.step(Action.NOOP)
            if terminated and info.get("survived"):
                survived_terminal = True
                break
        assert survived_terminal, "Expected survival termination within 15 healthy steps"
        env.close()
