"""Shared pytest fixtures for all test suites."""

from __future__ import annotations

import os
import pytest
import pytest_asyncio

# Force test environment
os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("MODELS_DIR", "/tmp/icu_test_models")
os.environ.setdefault("DATA_DIR", "/tmp/icu_test_data")
os.environ.setdefault("RUNS_DIR", "/tmp/icu_test_runs")

from backend.env.icu_env import ICUPatientEnv
from backend.training.replay_buffer import EpisodeDB


@pytest.fixture
def env():
    """Fresh ICU environment for each test."""
    e = ICUPatientEnv(seed=42)
    yield e
    e.close()


@pytest.fixture
def seeded_env():
    """Deterministic environment with fixed seed."""
    e = ICUPatientEnv(seed=0)
    yield e
    e.close()


@pytest_asyncio.fixture
async def episode_db(tmp_path):
    """Temporary in-memory SQLite DB for testing."""
    db = EpisodeDB(tmp_path / "test_episodes.db")
    await db.initialize()
    return db


@pytest.fixture
def sample_vitals_dict():
    return {
        "heart_rate": 95.0,
        "map": 72.0,
        "spo2": 94.0,
        "gcs": 13.0,
        "lactate": 2.5,
        "rr": 22.0,
        "temperature": 38.2,
        "fio2": 0.40,
        "peep": 5.0,
        "vasopressor": 0.1,
        "fluid_balance": 200.0,
    }
