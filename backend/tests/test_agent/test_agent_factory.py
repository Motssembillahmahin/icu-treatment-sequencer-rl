"""Agent factory and PPO wrapper tests."""

from __future__ import annotations

import pytest

from backend.agent.agent_factory import available_agents, create_agent
from backend.env.icu_env import ICUPatientEnv


@pytest.fixture
def single_env():
    env = ICUPatientEnv(seed=0)
    yield env
    env.close()


class TestAgentFactory:
    def test_available_agents(self):
        agents = available_agents()
        assert "ppo" in agents
        assert "dqn" in agents

    def test_unknown_agent_raises(self, single_env):
        with pytest.raises(ValueError, match="Unknown agent"):
            create_agent("invalid_agent", single_env)

    def test_create_ppo_agent(self, single_env):
        agent = create_agent("ppo", single_env, {"verbose": 0})
        assert agent is not None

    def test_ppo_predict_returns_valid_action(self, single_env):
        agent = create_agent("ppo", single_env, {"verbose": 0})
        obs, _ = single_env.reset()
        import numpy as np
        action, state = agent.predict(obs[np.newaxis, :], deterministic=True)
        assert 0 <= int(action.flat[0]) < 11

    def test_ppo_save_load(self, single_env, tmp_path):
        agent = create_agent("ppo", single_env, {"verbose": 0})
        path = tmp_path / "test_model"
        agent.save(path)
        assert (path.with_suffix(".zip")).exists() or path.exists()

        from backend.agent.agent_factory import load_agent
        loaded = load_agent("ppo", str(path) + ".zip", env=single_env)
        assert loaded is not None

        obs, _ = single_env.reset()
        import numpy as np
        action, _ = loaded.predict(obs[np.newaxis, :], deterministic=True)
        assert 0 <= int(action.flat[0]) < 11
