"""API integration tests — health and inference endpoints."""

from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from backend.api.main import create_app


@pytest_asyncio.fixture
async def client():
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestHealthEndpoint:
    async def test_health_ok(self, client):
        response = await client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "model_loaded" in data

    async def test_health_returns_json(self, client):
        response = await client.get("/api/v1/health")
        assert response.headers["content-type"].startswith("application/json")


class TestInferenceEndpoint:
    async def test_inference_without_model_returns_503(self, client):
        from backend.api import dependencies
        dependencies.clear_agent()

        payload = {
            "vitals": {
                "heart_rate": 110,
                "map": 58,
                "spo2": 91,
                "gcs": 13,
                "lactate": 3.5,
                "rr": 24,
                "temperature": 38.5,
                "fio2": 0.40,
                "peep": 5.0,
                "vasopressor": 0.0,
                "fluid_balance": 0.0,
            },
            "time_in_icu": 4.0,
            "episode_step_pct": 0.05,
        }
        response = await client.post("/api/v1/inference", json=payload)
        assert response.status_code == 503

    async def test_inference_invalid_payload_returns_422(self, client):
        response = await client.post("/api/v1/inference", json={"bad": "data"})
        assert response.status_code == 422


class TestEpisodesEndpoint:
    async def test_list_episodes_empty(self, client):
        response = await client.get("/api/v1/episodes")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    async def test_get_nonexistent_episode_404(self, client):
        response = await client.get("/api/v1/episodes/99999")
        assert response.status_code == 404


class TestMetricsEndpoint:
    async def test_get_metrics_empty(self, client):
        response = await client.get("/api/v1/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "points" in data
        assert "total_episodes" in data
