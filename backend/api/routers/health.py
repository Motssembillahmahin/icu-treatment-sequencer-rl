"""GET /api/v1/health — liveness and model status."""

from __future__ import annotations

from fastapi import APIRouter

from backend.api.dependencies import get_agent

router = APIRouter()


@router.get("/health")
async def health() -> dict:
    agent = get_agent()
    return {
        "status": "ok",
        "model_loaded": agent is not None,
    }
