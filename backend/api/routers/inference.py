"""POST /api/v1/inference — patient state → recommended action."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException

from backend.api.dependencies import get_agent
from backend.api.schemas.action import ActionRequest, ActionResponse, build_reasoning_tags
from backend.env.actions import Action
from backend.env.patient_state import PatientState
from backend.env.spaces import normalize, VITAL_NAMES

router = APIRouter()


def _vitals_to_obs(req: ActionRequest) -> np.ndarray:
    """Convert ActionRequest to the normalized observation vector."""
    raw = [
        req.vitals.heart_rate,
        req.vitals.map,
        req.vitals.spo2,
        req.vitals.gcs,
        req.vitals.lactate,
        req.vitals.rr,
        req.vitals.temperature,
        req.vitals.fio2,
        req.vitals.peep,
        req.vitals.vasopressor,
        req.vitals.fluid_balance,
        req.time_in_icu,
        req.episode_step_pct,
    ]
    return np.array(
        [normalize(v, name) for v, name in zip(raw, VITAL_NAMES)],
        dtype=np.float32,
    )


@router.post("/inference", response_model=ActionResponse)
async def inference(req: ActionRequest) -> ActionResponse:
    agent = get_agent()
    if agent is None:
        raise HTTPException(status_code=503, detail="No trained model loaded")

    obs = _vitals_to_obs(req)
    action_arr, _ = agent.predict(obs[np.newaxis, :], deterministic=req.deterministic)
    action_id = int(action_arr.flat[0])

    # Try to get action probabilities (PPO only)
    try:
        probs = agent.get_action_probs(obs[np.newaxis, :]).tolist()
        confidence = max(probs)
    except AttributeError:
        probs = [0.0] * 11
        probs[action_id] = 1.0
        confidence = 1.0

    action = Action(action_id)
    tags = build_reasoning_tags(action_id, req.vitals)

    return ActionResponse(
        action_id=action_id,
        action_name=action.name,
        confidence=confidence,
        all_action_probs=probs,
        reasoning_tags=tags,
    )
