"""Action request/response Pydantic schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field

from backend.api.schemas.patient import VitalsReading


class ActionRequest(BaseModel):
    vitals: VitalsReading
    time_in_icu: float = Field(0.0, ge=0, le=168)
    episode_step_pct: float = Field(0.0, ge=0, le=1)
    deterministic: bool = Field(True, description="Use deterministic policy")


class ActionResponse(BaseModel):
    action_id: int = Field(..., ge=0, le=10)
    action_name: str
    confidence: float = Field(..., ge=0, le=1, description="Max action probability")
    all_action_probs: list[float] = Field(..., description="Probability for each of 11 actions")
    reasoning_tags: list[str] = Field(default_factory=list, description="Clinical reasoning hints")


def build_reasoning_tags(action_id: int, vitals: VitalsReading) -> list[str]:
    """Generate human-readable clinical reasoning hints."""
    tags: list[str] = []
    from backend.env.actions import Action
    action = Action(action_id)

    if action.is_fluid and vitals.map < 65:
        tags.append("MAP below target — fluid resuscitation indicated")
    if action.is_vasopressor and vitals.map < 65:
        tags.append("Persistent hypotension — vasopressor support")
    if action == Action.FIO2_UP and vitals.spo2 < 92:
        tags.append("Low SpO2 — increasing FiO2")
    if action == Action.PEEP_UP and vitals.spo2 < 90:
        tags.append("Refractory hypoxia — increasing PEEP")
    if action == Action.NOOP and vitals.map >= 70 and vitals.spo2 >= 95:
        tags.append("Vitals within target — no intervention required")
    if action == Action.EXTUBATE and vitals.spo2 >= 95 and vitals.rr <= 20:
        tags.append("Weaning criteria met — attempting extubation")

    return tags
