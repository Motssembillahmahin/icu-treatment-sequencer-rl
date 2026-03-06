"""Patient state Pydantic schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class VitalsReading(BaseModel):
    heart_rate: float = Field(..., ge=20, le=250, description="Heart rate in bpm")
    map: float = Field(..., ge=20, le=160, description="Mean arterial pressure in mmHg")
    spo2: float = Field(..., ge=50, le=100, description="Oxygen saturation %")
    gcs: float = Field(..., ge=3, le=15, description="Glasgow Coma Scale score")
    lactate: float = Field(..., ge=0.5, le=20, description="Blood lactate mmol/L")
    rr: float = Field(..., ge=4, le=60, description="Respiratory rate breaths/min")
    temperature: float = Field(..., ge=34, le=42, description="Body temperature °C")
    fio2: float = Field(0.21, ge=0.21, le=1.0, description="FiO2 fraction")
    peep: float = Field(0.0, ge=0, le=20, description="PEEP cmH2O")
    vasopressor: float = Field(0.0, ge=0, le=1.0, description="Vasopressor dose (normalized)")
    fluid_balance: float = Field(0.0, ge=-5000, le=5000, description="Fluid balance mL")


class PatientStateResponse(VitalsReading):
    time_in_icu: float = Field(0.0, ge=0, le=168, description="Hours in ICU")
    episode_step_pct: float = Field(0.0, ge=0, le=1, description="Episode progress fraction")
    archetype: str = Field("unknown", description="Patient clinical archetype")
