"""
JeevanSync AI — Pydantic Schemas for ICU Data Ingestion
========================================================
Strict data validation for all incoming ICU data payloads.
Ensures type safety, range validation, and consistent serialization.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ── Enums ────────────────────────────────────────────────────────────────

class Gender(str, Enum):
    MALE = "M"
    FEMALE = "F"


class AdmissionType(str, Enum):
    ED = "ED"
    ELECTIVE = "Elective"
    TRANSFER = "Transfer"


class OxygenDevice(str, Enum):
    NONE = "none"
    NASAL = "nasal"
    MASK = "mask"
    HFNC = "hfnc"
    NIV = "niv"


# ── Vital Signs Payload ─────────────────────────────────────────────────

class VitalSignsPayload(BaseModel):
    """Single hourly vital signs reading from bedside monitors."""

    heart_rate: float = Field(..., ge=20, le=300, description="Heart rate (BPM)")
    respiratory_rate: float = Field(..., ge=4, le=60, description="Respiratory rate (breaths/min)")
    spo2_pct: float = Field(..., ge=50, le=100, description="SpO2 (%)")
    temperature_c: float = Field(..., ge=30.0, le=45.0, description="Temperature (°C)")
    systolic_bp: float = Field(..., ge=40, le=300, description="Systolic BP (mmHg)")
    diastolic_bp: float = Field(..., ge=20, le=200, description="Diastolic BP (mmHg)")
    oxygen_device: OxygenDevice = Field(default=OxygenDevice.NONE)
    oxygen_flow: float = Field(default=0.0, ge=0, le=100, description="O2 flow (L/min)")
    mobility_score: int = Field(default=2, ge=0, le=4, description="Mobility score (0-4)")
    nurse_alert: int = Field(default=0, ge=0, le=1, description="Nurse alert flag")

    @field_validator("oxygen_flow")
    @classmethod
    def validate_oxygen_flow(cls, v: float, info) -> float:
        """O2 flow must be 0 when no device is used."""
        device = info.data.get("oxygen_device")
        if device == OxygenDevice.NONE and v > 0:
            raise ValueError("oxygen_flow must be 0.0 when oxygen_device is 'none'")
        if device != OxygenDevice.NONE and v <= 0:
            raise ValueError("oxygen_flow must be > 0 when an oxygen device is in use")
        return v


# ── Lab Results Payload ──────────────────────────────────────────────────

class LabResultsPayload(BaseModel):
    """Single hourly lab panel results."""

    wbc_count: float = Field(..., ge=0, le=100, description="WBC count")
    lactate: float = Field(..., ge=0, le=30, description="Serum lactate (mmol/L)")
    creatinine: float = Field(..., ge=0, le=20, description="Serum creatinine (mg/dL)")
    crp_level: float = Field(..., ge=0, le=500, description="CRP (mg/L)")
    hemoglobin: float = Field(..., ge=2, le=25, description="Hemoglobin (g/dL)")
    sepsis_risk_score: float = Field(..., ge=0, le=1, description="Sepsis risk score (0-1)")


# ── Patient Demographics ────────────────────────────────────────────────

class PatientDemographics(BaseModel):
    """Static patient demographics (sent once at admission)."""

    age: int = Field(..., ge=0, le=120, description="Age at admission")
    gender: Gender
    comorbidity_index: int = Field(..., ge=0, le=20, description="Comorbidity burden index")
    admission_type: AdmissionType


# ── Combined ICU Data Payload ────────────────────────────────────────────

class ICUDataPayload(BaseModel):
    """
    Full ICU data payload combining vitals, labs, and demographics.
    This is the primary payload format for the batch ingestion endpoint.
    """

    patient_id: int = Field(..., ge=1, description="Unique patient identifier")
    hour_from_admission: int = Field(..., ge=0, le=72, description="Hours since admission")
    timestamp: Optional[datetime] = Field(
        default=None,
        description="Wall-clock timestamp (auto-generated if not provided)",
    )

    # Embedded measurements
    vitals: VitalSignsPayload
    labs: LabResultsPayload
    demographics: Optional[PatientDemographics] = None  # Only needed on first reading

    @field_validator("timestamp", mode="before")
    @classmethod
    def set_timestamp(cls, v):
        return v or datetime.utcnow()


# ── Batch Payload ────────────────────────────────────────────────────────

class BatchIngestionPayload(BaseModel):
    """Batch ingestion of multiple hourly readings."""

    records: list[ICUDataPayload] = Field(..., min_length=1, max_length=10_000)


# ── Flat Row Payload (direct CSV-style) ──────────────────────────────────

class FlatICURow(BaseModel):
    """
    Flat representation matching the CSV schema directly.
    Used for bulk uploads and the stream simulator.
    """

    patient_id: int = Field(..., ge=1)
    hour_from_admission: int = Field(..., ge=0)

    # Vitals
    heart_rate: float
    respiratory_rate: float
    spo2_pct: float
    temperature_c: float
    systolic_bp: float
    diastolic_bp: float
    oxygen_device: str = "none"
    oxygen_flow: float = 0.0
    mobility_score: int = 2
    nurse_alert: int = 0

    # Labs
    wbc_count: float
    lactate: float
    creatinine: float
    crp_level: float
    hemoglobin: float
    sepsis_risk_score: float

    # Demographics
    age: Optional[int] = None
    gender: Optional[str] = None
    comorbidity_index: Optional[int] = None
    admission_type: Optional[str] = None

    # Labels (for training data ingestion)
    deterioration_next_12h: Optional[int] = None


class FlatBatchPayload(BaseModel):
    """Batch of flat ICU rows for bulk ingestion."""

    records: list[FlatICURow] = Field(..., min_length=1, max_length=50_000)


# ── API Response Models ──────────────────────────────────────────────────

class IngestionResponse(BaseModel):
    """Standard API response for ingestion endpoints."""

    status: str = "accepted"
    records_received: int
    records_valid: int
    records_rejected: int = 0
    message: str = ""
    ingestion_id: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    service: str = "jeevansync-ingest-api"
    version: str = "1.0.0"
    kafka_connected: bool = False
    data_lake_accessible: bool = True
