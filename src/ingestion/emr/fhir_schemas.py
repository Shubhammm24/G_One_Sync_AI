"""
G_One_Sync AI — FHIR R4 Pydantic Schemas
============================================
Type-safe models for FHIR R4 resources consumed from EMR systems.
Includes LOINC code → field name mapping for clinical observations.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

# ── LOINC Code → G_One_Sync Field Mapping ────────────────────────────────
# These LOINC codes are the standard identifiers used in FHIR Observation
# resources for the vital signs and lab results our pipeline expects.

LOINC_TO_VITAL: dict[str, str] = {
    "8867-4": "heart_rate",          # Heart rate
    "9279-1": "respiratory_rate",    # Respiratory rate
    "2708-6": "spo2_pct",            # Oxygen saturation (SpO2)
    "59408-5": "spo2_pct",           # SpO2 by pulse oximetry (alternate)
    "8310-5": "temperature_c",       # Body temperature
    "8480-6": "systolic_bp",         # Systolic blood pressure
    "8462-4": "diastolic_bp",        # Diastolic blood pressure
}

LOINC_TO_LAB: dict[str, str] = {
    "6690-2": "wbc_count",           # White blood cell count
    "26464-8": "wbc_count",          # WBC (alternate)
    "2524-7": "lactate",             # Serum lactate
    "32693-4": "lactate",            # Lactate (alternate)
    "2160-0": "creatinine",          # Serum creatinine
    "1988-5": "crp_level",           # C-Reactive Protein
    "718-7": "hemoglobin",           # Hemoglobin
}

# Reverse map for quick lookup
LOINC_ALL: dict[str, str] = {**LOINC_TO_VITAL, **LOINC_TO_LAB}


# ── FHIR R4 Resource Models ─────────────────────────────────────────────


class FHIRCoding(BaseModel):
    """FHIR Coding element (code + system + display)."""

    system: str = ""
    code: str = ""
    display: str = ""


class FHIRCodeableConcept(BaseModel):
    """FHIR CodeableConcept — wraps one or more Coding elements."""

    coding: list[FHIRCoding] = Field(default_factory=list)
    text: str = ""

    def get_loinc_code(self) -> str | None:
        """Extract the LOINC code from coding entries."""
        for c in self.coding:
            if "loinc" in c.system.lower():
                return c.code
        # Fallback: return first code if any
        return self.coding[0].code if self.coding else None


class FHIRQuantity(BaseModel):
    """FHIR Quantity — value with unit."""

    value: float | None = None
    unit: str = ""
    system: str = ""
    code: str = ""


class FHIRReference(BaseModel):
    """FHIR Reference to another resource."""

    reference: str = ""     # e.g., "Patient/12345"
    display: str = ""

    def get_id(self) -> str:
        """Extract the resource ID from the reference string."""
        if "/" in self.reference:
            return self.reference.split("/")[-1]
        return self.reference


class FHIRPeriod(BaseModel):
    """FHIR Period — start and end timestamps."""

    start: datetime | None = None
    end: datetime | None = None


class FHIRObservationComponent(BaseModel):
    """Component within a multi-component Observation (e.g., BP systolic/diastolic)."""

    code: FHIRCodeableConcept = Field(default_factory=FHIRCodeableConcept)
    valueQuantity: FHIRQuantity | None = None


class ObservationStatus(str, Enum):
    """FHIR Observation status codes."""

    REGISTERED = "registered"
    PRELIMINARY = "preliminary"
    FINAL = "final"
    AMENDED = "amended"
    CORRECTED = "corrected"
    CANCELLED = "cancelled"
    ENTERED_IN_ERROR = "entered-in-error"
    UNKNOWN = "unknown"


class FHIRObservation(BaseModel):
    """
    FHIR R4 Observation resource.

    Used for vital signs and laboratory results.
    See: https://hl7.org/fhir/R4/observation.html
    """

    resourceType: str = "Observation"
    id: str = ""
    status: ObservationStatus = ObservationStatus.FINAL
    category: list[FHIRCodeableConcept] = Field(default_factory=list)
    code: FHIRCodeableConcept = Field(default_factory=FHIRCodeableConcept)
    subject: FHIRReference | None = None
    effectiveDateTime: datetime | None = None
    issued: datetime | None = None
    valueQuantity: FHIRQuantity | None = None
    component: list[FHIRObservationComponent] = Field(default_factory=list)

    def get_loinc_code(self) -> str | None:
        """Get the primary LOINC code for this observation."""
        return self.code.get_loinc_code()

    def get_value(self) -> float | None:
        """Get the numeric value of this observation."""
        if self.valueQuantity and self.valueQuantity.value is not None:
            return self.valueQuantity.value
        return None

    def get_patient_id(self) -> str:
        """Extract patient ID from subject reference."""
        if self.subject:
            return self.subject.get_id()
        return ""

    def is_vital_sign(self) -> bool:
        """Check if this observation is a vital sign category."""
        for cat in self.category:
            for coding in cat.coding:
                if coding.code == "vital-signs":
                    return True
        return False

    def is_laboratory(self) -> bool:
        """Check if this observation is a laboratory category."""
        for cat in self.category:
            for coding in cat.coding:
                if coding.code == "laboratory":
                    return True
        return False


class PatientGender(str, Enum):
    """FHIR administrative gender."""

    MALE = "male"
    FEMALE = "female"
    OTHER = "other"
    UNKNOWN = "unknown"


class FHIRPatient(BaseModel):
    """
    FHIR R4 Patient resource.

    See: https://hl7.org/fhir/R4/patient.html
    """

    resourceType: str = "Patient"
    id: str = ""
    gender: PatientGender = PatientGender.UNKNOWN
    birthDate: str | None = None  # YYYY-MM-DD

    def compute_age(self, reference_date: datetime | None = None) -> int | None:
        """Compute age in years from birthDate."""
        if not self.birthDate:
            return None
        ref = reference_date or datetime.now(timezone.utc)
        try:
            birth = datetime.strptime(self.birthDate, "%Y-%m-%d")
            age = ref.year - birth.year
            if (ref.month, ref.day) < (birth.month, birth.day):
                age -= 1
            return max(0, age)
        except ValueError:
            return None


class FHIRBundleEntry(BaseModel):
    """Single entry in a FHIR Bundle."""

    fullUrl: str = ""
    resource: dict[str, Any] = Field(default_factory=dict)


class FHIRBundle(BaseModel):
    """
    FHIR R4 Bundle resource.

    Used for search results and transaction responses.
    See: https://hl7.org/fhir/R4/bundle.html
    """

    resourceType: str = "Bundle"
    type: str = "searchset"
    total: int = 0
    entry: list[FHIRBundleEntry] = Field(default_factory=list)

    def get_observations(self) -> list[FHIRObservation]:
        """Extract all Observation resources from the bundle."""
        observations = []
        for e in self.entry:
            if e.resource.get("resourceType") == "Observation":
                observations.append(FHIRObservation(**e.resource))
        return observations

    def get_patients(self) -> list[FHIRPatient]:
        """Extract all Patient resources from the bundle."""
        patients = []
        for e in self.entry:
            if e.resource.get("resourceType") == "Patient":
                patients.append(FHIRPatient(**e.resource))
        return patients


# ── FHIR Subscription Models ────────────────────────────────────────────


class FHIRSubscriptionNotification(BaseModel):
    """
    Payload received when a FHIR Subscription fires.

    The EMR sends this to our webhook endpoint when a new
    Observation is created matching our subscription criteria.
    """

    subscription_id: str = Field(default="", alias="subscriptionId")
    resource_type: str = Field(default="Observation", alias="resourceType")
    resource_id: str = Field(default="", alias="resourceId")
    patient_reference: str = Field(default="", alias="patientReference")
    event_type: str = Field(default="create", alias="eventType")
    timestamp: datetime | None = None

    model_config = {"populate_by_name": True}

    def get_patient_id(self) -> str:
        """Extract patient ID from patient reference."""
        if "/" in self.patient_reference:
            return self.patient_reference.split("/")[-1]
        return self.patient_reference
