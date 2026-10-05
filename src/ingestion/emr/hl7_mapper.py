"""
G_One_Sync AI — HL7v2 → G_One_Sync Schema Mapper
====================================================
Converts parsed HL7v2 OBX segments into the existing Pydantic schemas.
Same pattern as FHIRMapper but for HL7v2 data.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from loguru import logger

from src.ingestion.schemas import (
    AdmissionType,
    FlatICURow,
    Gender,
    LabResultsPayload,
    PatientDemographics,
    VitalSignsPayload,
)


class HL7MapperError(Exception):
    """Raised when HL7v2 → G_One_Sync mapping fails."""


class HL7Mapper:
    """
    Maps HL7v2 OBX segments to G_One_Sync ingestion schemas.

    Uses the same LOINC codes as the FHIR mapper, since HL7v2 OBX-3
    often carries LOINC-coded observation identifiers.
    """

    # ── OBX Observation Code → Field Name ────────────────────────────
    # LOINC codes used in OBX-3 segments
    OBX_TO_VITAL: dict[str, str] = {
        "8867-4": "heart_rate",
        "9279-1": "respiratory_rate",
        "2708-6": "spo2_pct",
        "59408-5": "spo2_pct",
        "8310-5": "temperature_c",
        "8480-6": "systolic_bp",
        "8462-4": "diastolic_bp",
        # Common non-LOINC identifiers used by some EMR systems
        "HR": "heart_rate",
        "RR": "respiratory_rate",
        "SPO2": "spo2_pct",
        "TEMP": "temperature_c",
        "SBP": "systolic_bp",
        "DBP": "diastolic_bp",
    }

    OBX_TO_LAB: dict[str, str] = {
        "6690-2": "wbc_count",
        "26464-8": "wbc_count",
        "2524-7": "lactate",
        "32693-4": "lactate",
        "2160-0": "creatinine",
        "1988-5": "crp_level",
        "718-7": "hemoglobin",
        # Common non-LOINC identifiers
        "WBC": "wbc_count",
        "LAC": "lactate",
        "CREAT": "creatinine",
        "CRP": "crp_level",
        "HGB": "hemoglobin",
    }

    # ── Vital Signs Mapping ──────────────────────────────────────────

    @staticmethod
    def obx_segments_to_vitals(
        segments: list[dict[str, Any]],
    ) -> VitalSignsPayload | None:
        """
        Map OBX segments to VitalSignsPayload.

        Args:
            segments: List of parsed OBX segment dicts from HL7Parser

        Returns:
            VitalSignsPayload if required fields are present, None otherwise
        """
        vitals: dict[str, float] = {}

        for seg in segments:
            code = seg.get("observation_code", "")
            identifier = seg.get("observation_identifier", "")
            value = seg.get("value")

            if not isinstance(value, (int, float)):
                continue

            # Try code first, then identifier
            field = HL7Mapper.OBX_TO_VITAL.get(code)
            if not field:
                field = HL7Mapper.OBX_TO_VITAL.get(identifier)
            if not field:
                logger.debug("Skipping unrecognized vital OBX code: {} / {}", code, identifier)
                continue

            vitals[field] = float(value)

        required = {"heart_rate", "respiratory_rate", "spo2_pct",
                     "temperature_c", "systolic_bp", "diastolic_bp"}
        missing = required - set(vitals.keys())

        if missing:
            logger.warning("Missing required vitals from HL7: {}", missing)
            return None

        try:
            return VitalSignsPayload(**vitals)
        except Exception as e:
            logger.warning("HL7 vitals validation failed: {}", e)
            return None

    # ── Lab Results Mapping ──────────────────────────────────────────

    @staticmethod
    def obx_segments_to_labs(
        segments: list[dict[str, Any]],
    ) -> LabResultsPayload | None:
        """
        Map OBX segments to LabResultsPayload.

        Args:
            segments: List of parsed OBX segment dicts from HL7Parser

        Returns:
            LabResultsPayload if required fields are present, None otherwise
        """
        labs: dict[str, float] = {}

        for seg in segments:
            code = seg.get("observation_code", "")
            identifier = seg.get("observation_identifier", "")
            value = seg.get("value")

            if not isinstance(value, (int, float)):
                continue

            field = HL7Mapper.OBX_TO_LAB.get(code)
            if not field:
                field = HL7Mapper.OBX_TO_LAB.get(identifier)
            if not field:
                continue

            labs[field] = float(value)

        required = {"wbc_count", "lactate", "creatinine", "crp_level", "hemoglobin"}
        missing = required - set(labs.keys())

        if missing:
            logger.warning("Missing required labs from HL7: {}", missing)
            return None

        # Default sepsis_risk_score if not in HL7 data
        if "sepsis_risk_score" not in labs:
            labs["sepsis_risk_score"] = 0.0

        try:
            return LabResultsPayload(**labs)
        except Exception as e:
            logger.warning("HL7 labs validation failed: {}", e)
            return None

    # ── Patient Demographics Mapping ─────────────────────────────────

    @staticmethod
    def pid_segment_to_demographics(
        pid_data: dict[str, Any],
        comorbidity_index: int = 0,
    ) -> PatientDemographics | None:
        """
        Map PID segment data to PatientDemographics.

        Args:
            pid_data: Parsed PID segment dict from HL7Parser
            comorbidity_index: Comorbidity burden (not in HL7 PID)

        Returns:
            PatientDemographics if parseable, None otherwise
        """
        # Gender mapping
        gender_str = pid_data.get("gender", "").upper()
        gender = Gender.MALE if gender_str == "M" else Gender.FEMALE

        # Age from birth date
        birth_date = pid_data.get("birth_date", "")
        age: int | None = None
        if birth_date:
            try:
                # Handle both YYYY-MM-DD and YYYYMMDD formats
                if "-" in birth_date:
                    dob = datetime.strptime(birth_date, "%Y-%m-%d")
                elif len(birth_date) >= 8:
                    dob = datetime.strptime(birth_date[:8], "%Y%m%d")
                else:
                    dob = None

                if dob:
                    now = datetime.utcnow()
                    age = now.year - dob.year
                    if (now.month, now.day) < (dob.month, dob.day):
                        age -= 1
            except ValueError:
                logger.warning("Cannot parse HL7 birth date: {}", birth_date)

        if age is None:
            logger.warning("Cannot compute age from HL7 PID — using default 50")
            age = 50

        # Admission type
        adm_str = pid_data.get("admission_type", "ED")
        admission_map = {
            "ED": AdmissionType.ED,
            "Elective": AdmissionType.ELECTIVE,
            "Transfer": AdmissionType.TRANSFER,
        }
        admission_type = admission_map.get(adm_str, AdmissionType.ED)

        try:
            return PatientDemographics(
                age=age,
                gender=gender,
                comorbidity_index=comorbidity_index,
                admission_type=admission_type,
            )
        except Exception as e:
            logger.warning("HL7 demographics validation failed: {}", e)
            return None
