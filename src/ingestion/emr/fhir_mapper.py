"""
G_One_Sync AI — FHIR → G_One_Sync Schema Mapper
====================================================
Converts FHIR R4 Observation, Patient resources into the existing
Pydantic schemas used by the ingestion pipeline. This ensures all
downstream validation (range checks, O₂ flow rules) is preserved.
"""

from __future__ import annotations

from typing import Optional

from loguru import logger

from src.ingestion.emr.fhir_schemas import (
    LOINC_TO_LAB,
    LOINC_TO_VITAL,
    FHIRObservation,
    FHIRPatient,
    PatientGender,
)
from src.ingestion.schemas import (
    AdmissionType,
    FlatICURow,
    Gender,
    LabResultsPayload,
    PatientDemographics,
    VitalSignsPayload,
)


class FHIRMapperError(Exception):
    """Raised when FHIR → G_One_Sync mapping fails."""


class FHIRMapper:
    """
    Maps FHIR R4 resources to G_One_Sync ingestion schemas.

    Conversion flow:
        FHIR Observation (vital-signs) → VitalSignsPayload
        FHIR Observation (laboratory)  → LabResultsPayload
        FHIR Patient                   → PatientDemographics
    """

    # ── Vital Signs Mapping ──────────────────────────────────────────

    @staticmethod
    def observations_to_vitals(
        observations: list[FHIRObservation],
    ) -> VitalSignsPayload:
        """
        Map a list of FHIR vital signs Observations to VitalSignsPayload.

        Each Observation typically carries a single vital sign identified
        by its LOINC code. We aggregate all vitals for one time point.

        Args:
            observations: FHIR Observations with category 'vital-signs'

        Returns:
            VitalSignsPayload with mapped fields

        Raises:
            FHIRMapperError: If required vitals are missing
        """
        vitals: dict[str, float] = {}

        for obs in observations:
            loinc_code = obs.get_loinc_code()
            if not loinc_code:
                logger.debug("Skipping Observation/{} — no LOINC code", obs.id)
                continue

            field_name = LOINC_TO_VITAL.get(loinc_code)
            if not field_name:
                logger.debug(
                    "Skipping unknown vital LOINC code {} (Observation/{})",
                    loinc_code, obs.id,
                )
                continue

            # Handle multi-component observations (e.g., blood pressure)
            if obs.component:
                for comp in obs.component:
                    comp_loinc = comp.code.get_loinc_code()
                    comp_field = LOINC_TO_VITAL.get(comp_loinc or "")
                    if comp_field and comp.valueQuantity and comp.valueQuantity.value is not None:
                        vitals[comp_field] = comp.valueQuantity.value
            else:
                value = obs.get_value()
                if value is not None:
                    vitals[field_name] = value

        # Check required fields
        required = {"heart_rate", "respiratory_rate", "spo2_pct",
                     "temperature_c", "systolic_bp", "diastolic_bp"}
        missing = required - set(vitals.keys())
        if missing:
            raise FHIRMapperError(
                f"Missing required vital signs from FHIR: {missing}. "
                f"Available: {set(vitals.keys())}"
            )

        logger.debug("Mapped {} vital signs from {} FHIR Observations",
                      len(vitals), len(observations))

        return VitalSignsPayload(**vitals)

    # ── Lab Results Mapping ──────────────────────────────────────────

    @staticmethod
    def observations_to_labs(
        observations: list[FHIRObservation],
    ) -> LabResultsPayload:
        """
        Map a list of FHIR laboratory Observations to LabResultsPayload.

        Args:
            observations: FHIR Observations with category 'laboratory'

        Returns:
            LabResultsPayload with mapped fields

        Raises:
            FHIRMapperError: If required labs are missing
        """
        labs: dict[str, float] = {}

        for obs in observations:
            loinc_code = obs.get_loinc_code()
            if not loinc_code:
                logger.debug("Skipping Observation/{} — no LOINC code", obs.id)
                continue

            field_name = LOINC_TO_LAB.get(loinc_code)
            if not field_name:
                logger.debug(
                    "Skipping unknown lab LOINC code {} (Observation/{})",
                    loinc_code, obs.id,
                )
                continue

            value = obs.get_value()
            if value is not None:
                labs[field_name] = value

        # Check required fields
        required = {"wbc_count", "lactate", "creatinine", "crp_level", "hemoglobin"}
        missing = required - set(labs.keys())
        if missing:
            raise FHIRMapperError(
                f"Missing required lab results from FHIR: {missing}. "
                f"Available: {set(labs.keys())}"
            )

        # Default sepsis_risk_score if not in FHIR data
        if "sepsis_risk_score" not in labs:
            labs["sepsis_risk_score"] = 0.0

        logger.debug("Mapped {} lab results from {} FHIR Observations",
                      len(labs), len(observations))

        return LabResultsPayload(**labs)

    # ── Patient Demographics Mapping ─────────────────────────────────

    @staticmethod
    def patient_to_demographics(
        patient: FHIRPatient,
        comorbidity_index: int = 0,
        admission_type: str = "ED",
    ) -> PatientDemographics:
        """
        Map a FHIR Patient resource to PatientDemographics.

        Args:
            patient: FHIR Patient resource
            comorbidity_index: Comorbidity burden index (not in FHIR Patient — must be provided)
            admission_type: Admission type (not in FHIR Patient — must be provided)

        Returns:
            PatientDemographics with mapped fields
        """
        # Map FHIR gender to G_One_Sync Gender enum
        gender_map = {
            PatientGender.MALE: Gender.MALE,
            PatientGender.FEMALE: Gender.FEMALE,
        }
        gender = gender_map.get(patient.gender, Gender.MALE)

        # Compute age
        age = patient.compute_age()
        if age is None:
            logger.warning("Cannot compute age for Patient/{} — using default 50", patient.id)
            age = 50

        # Map admission type
        admission_type_map = {
            "ED": AdmissionType.ED,
            "Elective": AdmissionType.ELECTIVE,
            "Transfer": AdmissionType.TRANSFER,
        }
        mapped_admission = admission_type_map.get(admission_type, AdmissionType.ED)

        return PatientDemographics(
            age=age,
            gender=gender,
            comorbidity_index=comorbidity_index,
            admission_type=mapped_admission,
        )

    # ── Combined Flat Row ────────────────────────────────────────────

    @staticmethod
    def to_flat_icu_row(
        patient_id: int,
        hour_from_admission: int,
        vitals: VitalSignsPayload,
        labs: LabResultsPayload,
        demographics: PatientDemographics | None = None,
    ) -> FlatICURow:
        """
        Combine mapped vitals + labs + demographics into a FlatICURow.

        This produces the exact same payload format used by the existing
        /ingest/batch and /ingest/record endpoints, ensuring full
        compatibility with the downstream pipeline.

        Args:
            patient_id: Numeric patient identifier
            hour_from_admission: Hours since ICU admission
            vitals: Mapped vital signs
            labs: Mapped lab results
            demographics: Mapped demographics (optional)

        Returns:
            FlatICURow ready for ingestion
        """
        row_data: dict = {
            "patient_id": patient_id,
            "hour_from_admission": hour_from_admission,
            # Vitals
            "heart_rate": vitals.heart_rate,
            "respiratory_rate": vitals.respiratory_rate,
            "spo2_pct": vitals.spo2_pct,
            "temperature_c": vitals.temperature_c,
            "systolic_bp": vitals.systolic_bp,
            "diastolic_bp": vitals.diastolic_bp,
            "oxygen_device": vitals.oxygen_device.value,
            "oxygen_flow": vitals.oxygen_flow,
            "mobility_score": vitals.mobility_score,
            "nurse_alert": vitals.nurse_alert,
            # Labs
            "wbc_count": labs.wbc_count,
            "lactate": labs.lactate,
            "creatinine": labs.creatinine,
            "crp_level": labs.crp_level,
            "hemoglobin": labs.hemoglobin,
            "sepsis_risk_score": labs.sepsis_risk_score,
        }

        if demographics:
            row_data["age"] = demographics.age
            row_data["gender"] = demographics.gender.value
            row_data["comorbidity_index"] = demographics.comorbidity_index
            row_data["admission_type"] = demographics.admission_type.value

        return FlatICURow(**row_data)
