"""
G_One_Sync AI — EMR Integration Tests
=========================================
Tests for FHIR schemas, mappers, HL7v2 parser/mapper, and audit logger.
All tests use mock data — no real EMR or FHIR server required.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.ingestion.emr.fhir_schemas import (
    FHIRBundle,
    FHIRCodeableConcept,
    FHIRCoding,
    FHIRObservation,
    FHIRPatient,
    FHIRQuantity,
    FHIRReference,
    FHIRSubscriptionNotification,
    LOINC_TO_LAB,
    LOINC_TO_VITAL,
    ObservationStatus,
    PatientGender,
)
from src.ingestion.emr.fhir_mapper import FHIRMapper, FHIRMapperError
from src.ingestion.emr.hl7_mapper import HL7Mapper
from src.ingestion.emr.hl7_parser import HL7Parser, MLLP_START_BYTE, MLLP_END_BYTES
from src.ingestion.emr.audit_logger import AuditLogger

# ── Fixtures directory ───────────────────────────────────────────────

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "fhir_bundles"


# ── Helper: Build FHIR Observations ─────────────────────────────────


def _make_vital_observation(loinc_code: str, value: float, display: str = "") -> FHIRObservation:
    """Build a minimal FHIR vital signs Observation."""
    return FHIRObservation(
        id=f"obs-{loinc_code}",
        status=ObservationStatus.FINAL,
        category=[
            FHIRCodeableConcept(
                coding=[FHIRCoding(
                    system="http://terminology.hl7.org/CodeSystem/observation-category",
                    code="vital-signs",
                )]
            )
        ],
        code=FHIRCodeableConcept(
            coding=[FHIRCoding(system="http://loinc.org", code=loinc_code, display=display)]
        ),
        subject=FHIRReference(reference="Patient/12345"),
        valueQuantity=FHIRQuantity(value=value),
    )


def _make_lab_observation(loinc_code: str, value: float, display: str = "") -> FHIRObservation:
    """Build a minimal FHIR laboratory Observation."""
    return FHIRObservation(
        id=f"obs-{loinc_code}",
        status=ObservationStatus.FINAL,
        category=[
            FHIRCodeableConcept(
                coding=[FHIRCoding(
                    system="http://terminology.hl7.org/CodeSystem/observation-category",
                    code="laboratory",
                )]
            )
        ],
        code=FHIRCodeableConcept(
            coding=[FHIRCoding(system="http://loinc.org", code=loinc_code, display=display)]
        ),
        subject=FHIRReference(reference="Patient/12345"),
        valueQuantity=FHIRQuantity(value=value),
    )


def _make_complete_vitals() -> list[FHIRObservation]:
    """Build a complete set of vital signs observations."""
    return [
        _make_vital_observation("8867-4", 88.0, "Heart rate"),
        _make_vital_observation("9279-1", 18.0, "Respiratory rate"),
        _make_vital_observation("2708-6", 96.0, "SpO2"),
        _make_vital_observation("8310-5", 37.2, "Temperature"),
        _make_vital_observation("8480-6", 125.0, "Systolic BP"),
        _make_vital_observation("8462-4", 78.0, "Diastolic BP"),
    ]


def _make_complete_labs() -> list[FHIRObservation]:
    """Build a complete set of lab observations."""
    return [
        _make_lab_observation("6690-2", 9.5, "WBC"),
        _make_lab_observation("2524-7", 1.8, "Lactate"),
        _make_lab_observation("2160-0", 1.1, "Creatinine"),
        _make_lab_observation("1988-5", 25.0, "CRP"),
        _make_lab_observation("718-7", 13.2, "Hemoglobin"),
    ]


# ── FHIR Schema Tests ───────────────────────────────────────────────


class TestFHIRSchemas:
    """Tests for FHIR R4 Pydantic models."""

    def test_fhir_observation_parse(self):
        """Parse a single FHIR Observation from JSON fixture."""
        fixture_path = FIXTURES_DIR / "observation_vitals.json"
        with open(fixture_path) as f:
            data = json.load(f)

        obs = FHIRObservation(**data)
        assert obs.id == "obs-vitals-hr-001"
        assert obs.status == ObservationStatus.FINAL
        assert obs.get_loinc_code() == "8867-4"
        assert obs.get_value() == 88.0
        assert obs.is_vital_sign()
        assert not obs.is_laboratory()

    def test_fhir_lab_observation_parse(self):
        """Parse a lab Observation from JSON fixture."""
        fixture_path = FIXTURES_DIR / "observation_labs.json"
        with open(fixture_path) as f:
            data = json.load(f)

        obs = FHIRObservation(**data)
        assert obs.get_loinc_code() == "6690-2"
        assert obs.get_value() == 9.5
        assert obs.is_laboratory()
        assert not obs.is_vital_sign()

    def test_fhir_patient_parse(self):
        """Parse a Patient resource from JSON fixture."""
        fixture_path = FIXTURES_DIR / "patient.json"
        with open(fixture_path) as f:
            data = json.load(f)

        patient = FHIRPatient(**data)
        assert patient.id == "12345"
        assert patient.gender == PatientGender.MALE
        assert patient.birthDate == "1970-05-15"

        age = patient.compute_age()
        assert age is not None
        assert age >= 55  # Born in 1970

    def test_fhir_bundle_parse(self):
        """Parse a Bundle and extract Observations."""
        fixture_path = FIXTURES_DIR / "bundle_vitals.json"
        with open(fixture_path) as f:
            data = json.load(f)

        bundle = FHIRBundle(**data)
        assert bundle.total == 6
        assert len(bundle.entry) == 6

        observations = bundle.get_observations()
        assert len(observations) == 6

    def test_loinc_mapping_completeness(self):
        """Verify all required vital/lab fields have LOINC mappings."""
        vital_fields = {"heart_rate", "respiratory_rate", "spo2_pct",
                        "temperature_c", "systolic_bp", "diastolic_bp"}
        lab_fields = {"wbc_count", "lactate", "creatinine", "crp_level", "hemoglobin"}

        mapped_vitals = set(LOINC_TO_VITAL.values())
        mapped_labs = set(LOINC_TO_LAB.values())

        assert vital_fields.issubset(mapped_vitals), (
            f"Missing vital LOINC mappings: {vital_fields - mapped_vitals}"
        )
        assert lab_fields.issubset(mapped_labs), (
            f"Missing lab LOINC mappings: {lab_fields - mapped_labs}"
        )

    def test_fhir_reference_get_id(self):
        """Test extracting ID from FHIR reference string."""
        ref = FHIRReference(reference="Patient/12345")
        assert ref.get_id() == "12345"

        ref2 = FHIRReference(reference="Observation/abc-def-123")
        assert ref2.get_id() == "abc-def-123"

    def test_observation_get_patient_id(self):
        """Test extracting patient ID from Observation subject."""
        obs = _make_vital_observation("8867-4", 72.0)
        assert obs.get_patient_id() == "12345"

    def test_subscription_notification_parse(self):
        """Test FHIR Subscription notification parsing."""
        notif = FHIRSubscriptionNotification(
            subscriptionId="sub-001",
            resourceType="Observation",
            resourceId="obs-123",
            patientReference="Patient/42",
            eventType="create",
        )
        assert notif.get_patient_id() == "42"
        assert notif.resource_id == "obs-123"


# ── FHIR Mapper Tests ───────────────────────────────────────────────


class TestFHIRMapper:
    """Tests for FHIR → G_One_Sync schema mapping."""

    def test_vitals_mapping(self):
        """Map FHIR Observations to VitalSignsPayload."""
        observations = _make_complete_vitals()
        payload = FHIRMapper.observations_to_vitals(observations)

        assert payload.heart_rate == 88.0
        assert payload.respiratory_rate == 18.0
        assert payload.spo2_pct == 96.0
        assert payload.temperature_c == 37.2
        assert payload.systolic_bp == 125.0
        assert payload.diastolic_bp == 78.0

    def test_labs_mapping(self):
        """Map FHIR Observations to LabResultsPayload."""
        observations = _make_complete_labs()
        payload = FHIRMapper.observations_to_labs(observations)

        assert payload.wbc_count == 9.5
        assert payload.lactate == 1.8
        assert payload.creatinine == 1.1
        assert payload.crp_level == 25.0
        assert payload.hemoglobin == 13.2
        assert payload.sepsis_risk_score == 0.0  # Default when not in FHIR

    def test_demographics_mapping(self):
        """Map FHIR Patient to PatientDemographics."""
        patient = FHIRPatient(id="12345", gender=PatientGender.MALE, birthDate="1970-05-15")
        demographics = FHIRMapper.patient_to_demographics(patient)

        assert demographics.gender.value == "M"
        assert demographics.age >= 55
        assert demographics.admission_type.value == "ED"

    def test_demographics_female(self):
        """Map female FHIR Patient to G_One_Sync Gender."""
        patient = FHIRPatient(id="456", gender=PatientGender.FEMALE, birthDate="1985-01-01")
        demographics = FHIRMapper.patient_to_demographics(patient)
        assert demographics.gender.value == "F"

    def test_missing_vital_raises_error(self):
        """Missing required vital signs should raise FHIRMapperError."""
        # Only provide heart rate — missing RR, SpO2, temp, BP
        observations = [_make_vital_observation("8867-4", 72.0)]
        with pytest.raises(FHIRMapperError, match="Missing required vital"):
            FHIRMapper.observations_to_vitals(observations)

    def test_missing_lab_raises_error(self):
        """Missing required lab results should raise FHIRMapperError."""
        # Only provide WBC — missing lactate, creatinine, CRP, hemoglobin
        observations = [_make_lab_observation("6690-2", 8.0)]
        with pytest.raises(FHIRMapperError, match="Missing required lab"):
            FHIRMapper.observations_to_labs(observations)

    def test_unknown_loinc_code_skipped(self):
        """Unknown LOINC codes should be silently skipped."""
        observations = _make_complete_vitals()
        # Add an unknown observation
        observations.append(_make_vital_observation("99999-9", 42.0, "Unknown Vital"))
        # Should still work — unknown is just ignored
        payload = FHIRMapper.observations_to_vitals(observations)
        assert payload.heart_rate == 88.0

    def test_out_of_range_value_rejected(self):
        """Values outside clinical ranges should be rejected by Pydantic validation."""
        observations = [
            _make_vital_observation("8867-4", 999.0),  # HR > 300 (max)
            _make_vital_observation("9279-1", 18.0),
            _make_vital_observation("2708-6", 96.0),
            _make_vital_observation("8310-5", 37.2),
            _make_vital_observation("8480-6", 125.0),
            _make_vital_observation("8462-4", 78.0),
        ]
        with pytest.raises(Exception):  # Pydantic ValidationError
            FHIRMapper.observations_to_vitals(observations)

    def test_flat_row_roundtrip(self):
        """Map FHIR → VitalSignsPayload + LabResultsPayload → FlatICURow."""
        vitals = FHIRMapper.observations_to_vitals(_make_complete_vitals())
        labs = FHIRMapper.observations_to_labs(_make_complete_labs())
        patient = FHIRPatient(id="42", gender=PatientGender.MALE, birthDate="1980-06-15")
        demographics = FHIRMapper.patient_to_demographics(patient)

        row = FHIRMapper.to_flat_icu_row(
            patient_id=42,
            hour_from_admission=5,
            vitals=vitals,
            labs=labs,
            demographics=demographics,
        )

        assert row.patient_id == 42
        assert row.hour_from_admission == 5
        assert row.heart_rate == 88.0
        assert row.wbc_count == 9.5
        assert row.gender == "M"

    def test_flat_row_without_demographics(self):
        """FlatICURow should work without demographics."""
        vitals = FHIRMapper.observations_to_vitals(_make_complete_vitals())
        labs = FHIRMapper.observations_to_labs(_make_complete_labs())

        row = FHIRMapper.to_flat_icu_row(
            patient_id=1,
            hour_from_admission=0,
            vitals=vitals,
            labs=labs,
        )
        assert row.patient_id == 1
        assert row.age is None  # No demographics provided

    def test_bundle_fixture_end_to_end(self):
        """Parse a FHIR Bundle fixture and map to VitalSignsPayload."""
        fixture_path = FIXTURES_DIR / "bundle_vitals.json"
        with open(fixture_path) as f:
            data = json.load(f)

        bundle = FHIRBundle(**data)
        observations = bundle.get_observations()
        payload = FHIRMapper.observations_to_vitals(observations)

        assert payload.heart_rate == 88.0
        assert payload.systolic_bp == 125.0
        assert payload.diastolic_bp == 78.0


# ── HL7v2 Parser Tests ──────────────────────────────────────────────


class TestHL7Parser:
    """Tests for HL7v2 message parsing."""

    def test_strip_mllp_framing(self):
        """Strip MLLP start/end bytes from raw message."""
        raw = MLLP_START_BYTE + b"MSH|test message" + MLLP_END_BYTES
        result = HL7Parser.strip_mllp_framing(raw)
        assert result == "MSH|test message"

    def test_strip_mllp_no_framing(self):
        """Handle message without MLLP framing."""
        raw = b"MSH|test message"
        result = HL7Parser.strip_mllp_framing(raw)
        assert result == "MSH|test message"

    def test_build_ack_accepted(self):
        """Build an HL7 ACK with AA code."""
        ack = HL7Parser.build_ack("MSG001", "AA", "Message accepted")
        assert "MSH|" in ack
        assert "MSA|AA|MSG001|Message accepted" in ack

    def test_build_ack_error(self):
        """Build an HL7 ACK with AE code."""
        ack = HL7Parser.build_ack("MSG002", "AE", "Parse error")
        assert "MSA|AE|MSG002|Parse error" in ack

    def test_manual_parser_oru_r01(self):
        """Test manual (fallback) parsing of an ORU^R01 message."""
        message = (
            "MSH|^~\\&|MONITOR|ICU|||20261006120000||ORU^R01|MSG001|P|2.5\r"
            "PID|||12345^^^HOSP||DOE^JOHN||19700515|M\r"
            "OBX|1|NM|8867-4^Heart Rate^LN||88|/min|||||F\r"
            "OBX|2|NM|9279-1^Respiratory Rate^LN||18|/min|||||F\r"
            "OBX|3|NM|2708-6^SpO2^LN||96|%|||||F\r"
        )
        result = HL7Parser._parse_manual(message)

        assert result["message_type"] == "ORU^R01"
        assert len(result["observations"]) == 3
        assert result["observations"][0]["observation_code"] == "8867-4"
        assert result["observations"][0]["value"] == 88.0

    def test_manual_parser_adt_a01(self):
        """Test manual parsing of an ADT^A01 message."""
        message = (
            "MSH|^~\\&|ADT|HOSP|||20261006120000||ADT^A01|MSG002|P|2.5\r"
            "PID|||12345^^^HOSP||DOE^JOHN||19700515|M\r"
        )
        result = HL7Parser._parse_manual(message)

        assert result["message_type"] == "ADT^A01"
        assert result["patient"]["patient_id"] == "12345"
        assert result["patient"]["gender"] == "M"


# ── HL7v2 Mapper Tests ──────────────────────────────────────────────


class TestHL7Mapper:
    """Tests for HL7v2 → G_One_Sync schema mapping."""

    def _make_vital_segments(self) -> list[dict]:
        """Build a complete set of vital OBX segments."""
        return [
            {"observation_code": "8867-4", "observation_identifier": "Heart Rate", "value": 88.0, "unit": "/min"},
            {"observation_code": "9279-1", "observation_identifier": "Respiratory Rate", "value": 18.0, "unit": "/min"},
            {"observation_code": "2708-6", "observation_identifier": "SpO2", "value": 96.0, "unit": "%"},
            {"observation_code": "8310-5", "observation_identifier": "Temperature", "value": 37.2, "unit": "Cel"},
            {"observation_code": "8480-6", "observation_identifier": "Systolic BP", "value": 125.0, "unit": "mmHg"},
            {"observation_code": "8462-4", "observation_identifier": "Diastolic BP", "value": 78.0, "unit": "mmHg"},
        ]

    def _make_lab_segments(self) -> list[dict]:
        """Build a complete set of lab OBX segments."""
        return [
            {"observation_code": "6690-2", "observation_identifier": "WBC", "value": 9.5, "unit": "10*3/uL"},
            {"observation_code": "2524-7", "observation_identifier": "Lactate", "value": 1.8, "unit": "mmol/L"},
            {"observation_code": "2160-0", "observation_identifier": "Creatinine", "value": 1.1, "unit": "mg/dL"},
            {"observation_code": "1988-5", "observation_identifier": "CRP", "value": 25.0, "unit": "mg/L"},
            {"observation_code": "718-7", "observation_identifier": "Hemoglobin", "value": 13.2, "unit": "g/dL"},
        ]

    def test_obx_to_vitals(self):
        """Map HL7 OBX segments to VitalSignsPayload."""
        segments = self._make_vital_segments()
        payload = HL7Mapper.obx_segments_to_vitals(segments)

        assert payload is not None
        assert payload.heart_rate == 88.0
        assert payload.respiratory_rate == 18.0
        assert payload.spo2_pct == 96.0
        assert payload.temperature_c == 37.2
        assert payload.systolic_bp == 125.0
        assert payload.diastolic_bp == 78.0

    def test_obx_to_labs(self):
        """Map HL7 OBX segments to LabResultsPayload."""
        segments = self._make_lab_segments()
        payload = HL7Mapper.obx_segments_to_labs(segments)

        assert payload is not None
        assert payload.wbc_count == 9.5
        assert payload.lactate == 1.8
        assert payload.creatinine == 1.1
        assert payload.crp_level == 25.0
        assert payload.hemoglobin == 13.2
        assert payload.sepsis_risk_score == 0.0

    def test_missing_vitals_returns_none(self):
        """Missing required vitals should return None (not raise)."""
        segments = [
            {"observation_code": "8867-4", "value": 88.0},
            # Missing RR, SpO2, temp, BP
        ]
        result = HL7Mapper.obx_segments_to_vitals(segments)
        assert result is None

    def test_missing_labs_returns_none(self):
        """Missing required labs should return None."""
        segments = [
            {"observation_code": "6690-2", "value": 8.0},
        ]
        result = HL7Mapper.obx_segments_to_labs(segments)
        assert result is None

    def test_non_loinc_identifiers(self):
        """Test mapping with non-LOINC shorthand identifiers."""
        segments = [
            {"observation_code": "HR", "value": 72.0},
            {"observation_code": "RR", "value": 16.0},
            {"observation_code": "SPO2", "value": 98.0},
            {"observation_code": "TEMP", "value": 37.0},
            {"observation_code": "SBP", "value": 120.0},
            {"observation_code": "DBP", "value": 80.0},
        ]
        payload = HL7Mapper.obx_segments_to_vitals(segments)
        assert payload is not None
        assert payload.heart_rate == 72.0

    def test_pid_to_demographics(self):
        """Map HL7 PID segment to PatientDemographics."""
        pid_data = {
            "patient_id": "12345",
            "gender": "M",
            "birth_date": "1970-05-15",
            "admission_type": "ED",
        }
        demographics = HL7Mapper.pid_segment_to_demographics(pid_data)

        assert demographics is not None
        assert demographics.gender.value == "M"
        assert demographics.age >= 55
        assert demographics.admission_type.value == "ED"

    def test_pid_female_patient(self):
        """Map female PID segment."""
        pid_data = {"gender": "F", "birth_date": "1985-01-01"}
        demographics = HL7Mapper.pid_segment_to_demographics(pid_data)
        assert demographics is not None
        assert demographics.gender.value == "F"

    def test_non_numeric_value_skipped(self):
        """Non-numeric OBX values should be skipped."""
        segments = self._make_vital_segments()
        segments.append({
            "observation_code": "8867-4",
            "value": "not a number",  # Should be skipped
        })
        # The first valid segment should win
        payload = HL7Mapper.obx_segments_to_vitals(segments)
        assert payload is not None
        assert payload.heart_rate == 88.0


# ── Audit Logger Tests ──────────────────────────────────────────────


class TestAuditLogger:
    """Tests for audit trail."""

    def test_log_event(self):
        """Log an event and verify it's written."""
        with tempfile.TemporaryDirectory() as tmpdir:
            audit = AuditLogger(Path(tmpdir))
            audit.log_event(
                event_type="fhir_webhook_ingest",
                source="emr-fhir",
                patient_id=12345,
                record_count=1,
                metadata={"fhir_resource_id": "obs-001"},
            )

            events = audit.get_events()
            assert len(events) == 1
            assert events[0]["event_type"] == "fhir_webhook_ingest"
            assert events[0]["patient_id"] == "12345"
            assert events[0]["metadata"]["fhir_resource_id"] == "obs-001"

    def test_multiple_events(self):
        """Log multiple events and verify count."""
        with tempfile.TemporaryDirectory() as tmpdir:
            audit = AuditLogger(Path(tmpdir))
            for i in range(5):
                audit.log_event(
                    event_type="hl7_ingest",
                    source="emr-hl7",
                    patient_id=i,
                )

            events = audit.get_events()
            assert len(events) == 5

    def test_filter_by_event_type(self):
        """Filter events by type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            audit = AuditLogger(Path(tmpdir))
            audit.log_event(event_type="fhir_ingest", source="fhir", patient_id=1)
            audit.log_event(event_type="hl7_ingest", source="hl7", patient_id=2)
            audit.log_event(event_type="fhir_ingest", source="fhir", patient_id=3)

            fhir_events = audit.get_events(event_type="fhir_ingest")
            assert len(fhir_events) == 2

            hl7_events = audit.get_events(event_type="hl7_ingest")
            assert len(hl7_events) == 1

    def test_filter_by_patient_id(self):
        """Filter events by patient ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            audit = AuditLogger(Path(tmpdir))
            audit.log_event(event_type="ingest", source="fhir", patient_id=42)
            audit.log_event(event_type="ingest", source="fhir", patient_id=99)
            audit.log_event(event_type="ingest", source="fhir", patient_id=42)

            events = audit.get_events(patient_id="42")
            assert len(events) == 2

    def test_append_only(self):
        """Verify events are appended, not overwritten."""
        with tempfile.TemporaryDirectory() as tmpdir:
            audit = AuditLogger(Path(tmpdir))
            audit.log_event(event_type="first", source="test", patient_id=1)

            # Create a second logger instance pointing to same dir
            audit2 = AuditLogger(Path(tmpdir))
            audit2.log_event(event_type="second", source="test", patient_id=2)

            events = audit.get_events()
            assert len(events) == 2
            assert events[0]["event_type"] == "first"
            assert events[1]["event_type"] == "second"

    def test_event_count(self):
        """Test event counting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            audit = AuditLogger(Path(tmpdir))
            for i in range(10):
                audit.log_event(event_type="test", source="test", patient_id=i)

            assert audit.get_event_count() == 10

    def test_empty_date_returns_empty(self):
        """Querying a date with no events returns empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            audit = AuditLogger(Path(tmpdir))
            events = audit.get_events(date_str="2020-01-01")
            assert events == []
