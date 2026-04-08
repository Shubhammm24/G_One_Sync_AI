"""
JeevanSync AI — Ingestion Layer Tests
=======================================
Tests for schemas, data lake router, Kafka producer, and API endpoints.
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.ingestion.schemas import (
    FlatICURow,
    ICUDataPayload,
    LabResultsPayload,
    VitalSignsPayload,
    OxygenDevice,
    IngestionResponse,
)
from src.ingestion.data_lake_router import DataLakeRouter
from src.ingestion.kafka_producer import LocalFileQueue


# ── Schema Tests ─────────────────────────────────────────────────────────


class TestVitalSignsPayload:
    """Tests for VitalSignsPayload schema validation."""

    def test_valid_vitals(self):
        payload = VitalSignsPayload(
            heart_rate=72.0,
            respiratory_rate=16.0,
            spo2_pct=98.0,
            temperature_c=37.0,
            systolic_bp=120.0,
            diastolic_bp=80.0,
        )
        assert payload.heart_rate == 72.0
        assert payload.oxygen_device == OxygenDevice.NONE
        assert payload.oxygen_flow == 0.0

    def test_valid_vitals_with_oxygen(self):
        payload = VitalSignsPayload(
            heart_rate=90.0,
            respiratory_rate=22.0,
            spo2_pct=92.0,
            temperature_c=38.5,
            systolic_bp=100.0,
            diastolic_bp=60.0,
            oxygen_device=OxygenDevice.NASAL,
            oxygen_flow=3.0,
        )
        assert payload.oxygen_device == OxygenDevice.NASAL
        assert payload.oxygen_flow == 3.0

    def test_heart_rate_out_of_range(self):
        with pytest.raises(Exception):
            VitalSignsPayload(
                heart_rate=500.0,  # Out of range (max=300)
                respiratory_rate=16.0,
                spo2_pct=98.0,
                temperature_c=37.0,
                systolic_bp=120.0,
                diastolic_bp=80.0,
            )

    def test_spo2_out_of_range(self):
        with pytest.raises(Exception):
            VitalSignsPayload(
                heart_rate=72.0,
                respiratory_rate=16.0,
                spo2_pct=105.0,  # > 100%
                temperature_c=37.0,
                systolic_bp=120.0,
                diastolic_bp=80.0,
            )

    def test_oxygen_flow_mismatch(self):
        """O2 flow must be 0 when device is 'none'."""
        with pytest.raises(Exception):
            VitalSignsPayload(
                heart_rate=72.0,
                respiratory_rate=16.0,
                spo2_pct=98.0,
                temperature_c=37.0,
                systolic_bp=120.0,
                diastolic_bp=80.0,
                oxygen_device=OxygenDevice.NONE,
                oxygen_flow=5.0,  # Invalid: device is 'none'
            )


class TestLabResultsPayload:
    """Tests for LabResultsPayload schema validation."""

    def test_valid_labs(self):
        payload = LabResultsPayload(
            wbc_count=7.5,
            lactate=1.2,
            creatinine=0.9,
            crp_level=15.0,
            hemoglobin=13.5,
            sepsis_risk_score=0.15,
        )
        assert payload.lactate == 1.2
        assert payload.sepsis_risk_score == 0.15

    def test_sepsis_score_out_of_range(self):
        with pytest.raises(Exception):
            LabResultsPayload(
                wbc_count=7.5,
                lactate=1.2,
                creatinine=0.9,
                crp_level=15.0,
                hemoglobin=13.5,
                sepsis_risk_score=1.5,  # > 1.0
            )


class TestFlatICURow:
    """Tests for FlatICURow schema."""

    def test_valid_flat_row(self):
        row = FlatICURow(
            patient_id=1,
            hour_from_admission=5,
            heart_rate=72.0,
            respiratory_rate=16.0,
            spo2_pct=98.0,
            temperature_c=37.0,
            systolic_bp=120.0,
            diastolic_bp=80.0,
            wbc_count=7.5,
            lactate=1.2,
            creatinine=0.9,
            crp_level=15.0,
            hemoglobin=13.5,
            sepsis_risk_score=0.15,
        )
        assert row.patient_id == 1
        assert row.deterioration_next_12h is None  # Optional

    def test_flat_row_with_labels(self):
        row = FlatICURow(
            patient_id=42,
            hour_from_admission=10,
            heart_rate=95.0,
            respiratory_rate=24.0,
            spo2_pct=91.0,
            temperature_c=38.8,
            systolic_bp=95.0,
            diastolic_bp=55.0,
            wbc_count=15.0,
            lactate=3.5,
            creatinine=1.8,
            crp_level=120.0,
            hemoglobin=10.0,
            sepsis_risk_score=0.75,
            deterioration_next_12h=1,
        )
        assert row.deterioration_next_12h == 1


# ── Data Lake Router Tests ───────────────────────────────────────────────


class TestDataLakeRouter:
    """Tests for DataLakeRouter local storage."""

    def test_store_and_read(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            router = DataLakeRouter(base_dir=Path(tmpdir))

            # Store a record
            data = {
                "patient_id": 1,
                "hour_from_admission": 5,
                "heart_rate": 72.0,
                "spo2_pct": 98.0,
            }
            path = router.store(data, patient_id="1", data_type="icu-vitals")

            assert Path(path).exists()

            # Read the stored file
            with open(path, "r") as f:
                envelope = json.load(f)

            assert envelope["patient_id"] == "1"
            assert envelope["payload"]["heart_rate"] == 72.0

    def test_store_batch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            router = DataLakeRouter(base_dir=Path(tmpdir))

            records = [
                {"patient_id": 1, "heart_rate": 70.0},
                {"patient_id": 2, "heart_rate": 80.0},
                {"patient_id": 3, "heart_rate": 90.0},
            ]
            paths = router.store_batch(records, data_type="icu-batch")

            assert len(paths) == 3
            assert all(Path(p).exists() for p in paths)

    def test_partition_structure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            router = DataLakeRouter(base_dir=Path(tmpdir))

            router.store(
                {"patient_id": 1, "hr": 70},
                patient_id="1",
                data_type="icu-vitals",
            )

            # Check partition listing
            partitions = router.list_partitions(data_type="icu-vitals")
            assert len(partitions) > 0


# ── Local File Queue Tests ───────────────────────────────────────────────


class TestLocalFileQueue:
    """Tests for the local file-based Kafka fallback queue."""

    def test_produce_and_read(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = LocalFileQueue(Path(tmpdir))

            # Produce messages
            queue.produce("test-topic", {"key": "value1"}, "msg-1")
            queue.produce("test-topic", {"key": "value2"}, "msg-2")

            # Read the JSONL file
            topic_dir = Path(tmpdir) / "test-topic"
            jsonl_files = list(topic_dir.glob("*.jsonl"))
            assert len(jsonl_files) == 1

            with open(jsonl_files[0], "r") as f:
                lines = f.readlines()

            assert len(lines) == 2
            msg1 = json.loads(lines[0])
            assert msg1["value"]["key"] == "value1"


# ── API Response Model Tests ────────────────────────────────────────────


class TestResponseModels:
    """Tests for API response schema models."""

    def test_ingestion_response(self):
        resp = IngestionResponse(
            records_received=100,
            records_valid=98,
            records_rejected=2,
            message="Batch ingested",
        )
        assert resp.status == "accepted"
        assert resp.records_received == 100
