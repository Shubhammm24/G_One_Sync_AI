"""
JeevanSync AI — Data Lake Router
=================================
Routes raw ICU data payloads to immutable storage.
- Local filesystem in development (S3-compatible directory structure)
- AWS S3 in production (configurable via environment variables)
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from config.settings import data_settings


class DataLakeRouter:
    """
    Stores raw payloads in an immutable data lake with partitioned structure:

        datalake/
        ├── raw/
        │   ├── {data_type}/
        │   │   ├── date={YYYY-MM-DD}/
        │   │   │   ├── patient_id={id}/
        │   │   │   │   ├── {uuid}.json
    """

    def __init__(self, base_dir: Optional[Path] = None, use_s3: bool = False):
        self.base_dir = base_dir or data_settings.data_lake_dir
        self.use_s3 = use_s3
        self._s3_client = None

        if self.use_s3:
            self._init_s3()
        else:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            logger.info("DataLakeRouter initialized (local) at {}", self.base_dir)

    def _init_s3(self) -> None:
        """Initialize AWS S3 client for production."""
        try:
            import boto3
            self._s3_client = boto3.client("s3")
            logger.info("DataLakeRouter initialized (S3)")
        except Exception as e:
            logger.warning("S3 initialization failed ({}). Falling back to local.", e)
            self.use_s3 = False
            self.base_dir.mkdir(parents=True, exist_ok=True)

    def store(
        self,
        data: dict[str, Any],
        patient_id: str,
        data_type: str = "batch",
        timestamp: Optional[datetime] = None,
    ) -> str:
        """
        Store a raw data payload in the data lake.

        Args:
            data: Raw payload dictionary
            patient_id: Patient identifier
            data_type: Category (e.g., 'icu-vitals', 'icu-labs', 'icu-batch')
            timestamp: Ingestion timestamp (defaults to now)

        Returns:
            str: The storage path/key where data was written
        """
        ts = timestamp or datetime.utcnow()
        date_str = ts.strftime("%Y-%m-%d")
        record_id = uuid.uuid4().hex[:12]

        # Enrich payload with metadata
        envelope = {
            "ingestion_id": record_id,
            "ingestion_timestamp": ts.isoformat(),
            "source": "jeevansync-ingest-api",
            "schema_version": "1.0.0",
            "data_type": data_type,
            "patient_id": patient_id,
            "payload": data,
        }

        if self.use_s3:
            return self._store_s3(envelope, data_type, date_str, patient_id, record_id)
        else:
            return self._store_local(envelope, data_type, date_str, patient_id, record_id)

    def _store_local(
        self,
        envelope: dict,
        data_type: str,
        date_str: str,
        patient_id: str,
        record_id: str,
    ) -> str:
        """Write to local filesystem in S3-compatible partitioned layout."""
        partition_dir = (
            self.base_dir
            / "raw"
            / data_type
            / f"date={date_str}"
            / f"patient_id={patient_id}"
        )
        partition_dir.mkdir(parents=True, exist_ok=True)

        filepath = partition_dir / f"{record_id}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(envelope, f, default=str, indent=2)

        logger.debug("Stored to data lake: {}", filepath)
        return str(filepath)

    def _store_s3(
        self,
        envelope: dict,
        data_type: str,
        date_str: str,
        patient_id: str,
        record_id: str,
    ) -> str:
        """Write to AWS S3 bucket."""
        import os

        bucket = os.environ.get("JEEVAN_S3_BUCKET", "jeevansync-datalake")
        key = f"raw/{data_type}/date={date_str}/patient_id={patient_id}/{record_id}.json"

        self._s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(envelope, default=str).encode("utf-8"),
            ContentType="application/json",
        )

        logger.debug("Stored to S3: s3://{}/{}", bucket, key)
        return f"s3://{bucket}/{key}"

    def store_batch(
        self,
        records: list[dict],
        data_type: str = "icu-batch",
    ) -> list[str]:
        """
        Store a batch of records. Each record is stored individually.

        Args:
            records: List of data payloads
            data_type: Topic/category

        Returns:
            List of storage paths
        """
        paths = []
        for record in records:
            patient_id = str(record.get("patient_id", "unknown"))
            path = self.store(record, patient_id, data_type)
            paths.append(path)

        logger.info(
            "Batch stored: {} records to data lake (type={})", len(paths), data_type
        )
        return paths

    def list_partitions(
        self, data_type: str = "icu-batch", date_str: Optional[str] = None
    ) -> list[str]:
        """List available date partitions for a given data type."""
        type_dir = self.base_dir / "raw" / data_type
        if not type_dir.exists():
            return []

        partitions = []
        for d in sorted(type_dir.iterdir()):
            if d.is_dir() and d.name.startswith("date="):
                date_val = d.name.replace("date=", "")
                if date_str is None or date_val == date_str:
                    partitions.append(date_val)

        return partitions

    def read_partition(
        self, data_type: str, date_str: str, patient_id: Optional[str] = None
    ) -> list[dict]:
        """
        Read all records from a specific partition.

        Args:
            data_type: Data category
            date_str: Date partition (YYYY-MM-DD)
            patient_id: Optional - filter to specific patient

        Returns:
            List of deserialized record envelopes
        """
        partition_dir = self.base_dir / "raw" / data_type / f"date={date_str}"
        if not partition_dir.exists():
            return []

        records = []
        search_dirs = [partition_dir]
        if patient_id:
            specific = partition_dir / f"patient_id={patient_id}"
            search_dirs = [specific] if specific.exists() else []

        for search_dir in search_dirs:
            for json_file in search_dir.rglob("*.json"):
                with open(json_file, "r", encoding="utf-8") as f:
                    records.append(json.load(f))

        return records
