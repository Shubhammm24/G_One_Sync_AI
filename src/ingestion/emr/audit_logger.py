"""
G_One_Sync AI — Audit Logger
================================
Immutable, append-only audit trail for EMR data ingestion events.
Required for regulatory compliance (FDA/CE — Phase 9).

Writes JSON Lines to data/audit/YYYY-MM-DD.jsonl.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from loguru import logger


class AuditLogger:
    """
    Immutable audit trail for EMR data ingestion events.

    Each event is written as a single JSON line to a date-partitioned
    JSONL file. Files are append-only — no records are ever deleted
    or modified after writing.
    """

    def __init__(self, audit_dir: Path):
        self.audit_dir = audit_dir
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        logger.info("AuditLogger initialized at {}", self.audit_dir)

    def log_event(
        self,
        event_type: str,
        source: str,
        patient_id: int | str,
        record_count: int = 1,
        status: str = "success",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Log an audit event.

        Args:
            event_type: Type of event (e.g., 'fhir_webhook_ingest', 'hl7_ingest')
            source: Data source tag (e.g., 'emr-fhir', 'emr-hl7')
            patient_id: Patient identifier
            record_count: Number of records ingested
            status: Event status ('success', 'error', 'skipped')
            metadata: Additional context (resource IDs, error messages, etc.)
        """
        now = datetime.now(timezone.utc)

        entry = {
            "timestamp": now.isoformat() + "Z",
            "event_type": event_type,
            "source": source,
            "patient_id": str(patient_id),
            "record_count": record_count,
            "status": status,
            "metadata": metadata or {},
        }

        date_str = now.strftime("%Y-%m-%d")
        filepath = self.audit_dir / f"{date_str}.jsonl"

        with self._lock:
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")

        logger.debug(
            "Audit: {} | {} | patient={} | records={}",
            event_type, status, patient_id, record_count,
        )

    def get_events(
        self,
        date_str: str | None = None,
        event_type: str | None = None,
        patient_id: str | None = None,
    ) -> list[dict]:
        """
        Read audit events with optional filtering.

        Args:
            date_str: Date filter (YYYY-MM-DD). None = today.
            event_type: Filter by event type
            patient_id: Filter by patient ID

        Returns:
            List of audit event dicts
        """
        if date_str is None:
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        filepath = self.audit_dir / f"{date_str}.jsonl"
        if not filepath.exists():
            return []

        events = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if event_type and entry.get("event_type") != event_type:
                        continue
                    if patient_id and entry.get("patient_id") != str(patient_id):
                        continue
                    events.append(entry)
                except json.JSONDecodeError:
                    logger.warning("Malformed audit entry in {}", filepath.name)

        return events

    def get_event_count(self, date_str: str | None = None) -> int:
        """Get the total number of audit events for a date."""
        return len(self.get_events(date_str=date_str))
