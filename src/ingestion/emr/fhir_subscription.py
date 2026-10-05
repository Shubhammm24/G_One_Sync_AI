"""
G_One_Sync AI — FHIR Subscription Webhook Router
====================================================
FastAPI APIRouter that handles FHIR Subscription notifications.
When the EMR creates a new Observation, it POSTs a notification to
our webhook → we fetch the full resource → map → publish to Kafka.

Mounted onto the ingest API as: /emr/fhir/*
"""

from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, HTTPException, Query, Request
from loguru import logger

from config.settings import emr_settings, kafka_settings
from src.ingestion.data_lake_router import DataLakeRouter
from src.ingestion.emr.audit_logger import AuditLogger
from src.ingestion.emr.fhir_client import FHIRClient, FHIRRequestError
from src.ingestion.emr.fhir_mapper import FHIRMapper, FHIRMapperError
from src.ingestion.emr.fhir_schemas import FHIRSubscriptionNotification
from src.ingestion.kafka_producer import get_producer
from src.ingestion.schemas import IngestionResponse

# ── Router ───────────────────────────────────────────────────────────

fhir_router = APIRouter()

# Module-level singletons (initialized on first use)
_fhir_client: FHIRClient | None = None
_audit_logger: AuditLogger | None = None


def _get_fhir_client() -> FHIRClient:
    """Get or create the singleton FHIR client."""
    global _fhir_client
    if _fhir_client is None:
        _fhir_client = FHIRClient()
    return _fhir_client


def _get_audit_logger() -> AuditLogger:
    """Get or create the singleton audit logger."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger(emr_settings.audit_log_dir)
    return _audit_logger


def _get_producer(request: Request):
    """Get Kafka producer from app state or fallback to singleton."""
    return getattr(request.app.state, "producer", None) or get_producer()


def _get_data_lake(request: Request):
    """Get DataLake router from app state or fallback to singleton."""
    return getattr(request.app.state, "data_lake", None) or DataLakeRouter()


# ── Webhook Endpoint ────────────────────────────────────────────────


@fhir_router.post("/webhook", response_model=IngestionResponse)
async def fhir_webhook(
    notification: FHIRSubscriptionNotification,
    request: Request,
) -> IngestionResponse:
    """
    Receive a FHIR Subscription notification and ingest the referenced resource.

    Flow:
        1. Parse notification → extract resource reference
        2. Fetch the full Observation from the FHIR server
        3. Map to VitalSignsPayload / LabResultsPayload
        4. Publish to existing Kafka producer
        5. Store in existing data lake
        6. Write audit log entry

    Args:
        notification: FHIR Subscription webhook payload
        request: FastAPI request (provides access to app.state)

    Returns:
        IngestionResponse confirming receipt
    """
    start = time.time()
    ingestion_id = uuid.uuid4().hex[:12]
    client = _get_fhir_client()
    audit = _get_audit_logger()

    patient_id_str = notification.get_patient_id()
    resource_id = notification.resource_id

    logger.info(
        "FHIR webhook received: {} {} for patient {}",
        notification.event_type, notification.resource_type, patient_id_str,
    )

    try:
        # Fetch the full Observation
        observation = await client.get_observation(resource_id)

        # Determine patient_id as int (hash if non-numeric)
        try:
            patient_id_int = int(patient_id_str)
        except ValueError:
            patient_id_int = abs(hash(patient_id_str)) % 1_000_000

        producer = _get_producer(request)
        data_lake = _get_data_lake(request)

        # Map based on category
        records_valid = 0
        if observation.is_vital_sign():
            try:
                vitals = FHIRMapper.observations_to_vitals([observation])
            except FHIRMapperError:
                # Webhook sent a single vital observation — fetch recent observations to complete set
                recent_vitals = await client.get_vitals(patient_id_str, count=10)
                all_vitals = [observation] + [o for o in recent_vitals if o.id != observation.id]
                vitals = FHIRMapper.observations_to_vitals(all_vitals)

            record = {
                "patient_id": patient_id_int,
                "ingestion_id": ingestion_id,
                "source": emr_settings.emr_source_tag,
                "fhir_resource_id": resource_id,
                **vitals.model_dump(),
            }
            producer.produce_vitals(patient_id_int, record)
            data_lake.store(
                data=record,
                patient_id=str(patient_id_int),
                data_type=kafka_settings.vitals_topic,
            )
            records_valid = 1

        elif observation.is_laboratory():
            try:
                labs = FHIRMapper.observations_to_labs([observation])
            except FHIRMapperError:
                # Webhook sent a single lab observation — fetch recent observations to complete set
                recent_labs = await client.get_labs(patient_id_str, count=10)
                all_labs = [observation] + [o for o in recent_labs if o.id != observation.id]
                labs = FHIRMapper.observations_to_labs(all_labs)

            record = {
                "patient_id": patient_id_int,
                "ingestion_id": ingestion_id,
                "source": emr_settings.emr_source_tag,
                "fhir_resource_id": resource_id,
                **labs.model_dump(),
            }
            producer.produce_labs(patient_id_int, record)
            data_lake.store(
                data=record,
                patient_id=str(patient_id_int),
                data_type=kafka_settings.labs_topic,
            )
            records_valid = 1
        else:
            logger.warning(
                "Observation/{} has no recognized category — skipping", resource_id,
            )

        elapsed_ms = (time.time() - start) * 1000

        # Audit log
        audit.log_event(
            event_type="fhir_webhook_ingest",
            source=emr_settings.emr_source_tag,
            patient_id=patient_id_int,
            record_count=records_valid,
            metadata={
                "fhir_resource_id": resource_id,
                "ingestion_id": ingestion_id,
                "duration_ms": round(elapsed_ms, 2),
            },
        )

        return IngestionResponse(
            status="accepted",
            records_received=1,
            records_valid=records_valid,
            ingestion_id=ingestion_id,
            message=f"FHIR Observation/{resource_id} ingested for patient {patient_id_str}",
        )

    except FHIRRequestError as e:
        logger.error("FHIR fetch failed for Observation/{}: {}", resource_id, e)
        audit.log_event(
            event_type="fhir_webhook_error",
            source=emr_settings.emr_source_tag,
            patient_id=patient_id_str,
            metadata={"error": str(e), "resource_id": resource_id},
        )
        raise HTTPException(status_code=502, detail=f"FHIR server error: {e}") from e

    except FHIRMapperError as e:
        logger.warning("FHIR mapping failed for Observation/{}: {}", resource_id, e)
        audit.log_event(
            event_type="fhir_mapping_error",
            source=emr_settings.emr_source_tag,
            patient_id=patient_id_str,
            metadata={"error": str(e), "resource_id": resource_id},
        )
        raise HTTPException(status_code=422, detail=f"FHIR mapping error: {e}") from e


# ── Poll Endpoint ────────────────────────────────────────────────────


@fhir_router.post("/poll", response_model=IngestionResponse)
async def fhir_poll(
    request: Request,
    patient_id: str = Query(..., description="FHIR Patient resource ID"),
    hours_back: int = Query(default=1, ge=1, le=72, description="Hours of data to fetch"),
) -> IngestionResponse:
    """
    Manually poll the FHIR server for recent observations for a patient.

    This is a fallback mechanism when FHIR Subscriptions are not available.
    It fetches the most recent observations and ingests them.

    Args:
        request: FastAPI request
        patient_id: FHIR Patient ID
        hours_back: How many hours of data to fetch

    Returns:
        IngestionResponse with ingestion results
    """
    start = time.time()
    ingestion_id = uuid.uuid4().hex[:12]
    client = _get_fhir_client()
    audit = _get_audit_logger()

    logger.info("FHIR poll: patient={}, hours_back={}", patient_id, hours_back)

    try:
        # Fetch vitals and labs
        vitals_obs = await client.get_vitals(patient_id, count=hours_back * 10)
        labs_obs = await client.get_labs(patient_id, count=hours_back * 5)

        # Convert patient_id to int
        try:
            patient_id_int = int(patient_id)
        except ValueError:
            patient_id_int = abs(hash(patient_id)) % 1_000_000

        records_valid = 0
        records_rejected = 0

        producer = _get_producer(request)
        data_lake = _get_data_lake(request)

        # Process vitals
        if vitals_obs:
            try:
                vitals = FHIRMapper.observations_to_vitals(vitals_obs)
                record = {
                    "patient_id": patient_id_int,
                    "ingestion_id": ingestion_id,
                    "source": emr_settings.emr_source_tag,
                    **vitals.model_dump(),
                }
                producer.produce_vitals(patient_id_int, record)
                data_lake.store(
                    data=record,
                    patient_id=str(patient_id_int),
                    data_type=kafka_settings.vitals_topic,
                )
                records_valid += 1
            except FHIRMapperError as e:
                logger.warning("Vitals mapping failed during poll: {}", e)
                records_rejected += 1

        # Process labs
        if labs_obs:
            try:
                labs = FHIRMapper.observations_to_labs(labs_obs)
                record = {
                    "patient_id": patient_id_int,
                    "ingestion_id": ingestion_id,
                    "source": emr_settings.emr_source_tag,
                    **labs.model_dump(),
                }
                producer.produce_labs(patient_id_int, record)
                data_lake.store(
                    data=record,
                    patient_id=str(patient_id_int),
                    data_type=kafka_settings.labs_topic,
                )
                records_valid += 1
            except FHIRMapperError as e:
                logger.warning("Labs mapping failed during poll: {}", e)
                records_rejected += 1

        elapsed_ms = (time.time() - start) * 1000

        audit.log_event(
            event_type="fhir_poll_ingest",
            source=emr_settings.emr_source_tag,
            patient_id=patient_id_int,
            record_count=records_valid,
            metadata={
                "ingestion_id": ingestion_id,
                "vitals_observations": len(vitals_obs),
                "labs_observations": len(labs_obs),
                "duration_ms": round(elapsed_ms, 2),
            },
        )

        total_received = (1 if vitals_obs else 0) + (1 if labs_obs else 0)

        return IngestionResponse(
            status="accepted",
            records_received=total_received,
            records_valid=records_valid,
            records_rejected=records_rejected,
            ingestion_id=ingestion_id,
            message=(
                f"FHIR poll complete for patient {patient_id}: "
                f"{len(vitals_obs)} vitals + {len(labs_obs)} labs observations fetched"
            ),
        )

    except FHIRRequestError as e:
        logger.error("FHIR poll failed for patient {}: {}", patient_id, e)
        raise HTTPException(status_code=502, detail=f"FHIR server error: {e}") from e


# ── Health Check ─────────────────────────────────────────────────────


@fhir_router.get("/health")
async def fhir_health() -> dict:
    """
    Check FHIR server connectivity and configuration status.

    Returns:
        Dict with connectivity status and configuration info
    """
    client = _get_fhir_client()
    connectivity = await client.check_connectivity()

    return {
        "status": "healthy" if connectivity else "degraded",
        "emr_enabled": emr_settings.emr_enabled,
        "fhir_base_url": emr_settings.fhir_base_url,
        "fhir_subscription_enabled": emr_settings.fhir_subscription_enabled,
        "server_connectivity": connectivity,
    }
