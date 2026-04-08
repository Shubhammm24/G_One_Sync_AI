"""
JeevanSync AI — FastAPI Ingest API
====================================
High-performance REST API for receiving ICU data from bedside monitors,
lab information systems, and EMR feeds.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from config.settings import api_settings, kafka_settings
from src.ingestion.data_lake_router import DataLakeRouter
from src.ingestion.kafka_producer import get_producer
from src.ingestion.schemas import (
    BatchIngestionPayload,
    FlatBatchPayload,
    FlatICURow,
    HealthResponse,
    IngestionResponse,
    LabResultsPayload,
    VitalSignsPayload,
)


# ── Lifespan (startup / shutdown) ────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: initialize Kafka producer + data lake. Shutdown: flush."""
    logger.info("Starting JeevanSync Ingest API v{}", api_settings.version)

    # Initialize components
    app.state.producer = get_producer()
    app.state.data_lake = DataLakeRouter()

    logger.info(
        "Kafka status: {}",
        "connected" if app.state.producer.is_connected else "fallback (local queue)",
    )

    yield

    # Shutdown
    logger.info("Shutting down Ingest API...")
    app.state.producer.close()


# ── App Factory ──────────────────────────────────────────────────────────

app = FastAPI(
    title=api_settings.title,
    version=api_settings.version,
    description=(
        "ICU Data Ingestion API for JeevanSync AI.\n\n"
        "Receives high-frequency vital signs, lab results, and EMR data "
        "from ICU bedside monitors and clinical systems. Data is validated, "
        "published to a Kafka/Redpanda stream, and routed to the data lake."
    ),
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=api_settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health Check ─────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """System health and connectivity check."""
    return HealthResponse(
        status="healthy",
        kafka_connected=app.state.producer.is_connected,
        data_lake_accessible=True,
    )


# ── Vital Signs Ingestion ───────────────────────────────────────────────

@app.post(
    "/ingest/vitals",
    response_model=IngestionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Ingestion"],
)
async def ingest_vitals(
    patient_id: int = Query(..., ge=1, description="Patient identifier"),
    hour_from_admission: int = Query(..., ge=0, description="Hour from admission"),
    payload: VitalSignsPayload = ...,
):
    """
    Ingest a single vital signs reading for a patient.
    Publishes to the `icu-vitals` Kafka topic and stores in data lake.
    """
    ingestion_id = uuid.uuid4().hex[:12]

    record = {
        "patient_id": patient_id,
        "hour_from_admission": hour_from_admission,
        "ingestion_id": ingestion_id,
        **payload.model_dump(),
    }

    # Publish to stream
    app.state.producer.produce_vitals(patient_id, record)

    # Store in data lake
    app.state.data_lake.store(
        data=record,
        patient_id=str(patient_id),
        data_type=kafka_settings.vitals_topic,
    )

    return IngestionResponse(
        status="accepted",
        records_received=1,
        records_valid=1,
        ingestion_id=ingestion_id,
        message=f"Vital signs for patient {patient_id} at hour {hour_from_admission} accepted",
    )


# ── Lab Results Ingestion ────────────────────────────────────────────────

@app.post(
    "/ingest/labs",
    response_model=IngestionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Ingestion"],
)
async def ingest_labs(
    patient_id: int = Query(..., ge=1, description="Patient identifier"),
    hour_from_admission: int = Query(..., ge=0, description="Hour from admission"),
    payload: LabResultsPayload = ...,
):
    """
    Ingest a single lab results panel for a patient.
    Publishes to the `icu-labs` Kafka topic and stores in data lake.
    """
    ingestion_id = uuid.uuid4().hex[:12]

    record = {
        "patient_id": patient_id,
        "hour_from_admission": hour_from_admission,
        "ingestion_id": ingestion_id,
        **payload.model_dump(),
    }

    app.state.producer.produce_labs(patient_id, record)

    app.state.data_lake.store(
        data=record,
        patient_id=str(patient_id),
        data_type=kafka_settings.labs_topic,
    )

    return IngestionResponse(
        status="accepted",
        records_received=1,
        records_valid=1,
        ingestion_id=ingestion_id,
        message=f"Lab results for patient {patient_id} at hour {hour_from_admission} accepted",
    )


# ── Batch Ingestion (flat CSV-style rows) ────────────────────────────────

@app.post(
    "/ingest/batch",
    response_model=IngestionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Ingestion"],
)
async def ingest_batch(payload: FlatBatchPayload):
    """
    Ingest a batch of hourly ICU readings (flat format matching CSV schema).
    Each record is published to `icu-batch` and stored in the data lake.
    """
    ingestion_id = uuid.uuid4().hex[:12]
    valid_count = 0
    rejected_count = 0

    for record in payload.records:
        try:
            record_dict = record.model_dump(exclude_none=True)
            record_dict["ingestion_id"] = ingestion_id

            app.state.producer.produce_batch(record.patient_id, record_dict)

            app.state.data_lake.store(
                data=record_dict,
                patient_id=str(record.patient_id),
                data_type=kafka_settings.batch_topic,
            )
            valid_count += 1

        except Exception as e:
            logger.warning(
                "Record rejected (patient={}, hour={}): {}",
                record.patient_id,
                record.hour_from_admission,
                e,
            )
            rejected_count += 1

    # Flush producer to ensure all messages are delivered
    app.state.producer.flush(timeout=5.0)

    return IngestionResponse(
        status="accepted",
        records_received=len(payload.records),
        records_valid=valid_count,
        records_rejected=rejected_count,
        ingestion_id=ingestion_id,
        message=f"Batch ingestion complete: {valid_count} accepted, {rejected_count} rejected",
    )


# ── Single Flat Row Ingestion ────────────────────────────────────────────

@app.post(
    "/ingest/record",
    response_model=IngestionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Ingestion"],
)
async def ingest_single_record(record: FlatICURow):
    """
    Ingest a single flat ICU row (matching the CSV schema).
    """
    ingestion_id = uuid.uuid4().hex[:12]
    record_dict = record.model_dump(exclude_none=True)
    record_dict["ingestion_id"] = ingestion_id

    app.state.producer.produce_batch(record.patient_id, record_dict)
    app.state.data_lake.store(
        data=record_dict,
        patient_id=str(record.patient_id),
        data_type=kafka_settings.batch_topic,
    )

    return IngestionResponse(
        status="accepted",
        records_received=1,
        records_valid=1,
        ingestion_id=ingestion_id,
        message=f"Record for patient {record.patient_id} at hour {record.hour_from_admission} accepted",
    )


# ── Data Lake Query ──────────────────────────────────────────────────────

@app.get("/datalake/partitions", tags=["Data Lake"])
async def list_partitions(
    data_type: str = Query(default="icu-batch", description="Data category"),
):
    """List available date partitions in the data lake."""
    partitions = app.state.data_lake.list_partitions(data_type=data_type)
    return {"data_type": data_type, "partitions": partitions, "count": len(partitions)}


# ── CLI entrypoint ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.ingestion.ingest_api:app",
        host=api_settings.host,
        port=api_settings.port,
        reload=api_settings.reload,
    )
