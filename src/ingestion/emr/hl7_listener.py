"""
G_One_Sync AI — HL7v2 MLLP Listener
=======================================
Async TCP server implementing the HL7v2 Minimal Lower Layer Protocol (MLLP).
Receives HL7v2 messages from EMR systems, parses, maps, and publishes
to the existing Kafka/data-lake pipeline.

Runs as a standalone process:
    python -m src.ingestion.emr.hl7_listener
"""

from __future__ import annotations

import asyncio
import signal
import time
import uuid
from typing import Any

from loguru import logger

from config.settings import emr_settings, kafka_settings
from src.ingestion.data_lake_router import DataLakeRouter
from src.ingestion.emr.audit_logger import AuditLogger
from src.ingestion.emr.hl7_mapper import HL7Mapper
from src.ingestion.emr.hl7_parser import (
    MLLP_END_BYTES,
    MLLP_START_BYTE,
    HL7ParseError,
    HL7Parser,
)
from src.ingestion.kafka_producer import get_producer


class MLLPServer:
    """
    Async HL7v2 MLLP listener for receiving ADT/ORU messages.

    MLLP framing:
        <SB>message<EB><CR>
        SB = 0x0B (vertical tab)
        EB = 0x1C (file separator)
        CR = 0x0D (carriage return)

    Flow:
        1. Receive MLLP-framed message
        2. Parse with HL7Parser
        3. Map with HL7Mapper → VitalSignsPayload / LabResultsPayload
        4. Publish to Kafka via existing producer
        5. Store in data lake via existing router
        6. Write audit log
        7. Return HL7 ACK/NAK
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
    ):
        self.host = host or emr_settings.hl7_mllp_host
        self.port = port or emr_settings.hl7_mllp_port
        self._server: asyncio.AbstractServer | None = None
        self._running = False
        self._connections = 0
        self._messages_processed = 0

        # Pipeline components
        self._producer = get_producer()
        self._data_lake = DataLakeRouter()
        self._audit = AuditLogger(emr_settings.audit_log_dir)

        logger.info(
            "MLLPServer initialized | host={} | port={}",
            self.host, self.port,
        )

    async def handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """
        Handle a single MLLP client connection.

        Reads complete MLLP-framed messages from the stream and
        processes each one. Supports persistent connections (multiple
        messages per connection).
        """
        peer = writer.get_extra_info("peername")
        self._connections += 1
        logger.info("HL7 client connected: {} (active={})", peer, self._connections)

        buffer = b""

        try:
            while self._running:
                try:
                    chunk = await asyncio.wait_for(reader.read(65536), timeout=300.0)
                except asyncio.TimeoutError:
                    logger.debug("HL7 client {} timed out", peer)
                    break

                if not chunk:
                    break  # Client disconnected

                buffer += chunk

                # Process all complete messages in the buffer
                while MLLP_START_BYTE in buffer and MLLP_END_BYTES in buffer:
                    start_idx = buffer.index(MLLP_START_BYTE)
                    end_idx = buffer.index(MLLP_END_BYTES) + len(MLLP_END_BYTES)

                    raw_message = buffer[start_idx:end_idx]
                    buffer = buffer[end_idx:]

                    # Process and get ACK/NAK
                    ack_bytes = await self.process_message(raw_message)
                    writer.write(ack_bytes)
                    await writer.drain()

        except ConnectionResetError:
            logger.debug("HL7 client {} disconnected", peer)
        except Exception as e:
            logger.error("Error handling HL7 client {}: {}", peer, e)
        finally:
            self._connections -= 1
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
            logger.info("HL7 client disconnected: {} (active={})", peer, self._connections)

    async def process_message(self, raw: bytes) -> bytes:
        """
        Process a single MLLP-framed HL7v2 message.

        Args:
            raw: Raw MLLP-framed bytes

        Returns:
            MLLP-framed ACK or NAK response bytes
        """
        start = time.time()
        ingestion_id = uuid.uuid4().hex[:12]

        try:
            # Strip MLLP framing and parse
            message_str = HL7Parser.strip_mllp_framing(raw)
            parsed = HL7Parser.parse_message(message_str)

            message_type = parsed.get("message_type", "UNKNOWN")
            control_id = parsed.get("message_control_id", ingestion_id)

            logger.info("HL7 message received: type={} control_id={}", message_type, control_id)

            records_ingested = 0

            if message_type == "ORU^R01":
                records_ingested = self._process_oru(parsed, ingestion_id)
            elif message_type in ("ADT^A01", "ADT^A03"):
                self._process_adt(parsed, ingestion_id)
                records_ingested = 1
            else:
                logger.warning("Unsupported HL7 message type: {}", message_type)

            elapsed_ms = (time.time() - start) * 1000
            self._messages_processed += 1

            # Audit log
            self._audit.log_event(
                event_type=f"hl7_{message_type.lower().replace('^', '_')}_ingest",
                source="emr-hl7",
                patient_id=parsed.get("patient", {}).get("patient_id", "unknown"),
                record_count=records_ingested,
                metadata={
                    "message_type": message_type,
                    "control_id": control_id,
                    "ingestion_id": ingestion_id,
                    "duration_ms": round(elapsed_ms, 2),
                },
            )

            # Return ACK
            ack = HL7Parser.build_ack(control_id, "AA", "Message accepted")
            return MLLP_START_BYTE + ack.encode("utf-8") + MLLP_END_BYTES

        except HL7ParseError as e:
            logger.error("HL7 parse error: {}", e)
            self._audit.log_event(
                event_type="hl7_parse_error",
                source="emr-hl7",
                patient_id="unknown",
                status="error",
                metadata={"error": str(e)},
            )
            nak = HL7Parser.build_ack(ingestion_id, "AE", f"Parse error: {e}")
            return MLLP_START_BYTE + nak.encode("utf-8") + MLLP_END_BYTES

        except Exception as e:
            logger.error("HL7 processing error: {}", e)
            nak = HL7Parser.build_ack(ingestion_id, "AR", f"Processing error: {e}")
            return MLLP_START_BYTE + nak.encode("utf-8") + MLLP_END_BYTES

    def _process_oru(self, parsed: dict, ingestion_id: str) -> int:
        """Process ORU^R01 — map OBX segments to vitals/labs and publish."""
        observations = parsed.get("observations", [])
        if not observations:
            return 0

        # Extract patient_id from first observation
        patient_id_str = observations[0].get("patient_id", "0")
        try:
            patient_id = int(patient_id_str)
        except ValueError:
            patient_id = abs(hash(patient_id_str)) % 1_000_000

        records = 0

        # Try mapping to vitals
        vitals = HL7Mapper.obx_segments_to_vitals(observations)
        if vitals:
            record = {
                "patient_id": patient_id,
                "ingestion_id": ingestion_id,
                "source": "emr-hl7",
                **vitals.model_dump(),
            }
            self._producer.produce_vitals(patient_id, record)
            self._data_lake.store(
                data=record,
                patient_id=str(patient_id),
                data_type=kafka_settings.vitals_topic,
            )
            records += 1

        # Try mapping to labs
        labs = HL7Mapper.obx_segments_to_labs(observations)
        if labs:
            record = {
                "patient_id": patient_id,
                "ingestion_id": ingestion_id,
                "source": "emr-hl7",
                **labs.model_dump(),
            }
            self._producer.produce_labs(patient_id, record)
            self._data_lake.store(
                data=record,
                patient_id=str(patient_id),
                data_type=kafka_settings.labs_topic,
            )
            records += 1

        return records

    def _process_adt(self, parsed: dict, ingestion_id: str) -> None:
        """Process ADT^A01/A03 — store patient demographics."""
        patient_data = parsed.get("patient", {})
        if not patient_data:
            return

        patient_id_str = patient_data.get("patient_id", "0")
        try:
            patient_id = int(patient_id_str)
        except ValueError:
            patient_id = abs(hash(patient_id_str)) % 1_000_000

        record = {
            "patient_id": patient_id,
            "ingestion_id": ingestion_id,
            "source": "emr-hl7",
            "event": parsed.get("event", "admission"),
            **patient_data,
        }

        self._data_lake.store(
            data=record,
            patient_id=str(patient_id),
            data_type="icu-adt",
        )

    async def start(self) -> None:
        """Start the MLLP TCP server."""
        self._running = True

        self._server = await asyncio.start_server(
            self.handle_client,
            self.host,
            self.port,
        )

        addrs = [sock.getsockname() for sock in self._server.sockets]
        logger.info("🏥 HL7v2 MLLP server listening on {}", addrs)

        # Graceful shutdown on SIGINT/SIGTERM
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self.stop)
            except NotImplementedError:
                # Windows doesn't support add_signal_handler
                pass

        try:
            async with self._server:
                await self._server.serve_forever()
        except asyncio.CancelledError:
            pass
        finally:
            logger.info(
                "HL7v2 MLLP server stopped — {} messages processed",
                self._messages_processed,
            )

    def stop(self) -> None:
        """Signal the server to stop."""
        self._running = False
        if self._server:
            self._server.close()
        logger.info("HL7v2 MLLP server shutdown requested")


# ── CLI Entrypoint ───────────────────────────────────────────────────

if __name__ == "__main__":
    if not emr_settings.hl7_enabled:
        logger.warning(
            "HL7 MLLP listener is disabled. "
            "Set JEEVAN_EMR_HL7_ENABLED=true to enable."
        )
    else:
        logger.info("Starting HL7v2 MLLP listener...")
        server = MLLPServer()
        asyncio.run(server.start())
