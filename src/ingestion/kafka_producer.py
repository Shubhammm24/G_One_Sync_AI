"""
JeevanSync AI — Kafka / Redpanda Producer
==========================================
Publishes validated ICU data to Kafka-compatible topics.
Falls back to a local file-based queue when the broker is unavailable.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from config.settings import kafka_settings


class LocalFileQueue:
    """
    File-based fallback queue for development when Redpanda/Kafka is not running.
    Writes each message as a JSON-lines file partitioned by topic.
    """

    def __init__(self, queue_dir: Path):
        self.queue_dir = queue_dir
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        logger.info("LocalFileQueue initialized at {}", self.queue_dir)

    def produce(self, topic: str, value: dict, key: Optional[str] = None) -> None:
        topic_dir = self.queue_dir / topic
        topic_dir.mkdir(parents=True, exist_ok=True)

        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        filepath = topic_dir / f"{date_str}.jsonl"

        envelope = {
            "key": key or str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "value": value,
        }

        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(envelope, default=str) + "\n")

    def flush(self) -> None:
        pass  # No-op for file queue


class KafkaProducer:
    """
    Kafka/Redpanda producer with automatic fallback to local file queue.
    Uses confluent-kafka under the hood for production compatibility.
    """

    def __init__(self):
        self._producer = None
        self._fallback = None
        self._connected = False
        self._initialize()

    def _initialize(self) -> None:
        """Try to connect to Kafka/Redpanda broker; fall back to file queue."""
        try:
            from confluent_kafka import Producer

            conf = {
                "bootstrap.servers": kafka_settings.bootstrap_servers,
                "client.id": "jeevansync-producer",
                "acks": "all",
                "retries": 3,
                "retry.backoff.ms": 500,
                "linger.ms": 10,
                "batch.num.messages": 100,
            }
            self._producer = Producer(conf)

            # Quick connectivity check
            self._producer.list_topics(timeout=5)
            self._connected = True
            logger.info(
                "Kafka producer connected to {}", kafka_settings.bootstrap_servers
            )

        except Exception as e:
            logger.warning(
                "Kafka broker unavailable ({}). Using local file queue fallback.", str(e)
            )
            self._connected = False

            if kafka_settings.use_local_fallback:
                self._fallback = LocalFileQueue(kafka_settings.local_queue_dir)
            else:
                raise ConnectionError(
                    f"Kafka broker at {kafka_settings.bootstrap_servers} is unreachable "
                    "and local fallback is disabled."
                )

    @property
    def is_connected(self) -> bool:
        return self._connected

    def _delivery_callback(self, err, msg) -> None:
        """Callback for Kafka delivery reports."""
        if err:
            logger.error("Message delivery failed: {} [topic={}]", err, msg.topic())
        else:
            logger.debug(
                "Message delivered to {}/{} [offset={}]",
                msg.topic(),
                msg.partition(),
                msg.offset(),
            )

    def produce(
        self,
        topic: str,
        value: dict[str, Any],
        key: Optional[str] = None,
    ) -> None:
        """
        Publish a message to the specified topic.

        Args:
            topic: Kafka topic name
            value: Message payload (dict, will be JSON-serialized)
            key: Optional partition key (defaults to UUID)
        """
        message_key = key or str(uuid.uuid4())
        serialized = json.dumps(value, default=str).encode("utf-8")

        if self._connected and self._producer:
            try:
                self._producer.produce(
                    topic=topic,
                    value=serialized,
                    key=message_key.encode("utf-8"),
                    callback=self._delivery_callback,
                )
                self._producer.poll(0)  # Trigger delivery callbacks
            except BufferError:
                logger.warning("Kafka producer queue full — flushing and retrying")
                self._producer.flush(timeout=5)
                self._producer.produce(
                    topic=topic,
                    value=serialized,
                    key=message_key.encode("utf-8"),
                    callback=self._delivery_callback,
                )
        elif self._fallback:
            self._fallback.produce(topic, value, message_key)
        else:
            raise RuntimeError("No Kafka producer or fallback queue available")

    def produce_vitals(self, patient_id: int, data: dict) -> None:
        """Publish vital signs to the vitals topic."""
        self.produce(
            topic=kafka_settings.vitals_topic,
            value=data,
            key=str(patient_id),
        )

    def produce_labs(self, patient_id: int, data: dict) -> None:
        """Publish lab results to the labs topic."""
        self.produce(
            topic=kafka_settings.labs_topic,
            value=data,
            key=str(patient_id),
        )

    def produce_batch(self, patient_id: int, data: dict) -> None:
        """Publish full ICU batch record to the batch topic."""
        self.produce(
            topic=kafka_settings.batch_topic,
            value=data,
            key=str(patient_id),
        )

    def flush(self, timeout: float = 10.0) -> None:
        """Flush all pending messages."""
        if self._connected and self._producer:
            remaining = self._producer.flush(timeout=timeout)
            if remaining > 0:
                logger.warning("{} messages still in queue after flush", remaining)
        if self._fallback:
            self._fallback.flush()

    def close(self) -> None:
        """Gracefully shutdown the producer."""
        self.flush()
        logger.info("Kafka producer closed")


# ── Module-level singleton ───────────────────────────────────────────────
_producer_instance: Optional[KafkaProducer] = None


def get_producer() -> KafkaProducer:
    """Get or create the singleton Kafka producer instance."""
    global _producer_instance
    if _producer_instance is None:
        _producer_instance = KafkaProducer()
    return _producer_instance
