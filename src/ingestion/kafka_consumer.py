"""
JeevanSync AI — Kafka / Redpanda Consumer
==========================================
Subscribes to ICU data topics, validates incoming messages,
routes data to the data lake, and triggers preprocessing.
"""

from __future__ import annotations

import json
import signal
import sys
import threading
from typing import Callable, Optional

from loguru import logger

from config.settings import kafka_settings
from src.ingestion.data_lake_router import DataLakeRouter


class KafkaConsumer:
    """
    Kafka/Redpanda consumer that processes ICU data streams.
    Supports graceful shutdown and pluggable message handlers.
    """

    def __init__(
        self,
        topics: Optional[list[str]] = None,
        group_id: Optional[str] = None,
        message_handler: Optional[Callable] = None,
    ):
        self.topics = topics or [
            kafka_settings.vitals_topic,
            kafka_settings.labs_topic,
            kafka_settings.batch_topic,
        ]
        self.group_id = group_id or kafka_settings.consumer_group
        self.message_handler = message_handler or self._default_handler
        self._running = False
        self._consumer = None
        self._data_lake = DataLakeRouter()

        # Graceful shutdown on SIGINT / SIGTERM
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame) -> None:
        logger.info("Shutdown signal received — stopping consumer...")
        self._running = False

    def _initialize(self) -> None:
        """Initialize the confluent-kafka consumer."""
        try:
            from confluent_kafka import Consumer

            conf = {
                "bootstrap.servers": kafka_settings.bootstrap_servers,
                "group.id": self.group_id,
                "auto.offset.reset": kafka_settings.auto_offset_reset,
                "enable.auto.commit": True,
                "auto.commit.interval.ms": 5000,
                "session.timeout.ms": 30000,
                "max.poll.interval.ms": 300000,
            }
            self._consumer = Consumer(conf)
            self._consumer.subscribe(self.topics)
            logger.info(
                "Kafka consumer subscribed to topics: {} (group={})",
                self.topics,
                self.group_id,
            )
        except Exception as e:
            logger.error("Failed to initialize Kafka consumer: {}", e)
            raise

    def _default_handler(self, topic: str, key: str, value: dict) -> None:
        """
        Default message handler:
        1. Route raw payload to data lake
        2. Log the ingestion event
        """
        patient_id = value.get("patient_id", key)

        # Store in data lake
        self._data_lake.store(
            data=value,
            patient_id=str(patient_id),
            data_type=topic,
        )

        logger.debug(
            "Processed message: topic={}, patient_id={}, hour={}",
            topic,
            patient_id,
            value.get("hour_from_admission", "N/A"),
        )

    def consume(self, poll_timeout: float = 1.0) -> None:
        """
        Start the main consumer loop.
        Polls messages and dispatches to the handler.
        """
        self._initialize()
        self._running = True

        logger.info("Consumer loop started — polling every {}s", poll_timeout)
        messages_processed = 0

        try:
            while self._running:
                msg = self._consumer.poll(timeout=poll_timeout)

                if msg is None:
                    continue

                if msg.error():
                    from confluent_kafka import KafkaError

                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        logger.debug(
                            "End of partition: {}/{}",
                            msg.topic(),
                            msg.partition(),
                        )
                    else:
                        logger.error("Consumer error: {}", msg.error())
                    continue

                # Deserialize message
                try:
                    key = msg.key().decode("utf-8") if msg.key() else ""
                    value = json.loads(msg.value().decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    logger.error(
                        "Failed to deserialize message from {}: {}",
                        msg.topic(),
                        e,
                    )
                    continue

                # Dispatch to handler
                self.message_handler(msg.topic(), key, value)
                messages_processed += 1

                if messages_processed % 1000 == 0:
                    logger.info("Processed {} messages so far", messages_processed)

        finally:
            logger.info(
                "Consumer shutting down — {} total messages processed",
                messages_processed,
            )
            if self._consumer:
                self._consumer.close()

    def consume_async(self) -> threading.Thread:
        """Start the consumer in a background thread."""
        thread = threading.Thread(target=self.consume, daemon=True, name="kafka-consumer")
        thread.start()
        logger.info("Kafka consumer started in background thread")
        return thread

    def stop(self) -> None:
        """Signal the consumer to stop."""
        self._running = False


class LocalQueueConsumer:
    """
    Fallback consumer that reads from the local file-based queue.
    Used in development when Redpanda/Kafka is not available.
    """

    def __init__(self, message_handler: Optional[Callable] = None):
        self.queue_dir = kafka_settings.local_queue_dir
        self._data_lake = DataLakeRouter()
        self.message_handler = message_handler or self._default_handler

    def _default_handler(self, topic: str, key: str, value: dict) -> None:
        patient_id = value.get("patient_id", key)
        self._data_lake.store(
            data=value,
            patient_id=str(patient_id),
            data_type=topic,
        )

    def consume_all(self) -> int:
        """
        Process all messages in the local file queue.
        Returns the number of messages processed.
        """
        if not self.queue_dir.exists():
            logger.warning("Local queue directory does not exist: {}", self.queue_dir)
            return 0

        total = 0
        for topic_dir in self.queue_dir.iterdir():
            if not topic_dir.is_dir():
                continue

            topic = topic_dir.name
            for jsonl_file in sorted(topic_dir.glob("*.jsonl")):
                with open(jsonl_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            envelope = json.loads(line)
                            self.message_handler(
                                topic,
                                envelope.get("key", ""),
                                envelope.get("value", {}),
                            )
                            total += 1
                        except json.JSONDecodeError as e:
                            logger.error("Invalid JSON in {}: {}", jsonl_file.name, e)

        logger.info("LocalQueueConsumer processed {} messages", total)
        return total


# ── CLI entrypoint ───────────────────────────────────────────────────────
if __name__ == "__main__":
    if kafka_settings.use_local_fallback:
        logger.info("Starting local queue consumer...")
        consumer = LocalQueueConsumer()
        consumer.consume_all()
    else:
        logger.info("Starting Kafka consumer...")
        consumer = KafkaConsumer()
        consumer.consume()
