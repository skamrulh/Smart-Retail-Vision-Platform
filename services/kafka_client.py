"""
Optional Kafka integration — standalone stub.

Set env var KAFKA_ENABLED=true and KAFKA_BOOTSTRAP_SERVERS to activate.
Importing this module does NOT require any external config package.
"""
import os
import json
import logging

logger = logging.getLogger(__name__)


class KafkaClient:
    """
    Lightweight Kafka producer stub.

    In production, swap the stub implementation for a real aiokafka producer.
    """

    def __init__(self):
        self.enabled  = os.getenv("KAFKA_ENABLED", "false").lower() == "true"
        self.servers  = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        self._producer = None

        if self.enabled:
            self._try_connect()

    def _try_connect(self):
        try:
            from aiokafka import AIOKafkaProducer  # optional heavy dep
            # Producer is started inside an async context; store config only
            self._config = {"bootstrap_servers": self.servers}
            logger.info(f"Kafka configured (servers={self.servers})")
        except ImportError:
            logger.warning("aiokafka not installed — Kafka stub active.")
            self.enabled = False

    def send(self, topic: str, message: dict) -> None:
        """Fire-and-forget publish (sync stub). Replace with async producer in production."""
        if not self.enabled:
            return
        logger.debug(f"[KafkaStub] → {topic}: {json.dumps(message)[:120]}")
