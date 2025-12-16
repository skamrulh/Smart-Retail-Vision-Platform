# Optional Kafka integration stub
from core.config import settings


class KafkaClient:
    def __init__(self):
        self.enabled = settings.KAFKA_ENABLED

    def send(self, topic: str, message: dict):
        if not self.enabled:
            return
        print(f"[Kafka Stub] Sent to {topic}: {message}")
