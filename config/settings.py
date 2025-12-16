import os
import json
from pydantic import BaseSettings


class Settings(BaseSettings):
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379

    MODEL_PATH: str = "models/detector.onnx"
    OCR_MODEL_PATH: str = "models/ocr.onnx"

    KAFKA_ENABLED: bool = False

    def json_dumps(self, obj):
        return json.dumps(obj)

    def json_loads(self, s):
        return json.loads(s)


settings = Settings()
