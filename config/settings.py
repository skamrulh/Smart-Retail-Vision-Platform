"""
Central application settings.

Uses pydantic-settings so every field can be overridden via environment
variables or a .env file — no code changes needed for different deployments.
"""
import os
from pydantic_settings import BaseSettings   # was: from pydantic import BaseSettings (pydantic v1 only)


class Settings(BaseSettings):
    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # Redis
    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379
    REDIS_URL:  str = "redis://redis:6379"

    # Models
    MODEL_PATH:     str = "models/detector.onnx"
    OCR_MODEL_PATH: str = "models/ocr.onnx"

    # Optional integrations
    KAFKA_ENABLED: bool = False
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"

    # Storage
    MINIO_ENDPOINT:   str = "minio:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"

    # Database
    DATABASE_URL: str = "postgresql://sr_user:sr_pass@postgres:5432/sr_meta"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
