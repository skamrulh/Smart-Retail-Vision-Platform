"""
Synchronous Redis utility helpers for non-async service code.

Does NOT import from any internal config package; all settings are read
from environment variables so this module works standalone.
"""
import os
import json
import redis
from functools import lru_cache


@lru_cache(maxsize=1)
def get_redis_client() -> redis.Redis:
    return redis.Redis(
        host=os.getenv("REDIS_HOST", "redis"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        db=0,
        decode_responses=True,
    )


def publish(channel: str, message: dict) -> None:
    get_redis_client().publish(channel, json.dumps(message))


def set_json(key: str, value: dict, expire: int = None) -> None:
    get_redis_client().set(key, json.dumps(value), ex=expire)


def get_json(key: str):
    data = get_redis_client().get(key)
    return json.loads(data) if data else None
