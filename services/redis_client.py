import redis
from functools import lru_cache
from core.config import settings


@lru_cache()
def get_redis_client():
    return redis.Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        db=0,
        decode_responses=True
    )


def publish(channel: str, message: dict):
    client = get_redis_client()
    client.publish(channel, settings.json_dumps(message))


def set_json(key: str, value: dict, expire: int = None):
    client = get_redis_client()
    client.set(key, settings.json_dumps(value), ex=expire)


def get_json(key: str):
    client = get_redis_client()
    data = client.get(key)
    return settings.json_loads(data) if data else None
