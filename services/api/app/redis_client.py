import os
import json
import redis.asyncio as aioredis

class RedisClient:
    def __init__(self, url: str = None):
        url = url or os.getenv("REDIS_URL", "redis://redis:6379")
        self._r = aioredis.from_url(url, decode_responses=True)

    @classmethod
    def from_env(cls):
        return cls()

    async def publish_event(self, stream_name: str, event: dict):
        try:
            await self._r.xadd(stream_name, {"data": json.dumps(event)}, maxlen=10000, approximate=False)
        except Exception:
            
            pass

    async def publish_pubsub(self, channel: str, message: dict):
        try:
            await self._r.publish(channel, json.dumps(message))
        except Exception:
            pass
