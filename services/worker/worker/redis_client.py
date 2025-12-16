import os
import json
import redis

class RedisClient:
    def __init__(self, url: str = None):
        url = url or os.getenv("REDIS_URL", "redis://redis:6379")
        
        self._r = redis.from_url(url, decode_responses=True)

    def xadd(self, stream_name: str, mapping: dict):
        try:
            self._r.xadd(stream_name, mapping)
        except Exception:
            pass

    def xread(self, streams, count=1, block=1000):
        try:
            return self._r.xread(streams=streams, count=count, block=block)
        except Exception:
            return None
