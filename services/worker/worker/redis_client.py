"""
Async Redis client for the worker service.

Uses redis.asyncio so that both camera_consumer.py and processing.py
can correctly `await` all stream operations.
"""
import os
import json
import logging
import redis.asyncio as aioredis

logger = logging.getLogger(__name__)


class RedisClient:
    """Thin async wrapper around Redis Streams and pub/sub."""

    def __init__(self, url: str = None):
        url = url or os.getenv("REDIS_URL", "redis://redis:6379")
        # Connection is established lazily on first command
        self._r = aioredis.from_url(url, decode_responses=True)

    async def xadd(
        self,
        stream_name: str,
        mapping: dict,
        maxlen: int = 10_000,
    ) -> None:
        """Append an entry to a Redis Stream. Silently ignores connection errors."""
        try:
            await self._r.xadd(stream_name, mapping, maxlen=maxlen, approximate=True)
        except Exception as e:
            logger.warning(f"Redis xadd failed on stream '{stream_name}': {e}")

    async def xread(
        self,
        streams: dict,
        count: int = 1,
        block: int = 1000,
    ):
        """
        Read entries from one or more Redis Streams.

        Args:
            streams: dict mapping stream names to last-seen IDs,
                     e.g. {"camera:frames": "$"}.
            count:   Maximum entries to return per stream.
            block:   Milliseconds to block waiting for new entries.

        Returns:
            List of (stream_name, entries) tuples, or None on error.
        """
        try:
            return await self._r.xread(streams=streams, count=count, block=block)
        except Exception as e:
            logger.warning(f"Redis xread failed: {e}")
            return None

    async def close(self) -> None:
        await self._r.aclose()
