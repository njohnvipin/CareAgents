"""Small Redis helpers: cache and fixed-window rate limiter, with in-process fallbacks for dev."""

from __future__ import annotations

import time
from typing import Protocol

from redis.asyncio import Redis


class Cache(Protocol):
    async def get(self, key: str) -> str | None: ...
    async def set(self, key: str, value: str, ttl_s: int) -> None: ...
    async def incr(self, key: str) -> int: ...


class RedisCache:
    def __init__(self, redis: Redis):
        self.redis = redis

    async def get(self, key: str) -> str | None:
        value = await self.redis.get(key)
        return value.decode() if isinstance(value, bytes) else value

    async def set(self, key: str, value: str, ttl_s: int) -> None:
        await self.redis.set(key, value, ex=ttl_s)

    async def incr(self, key: str) -> int:
        return int(await self.redis.incr(key))


class LocalCache:
    def __init__(self) -> None:
        self._data: dict[str, tuple[str, float]] = {}

    async def get(self, key: str) -> str | None:
        item = self._data.get(key)
        if not item or item[1] < time.monotonic():
            return None
        return item[0]

    async def set(self, key: str, value: str, ttl_s: int) -> None:
        self._data[key] = (value, time.monotonic() + ttl_s)

    async def incr(self, key: str) -> int:
        current = int((await self.get(key)) or 0) + 1
        await self.set(key, str(current), 10**9)
        return current


class RateLimiter(Protocol):
    async def hit(self, key: str, limit: int, window_s: int) -> tuple[bool, int]: ...


class RedisRateLimiter:
    """Fixed window shared by every replica."""

    def __init__(self, redis: Redis):
        self.redis = redis

    async def hit(self, key: str, limit: int, window_s: int) -> tuple[bool, int]:
        now = int(time.time())
        bucket = now // window_s
        k = f"rl:{key}:{bucket}"
        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.incr(k)
            pipe.expire(k, window_s + 1)
            count, _ = await pipe.execute()
        retry_after = window_s - (now % window_s)
        return int(count) <= limit, retry_after


class LocalRateLimiter:
    """Per-process only; fine for dev, wrong behind a load balancer."""

    def __init__(self) -> None:
        self._counts: dict[str, int] = {}

    async def hit(self, key: str, limit: int, window_s: int) -> tuple[bool, int]:
        now = int(time.time())
        k = f"{key}:{now // window_s}"
        self._counts[k] = self._counts.get(k, 0) + 1
        if len(self._counts) > 10_000:
            self._counts = {k: self._counts[k]}
        return self._counts[k] <= limit, window_s - (now % window_s)
