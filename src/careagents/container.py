"""Composition root: builds every dependency once per process and tears them down cleanly."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from careagents.agents.runtime import AgentRuntime
from careagents.auth.security import TokenVerifier
from careagents.config import Settings
from careagents.data.cache import (
    Cache,
    LocalCache,
    LocalRateLimiter,
    RateLimiter,
    RedisCache,
    RedisRateLimiter,
)
from careagents.data.embeddings import Embedder, make_embedder
from careagents.data.models import AuditLogger, DocumentStore, Repositories
from careagents.data.search import DocumentSearch
from careagents.llm.client import LLMClient, OpenAICompatibleClient
from careagents.tools.catalog import build_tools

logger = logging.getLogger(__name__)


@dataclass
class Container:
    settings: Settings
    llm: LLMClient
    repos: Repositories
    store: DocumentStore
    audit: AuditLogger
    embedder: Embedder
    search: DocumentSearch
    limiter: RateLimiter
    verifier: TokenVerifier
    runtime: AgentRuntime
    redis: Any = None
    job_queue: Any = None
    _pool: Any = None
    _checkpoint_pool: Any = None
    _closers: list[Any] = field(default_factory=list)

    async def aclose(self) -> None:
        await self.llm.aclose()
        if self.job_queue is not None:
            await self.job_queue.aclose()
        if self.redis is not None:
            await self.redis.aclose()
        if self._checkpoint_pool is not None:
            await self._checkpoint_pool.close()
        if self._pool is not None:
            await self._pool.close()


async def build_container(settings: Settings, *, llm: LLMClient | None = None, with_queue: bool = True) -> Container:
    pool = None
    checkpoint_pool = None
    checkpointer: Any = None
    if settings.data_backend == "postgres":
        from careagents.data.postgres import (
            PgDocumentStore,
            PostgresAuditLogger,
            PostgresRepositories,
            create_pool,
        )
        pool = await create_pool(settings.database_url, settings.db_pool_min, settings.db_pool_max)
        repos: Repositories = PostgresRepositories(pool)
        store: DocumentStore = PgDocumentStore(pool)
        audit: AuditLogger = PostgresAuditLogger(pool)
        # LangGraph checkpoints live in the same database (tables created by `python -m careagents.migrate`).
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        from psycopg.rows import dict_row
        from psycopg_pool import AsyncConnectionPool
        checkpoint_pool = AsyncConnectionPool(
            conninfo=settings.database_url, min_size=1, max_size=settings.db_pool_max, open=False,
            kwargs={"autocommit": True, "prepare_threshold": None, "row_factory": dict_row})  # PgBouncer-safe
        await checkpoint_pool.open()
        from careagents.agents.graph import checkpoint_serde
        checkpointer = AsyncPostgresSaver(checkpoint_pool, serde=checkpoint_serde())  # type: ignore[arg-type]
    else:
        from careagents.data.memory import MemoryAuditLogger, MemoryDocumentStore, MemoryRepositories
        repos, store, audit = MemoryRepositories(), MemoryDocumentStore(), MemoryAuditLogger()

    redis = None
    cache: Cache | None
    limiter: RateLimiter
    if settings.redis_url:
        from redis.asyncio import Redis
        redis = Redis.from_url(settings.redis_url, socket_timeout=2, socket_connect_timeout=2)
        cache, limiter = RedisCache(redis), RedisRateLimiter(redis)
    else:
        cache, limiter = LocalCache(), LocalRateLimiter()

    embedder = make_embedder(settings)
    search = DocumentSearch(embedder, store, cache, settings.search_cache_ttl_s)
    llm = llm or OpenAICompatibleClient(settings)
    runtime = AgentRuntime(settings, llm, build_tools(), repos, search, audit, checkpointer=checkpointer)

    job_queue = None
    if with_queue and settings.redis_url:
        from arq import create_pool as create_arq_pool
        from arq.connections import RedisSettings
        job_queue = await create_arq_pool(RedisSettings.from_dsn(settings.redis_url))

    container = Container(settings=settings, llm=llm, repos=repos, store=store, audit=audit, embedder=embedder,
                          search=search, limiter=limiter, verifier=TokenVerifier(settings), runtime=runtime,
                          redis=redis, job_queue=job_queue, _pool=pool, _checkpoint_pool=checkpoint_pool)

    if settings.data_backend == "memory":  # dev convenience: memory store starts empty every boot
        from careagents.ingestion.pipeline import ingest_directory
        await ingest_directory(settings.documents_dir, store, embedder, search)
    return container
