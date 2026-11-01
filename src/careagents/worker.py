"""Background worker (arq on Redis): `arq careagents.worker.WorkerSettings`.

Ingestion (embedding whole document sets) is too slow for a request/response cycle, so the API
enqueues it here. Scale workers independently of the API.
"""

from __future__ import annotations

import logging
from typing import Any

from arq.connections import RedisSettings

from careagents.config import get_settings
from careagents.container import build_container
from careagents.ingestion.pipeline import ingest_directory
from careagents.logging_setup import configure_logging

logger = logging.getLogger(__name__)
LOCK_KEY = "ca:lock:ingest"


async def ingest_documents(ctx: dict[str, Any]) -> dict[str, Any]:
    container = ctx["container"]
    lock = ctx["redis"].lock(LOCK_KEY, timeout=1800, blocking=False)
    if not await lock.acquire():
        return {"status": "skipped", "reason": "another ingestion is running"}
    try:
        stats = await ingest_directory(container.settings.documents_dir, container.store, container.embedder,
                                       container.search)
        return {"status": "completed", **stats}
    finally:
        await lock.release()


async def startup(ctx: dict[str, Any]) -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    ctx["container"] = await build_container(settings, with_queue=False)


async def shutdown(ctx: dict[str, Any]) -> None:
    await ctx["container"].aclose()


class WorkerSettings:
    functions = [ingest_documents]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url or "redis://localhost:6379")
    max_jobs = 4
    job_timeout = 1800
    keep_result = 86400
