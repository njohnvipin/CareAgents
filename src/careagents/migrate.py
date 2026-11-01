"""Apply SQL migrations in order: `python -m careagents.migrate`.

Safe to run from several pods at once (advisory lock); each file applies in one transaction.
Run it as a pre-deploy Job, never from API startup.
"""

from __future__ import annotations

import asyncio
from importlib import resources

import asyncpg

from careagents.config import get_settings

LOCK_ID = 727_274_001


async def _setup_checkpointer(dsn: str) -> None:
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from psycopg import AsyncConnection
    from psycopg.rows import dict_row

    async with await AsyncConnection.connect(dsn, autocommit=True, row_factory=dict_row) as conn:
        await AsyncPostgresSaver(conn).setup()


async def migrate(dsn: str) -> list[str]:
    applied: list[str] = []
    con = await asyncpg.connect(dsn)
    try:
        await con.execute("SELECT pg_advisory_lock($1)", LOCK_ID)
        await con.execute("CREATE TABLE IF NOT EXISTS schema_migrations "
                          "(name text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())")
        done = {r["name"] for r in await con.fetch("SELECT name FROM schema_migrations")}
        files = sorted((f for f in resources.files("careagents.migrations").iterdir() if f.name.endswith(".sql")),
                       key=lambda f: f.name)
        for f in files:
            if f.name in done:
                continue
            async with con.transaction():
                await con.execute(f.read_text())
                await con.execute("INSERT INTO schema_migrations (name) VALUES ($1)", f.name)
            applied.append(f.name)
        await _setup_checkpointer(dsn)  # LangGraph's own tables, idempotent, still under our lock
    finally:
        await con.execute("SELECT pg_advisory_unlock($1)", LOCK_ID)
        await con.close()
    return applied


if __name__ == "__main__":
    print({"applied": asyncio.run(migrate(get_settings().database_url)), "langgraph_checkpointer": "ready"})
