"""Postgres backend (asyncpg) with pgvector for document search.

One datastore for records, vectors and audit keeps operations simple; split vectors out to a
dedicated store only if you outgrow pgvector's HNSW index (tens of millions of chunks).
Put PgBouncer (transaction mode) in front when replica count x pool size approaches max_connections.
"""

from __future__ import annotations

import json
from typing import Any

import asyncpg

from careagents.data.models import (
    Accumulator,
    ChunkIn,
    Claim,
    Member,
    Plan,
    PriorAuth,
    SearchHit,
    Service,
)


def _vec(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.7f}" for x in v) + "]"


async def create_pool(dsn: str, min_size: int, max_size: int) -> asyncpg.Pool:
    return await asyncpg.create_pool(dsn, min_size=min_size, max_size=max_size, command_timeout=10,
                                     statement_cache_size=0)  # 0 = safe behind PgBouncer


class PostgresRepositories:
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def ping(self) -> bool:
        async with self.pool.acquire() as con:
            return await con.fetchval("SELECT 1") == 1

    async def get_member(self, member_id: str) -> Member | None:
        row = await self.pool.fetchrow(
            "SELECT id, plan_id, active, coverage_start, coverage_end FROM members WHERE id=$1", member_id)
        return Member(**dict(row)) if row else None

    async def get_plan(self, plan_id: str) -> Plan | None:
        row = await self.pool.fetchrow(
            "SELECT id, name, plan_year, deductible, coinsurance_member_pct, oop_max FROM plans WHERE id=$1",
            plan_id)
        return Plan(**dict(row)) if row else None

    async def get_accumulator(self, member_id: str, plan_year: int) -> Accumulator | None:
        row = await self.pool.fetchrow(
            "SELECT member_id, plan_year, deductible_paid, oop_paid FROM accumulators "
            "WHERE member_id=$1 AND plan_year=$2", member_id, plan_year)
        return Accumulator(**dict(row)) if row else None

    async def get_service(self, code: str) -> Service | None:
        row = await self.pool.fetchrow(
            "SELECT code, name, aliases, category, requires_prior_auth, typical_charge FROM services WHERE code=$1",
            code)
        return Service(**dict(row)) if row else None

    async def find_services(self, query: str, limit: int = 5) -> list[Service]:
        rows = await self.pool.fetch(
            """
            SELECT code, name, aliases, category, requires_prior_auth, typical_charge
            FROM services
            WHERE code = $1
               OR name ILIKE '%' || $1 || '%'
               OR EXISTS (SELECT 1 FROM unnest(aliases) a WHERE $1 ILIKE '%' || a || '%' OR a ILIKE '%' || $1 || '%')
            ORDER BY (code = $1) DESC, length(name)
            LIMIT $2
            """,
            query.strip(), limit)
        return [Service(**dict(r)) for r in rows]

    async def list_claims(self, member_id: str, service_code: str | None = None,
                          status: str | None = None, limit: int = 20) -> list[Claim]:
        rows = await self.pool.fetch(
            """
            SELECT id, member_id, service_code, service_date, provider, billed_amount, status,
                   denial_code, denial_reason, member_paid, plan_paid
            FROM claims
            WHERE member_id = $1
              AND ($2::text IS NULL OR service_code = $2)
              AND ($3::text IS NULL OR status = $3)
            ORDER BY service_date DESC
            LIMIT $4
            """,
            member_id, service_code, status, limit)
        return [Claim(**dict(r)) for r in rows]

    async def get_claim(self, member_id: str, claim_id: str) -> Claim | None:
        # member_id in the WHERE clause is the isolation boundary: another member's claim is "not found".
        row = await self.pool.fetchrow(
            """
            SELECT id, member_id, service_code, service_date, provider, billed_amount, status,
                   denial_code, denial_reason, member_paid, plan_paid
            FROM claims WHERE id=$1 AND member_id=$2
            """, claim_id, member_id)
        return Claim(**dict(row)) if row else None

    async def get_prior_auths(self, member_id: str, service_code: str) -> list[PriorAuth]:
        rows = await self.pool.fetch(
            """
            SELECT id, member_id, service_code, status, requested_at, decided_at, valid_until
            FROM prior_authorizations WHERE member_id=$1 AND service_code=$2
            ORDER BY requested_at DESC
            """, member_id, service_code)
        return [PriorAuth(**dict(r)) for r in rows]

    async def save_case_run(self, record: dict[str, Any]) -> None:
        await self.pool.execute(
            """
            INSERT INTO case_runs (id, member_id, actor_id, question, outcome, routed_to, facts, answer,
                                   latency_ms, prompt_tokens, completion_tokens, llm_calls)
            VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8, $9, $10, $11, $12)
            ON CONFLICT (id) DO UPDATE SET
                outcome = EXCLUDED.outcome, facts = EXCLUDED.facts, answer = EXCLUDED.answer,
                latency_ms = case_runs.latency_ms + EXCLUDED.latency_ms,
                prompt_tokens = EXCLUDED.prompt_tokens, completion_tokens = EXCLUDED.completion_tokens,
                llm_calls = EXCLUDED.llm_calls
            """,
            record["case_id"], record["member_id"], record["actor_id"], record["question"], record["outcome"],
            record["routed_to"], json.dumps(record["facts"], default=str), record["answer"],
            record["latency_ms"], record["prompt_tokens"], record["completion_tokens"], record["llm_calls"])


    # ---- human review queue (the graph pauses with interrupt(); this is the index staff work from)
    async def enqueue_review(self, record: dict[str, Any]) -> None:
        await self.pool.execute(
            """
            INSERT INTO review_queue (case_id, member_id, reason, question, draft)
            VALUES ($1::uuid, $2, $3, $4, $5) ON CONFLICT (case_id) DO NOTHING
            """, record["case_id"], record["member_id"], record["reason"], record["question"], record.get("draft"))

    async def list_reviews(self, status: str = "pending", limit: int = 50) -> list[dict[str, Any]]:
        rows = await self.pool.fetch(
            """
            SELECT case_id::text, member_id, reason, question, draft, status, created_at, reviewer_id, decision,
                   decided_at
            FROM review_queue WHERE status = $1 ORDER BY created_at LIMIT $2
            """, status, limit)
        return [dict(r) for r in rows]

    async def get_review(self, case_id: str) -> dict[str, Any] | None:
        row = await self.pool.fetchrow(
            """
            SELECT case_id::text, member_id, reason, question, draft, status, created_at, reviewer_id, decision,
                   decided_at
            FROM review_queue WHERE case_id = $1::uuid
            """, case_id)
        return dict(row) if row else None

    async def complete_review(self, case_id: str, *, reviewer_id: str, decision: str) -> None:
        await self.pool.execute(
            """
            UPDATE review_queue SET status='done', reviewer_id=$2, decision=$3, decided_at=now()
            WHERE case_id=$1::uuid
            """, case_id, reviewer_id, decision)


class PgDocumentStore:
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def search(self, embedding: list[float], doc_type: str, top_k: int) -> list[SearchHit]:
        rows = await self.pool.fetch(
            """
            SELECT source, doc_type, chunk_index, content, 1 - (embedding <=> $1::vector) AS score
            FROM document_chunks WHERE doc_type = $2
            ORDER BY embedding <=> $1::vector
            LIMIT $3
            """, _vec(embedding), doc_type, top_k)
        return [SearchHit(**dict(r)) for r in rows]

    async def existing_hashes(self, source: str) -> dict[int, str]:
        rows = await self.pool.fetch(
            "SELECT chunk_index, content_hash FROM document_chunks WHERE source=$1", source)
        return {r["chunk_index"]: r["content_hash"] for r in rows}

    async def replace_chunks(self, source: str, doc_type: str, chunks: list[ChunkIn]) -> int:
        async with self.pool.acquire() as con, con.transaction():
            await con.execute("DELETE FROM document_chunks WHERE source=$1", source)
            await con.executemany(
                """
                INSERT INTO document_chunks (source, chunk_index, doc_type, content, content_hash, embedding)
                VALUES ($1, $2, $3, $4, $5, $6::vector)
                """,
                [(source, c.index, doc_type, c.content, c.content_hash, _vec(c.embedding)) for c in chunks])
        return len(chunks)


class PostgresAuditLogger:
    """Append-only access log (who touched which member's data, when, for which request)."""

    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def record(self, *, actor_id: str, member_id: str | None, action: str, request_id: str | None) -> None:
        await self.pool.execute(
            "INSERT INTO audit_log (actor_id, member_id, action, request_id) VALUES ($1, $2, $3, $4)",
            actor_id, member_id, action, request_id)
