"""Integration tests against real Postgres+pgvector and Redis.

Run with: CA_TEST_DATABASE_URL=postgresql://... CA_TEST_REDIS_URL=redis://... pytest -m integration
(docker compose up postgres redis && python -m careagents.migrate && python -m careagents.seed)
"""

import os

import pytest
from fakes import ScriptedLLM

from careagents.container import build_container

DB = os.getenv("CA_TEST_DATABASE_URL")
REDIS = os.getenv("CA_TEST_REDIS_URL")
pytestmark = [pytest.mark.integration,
              pytest.mark.skipif(not (DB and REDIS), reason="needs CA_TEST_DATABASE_URL and CA_TEST_REDIS_URL")]


@pytest.fixture
async def pg(settings):
    s = settings.model_copy(update={"data_backend": "postgres", "database_url": DB, "redis_url": REDIS})
    c = await build_container(s, llm=ScriptedLLM(), with_queue=False)
    await c.redis.flushdb()
    yield c
    await c.aclose()


async def test_record_queries(pg):
    r = pg.repos
    assert (await r.get_member("SARAH-1001")).plan_id == "HSG-2026"
    assert [c.id for c in await r.list_claims("SARAH-1001")] == ["CLM-8892", "CLM-7781"]
    assert [c.id for c in await r.list_claims("SARAH-1001", status="denied")] == ["CLM-8892"]
    assert await r.get_claim("JOHN-2002", "CLM-8892") is None
    assert (await r.find_services("knee mri"))[0].code == "73721"
    assert (await r.find_services("73721"))[0].code == "73721"


async def test_ingestion_is_idempotent_and_searchable(pg):
    from careagents.ingestion.pipeline import ingest_directory

    s = pg.settings
    await ingest_directory(s.documents_dir, pg.store, pg.embedder, pg.search)
    second = await ingest_directory(s.documents_dir, pg.store, pg.embedder, pg.search)
    assert second["chunks_written"] == 0 and second["skipped_unchanged"] == 2
    count = await pg.repos.pool.fetchval("SELECT count(*) FROM document_chunks")
    assert count == sum(len(v) for v in [await pg.store.existing_hashes("policy/healthsure_gold_2026.md"),
                                        await pg.store.existing_hashes("authorization/prior_auth_guidelines.md")])
    hits = await pg.search.search("MRI prior authorization imaging", "authorization", 2)
    assert hits and "MRI" in hits[0].content


async def test_search_cache_and_invalidation(pg):
    first = await pg.search.search("deductible amount", "policy", 2)
    keys = [k async for k in pg.redis.scan_iter("ca:search:*")]
    assert len(keys) == 1
    await pg.search.invalidate()
    assert await pg.search.search("deductible amount", "policy", 2) == first
    assert len([k async for k in pg.redis.scan_iter("ca:search:*")]) == 2  # new version, new key


async def test_end_to_end_persists_run_and_audit(pg):
    r = await pg.runtime.handle("How much will my knee MRI cost?", "SARAH-1001", "it-user", "it-req")
    assert r.outcome == "answered" and "$440.00" in r.answer
    row = await pg.repos.pool.fetchrow("SELECT outcome, routed_to, llm_calls FROM case_runs WHERE id=$1::uuid",
                                       r.case_id)
    assert row["outcome"] == "answered" and row["llm_calls"] > 0
    audits = await pg.repos.pool.fetch("SELECT action FROM audit_log WHERE request_id='it-req'")
    assert {"tool:estimate_cost", "tool:list_claims"} <= {a["action"] for a in audits}


async def test_redis_rate_limiter_shared_window(pg):
    results = [(await pg.limiter.hit("it-user", 2, 60))[0] for _ in range(3)]
    assert results == [True, True, False]


async def test_readiness_with_real_dependencies(pg):
    from fastapi.testclient import TestClient

    from careagents.api.app import create_app

    # TestClient runs its own event loop, so the app must build its own pools there.
    with TestClient(create_app(pg.settings)) as client:
        body = client.get("/readyz").json()
    assert body == {"status": "ok", "checks": {"database": True, "redis": True}}


async def test_paused_case_survives_full_restart(settings):
    """Pause in one process, tear everything down, resume from a fresh container (new pools)."""
    s = settings.model_copy(update={"data_backend": "postgres", "database_url": DB, "redis_url": REDIS,
                                    "human_review_enabled": True})
    first = await build_container(s, llm=ScriptedLLM(compose=lambda _: "You owe $12,345."), with_queue=False)
    paused = await first.runtime.handle("How much will my knee MRI cost?", "SARAH-1001", "it", "it-hitl")
    assert paused.outcome == "pending_review"
    await first.aclose()

    second = await build_container(s, llm=ScriptedLLM(), with_queue=False)
    try:
        queued = await second.repos.get_review(paused.case_id)
        assert queued["status"] == "pending" and queued["reason"] == "verification_failed"
        before = await second.runtime.get_case(paused.case_id)
        assert before.outcome == "pending_review" and "PA_MISSING" in {f.value for f in before.facts}
        done = await second.runtime.resume(paused.case_id, {"action": "edit", "answer": "About $440.",
                                                            "reviewer_id": "staff-1"})
        assert done.outcome == "answered" and done.answer == "About $440."
        assert (await second.repos.get_review(paused.case_id))["status"] == "done"
        row = await second.repos.pool.fetchrow("SELECT outcome, answer FROM case_runs WHERE id=$1::uuid",
                                               paused.case_id)
        assert dict(row) == {"outcome": "answered", "answer": "About $440."}  # upserted, not duplicated
    finally:
        await second.aclose()
