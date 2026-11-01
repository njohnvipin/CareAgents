"""Access control lives in tools, so test it there."""

from careagents.tools.base import ToolContext


def ctx_for(container, member_id):
    return ToolContext(member_id=member_id, actor_id="t", request_id="r", repos=container.repos,
                       search=container.search, audit=container.audit)


async def test_member_cannot_read_another_members_claim(make_container):
    c = await make_container()
    tools = c.runtime.agents["claims"].tools
    result = await tools["get_claim"].run(ctx_for(c, "JOHN-2002"), {"claim_id": "CLM-8892"})
    assert result.data == {"error": "not_found"}  # indistinguishable from a missing claim


async def test_model_cannot_inject_member_id(make_container):
    c = await make_container()
    tools = c.runtime.agents["claims"].tools
    result = await tools["list_claims"].run(ctx_for(c, "JOHN-2002"), {"member_id": "SARAH-1001"})
    assert result.data["error"] == "invalid_arguments"


async def test_tool_calls_are_audited(make_container):
    c = await make_container()
    tools = c.runtime.agents["claims"].tools
    await tools["list_claims"].run(ctx_for(c, "SARAH-1001"), {})
    assert c.audit.events[-1] == {"actor_id": "t", "member_id": "SARAH-1001", "action": "tool:list_claims",
                                  "request_id": "r"}


async def test_estimate_has_both_scenarios_when_auth_missing(make_container):
    c = await make_container()
    tools = c.runtime.agents["billing"].tools
    r = await tools["estimate_cost"].run(ctx_for(c, "SARAH-1001"), {"service_code": "73721"})
    facts = {f.key: f.value for f in r.facts}
    assert facts["estimate.73721.if_covered.member_pays"] == "440.00"
    assert facts["estimate.73721.if_auth_missing.member_may_owe_up_to"] == "1000.00"


async def test_estimate_single_scenario_when_auth_approved(make_container):
    c = await make_container()
    tools = c.runtime.agents["billing"].tools
    r = await tools["estimate_cost"].run(ctx_for(c, "JOHN-2002"), {"service_code": "73721"})
    keys = {f.key for f in r.facts}
    assert "estimate.73721.if_auth_missing.member_may_owe_up_to" not in keys


async def test_eligibility_formats_percent(make_container):
    c = await make_container()
    r = await c.runtime.agents["billing"].tools["get_eligibility"].run(ctx_for(c, "SARAH-1001"), {})
    assert r.data["coinsurance_member_pct"] == "20"
    assert r.data["deductible_remaining"] == "300.00"


async def test_policy_search_returns_relevant_chunk(make_container):
    c = await make_container()
    hits = await c.search.search("MRI imaging coverage prior authorization", "policy", 3)
    assert hits and any("MRI" in h.content for h in hits)


async def test_expired_authorization_counts_as_missing(make_container):
    from datetime import date

    c = await make_container()
    c.repos.auths[0].valid_until = date(2026, 1, 1)
    r = await c.runtime.agents["authorization"].tools["get_prior_auth_status"].run(
        ctx_for(c, "JOHN-2002"), {"service_code": "73721"})
    assert r.data["authorization_status"] == "expired"
    est = await c.runtime.agents["billing"].tools["estimate_cost"].run(ctx_for(c, "JOHN-2002"),
                                                                       {"service_code": "73721"})
    assert "if_auth_missing" in est.data["scenarios"]


def test_chunker_drops_heading_only_sections_and_prefixes_context():
    from careagents.ingestion.pipeline import chunk_markdown

    chunks = chunk_markdown("# Plan\n\n## MRI\nMRI is covered.\n\n## Empty\n\n## Deductible\n$500.")
    assert chunks == ["Plan > MRI\n## MRI\nMRI is covered.", "Plan > Deductible\n## Deductible\n$500."]
