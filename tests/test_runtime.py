from fakes import ScriptedLLM

from careagents.agents.specialist import dependency_waves
from careagents.agents.supervisor import keyword_route

Q = "Is my knee MRI covered, do I need prior authorization, and how much will I pay?"


async def run(c, q=Q, member="SARAH-1001"):
    return await c.runtime.handle(q, member, "tester", "req-1")


def test_keyword_router_uses_word_boundaries():
    # The prototype sent these to the authorization agent because "pa" matched inside "pay".
    assert keyword_route("How much do I pay for a visit?") == ["billing"]
    assert "authorization" not in keyword_route("Tell me about spacing of appointments")
    assert "authorization" in keyword_route("Do I need a PA for this?")


def test_billing_waits_for_claims_and_authorization():
    assert dependency_waves(["billing", "policy", "claims", "authorization"]) == [
        ["policy", "claims", "authorization"], ["billing"]]
    assert dependency_waves(["billing"]) == [["billing"]]


async def test_end_to_end_answer_is_grounded(make_container):
    c = await make_container()
    r = await run(c)
    assert r.outcome == "answered"
    assert r.routed_to == ["policy", "claims", "authorization", "billing"]
    assert "$440.00" in r.answer and "$1000.00" in r.answer
    facts = {f.key: f.value for f in r.facts}
    assert facts["claim.CLM-8892.denial_code"] == "PA_MISSING"
    assert facts["prior_auth.73721.status"] == "none_on_file"
    assert {s.kind for s in r.sources} == {"record", "document"}
    assert r.usage.llm_calls > 0


async def test_billing_sees_claims_facts(make_container):
    llm = ScriptedLLM()
    c = await make_container(llm=llm)
    await run(c)
    billing_call = next(call for call in llm.calls if call["kind"] == "specialist"
                        and any(t["function"]["name"] == "estimate_cost" for t in call["tools"]))
    assert "PA_MISSING" in billing_call["messages"][1]["content"]


async def test_hallucinated_amount_triggers_recompose(make_container):
    llm = ScriptedLLM(compose=["You will pay $999.", "You will pay $440.00 if authorized."])
    c = await make_container(llm=llm)
    r = await run(c)
    assert r.outcome == "answered" and "$999" not in r.answer
    composer_calls = [x for x in llm.calls if x["kind"] == "composer"]
    assert len(composer_calls) == 2 and "$999" in composer_calls[1]["messages"][-1]["content"]


async def test_unverifiable_answer_degrades_to_records(make_container):
    llm = ScriptedLLM(compose=lambda _: "You owe $12,345.")
    c = await make_container(llm=llm)
    r = await run(c)
    assert r.outcome == "degraded" and "verification_failed" in r.flags
    assert "$12,345" not in r.answer and "PA_MISSING" in r.answer


async def test_llm_outage_degrades_honestly(make_container):
    c = await make_container(llm=ScriptedLLM(fail_all=True))
    r = await run(c)
    assert r.outcome == "degraded" and "llm_unavailable" in r.flags
    assert "can't write a full explanation" in r.answer
    # Deterministic tools still ran, so the member sees their real records.
    assert "no authorization on file" in r.answer and "$440.00" in r.answer and "$1000.00" in r.answer


async def test_failed_agent_is_retried_once_then_reported(make_container):
    llm = ScriptedLLM(fail_agents={"claims"})
    c = await make_container(llm=llm)
    r = await run(c)
    claims_run = next(a for a in r.agents if a.agent == "claims")
    assert claims_run.status == "failed" and claims_run.attempts == 2
    assert r.outcome == "answered"  # other agents' grounded facts still produce an answer


async def test_emergency_short_circuits_before_any_llm_call(make_container):
    llm = ScriptedLLM()
    c = await make_container(llm=llm)
    r = await run(c, "I have chest pain right now, is the ER covered?")
    assert r.outcome == "escalated" and r.escalation_reason == "emergency"
    assert "911" in r.answer and llm.calls == []


async def test_planner_escalation(make_container):
    llm = ScriptedLLM(plan={"agents": [], "escalate": True, "reason": "clinical_advice"})
    c = await make_container(llm=llm)
    r = await run(c, "Should I take ibuprofen or get surgery?")
    assert r.outcome == "escalated" and r.escalation_reason == "clinical_advice"


async def test_invalid_plan_falls_back_to_keywords(make_container):
    llm = ScriptedLLM(plan={"agents": ["hacker_agent"], "escalate": False})
    c = await make_container(llm=llm)
    r = await run(c, "What is the status of my claim?")
    assert r.routed_to == ["claims"]


async def test_case_run_persisted(make_container):
    c = await make_container()
    r = await run(c)
    assert c.repos.case_runs[-1]["outcome"] == r.outcome


async def test_deadline_returns_partial_answer(make_container):
    import asyncio

    class Slow(ScriptedLLM):
        async def chat(self, *a, **kw):
            await asyncio.sleep(0.3)
            return await super().chat(*a, **kw)

    c = await make_container(llm=Slow(), request_deadline_s=0.5)
    r = await run(c)
    assert r.outcome == "timeout"
