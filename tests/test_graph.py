"""LangGraph-specific behaviour: structure, human-in-the-loop interrupts, streaming, resumability."""

import pytest
from fakes import ScriptedLLM

from careagents.agents.graph import initial_state, materialize
from careagents.agents.runtime import AgentRuntime, ReviewNotPendingError
from careagents.agents.state import Fact

Q = "Is my knee MRI covered, do I need prior authorization, and how much will I pay?"


async def run(c, q=Q, member="SARAH-1001"):
    return await c.runtime.handle(q, member, "tester", "req-1")


async def test_graph_has_expected_nodes(make_container):
    c = await make_container()
    nodes = set(c.runtime.graph.get_graph().nodes)
    assert {"plan", "dispatch", "policy", "claims", "authorization", "billing", "check", "compose", "verify",
            "finalize", "escalate", "request_review", "human_review", "apply_review"} <= nodes
    assert "human_review" in c.runtime.mermaid()


def test_conflicts_between_parallel_branches_are_detected():
    s = initial_state("c1", "m", "a", "q", None)
    s["fact_log"] = [Fact(key="claim.X.status", value="denied", source="claims"),
                     Fact(key="claim.X.status", value="approved", source="legacy")]
    case = materialize(s)
    assert case.conflicts and case.facts["claim.X.status"].value == "approved"


async def test_verification_failure_pauses_for_review_then_edit(make_container):
    llm = ScriptedLLM(compose=lambda _: "You owe $12,345.")
    c = await make_container(llm=llm, human_review_enabled=True)
    r = await run(c)
    assert r.outcome == "pending_review" and r.review_reason == "verification_failed"
    assert "specialist is reviewing" in r.answer and "$440.00" in r.answer  # verified records meanwhile
    assert "$12,345" not in r.answer
    review = await c.repos.get_review(r.case_id)
    assert review["status"] == "pending" and review["draft"] == "You owe $12,345."
    assert (await c.runtime.get_case(r.case_id)).outcome == "pending_review"

    done = await c.runtime.resume(r.case_id, {"action": "edit", "answer": "Your MRI would cost about $440.",
                                              "reviewer_id": "staff-7"})
    assert done.outcome == "answered" and done.answer == "Your MRI would cost about $440."
    assert "human_reviewed" in done.flags
    assert (await c.repos.get_review(r.case_id))["decision"] == "edit"
    assert {"actor_id": "staff-7", "member_id": "SARAH-1001", "action": "review_edit",
            "request_id": "req-1"} in c.audit.events
    with pytest.raises(ReviewNotPendingError):
        await c.runtime.resume(r.case_id, {"action": "approve"})


async def test_reject_returns_records_only(make_container):
    c = await make_container(llm=ScriptedLLM(compose=lambda _: "You owe $12,345."), human_review_enabled=True)
    r = await run(c)
    done = await c.runtime.resume(r.case_id, {"action": "reject", "reviewer_id": "staff-1"})
    assert done.outcome == "degraded" and "$12,345" not in done.answer and "$440.00" in done.answer


async def test_complaint_is_queued_for_staff(make_container):
    llm = ScriptedLLM(plan={"agents": [], "escalate": True, "reason": "complaint_or_legal"})
    c = await make_container(llm=llm, human_review_enabled=True)
    r = await run(c, "I want to file a formal grievance about my denied claim.")
    assert r.outcome == "pending_review" and r.escalation_reason == "complaint_or_legal"
    assert "representative" in r.answer
    done = await c.runtime.resume(r.case_id, {"action": "edit", "answer": "We've opened grievance G-1.",
                                              "reviewer_id": "staff-2"})
    assert done.answer == "We've opened grievance G-1."


async def test_review_disabled_keeps_old_behaviour(make_container):
    c = await make_container(llm=ScriptedLLM(compose=lambda _: "You owe $12,345."), human_review_enabled=False)
    r = await run(c)
    assert r.outcome == "degraded" and "verification_failed" in r.flags


async def test_resume_on_a_different_runtime_instance(make_container):
    """Simulates the reviewer's request landing on another pod: only the checkpointer is shared."""
    from careagents.tools.catalog import build_tools

    c = await make_container(llm=ScriptedLLM(compose=lambda _: "You owe $12,345."), human_review_enabled=True)
    r = await run(c)
    other_pod = AgentRuntime(c.settings, ScriptedLLM(), build_tools(), c.repos, c.search, c.audit,
                             checkpointer=c.runtime.graph.checkpointer)
    done = await other_pod.resume(r.case_id, {"action": "approve", "reviewer_id": "s"})
    assert done.outcome == "answered" and done.answer == "You owe $12,345."  # reviewer approved the draft
    assert done.routed_to == ["policy", "claims", "authorization", "billing"]  # full state came back


async def test_stream_emits_steps_in_dependency_order(make_container):
    c = await make_container()
    events = [e async for e in c.runtime.stream(Q, "SARAH-1001", "t", None)]
    assert events[0]["event"] == "started" and events[-1]["event"] == "result"
    steps = [e["node"] for e in events if e["event"] == "step"]
    assert steps[0] == "plan"
    assert steps.index("billing") > max(steps.index("claims"), steps.index("authorization"))
    assert steps.index("compose") > steps.index("billing") and steps[-1] == "verify"
    assert events[-1]["data"]["outcome"] == "answered"
    assert events[-1]["data"]["case_id"] == events[0]["case_id"]


async def test_stream_respects_deadline(make_container):
    import asyncio

    class Slow(ScriptedLLM):
        async def chat(self, *a, **kw):
            await asyncio.sleep(0.3)
            return await super().chat(*a, **kw)

    c = await make_container(llm=Slow(), request_deadline_s=0.5)
    events = [e async for e in c.runtime.stream(Q, "SARAH-1001", "t", None)]
    assert events[-1]["data"]["outcome"] == "timeout"


def test_checkpoint_serde_allowlist_round_trips_state_types_and_blocks_others():
    from careagents.agents.graph import checkpoint_serde
    from careagents.agents.state import AgentRun, Evidence, Usage

    serde = checkpoint_serde()
    for obj in (Fact(key="k", value="v", source="s"), Usage(llm_calls=2),
                AgentRun(agent="claims"), Evidence(source="p", doc_type="policy", chunk_index=0, text="t")):
        assert serde.loads_typed(serde.dumps_typed(obj)) == obj

    from careagents.agents.verifier import VerificationReport  # a dataclass that is not on the allowlist
    restored = serde.loads_typed(serde.dumps_typed(VerificationReport(ok=True)))
    assert not isinstance(restored, VerificationReport)
