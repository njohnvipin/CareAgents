"""The case workflow as a LangGraph StateGraph.

    START -> plan -+-> escalate -----------------------------+--> request_review -> human_review -> apply_review -> END
                   |                                         '--> END
                   '-> dispatch <-> {policy, claims, authorization, billing}   (parallel waves via Send)
                          '-> check -+-> dispatch (one re-run round for ungrounded agents)
                                     +-> finalize (models unreachable)                  -> END
                                     '-> compose -> verify -+-> END (verified)
                                             ^              +-> compose (fix issues)
                                             '--------------+-> request_review (failed / conflicting data)
                                                            '-> finalize (review disabled)

Design notes
- Specialists run in parallel, so they never mutate shared state. Each node rebuilds a CaseState
  from the graph state, does its work, and returns only the *delta*. Reducers merge the deltas.
- Facts are kept as an append-only log; CaseState replays it, so conflicts between parallel
  agents are still detected.
- Side effects that must happen once (queueing a review) live in their own node *before* the
  interrupt, because a node that calls interrupt() re-executes from the top when resumed.
- Every superstep is checkpointed, so a case can pause for days and resume on another pod.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send, interrupt

from careagents.agents.prompts import ESCALATION_MESSAGES
from careagents.agents.specialist import AGENT_ORDER, SPECS, SpecialistAgent
from careagents.agents.state import AgentRun, CaseState, Evidence, Fact, Usage
from careagents.agents.supervisor import Supervisor, fallback_answer
from careagents.agents.verifier import Verifier
from careagents.config import Settings
from careagents.data.models import AuditLogger, Repositories
from careagents.data.search import DocumentSearch
from careagents.llm.client import LLMError
from careagents.tools.base import ToolContext

REVIEW_ESCALATIONS = {"complaint_or_legal"}

# Types stored in checkpoints. LangGraph is moving to block unregistered types on load; without
# this allowlist an upgrade would make every case paused for review impossible to resume.
CHECKPOINT_TYPES = [("careagents.agents.state", name) for name in ("Fact", "Evidence", "AgentRun", "Usage")]


def checkpoint_serde() -> JsonPlusSerializer:
    return JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_TYPES)


# ----------------------------------------------------------------------------- state + reducers
def merge_runs(left: dict[str, AgentRun], right: dict[str, AgentRun]) -> dict[str, AgentRun]:
    return {**left, **right}


def add_usage(left: Usage, right: Usage) -> Usage:
    return Usage(llm_calls=left.llm_calls + right.llm_calls, prompt_tokens=left.prompt_tokens + right.prompt_tokens,
                 completion_tokens=left.completion_tokens + right.completion_tokens)


class GraphState(TypedDict, total=False):
    # identity (set once)
    case_id: str
    member_id: str
    actor_id: str
    question: str
    request_id: str | None
    # accumulated by parallel nodes (reducers)
    fact_log: Annotated[list[Fact], operator.add]
    evidence: Annotated[list[Evidence], operator.add]
    runs: Annotated[dict[str, AgentRun], merge_runs]
    usage: Annotated[Usage, add_usage]
    completed: Annotated[list[str], operator.add]  # "round:agent"
    # control
    plan: list[str]
    round: int
    rerun: list[str]
    route: str
    issues: list[str]
    compose_attempts: int
    draft: str
    escalation_reason: str | None
    fallback_reason: str | None
    review_reason: str | None
    review_decision: dict[str, Any] | None
    # result
    outcome: str
    answer: str
    flags: list[str]


def initial_state(case_id: str, member_id: str, actor_id: str, question: str,
                  request_id: str | None) -> GraphState:
    return GraphState(case_id=case_id, member_id=member_id, actor_id=actor_id, question=question,
                      request_id=request_id, fact_log=[], evidence=[], runs={}, usage=Usage(), completed=[],
                      plan=[], round=1, rerun=[], issues=[], compose_attempts=0, flags=[])


def materialize(s: GraphState) -> CaseState:
    """Rebuild the rich CaseState (prompts, verifier, fallback all use it) from graph state."""
    case = CaseState(case_id=s["case_id"], member_id=s["member_id"], actor_id=s["actor_id"],
                     question=s["question"], request_id=s.get("request_id"), plan=list(s.get("plan", [])),
                     issues=list(s.get("issues", [])))
    for fact in s.get("fact_log", []):
        case.add_fact(fact.model_copy())
    for ev in s.get("evidence", []):
        case.add_evidence(ev)
    case.runs = {k: v.model_copy() for k, v in s.get("runs", {}).items()}
    case.usage = s.get("usage", Usage()).model_copy()
    return case


@dataclass
class _Snapshot:
    facts: int
    evidence: int
    usage: Usage


def _snap(case: CaseState) -> _Snapshot:
    return _Snapshot(len(case.fact_log), len(case.evidence), case.usage.model_copy())


def _delta(case: CaseState, snap: _Snapshot, agents: list[str] | None = None) -> dict[str, Any]:
    u = case.usage
    update: dict[str, Any] = {
        "fact_log": case.fact_log[snap.facts:],
        "evidence": case.evidence[snap.evidence:],
        "usage": Usage(llm_calls=u.llm_calls - snap.usage.llm_calls,
                       prompt_tokens=u.prompt_tokens - snap.usage.prompt_tokens,
                       completion_tokens=u.completion_tokens - snap.usage.completion_tokens),
    }
    if agents:
        update["runs"] = {a: case.runs[a] for a in agents if a in case.runs}
    return update


# ----------------------------------------------------------------------------- graph
@dataclass
class GraphDeps:
    settings: Settings
    supervisor: Supervisor
    verifier: Verifier
    agents: dict[str, SpecialistAgent]
    repos: Repositories
    search: DocumentSearch
    audit: AuditLogger

    def ctx(self, s: GraphState) -> ToolContext:
        return ToolContext(member_id=s["member_id"], actor_id=s["actor_id"], request_id=s.get("request_id"),
                           repos=self.repos, search=self.search, audit=self.audit)


def build_graph(d: GraphDeps) -> StateGraph:
    s_ = d.settings

    async def plan(s: GraphState) -> dict[str, Any]:
        case = materialize(s)
        snap = _snap(case)
        p = await d.supervisor.plan(case)
        return {**_delta(case, snap), "plan": p.agents, "round": 1,
                "escalation_reason": p.reason if p.escalate else None}

    def after_plan(s: GraphState) -> str:
        return "escalate" if s.get("escalation_reason") else "dispatch"

    async def escalate(s: GraphState) -> dict[str, Any]:
        reason = s.get("escalation_reason") or "out_of_scope"
        await d.audit.record(actor_id=s["actor_id"], member_id=s["member_id"], action=f"escalation:{reason}",
                             request_id=s.get("request_id"))
        return {"outcome": "escalated", "answer": ESCALATION_MESSAGES[reason]}

    def after_escalate(s: GraphState) -> str:
        if s_.human_review_enabled and s.get("escalation_reason") in REVIEW_ESCALATIONS:
            return "request_review"
        return END

    async def dispatch(s: GraphState) -> dict[str, Any]:
        return {}

    def route_dispatch(s: GraphState) -> list[Send] | str:
        rnd = s.get("round", 1)
        targets = s.get("plan", []) if rnd == 1 else s.get("rerun", [])
        done = {c.split(":", 1)[1] for c in s.get("completed", []) if c.startswith(f"{rnd}:")}
        ready = [a for a in AGENT_ORDER if a in targets and a not in done
                 and all(dep in done or dep not in targets for dep in SPECS[a].depends_on)]
        return [Send(a, s) for a in ready] if ready else "check"

    def make_specialist(name: str) -> Callable[[GraphState], Any]:
        async def node(s: GraphState) -> dict[str, Any]:
            case = materialize(s)
            snap = _snap(case)
            hint = "Your previous attempt produced no verified data. Use your tools." if s.get("round", 1) > 1 else ""
            await d.agents[name].run(case, d.ctx(s), hint=hint)
            return {**_delta(case, snap, [name]), "completed": [f"{s.get('round', 1)}:{name}"]}
        node.__name__ = name
        return node

    async def check(s: GraphState) -> dict[str, Any]:
        case = materialize(s)
        if s.get("round", 1) == 1:
            structural = d.verifier.structural_check(case)
            if structural.rerun_agents:
                return {"rerun": structural.rerun_agents, "round": 2, "route": "dispatch"}
        snap = _snap(case)
        llm_down = [a for a in case.plan if a in case.runs and case.runs[a].status == "failed"]
        for name in llm_down:  # models unreachable: run each agent's core tools directly
            await d.agents[name].run_without_llm(case, d.ctx(s))
        update = _delta(case, snap, llm_down)
        if llm_down and len(llm_down) == len(case.plan):
            return {**update, "route": "finalize", "fallback_reason": "llm_unavailable"}
        if not case.facts and not case.evidence:
            return {**update, "route": "finalize", "fallback_reason": "llm_unavailable"}
        return {**update, "route": "compose"}

    async def compose(s: GraphState) -> dict[str, Any]:
        case = materialize(s)
        snap = _snap(case)
        try:
            draft = await d.supervisor.compose(case)
        except LLMError:
            return {**_delta(case, snap), "route": "finalize", "fallback_reason": "llm_unavailable"}
        return {**_delta(case, snap), "draft": draft, "compose_attempts": s.get("compose_attempts", 0) + 1,
                "route": "verify"}

    async def verify(s: GraphState) -> dict[str, Any]:
        case = materialize(s)
        snap = _snap(case)
        report = await d.verifier.check_draft(case, s["draft"])
        update = _delta(case, snap)
        if report.ok:
            if case.conflicts and s_.human_review_enabled:
                return {**update, "flags": report.flags, "route": "request_review", "review_reason": "data_conflict"}
            return {**update, "outcome": "answered", "answer": s["draft"], "flags": report.flags, "route": END}
        update["issues"] = report.issues
        if s.get("compose_attempts", 0) <= s_.max_verify_iterations:
            return {**update, "route": "compose"}
        if s_.human_review_enabled:
            return {**update, "route": "request_review", "review_reason": "verification_failed"}
        return {**update, "route": "finalize", "fallback_reason": "verification_failed"}

    async def finalize(s: GraphState) -> dict[str, Any]:
        reason = s.get("fallback_reason") or "llm_unavailable"
        return {"outcome": "degraded", "answer": fallback_answer(materialize(s), reason),
                "flags": [*s.get("flags", []), reason]}

    async def request_review(s: GraphState) -> dict[str, Any]:
        reason = s.get("review_reason") or s.get("escalation_reason") or "manual"
        case = materialize(s)
        interim = s.get("answer") if s.get("outcome") == "escalated" else fallback_answer(case, "under_review")
        await d.repos.enqueue_review({"case_id": s["case_id"], "member_id": s["member_id"], "reason": reason,
                                      "question": s["question"], "draft": s.get("draft")})
        await d.audit.record(actor_id=s["actor_id"], member_id=s["member_id"], action=f"review_requested:{reason}",
                             request_id=s.get("request_id"))
        return {"outcome": "pending_review", "answer": interim, "review_reason": reason}

    async def human_review(s: GraphState) -> dict[str, Any]:
        # Pauses the graph. Resumed by POST /v1/staff/reviews/{case_id} with Command(resume=decision).
        decision = interrupt({"case_id": s["case_id"], "reason": s.get("review_reason"), "draft": s.get("draft")})
        return {"review_decision": decision}

    async def apply_review(s: GraphState) -> dict[str, Any]:
        decision = s.get("review_decision") or {}
        action = decision.get("action")
        flags = [*s.get("flags", []), "human_reviewed"]
        if action == "approve" and s.get("draft"):
            outcome, answer = "answered", s["draft"]
        elif action in ("approve", "edit") and decision.get("answer"):
            outcome, answer = "answered", decision["answer"]
        else:  # reject: member gets their verified records and a pointer to member services
            outcome, answer = "degraded", fallback_answer(materialize(s), "verification_failed")
        reviewer = str(decision.get("reviewer_id", "unknown"))
        await d.repos.complete_review(s["case_id"], reviewer_id=reviewer, decision=action or "reject")
        await d.audit.record(actor_id=reviewer, member_id=s["member_id"], action=f"review_{action or 'reject'}",
                             request_id=s.get("request_id"))
        return {"outcome": outcome, "answer": answer, "flags": flags}

    g = StateGraph(GraphState)
    # The ignores below: LangGraph's node protocol types the parameter by the literal name "state".
    for name, fn in [("plan", plan), ("escalate", escalate), ("dispatch", dispatch), ("check", check),
                     ("compose", compose), ("verify", verify), ("finalize", finalize),
                     ("request_review", request_review), ("human_review", human_review),
                     ("apply_review", apply_review)]:
        g.add_node(name, fn)  # type: ignore[call-overload]
    for name in AGENT_ORDER:
        g.add_node(name, make_specialist(name))  # type: ignore[call-overload]
        g.add_edge(name, "dispatch")

    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", after_plan, ["escalate", "dispatch"])
    g.add_conditional_edges("escalate", after_escalate, ["request_review", END])
    g.add_conditional_edges("dispatch", route_dispatch, [*AGENT_ORDER, "check"])
    g.add_conditional_edges("check", lambda s: s["route"], ["dispatch", "compose", "finalize"])
    g.add_conditional_edges("compose", lambda s: s["route"], ["verify", "finalize"])
    g.add_conditional_edges("verify", lambda s: s["route"], ["compose", "request_review", "finalize", END])
    g.add_edge("finalize", END)
    g.add_edge("request_review", "human_review")
    g.add_edge("human_review", "apply_review")
    g.add_edge("apply_review", END)
    return g
