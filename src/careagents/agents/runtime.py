"""Runs cases through the compiled LangGraph workflow (see agents/graph.py).

The runtime owns what the graph shouldn't: the request deadline, streaming progress events,
resuming paused cases, persistence of case records, metrics and logging.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Any, Literal, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import BaseModel

from careagents.agents.graph import (
    GraphDeps,
    GraphState,
    build_graph,
    checkpoint_serde,
    initial_state,
    materialize,
)
from careagents.agents.prompts import PROMPT_VERSION
from careagents.agents.specialist import SPECS, SpecialistAgent
from careagents.agents.state import AgentRun, CaseState, Fact, Usage
from careagents.agents.supervisor import Supervisor, fallback_answer
from careagents.agents.verifier import Verifier
from careagents.config import Settings
from careagents.data.models import AuditLogger, Repositories
from careagents.data.search import DocumentSearch
from careagents.llm.client import LLMClient
from careagents.logging_setup import log_event, phi_hash
from careagents.metrics import CASE_LATENCY, CASES
from careagents.tools.base import Tool

logger = logging.getLogger(__name__)

Outcome = Literal["answered", "escalated", "degraded", "timeout", "pending_review"]

STEP_LABELS = {
    "plan": "Understanding your question",
    "policy": "Checking your plan documents",
    "claims": "Looking up your claims",
    "authorization": "Checking prior authorization",
    "billing": "Estimating your costs",
    "check": "Reviewing what we found",
    "compose": "Writing your answer",
    "verify": "Double-checking the answer against your records",
    "request_review": "Sending your question to a specialist",
    "escalate": "Routing your question",
}


class Source(BaseModel):
    source: str
    kind: str


class CaseResult(BaseModel):
    case_id: str
    outcome: Outcome
    answer: str
    routed_to: list[str]
    escalation_reason: str | None = None
    review_reason: str | None = None
    flags: list[str] = []
    facts: list[Fact] = []
    sources: list[Source] = []
    agents: list[AgentRun] = []
    usage: Usage = Usage()
    latency_ms: int = 0
    prompt_version: str = PROMPT_VERSION


class ReviewNotPendingError(Exception):
    pass


class AgentRuntime:
    def __init__(self, settings: Settings, llm: LLMClient, tools: dict[str, Tool], repos: Repositories,
                 search: DocumentSearch, audit: AuditLogger, checkpointer: BaseCheckpointSaver | None = None):
        self.s = settings
        self.repos = repos
        self.agents = {name: SpecialistAgent(spec, llm, tools, settings.max_tool_rounds)
                       for name, spec in SPECS.items()}
        deps = GraphDeps(settings=settings, supervisor=Supervisor(llm),
                         verifier=Verifier(llm, settings.llm_verifier_enabled), agents=self.agents,
                         repos=repos, search=search, audit=audit)
        self.graph = build_graph(deps).compile(checkpointer=checkpointer or InMemorySaver(serde=checkpoint_serde()))

    @staticmethod
    def _config(case_id: str) -> RunnableConfig:
        return {"configurable": {"thread_id": case_id}, "recursion_limit": 60}

    # ------------------------------------------------------------------ public API
    async def handle(self, question: str, member_id: str, actor_id: str, request_id: str | None) -> CaseResult:
        case_id = str(uuid.uuid4())
        return await self._run(case_id, initial_state(case_id, member_id, actor_id, question, request_id))

    async def stream(self, question: str, member_id: str, actor_id: str,
                     request_id: str | None) -> AsyncIterator[dict[str, Any]]:
        """Yields {"event": "step", ...} progress events, then {"event": "result", "data": CaseResult}.

        The graph runs in its own task so the request deadline is enforced inside graph code,
        never across a yield to the HTTP layer.
        """
        case_id = str(uuid.uuid4())
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        task = asyncio.create_task(self._run(
            case_id, initial_state(case_id, member_id, actor_id, question, request_id), on_step=queue.put_nowait))
        try:
            yield {"event": "started", "case_id": case_id}
            while not task.done() or not queue.empty():
                getter = asyncio.ensure_future(queue.get())
                await asyncio.wait({getter, task}, return_when=asyncio.FIRST_COMPLETED)
                if getter.done():
                    yield getter.result()
                else:
                    getter.cancel()
            yield {"event": "result", "data": task.result().model_dump(mode="json")}
        finally:
            if not task.done():  # client disconnected: stop spending tokens on this case
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    async def resume(self, case_id: str, decision: dict[str, Any]) -> CaseResult:
        snapshot = await self.graph.aget_state(self._config(case_id))
        if "human_review" not in (snapshot.next or ()):
            raise ReviewNotPendingError(case_id)
        return await self._run(case_id, Command(resume=decision))

    async def get_case(self, case_id: str) -> CaseResult | None:
        snapshot = await self.graph.aget_state(self._config(case_id))
        if not snapshot.values:
            return None
        return self._result(snapshot.values)

    async def case_owner(self, case_id: str) -> str | None:
        snapshot = await self.graph.aget_state(self._config(case_id))
        return snapshot.values.get("member_id") if snapshot.values else None

    def mermaid(self) -> str:
        return self.graph.get_graph().draw_mermaid()

    # ------------------------------------------------------------------ internals
    async def _run(self, case_id: str, graph_input: Any,
                   on_step: Callable[[dict[str, Any]], None] | None = None) -> CaseResult:
        start = time.perf_counter()
        config = self._config(case_id)
        try:
            async with asyncio.timeout(self.s.request_deadline_s):
                async for chunk in self.graph.astream(graph_input, config, stream_mode="updates"):
                    if on_step:
                        for node in chunk:
                            if node in STEP_LABELS:
                                on_step({"event": "step", "node": node, "label": STEP_LABELS[node]})
            values = (await self.graph.aget_state(config)).values
            result = self._result(values)
        except TimeoutError:
            # The checkpointer holds every completed superstep, so partial findings survive.
            values = (await self.graph.aget_state(config)).values
            result = self._result(values, timed_out=True)

        result.latency_ms = int((time.perf_counter() - start) * 1000)
        CASES.labels(result.outcome).inc()
        CASE_LATENCY.observe(result.latency_ms / 1000)
        log_event(logger, "case_completed", case_id=case_id, member=phi_hash(values.get("member_id")),
                  outcome=result.outcome, routed_to=result.routed_to, flags=result.flags,
                  latency_ms=result.latency_ms, llm_calls=result.usage.llm_calls)
        await self._persist(values, result)
        return result

    def _result(self, values: dict[str, Any], timed_out: bool = False) -> CaseResult:
        case: CaseState = materialize(cast(GraphState, values))
        outcome = values.get("outcome")
        answer = values.get("answer", "")
        if timed_out or not outcome:
            outcome, answer = "timeout", fallback_answer(case, "timeout")
        sources: dict[str, Source] = {}
        for f in case.facts.values():
            sources.setdefault(f.source, Source(source=f.source, kind="record"))
        for e in case.evidence:
            sources.setdefault(e.source, Source(source=e.source, kind="document"))
        flags = list(values.get("flags", []))
        if case.conflicts and "data_conflict" not in flags:
            flags.append("data_conflict")
        return CaseResult(case_id=case.case_id, outcome=cast(Outcome, outcome), answer=answer, routed_to=case.plan,
                          escalation_reason=values.get("escalation_reason"),
                          review_reason=values.get("review_reason"), flags=flags,
                          facts=list(case.facts.values()), sources=list(sources.values()),
                          agents=list(case.runs.values()), usage=case.usage)

    async def _persist(self, values: dict[str, Any], result: CaseResult) -> None:
        try:
            await self.repos.save_case_run({
                "case_id": uuid.UUID(result.case_id), "member_id": values["member_id"],
                "actor_id": values["actor_id"], "question": values["question"], "outcome": result.outcome,
                "routed_to": result.routed_to, "facts": [f.model_dump() for f in result.facts],
                "answer": result.answer, "latency_ms": result.latency_ms,
                "prompt_tokens": result.usage.prompt_tokens, "completion_tokens": result.usage.completion_tokens,
                "llm_calls": result.usage.llm_calls,
            })
        except Exception:  # persistence must never fail the member's request
            logger.exception("case_persist_failed", extra={"fields": {"case_id": result.case_id}})
