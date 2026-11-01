"""Specialist agents: an LLM tool-calling loop over a scoped set of tools."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from careagents.agents.llm_call import call_llm
from careagents.agents.prompts import SPECIALIST_SYSTEM, SPECIALIST_USER
from careagents.agents.state import AgentRun, CaseState
from careagents.llm.client import LLMClient, LLMError
from careagents.logging_setup import log_event
from careagents.metrics import AGENT_RUNS
from careagents.tools.base import Tool, ToolContext, ToolResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentSpec:
    name: str
    scope: str
    tools: tuple[str, ...]
    depends_on: tuple[str, ...] = ()


SPECS: dict[str, AgentSpec] = {
    "policy": AgentSpec("policy", "coverage rules, benefits, exclusions and cost-sharing rules in plan documents.",
                        ("lookup_service", "search_policy")),
    "claims": AgentSpec("claims", "the member's claims: status, denial reasons, amounts paid, appeal options.",
                        ("lookup_service", "list_claims", "get_claim")),
    "authorization": AgentSpec("authorization",
                               "prior authorization requirements, medical-necessity rules and the member's "
                               "authorization status.",
                               ("lookup_service", "get_prior_auth_status", "search_auth_guidelines")),
    # Billing runs after claims/authorization so it can see their facts (e.g. a PA_MISSING denial).
    "billing": AgentSpec("billing", "what the member will pay: deductible, coinsurance, out-of-pocket, estimates.",
                         ("lookup_service", "get_eligibility", "estimate_cost"),
                         depends_on=("claims", "authorization")),
}
AGENT_ORDER = tuple(SPECS)


def dependency_waves(agents: list[str]) -> list[list[str]]:
    """Group agents into waves; agents in a wave run concurrently."""
    remaining = [a for a in AGENT_ORDER if a in agents]
    done: set[str] = set()
    waves = []
    while remaining:
        ready = [a for a in remaining if all(d in done or d not in agents for d in SPECS[a].depends_on)]
        if not ready:  # defensive: a cycle would otherwise loop forever
            ready = remaining[:]
        waves.append(ready)
        done.update(ready)
        remaining = [a for a in remaining if a not in ready]
    return waves


class SpecialistAgent:
    def __init__(self, spec: AgentSpec, llm: LLMClient, tools: dict[str, Tool], max_tool_rounds: int):
        self.spec = spec
        self.llm = llm
        self.tools = {n: tools[n] for n in spec.tools}
        self.max_tool_rounds = max_tool_rounds

    async def run(self, state: CaseState, ctx: ToolContext, hint: str = "") -> AgentRun:
        run = state.runs.get(self.spec.name) or AgentRun(agent=self.spec.name)
        state.runs[self.spec.name] = run
        run.attempts += 1
        messages: list[dict] = [
            {"role": "system", "content": SPECIALIST_SYSTEM.format(name=self.spec.name, scope=self.spec.scope)},
            {"role": "user", "content": SPECIALIST_USER.format(question=state.question, facts=state.facts_block(),
                                                               hint=f"\nNote: {hint}" if hint else "")},
        ]
        specs = [t.spec() for t in self.tools.values()]
        try:
            for _ in range(self.max_tool_rounds):
                resp = await call_llm(self.llm, state, messages=messages, tools=specs, tier="fast")
                if not resp.tool_calls:
                    run.note = resp.content.strip()[:2000]
                    break
                messages.append(resp.assistant_message())
                results = await asyncio.gather(*(self._execute(tc.name, tc.arguments, ctx) for tc in resp.tool_calls))
                for tc, result in zip(resp.tool_calls, results, strict=True):
                    self._absorb(state, run, result)
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result.to_message()})
            else:
                run.note = run.note or "Stopped after reaching the tool-call limit."
            run.status, run.error = "ok", None
        except LLMError as exc:
            run.status, run.error = "failed", "llm_unavailable"
            log_event(logger, "agent_failed", logging.WARNING, agent=self.spec.name, error=str(exc)[:200])
        AGENT_RUNS.labels(self.spec.name, run.status).inc()
        return run

    async def run_without_llm(self, state: CaseState, ctx: ToolContext) -> AgentRun:
        """Degraded mode: call this agent's core tools directly when no model is reachable.

        Tools are deterministic and the database is usually healthy during an LLM outage, so the
        member still gets their real records instead of nothing.
        """
        run = state.runs.get(self.spec.name) or AgentRun(agent=self.spec.name)
        state.runs[self.spec.name] = run
        plan: list[tuple[str, dict]] = []
        codes: list[str] = []
        if "lookup_service" in self.tools and self.spec.name in ("authorization", "billing"):
            found = await self._execute("lookup_service", {"query": state.question[:300]}, ctx)
            self._absorb(state, run, found)
            codes = [m["code"] for m in found.data.get("matches", [])][:1]
        if self.spec.name == "policy":
            plan = [("search_policy", {"query": state.question[:300]})]
        elif self.spec.name == "claims":
            plan = [("list_claims", {})]
        elif self.spec.name == "authorization":
            plan = [("get_prior_auth_status", {"service_code": c}) for c in codes]
        elif self.spec.name == "billing":
            plan = [("get_eligibility", {})] + [("estimate_cost", {"service_code": c}) for c in codes]
        for name, args in plan:
            self._absorb(state, run, await self._execute(name, args, ctx))
        run.note = "Retrieved records directly (language model unavailable)."
        return run

    def _absorb(self, state: CaseState, run: AgentRun, result: ToolResult) -> None:
        run.tool_calls += 1
        for fact in result.facts:
            fact.agent = self.spec.name
            if state.add_fact(fact):
                run.facts_added += 1
        for ev in result.evidence:
            if state.add_evidence(ev):
                run.evidence_added += 1

    async def _execute(self, name: str, args: dict, ctx: ToolContext) -> ToolResult:
        tool = self.tools.get(name)
        if tool is None:
            return ToolResult(data={"error": f"tool '{name}' is not available to this agent"})
        return await tool.run(ctx, args)
