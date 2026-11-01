"""Scripted LLM double. Identifies the caller by shape (tools offered, json mode, tier)
rather than by prompt text, so prompt edits don't break tests."""

from __future__ import annotations

import itertools
import json
import re
from collections.abc import Callable
from typing import Any

from careagents.llm.client import LLMResponse, LLMUnavailableError, ToolCall

DEFAULT_TOOL_SCRIPTS: dict[str, list[tuple[str, dict[str, Any]]]] = {
    "search_policy": [("search_policy", {"query": "MRI coverage"})],
    "list_claims": [("list_claims", {})],
    "get_prior_auth_status": [("get_prior_auth_status", {"service_code": "73721"})],
    "estimate_cost": [("get_eligibility", {}), ("estimate_cost", {"service_code": "73721"})],
}
AGENT_BY_TOOL = {"search_policy": "policy", "list_claims": "claims", "get_prior_auth_status": "authorization",
                 "estimate_cost": "billing"}


def good_answer(user_msg: str) -> str:
    amounts = re.findall(r"if_covered\.member_pays = ([\d.]+)", user_msg)
    worst = re.findall(r"if_auth_missing\.member_may_owe_up_to = ([\d.]+)", user_msg)
    parts = ["Coverage: MRI is covered when medically necessary, after the deductible you pay 20%."]
    if amounts:
        parts.append(f"Your cost: about ${amounts[0]} if prior authorization is approved.")
    if worst:
        parts.append(f"Without prior authorization you may owe up to ${worst[0]}.")
    parts.append("Next steps: ask your doctor to submit a prior authorization request.")
    return "\n".join(parts)


class ScriptedLLM:
    def __init__(self, plan: dict[str, Any] | None = None,
                 compose: Callable[[str], str] | list[str] | None = None,
                 tool_scripts: dict[str, list[tuple[str, dict[str, Any]]]] | None = None,
                 fail_agents: set[str] | None = None, fail_all: bool = False, verifier_ok: bool = True):
        self.plan = plan or {"agents": ["policy", "claims", "authorization", "billing"], "escalate": False,
                             "reason": None}
        self._compose = compose or good_answer
        self._compose_iter = iter(compose) if isinstance(compose, list) else None
        self.tool_scripts = tool_scripts or DEFAULT_TOOL_SCRIPTS
        self.fail_agents = fail_agents or set()
        self.fail_all = fail_all
        self.verifier_ok = verifier_ok
        self.calls: list[dict[str, Any]] = []
        self._ids = itertools.count()

    async def aclose(self) -> None:
        pass

    async def chat(self, messages, *, tools=None, json_mode=False, tier="fast", temperature=0.1,
                   max_tokens=1024) -> LLMResponse:
        kind = "specialist" if tools else ("planner" if json_mode and tier == "fast"
                                           else "verifier" if json_mode else "composer")
        self.calls.append({"kind": kind, "messages": messages, "tools": tools})
        if self.fail_all:
            raise LLMUnavailableError("all models down")
        if kind == "planner":
            return LLMResponse(content=json.dumps(self.plan), prompt_tokens=50, completion_tokens=10)
        if kind == "verifier":
            body = {"ok": True, "issues": []} if self.verifier_ok else {"ok": False, "issues": ["contradiction"]}
            return LLMResponse(content=json.dumps(body), prompt_tokens=100, completion_tokens=10)
        if kind == "composer":
            user = messages[-1]["content"]
            text = next(self._compose_iter) if self._compose_iter else self._compose(user)
            return LLMResponse(content=text, prompt_tokens=300, completion_tokens=120)

        names = {t["function"]["name"] for t in tools}
        key = next(k for k in self.tool_scripts if k in names)
        agent = AGENT_BY_TOOL.get(key, key)
        if agent in self.fail_agents:
            raise LLMUnavailableError(f"{agent} model down")
        if any(m["role"] == "tool" for m in messages):
            return LLMResponse(content=f"{agent} findings recorded.", prompt_tokens=80, completion_tokens=20)
        calls = [ToolCall(id=f"call_{next(self._ids)}", name=n, arguments=a, raw_arguments=json.dumps(a))
                 for n, a in self.tool_scripts[key]]
        return LLMResponse(content="", tool_calls=calls, prompt_tokens=80, completion_tokens=20)
