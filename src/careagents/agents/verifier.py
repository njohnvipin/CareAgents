"""Verifier: deterministic grounding checks first, then an optional LLM audit."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

from careagents.agents.llm_call import call_llm
from careagents.agents.prompts import VERIFIER_SYSTEM, VERIFIER_USER
from careagents.agents.state import CaseState
from careagents.llm.client import LLMClient, LLMError
from careagents.metrics import VERIFIER_ISSUES

MONEY = re.compile(r"\$\s?(\d{1,3}(?:,\d{3})+|\d+)(\.\d{1,2})?")
PERCENT = re.compile(r"(\d{1,3}(?:\.\d+)?)\s?%")
NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
CENT = Decimal("0.01")


def _dec(text: str) -> Decimal | None:
    try:
        return Decimal(text.replace(",", "")).quantize(CENT)
    except InvalidOperation:
        return None


def grounded_numbers(state: CaseState) -> set[Decimal]:
    corpus = [state.question, *(f.value for f in state.facts.values()), *(e.text for e in state.evidence)]
    out: set[Decimal] = set()
    for text in corpus:
        for m in NUMBER.findall(text):
            d = _dec(m)
            if d is not None:
                out.add(d)
    return out


def ungrounded_amounts(answer: str, allowed: set[Decimal]) -> list[str]:
    bad = []
    for m in MONEY.finditer(answer):
        d = _dec(m.group(1) + (m.group(2) or ""))
        if d is not None and d not in allowed:
            bad.append(m.group(0))
    for m in PERCENT.finditer(answer):
        d = _dec(m.group(1))
        if d is not None and d not in allowed:
            bad.append(m.group(0))
    return bad


@dataclass
class VerificationReport:
    ok: bool
    issues: list[str] = field(default_factory=list)
    rerun_agents: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)


class Verifier:
    def __init__(self, llm: LLMClient, llm_enabled: bool = True):
        self.llm = llm
        self.llm_enabled = llm_enabled

    def structural_check(self, state: CaseState) -> VerificationReport:
        """Before composing: did every planned agent produce grounded output?"""
        rerun = []
        for name in state.plan:
            run = state.runs.get(name)
            if run is None or run.attempts >= 2:
                continue
            if run.status == "failed" or (run.facts_added + run.evidence_added == 0):
                rerun.append(name)
                VERIFIER_ISSUES.labels("ungrounded_agent").inc()
        return VerificationReport(ok=not rerun, rerun_agents=rerun)

    async def check_draft(self, state: CaseState, draft: str) -> VerificationReport:
        issues = []
        bad = ungrounded_amounts(draft, grounded_numbers(state))
        if bad:
            VERIFIER_ISSUES.labels("ungrounded_number").inc()
            issues.append("These amounts are not in the verified facts or plan excerpts: " + ", ".join(bad))
        flags = ["data_conflict"] if state.conflicts else []

        if not issues and self.llm_enabled:
            try:
                resp = await call_llm(
                    self.llm, state, tier="strong", json_mode=True, temperature=0.0, max_tokens=400,
                    messages=[{"role": "system", "content": VERIFIER_SYSTEM},
                              {"role": "user", "content": VERIFIER_USER.format(
                                  facts=state.facts_block(), evidence=state.evidence_block(), draft=draft)}])
                data = json.loads(resp.content)
                if data.get("ok") is False:
                    llm_issues = [str(i)[:300] for i in (data.get("issues") or [])][:5]
                    if llm_issues:
                        VERIFIER_ISSUES.labels("llm_audit").inc()
                        issues.extend(llm_issues)
            except (LLMError, ValueError, TypeError, AttributeError):
                flags.append("llm_audit_skipped")  # don't block an answer that passed deterministic checks
        return VerificationReport(ok=not issues, issues=issues, flags=flags)
