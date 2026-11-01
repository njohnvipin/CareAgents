"""Supervisor: plans which specialists to run, and composes the final answer from shared state."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from careagents.agents.llm_call import call_llm
from careagents.agents.prompts import COMPOSER_SYSTEM, COMPOSER_USER, PLANNER_SYSTEM
from careagents.agents.specialist import AGENT_ORDER, SPECS
from careagents.agents.state import CaseState
from careagents.llm.client import LLMClient, LLMError

ESCALATION_REASONS = {"emergency", "clinical_advice", "complaint_or_legal", "out_of_scope"}

# Deterministic safety net that runs before any model call. Word boundaries matter: the original
# prototype matched "pa" inside "pay", routing every cost question to the authorization agent.
EMERGENCY = re.compile(
    r"\b(chest pain|can'?t breathe|cannot breathe|trouble breathing|stroke|overdos\w*|suicid\w*|"
    r"kill myself|end my life|self[- ]harm|heavy bleeding|unconscious)\b", re.I)

KEYWORDS: dict[str, list[str]] = {
    "policy": [r"\bcover(ed|age|s)?\b", r"\bbenefits?\b", r"\bpolicy\b", r"\bexclu\w*", r"\bin[- ]network\b"],
    "claims": [r"\bclaims?\b", r"\bdenied\b", r"\bdenial\b", r"\beob\b", r"\bappeal\w*\b", r"\breimburs\w*"],
    "authorization": [r"\bprior auth\w*", r"\bpre-?auth\w*", r"\bauthori[sz]\w*", r"\bpa\b", r"\bmedical necessity\b",
                      r"\bapproval\b"],
    "billing": [r"\bhow much\b", r"\bcosts?\b", r"\bpay\b", r"\bowe\b", r"\bout[- ]of[- ]pocket\b", r"\bdeductible\b",
                r"\bcoinsurance\b", r"\bbill\w*\b", r"\bprice\b"],
}


def keyword_route(question: str) -> list[str]:
    q = question.lower()
    agents = [a for a in AGENT_ORDER if any(re.search(p, q) for p in KEYWORDS[a])]
    return agents or ["policy"]


@dataclass
class Plan:
    agents: list[str] = field(default_factory=list)
    escalate: bool = False
    reason: str | None = None
    source: str = "llm"


class Supervisor:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def plan(self, state: CaseState) -> Plan:
        if EMERGENCY.search(state.question):
            return Plan(escalate=True, reason="emergency", source="rules")
        try:
            resp = await call_llm(self.llm, state, tier="fast", json_mode=True, temperature=0.0, max_tokens=300,
                                  messages=[{"role": "system", "content": PLANNER_SYSTEM},
                                            {"role": "user", "content": state.question}])
            data = json.loads(resp.content)
            agents = [a for a in dict.fromkeys(data.get("agents") or []) if a in SPECS]
            reason = data.get("reason")
            if data.get("escalate") and reason in ESCALATION_REASONS:
                return Plan(agents=[], escalate=True, reason=reason)
            return Plan(agents=agents or keyword_route(state.question))
        except (LLMError, ValueError, TypeError, AttributeError):
            return Plan(agents=keyword_route(state.question), source="keywords")

    async def compose(self, state: CaseState) -> str:
        notes = "\n".join(f"- {name}: {run.note or run.status}" for name, run in state.runs.items()) or "(none)"
        failed = [n for n, r in state.runs.items() if r.status == "failed"]
        extra = []
        if failed:
            extra.append("Unavailable right now: " + ", ".join(failed) + " information.")
        if state.conflicts:
            extra.append("Records disagree on: " + "; ".join(state.conflicts) + ". Tell the member a representative "
                         "will confirm.")
        if state.issues:
            extra.append("A reviewer rejected your previous draft. Fix these problems:\n- " + "\n- ".join(state.issues))
        resp = await call_llm(
            self.llm, state, tier="strong", temperature=0.2, max_tokens=900,
            messages=[{"role": "system", "content": COMPOSER_SYSTEM},
                      {"role": "user", "content": COMPOSER_USER.format(
                          question=state.question, facts=state.facts_block(), evidence=state.evidence_block(),
                          notes=notes, issues=("\n" + "\n".join(extra)) if extra else "")}])
        answer = resp.content.strip()
        if not answer:
            raise LLMError("empty composition")
        return answer


AUTH_STATUS_TEXT = {"none_on_file": "no authorization on file", "approved": "approved", "pending": "pending",
                    "denied": "denied", "expired": "expired"}


def render_records(state: CaseState) -> list[str]:
    """Turn verified facts into plain sentences, grouped by topic. No model involved."""
    f = {k: v.value for k, v in state.facts.items()}
    lines: list[str] = []

    claim_ids = sorted({k.split(".")[1] for k in f if k.startswith("claim.") and k.endswith(".status")})
    for cid in claim_ids:
        g = lambda field, c=cid: f.get(f"claim.{c}.{field}")  # noqa: E731
        name = f.get(f"service.{g('service_code')}.name") or f"service {g('service_code')}"
        line = f"Claim {cid} ({name}, {g('service_date')}): {g('status')}"
        if g("reason"):
            line += f". {g('reason')}"
        if g("denial_code"):
            line += f" (code {g('denial_code')})"
        lines.append(line + f". Billed ${g('billed_amount')}, you paid ${g('member_paid')}, "
                     f"plan paid ${g('plan_paid')}.")

    for key, value in f.items():
        if key.startswith("prior_auth.") and key.endswith(".status"):
            code = key.split(".")[1]
            required = f.get(f"service.{code}.requires_prior_auth") == "true"
            name = f.get(f"service.{code}.name", f"service {code}")
            lines.append(f"{name}: prior authorization {'is' if required else 'is not'} required; status: "
                         f"{AUTH_STATUS_TEXT.get(value, value)}.")

    if "eligibility.deductible_remaining" in f:
        e = {k.removeprefix("eligibility."): v for k, v in f.items() if k.startswith("eligibility.")}
        lines.append(f"Plan {e.get('plan_name')}: deductible remaining ${e['deductible_remaining']} of "
                     f"${e.get('deductible')}; out-of-pocket remaining ${e.get('oop_remaining')} of "
                     f"${e.get('oop_max')}; your coinsurance is {e.get('coinsurance_member_pct')}%.")

    for key, value in f.items():
        if key.startswith("estimate.") and key.endswith(".if_covered.member_pays"):
            code = key.split(".")[1]
            name = f.get(f"service.{code}.name", f"service {code}")
            line = f"Estimated cost for {name} (charge ${f.get(f'estimate.{code}.charge')}): about ${value} if covered"
            worst = f.get(f"estimate.{code}.if_auth_missing.member_may_owe_up_to")
            if worst:
                line += f"; up to ${worst} if prior authorization is not obtained"
            lines.append(line + ".")

    sections = sorted({e.text.splitlines()[0] for e in state.evidence if e.doc_type == "policy"})[:3]
    if sections:
        lines.append("Relevant plan sections: " + "; ".join(sections) + ".")
    return lines


def fallback_answer(state: CaseState, reason: str) -> str:
    """Honest, deterministic answer used when the models are unavailable or a draft can't be verified.

    Unlike the prototype's canned MRI text, this only states what the system of record returned.
    """
    lead = {
        "timeout": "This is taking longer than expected, so here is what I could confirm from your records:",
        "llm_unavailable": "I can't write a full explanation right now, but here is what your records show:",
        "verification_failed": "I couldn't produce an explanation I could fully verify, so here are the "
                               "confirmed details from your records:",
        "under_review": "A member services specialist is reviewing your question to make sure the answer is "
                        "accurate. You'll see their reply here. In the meantime, here is what your records show:",
    }.get(reason, "Here is what your records show:")
    lines = render_records(state)
    body = "\n".join(f"- {line}" for line in lines) if lines else "- No records could be retrieved right now."
    return (f"{lead}\n\n{body}\n\nFor a full explanation, please contact member services using the number "
            "on the back of your ID card.")
