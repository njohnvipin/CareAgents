"""Shared case state: the blackboard every agent reads from and writes to.

Facts are written by *tools* (deterministic code reading the system of record), never by the
LLM. That is what lets the verifier check the final answer against something trustworthy.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Fact(BaseModel):
    key: str
    value: str
    source: str
    agent: str | None = None


class Evidence(BaseModel):
    source: str
    doc_type: str
    chunk_index: int
    text: str
    score: float | None = None


class AgentRun(BaseModel):
    agent: str
    status: Literal["ok", "failed"] = "ok"
    attempts: int = 0
    tool_calls: int = 0
    facts_added: int = 0
    evidence_added: int = 0
    note: str = ""
    error: str | None = None


class Usage(BaseModel):
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class CaseState(BaseModel):
    case_id: str
    member_id: str
    actor_id: str
    question: str
    request_id: str | None = None
    plan: list[str] = Field(default_factory=list)
    facts: dict[str, Fact] = Field(default_factory=dict)
    fact_log: list[Fact] = Field(default_factory=list)  # every fact in arrival order (graph deltas use this)
    conflicts: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    runs: dict[str, AgentRun] = Field(default_factory=dict)
    issues: list[str] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)

    def add_fact(self, fact: Fact) -> bool:
        existing = self.facts.get(fact.key)
        if existing and existing.value != fact.value:
            self.conflicts.append(f"{fact.key}: '{existing.value}' ({existing.source}) vs "
                                  f"'{fact.value}' ({fact.source})")
        is_new = existing is None
        self.facts[fact.key] = fact
        self.fact_log.append(fact)
        return is_new

    def add_evidence(self, ev: Evidence) -> bool:
        if any(e.source == ev.source and e.chunk_index == ev.chunk_index for e in self.evidence):
            return False
        self.evidence.append(ev)
        return True

    def facts_block(self) -> str:
        if not self.facts:
            return "(none)"
        return "\n".join(f"- {f.key} = {f.value}  [source: {f.source}]" for f in self.facts.values())

    def evidence_block(self, limit: int = 6, max_chars: int = 700) -> str:
        if not self.evidence:
            return "(none)"
        ranked = sorted(self.evidence, key=lambda e: -(e.score or 0))[:limit]
        return "\n\n".join(f"[{e.source} #{e.chunk_index}]\n{e.text[:max_chars]}" for e in ranked)
