"""Domain models and storage interfaces.

Design note: claims, eligibility and prior authorizations are *records*, so agents query them
through typed tools backed by SQL. Only unstructured text (policy documents, clinical
guidelines) goes through vector search. RAG over a claims export (as in the original prototype)
can't guarantee exact statuses or amounts and leaks across members.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any, Protocol

from pydantic import BaseModel


class Plan(BaseModel):
    id: str
    name: str
    plan_year: int
    deductible: Decimal
    coinsurance_member_pct: Decimal
    oop_max: Decimal


class Member(BaseModel):
    id: str
    plan_id: str
    active: bool
    coverage_start: date
    coverage_end: date | None = None


class Accumulator(BaseModel):
    member_id: str
    plan_year: int
    deductible_paid: Decimal = Decimal("0")
    oop_paid: Decimal = Decimal("0")


class Service(BaseModel):
    code: str
    name: str
    aliases: list[str] = []
    category: str
    requires_prior_auth: bool
    typical_charge: Decimal


class Claim(BaseModel):
    id: str
    member_id: str
    service_code: str
    service_date: date
    provider: str | None = None
    billed_amount: Decimal
    status: str
    denial_code: str | None = None
    denial_reason: str | None = None
    member_paid: Decimal = Decimal("0")
    plan_paid: Decimal = Decimal("0")


class PriorAuth(BaseModel):
    id: str
    member_id: str
    service_code: str
    status: str
    requested_at: datetime
    decided_at: datetime | None = None
    valid_until: date | None = None


class ChunkIn(BaseModel):
    index: int
    content: str
    content_hash: str
    embedding: list[float]


class SearchHit(BaseModel):
    source: str
    doc_type: str
    chunk_index: int
    content: str
    score: float


class Repositories(Protocol):
    async def ping(self) -> bool: ...
    async def get_member(self, member_id: str) -> Member | None: ...
    async def get_plan(self, plan_id: str) -> Plan | None: ...
    async def get_accumulator(self, member_id: str, plan_year: int) -> Accumulator | None: ...
    async def get_service(self, code: str) -> Service | None: ...
    async def find_services(self, query: str, limit: int = 5) -> list[Service]: ...
    async def list_claims(self, member_id: str, service_code: str | None = None,
                          status: str | None = None, limit: int = 20) -> list[Claim]: ...
    async def get_claim(self, member_id: str, claim_id: str) -> Claim | None: ...
    async def get_prior_auths(self, member_id: str, service_code: str) -> list[PriorAuth]: ...
    async def save_case_run(self, record: dict[str, Any]) -> None: ...
    async def enqueue_review(self, record: dict[str, Any]) -> None: ...
    async def list_reviews(self, status: str = "pending", limit: int = 50) -> list[dict[str, Any]]: ...
    async def get_review(self, case_id: str) -> dict[str, Any] | None: ...
    async def complete_review(self, case_id: str, *, reviewer_id: str, decision: str) -> None: ...


class DocumentStore(Protocol):
    async def search(self, embedding: list[float], doc_type: str, top_k: int) -> list[SearchHit]: ...
    async def existing_hashes(self, source: str) -> dict[int, str]: ...
    async def replace_chunks(self, source: str, doc_type: str, chunks: list[ChunkIn]) -> int: ...


class AuditLogger(Protocol):
    async def record(self, *, actor_id: str, member_id: str | None, action: str,
                     request_id: str | None) -> None: ...


def tokenize(text: str) -> set[str]:
    return {t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split() if len(t) > 1}
