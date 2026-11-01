"""In-memory backend for local development and tests. Mirrors seeds/demo.sql.

Not for production: state is per-process, so it breaks as soon as you run two replicas.
"""

from __future__ import annotations

import logging
import math
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

from careagents.data.models import (
    Accumulator,
    ChunkIn,
    Claim,
    Member,
    Plan,
    PriorAuth,
    SearchHit,
    Service,
    tokenize,
)
from careagents.logging_setup import log_event, phi_hash

logger = logging.getLogger(__name__)
D = Decimal


def demo_dataset() -> dict[str, Any]:
    plan = Plan(id="HSG-2026", name="HealthSure Gold 2026", plan_year=2026, deductible=D("500"),
                coinsurance_member_pct=D("20"), oop_max=D("5000"))
    members = [
        Member(id="SARAH-1001", plan_id=plan.id, active=True, coverage_start=date(2026, 1, 1)),
        Member(id="JOHN-2002", plan_id=plan.id, active=True, coverage_start=date(2026, 1, 1)),
    ]
    accumulators = [
        Accumulator(member_id="SARAH-1001", plan_year=2026, deductible_paid=D("200"), oop_paid=D("200")),
        Accumulator(member_id="JOHN-2002", plan_year=2026, deductible_paid=D("500"), oop_paid=D("900")),
    ]
    services = [
        Service(code="73721", name="MRI knee (lower extremity joint) without contrast",
                aliases=["mri", "knee mri", "mri knee", "magnetic resonance imaging"], category="advanced_imaging",
                requires_prior_auth=True, typical_charge=D("1000")),
        Service(code="99204", name="New patient office visit (orthopedic consultation)",
                aliases=["consultation", "office visit", "orthopedic consult"], category="office_visit",
                requires_prior_auth=False, typical_charge=D("200")),
        Service(code="97110", name="Physical therapy, therapeutic exercise",
                aliases=["physical therapy", "pt", "physio"], category="therapy",
                requires_prior_auth=False, typical_charge=D("120")),
    ]
    claims = [
        Claim(id="CLM-7781", member_id="SARAH-1001", service_code="99204", service_date=date(2026, 1, 10),
              provider="Lakeside Orthopedics", billed_amount=D("200"), status="approved",
              denial_reason="Applied to deductible", member_paid=D("200"), plan_paid=D("0")),
        Claim(id="CLM-8892", member_id="SARAH-1001", service_code="73721", service_date=date(2026, 2, 5),
              provider="North Dallas Imaging Center", billed_amount=D("1000"), status="denied",
              denial_code="PA_MISSING", denial_reason="Prior authorization not obtained before service date"),
        Claim(id="CLM-9001", member_id="JOHN-2002", service_code="97110", service_date=date(2026, 3, 2),
              provider="Uptown Physio", billed_amount=D("120"), status="approved",
              member_paid=D("24"), plan_paid=D("96")),
    ]
    auths = [
        PriorAuth(id="PA-5501", member_id="JOHN-2002", service_code="73721", status="approved",
                  requested_at=datetime(2026, 3, 10, tzinfo=UTC),
                  decided_at=datetime(2026, 3, 12, tzinfo=UTC), valid_until=date(2027, 3, 12)),
    ]
    return {"plans": [plan], "members": members, "accumulators": accumulators, "services": services,
            "claims": claims, "auths": auths}


class MemoryRepositories:
    def __init__(self, dataset: dict[str, Any] | None = None):
        ds = dataset or demo_dataset()
        self.plans = {p.id: p for p in ds["plans"]}
        self.members = {m.id: m for m in ds["members"]}
        self.accumulators = {(a.member_id, a.plan_year): a for a in ds["accumulators"]}
        self.services = {s.code: s for s in ds["services"]}
        self.claims: list[Claim] = list(ds["claims"])
        self.auths: list[PriorAuth] = list(ds["auths"])
        self.case_runs: list[dict[str, Any]] = []
        self.reviews: dict[str, dict[str, Any]] = {}

    async def ping(self) -> bool:
        return True

    async def get_member(self, member_id: str) -> Member | None:
        return self.members.get(member_id)

    async def get_plan(self, plan_id: str) -> Plan | None:
        return self.plans.get(plan_id)

    async def get_accumulator(self, member_id: str, plan_year: int) -> Accumulator | None:
        return self.accumulators.get((member_id, plan_year))

    async def get_service(self, code: str) -> Service | None:
        return self.services.get(code)

    async def find_services(self, query: str, limit: int = 5) -> list[Service]:
        q = tokenize(query)
        scored = []
        for s in self.services.values():
            vocab = tokenize(" ".join([s.code, s.name, *s.aliases]))
            score = len(q & vocab)
            if score:
                scored.append((score, s))
        scored.sort(key=lambda x: -x[0])
        return [s for _, s in scored[:limit]]

    async def list_claims(self, member_id: str, service_code: str | None = None,
                          status: str | None = None, limit: int = 20) -> list[Claim]:
        rows = [c for c in self.claims if c.member_id == member_id
                and (service_code is None or c.service_code == service_code)
                and (status is None or c.status == status)]
        return sorted(rows, key=lambda c: c.service_date, reverse=True)[:limit]

    async def get_claim(self, member_id: str, claim_id: str) -> Claim | None:
        return next((c for c in self.claims if c.id == claim_id and c.member_id == member_id), None)

    async def get_prior_auths(self, member_id: str, service_code: str) -> list[PriorAuth]:
        rows = [a for a in self.auths if a.member_id == member_id and a.service_code == service_code]
        return sorted(rows, key=lambda a: a.requested_at, reverse=True)

    async def save_case_run(self, record: dict[str, Any]) -> None:
        self.case_runs.append(record)

    async def enqueue_review(self, record: dict[str, Any]) -> None:
        self.reviews.setdefault(record["case_id"], {
            **record, "status": "pending", "created_at": datetime.now(UTC),
            "reviewer_id": None, "decision": None, "decided_at": None})

    async def list_reviews(self, status: str = "pending", limit: int = 50) -> list[dict[str, Any]]:
        rows = [r for r in self.reviews.values() if r["status"] == status]
        return sorted(rows, key=lambda r: r["created_at"])[:limit]

    async def get_review(self, case_id: str) -> dict[str, Any] | None:
        return self.reviews.get(case_id)

    async def complete_review(self, case_id: str, *, reviewer_id: str, decision: str) -> None:
        if case_id in self.reviews:
            self.reviews[case_id].update(status="done", reviewer_id=reviewer_id, decision=decision,
                                         decided_at=datetime.now(UTC))


class MemoryDocumentStore:
    def __init__(self) -> None:
        self.chunks: dict[tuple[str, int], tuple[str, str, str, list[float]]] = {}

    async def search(self, embedding: list[float], doc_type: str, top_k: int) -> list[SearchHit]:
        hits = []
        for (source, idx), (dtype, content, _h, vec) in self.chunks.items():
            if dtype != doc_type:
                continue
            score = sum(a * b for a, b in zip(embedding, vec, strict=False))
            if not math.isnan(score):
                hits.append(SearchHit(source=source, doc_type=dtype, chunk_index=idx, content=content, score=score))
        hits.sort(key=lambda h: -h.score)
        return hits[:top_k]

    async def existing_hashes(self, source: str) -> dict[int, str]:
        return {idx: h for (src, idx), (_d, _c, h, _v) in self.chunks.items() if src == source}

    async def replace_chunks(self, source: str, doc_type: str, chunks: list[ChunkIn]) -> int:
        for key in [k for k in self.chunks if k[0] == source]:
            del self.chunks[key]
        for c in chunks:
            self.chunks[(source, c.index)] = (doc_type, c.content, c.content_hash, c.embedding)
        return len(chunks)


class MemoryAuditLogger:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def record(self, *, actor_id: str, member_id: str | None, action: str, request_id: str | None) -> None:
        self.events.append({"actor_id": actor_id, "member_id": member_id, "action": action,
                            "request_id": request_id})
        log_event(logger, "audit", actor=actor_id, member=phi_hash(member_id), action=action)
