import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from careagents.agents.runtime import CaseResult, ReviewNotPendingError
from careagents.api.deps import get_container, get_principal, rate_limit, require_role
from careagents.api.schemas import ChatRequest, IngestResponse, ReviewDecision
from careagents.auth.security import Principal
from careagents.container import Container
from careagents.logging_setup import request_id_var

router = APIRouter()


@router.get("/livez", include_in_schema=False)
async def livez() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz", include_in_schema=False)
async def readyz(container: Container = Depends(get_container)) -> JSONResponse:
    # Readiness covers dependencies we own. The LLM provider is deliberately excluded: an
    # upstream outage should degrade answers, not pull every pod out of the load balancer.
    checks: dict[str, bool] = {}
    try:
        checks["database"] = await container.repos.ping()
    except Exception:
        checks["database"] = False
    if container.redis is not None:
        try:
            checks["redis"] = bool(await container.redis.ping())
        except Exception:
            checks["redis"] = False
    ok = all(checks.values())
    return JSONResponse({"status": "ok" if ok else "unavailable", "checks": checks},
                        status_code=200 if ok else 503)


async def _resolve_member(body: ChatRequest, principal: Principal, container: Container) -> str:
    if principal.role == "member":
        if body.member_id and body.member_id != principal.member_id:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "members can only access their own record")
        member_id = principal.member_id
    else:
        if not body.member_id:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "member_id is required for staff requests")
        member_id = body.member_id
    assert member_id is not None
    if await container.repos.get_member(member_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "member not found")
    return member_id


# ---------------------------------------------------------------- members (and staff on their behalf)
@router.post("/v1/chat", response_model=CaseResult, dependencies=[Depends(rate_limit)])
async def chat(body: ChatRequest, principal: Principal = Depends(get_principal),
               container: Container = Depends(get_container)) -> CaseResult:
    member_id = await _resolve_member(body, principal, container)
    return await container.runtime.handle(body.question, member_id, principal.subject, request_id_var.get())


@router.post("/v1/chat/stream", dependencies=[Depends(rate_limit)],
             responses={200: {"content": {"text/event-stream": {}}}})
async def chat_stream(body: ChatRequest, principal: Principal = Depends(get_principal),
                      container: Container = Depends(get_container)) -> StreamingResponse:
    """Server-sent events: `started`, one `step` per graph node as it finishes, then `result`."""
    member_id = await _resolve_member(body, principal, container)
    request_id = request_id_var.get()

    async def events() -> AsyncIterator[str]:
        async for evt in container.runtime.stream(body.question, member_id, principal.subject, request_id):
            yield f"event: {evt['event']}\ndata: {json.dumps(evt, default=str)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


@router.get("/v1/cases/{case_id}", response_model=CaseResult)
async def get_case(case_id: str, principal: Principal = Depends(get_principal),
                   container: Container = Depends(get_container)) -> CaseResult:
    """Poll a case, e.g. one waiting for staff review."""
    owner = await container.runtime.case_owner(case_id)
    if owner is None or (principal.role == "member" and owner != principal.member_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "case not found")  # don't reveal other members' cases
    if principal.role != "member":
        await container.audit.record(actor_id=principal.subject, member_id=owner, action="case:view",
                                     request_id=request_id_var.get())
    result = await container.runtime.get_case(case_id)
    assert result is not None
    return result


# ---------------------------------------------------------------- staff review queue (human-in-the-loop)
@router.get("/v1/staff/reviews")
async def list_reviews(status_filter: str = "pending", _: Principal = Depends(require_role("staff", "admin")),
                       container: Container = Depends(get_container)) -> list[dict]:
    if status_filter not in ("pending", "done"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "status_filter must be pending or done")
    rows = await container.repos.list_reviews(status_filter)
    return [{k: v for k, v in r.items() if k not in ("question", "draft")} for r in rows]  # list view: no PHI text


@router.get("/v1/staff/reviews/{case_id}")
async def get_review(case_id: str, principal: Principal = Depends(require_role("staff", "admin")),
                     container: Container = Depends(get_container)) -> dict:
    review = await container.repos.get_review(case_id)
    if review is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "review not found")
    await container.audit.record(actor_id=principal.subject, member_id=review["member_id"], action="review:view",
                                 request_id=request_id_var.get())
    case = await container.runtime.get_case(case_id)
    return {"review": review, "case": case.model_dump(mode="json") if case else None}


@router.post("/v1/staff/reviews/{case_id}", response_model=CaseResult)
async def decide_review(case_id: str, decision: ReviewDecision,
                        principal: Principal = Depends(require_role("staff", "admin")),
                        container: Container = Depends(get_container)) -> CaseResult:
    """Resume the paused graph with the reviewer's decision."""
    try:
        return await container.runtime.resume(case_id, {**decision.model_dump(), "reviewer_id": principal.subject})
    except ReviewNotPendingError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, "case is not waiting for review") from exc


# ---------------------------------------------------------------- operations
@router.post("/v1/admin/ingest", response_model=IngestResponse, status_code=202)
async def ingest(_: Principal = Depends(require_role("admin")),
                 container: Container = Depends(get_container)) -> IngestResponse:
    if container.job_queue is not None:
        job = await container.job_queue.enqueue_job("ingest_documents")
        # The worker holds a Redis lock, so concurrent requests can't run overlapping ingestions.
        return IngestResponse(status="queued", job_id=job.job_id if job else None)
    if container.settings.env in ("staging", "prod"):
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "job queue unavailable")
    from careagents.ingestion.pipeline import ingest_directory
    stats = await ingest_directory(container.settings.documents_dir, container.store, container.embedder,
                                   container.search)
    return IngestResponse(status="completed", stats=stats)


@router.get("/v1/admin/jobs/{job_id}")
async def job_status(job_id: str, _: Principal = Depends(require_role("admin")),
                     container: Container = Depends(get_container)) -> dict:
    if container.job_queue is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no job queue configured")
    from arq.jobs import Job
    job = Job(job_id, container.job_queue)
    info = await job.result_info()
    return {"job_id": job_id, "status": (await job.status()).value,
            "success": info.success if info else None,
            "result": info.result if info and info.success else None}


@router.get("/v1/admin/status")
async def system_status(request: Request, _: Principal = Depends(require_role("admin")),
                        container: Container = Depends(get_container)) -> dict:
    breakers: dict[str, str] = getattr(container.llm, "breaker_states", dict)()
    return {"env": container.settings.env, "llm_breakers": breakers,
            "fast_models": container.settings.llm_fast_models,
            "strong_models": container.settings.llm_strong_models,
            "human_review_enabled": container.settings.human_review_enabled}


@router.get("/v1/admin/graph", response_class=PlainTextResponse)
async def graph_diagram(_: Principal = Depends(require_role("admin")),
                        container: Container = Depends(get_container)) -> str:
    """The live LangGraph workflow as Mermaid source."""
    return container.runtime.mermaid()
