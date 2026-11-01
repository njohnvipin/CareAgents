"""ASGI application factory: `uvicorn careagents.api.app:create_app --factory`."""

from __future__ import annotations

import logging
import re
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from careagents import __version__
from careagents.api.routes import router
from careagents.config import Settings, get_settings
from careagents.container import Container, build_container
from careagents.logging_setup import configure_logging, log_event, request_id_var
from careagents.metrics import HTTP_LATENCY, HTTP_REQUESTS

logger = logging.getLogger("careagents.access")
SAFE_ID = re.compile(r"^[A-Za-z0-9._-]{8,64}$")


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Request id, access log (no bodies, no query strings), metrics and API security headers."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        incoming = request.headers.get("x-request-id", "")
        rid = incoming if SAFE_ID.match(incoming) else uuid.uuid4().hex
        token = request_id_var.set(rid)
        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
        finally:
            elapsed = time.perf_counter() - start
            route = request.scope.get("route")
            path = getattr(route, "path", "unmatched")
            HTTP_REQUESTS.labels(path, request.method, str(status)).inc()
            HTTP_LATENCY.labels(path, request.method).observe(elapsed)
            if path not in ("/livez", "/readyz"):
                log_event(logger, "http_request", method=request.method, route=path, status=status,
                          ms=int(elapsed * 1000))
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = rid
        response.headers["Cache-Control"] = "no-store"  # responses can contain PHI
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owned = container is None
        app.state.container = container or await build_container(settings)
        log_event(logging.getLogger(__name__), "startup", env=settings.env, version=__version__,
                  data_backend=settings.data_backend, models=settings.llm_fast_models)
        try:
            yield
        finally:
            if owned:
                await app.state.container.aclose()

    public_docs = settings.env in ("dev", "test")
    app = FastAPI(title="CareAgents", version=__version__, lifespan=lifespan,
                  docs_url="/docs" if public_docs else None, redoc_url=None,
                  openapi_url="/openapi.json" if public_docs else None)
    app.add_middleware(RequestContextMiddleware)
    if settings.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["GET", "POST"],
                           allow_headers=["Authorization", "Content-Type", "X-Request-ID"])

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        # Don't echo the input back: it may contain PHI.
        errors = [{"loc": e["loc"], "msg": e["msg"]} for e in exc.errors()]
        return JSONResponse({"error": "invalid_request", "details": errors, "request_id": request_id_var.get()},
                            status_code=422)

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        logging.getLogger(__name__).exception("unhandled_error")
        return JSONResponse({"error": "internal_error", "request_id": request_id_var.get()}, status_code=500)

    app.include_router(router)
    # Scraped from inside the cluster; don't route /metrics through the public ingress.
    app.mount("/metrics", make_asgi_app())
    if settings.otel_enabled:
        from careagents.tracing import setup_tracing
        setup_tracing(app, settings.service_name)
    return app
