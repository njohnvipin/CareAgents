"""Optional OpenTelemetry tracing (pip install careagents[otel]; set CA_OTEL_ENABLED=true and the
standard OTEL_EXPORTER_OTLP_ENDPOINT). Traces cover HTTP requests and every outbound LLM call."""

import logging

from fastapi import FastAPI

logger = logging.getLogger(__name__)


def setup_tracing(app: FastAPI, service_name: str) -> None:
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        logger.warning("otel_enabled_but_not_installed")
        return
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    # Request/response bodies are not captured, so PHI does not end up in traces.
    FastAPIInstrumentor.instrument_app(app, excluded_urls="livez,readyz,metrics")
    HTTPXClientInstrumentor().instrument()
