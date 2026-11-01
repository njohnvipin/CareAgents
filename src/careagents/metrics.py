"""Prometheus metrics. Labels are kept low-cardinality (route templates, model ids, agent names)."""

from prometheus_client import Counter, Histogram

HTTP_REQUESTS = Counter("ca_http_requests_total", "HTTP requests", ["route", "method", "status"])
HTTP_LATENCY = Histogram(
    "ca_http_request_seconds", "HTTP latency", ["route", "method"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 15, 30, 60),
)
LLM_REQUESTS = Counter("ca_llm_requests_total", "LLM HTTP calls by outcome", ["model", "outcome"])
LLM_LATENCY = Histogram(
    "ca_llm_request_seconds", "LLM call latency", ["model"], buckets=(0.25, 0.5, 1, 2, 4, 8, 16, 32)
)
LLM_TOKENS = Counter("ca_llm_tokens_total", "LLM tokens", ["model", "kind"])
AGENT_RUNS = Counter("ca_agent_runs_total", "Specialist agent runs", ["agent", "status"])
TOOL_CALLS = Counter("ca_tool_calls_total", "Tool invocations", ["tool", "status"])
CASES = Counter("ca_cases_total", "Completed cases by outcome", ["outcome"])
CASE_LATENCY = Histogram("ca_case_seconds", "End-to-end case latency", buckets=(1, 2, 4, 8, 15, 30, 45, 60))
VERIFIER_ISSUES = Counter("ca_verifier_issues_total", "Verifier findings", ["kind"])
RATE_LIMITED = Counter("ca_rate_limited_total", "Requests rejected by the rate limiter")
