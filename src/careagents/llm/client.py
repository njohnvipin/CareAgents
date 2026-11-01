"""Resilient client for any OpenAI-compatible chat-completions API (Groq by default).

Failure handling, in order:
  1. Per-call timeout.
  2. Retries with exponential backoff + jitter on 429 / 5xx / transport errors (honours Retry-After).
  3. Per-model circuit breaker so a dead model stops costing latency.
  4. Fallback down an ordered model chain. This is what keeps the service alive through a
     model deprecation (e.g. Groq retiring llama-3.1-8b-instant) instead of failing every call.
  5. A process-wide semaphore bounds concurrent calls so a traffic spike can't stampede the provider.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import httpx

from careagents.config import Settings
from careagents.logging_setup import log_event
from careagents.metrics import LLM_LATENCY, LLM_REQUESTS, LLM_TOKENS

logger = logging.getLogger(__name__)

Tier = Literal["fast", "strong"]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: str


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def assistant_message(self) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": "assistant", "content": self.content or ""}
        if self.tool_calls:
            msg["tool_calls"] = [
                {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": tc.raw_arguments}}
                for tc in self.tool_calls
            ]
        return msg


class LLMError(Exception):
    """Base error; callers only need to catch this."""


class LLMUnavailableError(LLMError):
    """Every model in the chain failed or is circuit-broken."""


class _RetryableError(LLMError):
    def __init__(self, msg: str, retry_after: float | None = None):
        super().__init__(msg)
        self.retry_after = retry_after


class _FatalError(LLMError):
    pass


class LLMClient(Protocol):
    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        json_mode: bool = False,
        tier: Tier = "fast",
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> LLMResponse: ...

    async def aclose(self) -> None: ...


class CircuitBreaker:
    """Closed -> open after N consecutive failures -> half-open after cooldown (one trial call).

    State is per process. With many replicas that is usually fine (each learns quickly); a shared
    Redis-backed breaker is a drop-in replacement if you need fleet-wide state.
    """

    def __init__(self, threshold: int, cooldown_s: float):
        self._threshold = threshold
        self._cooldown = cooldown_s
        self._failures = 0
        self._opened_at: float | None = None

    @property
    def state(self) -> str:
        if self._opened_at is None:
            return "closed"
        return "half_open" if self._cooldown_elapsed() else "open"

    def _cooldown_elapsed(self) -> bool:
        return self._opened_at is not None and time.monotonic() - self._opened_at >= self._cooldown

    def allow(self) -> bool:
        return self._opened_at is None or self._cooldown_elapsed()

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self._threshold:
            self._opened_at = time.monotonic()


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after")
    try:
        return min(float(value), 30.0) if value else None
    except ValueError:
        return None


class OpenAICompatibleClient:
    def __init__(self, settings: Settings, http: httpx.AsyncClient | None = None):
        self._s = settings
        self._http = http or httpx.AsyncClient(
            base_url=settings.llm_base_url,
            timeout=httpx.Timeout(settings.llm_timeout_s, connect=5.0),
            limits=httpx.Limits(
                max_connections=settings.llm_max_concurrency * 2,
                max_keepalive_connections=settings.llm_max_concurrency,
            ),
            headers={"Authorization": f"Bearer {settings.llm_api_key.get_secret_value()}"},
        )
        self._sem = asyncio.Semaphore(settings.llm_max_concurrency)
        self._breakers: dict[str, CircuitBreaker] = {}

    def breaker_states(self) -> dict[str, str]:
        models = dict.fromkeys(self._s.llm_fast_models + self._s.llm_strong_models)
        return {m: self._breaker(m).state for m in models}

    def _breaker(self, model: str) -> CircuitBreaker:
        if model not in self._breakers:
            self._breakers[model] = CircuitBreaker(self._s.llm_breaker_threshold, self._s.llm_breaker_cooldown_s)
        return self._breakers[model]

    async def aclose(self) -> None:
        await self._http.aclose()

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        json_mode: bool = False,
        tier: Tier = "fast",
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        chain = self._s.llm_strong_models if tier == "strong" else self._s.llm_fast_models
        errors: list[str] = []
        for model in chain:
            breaker = self._breaker(model)
            if not breaker.allow():
                LLM_REQUESTS.labels(model, "breaker_open").inc()
                errors.append(f"{model}: circuit open")
                continue
            payload = self._payload(model, messages, tools, json_mode, temperature, max_tokens)
            try:
                response = await self._with_retries(model, payload)
            except LLMError as exc:
                breaker.record_failure()
                errors.append(f"{model}: {exc}")
                log_event(logger, "llm_model_failed", logging.WARNING, model=model, error=str(exc)[:300],
                          breaker=breaker.state)
                continue
            breaker.record_success()
            return response
        raise LLMUnavailableError("; ".join(errors) or "no models configured")

    def _payload(self, model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None,
                 json_mode: bool, temperature: float, max_tokens: int) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if self._s.llm_reasoning_effort and model.startswith("openai/gpt-oss"):
            payload["reasoning_effort"] = self._s.llm_reasoning_effort
        return payload

    async def _with_retries(self, model: str, payload: dict[str, Any]) -> LLMResponse:
        attempt = 0
        while True:
            try:
                return await self._once(model, payload)
            except _RetryableError as exc:
                attempt += 1
                if attempt > self._s.llm_max_retries:
                    raise
                delay = exc.retry_after or min(8.0, 0.5 * 2 ** (attempt - 1))
                await asyncio.sleep(delay + random.uniform(0, delay * 0.25))  # noqa: S311 (jitter, not crypto)

    async def _once(self, model: str, payload: dict[str, Any]) -> LLMResponse:
        start = time.perf_counter()
        try:
            async with self._sem:
                r = await self._http.post("/chat/completions", json=payload)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            LLM_REQUESTS.labels(model, "transport_error").inc()
            raise _RetryableError(f"transport: {type(exc).__name__}") from exc
        finally:
            LLM_LATENCY.labels(model).observe(time.perf_counter() - start)

        if r.status_code == 429 or r.status_code >= 500:
            LLM_REQUESTS.labels(model, f"http_{r.status_code}").inc()
            raise _RetryableError(f"http {r.status_code}", _retry_after(r))
        if r.status_code >= 400:
            # e.g. 400/404 for a decommissioned model: don't retry, fall through to the next model.
            LLM_REQUESTS.labels(model, f"http_{r.status_code}").inc()
            raise _FatalError(f"http {r.status_code}: {r.text[:300]}")

        try:
            data = r.json()
            message = data["choices"][0]["message"]
        except (ValueError, KeyError, IndexError) as exc:
            LLM_REQUESTS.labels(model, "bad_response").inc()
            raise _RetryableError("malformed response body") from exc

        LLM_REQUESTS.labels(model, "ok").inc()
        usage = data.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        LLM_TOKENS.labels(model, "prompt").inc(prompt_tokens)
        LLM_TOKENS.labels(model, "completion").inc(completion_tokens)

        tool_calls = []
        for raw in message.get("tool_calls") or []:
            fn = raw.get("function") or {}
            raw_args = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw_args)
                if not isinstance(args, dict):
                    raise ValueError("arguments must be an object")
            except ValueError:
                args = {"__invalid_json__": raw_args[:500]}
            tool_calls.append(ToolCall(id=raw.get("id", ""), name=fn.get("name", ""), arguments=args,
                                       raw_arguments=raw_args))
        return LLMResponse(
            content=message.get("content") or "",
            tool_calls=tool_calls,
            model=data.get("model", model),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
