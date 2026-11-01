import httpx
import pytest

from careagents.config import Settings
from careagents.llm.client import LLMUnavailableError, OpenAICompatibleClient


def ok_body(model, content="hi"):
    return {"model": model, "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2}}


def make(handler, **overrides):
    s = Settings(env="test", llm_api_key="k", llm_fast_models=["m1", "m2"], llm_max_retries=2,
                 llm_breaker_threshold=2, llm_breaker_cooldown_s=60, **overrides)
    http = httpx.AsyncClient(base_url="http://llm", transport=httpx.MockTransport(handler))
    return OpenAICompatibleClient(s, http=http)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    async def instant(_):
        return None
    monkeypatch.setattr("careagents.llm.client.asyncio.sleep", instant)


async def test_retries_transient_errors():
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(503) if len(calls) < 3 else httpx.Response(200, json=ok_body("m1"))

    r = await make(handler).chat([{"role": "user", "content": "x"}])
    assert r.content == "hi" and len(calls) == 3


async def test_falls_back_when_model_is_decommissioned():
    seen = []

    def handler(req):
        model = __import__("json").loads(req.content)["model"]
        seen.append(model)
        if model == "m1":
            return httpx.Response(404, json={"error": {"code": "model_decommissioned"}})
        return httpx.Response(200, json=ok_body(model, "from m2"))

    r = await make(handler).chat([{"role": "user", "content": "x"}])
    assert r.content == "from m2" and seen == ["m1", "m2"]  # 404 is not retried


async def test_breaker_skips_dead_model():
    seen = []

    def handler(req):
        model = __import__("json").loads(req.content)["model"]
        seen.append(model)
        return httpx.Response(400) if model == "m1" else httpx.Response(200, json=ok_body(model))

    client = make(handler)
    for _ in range(3):
        await client.chat([{"role": "user", "content": "x"}])
    assert seen.count("m1") == 2  # breaker opened after threshold=2
    assert client.breaker_states()["m1"] == "open"


async def test_all_models_down_raises():
    client = make(lambda req: httpx.Response(500))
    with pytest.raises(LLMUnavailableError):
        await client.chat([{"role": "user", "content": "x"}])


async def test_parses_tool_calls_and_bad_json():
    body = {"model": "m1", "choices": [{"message": {"content": None, "tool_calls": [
        {"id": "a", "function": {"name": "list_claims", "arguments": "{}"}},
        {"id": "b", "function": {"name": "get_claim", "arguments": "{not json"}}]}}]}
    r = await make(lambda req: httpx.Response(200, json=body)).chat([{"role": "user", "content": "x"}])
    assert r.tool_calls[0].arguments == {} and "__invalid_json__" in r.tool_calls[1].arguments


async def test_payload_shape():
    captured = {}

    def handler(req):
        captured.update(__import__("json").loads(req.content))
        return httpx.Response(200, json=ok_body("m1"))

    client = make(handler, llm_reasoning_effort="low")
    await client.chat([{"role": "user", "content": "x"}], json_mode=True,
                      tools=[{"type": "function", "function": {"name": "t", "parameters": {}}}])
    assert captured["response_format"] == {"type": "json_object"} and captured["tool_choice"] == "auto"
    assert "reasoning_effort" not in captured  # only sent to gpt-oss models
