"""Every LLM call goes through here so usage is attributed to the case."""

from typing import Any

from careagents.agents.state import CaseState
from careagents.llm.client import LLMClient, LLMResponse


async def call_llm(llm: LLMClient, state: CaseState, **kwargs: Any) -> LLMResponse:
    resp = await llm.chat(**kwargs)
    state.usage.llm_calls += 1
    state.usage.prompt_tokens += resp.prompt_tokens
    state.usage.completion_tokens += resp.completion_tokens
    return resp
