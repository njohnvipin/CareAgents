"""Tool framework.

Security properties enforced here, not in prompts:
  * member_id comes from the authenticated request via ToolContext. It is never a tool argument,
    and argument models forbid extra fields, so the model cannot ask for another member's data.
  * Every call that touches member data is written to the audit log before it executes.
  * Tool failures become structured errors the model can see; exceptions never reach the user.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from careagents.agents.state import Evidence, Fact
from careagents.data.models import AuditLogger, Repositories
from careagents.data.search import DocumentSearch
from careagents.metrics import TOOL_CALLS

logger = logging.getLogger(__name__)


class Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


@dataclass
class ToolContext:
    member_id: str
    actor_id: str
    request_id: str | None
    repos: Repositories
    search: DocumentSearch
    audit: AuditLogger


@dataclass
class ToolResult:
    data: dict[str, Any]
    facts: list[Fact] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)

    def to_message(self) -> str:
        return json.dumps(self.data, default=str)[:8000]


Handler = Callable[[ToolContext, Any], Awaitable[ToolResult]]


class Tool:
    def __init__(self, name: str, description: str, args_model: type[Args], handler: Handler,
                 touches_member_data: bool = True):
        self.name = name
        self.description = description
        self.args_model = args_model
        self.handler = handler
        self.touches_member_data = touches_member_data

    def spec(self) -> dict[str, Any]:
        schema = self.args_model.model_json_schema()
        schema.pop("title", None)
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                                                 "parameters": schema}}

    async def run(self, ctx: ToolContext, raw_args: dict[str, Any]) -> ToolResult:
        try:
            args = self.args_model.model_validate(raw_args)
        except ValidationError as exc:
            TOOL_CALLS.labels(self.name, "invalid_args").inc()
            errors = [{"loc": e["loc"], "msg": e["msg"]} for e in exc.errors()]
            return ToolResult(data={"error": "invalid_arguments", "details": errors})
        if self.touches_member_data:
            await ctx.audit.record(actor_id=ctx.actor_id, member_id=ctx.member_id, action=f"tool:{self.name}",
                                   request_id=ctx.request_id)
        try:
            result = await self.handler(ctx, args)
        except Exception:
            logger.exception("tool_failed", extra={"fields": {"tool": self.name}})
            TOOL_CALLS.labels(self.name, "error").inc()
            return ToolResult(data={"error": "tool_failed", "message": "This lookup is temporarily unavailable."})
        TOOL_CALLS.labels(self.name, "ok").inc()
        return result
