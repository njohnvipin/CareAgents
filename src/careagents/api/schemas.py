from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    member_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9-]{3,40}$",
                                  description="Staff only: the member being helped.")

    @field_validator("question")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = " ".join(v.split())
        if not v:
            raise ValueError("question is empty")
        return v


class ReviewDecision(BaseModel):
    action: Literal["approve", "edit", "reject"]
    answer: str | None = Field(default=None, max_length=4000, description="Required for edit")
    note: str | None = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def _edit_needs_answer(self) -> "ReviewDecision":
        if self.action == "edit" and not (self.answer and self.answer.strip()):
            raise ValueError("edit requires an answer")
        return self


class IngestResponse(BaseModel):
    status: str
    job_id: str | None = None
    stats: dict[str, int] | None = None


class ErrorResponse(BaseModel):
    error: str
    request_id: str | None = None
