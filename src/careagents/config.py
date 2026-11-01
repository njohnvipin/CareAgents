"""Typed configuration. Every setting comes from the environment with the CA_ prefix.

Lists are given as JSON, e.g. CA_LLM_FAST_MODELS='["openai/gpt-oss-20b","openai/gpt-oss-120b"]'.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CA_", env_file=".env", extra="ignore")

    env: Literal["dev", "test", "staging", "prod"] = "dev"
    service_name: str = "careagents"
    log_level: str = "INFO"

    # --- LLM (any OpenAI-compatible endpoint; Groq by default) ---
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_api_key: SecretStr = SecretStr("")
    # Ordered fallback chains. The first healthy model wins.
    llm_fast_models: list[str] = ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]
    llm_strong_models: list[str] = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
    llm_reasoning_effort: Literal["low", "medium", "high"] | None = "low"
    llm_timeout_s: float = 30.0
    llm_max_retries: int = 3
    llm_max_concurrency: int = 32  # per process; size against provider rate limits
    llm_breaker_threshold: int = 5
    llm_breaker_cooldown_s: float = 30.0

    # --- Agent runtime ---
    request_deadline_s: float = 45.0
    max_tool_rounds: int = 4
    max_verify_iterations: int = 2
    llm_verifier_enabled: bool = True
    # Send cases that fail verification, have conflicting records, or are complaints to a staff
    # review queue (LangGraph interrupt) instead of auto-answering with records only.
    human_review_enabled: bool = True

    # --- Data ---
    data_backend: Literal["memory", "postgres"] = "memory"
    database_url: str = "postgresql://careagents:careagents@localhost:5432/careagents"
    db_pool_min: int = 2
    db_pool_max: int = 10
    redis_url: str | None = None
    documents_dir: Path = Path("data/documents")
    embeddings_backend: Literal["hashing", "sentence-transformers"] = "hashing"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384  # must match the vector(...) column in migrations
    search_cache_ttl_s: int = 3600

    # --- Auth ---
    auth_disabled: bool = False  # dev only; refused in staging/prod
    dev_member_id: str = "SARAH-1001"
    jwt_secret: SecretStr | None = None  # HS256 (dev / internal)
    jwt_public_key: str | None = None  # RS256/ES256 PEM from your IdP (prod)
    jwt_algorithms: list[str] = ["HS256"]
    jwt_audience: str = "careagents"
    jwt_issuer: str | None = None

    # --- Edge ---
    rate_limit_per_minute: int = 20
    cors_origins: list[str] = []

    # --- Tracing ---
    otel_enabled: bool = False

    @model_validator(mode="after")
    def _guard_production(self) -> "Settings":
        if self.env not in ("staging", "prod"):
            return self
        problems = []
        if self.auth_disabled:
            problems.append("CA_AUTH_DISABLED must be false")
        if self.data_backend != "postgres":
            problems.append("CA_DATA_BACKEND must be postgres")
        if self.embeddings_backend == "hashing":
            problems.append("CA_EMBEDDINGS_BACKEND=hashing is lexical only; use sentence-transformers")
        if not self.redis_url:
            problems.append("CA_REDIS_URL is required (rate limits, cache, job queue)")
        if not self.llm_api_key.get_secret_value():
            problems.append("CA_LLM_API_KEY is required")
        if not (self.jwt_secret or self.jwt_public_key):
            problems.append("CA_JWT_SECRET or CA_JWT_PUBLIC_KEY is required")
        if problems:
            raise ValueError(f"Unsafe {self.env} configuration: " + "; ".join(problems))
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
