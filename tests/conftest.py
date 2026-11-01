from pathlib import Path

import pytest
from fakes import ScriptedLLM

from careagents.config import Settings
from careagents.container import build_container

JWT_SECRET = "test-secret-that-is-long-enough-for-hs256-0123456789"
DOCS = Path(__file__).resolve().parents[1] / "data" / "documents"


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(env="test", data_backend="memory", redis_url=None, jwt_secret=JWT_SECRET,
                    documents_dir=DOCS, request_deadline_s=10, rate_limit_per_minute=1000,
                    llm_verifier_enabled=True, human_review_enabled=False)


@pytest.fixture
def make_container(settings):
    created = []

    async def _make(llm=None, **overrides):
        s = settings.model_copy(update=overrides) if overrides else settings
        c = await build_container(s, llm=llm or ScriptedLLM(), with_queue=False)
        created.append(c)
        return c

    yield _make
