.PHONY: install dev lint test test-all up down migrate seed token eval graph

install:
	pip install -e ".[dev]"

dev:  ## Run the API locally with in-memory data and no auth
	CA_DATA_BACKEND=memory CA_AUTH_DISABLED=true uvicorn careagents.api.app:create_app --factory --reload

lint:
	ruff check src tests eval && mypy src

test:
	pytest -q

test-all:  ## Includes Postgres/Redis integration tests (needs `make up`)
	CA_TEST_DATABASE_URL=postgresql://careagents:careagents@localhost:5432/careagents \
	CA_TEST_REDIS_URL=redis://localhost:6379/0 pytest -q

up:
	docker compose up --build -d

down:
	docker compose down

migrate:
	python -m careagents.migrate

seed:
	python -m careagents.seed

token:  ## make token ROLE=member MEMBER=SARAH-1001
	@python -m careagents.devtools --role $(or $(ROLE),member) $(if $(MEMBER),--member-id $(MEMBER),)

eval:
	python eval/run_eval.py

graph:  ## Regenerate docs/graph.mmd from the compiled LangGraph workflow
	@PYTHONPATH=src:tests python -c "import asyncio; from pathlib import Path; from careagents.config import Settings; \
	from careagents.container import build_container; from fakes import ScriptedLLM; \
	c = asyncio.run(build_container(Settings(env='test', documents_dir=Path('data/documents'), log_level='ERROR'), llm=ScriptedLLM(), with_queue=False)); \
	print(c.runtime.mermaid())" | sed 's|<p>||g; s|</p>||g' > docs/graph.mmd
