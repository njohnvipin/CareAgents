# syntax=docker/dockerfile:1.7
ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /src
COPY pyproject.toml README.md ./
COPY src ./src
# EXTRAS="[embeddings]" adds sentence-transformers (CPU torch wheels keep the image smaller).
ARG EXTRAS=""
RUN python -m venv /venv \
 && /venv/bin/pip install --upgrade pip \
 && /venv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cpu ".${EXTRAS}"
# Bake the embedding model into the image: pods never download weights at runtime.
ARG EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RUN mkdir -p /models && if [ -n "$EXTRAS" ]; then \
      HF_HOME=/models /venv/bin/python -c "from sentence_transformers import SentenceTransformer as S; S('${EMBEDDING_MODEL}')"; \
    fi

FROM python:${PYTHON_VERSION}-slim
RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin app
COPY --from=build /venv /venv
COPY --from=build /models /models
COPY data/documents /app/data/documents
WORKDIR /app
ENV PATH=/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/models HF_HUB_OFFLINE=1 \
    CA_DOCUMENTS_DIR=/app/data/documents
USER 10001
EXPOSE 8000
# One uvicorn process per container; scale with replicas, not workers (keeps metrics and breakers simple).
CMD ["uvicorn", "careagents.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", \
     "--proxy-headers", "--timeout-graceful-shutdown", "30", "--no-server-header"]
