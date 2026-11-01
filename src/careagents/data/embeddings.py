"""Embedding backends behind one interface."""

from __future__ import annotations

import asyncio
import hashlib
import math
from typing import Any, Protocol

from careagents.config import Settings
from careagents.data.models import tokenize


class Embedder(Protocol):
    dim: int

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashingEmbedder:
    """Deterministic feature-hashing embedder (unigrams + bigrams).

    Lexical, not semantic. It exists so dev and CI run without downloading a model.
    Config validation refuses it in staging/prod.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = sorted(tokenize(text))
        grams = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:], strict=False)]
        for g in grams:
            h = int(hashlib.md5(g.encode(), usedforsecurity=False).hexdigest(), 16)
            vec[h % self.dim] += 1.0 if (h >> 8) & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]


class SentenceTransformerEmbedder:
    """Local sentence-transformers model, loaded lazily and run off the event loop."""

    def __init__(self, model_name: str, dim: int):
        self.model_name = model_name
        self.dim = dim
        self._model: Any = None
        self._lock = asyncio.Lock()

    async def _load(self) -> Any:
        async with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer  # optional extra

                self._model = await asyncio.to_thread(SentenceTransformer, self.model_name)
        return self._model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        model = await self._load()
        vectors = await asyncio.to_thread(model.encode, texts, normalize_embeddings=True, batch_size=32)
        out = [list(map(float, v)) for v in vectors]
        if out and len(out[0]) != self.dim:
            raise RuntimeError(f"embedding dim {len(out[0])} != configured {self.dim}; update migration + config")
        return out


def make_embedder(settings: Settings) -> Embedder:
    if settings.embeddings_backend == "sentence-transformers":
        return SentenceTransformerEmbedder(settings.embedding_model, settings.embedding_dim)
    return HashingEmbedder(settings.embedding_dim)
