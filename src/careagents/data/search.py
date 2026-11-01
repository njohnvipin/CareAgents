"""Document retrieval with a versioned cache.

Ingestion bumps the version, which invalidates every cached result at once without a scan.
"""

from __future__ import annotations

import hashlib
import json

from careagents.data.cache import Cache
from careagents.data.embeddings import Embedder
from careagents.data.models import DocumentStore, SearchHit

VERSION_KEY = "ca:docs:version"


class DocumentSearch:
    def __init__(self, embedder: Embedder, store: DocumentStore, cache: Cache | None = None, ttl_s: int = 3600):
        self.embedder = embedder
        self.store = store
        self.cache = cache
        self.ttl_s = ttl_s

    async def search(self, query: str, doc_type: str, top_k: int = 4) -> list[SearchHit]:
        key = None
        if self.cache:
            version = await self.cache.get(VERSION_KEY) or "0"
            digest = hashlib.sha256(f"{doc_type}|{top_k}|{query.strip().lower()}".encode()).hexdigest()[:32]
            key = f"ca:search:{version}:{digest}"
            cached = await self.cache.get(key)
            if cached:
                return [SearchHit(**h) for h in json.loads(cached)]
        [embedding] = await self.embedder.embed([query])
        hits = await self.store.search(embedding, doc_type, top_k)
        if self.cache and key:
            await self.cache.set(key, json.dumps([h.model_dump() for h in hits]), self.ttl_s)
        return hits

    async def invalidate(self) -> None:
        if self.cache:
            await self.cache.incr(VERSION_KEY)
