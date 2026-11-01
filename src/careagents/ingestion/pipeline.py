"""Idempotent document ingestion.

Layout: <documents_dir>/<doc_type>/<file>.md, where doc_type is "policy" or "authorization".
Chunks follow markdown headings, so a chunk never splits a coverage rule from its heading.
Unchanged files are skipped via content hashes, so re-running is cheap and never duplicates
(the prototype appended a fresh copy of every chunk on each run).
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

from careagents.data.embeddings import Embedder
from careagents.data.models import ChunkIn, DocumentStore
from careagents.data.search import DocumentSearch
from careagents.logging_setup import log_event

logger = logging.getLogger(__name__)
DOC_TYPES = ("policy", "authorization")
HEADING = re.compile(r"^(#{1,3})\s+(.*)$", re.M)


def chunk_markdown(text: str, max_chars: int = 1200, overlap: int = 150) -> list[str]:
    title_match = re.search(r"^#\s+(.*)$", text, re.M)
    title = title_match.group(1).strip() if title_match else ""
    positions = [m.start() for m in HEADING.finditer(text)] or [0]
    if positions[0] != 0:
        positions.insert(0, 0)
    sections = [text[a:b].strip() for a, b in zip(positions, positions[1:] + [len(text)], strict=False)]
    chunks: list[str] = []
    for section in filter(None, sections):
        lines = section.splitlines()
        if not "".join(lines[1:]).strip():
            continue  # heading with no body (e.g. the document title): pure noise for retrieval
        heading = lines[0].lstrip("# ").strip()
        prefix = f"{title} > {heading}\n" if title and heading != title else ""
        if len(section) <= max_chars:
            chunks.append(prefix + section)
            continue
        start = 0
        while start < len(section):
            chunks.append(prefix + section[start:start + max_chars])
            start += max_chars - overlap
    return chunks


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


async def ingest_directory(root: Path, store: DocumentStore, embedder: Embedder,
                           search: DocumentSearch | None = None) -> dict[str, int]:
    stats = {"files": 0, "skipped_unchanged": 0, "chunks_written": 0}
    for doc_type in DOC_TYPES:
        folder = root / doc_type
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.md")):
            stats["files"] += 1
            source = f"{doc_type}/{path.name}"
            chunks = chunk_markdown(path.read_text(encoding="utf-8"))
            hashes = [_hash(c) for c in chunks]
            existing = await store.existing_hashes(source)
            if existing == dict(enumerate(hashes)):
                stats["skipped_unchanged"] += 1
                continue
            vectors: list[list[float]] = []
            for i in range(0, len(chunks), 64):
                vectors.extend(await embedder.embed(chunks[i:i + 64]))
            stats["chunks_written"] += await store.replace_chunks(source, doc_type, [
                ChunkIn(index=i, content=c, content_hash=h, embedding=v)
                for i, (c, h, v) in enumerate(zip(chunks, hashes, vectors, strict=True))])
    if search and stats["chunks_written"]:
        await search.invalidate()
    log_event(logger, "ingestion_completed", **stats)
    return stats
