"""
Abacus Digital Chatbot - Vector Store
pgvector-backed RAG retrieval, in the same Postgres database as everything else.

Two physically separate tables enforce PRD 9.5: the public prospect index and the
authenticated client index are never queried in the same call. A retrieval is always
bound to one namespace, and the client namespace additionally filters on a real,
indexed, NOT NULL client_id column — not just a key inside a metadata blob — so one
client can never retrieve another's data even if a filter were accidentally omitted
somewhere upstream.

Public methods are now async (asyncpg is asyncio-native), unlike the previous
Chroma-backed version which needed `run_in_executor` to keep local CPU-bound embedding
inference off the event loop. Embeddings now come from an HTTP API call (see
embeddings.py), which is genuinely async I/O — callers no longer need an executor at all.
"""

import json
import logging
from typing import List, Dict, Any, Optional

import asyncpg

from .config import settings
from .database import db
from .embeddings import embedding_client

logger = logging.getLogger(__name__)

PUBLIC_NAMESPACE = "public"
CLIENT_NAMESPACE = "client"

# Small enough that a rate-limit failure partway through a large re-index only costs
# this many chunks of progress, not the whole job.
INDEX_SUB_BATCH_SIZE = 25

_TABLE_NAMES = {
    PUBLIC_NAMESPACE: "public_documents",
    CLIENT_NAMESPACE: "client_documents",
}


def _to_vector_literal(embedding: List[float]) -> str:
    """asyncpg has no built-in pgvector codec; a bracketed literal + ::vector cast works."""
    return "[" + ",".join(repr(float(x)) for x in embedding) + "]"


def _clean_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Drop None values and coerce anything non-JSON-scalar to a string."""
    clean = {}
    for k, v in (metadata or {}).items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
        else:
            clean[k] = str(v)
    return clean


class VectorStore:
    """pgvector-backed retrieval for both the public and client knowledge bases."""

    def __init__(self):
        self._ready = False

    async def ensure_schema(self):
        """Create the pgvector extension and document tables if they don't exist."""
        if self._ready:
            return
        pool = db._pool
        dims = settings.embedding_dimensions

        await pool.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await pool.execute(f"""
            CREATE TABLE IF NOT EXISTS public_documents (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                embedding vector({dims})
            )
        """)
        await pool.execute(f"""
            CREATE TABLE IF NOT EXISTS client_documents (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                embedding vector({dims}),
                client_id TEXT NOT NULL
            )
        """)
        await pool.execute(
            "CREATE INDEX IF NOT EXISTS idx_client_documents_client_id ON client_documents(client_id)"
        )
        self._ready = True

    async def index_documents(
        self,
        chunks: List[Dict[str, Any]],
        namespace: str = PUBLIC_NAMESPACE,
        force: bool = False,
    ) -> int:
        """
        Embed and upsert knowledge base chunks.

        Each chunk needs: id, text, metadata (client chunks also need
        metadata["client_id"]). Chunks whose text is unchanged since the last index are
        skipped so a scheduled re-index only pays to embed what actually moved.

        Embedded and written in small sub-batches rather than all at once: the free-tier
        embedding API rate-limits under a large job, and a failure partway through a
        single big batch would otherwise discard everything already embedded in that
        call. Writing incrementally means a call that runs out of time (or hits a
        run of 429s) still commits whatever it got done, and a retry naturally picks up
        the remainder via the unchanged-text skip above — no separate resume state needed.

        Returns the number of documents actually embedded (may be less than len(chunks)
        if a later sub-batch fails after earlier ones succeeded; the exception still
        propagates once nothing more can be written, but already-committed rows stay).
        """
        await self.ensure_schema()
        if not chunks:
            return 0

        table = _TABLE_NAMES[namespace]
        pool = db._pool

        to_index = chunks
        if not force:
            existing_rows = await pool.fetch(f"SELECT id, text FROM {table}")
            existing_text = {r["id"]: r["text"] for r in existing_rows}
            to_index = [c for c in chunks if existing_text.get(c["id"]) != c["text"]]
            if not to_index:
                logger.info(f"[{namespace}] No changed documents to index")
                return 0

        total_indexed = 0
        for i in range(0, len(to_index), INDEX_SUB_BATCH_SIZE):
            sub = to_index[i:i + INDEX_SUB_BATCH_SIZE]
            texts = [c["text"] for c in sub]
            embeddings = await embedding_client.embed(texts, task_type="document")
            await self._write_documents(table, namespace, sub, embeddings, pool)
            total_indexed += len(sub)

        logger.info(f"[{namespace}] Indexed {total_indexed} documents")
        return total_indexed

    async def _write_documents(
        self,
        table: str,
        namespace: str,
        docs: List[Dict[str, Any]],
        embeddings: List[List[float]],
        pool,
    ) -> None:
        if namespace == CLIENT_NAMESPACE:
            await pool.executemany(
                f"""INSERT INTO {table} (id, text, metadata, embedding, client_id)
                    VALUES ($1, $2, $3::jsonb, $4::vector, $5)
                    ON CONFLICT (id) DO UPDATE SET
                      text = EXCLUDED.text, metadata = EXCLUDED.metadata,
                      embedding = EXCLUDED.embedding, client_id = EXCLUDED.client_id""",
                [
                    (
                        c["id"], c["text"], json.dumps(_clean_metadata(c["metadata"])),
                        _to_vector_literal(vec), _clean_metadata(c["metadata"]).get("client_id", ""),
                    )
                    for c, vec in zip(docs, embeddings)
                ],
            )
        else:
            await pool.executemany(
                f"""INSERT INTO {table} (id, text, metadata, embedding)
                    VALUES ($1, $2, $3::jsonb, $4::vector)
                    ON CONFLICT (id) DO UPDATE SET
                      text = EXCLUDED.text, metadata = EXCLUDED.metadata, embedding = EXCLUDED.embedding""",
                [
                    (c["id"], c["text"], json.dumps(_clean_metadata(c["metadata"])), _to_vector_literal(vec))
                    for c, vec in zip(docs, embeddings)
                ],
            )

    async def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        namespace: str = PUBLIC_NAMESPACE,
    ) -> List[Dict[str, Any]]:
        """
        Search one namespace for relevant documents.

        Returns list of {"id", "text", "metadata", "score"} ordered by relevance
        (cosine similarity, higher = more relevant — same convention as before).
        """
        await self.ensure_schema()
        table = _TABLE_NAMES[namespace]
        pool = db._pool

        count = await pool.fetchval(f"SELECT COUNT(*) FROM {table}")
        if not count:
            logger.warning(f"[{namespace}] Vector store is empty, no documents to search")
            return []

        query_vec = _to_vector_literal((await embedding_client.embed([query], task_type="query"))[0])

        if filter_metadata:
            rows = await pool.fetch(
                f"""SELECT id, text, metadata, 1 - (embedding <=> $1::vector) AS score
                    FROM {table}
                    WHERE metadata @> $2::jsonb
                    ORDER BY embedding <=> $1::vector
                    LIMIT $3""",
                query_vec, json.dumps(filter_metadata), min(top_k, count),
            )
        else:
            rows = await pool.fetch(
                f"""SELECT id, text, metadata, 1 - (embedding <=> $1::vector) AS score
                    FROM {table}
                    ORDER BY embedding <=> $1::vector
                    LIMIT $2""",
                query_vec, min(top_k, count),
            )

        return [
            {
                "id": row["id"],
                "text": row["text"],
                "metadata": json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"],
                "score": float(row["score"]),
            }
            for row in rows
        ]

    async def search_client(
        self,
        query: str,
        client_id: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search the client namespace, hard-scoped to a single client_id via a real,
        indexed column — not a metadata-key filter that could silently no-op.

        This is the only supported way to read the client index.
        """
        if not client_id:
            raise ValueError("client_id is required for client knowledge base retrieval")

        await self.ensure_schema()
        pool = db._pool

        count = await pool.fetchval(
            "SELECT COUNT(*) FROM client_documents WHERE client_id = $1", client_id
        )
        if not count:
            return []

        query_vec = _to_vector_literal((await embedding_client.embed([query], task_type="query"))[0])
        rows = await pool.fetch(
            """SELECT id, text, metadata, 1 - (embedding <=> $1::vector) AS score
               FROM client_documents
               WHERE client_id = $2
               ORDER BY embedding <=> $1::vector
               LIMIT $3""",
            query_vec, client_id, min(top_k, count),
        )
        return [
            {
                "id": row["id"],
                "text": row["text"],
                "metadata": json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"],
                "score": float(row["score"]),
            }
            for row in rows
        ]

    async def reindex(self, chunks: List[Dict[str, Any]], namespace: str = PUBLIC_NAMESPACE) -> int:
        """
        Reindex a namespace against a fresh chunk set: drop any stored document whose id
        isn't in the new set (stale/removed pages), then index the rest incrementally.

        Unlike a wipe-and-force-embed-everything, this stays resumable under the
        embedding API's rate limits — unchanged chunks are skipped on each retry instead
        of the whole namespace being re-embedded from scratch every time.
        """
        await self.ensure_schema()
        table = _TABLE_NAMES[namespace]
        pool = db._pool

        current_ids = [c["id"] for c in chunks]
        if current_ids:
            await pool.execute(f"DELETE FROM {table} WHERE id != ALL($1::text[])", current_ids)
        else:
            await pool.execute(f"DELETE FROM {table}")

        count = await self.index_documents(chunks, namespace=namespace, force=False)
        logger.info(f"[{namespace}] Reindexed ({count} embedded, stale entries pruned)")
        return count

    async def delete_client_documents(self, client_id: str) -> int:
        """Remove every client-index document belonging to one client."""
        await self.ensure_schema()
        status = await db._pool.execute(
            "DELETE FROM client_documents WHERE client_id = $1", client_id
        )
        # asyncpg execute() returns a tag like "DELETE 3"
        try:
            return int(status.split()[-1])
        except (ValueError, IndexError):
            return 0

    async def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        await self.ensure_schema()
        pool = db._pool
        stats: Dict[str, Any] = {}
        for namespace, table in _TABLE_NAMES.items():
            try:
                count = await pool.fetchval(f"SELECT COUNT(*) FROM {table}")
                stats[namespace] = {"table_name": table, "documents": count}
            except Exception as e:  # pragma: no cover - only on a broken schema
                stats[namespace] = {"table_name": table, "error": str(e)}
        stats["total_documents"] = sum(
            v.get("documents", 0) for v in stats.values() if isinstance(v, dict)
        )
        return stats


# Singleton
vector_store = VectorStore()
