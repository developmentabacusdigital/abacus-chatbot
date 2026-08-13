"""
Abacus Digital Chatbot - Indexing Service
Builds the public knowledge base (static doc content + live site crawl) and the
access-controlled client knowledge base, plus the scheduled re-index loop (PRD 7.1).
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional

from .config import settings
from .knowledge_base import get_knowledge_chunks, build_client_chunks, save_knowledge_base
from .site_crawler import site_crawler
from .vector_store import vector_store, PUBLIC_NAMESPACE, CLIENT_NAMESPACE
from .database import db
from .embeddings import embedding_client, EmbeddingError

logger = logging.getLogger(__name__)


class Indexer:
    """Owns knowledge base construction and refresh."""

    def __init__(self):
        self.last_run: Optional[str] = None
        self.last_result: Dict[str, Any] = {}
        self._task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    async def index_public(self, crawl_site: bool = True, full: bool = False) -> Dict[str, Any]:
        """
        Index the public prospect knowledge base.

        Static chunks from the company doc are always included; live site content is
        layered on top when crawling is enabled. A crawl failure degrades to the static
        base rather than emptying the index.
        """
        async with self._lock:
            static_chunks = get_knowledge_chunks()
            site_chunks = []

            if crawl_site and settings.crawl_enabled:
                site_chunks = await site_crawler.crawl()
                if not site_chunks:
                    logger.warning("Crawl returned nothing; keeping static knowledge base only")

            chunks = static_chunks + site_chunks

            if not embedding_client.configured:
                logger.warning(
                    "GOOGLE_API_KEY is not set — skipping embedding. RAG answers won't "
                    "be grounded until it's configured."
                )
                self.last_result = {
                    "static_chunks": len(static_chunks),
                    "site_chunks": len(site_chunks),
                    "total_chunks": len(chunks),
                    "documents_embedded": 0,
                    "full_reindex": full,
                    "error": "embedding_provider_not_configured",
                    "completed_at": datetime.utcnow().isoformat(),
                }
                return self.last_result

            try:
                if full:
                    indexed = await vector_store.reindex(chunks, namespace=PUBLIC_NAMESPACE)
                else:
                    indexed = await vector_store.index_documents(chunks, namespace=PUBLIC_NAMESPACE)
            except EmbeddingError as e:
                logger.error(f"Public index refresh failed: {e}")
                self.last_result = {
                    "static_chunks": len(static_chunks),
                    "site_chunks": len(site_chunks),
                    "total_chunks": len(chunks),
                    "documents_embedded": 0,
                    "full_reindex": full,
                    "error": str(e),
                    "completed_at": datetime.utcnow().isoformat(),
                }
                return self.last_result

            self.last_run = datetime.utcnow().isoformat()
            self.last_result = {
                "static_chunks": len(static_chunks),
                "site_chunks": len(site_chunks),
                "total_chunks": len(chunks),
                "documents_embedded": indexed,
                "full_reindex": full,
                "completed_at": self.last_run,
            }
            logger.info(f"Public index refreshed: {self.last_result}")
            return self.last_result

    async def index_client(self, client_id: str) -> Dict[str, Any]:
        """(Re)index one client's project data into the client namespace."""
        client = await db.get_client(client_id)
        if not client:
            return {"error": "client not found", "documents_embedded": 0}

        projects = await db.get_client_projects(client_id)
        chunks = build_client_chunks(client, projects)

        if not embedding_client.configured:
            return {"client_id": client_id, "projects": len(projects),
                    "documents_embedded": 0, "error": "embedding_provider_not_configured"}

        try:
            # Replace this client's documents wholesale so deleted projects disappear
            await vector_store.delete_client_documents(client_id)
            indexed = await vector_store.index_documents(chunks, namespace=CLIENT_NAMESPACE, force=True)
        except EmbeddingError as e:
            logger.error(f"Client index refresh failed for {client_id}: {e}")
            return {"client_id": client_id, "projects": len(projects),
                    "documents_embedded": 0, "error": str(e)}

        return {
            "client_id": client_id,
            "projects": len(projects),
            "documents_embedded": indexed,
        }

    async def index_all_clients(self) -> Dict[str, Any]:
        """Reindex every client's data (used at startup and after bulk imports)."""
        projects = await db.get_all_projects()
        client_ids = {p.client_id for p in projects}
        total = 0
        for cid in client_ids:
            result = await self.index_client(cid)
            total += result.get("documents_embedded", 0)
        return {"clients": len(client_ids), "documents_embedded": total}

    async def bootstrap(self):
        """
        Startup indexing: static content immediately, crawl in the background.

        A missing/invalid embedding provider must never crash startup — the same
        graceful-degradation the app already applies to a missing OPENROUTER_API_KEY.
        RAG answers just won't be grounded until GOOGLE_API_KEY is configured.
        """
        save_knowledge_base()  # best-effort local snapshot; no-ops on a read-only fs

        if not embedding_client.configured:
            logger.warning(
                "GOOGLE_API_KEY is not set — starting without a knowledge base index. "
                "Get a free key at https://aistudio.google.com/apikey, then call "
                "/api/admin/reindex."
            )
            return

        try:
            static_chunks = get_knowledge_chunks()
            await vector_store.index_documents(static_chunks, namespace=PUBLIC_NAMESPACE)
            logger.info(f"Static knowledge base ready ({len(static_chunks)} chunks)")
            await self.index_all_clients()
        except EmbeddingError as e:
            logger.error(f"Startup indexing failed, continuing without it: {e}")

    async def _schedule_loop(self):
        """Periodic re-index so published content changes reach the bot (PRD 7.1)."""
        interval = max(1, settings.reindex_interval_hours) * 3600
        # Give the app a moment to finish booting before the first crawl
        await asyncio.sleep(30)
        while True:
            try:
                await self.index_public(crawl_site=True)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Scheduled re-index failed: {e}")
            await asyncio.sleep(interval)

    def start_scheduler(self):
        """
        In-process background loop, for local dev / any always-on host.

        On Vercel this is a no-op — a serverless function has no persistent process for
        an infinite asyncio loop to live in. Use the Vercel Cron job wired to
        /api/cron/reindex instead (see main.py); VERCEL is set automatically in that
        environment, which is what this checks.
        """
        if os.environ.get("VERCEL"):
            return
        if self._task or not settings.crawl_enabled:
            return
        self._task = asyncio.create_task(self._schedule_loop())
        logger.info(
            f"Re-index scheduler started (every {settings.reindex_interval_hours}h)"
        )

    async def stop_scheduler(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None


indexer = Indexer()
