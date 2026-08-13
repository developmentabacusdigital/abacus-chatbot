"""
Shared fixtures.

Tests run against a real, throwaway embedded Postgres instance (via `pgserver`) rather
than mocks — the migration from SQLite to Postgres changed enough SQL (upsert syntax,
placeholders, boolean handling) that faking the DB layer would defeat the point.

One Postgres instance is started for the whole test session (spinning one up per test
would be far too slow); the `database` fixture truncates every table before each test
for isolation instead of reconnecting. Because the app's `db` and `vector_store`
singletons are mutated in place rather than replaced, every module that already does
`from .database import db` sees the test instance automatically — no per-test
monkeypatching required.

Tests never call OpenRouter; LLM calls go through FakeRouter below.
"""

import os
import shutil
import sys
import tempfile

import pytest
import pytest_asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings  # noqa: E402
from app.database import db as real_db  # noqa: E402
from app.vector_store import vector_store as real_vector_store  # noqa: E402
from app.embeddings import embedding_client as real_embedding_client  # noqa: E402

TABLES = [
    "messages", "leads", "briefs", "emails", "escalations", "client_projects",
    "magic_links", "client_sessions", "clients", "sessions",
    "public_documents", "client_documents",
]

# Small, fast, and matches FakeEmbeddingClient's output — must be set before
# ensure_schema() creates the pgvector column, which fixes its dimension at CREATE time.
settings.embedding_dimensions = 8


class FakeEmbeddingClient:
    """Deterministic pseudo-embeddings — same text always maps to the same vector,
    so relevance ordering in vector-store tests is meaningful without a real API key."""

    configured = True

    async def embed(self, texts, task_type="document"):
        return [self._vector(t) for t in texts]

    async def embed_one(self, text, task_type="query"):
        return self._vector(text)

    @staticmethod
    def _vector(text, dims=8):
        import hashlib
        h = hashlib.sha256(text.encode()).digest()
        return [(h[i % len(h)] - 128) / 128.0 for i in range(dims)]


@pytest_asyncio.fixture(scope="session")
async def _test_postgres():
    """One embedded Postgres + schema, shared across the whole test session."""
    import pgserver

    # embedding_client is a plain module-level singleton (no dependency injection),
    # so swapping its methods here — same trick as mutating the db/vector_store
    # singletons below — makes every caller use the fake without a real API key.
    fake = FakeEmbeddingClient()
    real_embedding_client.embed = fake.embed
    real_embedding_client.embed_one = fake.embed_one

    data_dir = tempfile.mkdtemp(prefix="abacus_test_pg_")
    server = pgserver.get_server(data_dir)

    real_db.database_url = server.get_uri()
    await real_db.connect()
    await real_vector_store.ensure_schema()

    yield real_db

    await real_db.disconnect()
    server.cleanup()
    shutil.rmtree(data_dir, ignore_errors=True)


@pytest_asyncio.fixture
async def database(_test_postgres):
    """Truncate every table before each test so tests never see each other's data."""
    await _test_postgres._pool.execute(
        f"TRUNCATE {', '.join(TABLES)} RESTART IDENTITY CASCADE"
    )
    yield _test_postgres


class FakeRouter:
    """Stands in for the LLM router: returns queued payloads, records calls."""

    def __init__(self, json_responses=None, text_responses=None):
        self.json_responses = list(json_responses or [])
        self.text_responses = list(text_responses or [])
        self.calls = []
        self.configured = True

    async def generate_json(self, messages, task_type="general", **kwargs):
        self.calls.append((task_type, messages))
        data = self.json_responses.pop(0) if self.json_responses else None
        return {"data": data, "model_used": f"fake/{task_type}", "cost": 0.0001, "raw": ""}

    async def generate(self, messages, task_type="general", **kwargs):
        self.calls.append((task_type, messages))
        content = self.text_responses.pop(0) if self.text_responses else "fake reply"
        return {
            "content": content,
            "model_used": f"fake/{task_type}",
            "cost": 0.0001,
            "usage": {},
            "ok": True,
        }
