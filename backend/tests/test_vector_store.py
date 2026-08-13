"""
pgvector-backed retrieval: indexing, unchanged-content skip, relevance ranking,
metadata filtering, and — most importantly — that a client can never retrieve another
client's data (PRD 9.5). Runs against the real embedded Postgres from conftest.py, with
deterministic fake embeddings (see FakeEmbeddingClient) so ranking is meaningful
without a live Google API key.
"""

import pytest

from app.vector_store import vector_store, PUBLIC_NAMESPACE, CLIENT_NAMESPACE


@pytest.mark.asyncio
async def test_index_and_search_public(database):
    chunks = [
        {"id": "c1", "text": "Web design for manufacturers",
         "metadata": {"service": "Web Design", "section": "service_overview"}},
        {"id": "c2", "text": "AI automation for business processes",
         "metadata": {"service": "AI & Automation", "section": "service_overview"}},
    ]
    indexed = await vector_store.index_documents(chunks, namespace=PUBLIC_NAMESPACE)
    assert indexed == 2

    results = await vector_store.search("web design", top_k=5)
    assert {r["id"] for r in results} == {"c1", "c2"}
    assert all(-1.0 <= r["score"] <= 1.0 for r in results)


@pytest.mark.asyncio
async def test_unchanged_content_is_not_re_embedded(database):
    chunks = [{"id": "c1", "text": "Same text", "metadata": {}}]
    first = await vector_store.index_documents(chunks, namespace=PUBLIC_NAMESPACE)
    second = await vector_store.index_documents(chunks, namespace=PUBLIC_NAMESPACE)
    assert first == 1
    assert second == 0  # nothing changed, so nothing re-embedded


@pytest.mark.asyncio
async def test_changed_content_is_re_embedded(database):
    await vector_store.index_documents(
        [{"id": "c1", "text": "Old text", "metadata": {}}], namespace=PUBLIC_NAMESPACE,
    )
    reindexed = await vector_store.index_documents(
        [{"id": "c1", "text": "New text", "metadata": {}}], namespace=PUBLIC_NAMESPACE,
    )
    assert reindexed == 1

    results = await vector_store.search("New text", top_k=1)
    assert results[0]["text"] == "New text"


@pytest.mark.asyncio
async def test_metadata_filter_narrows_results(database):
    chunks = [
        {"id": "a", "text": "overview one", "metadata": {"section": "service_overview"}},
        {"id": "b", "text": "overview two", "metadata": {"section": "service_overview"}},
        {"id": "c", "text": "problem one", "metadata": {"section": "problems_solved"}},
    ]
    await vector_store.index_documents(chunks, namespace=PUBLIC_NAMESPACE)

    filtered = await vector_store.search(
        "anything", top_k=10, filter_metadata={"section": "service_overview"}
    )
    assert {r["id"] for r in filtered} == {"a", "b"}


@pytest.mark.asyncio
async def test_search_on_empty_namespace_returns_nothing(database):
    assert await vector_store.search("anything", top_k=5) == []


@pytest.mark.asyncio
async def test_reindex_replaces_all_documents(database):
    await vector_store.index_documents(
        [{"id": "old", "text": "old doc", "metadata": {}}], namespace=PUBLIC_NAMESPACE,
    )
    count = await vector_store.reindex(
        [{"id": "new", "text": "new doc", "metadata": {}}], namespace=PUBLIC_NAMESPACE,
    )
    assert count == 1

    results = await vector_store.search("doc", top_k=10)
    assert {r["id"] for r in results} == {"new"}


# --- Client isolation (PRD 9.5) ---

@pytest.mark.asyncio
async def test_client_only_sees_their_own_documents(database):
    await vector_store.index_documents(
        [{"id": "ca", "text": "Client A project status", "metadata": {"client_id": "client-a"}}],
        namespace=CLIENT_NAMESPACE,
    )
    await vector_store.index_documents(
        [{"id": "cb", "text": "Client B project status", "metadata": {"client_id": "client-b"}}],
        namespace=CLIENT_NAMESPACE,
    )

    a_results = await vector_store.search_client("project status", client_id="client-a", top_k=10)
    b_results = await vector_store.search_client("project status", client_id="client-b", top_k=10)

    assert [r["id"] for r in a_results] == ["ca"]
    assert [r["id"] for r in b_results] == ["cb"]


@pytest.mark.asyncio
async def test_client_search_never_falls_back_to_public_index(database):
    """The public and client indexes are physically separate tables — confirm a
    client query can't accidentally surface public prospect content."""
    await vector_store.index_documents(
        [{"id": "pub", "text": "public marketing content", "metadata": {}}],
        namespace=PUBLIC_NAMESPACE,
    )
    results = await vector_store.search_client("marketing content", client_id="client-a", top_k=10)
    assert results == []


@pytest.mark.asyncio
async def test_delete_client_documents_only_affects_that_client(database):
    await vector_store.index_documents(
        [{"id": "ca", "text": "doc a", "metadata": {"client_id": "client-a"}}],
        namespace=CLIENT_NAMESPACE,
    )
    await vector_store.index_documents(
        [{"id": "cb", "text": "doc b", "metadata": {"client_id": "client-b"}}],
        namespace=CLIENT_NAMESPACE,
    )

    deleted = await vector_store.delete_client_documents("client-a")
    assert deleted == 1

    assert await vector_store.search_client("doc", client_id="client-a", top_k=10) == []
    remaining = await vector_store.search_client("doc", client_id="client-b", top_k=10)
    assert [r["id"] for r in remaining] == ["cb"]


@pytest.mark.asyncio
async def test_get_stats_reports_both_namespaces_separately(database):
    await vector_store.index_documents(
        [{"id": "p1", "text": "x", "metadata": {}}], namespace=PUBLIC_NAMESPACE,
    )
    await vector_store.index_documents(
        [{"id": "c1", "text": "y", "metadata": {"client_id": "client-a"}}],
        namespace=CLIENT_NAMESPACE,
    )

    stats = await vector_store.get_stats()
    assert stats["public"]["documents"] == 1
    assert stats["client"]["documents"] == 1
    assert stats["total_documents"] == 2
