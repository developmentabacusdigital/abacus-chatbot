"""
Abacus Digital Chatbot - Embeddings
API-based embedding generation for RAG (replaces local sentence-transformers).

Local inference (all-MiniLM-L6-v2 via sentence-transformers/torch) doesn't fit a
serverless deployment: the model + torch are far too large for typical function
size limits, and there's no persistent disk to cache them between cold starts.
This module calls Gemini's free-tier embedding endpoint instead, isolated behind
one interface so swapping providers later only means editing this file.
"""

import asyncio
import logging
from typing import List, Literal

import httpx

from .config import settings

logger = logging.getLogger(__name__)

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models"

# gemini-embedding-001 only supports the single-text embedContent method (no synchronous
# batchEmbedContents), so a "batch" here is just this many concurrent embedContent calls.
# Kept modest — the free tier's per-minute quota is easy to blow through with high
# concurrency, and a 429 here costs an entire indexing sub-batch's worth of retries.
MAX_BATCH_SIZE = 5

MAX_RETRIES = 5

TaskType = Literal["document", "query"]

_GEMINI_TASK_TYPE = {
    "document": "RETRIEVAL_DOCUMENT",
    "query": "RETRIEVAL_QUERY",
}


class EmbeddingError(RuntimeError):
    """Raised when the embedding provider can't be reached or returns something unusable."""


class EmbeddingClient:
    """Thin async client over the embedding provider, with local retry/batching."""

    def __init__(self):
        self._client: httpx.AsyncClient | None = None

    async def initialize(self):
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0))

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    @property
    def configured(self) -> bool:
        return bool(settings.google_api_key)

    async def embed(self, texts: List[str], task_type: TaskType = "document") -> List[List[float]]:
        """
        Embed a batch of texts. Order of the returned vectors matches `texts`.

        Raises EmbeddingError if the provider isn't configured or a batch fails after
        retrying — callers (indexing, retrieval) should treat this as fatal for that
        operation rather than silently returning zero vectors, which would corrupt
        similarity search results without any visible symptom.
        """
        if not texts:
            return []
        if not self.configured:
            raise EmbeddingError(
                "GOOGLE_API_KEY is not set; cannot generate embeddings. "
                "Get a free key at https://aistudio.google.com/apikey"
            )

        await self.initialize()

        vectors: List[List[float]] = []
        for i in range(0, len(texts), MAX_BATCH_SIZE):
            batch = texts[i:i + MAX_BATCH_SIZE]
            vectors.extend(await self._embed_batch(batch, task_type))
        return vectors

    async def embed_one(self, text: str, task_type: TaskType = "query") -> List[float]:
        result = await self.embed([text], task_type=task_type)
        return result[0]

    async def _embed_batch(self, texts: List[str], task_type: TaskType) -> List[List[float]]:
        semaphore = asyncio.Semaphore(MAX_BATCH_SIZE)
        results = await asyncio.gather(*(
            self._embed_single(text, task_type, semaphore) for text in texts
        ))
        return list(results)

    async def _embed_single(
        self, text: str, task_type: TaskType, semaphore: asyncio.Semaphore
    ) -> List[float]:
        model = settings.embedding_model
        url = f"{GEMINI_API_URL}/{model}:embedContent?key={settings.google_api_key}"
        gemini_task = _GEMINI_TASK_TYPE.get(task_type, "RETRIEVAL_DOCUMENT")

        payload = {
            "model": f"models/{model}",
            "content": {"parts": [{"text": text}]},
            "taskType": gemini_task,
            "outputDimensionality": settings.embedding_dimensions,
        }

        last_error: Exception | None = None
        async with semaphore:
            for attempt in range(MAX_RETRIES):
                try:
                    response = await self._client.post(url, json=payload)
                    if response.status_code == 429 and attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(self._retry_delay(response, attempt))
                        continue
                    response.raise_for_status()
                    data = response.json()
                    values = (data.get("embedding") or {}).get("values")
                    if not values:
                        raise EmbeddingError("Embedding API returned no vector")
                    return values
                except httpx.HTTPStatusError as e:
                    last_error = e
                    logger.error(f"Embedding API error: {e.response.status_code} - {e.response.text[:300]}")
                    if e.response.status_code != 429:
                        break
                except Exception as e:
                    last_error = e
                    logger.error(f"Embedding request failed: {e}")
                    break

        raise EmbeddingError(f"Failed to generate embeddings: {last_error}")

    @staticmethod
    def _retry_delay(response: httpx.Response, attempt: int) -> float:
        """Honor the API's Retry-After if it sent one; otherwise back off exponentially."""
        retry_after = response.headers.get("retry-after")
        if retry_after:
            try:
                return float(retry_after)
            except ValueError:
                pass
        return min(2.0 * (2 ** attempt), 30.0)


# Singleton
embedding_client = EmbeddingClient()
