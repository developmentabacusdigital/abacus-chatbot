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

# Gemini's batchEmbedContents caps requests per call; chunk larger jobs to stay under it.
MAX_BATCH_SIZE = 100

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
        model = settings.embedding_model
        url = f"{GEMINI_API_URL}/{model}:batchEmbedContents?key={settings.google_api_key}"
        gemini_task = _GEMINI_TASK_TYPE.get(task_type, "RETRIEVAL_DOCUMENT")

        payload = {
            "requests": [
                {
                    "model": f"models/{model}",
                    "content": {"parts": [{"text": text}]},
                    "taskType": gemini_task,
                    "outputDimensionality": settings.embedding_dimensions,
                }
                for text in texts
            ]
        }

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = await self._client.post(url, json=payload)
                if response.status_code == 429 and attempt < 2:
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue
                response.raise_for_status()
                data = response.json()
                embeddings = data.get("embeddings")
                if not embeddings or len(embeddings) != len(texts):
                    raise EmbeddingError(
                        f"Embedding API returned {len(embeddings or [])} vectors for "
                        f"{len(texts)} inputs"
                    )
                return [e["values"] for e in embeddings]
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


# Singleton
embedding_client = EmbeddingClient()
