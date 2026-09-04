"""
Abacus Digital Chatbot - RAG Engine
Retrieval-Augmented Generation for grounded Q&A.

Public and client retrieval are separate methods over separate collections; neither
ever falls back to the other (PRD 9.5).
"""

import logging
from typing import List, Dict, Any, Optional

from .config import SYSTEM_PROMPT_BASE, CLIENT_SUPPORT_SYSTEM_PROMPT
from .vector_store import vector_store
from .llm_router import llm_router
from .embeddings import EmbeddingError

logger = logging.getLogger(__name__)

NO_ANSWER_RESPONSE = (
    "I don't have specific information about that in my knowledge base. I'd be happy to "
    "connect you with our team who can give you a detailed answer — would you like to get in touch?"
)


class RAGEngine:
    """Handles RAG retrieval and grounded answer generation."""

    def __init__(self):
        # Below this cosine similarity we treat retrieval as a miss and say so (PRD 7.1)
        self.min_relevance_score = 0.3

    async def _search(self, **kwargs) -> List[Dict[str, Any]]:
        """
        pgvector search is genuinely async I/O now (embedding via HTTP call, retrieval
        via asyncpg) — no executor needed, unlike the old CPU-bound local-model version.

        A missing/broken embedding provider degrades to "no results" here rather than
        raising, so the caller's existing no-grounded-answer handling (an honest "I
        don't know, want to talk to the team?") kicks in instead of a generic error.
        """
        try:
            return await vector_store.search(**kwargs)
        except EmbeddingError as e:
            logger.error(f"Retrieval unavailable: {e}")
            return []

    async def answer(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Answer a prospect question from the public index.

        Returns {"answer", "sources", "source_link", "model_used", "cost", "grounded"}.
        "sources" is a plain list of labels (used internally to track service interest);
        "source_link" is a single {label, url} citation to the best-matching live page,
        for the widget to render as one discreet link rather than a row of chips.
        """
        results = await self._search(query=query, top_k=top_k)
        relevant = [r for r in results if r["score"] >= self.min_relevance_score]

        if not relevant:
            return {
                "answer": NO_ANSWER_RESPONSE,
                "sources": [],
                "source_link": None,
                "model_used": "none",
                "cost": 0.0,
                "grounded": False,
            }

        context_parts = []
        sources: List[str] = []
        source_link: Optional[Dict[str, str]] = None

        for r in relevant:
            context_parts.append(r["text"])
            meta = r["metadata"] or {}

            label = meta.get("title") or meta.get("service") or meta.get("source") or ""
            if label and label != "general" and label not in sources:
                sources.append(label)

            # First live-crawled page wins the citation link; results are already
            # ordered by relevance, so this is the strongest match with a real URL.
            url = meta.get("source_url")
            if url and source_link is None:
                source_link = {"label": meta.get("title") or "Learn more", "url": url}

        context = "\n\n---\n\n".join(context_parts)

        system_prompt = f"""{SYSTEM_PROMPT_BASE}

RETRIEVED CONTEXT (use ONLY this information to answer):
{context}

INSTRUCTIONS:
- Answer the user's question based ONLY on the context above
- If the context doesn't fully answer the question, say what you know and note what's not covered
- Cite which service or page your answer comes from
- Keep your answer concise and helpful
- If asked about pricing, never give a number or range — explain that it depends on scope and can only come from the team after a discovery call, and end that sentence with a single *"""

        messages = [{"role": "system", "content": system_prompt}]
        if conversation_history:
            messages.extend(conversation_history[-6:])
        messages.append({"role": "user", "content": query})

        result = await llm_router.generate(
            messages=messages,
            task_type="rag_answer",
            temperature=0.5,
            max_tokens=800,
        )

        return {
            "answer": result["content"],
            "sources": sources[:4],
            "source_link": source_link,
            "model_used": result["model_used"],
            "cost": result["cost"],
            "grounded": result.get("ok", True),
        }

    async def answer_for_client(
        self,
        query: str,
        client_id: str,
        client_name: str,
        client_company: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Answer an authenticated client question from the client index only.

        Retrieval is hard-scoped to client_id; the public index is never consulted here.
        """
        try:
            results = await vector_store.search_client(query=query, client_id=client_id, top_k=top_k)
        except EmbeddingError as e:
            logger.error(f"Client retrieval unavailable: {e}")
            results = []
        relevant = [r for r in results if r["score"] >= self.min_relevance_score]

        if not relevant:
            return {
                "answer": "",
                "sources": [],
                "model_used": "none",
                "cost": 0.0,
                "grounded": False,
            }

        context = "\n\n---\n\n".join(r["text"] for r in relevant)
        sources = []
        for r in relevant:
            label = (r["metadata"] or {}).get("project_name")
            if label and label not in sources:
                sources.append(label)

        system_prompt = CLIENT_SUPPORT_SYSTEM_PROMPT.format(
            client_name=client_name or "there",
            client_company=client_company or "your company",
            context=context,
        )

        messages = [{"role": "system", "content": system_prompt}]
        if conversation_history:
            messages.extend(conversation_history[-6:])
        messages.append({"role": "user", "content": query})

        result = await llm_router.generate(
            messages=messages,
            task_type="client_support",
            temperature=0.3,
            max_tokens=700,
        )

        return {
            "answer": result["content"],
            "sources": sources,
            "model_used": result["model_used"],
            "cost": result["cost"],
            "grounded": result.get("ok", True),
        }

    async def retrieve_context(self, query: str, top_k: int = 5) -> str:
        """Plain retrieval, used by the intake agent to ground its questions."""
        results = await self._search(query=query, top_k=top_k)
        relevant = [r for r in results if r["score"] >= self.min_relevance_score]
        return "\n\n---\n\n".join(r["text"] for r in relevant[:top_k])

    async def match_services(self, description: str) -> List[Dict[str, Any]]:
        """
        Retrieve the service lines most similar to a project description.
        Used as retrieval input to the reasoning-based service matcher.
        """
        results = await self._search(
            query=description,
            top_k=8,
            filter_metadata={"section": "service_overview"},
        )

        services = []
        seen = set()
        for r in results:
            service_name = (r["metadata"] or {}).get("service", "")
            if service_name and service_name not in seen and r["score"] >= 0.3:
                seen.add(service_name)
                services.append({
                    "service": service_name,
                    "relevance": round(r["score"], 3),
                    "description": r["text"][:200],
                })

        return services


# Singleton
rag_engine = RAGEngine()
