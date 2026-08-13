"""
Abacus Digital Chatbot - Intent Classifier
Classifies user messages into action categories.
"""

import json
import logging
from typing import Dict, Any

from .llm_router import llm_router
from .models import Intent

logger = logging.getLogger(__name__)

# Keywords for fast-path classification (no LLM needed)
GREETING_KEYWORDS = {"hi", "hello", "hey", "howdy", "greetings", "good morning", "good afternoon", "good evening", "sup", "yo"}
FAREWELL_KEYWORDS = {"bye", "goodbye", "see you", "thanks bye", "that's all", "exit", "quit"}
# Multi-word phrases only: bare "call"/"book" appear constantly in ordinary sentences
BOOKING_KEYWORDS = {
    "book a call", "book a meeting", "book a time", "schedule a call",
    "schedule a meeting", "set up a call", "discovery call", "consultation",
    "book an appointment", "talk to sales", "get on a call",
}
DELETION_KEYWORDS = {"delete my data", "remove my data", "forget me", "gdpr", "erase my data", "data deletion", "delete this chat"}
HANDOFF_KEYWORDS = {
    "speak to a human", "talk to a human", "speak to someone", "talk to someone",
    "real person", "speak to a person", "human agent", "customer service",
}


class IntentClassifier:
    """Classifies user intent from messages."""

    async def classify(
        self,
        message: str,
        conversation_history: list = None,
        current_state: str = "greeting",
    ) -> Dict[str, Any]:
        """
        Classify the intent of a user message.

        Returns:
        {
            "intent": Intent enum value,
            "confidence": float (0-1),
            "details": str (optional additional context)
        }
        """
        message_lower = message.lower().strip()

        # Fast-path keyword matching (no LLM cost)
        fast_result = self._fast_classify(message_lower)
        if fast_result:
            return fast_result

        # LLM-based classification for ambiguous messages
        return await self._llm_classify(message, conversation_history, current_state)

    def _fast_classify(self, message_lower: str) -> Dict[str, Any] | None:
        """
        Attempt fast keyword-based classification.

        Checked most-specific-first, and only on short messages for the greeting and
        farewell cases — "hi, our website is losing customers" is a qualification
        message, not a greeting.
        """
        word_count = len(message_lower.split())

        # Data deletion (most specific, and legally important to get right)
        if any(kw in message_lower for kw in DELETION_KEYWORDS):
            return {"intent": Intent.DATA_DELETION, "confidence": 0.95, "details": ""}

        # Explicit request for a person
        if any(kw in message_lower for kw in HANDOFF_KEYWORDS):
            return {"intent": Intent.HUMAN_HANDOFF, "confidence": 0.9, "details": ""}

        # Booking (multi-word phrases only)
        if any(kw in message_lower for kw in BOOKING_KEYWORDS):
            return {"intent": Intent.BOOKING, "confidence": 0.85, "details": ""}

        # Greetings — only when the message is essentially just a greeting
        if word_count <= 4 and (
            message_lower.rstrip("!.? ") in GREETING_KEYWORDS
            or any(message_lower.startswith(g + " ") for g in GREETING_KEYWORDS)
            or message_lower in GREETING_KEYWORDS
        ):
            return {"intent": Intent.GREETING, "confidence": 0.95, "details": ""}

        # Farewells — likewise, short messages only
        if word_count <= 6 and any(kw in message_lower for kw in FAREWELL_KEYWORDS):
            return {"intent": Intent.FAREWELL, "confidence": 0.9, "details": ""}

        return None

    async def _llm_classify(
        self,
        message: str,
        conversation_history: list = None,
        current_state: str = "greeting",
    ) -> Dict[str, Any]:
        """Use LLM for intent classification."""
        conv_context = ""
        if conversation_history:
            recent = conversation_history[-4:]
            conv_context = "\n".join(
                f"{m['role'].upper()}: {m['content'][:100]}" for m in recent
            )

        prompt = f"""Classify the user's message into exactly ONE intent category.

Current conversation state: {current_state}

Recent conversation context:
{conv_context}

User message: "{message}"

Categories:
- "question": User is asking a question about Abacus Digital's services, capabilities, process, or general information
- "qualification": User is expressing a business need, describing their situation, or providing details about what they need help with (e.g., "I need a website", "we're looking for help with marketing")
- "booking": User wants to schedule a call, meeting, or consultation
- "project_intake": User is describing a specific project with details about goals, budget, timeline, or requirements
- "general": General conversation, small talk, or messages that don't fit other categories
- "greeting": Hello, hi, hey, etc.
- "farewell": Goodbye, thanks, that's all, etc.
- "data_deletion": User wants their data deleted
- "human_handoff": User explicitly wants to speak with a person

Respond with ONLY a JSON object:
{{"intent": "<category>", "confidence": <0.0-1.0>, "details": "<brief reason>"}}"""

        result = await llm_router.generate_json(
            messages=[{"role": "user", "content": prompt}],
            task_type="intent_classification",
            temperature=0.1,
            max_tokens=150,
        )

        parsed = result["data"]
        if parsed:
            try:
                intent = Intent(parsed.get("intent", "general"))
            except ValueError:
                intent = Intent.GENERAL
            try:
                confidence = float(parsed.get("confidence", 0.5))
            except (TypeError, ValueError):
                confidence = 0.5
            return {
                "intent": intent,
                "confidence": confidence,
                "details": str(parsed.get("details", ""))[:200],
            }

        # Fallback: treat as a question if it reads like one, else general
        if "?" in message:
            return {"intent": Intent.QUESTION, "confidence": 0.6, "details": "fallback - has question mark"}
        return {"intent": Intent.GENERAL, "confidence": 0.5, "details": "fallback - parse error"}


# Singleton
intent_classifier = IntentClassifier()
