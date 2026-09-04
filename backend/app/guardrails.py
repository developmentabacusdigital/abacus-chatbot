"""
Abacus Digital Chatbot - Guardrails
Safety guardrails, rate limiting, and input validation.
"""

import re
import logging
import time
from collections import defaultdict
from typing import Optional, Tuple

from .config import settings

logger = logging.getLogger(__name__)


class RateLimiter:
    """In-memory rate limiter for API requests."""

    def __init__(self):
        # {ip: [(timestamp, session_id), ...]}
        self._requests: dict = defaultdict(list)
        # {session_id: message_count}
        self._session_counts: dict = defaultdict(int)

    def check_rate_limit(
        self,
        ip_address: str,
        session_id: str,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if a request is within rate limits.

        Returns: (is_allowed, error_message)
        """
        now = time.time()

        # Clean old entries (older than 1 hour)
        cutoff = now - 3600
        self._requests[ip_address] = [
            (ts, sid) for ts, sid in self._requests[ip_address]
            if ts > cutoff
        ]

        # Check session message limit
        if self._session_counts[session_id] >= settings.max_messages_per_session:
            return False, "You've reached the message limit for this session. Please start a new conversation or contact us directly."

        # Check IP session limit
        unique_sessions = set(
            sid for _, sid in self._requests[ip_address]
        )
        if len(unique_sessions) >= settings.max_sessions_per_ip_per_hour:
            if session_id not in unique_sessions:
                return False, "Too many conversations from this location. Please try again later or contact us directly."

        # Record this request
        self._requests[ip_address].append((now, session_id))
        self._session_counts[session_id] += 1

        return True, None

    def reset_session(self, session_id: str):
        """Reset message count for a session."""
        self._session_counts.pop(session_id, None)


class ContentGuardrails:
    """Content safety and validation guardrails."""

    # Patterns for sensitive data we should never accept or store
    SENSITIVE_PATTERNS = [
        (r'\b\d{3}-\d{2}-\d{4}\b', "SSN"),          # SSN
        (r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', "credit card"),  # Credit card
        (r'\b\d{9}\b', "possible SSN"),                # 9-digit number
    ]

    # Topics the bot should not engage with
    RESTRICTED_TOPICS = [
        "competitor pricing",
        "guarantee",
        "guaranteed results",
        "sign the contract",
        "discount code",
        "free work",
    ]

    @staticmethod
    def validate_input(message: str) -> Tuple[bool, Optional[str]]:
        """
        Validate user input for safety concerns.

        Returns: (is_safe, warning_message)
        """
        if not message or not message.strip():
            return False, "Please enter a message."

        if len(message) > 5000:
            return False, "Message is too long. Please keep your message under 5000 characters."

        # Check for sensitive data
        for pattern, data_type in ContentGuardrails.SENSITIVE_PATTERNS:
            if re.search(pattern, message):
                return False, f"For your security, please don't share {data_type} numbers in chat. We never need this information."

        return True, None

    # Any dollar figure at all — the bot must never quote a price, so this isn't
    # narrowed to "exact" pricing phrasing the way it used to be. Backstop for the
    # system prompts' own "never give pricing" rule, not a replacement for it: an LLM
    # can still slip past its instructions, so this catches whatever gets through.
    _DOLLAR_AMOUNT = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?(?:\s?[kK])?\b(?!\*)")
    _PRICE_TOPIC = re.compile(r"\b(?:quote|quotation|pricing|price|prices|cost|costs)\b", re.IGNORECASE)
    _DISCLAIMER = (
        "\n\n*Pricing depends on your specific requirements — the team will give you an "
        "exact number after a discovery call."
    )

    @staticmethod
    def sanitize_output(response: str) -> str:
        """
        Backstop against a quotation slipping past the system prompt's "never give
        pricing" rule: any dollar figure gets a footnote asterisk right where it
        appears (rather than being silently deleted, which risks a broken sentence),
        and any mention of pricing as a topic — figure or not — gets the disclaimer
        appended once.
        """
        dollar_matches = list(ContentGuardrails._DOLLAR_AMOUNT.finditer(response))
        if dollar_matches:
            offset = 0
            for m in dollar_matches:
                insert_at = m.end() + offset
                response = response[:insert_at] + "*" + response[insert_at:]
                offset += 1

        mentions_pricing = bool(dollar_matches) or bool(ContentGuardrails._PRICE_TOPIC.search(response))
        if mentions_pricing and "Pricing depends on your specific requirements" not in response:
            response += ContentGuardrails._DISCLAIMER

        return response

    @staticmethod
    def check_commitment_language(response: str) -> str:
        """Add disclaimers if the response contains commitment language."""
        commitment_words = ["guarantee", "promise", "commit", "assured"]
        if any(word in response.lower() for word in commitment_words):
            response += "\n\n*Please note: Specific commitments and guarantees are discussed during our formal engagement process.*"
        return response


# Singletons
rate_limiter = RateLimiter()
content_guardrails = ContentGuardrails()
