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

    @staticmethod
    def sanitize_output(response: str) -> str:
        """
        Sanitize bot output to enforce guardrails.
        Catches any pricing commitments or unauthorized promises.
        """
        # Check for exact pricing patterns that shouldn't be in responses
        price_patterns = [
            r'\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s*(?:per|/)',  # $X per/month
            r'(?:costs?|price[ds]?|charge[ds]?)\s+(?:exactly|precisely)\s+\$',
        ]

        for pattern in price_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                response += "\n\n*Note: Exact pricing depends on your specific requirements. The numbers mentioned are general ranges — please speak with our team for a detailed quote.*"
                break

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
