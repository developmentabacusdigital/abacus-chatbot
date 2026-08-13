"""
Soft email capture.

Asked once, near the start of a session: "This will help me follow up in case you need
to step away." If the visitor gives an email, it's stored; if not, the bot says so and
moves on with whatever they actually said. Extraction and decline-detection are plain
regex/keyword logic — cheap, deterministic, and doesn't need a model call on every turn.
"""

import re
from typing import Optional

EMAIL_ASK = (
    "Before we go further — what's a good email for you? "
    "This will help me follow up in case you need to step away."
)

FOUND_ACK = "Got it — thanks! I've noted that so we can follow up if needed."
DECLINE_ACK = "No problem — happy to keep going without it."

_EMAIL_PATTERN = re.compile(r"[\w.+-]+@(?:[\w-]+\.)+[a-zA-Z]{2,}")

_DECLINE_PHRASES = (
    "no", "nah", "nope", "no thanks", "no thank you", "not now", "not right now",
    "rather not", "prefer not", "skip", "pass", "i'd rather not", "id rather not",
    "no email", "won't share", "wont share", "not sharing", "none", "n/a",
)


def extract_email(text: str) -> Optional[str]:
    """Pull the first email address out of free text, or None."""
    if not text:
        return None
    match = _EMAIL_PATTERN.search(text)
    return match.group(0) if match else None


def looks_like_decline(text: str) -> bool:
    """
    True if the message reads as a short refusal to share an email, with nothing else
    of substance in it. A longer message that happens to contain "no" (e.g. "no, but do
    you do SEO?") is NOT a bare decline — it should still be routed normally.
    """
    if not text:
        return False
    normalized = text.strip().lower().rstrip(".!? ")
    if normalized in _DECLINE_PHRASES:
        return True
    # Short message starting with a decline phrase and nothing substantial after it
    word_count = len(normalized.split())
    if word_count <= 4:
        return any(normalized.startswith(p) for p in _DECLINE_PHRASES)
    return False


def strip_email(text: str, email: str) -> str:
    """Remove a matched email from a message, for detecting leftover content."""
    return text.replace(email, "").strip()
