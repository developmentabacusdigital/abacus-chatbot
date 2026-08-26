"""
Suggestion-chip catalogs for the widget's quick-reply bubbles.

When the bot asks a specific, finite-answer question (budget band, timeline,
decision-maker status) it attaches a short list of likely replies. Clicking one sends
its text exactly as if the visitor had typed it — the orchestrator doesn't need to know
a click produced the message rather than a keystroke.

Open-ended fields (goals, pain point, current state) get no suggestions; forcing chips
onto a "tell me more" question produces answers that are too narrow to be useful.
"""

from typing import Any, Dict, List, Optional

BUDGET_SUGGESTIONS = [
    "Under $1k", "$1k – $5k", "$5k – $15k", "$15k – $50k", "Over $50k", "Not sure yet",
]

TIMELINE_SUGGESTIONS = [
    "ASAP", "1–3 months", "3–6 months", "6+ months", "Just exploring",
]

AUTHORITY_SUGGESTIONS = [
    "Yes, I decide", "No, someone else decides", "We decide together",
]

GREETING_SUGGESTIONS = [
    "See our services", "I have a project in mind", "Book a call", "Just browsing",
]

EMAIL_ASK_SUGGESTIONS = ["No thanks, let's continue"]

# Generic nets for turns that don't map onto a qualification/intake field at all —
# a RAG answer, a conversational aside, a self-serve hand-off. Kept short and broadly
# applicable rather than trying to be clever without the actual question text to key off.
QUESTION_FOLLOWUP_SUGGESTIONS = ["Tell me about pricing", "Book a call", "See our services"]
SELF_SERVE_SUGGESTIONS = ["Book a call", "See our services", "Ask something else"]
FAREWELL_SUGGESTIONS = ["Book a call", "Ask another question"]

FIELD_SUGGESTIONS: Dict[str, List[str]] = {
    "budget_band": BUDGET_SUGGESTIONS,
    "timeline": TIMELINE_SUGGESTIONS,
    "decision_maker": AUTHORITY_SUGGESTIONS,
}


def suggestions_for_field(field: Optional[str]) -> List[str]:
    """Suggestion chips for the field the bot is currently asking about, if any."""
    if not field:
        return []
    return FIELD_SUGGESTIONS.get(field, [])


def clean_suggestions(raw: Any, limit: int = 5) -> List[str]:
    """
    Validate a model-proposed suggestion list: must actually be a list of short,
    non-empty, distinct strings. Anything malformed collapses to an empty list so the
    caller falls back to a deterministic default rather than showing garbage chips.
    """
    if not isinstance(raw, list):
        return []
    seen = set()
    cleaned: List[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        text = item.strip()
        key = text.lower()
        if not text or len(text) > 40 or key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
        if len(cleaned) == limit:
            break
    return cleaned
