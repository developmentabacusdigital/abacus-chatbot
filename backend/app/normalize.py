"""
Coerce free-text budget and timeline answers into the enum bands.

Models are asked for enum values but routinely return what the visitor actually said
("around £20k", "in about 3 months"). Storing that verbatim understates the
qualification score — unrecognised strings only earn partial credit — and makes the
dashboard and CSV impossible to filter. Normalising at the boundary keeps one
vocabulary in the database while the transcript preserves the visitor's own words.
"""

import re
from typing import Optional

from .models import BudgetBand, Timeline

BUDGET_VALUES = {b.value for b in BudgetBand}
TIMELINE_VALUES = {t.value for t in Timeline}

_UNKNOWN = {"", "none", "null", "unknown", "n/a", "na", "not sure", "unsure",
            "no idea", "tbd", "to be confirmed", "not disclosed"}

# "20k", "$20,000", "£15-20k", "between 10 and 15k"
_AMOUNT = re.compile(r"(\d[\d,.]*)\s*([km])?", re.IGNORECASE)
_MONTHS = re.compile(r"(\d+)\s*(?:-|to|–)?\s*(\d+)?\s*month", re.IGNORECASE)
_WEEKS = re.compile(r"(\d+)\s*(?:-|to|–)?\s*(\d+)?\s*week", re.IGNORECASE)
_YEARS = re.compile(r"(\d+)\s*(?:-|to|–)?\s*(\d+)?\s*year", re.IGNORECASE)


def _clean(value) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    return None if text in _UNKNOWN else text


def normalize_budget(value) -> Optional[str]:
    """Map a budget answer onto a BudgetBand value, or None if nothing usable."""
    text = _clean(value)
    if text is None:
        return None
    if text in BUDGET_VALUES:
        return text

    amounts = []
    for raw, suffix in _AMOUNT.findall(text):
        try:
            number = float(raw.replace(",", ""))
        except ValueError:
            continue
        suffix = (suffix or "").lower()
        if suffix == "k":
            number *= 1_000
        elif suffix == "m":
            number *= 1_000_000
        # A bare small number in a money context almost always means thousands
        elif number < 500:
            number *= 1_000
        amounts.append(number)

    if not amounts:
        return BudgetBand.NOT_SURE.value if "budget" in text else None

    # A range bands on its midpoint. Taking the top of "10 to 15k" would promote the
    # lead into a higher band than the visitor actually committed to.
    amount = (min(amounts) + max(amounts)) / 2
    if amount < 1_000:
        return BudgetBand.UNDER_1K.value
    if amount < 5_000:
        return BudgetBand.ONE_TO_5K.value
    if amount < 15_000:
        return BudgetBand.FIVE_TO_15K.value
    if amount < 50_000:
        return BudgetBand.FIFTEEN_TO_50K.value
    return BudgetBand.OVER_50K.value


def normalize_timeline(value) -> Optional[str]:
    """Map a timeline answer onto a Timeline value, or None if nothing usable."""
    text = _clean(value)
    if text is None:
        return None
    if text in TIMELINE_VALUES:
        return text

    if any(word in text for word in ("asap", "immediately", "urgent", "right away", "yesterday")):
        return Timeline.IMMEDIATE.value
    if any(word in text for word in ("just exploring", "exploring", "browsing",
                                     "researching", "no rush", "someday")):
        return Timeline.JUST_EXPLORING.value

    years = _YEARS.search(text)
    if years:
        return Timeline.LONG_TERM.value

    weeks = _WEEKS.search(text)
    if weeks:
        count = max(int(g) for g in weeks.groups() if g)
        return Timeline.IMMEDIATE.value if count <= 2 else Timeline.SHORT_TERM.value

    months = _MONTHS.search(text)
    if months:
        count = max(int(g) for g in months.groups() if g)
        if count <= 3:
            return Timeline.SHORT_TERM.value
        if count <= 6:
            return Timeline.MEDIUM_TERM.value
        return Timeline.LONG_TERM.value

    if "next quarter" in text or "this quarter" in text:
        return Timeline.SHORT_TERM.value
    if "next year" in text:
        return Timeline.LONG_TERM.value

    return None


def normalize_fields(fields: dict) -> dict:
    """Normalise budget_band/timeline in a dict of extracted fields, in place-safe form."""
    out = dict(fields)
    if "budget_band" in out:
        normalized = normalize_budget(out["budget_band"])
        if normalized:
            out["budget_band"] = normalized
        else:
            out.pop("budget_band")
    if "timeline" in out:
        normalized = normalize_timeline(out["timeline"])
        if normalized:
            out["timeline"] = normalized
        else:
            out.pop("timeline")
    return out
