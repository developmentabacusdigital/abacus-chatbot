"""
Abacus Digital Chatbot - Lead Qualifier
Conversational lead qualification with scoring.
"""

import json
import logging
from typing import Dict, Any, Optional, List

from .config import QUALIFICATION_PROMPT
from .llm_router import llm_router
from .models import ConversationState
from .normalize import normalize_fields

logger = logging.getLogger(__name__)

# Qualification score weights
SCORE_WEIGHTS = {
    "budget": 0.30,
    "timeline": 0.25,
    "authority": 0.25,
    "fit": 0.20,
}

# Budget band scores
BUDGET_SCORES = {
    "under_1k": 0.2,
    "1k_to_5k": 0.5,
    "5k_to_15k": 0.7,
    "15k_to_50k": 0.9,
    "over_50k": 1.0,
    "not_sure": 0.4,
}

# Timeline scores
TIMELINE_SCORES = {
    "immediate": 1.0,
    "short_term": 0.8,
    "medium_term": 0.5,
    "long_term": 0.3,
    "just_exploring": 0.1,
    "not_sure": 0.3,
}

# Qualification threshold for booking
QUALIFICATION_THRESHOLD = 0.5

# Order in which qualification chases missing fields — also the order suggestion chips
# are offered in, so the chip on screen always matches the question just asked.
FIELD_ORDER = ["business_type", "pain_point", "budget_band", "timeline", "decision_maker"]


class LeadQualifier:
    """Manages conversational lead qualification."""

    async def qualify(
        self,
        user_message: str,
        state: ConversationState,
    ) -> Dict[str, Any]:
        """
        Process a user message in the qualification flow.

        Returns:
        {
            "response": str,               # Bot response
            "extracted_data": dict,         # Any new data extracted
            "qualification_score": float,   # Updated score
            "is_qualified": bool,           # Whether they've hit threshold
            "should_offer_booking": bool,   # Whether to transition to booking
            "model_used": str,
            "cost": float,
        }
        """
        # Build conversation context
        conv_text = ""
        for msg in state.messages[-10:]:
            conv_text += f"{msg['role'].upper()}: {msg['content']}\n"

        # Ask LLM to continue qualification AND extract data
        system_prompt = f"""You are qualifying a lead for Abacus Digital. Be conversational and natural — NOT a rigid form.

Current collected data:
{json.dumps(state.qualification_data, indent=2)}

Fields still needed (only ask about missing ones):
{self._get_missing_fields(state.qualification_data)}

INSTRUCTIONS:
1. Respond naturally to the user's message
2. Extract any qualification data from their message
3. Ask about ONE missing field naturally (don't interrogate)
4. If you have enough data (at least business_type + pain_point + one of budget/timeline), you can suggest booking a call
5. Be empathetic and acknowledge what they share

Respond as a JSON object:
{{
    "response": "<your conversational response>",
    "extracted_data": {{
        "business_type": "<if mentioned, else null>",
        "pain_point": "<if mentioned, else null>",
        "budget_band": "<under_1k|1k_to_5k|5k_to_15k|15k_to_50k|over_50k|not_sure if mentioned, else null>",
        "timeline": "<immediate|short_term|medium_term|long_term|just_exploring|not_sure if mentioned, else null>",
        "decision_maker": <true|false|null>,
        "company": "<if mentioned, else null>",
        "name": "<if mentioned, else null>",
        "email": "<if mentioned, else null>",
        "service_interest": "<if mentioned, else null>"
    }},
    "enough_data_for_booking": <true|false>
}}"""

        messages = [
            {"role": "system", "content": system_prompt},
        ]

        # Add recent conversation for context
        for msg in state.messages[-6:]:
            messages.append(msg)

        messages.append({"role": "user", "content": user_message})

        result = await llm_router.generate_json(
            messages=messages,
            task_type="qualification",
            temperature=0.7,
            max_tokens=600,
        )

        parsed = result["data"]
        if not parsed:
            # Both models failed to return usable JSON; keep the conversation moving
            # rather than echoing raw model output at the visitor.
            return {
                "response": self._next_question(state.qualification_data),
                "extracted_data": {},
                "qualification_score": state.qualification_score,
                "is_qualified": state.is_qualified,
                "should_offer_booking": False,
                "model_used": result["model_used"],
                "cost": result["cost"],
            }

        # Merge extracted data, banding free-text budget/timeline answers
        extracted = normalize_fields(parsed.get("extracted_data", {}))
        new_data = {}
        for key, value in extracted.items():
            if value is not None and value != "null" and str(value).lower() != "null":
                new_data[key] = value
                state.qualification_data[key] = value

        # Calculate qualification score
        score = self._calculate_score(state.qualification_data)
        state.qualification_score = score

        is_qualified = score >= QUALIFICATION_THRESHOLD
        state.is_qualified = is_qualified

        enough_data = parsed.get("enough_data_for_booking", False)
        should_book = is_qualified and enough_data and not state.booking_offered

        return {
            "response": parsed.get("response", "Could you tell me more about what you're looking for?"),
            "extracted_data": new_data,
            "qualification_score": score,
            "is_qualified": is_qualified,
            "should_offer_booking": should_book,
            "model_used": result["model_used"],
            "cost": result["cost"],
        }

    def score_from_data(self, data: Dict) -> float:
        """Public scoring entry point (also used by the intake agent's brief step)."""
        return self._calculate_score(data)

    def next_missing_field(self, data: Dict) -> Optional[str]:
        """Which field the bot is about to ask for next — drives suggestion chips."""
        for field in FIELD_ORDER:
            if not data.get(field):
                return field
        return None

    @staticmethod
    def _next_question(data: Dict) -> str:
        """Deterministic next question, used when the model fails to return JSON."""
        prompts = {
            "business_type": "What kind of business are you running?",
            "pain_point": "What's the main challenge you're hoping to solve?",
            "budget_band": "Do you have a rough budget range in mind for this?",
            "timeline": "When are you hoping to get this underway?",
            "decision_maker": "Are you the one making the call on this, or will others be involved?",
        }
        for field, question in prompts.items():
            if not data.get(field):
                return question
        return "Thanks — that's really helpful. Anything else you'd like our team to know?"

    def _get_missing_fields(self, data: Dict) -> str:
        """Get a list of fields still needed."""
        fields = {
            "business_type": "What type of business are they?",
            "pain_point": "What's their main challenge or pain point?",
            "budget_band": "What's their rough budget range?",
            "timeline": "When do they need this done?",
            "decision_maker": "Are they the decision-maker?",
        }
        missing = []
        for field, description in fields.items():
            if field not in data or data[field] is None:
                missing.append(f"- {field}: {description}")

        return "\n".join(missing) if missing else "All key fields collected!"

    def _calculate_score(self, data: Dict) -> float:
        """
        Calculate qualification score from collected data.
        Weighted: budget 30%, timeline 25%, authority 25%, fit 20%.
        """
        score = 0.0

        # Budget component
        budget = data.get("budget_band", "")
        if budget in BUDGET_SCORES:
            score += BUDGET_SCORES[budget] * SCORE_WEIGHTS["budget"]
        elif budget:
            # Unknown budget string - give partial credit
            score += 0.3 * SCORE_WEIGHTS["budget"]

        # Timeline component
        timeline = data.get("timeline", "")
        if timeline in TIMELINE_SCORES:
            score += TIMELINE_SCORES[timeline] * SCORE_WEIGHTS["timeline"]
        elif timeline:
            score += 0.3 * SCORE_WEIGHTS["timeline"]

        # Authority component
        dm = data.get("decision_maker")
        if dm is True:
            score += 1.0 * SCORE_WEIGHTS["authority"]
        elif dm is False:
            score += 0.4 * SCORE_WEIGHTS["authority"]
        # If unknown, no credit

        # Fit component (do they have a real business need?)
        fit = 0.0
        if data.get("business_type"):
            fit += 0.3
        if data.get("pain_point"):
            fit += 0.4
        if data.get("service_interest"):
            fit += 0.3
        score += fit * SCORE_WEIGHTS["fit"]

        return round(score, 3)

    def get_qualification_summary(self, data: Dict, score: float) -> str:
        """Generate a human-readable qualification summary."""
        lines = ["📋 **Lead Qualification Summary**\n"]

        if data.get("name"):
            lines.append(f"**Name:** {data['name']}")
        if data.get("company"):
            lines.append(f"**Company:** {data['company']}")
        if data.get("business_type"):
            lines.append(f"**Business Type:** {data['business_type']}")
        if data.get("pain_point"):
            lines.append(f"**Pain Point:** {data['pain_point']}")
        if data.get("service_interest"):
            lines.append(f"**Service Interest:** {data['service_interest']}")
        if data.get("budget_band"):
            lines.append(f"**Budget Range:** {data['budget_band'].replace('_', ' ').title()}")
        if data.get("timeline"):
            lines.append(f"**Timeline:** {data['timeline'].replace('_', ' ').title()}")

        lines.append(f"\n**Qualification Score:** {score:.0%}")

        if score >= QUALIFICATION_THRESHOLD:
            lines.append("✅ **Status:** Qualified for discovery call")
        else:
            lines.append("📌 **Status:** Not yet qualified — needs more information")

        return "\n".join(lines)


# Singleton
lead_qualifier = LeadQualifier()
