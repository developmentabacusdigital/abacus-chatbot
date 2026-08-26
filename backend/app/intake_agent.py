"""
Abacus Digital Chatbot - Agentic Intake (PRD Phase 2 / 7.6)

Runs a multi-turn structured discovery conversation, reasons over the nine service
lines to recommend a fit (including bundles), and emits a ProjectBrief that is stored
as the system of record and optionally emailed to the visitor for confirmation.
"""

import logging
from typing import Dict, Any, List, Optional

from .config import (
    INTAKE_SYSTEM_PROMPT, BRIEF_GENERATION_PROMPT, SERVICE_MATCHING_PROMPT,
)
from .knowledge_base import (
    get_service_catalog_text, get_bundling_logic_text, get_service_line_names,
)
from .llm_router import llm_router
from .rag_engine import rag_engine
from .models import ConversationState, ProjectBrief
from .normalize import normalize_fields, normalize_budget, normalize_timeline
from .suggestions import clean_suggestions, suggestions_for_field

logger = logging.getLogger(__name__)

# Fields the discovery conversation is trying to fill, in the order we chase them
DISCOVERY_FIELDS = {
    "goals": "What business outcome are they trying to achieve?",
    "current_state": "What exists today — current site, stack, or process?",
    "budget_band": "Roughly what budget band are they working with?",
    "timeline": "When do they need this live?",
    "constraints": "Any technical, organisational, or compliance constraints?",
    "success_criteria": "How will they judge whether it worked?",
    "email": "Contact email so the team can follow up",
}

# Discovery is considered complete once these are known
REQUIRED_FOR_BRIEF = ("goals", "current_state", "budget_band", "timeline")

INTAKE_TRIGGER_PHRASES = (
    "i need", "we need", "looking for", "looking to", "we want", "i want",
    "our website", "my website", "rebuild", "redesign", "automate", "project",
    "quote", "proposal", "scope", "help us", "help me with",
)


def looks_like_project_request(message: str) -> bool:
    """Cheap pre-check for whether a message describes a concrete project need."""
    lowered = message.lower()
    if len(lowered.split()) < 4:
        return False
    return any(phrase in lowered for phrase in INTAKE_TRIGGER_PHRASES)


class IntakeAgent:
    """Structured discovery agent for inbound project requests."""

    async def run_turn(
        self,
        user_message: str,
        state: ConversationState,
    ) -> Dict[str, Any]:
        """
        Advance the discovery conversation by one turn.

        Returns {"response", "extracted", "discovery_complete", "model_used", "cost"}.
        """
        context = await rag_engine.retrieve_context(user_message, top_k=4)

        prompt = INTAKE_SYSTEM_PROMPT.format(
            service_lines="\n".join(f"- {name}" for name in get_service_line_names()),
            context=context or "(no relevant context retrieved)",
            collected=self._format_collected(state.intake_data),
            missing=self._format_missing(state.intake_data),
        )

        messages: List[Dict[str, str]] = [{"role": "system", "content": prompt}]
        messages.extend(state.messages[-8:])
        messages.append({"role": "user", "content": user_message})

        result = await llm_router.generate_json(
            messages=messages,
            task_type="intake_reasoning",
            temperature=0.6,
            max_tokens=900,
        )

        data = result["data"]
        if not data:
            # Discovery must not dead-end on a parse failure; ask the next missing thing directly
            return {
                "response": self._fallback_question(state.intake_data),
                "extracted": {},
                "discovery_complete": False,
                "suggestions": suggestions_for_field(self.next_missing_field(state.intake_data)),
                "model_used": result["model_used"],
                "cost": result["cost"],
            }

        extracted = normalize_fields({
            k: v for k, v in (data.get("extracted") or {}).items()
            if v not in (None, "", "null", "None")
        })
        state.intake_data.update(extracted)

        complete = self._is_complete(state.intake_data) and bool(
            data.get("discovery_complete", False)
        )

        # Prefer the model's own suggestions — generated alongside the question it just
        # asked, so they track it exactly instead of guessing from field order.
        suggestions = clean_suggestions(data.get("suggested_replies"))
        if not suggestions:
            suggestions = suggestions_for_field(self.next_missing_field(state.intake_data))

        return {
            "response": data.get("response") or self._fallback_question(state.intake_data),
            "extracted": extracted,
            "discovery_complete": complete,
            "suggestions": suggestions,
            "model_used": result["model_used"],
            "cost": result["cost"],
        }

    async def match_services(
        self,
        description: str,
        state: Optional[ConversationState] = None,
    ) -> Dict[str, Any]:
        """
        Reason over the service lines to pick a primary service and any bundle.

        Retrieval narrows the candidates; a stronger model makes the call (PRD 9.3).
        """
        retrieved = await rag_engine.match_services(description)
        context = "\n".join(
            f"- {s['service']} (similarity {s['relevance']}): {s['description']}"
            for s in retrieved
        ) or "(no strong retrieval matches)"

        prompt = SERVICE_MATCHING_PROMPT.format(
            service_catalog=get_service_catalog_text(),
            bundling_logic=get_bundling_logic_text(),
            context=context,
            description=description,
        )

        result = await llm_router.generate_json(
            messages=[{"role": "user", "content": prompt}],
            task_type="service_matching",
            temperature=0.2,
            max_tokens=500,
        )

        data = result["data"] or {}
        primary = data.get("primary_service") or (
            retrieved[0]["service"] if retrieved else "Web Design & Development"
        )
        supporting = [s for s in (data.get("supporting_services") or []) if s and s != primary]

        return {
            "primary_service": primary,
            "supporting_services": supporting,
            "bundle_rationale": data.get("bundle_rationale", ""),
            "confidence": float(data.get("confidence", 0.5) or 0.5),
            "retrieved": retrieved,
            "model_used": result["model_used"],
            "cost": result["cost"],
        }

    async def generate_brief(self, state: ConversationState) -> Dict[str, Any]:
        """Produce the final structured brief for the sales team."""
        description = self._project_description(state)
        match = await self.match_services(description, state)

        services_text = f"Primary: {match['primary_service']}"
        if match["supporting_services"]:
            services_text += f"\nSupporting: {', '.join(match['supporting_services'])}"
        if match["bundle_rationale"]:
            services_text += f"\nRationale: {match['bundle_rationale']}"

        transcript = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in state.messages[-24:]
        )

        prompt = BRIEF_GENERATION_PROMPT.format(
            discovery=self._format_collected(state.intake_data),
            services=services_text,
            transcript=transcript,
        )

        result = await llm_router.generate_json(
            messages=[{"role": "user", "content": prompt}],
            task_type="brief_generation",
            temperature=0.3,
            max_tokens=1400,
        )

        data = result["data"] or {}
        recommended = data.get("recommended_services") or (
            [match["primary_service"]] + match["supporting_services"]
        )

        brief = ProjectBrief(
            session_id=state.session_id,
            title=data.get("title") or f"Project enquiry — {match['primary_service']}",
            summary=data.get("summary") or self._fallback_summary(state),
            goals=_as_list(data.get("goals"), state.intake_data.get("goals")),
            current_state=data.get("current_state")
                or state.intake_data.get("current_state")
                or "Not disclosed",
            constraints=_as_list(data.get("constraints"), state.intake_data.get("constraints")),
            budget_band=normalize_budget(
                data.get("budget_band") or state.intake_data.get("budget_band")
            ),
            timeline=normalize_timeline(
                data.get("timeline") or state.intake_data.get("timeline")
            ),
            success_criteria=_as_list(
                data.get("success_criteria"), state.intake_data.get("success_criteria")
            ),
            recommended_services=recommended,
            bundle_rationale=data.get("bundle_rationale") or match["bundle_rationale"],
            open_questions=_as_list(data.get("open_questions"), None),
            risk_flags=_as_list(data.get("risk_flags"), None),
        )

        return {
            "brief": brief,
            "match": match,
            "model_used": result["model_used"],
            "cost": result["cost"] + match["cost"],
        }

    # --- helpers ---

    @staticmethod
    def _project_description(state: ConversationState) -> str:
        d = state.intake_data
        parts = [
            f"Goals: {d.get('goals', 'unknown')}",
            f"Current state: {d.get('current_state', 'unknown')}",
            f"Pain point: {d.get('pain_point', 'unknown')}",
            f"Business type: {d.get('business_type', 'unknown')}",
            f"Constraints: {d.get('constraints', 'none stated')}",
            f"Budget: {d.get('budget_band', 'unknown')}",
            f"Timeline: {d.get('timeline', 'unknown')}",
        ]
        user_turns = [m["content"] for m in state.messages if m.get("role") == "user"]
        if user_turns:
            parts.append("Visitor's own words: " + " | ".join(user_turns[-5:]))
        return "\n".join(parts)

    @staticmethod
    def _format_collected(data: Dict[str, Any]) -> str:
        if not data:
            return "(nothing collected yet)"
        return "\n".join(f"- {k}: {v}" for k, v in data.items() if v)

    @staticmethod
    def _format_missing(data: Dict[str, Any]) -> str:
        missing = [
            f"- {field}: {question}"
            for field, question in DISCOVERY_FIELDS.items()
            if not data.get(field)
        ]
        return "\n".join(missing) if missing else "(everything needed has been collected)"

    @staticmethod
    def _is_complete(data: Dict[str, Any]) -> bool:
        return all(data.get(f) for f in REQUIRED_FOR_BRIEF)

    @staticmethod
    def next_missing_field(data: Dict[str, Any]) -> Optional[str]:
        """
        Which discovery field is about to be asked about — drives suggestion chips.
        Only budget_band/timeline have a finite, chip-friendly answer set; open fields
        like goals or current_state naturally get no suggestions from the caller.
        """
        for field in DISCOVERY_FIELDS:
            if not data.get(field):
                return field
        return None

    @staticmethod
    def _fallback_question(data: Dict[str, Any]) -> str:
        for field, question in DISCOVERY_FIELDS.items():
            if not data.get(field):
                return {
                    "goals": "Got it — what's the main outcome you're hoping this project delivers?",
                    "current_state": "Understood. What do you have in place today — an existing site or process I should know about?",
                    "budget_band": "That helps. Do you have a rough budget band in mind for this?",
                    "timeline": "And when are you hoping to have this live?",
                    "constraints": "Are there any constraints on your side I should factor in — existing tools, compliance, or internal resourcing?",
                    "success_criteria": "How would you judge whether this project worked?",
                    "email": "What's the best email for our team to follow up on?",
                }[field]
        return "Thanks — that gives me a good picture. Anything else you'd like the team to know?"

    @staticmethod
    def _fallback_summary(state: ConversationState) -> str:
        d = state.intake_data
        return (
            f"Visitor described a project with goals: {d.get('goals', 'not stated')}. "
            f"Current state: {d.get('current_state', 'not disclosed')}. "
            f"Budget {d.get('budget_band', 'not stated')}, timeline {d.get('timeline', 'not stated')}."
        )


def _as_list(value: Any, fallback: Any = None) -> List[str]:
    """Coerce a model's answer into a clean list of strings."""
    source = value if value not in (None, "", []) else fallback
    if not source:
        return []
    if isinstance(source, str):
        return [source.strip()] if source.strip() else []
    if isinstance(source, list):
        return [str(v).strip() for v in source if str(v).strip()]
    return [str(source)]


intake_agent = IntakeAgent()
