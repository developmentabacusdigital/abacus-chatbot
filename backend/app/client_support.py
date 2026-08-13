"""
Abacus Digital Chatbot - Client Support (PRD Phase 3 / 7.7)

Handles the authenticated client surface. Retrieval is scoped to the signed-in client's
own project data and never touches the public prospect index. Anything the client index
can't answer — or anything commercial, contractual, or emotionally charged — escalates
to a human account manager rather than being improvised.
"""

import logging
from typing import Dict, Any

from .config import settings, ESCALATION_MESSAGE
from .database import db
from .email_service import email_service
from .models import ConversationState, SessionStatus
from .rag_engine import rag_engine

logger = logging.getLogger(__name__)

# Phrases that should go straight to a human, no retrieval attempted
ESCALATION_TRIGGERS = (
    "speak to someone", "speak to a human", "talk to a person", "talk to someone",
    "account manager", "escalate", "complaint", "complain", "refund", "invoice",
    "billing", "contract", "cancel", "terminate", "legal", "unacceptable",
    "not happy", "unhappy", "frustrated", "this is ridiculous",
)


class ClientSupport:
    """Answers authenticated client questions and manages escalation."""

    async def handle(self, message: str, state: ConversationState) -> Dict[str, Any]:
        """Process one authenticated client turn."""
        if state.escalated:
            return self._already_escalated()

        if self._should_escalate(message):
            return await self.escalate(
                state=state,
                reason=f"Client requested human support or raised a commercial issue: {message[:200]}",
            )

        result = await rag_engine.answer_for_client(
            query=message,
            client_id=state.client_id,
            client_name=state.client_name or "",
            client_company=state.client_company or "",
            conversation_history=state.messages,
        )

        if not result["grounded"] or not result["answer"].strip():
            escalation = await self.escalate(
                state=state,
                reason=f"No grounded answer in the client knowledge base for: {message[:200]}",
            )
            escalation["message"] = (
                "I don't have that detail in your project records. "
                + escalation["message"]
            )
            escalation["cost"] = escalation.get("cost", 0.0) + result["cost"]
            return escalation

        return {
            "message": result["answer"],
            "sources": result["sources"],
            "model_used": result["model_used"],
            "cost": result["cost"],
            "escalated": False,
        }

    async def escalate(self, state: ConversationState, reason: str) -> Dict[str, Any]:
        """Hand the conversation to the client's account manager."""
        state.escalated = True

        client = await db.get_client(state.client_id) if state.client_id else None
        manager = (
            (client.account_manager_email if client else None)
            or settings.account_manager_email
        )

        summary = self.transcript_digest(state)

        await db.create_escalation(
            session_id=state.session_id,
            reason=reason,
            client_id=state.client_id,
            transcript_summary=summary,
        )
        await db.update_session(state.session_id, status=SessionStatus.ESCALATED.value)

        try:
            await email_service.notify_escalation(
                manager_email=manager,
                client_name=state.client_name or "Client",
                client_company=state.client_company or "",
                reason=reason,
                transcript_summary=summary,
                session_id=state.session_id,
            )
        except Exception as e:
            # The escalation record is already saved; email failure must not lose it
            logger.error(f"Escalation email failed for session {state.session_id}: {e}")

        return {
            "message": ESCALATION_MESSAGE.format(manager_email=manager),
            "sources": [],
            "model_used": "template",
            "cost": 0.0,
            "escalated": True,
        }

    @staticmethod
    def _should_escalate(message: str) -> bool:
        lowered = message.lower()
        return any(trigger in lowered for trigger in ESCALATION_TRIGGERS)

    @staticmethod
    def _already_escalated() -> Dict[str, Any]:
        return {
            "message": (
                "Your account manager already has this conversation and will be in touch. "
                "Is there anything else I can look up in your project records meanwhile?"
            ),
            "sources": [],
            "model_used": "template",
            "cost": 0.0,
            "escalated": True,
        }

    @staticmethod
    def transcript_digest(state: ConversationState, turns: int = 12) -> str:
        return "\n".join(
            f"{m['role'].upper()}: {m['content'][:500]}"
            for m in state.messages[-turns:]
        )

    def greeting(self, state: ConversationState) -> str:
        name = state.client_name or "there"
        company = f" at {state.client_company}" if state.client_company else ""
        return (
            f"Hi {name}{company} 👋 You're signed in to the Abacus Digital client portal. "
            "I can check your project status, deliverables, and support docs. "
            "If you'd rather speak to your account manager, just say so and I'll pass it on."
        )


client_support = ClientSupport()
