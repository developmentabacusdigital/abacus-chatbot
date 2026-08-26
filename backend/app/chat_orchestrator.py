"""
Abacus Digital Chatbot - Chat Orchestrator
State machine routing conversation flow: intent -> RAG / qualification / intake /
booking / client support -> persistence -> CRM.

Conversation state lives in memory for speed but is rehydrated from the database on a
miss, so a free-tier cold start doesn't lose an in-flight conversation.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from .models import (
    ConversationState, Intent, SessionStatus, SurfaceType,
    ChatRequest, ChatResponse, MessageRecord, LeadRecord,
    LEAD_WRITABLE_FIELDS,
)
from .config import (
    CONSENT_NOTICE, SYSTEM_PROMPT_BASE, TRANSCRIPT_SUMMARY_PROMPT,
    TITLE_GENERATION_PROMPT, settings,
)
from .intent_classifier import intent_classifier
from .rag_engine import rag_engine
from .lead_qualifier import lead_qualifier, QUALIFICATION_THRESHOLD
from .intake_agent import intake_agent, looks_like_project_request
from .booking_handler import booking_handler
from .client_support import client_support
from .guardrails import content_guardrails
from .llm_router import llm_router
from .email_service import email_service, is_valid_email
from .email_capture import extract_email, looks_like_decline, strip_email, EMAIL_ASK, FOUND_ACK, DECLINE_ACK
from .suggestions import (
    suggestions_for_field, GREETING_SUGGESTIONS, EMAIL_ASK_SUGGESTIONS,
    QUESTION_FOLLOWUP_SUGGESTIONS, SELF_SERVE_SUGGESTIONS, FAREWELL_SUGGESTIONS,
)
from .crm_sync import crm_sync
from .database import db

logger = logging.getLogger(__name__)

# Bound the in-memory state cache so a busy widget can't grow it without limit
MAX_CACHED_SESSIONS = 500

# A chat only gets a title once it has at least this many turns of substance
TITLE_MIN_MESSAGE_COUNT = 2

# How many turns to give the conversation to naturally surface a requirement (pain
# point / goals) before the lead-notification email falls back to an auto-summary
# of the transcript instead of waiting indefinitely.
LEAD_NOTIFY_SUMMARY_FALLBACK_AFTER = 6

# Unlike the soft EMAIL_ASK earlier in a session, this one is mandatory: no Calendly
# link goes out without somewhere to send the confirmation and calendar invite.
BOOKING_EMAIL_ASK = (
    "Before I hand you off to book a call — what's the best email to send your "
    "confirmation and calendar invite to?"
)
BOOKING_EMAIL_RETRY = (
    "I'll need a valid email address to send your booking confirmation and calendar "
    "invite — what's the best one to use?"
)


class ChatOrchestrator:
    """Orchestrates the full conversation flow."""

    def __init__(self):
        self._sessions: Dict[str, ConversationState] = {}

    # --- State handling ---

    async def get_state(self, session_id: str) -> ConversationState:
        """Fetch conversation state, rebuilding it from the database if not cached."""
        state = self._sessions.get(session_id)
        if state:
            return state

        state = ConversationState(session_id=session_id)

        session = await db.get_session(session_id)
        if session:
            state.consent_given = session.consent_given
            state.source_page = session.source_page
            state.surface = session.surface
            state.client_id = session.client_id
            state.consent_shown = session.total_messages > 0
            state.escalated = session.status == SessionStatus.ESCALATED
            state.booking_offered = session.status in (
                SessionStatus.BOOKED, SessionStatus.QUALIFIED
            )
            state.visitor_id = session.visitor_id
            state.title = session.title
            state.title_generated = bool(session.title)
            state.category = session.category or "other"

            messages = await db.get_session_messages(session_id)
            state.messages = [{"role": m.role, "content": m.content} for m in messages]
            state.message_count = len(messages)

            lead = await db.get_lead_by_session(session_id)
            if lead:
                state.qualification_score = lead.qualification_score or 0.0
                state.is_qualified = state.qualification_score >= QUALIFICATION_THRESHOLD
                state.qualification_data = {
                    k: v for k, v in lead.model_dump().items()
                    if k in LEAD_WRITABLE_FIELDS and v is not None
                }

            if session.client_id:
                client = await db.get_client(session.client_id)
                if client:
                    state.client_name = client.name
                    state.client_company = client.company

            # A brand-new session (no messages yet) for a visitor we've seen before gets
            # primed with what we already know, so the bot doesn't re-ask settled
            # questions and doesn't ask for an email it already has (PRD-adjacent:
            # "each session is a new chat" but with continuity of what's been learned).
            if session.total_messages == 0 and session.visitor_id and session.surface == SurfaceType.PUBLIC:
                await self._prime_from_visitor_memory(state, session.visitor_id, session_id)

        self._cache(state)
        return state

    async def _prime_from_visitor_memory(
        self, state: ConversationState, visitor_id: str, session_id: str
    ) -> None:
        """Carry known facts and a light context note forward into a fresh chat."""
        memory = await db.get_visitor_memory(visitor_id, exclude_session_id=session_id)
        known = memory.get("known_fields") or {}
        if known:
            state.qualification_data.update({k: v for k, v in known.items() if v})
        if memory.get("email"):
            # We already have it — don't ask again this session.
            state.email_capture_asked = True

        summaries = memory.get("summaries") or []
        if summaries:
            state.visitor_memory = (
                "This visitor has chatted with us before. Prior conversation summary: "
                + " | ".join(summaries[:2])
            )

    def _cache(self, state: ConversationState):
        if len(self._sessions) >= MAX_CACHED_SESSIONS:
            # Drop the oldest inserted entry; it can be rehydrated from the DB if needed
            self._sessions.pop(next(iter(self._sessions)), None)
        self._sessions[state.session_id] = state

    # --- Public prospect surface ---

    async def process_message(
        self,
        request: ChatRequest,
        ip_address: str = "unknown",
    ) -> ChatResponse:
        """Process an incoming public chat message through the full pipeline."""
        state = await self.get_state(request.session_id)
        if request.source_page:
            state.source_page = request.source_page
        if request.visitor_id and not state.visitor_id:
            state.visitor_id = request.visitor_id
            await db.update_session(request.session_id, visitor_id=request.visitor_id)

        if request.consent_given and not state.consent_given:
            state.consent_given = True
            await db.update_session(request.session_id, consent_given=True)

        # Validate input before anything is stored
        is_safe, warning = content_guardrails.validate_input(request.message)
        if not is_safe:
            return ChatResponse(session_id=request.session_id, message=warning)

        # First contact: flag the consent notice for the widget to display alongside the
        # reply. Nothing personal is requested before it has been shown.
        first_contact = not state.consent_shown
        if first_contact:
            state.consent_shown = True
            state.consent_given = True
            await db.update_session(request.session_id, consent_given=True)

        await self._save_message(request.session_id, "user", request.message)
        state.messages.append({"role": "user", "content": request.message})
        state.message_count += 1

        # If we're waiting on the mandatory pre-booking email, resolve it first — this
        # takes priority over normal routing since nothing else matters until we have
        # somewhere to send the confirmation.
        ack_prefix: Optional[str] = None
        if state.booking_email_pending:
            response_data = await self._resolve_booking_email(request.message, state)
            intent = Intent.BOOKING
            state.current_intent = intent
        else:
            # If we're waiting on a reply to the soft email ask, resolve it first. Whatever
            # is left over from the visitor's message (if anything) still gets routed
            # normally — declining or giving an email doesn't swallow a real question.
            routed_message = request.message
            if state.email_capture_pending:
                state.email_capture_pending = False
                ack_prefix, routed_message = await self._resolve_email_capture(
                    request.message, state
                )

            if routed_message.strip():
                classification = await intent_classifier.classify(
                    message=routed_message,
                    conversation_history=state.messages,
                    current_state=state.current_intent.value,
                )
                intent = classification["intent"]

                # Once discovery has started, stay in it unless the visitor clearly leaves
                if state.in_intake and intent in (
                    Intent.QUALIFICATION, Intent.GENERAL, Intent.PROJECT_INTAKE, Intent.QUESTION
                ):
                    intent = Intent.PROJECT_INTAKE

                state.current_intent = intent
                response_data = await self._route_intent(intent, routed_message, state)
            else:
                # Pure decline/ack with nothing else in the message — nothing to route.
                intent = Intent.GENERAL
                response_data = {
                    "message": "What can I help you with today?",
                    "model_used": "template",
                    "cost": 0.0,
                }

        # A handler may redirect (e.g. the classifier says "general" but the message is a
        # project brief). Report where the turn actually went, not where it was filed.
        if response_data.get("intent_override"):
            intent = response_data["intent_override"]
            state.current_intent = intent

        if ack_prefix:
            response_data["message"] = f"{ack_prefix} {response_data['message']}"
            response_data["suggestions"] = []

        # First-ever message in this session: ask for an email, softly, once — unless we
        # already have it (cross-session memory) or the turn is an urgent flow.
        if (
            not state.email_capture_asked
            and not state.email_capture_pending
            and not state.booking_email_pending
            and state.message_count == 1
            and intent not in (Intent.DATA_DELETION, Intent.HUMAN_HANDOFF)
        ):
            response_data["message"] = response_data["message"].rstrip() + "\n\n" + EMAIL_ASK
            response_data["suggestions"] = EMAIL_ASK_SUGGESTIONS
            state.email_capture_asked = True
            state.email_capture_pending = True

        response_data["message"] = content_guardrails.sanitize_output(response_data["message"])
        response_data["message"] = content_guardrails.check_commitment_language(
            response_data["message"]
        )

        await self._save_message(
            session_id=request.session_id,
            role="assistant",
            content=response_data["message"],
            intent=intent.value,
            model_used=response_data.get("model_used"),
            cost=response_data.get("cost", 0.0),
        )
        state.messages.append({"role": "assistant", "content": response_data["message"]})
        state.message_count += 1
        state.total_cost += response_data.get("cost", 0.0)

        await db.increment_session_messages(
            request.session_id, cost=response_data.get("cost", 0.0)
        )

        await self._update_category(state)
        await self._maybe_notify_new_lead(state)
        generated_title = await self._maybe_generate_title(state)

        return ChatResponse(
            session_id=request.session_id,
            message=response_data["message"],
            intent=intent.value,
            source_link=response_data.get("source_link"),
            suggestions=response_data.get("suggestions") or [],
            show_booking=response_data.get("show_booking", False),
            booking_url=response_data.get("booking_url"),
            qualification_score=(
                state.qualification_score if state.qualification_score > 0 else None
            ),
            show_consent=first_contact,
            brief_ready=response_data.get("brief_ready", False),
            title=generated_title or state.title,
        )

    async def _resolve_email_capture(
        self, message: str, state: ConversationState
    ) -> "tuple[Optional[str], str]":
        """
        Interpret a reply to the soft email ask.

        Returns (acknowledgement_or_None, message_to_route). An email or a bare decline
        consumes the turn (nothing left to route); anything else is treated as the
        visitor having ignored the ask and just kept talking, so it's routed untouched.
        """
        email = extract_email(message)
        if email:
            await db.update_lead_fields(state.session_id, {"email": email})
            state.qualification_data["email"] = email
            leftover = strip_email(message, email)
            if len(leftover.split()) >= 3:
                return FOUND_ACK, leftover
            return FOUND_ACK, ""

        if looks_like_decline(message):
            return DECLINE_ACK, ""

        return None, message

    async def _offer_booking(
        self,
        state: ConversationState,
        lead_data: Dict[str, Any],
        qualification_score: float,
    ) -> Dict[str, Any]:
        """
        Produce the booking segment for a turn.

        If we already have an email on file, this is the real Calendly link. Otherwise
        it's a mandatory ask — unlike the soft capture earlier in the session, this one
        can't be declined, since a booking confirmation needs somewhere to go. The reply
        is picked up by _resolve_booking_email on the visitor's next message.
        """
        email = lead_data.get("email") or state.intake_data.get("email")
        if not is_valid_email(email or ""):
            state.booking_email_pending = True
            return {"message": BOOKING_EMAIL_ASK, "show_booking": False, "booking_url": None}

        state.booking_email_pending = False
        booking = booking_handler.generate_booking_response(
            lead_data=lead_data, qualification_score=qualification_score
        )
        return {
            "message": booking["message"],
            "show_booking": True,
            "booking_url": booking["booking_url"],
        }

    async def _resolve_booking_email(
        self, message: str, state: ConversationState
    ) -> Dict[str, Any]:
        """Interpret a reply to the mandatory pre-booking email ask."""
        email = extract_email(message)
        if not email:
            return {
                "message": BOOKING_EMAIL_RETRY,
                "show_booking": False,
                "suggestions": [],
                "model_used": "template",
                "cost": 0.0,
            }

        state.booking_email_pending = False
        # This reply also answers the soft ask, if it was somehow still outstanding —
        # don't leave it pending to misfire against a later, unrelated message.
        state.email_capture_asked = True
        state.email_capture_pending = False
        await db.update_lead_fields(state.session_id, {"email": email})
        state.qualification_data["email"] = email

        booking = booking_handler.generate_booking_response(
            lead_data=state.qualification_data,
            qualification_score=state.qualification_score,
        )
        state.booking_offered = True
        await db.update_session(state.session_id, status=SessionStatus.BOOKED.value)
        await self._maybe_email_booking(state)

        return {
            "message": f"Thanks! {booking['message']}",
            "show_booking": True,
            "booking_url": booking["booking_url"],
            "suggestions": [],
            "model_used": "template",
            "cost": 0.0,
        }

    async def _update_category(self, state: ConversationState) -> None:
        """
        Coarse classification for the dashboard: "lead" once there's real qualifying
        data or a discovery in progress, "query" for pure Q&A, "other" otherwise.
        Only writes to the database when the classification actually changes.
        """
        if (
            state.qualification_score > 0
            or state.in_intake
            or state.brief_generated
            or state.booking_offered
        ):
            category = "lead"
        elif state.current_intent in (Intent.QUESTION, Intent.GENERAL) and state.message_count >= 2:
            category = "query"
        else:
            category = "other"

        if category != state.category:
            state.category = category
            await db.update_session(state.session_id, category=category)

    async def _maybe_notify_new_lead(self, state: ConversationState) -> None:
        """
        Fire once per session, once name, email, a requirement, and a qualification
        score are all in hand — checked after every turn so it fires regardless of
        which handler (soft capture, booking gate, qualification, or intake) supplied
        the missing piece.

        "Requirement" comes from pain_point/goals/current_state if the conversation has
        surfaced one; if it hasn't after a few turns, an auto-generated transcript
        summary fills in instead rather than waiting indefinitely. Score is always
        computed fresh from whatever's known — it's a weighted composite, so it's never
        blank, just possibly low.
        """
        if state.lead_notified or not settings.lead_notification_email:
            return

        name = state.qualification_data.get("name") or state.intake_data.get("name")
        email = state.qualification_data.get("email") or state.intake_data.get("email")
        if not name or not is_valid_email(email or ""):
            return

        lead_data = {**state.qualification_data, **state.intake_data}
        requirement = (
            lead_data.get("pain_point") or lead_data.get("goals") or lead_data.get("current_state")
        )

        if not requirement:
            if state.message_count < LEAD_NOTIFY_SUMMARY_FALLBACK_AFTER:
                return  # give the conversation more turns to surface one naturally
            summary = await self._summarize_session(state.session_id)
            requirement = (summary or {}).get("summary")
            if not requirement:
                return  # nothing to report yet even via summary; try again next turn

        state.lead_notified = True
        lead_data["pain_point"] = requirement
        lead_data["qualification_score"] = lead_qualifier.score_from_data(lead_data)

        try:
            await email_service.notify_new_lead(
                to_email=settings.lead_notification_email,
                lead_data=lead_data,
                session_id=state.session_id,
            )
        except Exception as e:
            logger.error(f"Failed to send new-lead notification for {state.session_id}: {e}")

    async def _maybe_generate_title(self, state: ConversationState) -> Optional[str]:
        """Generate a short AI title for the chat-list panel, once, cheaply."""
        if state.title_generated or state.message_count < TITLE_MIN_MESSAGE_COUNT:
            return None
        state.title_generated = True

        transcript = "\n".join(
            f"{m['role'].upper()}: {m['content'][:300]}" for m in state.messages[:6]
        )
        result = await llm_router.generate(
            messages=[{
                "role": "user",
                "content": TITLE_GENERATION_PROMPT.format(transcript=transcript),
            }],
            task_type="summarization",
            temperature=0.4,
            max_tokens=24,
        )

        if not result.get("ok", True):
            # Router exhausted its fallbacks and returned the generic error message —
            # never let that become a chat title.
            return None

        title = self._clean_title(result.get("content", ""))
        if not title:
            return None

        state.title = title
        await db.update_session(state.session_id, title=title)
        if result.get("cost"):
            state.total_cost += result["cost"]
            await db.increment_session_messages(state.session_id, cost=result["cost"])
        return title

    @staticmethod
    def _clean_title(raw: str) -> Optional[str]:
        title = (raw or "").strip().strip('"\'` ').split("\n")[0].strip()
        title = title.rstrip(".")
        if not title or len(title) < 3:
            return None
        return title[:60]

    # --- Authenticated client surface (Phase 3) ---

    async def process_client_message(
        self,
        session_id: str,
        message: str,
        client_id: str,
        client_name: Optional[str],
        client_company: Optional[str],
        source_page: Optional[str] = None,
    ) -> ChatResponse:
        """
        Process a message on the authenticated client surface.

        This path never calls the intent classifier's prospect routes, never touches the
        public knowledge base, and never runs lead qualification (PRD 7.7).
        """
        state = await self.get_state(session_id)
        state.surface = SurfaceType.CLIENT
        state.client_id = client_id
        state.client_name = client_name
        state.client_company = client_company
        if source_page:
            state.source_page = source_page

        is_safe, warning = content_guardrails.validate_input(message)
        if not is_safe:
            return ChatResponse(session_id=session_id, message=warning)

        await self._save_message(session_id, "user", message)
        state.messages.append({"role": "user", "content": message})
        state.message_count += 1

        result = await client_support.handle(message, state)
        result["message"] = content_guardrails.check_commitment_language(result["message"])

        await self._save_message(
            session_id=session_id,
            role="assistant",
            content=result["message"],
            intent="client_support",
            model_used=result.get("model_used"),
            cost=result.get("cost", 0.0),
        )
        state.messages.append({"role": "assistant", "content": result["message"]})
        state.message_count += 1
        await db.increment_session_messages(session_id, cost=result.get("cost", 0.0))

        return ChatResponse(
            session_id=session_id,
            message=result["message"],
            intent="client_support",
            escalated=result.get("escalated", False),
        )

    # --- Routing ---

    async def _route_intent(
        self,
        intent: Intent,
        message: str,
        state: ConversationState,
    ) -> Dict[str, Any]:
        """Route to the appropriate handler based on intent."""
        if intent == Intent.GREETING:
            return await self._handle_greeting(message, state)

        if intent == Intent.QUESTION:
            return await self._handle_question(message, state)

        if intent == Intent.PROJECT_INTAKE:
            return await self._handle_intake(message, state)

        if intent in (Intent.QUALIFICATION, Intent.GENERAL):
            # A message that reads like a concrete project brief goes to the discovery
            # agent rather than slot-filling or a generic reply. The cheap classifier
            # routinely files "we need a rebuild" under general, and that is exactly the
            # message intake exists to catch (PRD 7.6 trigger).
            if looks_like_project_request(message):
                return await self._handle_intake(message, state)
            if intent == Intent.GENERAL:
                return await self._handle_general(message, state)
            return await self._handle_qualification(message, state)

        if intent == Intent.BOOKING:
            return await self._handle_booking(message, state)

        if intent == Intent.FAREWELL:
            return await self._handle_farewell(message, state)

        if intent == Intent.DATA_DELETION:
            return await self._handle_data_deletion(message, state)

        if intent == Intent.HUMAN_HANDOFF:
            return await self._handle_human_handoff(message, state)

        return await self._handle_general(message, state)

    async def _handle_greeting(self, message: str, state: ConversationState) -> Dict[str, Any]:
        """Handle greeting messages."""
        if not state.consent_given:
            return {"message": CONSENT_NOTICE, "model_used": "template", "cost": 0.0}

        greeting_response = """Hello! 👋 Welcome to Abacus Digital. I'm here to help you learn about our services and find the right solution for your business.

Here's what I can help with:
• **Learn about our services** — Web Design, AI & Automation, Lead Generation, and more
• **Get recommendations** — Tell me about your business challenge and I'll suggest the right approach
• **Book a call** — Schedule a free discovery call with our team

What brings you here today?"""

        return {
            "message": greeting_response,
            "model_used": "template",
            "cost": 0.0,
            "suggestions": GREETING_SUGGESTIONS,
        }

    async def _handle_question(self, message: str, state: ConversationState) -> Dict[str, Any]:
        """Handle questions using RAG."""
        result = await rag_engine.answer(
            query=message,
            conversation_history=state.messages,
        )

        for source in result["sources"]:
            if source not in state.service_interests:
                state.service_interests.append(source)

        if state.service_interests:
            await db.update_lead_fields(
                state.session_id,
                {"service_interest": ", ".join(state.service_interests[:3])},
            )

        return {
            "message": result["answer"],
            "source_link": result.get("source_link"),
            "model_used": result["model_used"],
            "cost": result["cost"],
            "suggestions": self._followup_suggestions(result),
        }

    @staticmethod
    def _followup_suggestions(rag_result: Dict[str, Any]) -> List[str]:
        """
        Deterministic (no extra LLM call) follow-up chips for a RAG answer — tailored to
        whatever it was grounded in when we have a source, a generic-but-relevant set
        when we don't. Every RAG turn gets something rather than a chip-less dead end.
        """
        sources = rag_result.get("sources") or []
        if sources:
            return [f"Tell me more about {sources[0]}", "Book a call", "See other services"]
        return QUESTION_FOLLOWUP_SUGGESTIONS

    async def _handle_qualification(self, message: str, state: ConversationState) -> Dict[str, Any]:
        """Handle conversational lead qualification."""
        result = await lead_qualifier.qualify(user_message=message, state=state)

        # Persist progressively so an abandoned chat still leaves usable data (PRD 7.5)
        fields = dict(result["extracted_data"])
        if result["qualification_score"]:
            fields["qualification_score"] = result["qualification_score"]
        if fields:
            await db.update_lead_fields(state.session_id, fields)

        response: Dict[str, Any] = {
            "message": result["response"],
            "model_used": result["model_used"],
            "cost": result["cost"],
        }

        if result["should_offer_booking"]:
            booking = await self._offer_booking(
                state, state.qualification_data, result["qualification_score"]
            )
            response["message"] += "\n\n" + booking["message"]
            response["show_booking"] = booking["show_booking"]
            if booking["show_booking"]:
                response["booking_url"] = booking["booking_url"]
                response["suggestions"] = ["Ask another question", "That's all, thanks"]
                state.booking_offered = True
                await db.update_session(state.session_id, status=SessionStatus.QUALIFIED.value)
                await self._maybe_email_booking(state)
            else:
                response["suggestions"] = []

        elif result["is_qualified"] is False and state.message_count > 6:
            # Below threshold after a real conversation: give self-serve resources (PRD 7.2)
            response["message"] += "\n\n" + booking_handler.generate_self_serve_response(
                state.qualification_data
            )
            response["suggestions"] = SELF_SERVE_SUGGESTIONS
        else:
            # The qualifier already generated suggestions matched to whatever it just
            # asked (with a field-based fallback baked in) — use them as-is.
            response["suggestions"] = result.get("suggestions", [])

        return response

    async def _handle_intake(self, message: str, state: ConversationState) -> Dict[str, Any]:
        """Run the Phase 2 agentic discovery flow."""
        state.in_intake = True

        turn = await intake_agent.run_turn(message, state)
        cost = turn["cost"]

        # Discovery fields that map onto the lead record are persisted as they arrive,
        # along with a running score — an abandoned discovery still leaves a ranked lead
        # rather than one sitting at zero (PRD 7.5).
        if turn["extracted"]:
            state.qualification_data.update(turn["extracted"])
            fields = dict(turn["extracted"])

            score = lead_qualifier.score_from_data(state.qualification_data)
            state.qualification_score = score
            state.is_qualified = score >= QUALIFICATION_THRESHOLD
            if score > 0:
                fields["qualification_score"] = score

            await db.update_lead_fields(state.session_id, fields)

        response: Dict[str, Any] = {
            "message": turn["response"],
            "model_used": turn["model_used"],
            "cost": cost,
            "intent_override": Intent.PROJECT_INTAKE,
            # Generated alongside the question intake_agent just asked, so it can't
            # drift out of sync with it the way a field-order guess could.
            "suggestions": turn.get("suggestions", []),
        }

        if turn["discovery_complete"] and not state.brief_generated:
            brief_result = await intake_agent.generate_brief(state)
            brief = brief_result["brief"]
            cost += brief_result["cost"]

            await db.save_brief(brief)
            state.brief_generated = True
            state.in_intake = False

            score = lead_qualifier.score_from_data(state.qualification_data)
            state.qualification_score = score
            state.is_qualified = score >= QUALIFICATION_THRESHOLD

            await db.update_lead_fields(state.session_id, {
                "service_interest": ", ".join(brief.recommended_services[:3]),
                "qualification_score": score,
                "budget_band": brief.budget_band,
                "timeline": brief.timeline,
                "next_step": "Review project brief and run discovery call",
            })
            await db.update_session(state.session_id, status=SessionStatus.QUALIFIED.value)

            booking = await self._offer_booking(state, state.qualification_data, score)

            response["message"] = (
                f"{turn['response']}\n\n"
                f"Here's the brief I've put together for our team:\n\n"
                f"**{brief.title}**\n{brief.summary}\n\n"
                f"**Recommended:** {', '.join(brief.recommended_services)}\n"
                f"{brief.bundle_rationale}\n\n"
                f"{booking['message']}"
            )
            response["show_booking"] = booking["show_booking"]
            if booking["show_booking"]:
                response["booking_url"] = booking["booking_url"]
                response["suggestions"] = ["Ask another question", "That's all, thanks"]
            else:
                response["suggestions"] = []
            response["brief_ready"] = True
            response["cost"] = cost
            state.booking_offered = True

            await self._email_brief(state, brief)
            await crm_sync.sync_session(state.session_id)

        return response

    async def _handle_booking(self, message: str, state: ConversationState) -> Dict[str, Any]:
        """Handle direct booking requests."""
        booking = await self._offer_booking(
            state, state.qualification_data, state.qualification_score
        )
        state.booking_offered = True

        if booking["show_booking"]:
            await db.update_session(state.session_id, status=SessionStatus.BOOKED.value)
            await self._maybe_email_booking(state)

        result: Dict[str, Any] = {
            "message": booking["message"],
            "show_booking": booking["show_booking"],
            "booking_url": booking["booking_url"],
            "model_used": "template",
            "cost": 0.0,
            # A required email ask has no sensible multiple-choice chips; once the
            # actual link is shown, offer a couple of ways to keep the chat going.
            "suggestions": [] if not booking["show_booking"] else ["Ask another question", "That's all, thanks"],
        }
        return result

    async def _handle_farewell(self, message: str, state: ConversationState) -> Dict[str, Any]:
        """Handle farewell messages and close out the session record."""
        farewell = "Thanks for chatting with us! If you need anything else, feel free to come back anytime. Have a great day! 😊"

        if state.qualification_data and not state.booking_offered:
            farewell += (
                f"\n\nWhenever you're ready, you can "
                f"[book a discovery call]({settings.calendly_url}) with our team."
            )

        await self.finalize_session(state.session_id, SessionStatus.COMPLETED)

        return {
            "message": farewell,
            "model_used": "template",
            "cost": 0.0,
            "suggestions": FAREWELL_SUGGESTIONS,
        }

    async def _handle_data_deletion(self, message: str, state: ConversationState) -> Dict[str, Any]:
        """Handle data deletion requests (PRD 7.5)."""
        lead = await db.get_lead_by_session(state.session_id)

        if lead and lead.email:
            deleted = await db.delete_by_email(lead.email)
            self._sessions.pop(state.session_id, None)
            return {
                "message": (
                    f"Done — I've deleted the {deleted} record(s) associated with {lead.email}, "
                    "including this conversation's transcript. If you'd like anything else removed, "
                    "email us at https://www.abacusdigital.net/contact."
                ),
                "model_used": "template",
                "cost": 0.0,
            }

        return {
            "message": (
                "I can delete everything from this conversation right away, or all records tied to "
                "your email address. Reply with the email address you'd like removed, or say "
                "\"delete this chat\" and I'll clear this session. You can also reach us at "
                "https://www.abacusdigital.net/contact."
            ),
            "model_used": "template",
            "cost": 0.0,
        }

    async def _handle_human_handoff(self, message: str, state: ConversationState) -> Dict[str, Any]:
        """A prospect explicitly asked for a person."""
        await db.create_escalation(
            session_id=state.session_id,
            reason="Prospect asked to speak with a human",
            transcript_summary=client_support.transcript_digest(state),
        )
        await db.update_session(state.session_id, status=SessionStatus.ESCALATED.value)

        return {
            "message": (
                "Of course — I've flagged this for our team. The fastest route is to "
                f"[book a call]({settings.calendly_url}), or you can reach us any time at "
                "https://www.abacusdigital.net/contact. "
                "If you leave your email here, someone will follow up directly."
            ),
            "model_used": "template",
            "cost": 0.0,
        }

    async def _handle_general(self, message: str, state: ConversationState) -> Dict[str, Any]:
        """General messages: try RAG first, then a conversational reply."""
        rag_result = await rag_engine.answer(
            query=message,
            conversation_history=state.messages,
        )

        if rag_result["grounded"] and rag_result["sources"]:
            return {
                "message": rag_result["answer"],
                "source_link": rag_result.get("source_link"),
                "model_used": rag_result["model_used"],
                "cost": rag_result["cost"],
                "suggestions": self._followup_suggestions(rag_result),
            }

        memory_note = f"\n\n{state.visitor_memory}" if state.visitor_memory else ""
        messages = [{
            "role": "system",
            "content": f"""{SYSTEM_PROMPT_BASE}{memory_note}

The user sent a general message. Respond helpfully and try to guide them toward:
1. Learning about Abacus Digital's services
2. Sharing their business needs so you can help
3. Booking a discovery call if appropriate

Be concise and friendly.""",
        }]
        messages.extend(state.messages[-6:])
        messages.append({"role": "user", "content": message})

        result = await llm_router.generate(
            messages=messages,
            task_type="general",
            temperature=0.7,
            max_tokens=500,
        )

        return {
            "message": result["content"],
            "model_used": result["model_used"],
            "cost": result["cost"],
            "suggestions": GREETING_SUGGESTIONS,
        }

    # --- Session close-out ---

    async def finalize_session(
        self,
        session_id: str,
        status: SessionStatus = SessionStatus.COMPLETED,
    ) -> Dict[str, Any]:
        """
        Close a session: summarise the transcript, write it onto the lead, and mirror
        to the CRM. Safe to call more than once.
        """
        session = await db.get_session(session_id)
        if not session:
            return {"finalized": False, "reason": "unknown session"}
        if session.ended_at:
            return {"finalized": False, "reason": "already finalized"}

        await db.update_session(
            session_id,
            status=status.value,
            ended_at=datetime.utcnow().isoformat(),
        )

        summary = await self._summarize_session(session_id)
        if summary:
            await db.update_session(
                session_id,
                summary=summary.get("summary"),
                next_step=summary.get("next_step"),
            )
            fields = {
                "transcript_summary": summary.get("summary"),
                "next_step": summary.get("next_step"),
            }
            if summary.get("service_interest"):
                fields["service_interest"] = summary["service_interest"]
            await db.update_lead_fields(session_id, fields)

        sync = await crm_sync.sync_session(session_id)
        self._sessions.pop(session_id, None)

        return {"finalized": True, "summary": summary, "crm": sync}

    async def _summarize_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Summarise a transcript for the sales team (PRD 7.4)."""
        messages = await db.get_session_messages(session_id)
        user_turns = [m for m in messages if m.role == "user"]
        if len(user_turns) < 1:
            return None

        transcript = "\n".join(
            f"{m.role.upper()}: {m.content[:600]}" for m in messages[-30:]
        )

        result = await llm_router.generate_json(
            messages=[{
                "role": "user",
                "content": TRANSCRIPT_SUMMARY_PROMPT.format(transcript=transcript),
            }],
            task_type="summarization",
            temperature=0.2,
            max_tokens=400,
        )

        if not result["data"]:
            # Still record something useful rather than leaving the field empty
            return {
                "summary": f"{len(user_turns)} visitor message(s); automatic summary unavailable.",
                "service_interest": None,
                "next_step": "Review transcript manually",
            }

        data = result["data"]
        return {
            "summary": data.get("summary"),
            "service_interest": data.get("service_interest"),
            "next_step": data.get("next_step"),
        }

    async def sweep_abandoned(self, idle_minutes: int = 30) -> int:
        """Finalize sessions that went quiet, so abandoned chats still produce a record."""
        stale = await db.get_stale_sessions(minutes=idle_minutes)
        for session_id in stale:
            try:
                await self.finalize_session(session_id, SessionStatus.ABANDONED)
            except Exception as e:
                logger.error(f"Failed finalizing abandoned session {session_id}: {e}")
        if stale:
            logger.info(f"Finalized {len(stale)} abandoned session(s)")
        return len(stale)

    # --- Email helpers ---

    async def _email_brief(self, state: ConversationState, brief) -> None:
        """Queue the brief confirmation email if we have an address (PRD 7.6)."""
        email = state.intake_data.get("email") or state.qualification_data.get("email")
        if not is_valid_email(email or ""):
            return
        try:
            await email_service.draft_intake_followup(
                to_email=email,
                brief=brief,
                session_id=state.session_id,
                visitor_name=state.intake_data.get("name") or state.qualification_data.get("name"),
            )
        except Exception as e:
            logger.error(f"Failed to draft intake follow-up email: {e}")

    async def _maybe_email_booking(self, state: ConversationState) -> None:
        """Confirm booking details by email when we have one (PRD 7.3)."""
        email = state.qualification_data.get("email") or state.intake_data.get("email")
        if not is_valid_email(email or ""):
            return
        try:
            await email_service.draft_booking_confirmation(
                to_email=email,
                session_id=state.session_id,
                visitor_name=state.qualification_data.get("name"),
            )
        except Exception as e:
            logger.error(f"Failed to draft booking confirmation email: {e}")

    # --- Persistence helpers ---

    async def _save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        intent: str = None,
        model_used: str = None,
        cost: float = 0.0,
    ):
        """Save a message to the database."""
        await db.save_message(MessageRecord(
            session_id=session_id,
            role=role,
            content=content,
            intent=intent,
            model_used=model_used,
            cost=cost,
        ))

    def cleanup_session(self, session_id: str):
        """Remove a session from memory (but keep it in the database)."""
        self._sessions.pop(session_id, None)


# Singleton
chat_orchestrator = ChatOrchestrator()
