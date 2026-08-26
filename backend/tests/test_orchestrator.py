"""Orchestrator flow tests with the LLM router and vector store stubbed out."""

import pytest

from app.models import (
    ChatRequest, SessionRecord, SessionStatus, SurfaceType, ConversationState,
)
from conftest import FakeRouter


@pytest.fixture(autouse=True)
def wire_database(database):
    """
    Reset in-memory conversation-state cache between tests.

    No monkeypatching needed here anymore: chat_orchestrator/client_support/crm_sync/
    email_service all import the real `db` singleton directly, and the `database`
    fixture truncates its tables before each test rather than swapping in a different
    instance — so every module already sees the clean, test-scoped state.
    """
    from app import chat_orchestrator as orch_mod
    orch_mod.chat_orchestrator._sessions.clear()
    return database


@pytest.fixture
def orchestrator():
    from app.chat_orchestrator import chat_orchestrator
    chat_orchestrator._sessions.clear()
    return chat_orchestrator


async def _new_session(database, surface=SurfaceType.PUBLIC, client_id=None):
    session = SessionRecord(surface=surface, client_id=client_id)
    await database.create_session(session)
    return session.id


@pytest.mark.asyncio
async def test_question_uses_rag_and_persists_transcript(orchestrator, database, monkeypatch):
    from app import chat_orchestrator as mod

    async def fake_answer(query, conversation_history=None, top_k=5):
        return {
            "answer": "We build responsive B2B sites.",
            "sources": ["Web Design & Development"],
            "source_link": {
                "label": "Web Design & Development",
                "url": "https://www.abacusdigital.net/all-services/web-design",
            },
            "model_used": "fake", "cost": 0.0002, "grounded": True,
        }

    monkeypatch.setattr(mod.rag_engine, "answer", fake_answer)
    monkeypatch.setattr(
        mod.intent_classifier, "classify",
        lambda **kwargs: _as_coro({"intent": mod.Intent.QUESTION, "confidence": 0.9}),
    )

    session_id = await _new_session(database)
    # First turn's suggestions get overwritten by the soft email ask — assert the
    # follow-up chips on a second turn, once that one-time ask is out of the way.
    await orchestrator.process_message(
        ChatRequest(session_id=session_id, message="Hi there")
    )
    response = await orchestrator.process_message(
        ChatRequest(session_id=session_id, message="Do you build B2B websites?")
    )

    assert "responsive B2B sites" in response.message
    assert response.source_link == {
        "label": "Web Design & Development",
        "url": "https://www.abacusdigital.net/all-services/web-design",
    }
    # A grounded answer always carries follow-up chips, tailored to its source
    assert response.suggestions
    assert "Web Design & Development" in response.suggestions[0]

    transcript = await database.get_session_transcript(session_id)
    assert [t["role"] for t in transcript] == ["user", "assistant", "user", "assistant"]

    # Interest is captured even though the visitor never entered qualification
    lead = await database.get_lead_by_session(session_id)
    assert lead.service_interest == "Web Design & Development"


@pytest.mark.asyncio
async def test_service_names_in_bot_replies_become_real_links(orchestrator, database, monkeypatch):
    """
    Any service name the bot mentions in its own generated text — not just the
    dedicated source_link — should turn into a markdown link to that service's real
    page, wherever in the conversation it comes up.
    """
    from app import chat_orchestrator as mod

    async def fake_answer(query, conversation_history=None, top_k=5):
        return {
            "answer": "Cybersecurity and Brand Identity would both help here.",
            "sources": [], "source_link": None,
            "model_used": "fake", "cost": 0.0002, "grounded": True,
        }

    monkeypatch.setattr(mod.rag_engine, "answer", fake_answer)
    monkeypatch.setattr(
        mod.intent_classifier, "classify",
        lambda **kwargs: _as_coro({"intent": mod.Intent.QUESTION, "confidence": 0.9}),
    )

    session_id = await _new_session(database)
    response = await orchestrator.process_message(
        ChatRequest(session_id=session_id, message="What would help my business?")
    )

    assert "[Cybersecurity](https://www.abacusdigital.net/all-services/cybersecurity)" in response.message
    assert "[Brand Identity](https://www.abacusdigital.net/all-services/brand-identity)" in response.message


@pytest.mark.asyncio
async def test_unsafe_input_is_never_stored(orchestrator, database):
    session_id = await _new_session(database)

    response = await orchestrator.process_message(
        ChatRequest(session_id=session_id, message="my ssn is 123-45-6789")
    )

    assert "don't share" in response.message
    assert await database.get_session_transcript(session_id) == []


@pytest.mark.asyncio
async def test_qualification_persists_partial_data_and_score(orchestrator, database, monkeypatch):
    from app import chat_orchestrator as mod

    monkeypatch.setattr(
        mod.intent_classifier, "classify",
        lambda **kwargs: _as_coro({"intent": mod.Intent.QUALIFICATION, "confidence": 0.9}),
    )
    monkeypatch.setattr(mod.lead_qualifier, "llm_router", FakeRouter(), raising=False)

    fake = FakeRouter(json_responses=[{
        "response": "Got it — what's your timeline?",
        "extracted_data": {
            "business_type": "manufacturing",
            "pain_point": "no inbound leads",
            "budget_band": "5k_to_15k",
        },
        "enough_data_for_booking": False,
    }])
    monkeypatch.setattr("app.lead_qualifier.llm_router", fake)

    session_id = await _new_session(database)
    await orchestrator.process_message(ChatRequest(
        session_id=session_id,
        message="We manufacture valves and we're not getting any enquiries",
    ))

    lead = await database.get_lead_by_session(session_id)
    assert lead.business_type == "manufacturing"
    assert lead.budget_band == "5k_to_15k"
    # Score is written as data arrives, not only at the end (PRD 7.5)
    assert lead.qualification_score > 0


@pytest.mark.asyncio
async def test_suggestions_match_whatever_question_the_model_actually_asked(
    orchestrator, database, monkeypatch
):
    """
    Regression test: suggestion chips must come from the same model call that produced
    the question, not a fixed field-order guess. Previously a model question about
    social media / online presence could still get budget-band chips attached, because
    chips were picked by "what field is structurally next" rather than by what was
    actually asked.
    """
    from app import chat_orchestrator as mod

    monkeypatch.setattr(
        mod.intent_classifier, "classify",
        lambda **kwargs: _as_coro({"intent": mod.Intent.QUALIFICATION, "confidence": 0.9}),
    )
    fake = FakeRouter(json_responses=[
        {
            "response": "Got it, thanks for sharing that.",
            "extracted_data": {},
            "enough_data_for_booking": False,
            "suggested_replies": [],
        },
        {
            "response": (
                "Attracting more local customers is something a well-designed website can "
                "help with. Do you currently have any online presence, like a social media "
                "page or a listing on Google Maps?"
            ),
            "extracted_data": {"business_type": "retail", "pain_point": "not enough local customers"},
            "enough_data_for_booking": False,
            "suggested_replies": ["Yes, social media", "Yes, Google Maps", "No online presence yet"],
        },
    ])
    monkeypatch.setattr("app.lead_qualifier.llm_router", fake)

    session_id = await _new_session(database)
    # First turn's suggestions get overwritten by the soft email ask — the mismatch bug
    # only shows up once that one-time ask is out of the way.
    await orchestrator.process_message(ChatRequest(
        session_id=session_id, message="Hi, I run a local shop",
    ))
    response = await orchestrator.process_message(ChatRequest(
        session_id=session_id,
        message="Yes, that's exactly right",
    ))

    assert response.suggestions == ["Yes, social media", "Yes, Google Maps", "No online presence yet"]
    # Must NOT have fallen back to budget-band chips, which don't match this question
    assert "Under $1k" not in response.suggestions


@pytest.mark.asyncio
async def test_booking_is_withheld_until_an_email_is_given(orchestrator, database, monkeypatch):
    """
    A qualified lead with no email on file must be asked for one before the Calendly
    link appears; giving it on the next turn unblocks the real booking response.
    """
    from app import chat_orchestrator as mod

    monkeypatch.setattr(
        mod.intent_classifier, "classify",
        lambda **kwargs: _as_coro({"intent": mod.Intent.QUALIFICATION, "confidence": 0.9}),
    )
    fake = FakeRouter(json_responses=[{
        "response": "Great, that's everything I need.",
        "extracted_data": {
            "business_type": "manufacturing",
            "pain_point": "no inbound leads",
            "budget_band": "over_50k",
            "timeline": "immediate",
            "decision_maker": "yes",
        },
        "enough_data_for_booking": True,
    }])
    monkeypatch.setattr("app.lead_qualifier.llm_router", fake)

    session_id = await _new_session(database)
    first = await orchestrator.process_message(ChatRequest(
        session_id=session_id,
        message="We manufacture valves, ready to start immediately, budget's not an issue",
    ))

    assert first.show_booking is False
    assert "email" in first.message.lower()

    state = orchestrator._sessions[session_id]
    assert state.booking_email_pending is True

    second = await orchestrator.process_message(ChatRequest(
        session_id=session_id,
        message="sure, it's jane@example.com",
    ))

    assert second.show_booking is True
    assert second.booking_url

    lead = await database.get_lead_by_session(session_id)
    assert lead.email == "jane@example.com"


@pytest.mark.asyncio
async def test_team_is_notified_once_name_and_email_are_both_known(
    orchestrator, database, monkeypatch
):
    """
    The internal lead-notification email should fire exactly once, the turn both a
    name and an email become known — not before, and not again on a later turn.
    """
    from app import chat_orchestrator as mod

    monkeypatch.setattr(mod.settings, "lead_notification_email", "sales@abacusdigital.net")

    monkeypatch.setattr(
        mod.intent_classifier, "classify",
        lambda **kwargs: _as_coro({"intent": mod.Intent.PROJECT_INTAKE, "confidence": 0.9}),
    )

    async def fake_turn(message, state):
        state.intake_data.update({
            "name": "Priya Shah", "business_type": "manufacturing",
            "pain_point": "quoting takes too long",
        })
        return {
            "response": "Thanks Priya! What's your email so the team can follow up?",
            "extracted": {
                "name": "Priya Shah", "business_type": "manufacturing",
                "pain_point": "quoting takes too long",
            },
            "discovery_complete": False,
            "model_used": "fake", "cost": 0.001,
        }

    monkeypatch.setattr(mod.intake_agent, "run_turn", fake_turn)

    session_id = await _new_session(database)
    await orchestrator.process_message(ChatRequest(
        session_id=session_id,
        message="We're a manufacturing company, I'm Priya",
    ))

    # Name only so far — nothing sent yet
    assert await database.get_emails() == []

    async def fake_turn_with_email(message, state):
        state.intake_data.update({"email": "priya@example.com"})
        return {
            "response": "Got it, thanks!",
            "extracted": {"email": "priya@example.com"},
            "discovery_complete": False,
            "model_used": "fake", "cost": 0.001,
        }

    monkeypatch.setattr(mod.intake_agent, "run_turn", fake_turn_with_email)

    await orchestrator.process_message(ChatRequest(
        session_id=session_id,
        message="priya@example.com",
    ))

    emails = await database.get_emails()
    assert len(emails) == 1
    assert emails[0].to_email == "sales@abacusdigital.net"
    assert "Priya Shah" in emails[0].body
    assert "quoting takes too long" in emails[0].body
    # Score is computed fresh from whatever's known, never left blank
    assert "Qualification score: —" not in emails[0].body


@pytest.mark.asyncio
async def test_lead_notification_waits_for_a_requirement_then_falls_back_to_a_summary(
    orchestrator, database, monkeypatch
):
    """
    If a requirement (pain point / goals) never naturally surfaces, the notification
    must not fire on name+email alone with a blank requirement — it should wait a few
    turns, then fall back to an auto-generated transcript summary rather than never
    sending at all.
    """
    from app import chat_orchestrator as mod

    monkeypatch.setattr(mod.settings, "lead_notification_email", "sales@abacusdigital.net")
    monkeypatch.setattr(mod, "LEAD_NOTIFY_SUMMARY_FALLBACK_AFTER", 4)

    monkeypatch.setattr(
        mod.intent_classifier, "classify",
        lambda **kwargs: _as_coro({"intent": mod.Intent.QUALIFICATION, "confidence": 0.9}),
    )
    fake = FakeRouter(json_responses=[
        {
            "response": "Nice to meet you, Sam!",
            "extracted_data": {"name": "Sam", "email": "sam@example.com"},
            "enough_data_for_booking": False,
        },
        {
            "response": "Understood, thanks.",
            "extracted_data": {},
            "enough_data_for_booking": False,
        },
    ])
    monkeypatch.setattr("app.lead_qualifier.llm_router", fake)

    session_id = await _new_session(database)
    await orchestrator.process_message(ChatRequest(
        session_id=session_id, message="Hi, I'm Sam, my email is sam@example.com",
    ))
    # Name + email known, but no requirement yet, and below the turn threshold
    assert await database.get_emails() == []

    await orchestrator.process_message(ChatRequest(
        session_id=session_id, message="Sure, sounds fine",
    ))

    emails = await database.get_emails()
    assert len(emails) == 1
    assert emails[0].to_email == "sales@abacusdigital.net"
    # No real summary model is configured in tests, so the deterministic placeholder
    # is what should have been used as the requirement.
    assert "automatic summary unavailable" in emails[0].body

    # A third turn must not send a second notification
    await orchestrator.process_message(ChatRequest(
        session_id=session_id,
        message="Anything else I should know?",
    ))
    assert len(await database.get_emails()) == 1


@pytest.mark.asyncio
async def test_intake_completion_produces_and_stores_a_brief(orchestrator, database, monkeypatch):
    from app import chat_orchestrator as mod
    from app.models import ProjectBrief

    monkeypatch.setattr(
        mod.intent_classifier, "classify",
        lambda **kwargs: _as_coro({"intent": mod.Intent.PROJECT_INTAKE, "confidence": 0.9}),
    )

    async def fake_turn(message, state):
        state.intake_data.update({
            "goals": "double inbound enquiries",
            "current_state": "old WordPress site",
            "budget_band": "15k_to_50k",
            "timeline": "short_term",
            "email": "dana@example.com",
        })
        return {
            "response": "That's everything I need.",
            "extracted": {
                "business_type": "manufacturing", "email": "dana@example.com",
                "budget_band": "15k_to_50k", "timeline": "short_term",
            },
            "discovery_complete": True,
            "model_used": "fake", "cost": 0.001,
        }

    async def fake_brief(state):
        return {
            "brief": ProjectBrief(
                session_id=state.session_id,
                title="Manufacturing site rebuild",
                summary="Rebuild to drive inbound enquiries.",
                goals=["double inbound enquiries"],
                budget_band="15k_to_50k",
                timeline="short_term",
                recommended_services=["Web Design & Development",
                                      "Lead Generation & Performance Marketing"],
                bundle_rationale="The site needs traffic to convert.",
            ),
            "match": {}, "model_used": "fake", "cost": 0.002,
        }

    monkeypatch.setattr(mod.intake_agent, "run_turn", fake_turn)
    monkeypatch.setattr(mod.intake_agent, "generate_brief", fake_brief)

    session_id = await _new_session(database)
    response = await orchestrator.process_message(ChatRequest(
        session_id=session_id,
        message="We need a new website to bring in more enquiries",
    ))

    assert response.brief_ready is True
    assert response.show_booking is True

    brief = await database.get_brief_by_session(session_id)
    assert brief.title == "Manufacturing site rebuild"
    assert "Lead Generation & Performance Marketing" in brief.recommended_services

    session = await database.get_session(session_id)
    assert session.status == SessionStatus.QUALIFIED

    # A follow-up email is drafted for approval, not sent (PRD 7.6)
    emails = await database.get_emails()
    assert emails and emails[0].to_email == "dana@example.com"
    assert emails[0].status.value == "pending_approval"


@pytest.mark.asyncio
async def test_intake_scores_the_lead_before_discovery_finishes(
    orchestrator, database, monkeypatch
):
    """An abandoned discovery must still leave a ranked lead, not one sitting at zero."""
    from app import chat_orchestrator as mod

    monkeypatch.setattr(
        mod.intent_classifier, "classify",
        lambda **kwargs: _as_coro({"intent": mod.Intent.PROJECT_INTAKE, "confidence": 0.9}),
    )

    async def partial_turn(message, state):
        extracted = {"budget_band": "15k_to_50k", "timeline": "short_term",
                     "business_type": "manufacturing"}
        state.intake_data.update(extracted)
        return {"response": "And what does success look like?", "extracted": extracted,
                "discovery_complete": False, "model_used": "fake", "cost": 0.001}

    monkeypatch.setattr(mod.intake_agent, "run_turn", partial_turn)

    session_id = await _new_session(database)
    response = await orchestrator.process_message(ChatRequest(
        session_id=session_id, message="We need our quoting process automated",
    ))

    assert response.brief_ready is False
    lead = await database.get_lead_by_session(session_id)
    assert lead.budget_band == "15k_to_50k"
    assert lead.qualification_score > 0


@pytest.mark.asyncio
async def test_finalize_session_writes_summary_and_next_step(orchestrator, database, monkeypatch):
    from app import chat_orchestrator as mod

    monkeypatch.setattr("app.chat_orchestrator.llm_router", FakeRouter(json_responses=[{
        "summary": "Manufacturer exploring a site rebuild; left before booking.",
        "service_interest": "Web Design & Development",
        "next_step": "Send manufacturing case studies",
    }]))

    session_id = await _new_session(database)
    from app.models import MessageRecord
    await database.save_message(MessageRecord(
        session_id=session_id, role="user", content="we need a new site",
    ))

    result = await orchestrator.finalize_session(session_id, SessionStatus.ABANDONED)
    assert result["finalized"]

    session = await database.get_session(session_id)
    assert session.status == SessionStatus.ABANDONED
    assert session.ended_at
    assert "site rebuild" in session.summary

    lead = await database.get_lead_by_session(session_id)
    assert lead.next_step == "Send manufacturing case studies"

    # Finalizing twice must not double-write
    assert (await orchestrator.finalize_session(session_id))["finalized"] is False


@pytest.mark.asyncio
async def test_state_rehydrates_after_a_cold_start(orchestrator, database, monkeypatch):
    from app import chat_orchestrator as mod
    from app.models import MessageRecord

    session_id = await _new_session(database)
    await database.save_message(MessageRecord(session_id=session_id, role="user", content="hello"))
    await database.save_message(MessageRecord(
        session_id=session_id, role="assistant", content="hi there",
    ))
    await database.update_lead_fields(session_id, {
        "business_type": "logistics", "qualification_score": 0.55,
    })

    # Simulate a process restart: in-memory cache is empty
    orchestrator._sessions.clear()
    state = await orchestrator.get_state(session_id)

    assert len(state.messages) == 2
    assert state.qualification_data["business_type"] == "logistics"
    assert state.qualification_score == pytest.approx(0.55)
    assert state.is_qualified is True


@pytest.mark.asyncio
async def test_data_deletion_intent_removes_the_record(orchestrator, database, monkeypatch):
    from app import chat_orchestrator as mod

    monkeypatch.setattr(
        mod.intent_classifier, "classify",
        lambda **kwargs: _as_coro({"intent": mod.Intent.DATA_DELETION, "confidence": 0.95}),
    )

    session_id = await _new_session(database)
    await database.update_lead_fields(session_id, {"email": "erase@example.com"})

    response = await orchestrator.process_message(
        ChatRequest(session_id=session_id, message="delete my data")
    )

    assert "deleted" in response.message.lower()
    assert await database.get_lead_by_session(session_id) is None


# --- Phase 3 ---

@pytest.mark.asyncio
async def test_client_chat_answers_from_the_client_index_only(orchestrator, database, monkeypatch):
    from app import client_support as cs_mod
    from app.models import ClientRecord

    client = await database.upsert_client(ClientRecord(
        email="sam@acme.com", name="Sam", company="Acme",
    ))

    called = {}

    async def fake_client_answer(query, client_id, client_name, client_company,
                                 conversation_history=None, top_k=5):
        called["client_id"] = client_id
        return {"answer": "Your rebuild is 60% through the build phase.",
                "sources": ["Rebuild"], "model_used": "fake", "cost": 0.0003, "grounded": True}

    def public_answer_must_not_run(*args, **kwargs):
        raise AssertionError("public RAG must never be called on the client surface")

    monkeypatch.setattr(cs_mod.rag_engine, "answer_for_client", fake_client_answer)
    monkeypatch.setattr(cs_mod.rag_engine, "answer", public_answer_must_not_run)

    session_id = await _new_session(database, SurfaceType.CLIENT, client.id)
    response = await orchestrator.process_client_message(
        session_id=session_id, message="How's my project going?",
        client_id=client.id, client_name="Sam", client_company="Acme",
    )

    assert "60%" in response.message
    assert called["client_id"] == client.id
    assert response.escalated is False


@pytest.mark.asyncio
async def test_client_request_for_a_human_escalates(orchestrator, database, monkeypatch):
    from app.models import ClientRecord

    client = await database.upsert_client(ClientRecord(
        email="jo@acme.com", name="Jo", company="Acme",
        account_manager_email="manager@abacusdigital.net",
    ))

    session_id = await _new_session(database, SurfaceType.CLIENT, client.id)
    response = await orchestrator.process_client_message(
        session_id=session_id, message="I want to speak to my account manager about the invoice",
        client_id=client.id, client_name="Jo", client_company="Acme",
    )

    assert response.escalated is True
    assert "manager@abacusdigital.net" in response.message

    escalations = await database.get_escalations()
    assert len(escalations) == 1
    assert escalations[0]["client_id"] == client.id

    session = await database.get_session(session_id)
    assert session.status == SessionStatus.ESCALATED


@pytest.mark.asyncio
async def test_ungrounded_client_question_escalates_rather_than_guessing(
    orchestrator, database, monkeypatch
):
    from app import client_support as cs_mod
    from app.models import ClientRecord

    client = await database.upsert_client(ClientRecord(email="pat@acme.com", name="Pat"))

    async def no_match(**kwargs):
        return {"answer": "", "sources": [], "model_used": "none", "cost": 0.0, "grounded": False}

    monkeypatch.setattr(cs_mod.rag_engine, "answer_for_client", no_match)

    session_id = await _new_session(database, SurfaceType.CLIENT, client.id)
    response = await orchestrator.process_client_message(
        session_id=session_id, message="What did we agree about the mobile app?",
        client_id=client.id, client_name="Pat", client_company="Acme",
    )

    assert response.escalated is True
    assert "don't have that detail" in response.message


def _as_coro(value):
    """Wrap a plain value in an awaitable, for monkeypatched async functions."""
    async def _inner():
        return value
    return _inner()


# --- Soft email capture ---

@pytest.mark.asyncio
async def test_first_message_gets_a_soft_email_ask(orchestrator, database, monkeypatch):
    from app import chat_orchestrator as mod
    from app.email_capture import EMAIL_ASK

    monkeypatch.setattr(
        mod.intent_classifier, "classify",
        lambda **kwargs: _as_coro({"intent": mod.Intent.GREETING, "confidence": 0.9}),
    )

    session_id = await _new_session(database)
    response = await orchestrator.process_message(
        ChatRequest(session_id=session_id, message="hi")
    )

    assert EMAIL_ASK in response.message
    assert response.suggestions == ["No thanks, let's continue"]

    state = await orchestrator.get_state(session_id)
    assert state.email_capture_asked is True
    assert state.email_capture_pending is True


@pytest.mark.asyncio
async def test_email_reply_is_stored_and_acknowledged(orchestrator, database, monkeypatch):
    from app import chat_orchestrator as mod
    from app.email_capture import FOUND_ACK

    monkeypatch.setattr(
        mod.intent_classifier, "classify",
        lambda **kwargs: _as_coro({"intent": mod.Intent.GREETING, "confidence": 0.9}),
    )

    session_id = await _new_session(database)
    await orchestrator.process_message(ChatRequest(session_id=session_id, message="hi"))

    response = await orchestrator.process_message(
        ChatRequest(session_id=session_id, message="dana@example.com")
    )

    assert FOUND_ACK in response.message
    lead = await database.get_lead_by_session(session_id)
    assert lead.email == "dana@example.com"

    state = await orchestrator.get_state(session_id)
    assert state.email_capture_pending is False


@pytest.mark.asyncio
async def test_decline_is_acknowledged_and_no_email_stored(orchestrator, database, monkeypatch):
    from app import chat_orchestrator as mod
    from app.email_capture import DECLINE_ACK

    monkeypatch.setattr(
        mod.intent_classifier, "classify",
        lambda **kwargs: _as_coro({"intent": mod.Intent.GREETING, "confidence": 0.9}),
    )

    session_id = await _new_session(database)
    await orchestrator.process_message(ChatRequest(session_id=session_id, message="hi"))

    response = await orchestrator.process_message(
        ChatRequest(session_id=session_id, message="no thanks")
    )

    assert DECLINE_ACK in response.message
    lead = await database.get_lead_by_session(session_id)
    assert lead is None or lead.email is None


@pytest.mark.asyncio
async def test_email_plus_real_content_still_gets_routed(orchestrator, database, monkeypatch):
    from app import chat_orchestrator as mod
    from app.email_capture import FOUND_ACK

    monkeypatch.setattr(
        mod.intent_classifier, "classify",
        lambda **kwargs: _as_coro({"intent": mod.Intent.GREETING, "confidence": 0.9}),
    )
    session_id = await _new_session(database)
    await orchestrator.process_message(ChatRequest(session_id=session_id, message="hi"))

    async def fake_answer(query, conversation_history=None, top_k=5):
        return {
            "answer": "Yes, we build ecommerce sites.",
            "sources": ["Web Design & Development"],
            "source_link": None, "model_used": "fake", "cost": 0.0001, "grounded": True,
        }

    monkeypatch.setattr(mod.rag_engine, "answer", fake_answer)
    monkeypatch.setattr(
        mod.intent_classifier, "classify",
        lambda **kwargs: _as_coro({"intent": mod.Intent.QUESTION, "confidence": 0.9}),
    )

    response = await orchestrator.process_message(ChatRequest(
        session_id=session_id,
        message="dana@example.com, do you build ecommerce sites?",
    ))

    assert FOUND_ACK in response.message
    assert "ecommerce sites" in response.message
    lead = await database.get_lead_by_session(session_id)
    assert lead.email == "dana@example.com"


@pytest.mark.asyncio
async def test_returning_visitor_is_not_asked_for_email_again(orchestrator, database, monkeypatch):
    from app import chat_orchestrator as mod
    from app.email_capture import EMAIL_ASK
    from app.models import SessionRecord

    visitor_id = "returning-visitor"
    old = SessionRecord(visitor_id=visitor_id)
    await database.create_session(old)
    await database.update_lead_fields(old.id, {
        "email": "known@example.com", "business_type": "manufacturing",
    })

    new_session = SessionRecord(visitor_id=visitor_id)
    await database.create_session(new_session)

    monkeypatch.setattr(
        mod.intent_classifier, "classify",
        lambda **kwargs: _as_coro({"intent": mod.Intent.GREETING, "confidence": 0.9}),
    )

    response = await orchestrator.process_message(
        ChatRequest(session_id=new_session.id, visitor_id=visitor_id, message="hi")
    )

    assert EMAIL_ASK not in response.message

    state = await orchestrator.get_state(new_session.id)
    assert state.qualification_data.get("email") == "known@example.com"
    assert state.qualification_data.get("business_type") == "manufacturing"


# --- Category classification ---

@pytest.mark.asyncio
async def test_category_becomes_query_for_a_grounded_question(orchestrator, database, monkeypatch):
    from app import chat_orchestrator as mod

    async def fake_answer(query, conversation_history=None, top_k=5):
        return {
            "answer": "We build responsive sites.",
            "sources": ["Web Design & Development"], "source_link": None,
            "model_used": "fake", "cost": 0.0001, "grounded": True,
        }

    monkeypatch.setattr(mod.rag_engine, "answer", fake_answer)
    monkeypatch.setattr(
        mod.intent_classifier, "classify",
        lambda **kwargs: _as_coro({"intent": mod.Intent.QUESTION, "confidence": 0.9}),
    )

    session_id = await _new_session(database)
    await orchestrator.process_message(
        ChatRequest(session_id=session_id, message="What services do you offer?")
    )

    session = await database.get_session(session_id)
    assert session.category == "query"


@pytest.mark.asyncio
async def test_category_becomes_lead_once_qualification_scores(orchestrator, database, monkeypatch):
    from app import chat_orchestrator as mod

    monkeypatch.setattr(
        mod.intent_classifier, "classify",
        lambda **kwargs: _as_coro({"intent": mod.Intent.QUALIFICATION, "confidence": 0.9}),
    )
    monkeypatch.setattr("app.lead_qualifier.llm_router", FakeRouter(json_responses=[{
        "response": "Got it, thanks.",
        "extracted_data": {"budget_band": "15k_to_50k", "timeline": "immediate"},
        "enough_data_for_booking": False,
    }]))

    session_id = await _new_session(database)
    await orchestrator.process_message(ChatRequest(
        session_id=session_id, message="We have a healthy budget and need this done fast",
    ))

    session = await database.get_session(session_id)
    assert session.category == "lead"


# --- AI-generated chat titles ---

@pytest.mark.asyncio
async def test_title_generated_after_first_exchange(orchestrator, database, monkeypatch):
    from app import chat_orchestrator as mod

    monkeypatch.setattr(
        mod.intent_classifier, "classify",
        lambda **kwargs: _as_coro({"intent": mod.Intent.GREETING, "confidence": 0.9}),
    )
    monkeypatch.setattr(
        "app.chat_orchestrator.llm_router",
        FakeRouter(text_responses=["Website rebuild inquiry"]),
    )

    session_id = await _new_session(database)
    response = await orchestrator.process_message(
        ChatRequest(session_id=session_id, message="hi")
    )

    assert response.title == "Website rebuild inquiry"
    session = await database.get_session(session_id)
    assert session.title == "Website rebuild inquiry"


@pytest.mark.asyncio
async def test_title_never_becomes_the_generic_error_message(orchestrator, database, monkeypatch):
    from app import chat_orchestrator as mod

    monkeypatch.setattr(
        mod.intent_classifier, "classify",
        lambda **kwargs: _as_coro({"intent": mod.Intent.GREETING, "confidence": 0.9}),
    )

    class FailingRouter:
        async def generate(self, **kwargs):
            return {
                "content": "I'm having trouble processing your request right now.",
                "model_used": "error", "cost": 0.0, "usage": {}, "ok": False,
            }

    monkeypatch.setattr("app.chat_orchestrator.llm_router", FailingRouter())

    session_id = await _new_session(database)
    response = await orchestrator.process_message(
        ChatRequest(session_id=session_id, message="hi")
    )

    assert response.title is None
    session = await database.get_session(session_id)
    assert session.title is None
