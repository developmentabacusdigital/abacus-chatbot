"""Pure-logic tests: routing, scoring, guardrails, JSON extraction, chunking, isolation."""

import pytest

from app.llm_router import extract_json
from app.intent_classifier import intent_classifier
from app.lead_qualifier import lead_qualifier, QUALIFICATION_THRESHOLD
from app.intake_agent import intake_agent, looks_like_project_request
from app.guardrails import content_guardrails, RateLimiter
from app.site_crawler import SiteCrawler
from app.knowledge_base import (
    get_knowledge_chunks, get_service_catalog_text, build_client_chunks,
    linkify_services, get_service_url, SERVICE_URLS,
)
from app.email_service import EmailService, is_valid_email
from app.models import Intent, ClientRecord, ClientProject, ProjectBrief


# --- JSON extraction (cheap models wrap output in prose/fences) ---

@pytest.mark.parametrize("raw,expected", [
    ('{"intent": "question"}', "question"),
    ('```json\n{"intent": "question"}\n```', "question"),
    ('Sure! Here you go:\n{"intent": "question"}\nHope that helps.', "question"),
])
def test_extract_json_variants(raw, expected):
    assert extract_json(raw)["intent"] == expected


def test_extract_json_returns_none_on_garbage():
    assert extract_json("no json here at all") is None
    assert extract_json("") is None


# --- Intent fast-path ---

@pytest.mark.asyncio
@pytest.mark.parametrize("message,expected", [
    ("hi", Intent.GREETING),
    ("Hello there", Intent.GREETING),
    ("delete my data please", Intent.DATA_DELETION),
    ("can I speak to a human", Intent.HUMAN_HANDOFF),
    ("I'd like to book a call", Intent.BOOKING),
    ("bye", Intent.FAREWELL),
])
async def test_fast_path_intents(message, expected):
    result = await intent_classifier.classify(message)
    assert result["intent"] == expected


@pytest.mark.asyncio
async def test_greeting_prefix_does_not_swallow_a_real_enquiry():
    """"hi, our site is broken" is a business message, not a greeting."""
    result = intent_classifier._fast_classify(
        "hi, our website is losing customers and we need help fixing the funnel"
    )
    assert result is None


@pytest.mark.asyncio
async def test_bare_call_word_is_not_a_booking_request():
    """"call" appears constantly in ordinary sentences; only phrases should match."""
    assert intent_classifier._fast_classify("we get a lot of cold call complaints") is None


# --- Qualification scoring ---

def test_score_rises_with_budget_timeline_authority():
    low = lead_qualifier.score_from_data({
        "budget_band": "under_1k", "timeline": "just_exploring",
    })
    high = lead_qualifier.score_from_data({
        "budget_band": "15k_to_50k", "timeline": "immediate", "decision_maker": True,
        "business_type": "manufacturing", "pain_point": "no inbound leads",
        "service_interest": "Lead Generation & Performance Marketing",
    })
    assert 0 < low < QUALIFICATION_THRESHOLD < high <= 1.0


def test_empty_data_scores_zero():
    assert lead_qualifier.score_from_data({}) == 0.0


def test_next_question_walks_missing_fields():
    assert "business" in lead_qualifier._next_question({}).lower()
    q = lead_qualifier._next_question({"business_type": "saas", "pain_point": "churn"})
    assert "budget" in q.lower()


# --- Intake trigger ---

@pytest.mark.parametrize("message,expected", [
    ("I need a new ecommerce website for my manufacturing business", True),
    ("we're looking for help with automating our quoting process", True),
    ("ok", False),
    ("thanks", False),
])
def test_project_request_detection(message, expected):
    assert looks_like_project_request(message) is expected


def test_intake_missing_fields_shrink_as_data_arrives():
    empty = intake_agent._format_missing({})
    partial = intake_agent._format_missing({"goals": "more leads", "budget_band": "5k_to_15k"})
    assert "goals" in empty
    assert "goals" not in partial
    assert "timeline" in partial


def test_intake_completion_requires_core_fields():
    assert not intake_agent._is_complete({"goals": "x"})
    assert intake_agent._is_complete({
        "goals": "x", "current_state": "y", "budget_band": "5k_to_15k", "timeline": "immediate",
    })


# --- Guardrails ---

def test_rejects_sensitive_data():
    ok, warning = content_guardrails.validate_input("my card is 4111 1111 1111 1111")
    assert not ok and "credit card" in warning


def test_rejects_empty_and_overlong_input():
    assert not content_guardrails.validate_input("   ")[0]
    assert not content_guardrails.validate_input("x" * 5001)[0]


def test_accepts_ordinary_message():
    assert content_guardrails.validate_input("What does a website rebuild involve?")[0]


def test_pricing_claim_gets_a_disclaimer():
    out = content_guardrails.sanitize_output("It costs $2,000 per month.")
    assert "pricing depends on your specific requirements" in out.lower()


def test_dollar_figures_get_a_footnote_asterisk():
    out = content_guardrails.sanitize_output("Typical projects run $5,000 to $15,000.")
    assert "$5,000*" in out
    assert "$15,000*" in out


def test_pricing_topic_without_a_dollar_figure_still_gets_disclaimer():
    # No number at all, but the model still shouldn't get away with dodging the
    # disclaimer just because it phrased pricing without a literal "$" figure.
    out = content_guardrails.sanitize_output("Our pricing depends on the scope of work.")
    assert "team will give you an exact number" in out.lower()


def test_disclaimer_is_not_duplicated_if_already_present():
    out = content_guardrails.sanitize_output("It costs $2,000.")
    out_again = content_guardrails.sanitize_output(out)
    assert out_again.count("Pricing depends on your specific requirements") == 1


def test_commitment_language_gets_a_disclaimer():
    out = content_guardrails.check_commitment_language("We guarantee first-page rankings.")
    assert "guarantees are discussed" in out.lower()


def test_rate_limiter_caps_messages_per_session(monkeypatch):
    from app import guardrails
    monkeypatch.setattr(guardrails.settings, "max_messages_per_session", 3)
    limiter = RateLimiter()

    for _ in range(3):
        assert limiter.check_rate_limit("1.2.3.4", "session-a")[0]

    allowed, message = limiter.check_rate_limit("1.2.3.4", "session-a")
    assert not allowed and "message limit" in message


def test_rate_limiter_caps_sessions_per_ip(monkeypatch):
    from app import guardrails
    monkeypatch.setattr(guardrails.settings, "max_sessions_per_ip_per_hour", 2)
    limiter = RateLimiter()

    limiter.check_rate_limit("9.9.9.9", "s1")
    limiter.check_rate_limit("9.9.9.9", "s2")

    allowed, _ = limiter.check_rate_limit("9.9.9.9", "s3")
    assert not allowed


# --- Knowledge base / crawler ---

def test_knowledge_chunks_cover_every_service():
    chunks = get_knowledge_chunks()
    services = {c["metadata"]["service"] for c in chunks}

    assert len(chunks) > 20
    assert "Web Design & Development" in services
    assert "Engineering Services" in services
    assert all(c["id"] and c["text"] for c in chunks)


def test_chunk_ids_are_unique():
    ids = [c["id"] for c in get_knowledge_chunks()]
    assert len(ids) == len(set(ids))


def test_service_catalog_lists_all_nine_capability_areas():
    catalog = get_service_catalog_text()
    for name in ("Web Design", "AI & Automation", "Cybersecurity", "Engineering",
                 "Brand Identity", "Software Solutions", "Digital Business Transformation"):
        assert name in catalog


def test_linkify_services_turns_mentions_into_markdown_links():
    text = "We recommend Brand Identity alongside Web Design & Development for this."
    linked = linkify_services(text)
    assert f"[Brand Identity]({SERVICE_URLS['Brand Identity']})" in linked
    assert f"[Web Design & Development]({SERVICE_URLS['Web Design & Development']})" in linked


def test_linkify_services_does_not_double_link_existing_markdown():
    already_linked = "See [Cybersecurity](https://example.com/custom-link) for more."
    result = linkify_services(already_linked)
    # The existing link's URL must survive untouched, not get swapped for the real one
    assert "https://example.com/custom-link" in result
    assert result.count("[Cybersecurity]") == 1


def test_linkify_services_ignores_capability_areas_with_no_dedicated_page():
    text = "Software Solutions and Digital Business Transformation are also options."
    linked = linkify_services(text)
    assert linked == text  # no page exists for either, so nothing should change


def test_get_service_url_returns_none_for_unknown_service():
    assert get_service_url("Not A Real Service") is None
    assert get_service_url("Cybersecurity") == SERVICE_URLS["Cybersecurity"]


def test_crawler_only_indexes_public_paths():
    crawler = SiteCrawler("https://www.abacusdigital.net")

    assert crawler._is_indexable("https://www.abacusdigital.net/all-services/web-design")
    assert crawler._is_indexable("https://www.abacusdigital.net/blog/seo-basics")
    # Client-only and asset paths stay out of the public prospect index
    assert not crawler._is_indexable("https://www.abacusdigital.net/client/portal")
    assert not crawler._is_indexable("https://www.abacusdigital.net/all-services/deck.pdf")
    assert not crawler._is_indexable("https://someoneelse.com/blog/post")


def test_crawler_chunks_carry_source_url_for_citation():
    chunks = SiteCrawler().chunk_page({
        "url": "https://www.abacusdigital.net/blog/post",
        "title": "A Post",
        "text": "\n".join(f"Paragraph {i} with enough words to matter." for i in range(40)),
    })

    assert len(chunks) > 1
    assert all(c["metadata"]["source_url"].endswith("/blog/post") for c in chunks)
    assert all("https://www.abacusdigital.net/blog/post" in c["text"] for c in chunks)
    assert chunks[0]["metadata"]["section"] == "blog"


# --- Phase 3 isolation ---

def test_client_chunks_are_tagged_with_client_id():
    client = ClientRecord(id="c1", email="a@b.com", name="Sam", company="Acme")
    projects = [ClientProject(client_id="c1", name="Rebuild", deliverables=["Design"])]

    chunks = build_client_chunks(client, projects)

    assert chunks
    assert all(c["metadata"]["client_id"] == "c1" for c in chunks)
    assert any("Rebuild" in c["text"] for c in chunks)


@pytest.mark.asyncio
async def test_client_search_requires_a_client_id():
    from app.vector_store import vector_store
    with pytest.raises(ValueError):
        await vector_store.search_client(query="status", client_id="")


# --- Email ---

@pytest.mark.parametrize("address,valid", [
    ("dana@example.com", True),
    ("dana@example.co.uk", True),
    ("not-an-email", False),
    ("", False),
])
def test_email_validation(address, valid):
    assert is_valid_email(address) is valid


def test_email_html_escapes_visitor_content():
    html = EmailService._to_html('Hi <script>alert("x")</script>')
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_email_from_header_parsing():
    assert EmailService._parse_from("Abacus Digital <hi@abacusdigital.net>") == (
        "Abacus Digital", "hi@abacusdigital.net"
    )
    assert EmailService._parse_from("hi@abacusdigital.net")[1] == "hi@abacusdigital.net"


# --- Brief rendering ---

def test_brief_markdown_includes_every_section():
    brief = ProjectBrief(
        session_id="s1", title="Rebuild", summary="A summary.",
        goals=["More leads"], recommended_services=["Web Design & Development"],
    )
    markdown = brief.to_markdown()

    for heading in ("## Goals", "## Constraints", "## Recommended Services", "## Risk Flags"):
        assert heading in markdown
    assert "More leads" in markdown
    # Empty lists render explicitly rather than silently disappearing
    assert "(none captured)" in markdown


# --- Budget / timeline normalisation ---
# Models are asked for enum bands but return what the visitor said ("around £20k").
# Storing that verbatim understated the score and broke dashboard filtering.

@pytest.mark.parametrize("raw,expected", [
    ("15k_to_50k", "15k_to_50k"),           # already banded
    ("around £20k", "15k_to_50k"),
    ("$20,000", "15k_to_50k"),
    ("about 8k", "5k_to_15k"),
    ("10 to 15k", "5k_to_15k"),             # bands on the upper figure
    ("2500", "1k_to_5k"),
    ("we could stretch to 60k", "over_50k"),
    ("$800", "under_1k"),
    ("no idea", None),
    (None, None),
])
def test_normalize_budget(raw, expected):
    from app.normalize import normalize_budget
    assert normalize_budget(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("short_term", "short_term"),
    ("in about 3 months", "short_term"),
    ("2 weeks", "immediate"),
    ("asap", "immediate"),
    ("4-6 months", "medium_term"),
    ("next year", "long_term"),
    ("about 2 years", "long_term"),
    ("just exploring for now", "just_exploring"),
    ("no idea", None),
    (None, None),
])
def test_normalize_timeline(raw, expected):
    from app.normalize import normalize_timeline
    assert normalize_timeline(raw) == expected


def test_normalize_fields_drops_unusable_values():
    from app.normalize import normalize_fields
    out = normalize_fields({
        "budget_band": "around £20k",
        "timeline": "who knows",
        "pain_point": "no enquiries",
    })
    assert out["budget_band"] == "15k_to_50k"
    # An unparseable value is dropped rather than overwriting a good stored one
    assert "timeline" not in out
    assert out["pain_point"] == "no enquiries"


def test_banded_budget_scores_higher_than_free_text():
    """The defect this fixes: '£20k' only earned partial credit, understating the lead."""
    banded = lead_qualifier.score_from_data({"budget_band": "15k_to_50k"})
    free_text = lead_qualifier.score_from_data({"budget_band": "£20k"})
    assert banded > free_text


# --- Soft email capture ---

@pytest.mark.parametrize("text,expected", [
    ("dana@example.com", "dana@example.com"),
    ("my email is dana@example.co.uk, thanks", "dana@example.co.uk"),
    ("no email here", None),
    ("", None),
])
def test_extract_email(text, expected):
    from app.email_capture import extract_email
    assert extract_email(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("no", True),
    ("No thanks", True),
    ("nah, rather not", True),
    ("skip", True),
    ("no, but do you do SEO?", False),   # a real question riding along "no" is not a bare decline
    ("dana@example.com", False),
    ("we need a website rebuild", False),
])
def test_looks_like_decline(text, expected):
    from app.email_capture import looks_like_decline
    assert looks_like_decline(text) is expected


def test_strip_email_leaves_the_rest_of_the_message():
    from app.email_capture import strip_email
    assert strip_email("dana@example.com, also what's your pricing?", "dana@example.com") == \
        ", also what's your pricing?"


# --- Suggestion chips ---

def test_suggestions_only_offered_for_finite_answer_fields():
    from app.suggestions import suggestions_for_field, BUDGET_SUGGESTIONS
    assert suggestions_for_field("budget_band") == BUDGET_SUGGESTIONS
    assert suggestions_for_field("timeline")
    assert suggestions_for_field("decision_maker")
    # Open-ended fields get no chips — forcing them narrows a "tell me more" answer
    assert suggestions_for_field("pain_point") == []
    assert suggestions_for_field("goals") == []
    assert suggestions_for_field(None) == []


def test_qualifier_next_missing_field_matches_suggestion_catalog():
    from app.suggestions import suggestions_for_field
    assert lead_qualifier.next_missing_field({}) == "business_type"
    field = lead_qualifier.next_missing_field({"business_type": "saas", "pain_point": "churn"})
    assert field == "budget_band"
    assert suggestions_for_field(field)  # budget has chips
    assert lead_qualifier.next_missing_field({
        "business_type": "x", "pain_point": "y", "budget_band": "5k_to_15k",
        "timeline": "short_term", "decision_maker": True,
    }) is None


def test_intake_next_missing_field_walks_discovery_order():
    assert intake_agent.next_missing_field({}) == "goals"
    assert intake_agent.next_missing_field({"goals": "more leads"}) == "current_state"
    assert intake_agent.next_missing_field({
        "goals": "x", "current_state": "y", "budget_band": "5k_to_15k", "timeline": "short_term",
        "constraints": "none", "success_criteria": "more leads", "email": "d@example.com",
    }) is None
