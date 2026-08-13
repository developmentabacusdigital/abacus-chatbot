"""Database layer: progressive capture, dedup, deletion, metrics, client isolation."""

import pytest

from app.models import (
    SessionRecord, LeadRecord, MessageRecord, ProjectBrief,
    ClientRecord, ClientProject, SessionStatus, SurfaceType,
)


@pytest.mark.asyncio
async def test_session_roundtrip(database):
    session = SessionRecord(source_page="https://abacusdigital.net/all-services")
    await database.create_session(session)

    fetched = await database.get_session(session.id)
    assert fetched.source_page == "https://abacusdigital.net/all-services"
    assert fetched.status == SessionStatus.ACTIVE
    assert fetched.surface == SurfaceType.PUBLIC

    await database.update_session(session.id, status=SessionStatus.QUALIFIED.value,
                                  summary="Wants a rebuild")
    updated = await database.get_session(session.id)
    assert updated.status == SessionStatus.QUALIFIED
    assert updated.summary == "Wants a rebuild"


@pytest.mark.asyncio
async def test_update_session_rejects_unknown_field(database):
    session = SessionRecord()
    await database.create_session(session)

    # Would be a SQL injection vector if interpolated blindly
    await database.update_session(session.id, **{"status = 'x'; DROP TABLE sessions; --": "y"})

    assert await database.get_session(session.id) is not None


@pytest.mark.asyncio
async def test_progressive_lead_capture(database):
    session = SessionRecord()
    await database.create_session(session)

    await database.update_lead_fields(session.id, {"business_type": "manufacturing"})
    await database.update_lead_fields(session.id, {"pain_point": "no inbound leads"})
    await database.update_lead_fields(session.id, {"budget_band": "5k_to_15k",
                                                   "qualification_score": 0.62})

    lead = await database.get_lead_by_session(session.id)
    assert lead.business_type == "manufacturing"
    assert lead.pain_point == "no inbound leads"
    assert lead.qualification_score == pytest.approx(0.62)


@pytest.mark.asyncio
async def test_lead_field_whitelist_blocks_injection(database):
    session = SessionRecord()
    await database.create_session(session)
    await database.update_lead_fields(session.id, {"business_type": "retail"})

    # An LLM-extracted key that isn't a real column must be dropped, not executed
    await database.update_lead_fields(session.id, {
        "name = 'x' WHERE 1=1; DROP TABLE leads; --": "boom",
        "unknown_field": "ignored",
        "pain_point": "slow site",
    })

    lead = await database.get_lead_by_session(session.id)
    assert lead.pain_point == "slow site"
    assert lead.business_type == "retail"


@pytest.mark.asyncio
async def test_lead_dedup_by_email(database):
    for source in ("a", "b"):
        s = SessionRecord(source_page=source)
        await database.create_session(s)
        await database.create_or_update_lead(LeadRecord(
            session_id=s.id, email="repeat@example.com", company=f"Co {source}",
        ))

    leads = await database.get_all_leads()
    assert len([l for l in leads if l.email == "repeat@example.com"]) == 1


@pytest.mark.asyncio
async def test_delete_by_session_removes_everything(database):
    session = SessionRecord()
    await database.create_session(session)
    await database.save_message(MessageRecord(session_id=session.id, role="user", content="hi"))
    await database.update_lead_fields(session.id, {"email": "gone@example.com"})
    await database.save_brief(ProjectBrief(session_id=session.id, title="T"))

    deleted = await database.delete_by_session(session.id)

    assert deleted >= 3
    assert await database.get_session(session.id) is None
    assert await database.get_lead_by_session(session.id) is None
    assert await database.get_brief_by_session(session.id) is None
    assert await database.get_session_messages(session.id) == []


@pytest.mark.asyncio
async def test_delete_by_email_spans_sessions(database):
    for _ in range(2):
        s = SessionRecord()
        await database.create_session(s)
        await database.update_lead_fields(s.id, {"email": "wipe@example.com"})
        await database.save_message(MessageRecord(session_id=s.id, role="user", content="x"))

    assert await database.delete_by_email("wipe@example.com") > 0
    assert not [l for l in await database.get_all_leads() if l.email == "wipe@example.com"]


@pytest.mark.asyncio
async def test_csv_export_has_header_and_rows(database):
    s = SessionRecord()
    await database.create_session(s)
    await database.update_lead_fields(s.id, {"name": "Dana", "email": "dana@example.com"})

    csv_text = await database.export_leads_csv()
    lines = csv_text.strip().split("\n")

    assert lines[0].startswith("session_id,category,name,email")
    assert any("dana@example.com" in line for line in lines[1:])


@pytest.mark.asyncio
async def test_csv_export_escapes_newlines_in_summary(database):
    s = SessionRecord()
    await database.create_session(s)
    await database.update_lead_fields(s.id, {
        "email": "multi@example.com",
        "transcript_summary": "line one\nline two",
    })

    csv_text = await database.export_leads_csv()
    assert "line one line two" in csv_text


@pytest.mark.asyncio
async def test_metrics(database):
    s = SessionRecord()
    await database.create_session(s)
    await database.update_lead_fields(s.id, {
        "email": "m@example.com", "business_type": "saas",
        "pain_point": "churn", "qualification_score": 0.8,
    })

    metrics = await database.get_metrics()
    assert metrics["total_leads"] == 1
    assert metrics["qualified_leads"] == 1
    assert metrics["record_completeness"] == 1.0


@pytest.mark.asyncio
async def test_client_projects_and_magic_link(database):
    client = await database.upsert_client(ClientRecord(
        email="Client@Example.com", name="Sam", company="Acme",
    ))
    await database.upsert_project(ClientProject(
        client_id=client.id, name="Rebuild", deliverables=["Design", "Build"],
    ))

    # Lookup is case-insensitive
    assert (await database.get_client_by_email("client@example.com")).id == client.id

    projects = await database.get_client_projects(client.id)
    assert projects[0].deliverables == ["Design", "Build"]

    token = await database.create_magic_link(client.id, ttl_minutes=10)
    assert await database.redeem_magic_link(token) == client.id
    # Single use
    assert await database.redeem_magic_link(token) is None


@pytest.mark.asyncio
async def test_expired_magic_link_rejected(database):
    client = await database.upsert_client(ClientRecord(email="exp@example.com"))
    token = await database.create_magic_link(client.id, ttl_minutes=-1)
    assert await database.redeem_magic_link(token) is None


@pytest.mark.asyncio
async def test_client_session_token(database):
    client = await database.upsert_client(ClientRecord(email="sess@example.com"))
    session = await database.create_client_session(client.id, ttl_hours=1)

    resolved = await database.get_client_by_session_token(session["token"])
    assert resolved.id == client.id

    await database.revoke_client_session(session["token"])
    assert await database.get_client_by_session_token(session["token"]) is None


@pytest.mark.asyncio
async def test_stale_session_detection(database):
    fresh = SessionRecord()
    await database.create_session(fresh)
    stale = SessionRecord(started_at="2020-01-01T00:00:00")
    await database.create_session(stale)

    ids = await database.get_stale_sessions(minutes=30)
    assert stale.id in ids
    assert fresh.id not in ids


# --- Chat list / visitor memory (session-per-chat feature) ---

@pytest.mark.asyncio
async def test_sessions_by_visitor_lists_newest_first_with_snippet(database):
    visitor = "visitor-abc"
    s1 = SessionRecord(visitor_id=visitor, title="First chat")
    await database.create_session(s1)
    await database.save_message(MessageRecord(session_id=s1.id, role="user", content="hi there"))
    await database.increment_session_messages(s1.id)

    s2 = SessionRecord(visitor_id=visitor, title="Second chat")
    await database.create_session(s2)
    await database.save_message(MessageRecord(session_id=s2.id, role="user", content="follow up"))
    await database.increment_session_messages(s2.id)

    chats = await database.get_sessions_by_visitor(visitor)
    assert [c["session_id"] for c in chats] == [s2.id, s1.id]
    assert chats[0]["title"] == "Second chat"
    assert "follow up" in chats[0]["snippet"]


@pytest.mark.asyncio
async def test_sessions_by_visitor_excludes_empty_and_other_visitors(database):
    visitor = "visitor-xyz"
    empty = SessionRecord(visitor_id=visitor)
    await database.create_session(empty)  # never gets a message

    other = SessionRecord(visitor_id="someone-else")
    await database.create_session(other)
    await database.save_message(MessageRecord(session_id=other.id, role="user", content="hi"))
    await database.increment_session_messages(other.id)

    chats = await database.get_sessions_by_visitor(visitor)
    assert chats == []


@pytest.mark.asyncio
async def test_session_for_visitor_enforces_ownership(database):
    owner = SessionRecord(visitor_id="owner-1")
    await database.create_session(owner)

    assert (await database.get_session_for_visitor(owner.id, "owner-1")) is not None
    # A different visitor_id must not be able to read someone else's transcript
    assert (await database.get_session_for_visitor(owner.id, "intruder")) is None


@pytest.mark.asyncio
async def test_visitor_memory_carries_email_and_summary_forward(database):
    visitor = "visitor-memory"
    old = SessionRecord(visitor_id=visitor)
    await database.create_session(old)
    # summary/next_step are written at finalize time via update_session, not at creation
    await database.update_session(old.id, summary="Asked about web design pricing.")
    await database.update_lead_fields(old.id, {
        "email": "returning@example.com", "business_type": "manufacturing",
    })

    new = SessionRecord(visitor_id=visitor)
    await database.create_session(new)

    memory = await database.get_visitor_memory(visitor, exclude_session_id=new.id)
    assert memory["email"] == "returning@example.com"
    assert memory["known_fields"]["business_type"] == "manufacturing"
    assert "web design pricing" in memory["summaries"][0]


@pytest.mark.asyncio
async def test_delete_by_visitor_removes_all_their_sessions(database):
    visitor = "visitor-delete-me"
    for _ in range(2):
        s = SessionRecord(visitor_id=visitor)
        await database.create_session(s)
        await database.save_message(MessageRecord(session_id=s.id, role="user", content="x"))

    other = SessionRecord(visitor_id="keep-me")
    await database.create_session(other)

    deleted = await database.delete_by_visitor(visitor)
    assert deleted > 0
    assert await database.get_sessions_by_visitor(visitor) == []
    assert await database.get_session(other.id) is not None


@pytest.mark.asyncio
async def test_metrics_breaks_down_sessions_by_category(database):
    lead = SessionRecord(category="lead")
    query = SessionRecord(category="query")
    await database.create_session(lead)
    await database.create_session(query)

    metrics = await database.get_metrics()
    assert metrics["sessions_by_category"]["lead"] == 1
    assert metrics["sessions_by_category"]["query"] == 1
