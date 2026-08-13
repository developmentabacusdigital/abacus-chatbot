"""
Abacus Digital Chatbot - FastAPI Application
Main entry point: public widget API, authenticated client API, and internal admin API.
"""

import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, PlainTextResponse, RedirectResponse

from .config import settings
from .database import db
from .vector_store import vector_store
from .llm_router import llm_router
from .indexer import indexer
from .chat_orchestrator import chat_orchestrator
from .email_service import email_service
from .crm_sync import crm_sync
from .guardrails import rate_limiter
from .auth import require_admin, require_cron, require_client, client_auth
from .models import (
    ChatRequest, ChatResponse, SessionRecord, SessionStatus, SurfaceType,
    DataDeletionRequest, DataDeletionResponse,
    MagicLinkRequest, MagicLinkResponse, ClientVerifyRequest, ClientVerifyResponse,
    ClientChatRequest, ClientRecord, ClientProject, ReindexRequest,
    EmailApprovalRequest, ChatListResponse, ChatTranscriptResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WIDGET_DIR = os.path.join(BASE_DIR, "widget")
DASHBOARD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dashboard")

# How long a session can sit idle before it's finalized as abandoned
ABANDON_AFTER_MINUTES = 30
SWEEP_INTERVAL_SECONDS = 600


async def _sweep_loop():
    """Finalize abandoned sessions so every conversation ends with a lead record."""
    while True:
        await asyncio.sleep(SWEEP_INTERVAL_SECONDS)
        try:
            await chat_orchestrator.sweep_abandoned(ABANDON_AFTER_MINUTES)
            await db.purge_expired_tokens()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Session sweep failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info("🚀 Starting Abacus Digital Chatbot...")

    await db.connect()
    logger.info("✅ Database connected")

    await llm_router.initialize()
    if not llm_router.configured:
        logger.warning("⚠️  OPENROUTER_API_KEY is not set — the bot cannot generate answers")

    await email_service.initialize()
    logger.info(
        f"✉️  Email provider: {settings.email_provider} "
        f"({'configured' if email_service.configured else 'not configured'})"
    )

    logger.info("📚 Indexing knowledge base...")
    await indexer.bootstrap()
    logger.info(f"✅ Vector store: {await vector_store.get_stats()}")

    # In-process background loops are for local dev / any always-on host; they're
    # no-ops on Vercel (checked via the VERCEL env var Vercel sets automatically),
    # which relies on the /api/cron/* endpoints below instead.
    indexer.start_scheduler()
    crm_sync.start_scheduler()
    sweep_task = None if os.environ.get("VERCEL") else asyncio.create_task(_sweep_loop())

    logger.info("🤖 Abacus Digital Chatbot is ready!")

    yield

    logger.info("Shutting down...")
    if sweep_task:
        sweep_task.cancel()
    await indexer.stop_scheduler()
    await crm_sync.stop_scheduler()
    await crm_sync.close()
    await email_service.close()
    await llm_router.close()
    await db.disconnect()
    logger.info("Goodbye!")


app = FastAPI(
    title="Abacus Digital Chatbot API",
    description="AI chatbot backend for the Abacus Digital website",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


def get_client_ip(request: Request) -> str:
    """Extract client IP from request."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# ========== Health ==========

@app.get("/", response_class=JSONResponse)
async def root():
    return {"status": "ok", "service": "Abacus Digital Chatbot", "version": "1.0.0"}


@app.get("/health")
async def health():
    """Detailed health check. The widget uses this to decide whether to degrade."""
    return {
        "status": "healthy",
        "database": "connected" if db._pool else "disconnected",
        "vector_store": await vector_store.get_stats(),
        "llm_router": {
            "configured": llm_router.configured,
            "calls": llm_router.call_count,
            "session_cost_usd": round(llm_router.total_cost, 6),
        },
        "email": {"provider": settings.email_provider, "configured": email_service.configured},
        "crm_mirror": {"provider": settings.crm_provider, "enabled": crm_sync.enabled},
        "last_index": indexer.last_result or None,
    }


# ========== Public chat (widget) ==========

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, req: Request):
    """Process a chat message. This is the main endpoint the widget calls."""
    ip = get_client_ip(req)

    if not request.session_id:
        request.session_id = str(uuid.uuid4())

    allowed, error = rate_limiter.check_rate_limit(ip, request.session_id)
    if not allowed:
        return ChatResponse(session_id=request.session_id, message=error)

    existing = await db.get_session(request.session_id)
    if not existing:
        await db.create_session(SessionRecord(
            id=request.session_id,
            source_page=request.source_page,
            consent_given=request.consent_given,
            visitor_id=request.visitor_id,
            ip_address=ip,
            user_agent=req.headers.get("user-agent", "")[:400],
        ))
    elif existing.surface == SurfaceType.CLIENT:
        # A client session id must not be reusable on the anonymous endpoint
        raise HTTPException(status_code=403, detail="This session belongs to the client portal")

    try:
        return await chat_orchestrator.process_message(request=request, ip_address=ip)
    except Exception as e:
        logger.error(f"Error processing chat: {e}", exc_info=True)
        return ChatResponse(
            session_id=request.session_id,
            message=(
                "I'm sorry, I ran into an issue. Please try again, or reach out to our team "
                "at https://www.abacusdigital.net/contact."
            ),
        )


@app.post("/api/sessions/{session_id}/end")
async def end_session(session_id: str):
    """Widget close / page unload: summarise and close out the session."""
    result = await chat_orchestrator.finalize_session(session_id, SessionStatus.COMPLETED)
    return result


@app.get("/api/chats", response_model=ChatListResponse)
async def list_chats(visitor_id: str = Query(...), exclude: Optional[str] = None):
    """
    Chat-list panel for the widget: every past chat belonging to this browser-scoped
    visitor. No admin key — a visitor_id is an unguessable client-generated UUID, the
    same trust model already used for session_id on the anonymous endpoint.
    """
    chats = await db.get_sessions_by_visitor(visitor_id, limit=50, exclude_session_id=exclude)
    return ChatListResponse(chats=chats)


@app.get("/api/chats/{session_id}", response_model=ChatTranscriptResponse)
async def get_chat(session_id: str, visitor_id: str = Query(...)):
    """
    Fetch one past chat's transcript for the visitor who owns it.

    Ownership is enforced server-side (session.visitor_id must match), so guessing a
    session_id without also knowing the visitor_id that created it doesn't work.
    """
    session = await db.get_session_for_visitor(session_id, visitor_id)
    if not session:
        raise HTTPException(status_code=404, detail="Chat not found")

    transcript = await db.get_session_transcript(session_id)
    return ChatTranscriptResponse(session_id=session_id, title=session.title, transcript=transcript)


@app.delete("/api/data", response_model=DataDeletionResponse)
async def delete_data(request: DataDeletionRequest):
    """Delete user data (PRD 7.5)."""
    if not request.email and not request.session_id and not request.visitor_id:
        raise HTTPException(
            status_code=400, detail="One of email, session_id, or visitor_id is required"
        )

    total = 0
    if request.email:
        total += await db.delete_by_email(request.email)
    if request.session_id:
        total += await db.delete_by_session(request.session_id)
        chat_orchestrator.cleanup_session(request.session_id)
    if request.visitor_id:
        total += await db.delete_by_visitor(request.visitor_id)

    return DataDeletionResponse(
        success=total > 0,
        records_deleted=total,
        message=f"Deleted {total} records." if total else "No matching records found.",
    )


# ========== Client portal (Phase 3) ==========

@app.post("/api/client/login", response_model=MagicLinkResponse)
async def client_login(request: MagicLinkRequest):
    """Request a magic sign-in link."""
    result = await client_auth.request_magic_link(str(request.email))
    return MagicLinkResponse(**result)


@app.post("/api/client/verify", response_model=ClientVerifyResponse)
async def client_verify(request: ClientVerifyRequest):
    """Redeem a magic link for a session token."""
    result = await client_auth.verify_token(request.token)
    return ClientVerifyResponse(**result)


@app.post("/api/client/chat", response_model=ChatResponse)
async def client_chat(
    request: ClientChatRequest,
    req: Request,
    client: ClientRecord = Depends(require_client),
):
    """Authenticated client support chat. Isolated from the public index (PRD 7.7)."""
    session_id = request.session_id or str(uuid.uuid4())

    existing = await db.get_session(session_id)
    if existing:
        # A session token may only continue its own client's session
        if existing.surface != SurfaceType.CLIENT or existing.client_id != client.id:
            raise HTTPException(status_code=403, detail="Session does not belong to this client")
    else:
        await db.create_session(SessionRecord(
            id=session_id,
            source_page=request.source_page,
            consent_given=True,
            surface=SurfaceType.CLIENT,
            client_id=client.id,
            ip_address=get_client_ip(req),
            user_agent=req.headers.get("user-agent", "")[:400],
        ))

    try:
        return await chat_orchestrator.process_client_message(
            session_id=session_id,
            message=request.message,
            client_id=client.id,
            client_name=client.name,
            client_company=client.company,
            source_page=request.source_page,
        )
    except Exception as e:
        logger.error(f"Error processing client chat: {e}", exc_info=True)
        return ChatResponse(
            session_id=session_id,
            message=(
                "Something went wrong on our side. Your account manager is the fastest route "
                "right now — reach them at " + settings.account_manager_email + "."
            ),
        )


@app.get("/api/client/me")
async def client_me(client: ClientRecord = Depends(require_client)):
    """Current client profile and projects, for the portal header."""
    projects = await db.get_client_projects(client.id)
    return {
        "client": {
            "name": client.name,
            "company": client.company,
            "email": client.email,
            "account_manager_email": client.account_manager_email or settings.account_manager_email,
        },
        "projects": [
            {"name": p.name, "status": p.status, "service_line": p.service_line,
             "target_date": p.target_date}
            for p in projects
        ],
    }


@app.post("/api/client/logout")
async def client_logout(request: Request, client: ClientRecord = Depends(require_client)):
    """Revoke the current client session token."""
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        await db.revoke_client_session(auth_header[7:].strip())
    return {"success": True}


# ========== Internal admin API (PRD 7.4) ==========

@app.get("/api/admin/leads", dependencies=[Depends(require_admin)])
async def list_leads(limit: int = Query(50, le=500), offset: int = 0):
    """List leads for the internal dashboard."""
    leads = await db.get_all_leads(limit=limit, offset=offset)
    return {"leads": [l.model_dump() for l in leads], "count": len(leads),
            "limit": limit, "offset": offset}


@app.get("/api/admin/leads.csv", dependencies=[Depends(require_admin)],
         response_class=PlainTextResponse)
async def export_leads():
    """CSV export of every lead."""
    csv_text = await db.export_leads_csv()
    return PlainTextResponse(
        csv_text,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="abacus-leads.csv"'},
    )


@app.get("/api/admin/metrics", dependencies=[Depends(require_admin)])
async def metrics():
    """Success metrics (PRD Section 10)."""
    data = await db.get_metrics()
    data["last_index"] = indexer.last_result or {}
    data["vector_store"] = await vector_store.get_stats()
    return data


@app.get("/api/admin/sessions/{session_id}", dependencies=[Depends(require_admin)])
async def get_session(session_id: str):
    """Session details, transcript, lead, and brief."""
    session = await db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    lead = await db.get_lead_by_session(session_id)
    brief = await db.get_brief_by_session(session_id)

    return {
        "session": session.model_dump(),
        "transcript": await db.get_session_transcript(session_id),
        "lead": lead.model_dump() if lead else None,
        "brief": brief.model_dump() if brief else None,
    }


@app.get("/api/admin/briefs", dependencies=[Depends(require_admin)])
async def list_briefs(limit: int = Query(50, le=200)):
    """Project briefs produced by the intake agent."""
    briefs = await db.get_all_briefs(limit=limit)
    return {"briefs": [b.model_dump() for b in briefs], "count": len(briefs)}


@app.get("/api/admin/emails", dependencies=[Depends(require_admin)])
async def list_emails(status: Optional[str] = None, limit: int = Query(50, le=200)):
    """Outbound email queue, including drafts awaiting approval."""
    emails = await db.get_emails(status=status, limit=limit)
    return {"emails": [e.model_dump() for e in emails], "count": len(emails)}


@app.post("/api/admin/emails/approve", dependencies=[Depends(require_admin)])
async def approve_email(request: EmailApprovalRequest):
    """Approve (and send) or reject a queued follow-up email."""
    if request.approve:
        record = await email_service.approve_and_send(request.email_id, request.edited_body)
    else:
        record = await email_service.reject(request.email_id)

    if not record:
        raise HTTPException(status_code=404, detail="Email not found")
    return record.model_dump()


@app.get("/api/admin/escalations", dependencies=[Depends(require_admin)])
async def list_escalations(resolved: bool = False):
    """Open human-handoff requests."""
    return {"escalations": await db.get_escalations(resolved=resolved)}


@app.post("/api/admin/reindex", dependencies=[Depends(require_admin)])
async def reindex(request: ReindexRequest):
    """
    Re-index the public knowledge base.
    Wire this to a CMS publish webhook, or let the daily scheduler drive it (PRD 7.1).
    """
    return await indexer.index_public(crawl_site=request.crawl_site, full=request.full)


@app.post("/api/admin/crm-sync", dependencies=[Depends(require_admin)])
async def trigger_crm_sync():
    """Manually mirror leads into the configured CRM."""
    return await crm_sync.sync_all()


@app.post("/api/admin/clients", dependencies=[Depends(require_admin)])
async def upsert_client(client: ClientRecord):
    """Create or update a client for the Phase 3 portal."""
    saved = await db.upsert_client(client)
    await indexer.index_client(saved.id)
    return saved.model_dump()


@app.post("/api/admin/clients/{client_id}/projects", dependencies=[Depends(require_admin)])
async def upsert_project(client_id: str, project: ClientProject):
    """Create or update a client project and reindex that client's knowledge base."""
    if not await db.get_client(client_id):
        raise HTTPException(status_code=404, detail="Client not found")
    project.client_id = client_id
    saved = await db.upsert_project(project)
    index_result = await indexer.index_client(client_id)
    return {"project": saved.model_dump(), "index": index_result}


@app.get("/api/admin/clients", dependencies=[Depends(require_admin)])
async def list_clients():
    """All clients with their project counts."""
    projects = await db.get_all_projects()
    by_client = {}
    for p in projects:
        by_client.setdefault(p.client_id, []).append(p)

    clients = []
    for client_id in by_client:
        client = await db.get_client(client_id)
        if client:
            clients.append({
                **client.model_dump(),
                "project_count": len(by_client[client_id]),
            })
    return {"clients": clients}


# ========== Scheduled jobs (Vercel Cron) ==========
# GET endpoints (Vercel Cron always sends GET, no body) mirroring the in-process
# background loops in indexer.py/crm_sync.py, for hosts with no persistent process for
# those loops to live in. See vercel.json for the actual schedule; Hobby-tier Vercel
# only allows daily cron granularity, Pro allows more frequent.

@app.get("/api/cron/reindex", dependencies=[Depends(require_cron)])
async def cron_reindex():
    """Daily public knowledge base re-index (PRD 7.1)."""
    return await indexer.index_public(crawl_site=True, full=False)


@app.get("/api/cron/crm-sync", dependencies=[Depends(require_cron)])
async def cron_crm_sync():
    """Mirror leads into the configured CRM, if one is enabled."""
    return await crm_sync.sync_all()


@app.get("/api/cron/sweep", dependencies=[Depends(require_cron)])
async def cron_sweep():
    """Finalize abandoned sessions and purge expired auth tokens."""
    finalized = await chat_orchestrator.sweep_abandoned(ABANDON_AFTER_MINUTES)
    purged = await db.purge_expired_tokens()
    return {"sessions_finalized": finalized, "tokens_purged": purged}


# ========== Static files ==========

if os.path.isdir(WIDGET_DIR):
    app.mount("/widget", StaticFiles(directory=WIDGET_DIR, html=True), name="widget")

if os.path.isdir(DASHBOARD_DIR):
    app.mount("/dashboard", StaticFiles(directory=DASHBOARD_DIR, html=True), name="dashboard")


@app.get("/admin")
async def admin_redirect():
    return RedirectResponse(url="/dashboard/")
