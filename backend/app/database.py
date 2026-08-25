"""
Abacus Digital Chatbot - Database Layer
Async Postgres (Neon or any standard Postgres) for sessions, leads, messages, briefs,
emails and clients. This store is the CRM system of record (PRD 7.4/7.5).

Public method signatures are unchanged from the previous SQLite implementation, so
every caller elsewhere in the app (chat_orchestrator, auth, main, etc.) works as-is —
only the SQL dialect and connection layer changed underneath.

Postgres handles concurrent writers natively, so the SQLite-era `asyncio.Lock` around
every write is gone; asyncpg's connection pool is safe to use from concurrent requests
without one. Where the old code did a check-then-write (SELECT to decide INSERT vs
UPDATE), that's now a real `ON CONFLICT` upsert instead — the old pattern was a race
condition waiting to happen once more than one request could run at a time.
"""

import csv
import io
import json
import logging
import re
import secrets
import uuid
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta, timezone

import asyncpg

from .config import settings
from .models import (
    SessionRecord, LeadRecord, MessageRecord, ProjectBrief, EmailRecord,
    ClientRecord, ClientProject, LeadSummary, SessionStatus, SurfaceType,
    EmailStatus, LEAD_WRITABLE_FIELDS,
)

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat()


def _parse_iso(value: str) -> datetime:
    """Timestamps are stored as plain ISO strings (no tz); parse back the same way."""
    return datetime.fromisoformat(value)


def _rowcount(status: str) -> int:
    """asyncpg's execute() returns a command tag like 'DELETE 3' or 'UPDATE 1'."""
    match = re.search(r"(\d+)$", status or "")
    return int(match.group(1)) if match else 0


class Database:
    """Async Postgres database manager."""

    def __init__(self, database_url: str = None):
        self.database_url = database_url or settings.database_url
        self._pool: Optional[asyncpg.Pool] = None
        self._local_pg_server = None

    async def connect(self):
        """Initialize the connection pool and create tables."""
        if not self.database_url:
            try:
                import pgserver
                import os
                pg_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "postgres"))
                os.makedirs(pg_data_dir, exist_ok=True)
                logger.info(f"DATABASE_URL not set; starting local embedded Postgres at {pg_data_dir}...")
                self._local_pg_server = pgserver.get_server(pg_data_dir)
                self.database_url = self._local_pg_server.get_uri()
                logger.info(f"Local Postgres running at {self.database_url}")
            except Exception as e:
                logger.warning(f"Could not start embedded pgserver: {e}")
                raise RuntimeError(
                    "DATABASE_URL is not set and local pgserver failed to start. "
                    "Use Neon's pooled connection string or set DATABASE_URL."
                ) from e
        self._pool = await asyncpg.create_pool(
            self.database_url,
            min_size=settings.database_pool_min_size,
            max_size=settings.database_pool_max_size,
            command_timeout=30,
        )
        await self._create_tables()
        await self._migrate()

    async def disconnect(self):
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
        if self._local_pg_server:
            try:
                self._local_pg_server.cleanup()
            except Exception:
                pass
            self._local_pg_server = None

    async def _create_tables(self):
        """Create database tables if they don't exist."""
        await self._pool.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                seq BIGSERIAL,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                source_page TEXT,
                consent_given BOOLEAN DEFAULT FALSE,
                status TEXT DEFAULT 'active',
                ip_address TEXT,
                user_agent TEXT,
                total_messages INTEGER DEFAULT 0,
                total_cost REAL DEFAULT 0.0
            );

            CREATE TABLE IF NOT EXISTS leads (
                id TEXT PRIMARY KEY,
                seq BIGSERIAL,
                session_id TEXT NOT NULL UNIQUE,
                name TEXT,
                email TEXT,
                phone TEXT,
                company TEXT,
                business_type TEXT,
                pain_point TEXT,
                budget_band TEXT,
                timeline TEXT,
                decision_maker BOOLEAN,
                service_interest TEXT,
                qualification_score REAL DEFAULT 0.0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                intent TEXT,
                model_used TEXT,
                cost REAL DEFAULT 0.0
            );

            CREATE TABLE IF NOT EXISTS briefs (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                title TEXT,
                summary TEXT,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS emails (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                to_email TEXT NOT NULL,
                subject TEXT NOT NULL,
                body TEXT NOT NULL,
                status TEXT NOT NULL,
                provider_message_id TEXT,
                error TEXT,
                created_at TEXT NOT NULL,
                sent_at TEXT
            );

            CREATE TABLE IF NOT EXISTS clients (
                id TEXT PRIMARY KEY,
                email TEXT NOT NULL UNIQUE,
                name TEXT,
                company TEXT,
                account_manager_email TEXT,
                active BOOLEAN DEFAULT TRUE,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS client_projects (
                id TEXT PRIMARY KEY,
                client_id TEXT NOT NULL,
                name TEXT NOT NULL,
                status TEXT,
                service_line TEXT,
                started_at TEXT,
                target_date TEXT,
                progress_notes TEXT,
                deliverables TEXT,
                support_docs TEXT,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS magic_links (
                token TEXT PRIMARY KEY,
                client_id TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                used BOOLEAN DEFAULT FALSE,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS client_sessions (
                token TEXT PRIMARY KEY,
                client_id TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS escalations (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                client_id TEXT,
                reason TEXT,
                transcript_summary TEXT,
                resolved BOOLEAN DEFAULT FALSE,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_leads_email ON leads(email);
            CREATE INDEX IF NOT EXISTS idx_leads_session ON leads(session_id);
            CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
            CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);
            CREATE INDEX IF NOT EXISTS idx_sessions_ip ON sessions(ip_address);
            CREATE INDEX IF NOT EXISTS idx_briefs_session ON briefs(session_id);
            CREATE INDEX IF NOT EXISTS idx_emails_status ON emails(status);
            CREATE INDEX IF NOT EXISTS idx_projects_client ON client_projects(client_id);
            CREATE INDEX IF NOT EXISTS idx_client_sessions_client ON client_sessions(client_id);
        """)

    async def _migrate(self):
        """
        Add columns introduced after the first schema version.
        Postgres' `ADD COLUMN IF NOT EXISTS` is idempotent on its own, so this no
        longer needs the SQLite-era "check information_schema, then maybe ALTER" dance.
        """
        statements = [
            "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS surface TEXT DEFAULT 'public'",
            "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS client_id TEXT",
            "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS summary TEXT",
            "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS next_step TEXT",
            "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS visitor_id TEXT",
            "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS title TEXT",
            "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS category TEXT DEFAULT 'other'",
            "ALTER TABLE leads ADD COLUMN IF NOT EXISTS transcript_summary TEXT",
            "ALTER TABLE leads ADD COLUMN IF NOT EXISTS next_step TEXT",
            "CREATE INDEX IF NOT EXISTS idx_sessions_visitor ON sessions(visitor_id)",
        ]
        for stmt in statements:
            await self._pool.execute(stmt)

    # --- Session Operations ---

    async def create_session(self, session: SessionRecord) -> SessionRecord:
        """Create a new chat session."""
        await self._pool.execute(
            """INSERT INTO sessions
               (id, started_at, source_page, consent_given, status, surface,
                client_id, ip_address, user_agent, visitor_id, title, category)
               VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)""",
            session.id, session.started_at, session.source_page,
            bool(session.consent_given), session.status.value, session.surface.value,
            session.client_id, session.ip_address, session.user_agent,
            session.visitor_id, session.title, session.category,
        )
        return session

    async def get_session(self, session_id: str) -> Optional[SessionRecord]:
        """Retrieve a session by ID."""
        row = await self._pool.fetchrow("SELECT * FROM sessions WHERE id = $1", session_id)
        return self._session_from_row(row) if row else None

    @staticmethod
    def _session_from_row(row) -> SessionRecord:
        data = dict(row)
        return SessionRecord(
            id=data["id"],
            started_at=data["started_at"],
            ended_at=data["ended_at"],
            source_page=data["source_page"],
            consent_given=bool(data["consent_given"]),
            status=SessionStatus(data["status"]),
            surface=SurfaceType(data["surface"] or "public"),
            client_id=data["client_id"],
            ip_address=data["ip_address"],
            user_agent=data["user_agent"],
            total_messages=data["total_messages"],
            total_cost=data["total_cost"],
            summary=data["summary"],
            next_step=data["next_step"],
            visitor_id=data.get("visitor_id"),
            title=data.get("title"),
            category=data.get("category") or "other",
        )

    SESSION_WRITABLE_FIELDS = frozenset({
        "ended_at", "source_page", "consent_given", "status", "surface",
        "client_id", "summary", "next_step", "title", "category", "visitor_id",
    })

    async def update_session(self, session_id: str, **kwargs):
        """Update session fields. Field names are whitelisted before hitting SQL."""
        clean: Dict[str, Any] = {}
        for key, value in kwargs.items():
            if key not in self.SESSION_WRITABLE_FIELDS:
                logger.warning(f"Ignoring non-writable session field: {key}")
                continue
            if key == "consent_given":
                value = bool(value)
            elif isinstance(value, (SessionStatus, SurfaceType)):
                value = value.value
            clean[key] = value

        if not clean:
            return

        set_clauses = [f"{k} = ${i + 1}" for i, k in enumerate(clean.keys())]
        values = list(clean.values())
        values.append(session_id)
        await self._pool.execute(
            f"UPDATE sessions SET {', '.join(set_clauses)} WHERE id = ${len(values)}",
            *values,
        )

    async def increment_session_messages(self, session_id: str, cost: float = 0.0):
        """Increment message count and add cost for a session."""
        await self._pool.execute(
            """UPDATE sessions
               SET total_messages = total_messages + 1, total_cost = total_cost + $1
               WHERE id = $2""",
            cost, session_id,
        )

    async def get_sessions_by_visitor(
        self, visitor_id: str, limit: int = 30, exclude_session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Chat-list rows for one visitor, newest first.

        Each row includes a snippet: the stored summary once a chat is finalized, else
        the visitor's own most recent message, so the list is useful mid-conversation too.
        """
        rows = await self._pool.fetch(
            """SELECT s.id, s.title, s.summary, s.status, s.category,
                      s.started_at, s.total_messages,
                      COALESCE(s.ended_at, s.started_at) AS updated_at,
                      (SELECT m.content FROM messages m
                        WHERE m.session_id = s.id AND m.role = 'user'
                        ORDER BY m.timestamp DESC LIMIT 1) AS last_user_message
               FROM sessions s
               WHERE s.visitor_id = $1 AND s.surface = 'public' AND s.id != $2
                 AND s.total_messages > 0
               ORDER BY updated_at DESC, s.seq DESC
               LIMIT $3""",
            visitor_id, exclude_session_id or "", limit,
        )
        result = []
        for row in rows:
            data = dict(row)
            snippet = data["summary"] or data["last_user_message"] or ""
            result.append({
                "session_id": data["id"],
                "title": data["title"],
                "snippet": snippet[:140],
                "status": data["status"],
                "category": data["category"] or "other",
                "started_at": data["started_at"],
                "updated_at": data["updated_at"],
                "message_count": data["total_messages"] or 0,
            })
        return result

    async def get_session_for_visitor(
        self, session_id: str, visitor_id: str
    ) -> Optional[SessionRecord]:
        """Fetch a session only if it belongs to the given visitor (ownership check)."""
        row = await self._pool.fetchrow(
            "SELECT * FROM sessions WHERE id = $1 AND visitor_id = $2 AND surface = 'public'",
            session_id, visitor_id,
        )
        return self._session_from_row(row) if row else None

    async def get_visitor_memory(self, visitor_id: str, exclude_session_id: str) -> Dict[str, Any]:
        """
        Condensed cross-session context for a returning visitor (used to prime a new
        chat so it doesn't re-ask what's already known, and to avoid re-asking for email).

        Returns {"email": ..., "known_fields": {...}, "summaries": [...]}.
        """
        lead_row = await self._pool.fetchrow(
            """SELECT l.email, l.name, l.company, l.business_type, l.pain_point,
                      l.service_interest, l.budget_band, l.timeline
               FROM leads l
               JOIN sessions s ON s.id = l.session_id
               WHERE s.visitor_id = $1 AND s.id != $2 AND l.email IS NOT NULL
               ORDER BY l.updated_at DESC, l.seq DESC LIMIT 1""",
            visitor_id, exclude_session_id,
        )
        known_fields = {}
        if lead_row:
            known_fields = {k: v for k, v in dict(lead_row).items() if v}

        summary_rows = await self._pool.fetch(
            """SELECT summary FROM sessions
               WHERE visitor_id = $1 AND id != $2 AND summary IS NOT NULL
               ORDER BY COALESCE(ended_at, started_at) DESC, seq DESC LIMIT 3""",
            visitor_id, exclude_session_id,
        )
        summaries = [row["summary"] for row in summary_rows if row["summary"]]

        return {
            "email": known_fields.get("email"),
            "known_fields": known_fields,
            "summaries": summaries,
        }

    async def get_stale_sessions(self, minutes: int = 30) -> List[str]:
        """Active sessions with no activity for `minutes`, for abandonment sweep."""
        cutoff = (datetime.utcnow() - timedelta(minutes=minutes)).isoformat()
        rows = await self._pool.fetch(
            """SELECT s.id FROM sessions s
               WHERE s.status = 'active'
                 AND COALESCE(
                       (SELECT MAX(m.timestamp) FROM messages m WHERE m.session_id = s.id),
                       s.started_at
                     ) < $1""",
            cutoff,
        )
        return [row["id"] for row in rows]

    # --- Lead Operations ---

    async def create_or_update_lead(self, lead: LeadRecord) -> LeadRecord:
        """Create or update a lead record. Deduplicates by email where possible."""
        if lead.email:
            existing = await self._get_lead_by_email(lead.email)
            if existing:
                await self._merge_lead(existing["id"], lead)
                lead.id = existing["id"]
                return lead

        await self._pool.execute(
            """INSERT INTO leads
               (id, session_id, name, email, phone, company, business_type,
                pain_point, budget_band, timeline, decision_maker,
                service_interest, qualification_score, transcript_summary,
                next_step, created_at, updated_at)
               VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17)
               ON CONFLICT (session_id) DO UPDATE SET
                 name = EXCLUDED.name, email = EXCLUDED.email, phone = EXCLUDED.phone,
                 company = EXCLUDED.company, business_type = EXCLUDED.business_type,
                 pain_point = EXCLUDED.pain_point, budget_band = EXCLUDED.budget_band,
                 timeline = EXCLUDED.timeline, decision_maker = EXCLUDED.decision_maker,
                 service_interest = EXCLUDED.service_interest,
                 qualification_score = EXCLUDED.qualification_score,
                 transcript_summary = EXCLUDED.transcript_summary,
                 next_step = EXCLUDED.next_step, updated_at = EXCLUDED.updated_at""",
            lead.id, lead.session_id, lead.name, lead.email, lead.phone,
            lead.company, lead.business_type, lead.pain_point,
            lead.budget_band, lead.timeline, lead.decision_maker,
            lead.service_interest, lead.qualification_score,
            lead.transcript_summary, lead.next_step,
            lead.created_at, lead.updated_at,
        )
        return lead

    async def _get_lead_by_email(self, email: str) -> Optional[dict]:
        """Find an existing lead by email."""
        row = await self._pool.fetchrow(
            "SELECT * FROM leads WHERE email = $1 ORDER BY updated_at DESC LIMIT 1", email,
        )
        return dict(row) if row else None

    async def _merge_lead(self, existing_id: str, new_lead: LeadRecord):
        """Merge new lead data into existing record (non-null fields win)."""
        updates: Dict[str, Any] = {}
        for field in ["name", "phone", "company", "business_type", "pain_point",
                      "budget_band", "timeline", "service_interest",
                      "transcript_summary", "next_step"]:
            val = getattr(new_lead, field)
            if val is not None:
                updates[field] = val
        if new_lead.decision_maker is not None:
            updates["decision_maker"] = new_lead.decision_maker
        if new_lead.qualification_score > 0:
            updates["qualification_score"] = new_lead.qualification_score
        updates["updated_at"] = _now()
        updates["session_id"] = new_lead.session_id

        set_clauses = [f"{k} = ${i + 1}" for i, k in enumerate(updates.keys())]
        values = list(updates.values()) + [existing_id]
        await self._pool.execute(
            f"UPDATE leads SET {', '.join(set_clauses)} WHERE id = ${len(values)}",
            *values,
        )

    async def get_lead_by_session(self, session_id: str) -> Optional[LeadRecord]:
        """Get lead record for a session."""
        row = await self._pool.fetchrow(
            "SELECT * FROM leads WHERE session_id = $1 ORDER BY updated_at DESC LIMIT 1",
            session_id,
        )
        if not row:
            return None
        data = dict(row)
        if data.get("decision_maker") is not None:
            data["decision_maker"] = bool(data["decision_maker"])
        data.pop("seq", None)
        return LeadRecord(**data)

    async def update_lead_fields(self, session_id: str, fields: Dict[str, Any]):
        """
        Update one or more fields on a lead (progressive capture, PRD 7.5).

        Field names originate from LLM-extracted data, so they are validated against
        LEAD_WRITABLE_FIELDS before being interpolated into the statement. Backed by a
        real upsert (leads.session_id is UNIQUE) so concurrent calls for the same
        session can't race into duplicate rows the way a check-then-write would.
        """
        clean = {
            k: v for k, v in fields.items()
            if k in LEAD_WRITABLE_FIELDS and v is not None
        }
        rejected = set(fields) - set(clean) - {k for k, v in fields.items() if v is None}
        if rejected:
            logger.warning(f"Rejected non-writable lead fields: {sorted(rejected)}")
        if not clean:
            return

        if "decision_maker" in clean:
            clean["decision_maker"] = bool(clean["decision_maker"])

        now = _now()
        columns = ["id", "session_id", "created_at", "updated_at"] + list(clean.keys())
        values = [str(uuid.uuid4()), session_id, now, now] + list(clean.values())
        placeholders = [f"${i + 1}" for i in range(len(values))]

        update_set = ", ".join(f"{k} = EXCLUDED.{k}" for k in clean.keys())
        update_set += ", updated_at = EXCLUDED.updated_at"

        await self._pool.execute(
            f"""INSERT INTO leads ({', '.join(columns)})
                VALUES ({', '.join(placeholders)})
                ON CONFLICT (session_id) DO UPDATE SET {update_set}""",
            *values,
        )

    async def update_lead_field(self, session_id: str, field: str, value: Any):
        """Update a single lead field (thin wrapper over update_lead_fields)."""
        await self.update_lead_fields(session_id, {field: value})

    # --- Message Operations ---

    async def save_message(self, message: MessageRecord):
        """Save a chat message."""
        await self._pool.execute(
            """INSERT INTO messages (id, session_id, role, content, timestamp, intent, model_used, cost)
               VALUES ($1,$2,$3,$4,$5,$6,$7,$8)""",
            message.id, message.session_id, message.role, message.content,
            message.timestamp, message.intent, message.model_used, message.cost,
        )

    async def get_session_messages(self, session_id: str) -> List[MessageRecord]:
        """Get all messages for a session."""
        rows = await self._pool.fetch(
            "SELECT * FROM messages WHERE session_id = $1 ORDER BY timestamp ASC", session_id,
        )
        return [MessageRecord(**dict(row)) for row in rows]

    # --- Brief Operations (Phase 2) ---

    async def save_brief(self, brief: ProjectBrief) -> ProjectBrief:
        """Persist a structured project brief."""
        await self._pool.execute(
            """INSERT INTO briefs (id, session_id, title, summary, payload, created_at)
               VALUES ($1,$2,$3,$4,$5,$6)
               ON CONFLICT (id) DO UPDATE SET
                 title = EXCLUDED.title, summary = EXCLUDED.summary, payload = EXCLUDED.payload""",
            brief.id, brief.session_id, brief.title, brief.summary,
            brief.model_dump_json(), brief.created_at,
        )
        return brief

    async def get_brief_by_session(self, session_id: str) -> Optional[ProjectBrief]:
        row = await self._pool.fetchrow(
            "SELECT payload FROM briefs WHERE session_id = $1 ORDER BY created_at DESC LIMIT 1",
            session_id,
        )
        return ProjectBrief(**json.loads(row["payload"])) if row else None

    async def get_all_briefs(self, limit: int = 50) -> List[ProjectBrief]:
        rows = await self._pool.fetch(
            "SELECT payload FROM briefs ORDER BY created_at DESC LIMIT $1", limit,
        )
        return [ProjectBrief(**json.loads(r["payload"])) for r in rows]

    # --- Email Operations (Phase 2) ---

    async def save_email(self, email: EmailRecord) -> EmailRecord:
        await self._pool.execute(
            """INSERT INTO emails
               (id, session_id, to_email, subject, body, status,
                provider_message_id, error, created_at, sent_at)
               VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
               ON CONFLICT (id) DO UPDATE SET
                 status = EXCLUDED.status, provider_message_id = EXCLUDED.provider_message_id,
                 error = EXCLUDED.error, sent_at = EXCLUDED.sent_at, body = EXCLUDED.body""",
            email.id, email.session_id, email.to_email, email.subject, email.body,
            email.status.value, email.provider_message_id, email.error,
            email.created_at, email.sent_at,
        )
        return email

    async def get_email(self, email_id: str) -> Optional[EmailRecord]:
        row = await self._pool.fetchrow("SELECT * FROM emails WHERE id = $1", email_id)
        if not row:
            return None
        data = dict(row)
        data["status"] = EmailStatus(data["status"])
        return EmailRecord(**data)

    async def get_emails(self, status: Optional[str] = None, limit: int = 50) -> List[EmailRecord]:
        if status:
            rows = await self._pool.fetch(
                "SELECT * FROM emails WHERE status = $1 ORDER BY created_at DESC LIMIT $2",
                status, limit,
            )
        else:
            rows = await self._pool.fetch(
                "SELECT * FROM emails ORDER BY created_at DESC LIMIT $1", limit,
            )
        out = []
        for row in rows:
            data = dict(row)
            data["status"] = EmailStatus(data["status"])
            out.append(EmailRecord(**data))
        return out

    # --- Client Operations (Phase 3) ---

    async def upsert_client(self, client: ClientRecord) -> ClientRecord:
        existing = await self.get_client_by_email(client.email)
        if existing:
            client.id = existing.id
            await self._pool.execute(
                """UPDATE clients SET name = $1, company = $2, account_manager_email = $3, active = $4
                   WHERE id = $5""",
                client.name, client.company, client.account_manager_email,
                bool(client.active), client.id,
            )
            return client

        await self._pool.execute(
            """INSERT INTO clients (id, email, name, company, account_manager_email, active, created_at)
               VALUES ($1,$2,$3,$4,$5,$6,$7)""",
            client.id, client.email.lower(), client.name, client.company,
            client.account_manager_email, bool(client.active), client.created_at,
        )
        return client

    async def get_client_by_email(self, email: str) -> Optional[ClientRecord]:
        row = await self._pool.fetchrow(
            "SELECT * FROM clients WHERE email = $1", email.lower().strip(),
        )
        if not row:
            return None
        data = dict(row)
        data["active"] = bool(data["active"])
        return ClientRecord(**data)

    async def get_client(self, client_id: str) -> Optional[ClientRecord]:
        row = await self._pool.fetchrow("SELECT * FROM clients WHERE id = $1", client_id)
        if not row:
            return None
        data = dict(row)
        data["active"] = bool(data["active"])
        return ClientRecord(**data)

    async def upsert_project(self, project: ClientProject) -> ClientProject:
        await self._pool.execute(
            """INSERT INTO client_projects
               (id, client_id, name, status, service_line, started_at, target_date,
                progress_notes, deliverables, support_docs, updated_at)
               VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
               ON CONFLICT (id) DO UPDATE SET
                 name = EXCLUDED.name, status = EXCLUDED.status,
                 service_line = EXCLUDED.service_line, started_at = EXCLUDED.started_at,
                 target_date = EXCLUDED.target_date, progress_notes = EXCLUDED.progress_notes,
                 deliverables = EXCLUDED.deliverables, support_docs = EXCLUDED.support_docs,
                 updated_at = EXCLUDED.updated_at""",
            project.id, project.client_id, project.name, project.status,
            project.service_line, project.started_at, project.target_date,
            project.progress_notes, json.dumps(project.deliverables),
            json.dumps(project.support_docs), project.updated_at,
        )
        return project

    async def get_client_projects(self, client_id: str) -> List[ClientProject]:
        rows = await self._pool.fetch(
            "SELECT * FROM client_projects WHERE client_id = $1 ORDER BY updated_at DESC",
            client_id,
        )
        out = []
        for row in rows:
            data = dict(row)
            data["deliverables"] = json.loads(data["deliverables"] or "[]")
            data["support_docs"] = json.loads(data["support_docs"] or "[]")
            out.append(ClientProject(**data))
        return out

    async def get_all_projects(self) -> List[ClientProject]:
        rows = await self._pool.fetch("SELECT * FROM client_projects")
        out = []
        for row in rows:
            data = dict(row)
            data["deliverables"] = json.loads(data["deliverables"] or "[]")
            data["support_docs"] = json.loads(data["support_docs"] or "[]")
            out.append(ClientProject(**data))
        return out

    # --- Auth tokens (Phase 3) ---

    async def create_magic_link(self, client_id: str, ttl_minutes: int) -> str:
        token = secrets.token_urlsafe(32)
        expires = (datetime.utcnow() + timedelta(minutes=ttl_minutes)).isoformat()
        await self._pool.execute(
            "INSERT INTO magic_links (token, client_id, expires_at, used, created_at) VALUES ($1,$2,$3,FALSE,$4)",
            token, client_id, expires, _now(),
        )
        return token

    async def redeem_magic_link(self, token: str) -> Optional[str]:
        """Consume a magic link, returning the client_id if it is valid and unused."""
        row = await self._pool.fetchrow("SELECT * FROM magic_links WHERE token = $1", token)
        if not row or row["used"]:
            return None
        if _parse_iso(row["expires_at"]) < datetime.utcnow():
            return None
        await self._pool.execute("UPDATE magic_links SET used = TRUE WHERE token = $1", token)
        return row["client_id"]

    async def create_client_session(self, client_id: str, ttl_hours: int) -> Dict[str, str]:
        token = secrets.token_urlsafe(40)
        expires = (datetime.utcnow() + timedelta(hours=ttl_hours)).isoformat()
        await self._pool.execute(
            "INSERT INTO client_sessions (token, client_id, expires_at, created_at) VALUES ($1,$2,$3,$4)",
            token, client_id, expires, _now(),
        )
        return {"token": token, "expires_at": expires}

    async def get_client_by_session_token(self, token: str) -> Optional[ClientRecord]:
        row = await self._pool.fetchrow("SELECT * FROM client_sessions WHERE token = $1", token)
        if not row:
            return None
        if _parse_iso(row["expires_at"]) < datetime.utcnow():
            return None
        return await self.get_client(row["client_id"])

    async def revoke_client_session(self, token: str):
        await self._pool.execute("DELETE FROM client_sessions WHERE token = $1", token)

    async def purge_expired_tokens(self) -> int:
        now = _now()
        s1 = await self._pool.execute("DELETE FROM magic_links WHERE expires_at < $1", now)
        s2 = await self._pool.execute("DELETE FROM client_sessions WHERE expires_at < $1", now)
        return _rowcount(s1) + _rowcount(s2)

    # --- Escalations (Phase 3) ---

    async def create_escalation(
        self, session_id: str, reason: str,
        client_id: Optional[str] = None, transcript_summary: Optional[str] = None,
    ) -> str:
        esc_id = str(uuid.uuid4())
        await self._pool.execute(
            """INSERT INTO escalations (id, session_id, client_id, reason, transcript_summary, resolved, created_at)
               VALUES ($1,$2,$3,$4,$5,FALSE,$6)""",
            esc_id, session_id, client_id, reason, transcript_summary, _now(),
        )
        return esc_id

    async def get_escalations(self, resolved: bool = False, limit: int = 50) -> List[Dict[str, Any]]:
        rows = await self._pool.fetch(
            "SELECT * FROM escalations WHERE resolved = $1 ORDER BY created_at DESC LIMIT $2",
            resolved, limit,
        )
        return [dict(row) for row in rows]

    # --- Dashboard / Export Operations ---

    async def get_all_leads(self, limit: int = 100, offset: int = 0) -> List[LeadSummary]:
        """Get all leads for the internal dashboard."""
        rows = await self._pool.fetch(
            """SELECT l.*, s.status AS session_status, s.category AS session_category,
                      s.started_at AS session_started, s.total_messages,
                      (SELECT COUNT(*) FROM briefs b WHERE b.session_id = l.session_id) AS brief_count
               FROM leads l
               LEFT JOIN sessions s ON l.session_id = s.id
               ORDER BY l.updated_at DESC
               LIMIT $1 OFFSET $2""",
            limit, offset,
        )
        return [
            LeadSummary(
                session_id=row["session_id"],
                name=row["name"],
                email=row["email"],
                company=row["company"],
                business_type=row["business_type"],
                pain_point=row["pain_point"],
                service_interest=row["service_interest"],
                budget_band=row["budget_band"],
                timeline=row["timeline"],
                qualification_score=row["qualification_score"] or 0.0,
                transcript_summary=row["transcript_summary"],
                next_step=row["next_step"],
                status=row["session_status"] or "unknown",
                category=row["session_category"] or "other",
                started_at=row["session_started"] or row["created_at"],
                message_count=row["total_messages"] or 0,
                has_brief=bool(row["brief_count"]),
            )
            for row in rows
        ]

    async def export_leads_csv(self) -> str:
        """Export every lead as CSV for the team (PRD 7.4)."""
        leads = await self.get_all_leads(limit=10000)
        buf = io.StringIO()
        writer = csv.writer(buf, lineterminator="\n")
        writer.writerow([
            "session_id", "category", "name", "email", "company", "business_type", "pain_point",
            "service_interest", "budget_band", "timeline", "qualification_score",
            "status", "started_at", "message_count", "has_brief",
            "next_step", "transcript_summary",
        ])
        for l in leads:
            writer.writerow([
                l.session_id, l.category, l.name or "", l.email or "", l.company or "",
                l.business_type or "", l.pain_point or "", l.service_interest or "",
                l.budget_band or "", l.timeline or "", f"{l.qualification_score:.3f}",
                l.status, l.started_at, l.message_count, "yes" if l.has_brief else "no",
                l.next_step or "", (l.transcript_summary or "").replace("\n", " "),
            ])
        return buf.getvalue()

    async def get_session_transcript(self, session_id: str) -> List[Dict[str, str]]:
        """Get the full transcript for a session."""
        messages = await self.get_session_messages(session_id)
        return [{"role": m.role, "content": m.content, "timestamp": m.timestamp}
                for m in messages]

    async def get_metrics(self) -> Dict[str, Any]:
        """Aggregate metrics for the dashboard (PRD Section 10)."""
        async def scalar(sql: str, *params) -> Any:
            return await self._pool.fetchval(sql, *params)

        total_sessions = await scalar("SELECT COUNT(*) FROM sessions WHERE surface = 'public'")
        total_leads = await scalar("SELECT COUNT(*) FROM leads")
        qualified = await scalar("SELECT COUNT(*) FROM leads WHERE qualification_score >= 0.5")
        booked = await scalar("SELECT COUNT(*) FROM sessions WHERE status = 'booked'")
        briefs = await scalar("SELECT COUNT(*) FROM briefs")
        total_cost = await scalar("SELECT COALESCE(SUM(total_cost), 0) FROM sessions")
        escalations = await scalar("SELECT COUNT(*) FROM escalations WHERE resolved = FALSE")
        client_sessions = await scalar("SELECT COUNT(*) FROM sessions WHERE surface = 'client'")
        complete_leads = await scalar(
            """SELECT COUNT(*) FROM leads
               WHERE (email IS NOT NULL OR name IS NOT NULL)
                 AND business_type IS NOT NULL AND pain_point IS NOT NULL"""
        )
        leads_category = await scalar(
            "SELECT COUNT(*) FROM sessions WHERE surface = 'public' AND category = 'lead'"
        )
        queries_category = await scalar(
            "SELECT COUNT(*) FROM sessions WHERE surface = 'public' AND category = 'query'"
        )
        other_category = await scalar(
            "SELECT COUNT(*) FROM sessions WHERE surface = 'public' AND category = 'other'"
        )

        return {
            "total_sessions": total_sessions,
            "total_leads": total_leads,
            "qualified_leads": qualified,
            "qualification_rate": round(qualified / total_leads, 3) if total_leads else 0.0,
            "bookings": booked,
            "briefs_generated": briefs,
            "record_completeness": round(complete_leads / total_leads, 3) if total_leads else 0.0,
            "total_llm_cost_usd": round(total_cost, 6),
            "cost_per_conversation_usd": round(total_cost / total_sessions, 6) if total_sessions else 0.0,
            "open_escalations": escalations,
            "client_sessions": client_sessions,
            "sessions_by_category": {
                "lead": leads_category, "query": queries_category, "other": other_category,
            },
        }

    # --- Data Deletion (GDPR) ---

    async def delete_by_email(self, email: str) -> int:
        """Delete all data associated with an email."""
        rows = await self._pool.fetch("SELECT session_id FROM leads WHERE email = $1", email)
        session_ids = [row["session_id"] for row in rows]

        total = 0
        for sid in session_ids:
            total += await self.delete_by_session(sid)

        # Catch orphaned lead rows that share the email but no live session
        status = await self._pool.execute("DELETE FROM leads WHERE email = $1", email)
        total += _rowcount(status)
        return total

    async def delete_by_session(self, session_id: str) -> int:
        """Delete all data for a specific session."""
        total = 0
        for table, col in [
            ("messages", "session_id"), ("leads", "session_id"),
            ("briefs", "session_id"), ("emails", "session_id"),
            ("escalations", "session_id"), ("sessions", "id"),
        ]:
            status = await self._pool.execute(f"DELETE FROM {table} WHERE {col} = $1", session_id)
            total += _rowcount(status)
        return total

    async def delete_by_visitor(self, visitor_id: str) -> int:
        """Delete every session (and its data) tied to a browser-scoped visitor_id."""
        rows = await self._pool.fetch("SELECT id FROM sessions WHERE visitor_id = $1", visitor_id)
        total = 0
        for row in rows:
            total += await self.delete_by_session(row["id"])
        return total

    # --- Rate Limiting ---

    async def count_sessions_by_ip(self, ip_address: str, hours: int = 1) -> int:
        """Count recent sessions from an IP for rate limiting."""
        cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
        count = await self._pool.fetchval(
            "SELECT COUNT(*) FROM sessions WHERE ip_address = $1 AND started_at > $2",
            ip_address, cutoff,
        )
        return count or 0


# Singleton instance
db = Database()
