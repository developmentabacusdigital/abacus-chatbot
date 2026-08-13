"""
Abacus Digital Chatbot - Authentication (PRD Phase 3 / 7.7)

Magic-link auth for existing clients, plus the admin API key guard for the internal
dashboard. Deliberately minimal: no passwords are stored, links are single-use and
short-lived, and session tokens are opaque random strings held server-side.
"""

import logging
import secrets
from typing import Optional

from fastapi import Header, HTTPException, Depends

from .config import settings
from .database import db
from .email_service import email_service
from .models import ClientRecord

logger = logging.getLogger(__name__)


# --- Admin guard ---

async def require_admin(x_admin_key: Optional[str] = Header(default=None)) -> bool:
    """
    Guard for internal dashboard/API routes.

    If ADMIN_API_KEY is unset the admin surface is refused outright rather than left
    open — an unauthenticated lead database is worse than a broken dashboard.
    """
    if not settings.admin_api_key:
        raise HTTPException(
            status_code=503,
            detail="Admin API is disabled. Set ADMIN_API_KEY to enable it.",
        )
    if not x_admin_key or not secrets.compare_digest(x_admin_key, settings.admin_api_key):
        raise HTTPException(status_code=401, detail="Invalid or missing admin key")
    return True


async def require_cron(
    authorization: Optional[str] = Header(default=None),
    x_admin_key: Optional[str] = Header(default=None),
) -> bool:
    """
    Guard for scheduled-job endpoints (/api/cron/*).

    Vercel automatically sends "Authorization: Bearer <CRON_SECRET>" when it invokes a
    cron-triggered request, once CRON_SECRET is set as a project env var — that's the
    primary path. The admin key also works so the same endpoints can be triggered by
    hand (curl, the dashboard) without needing a second secret to manage.
    """
    if settings.cron_secret:
        expected = f"Bearer {settings.cron_secret}"
        if authorization and secrets.compare_digest(authorization, expected):
            return True
    if settings.admin_api_key and x_admin_key:
        if secrets.compare_digest(x_admin_key, settings.admin_api_key):
            return True
    raise HTTPException(status_code=401, detail="Invalid or missing cron/admin credentials")


# --- Client auth ---

class ClientAuth:
    """Issues and verifies client magic links and session tokens."""

    async def request_magic_link(self, email: str) -> dict:
        """
        Start a client login.

        The response is identical whether or not the address belongs to a client, so
        the endpoint can't be used to enumerate the client list.
        """
        generic = {
            "success": True,
            "message": (
                "If that email is registered with us, a sign-in link is on its way. "
                "It expires in %d minutes." % settings.magic_link_ttl_minutes
            ),
            "debug_token": None,
        }

        if not settings.client_auth_enabled:
            return {
                "success": False,
                "message": "Client sign-in is not enabled yet.",
                "debug_token": None,
            }

        client = await db.get_client_by_email(email)
        if not client or not client.active:
            logger.info(f"Magic link requested for unknown/inactive address: {email}")
            return generic

        token = await db.create_magic_link(client.id, settings.magic_link_ttl_minutes)
        record = await email_service.send_magic_link(client.email, token)

        # Without a configured provider there is no way to deliver the link; surface it
        # locally so the flow is testable, and never when email is actually working.
        if not email_service.configured:
            logger.warning(
                "Email provider not configured — returning magic link token in the response. "
                "Configure EMAIL_PROVIDER before going live."
            )
            generic["debug_token"] = token
        elif record.status.value == "failed":
            logger.error(f"Magic link email failed: {record.error}")

        return generic

    async def verify_token(self, token: str) -> dict:
        """Redeem a magic link and issue a client session token."""
        client_id = await db.redeem_magic_link(token)
        if not client_id:
            return {
                "success": False,
                "message": "That sign-in link is invalid, already used, or expired. Please request a new one.",
            }

        client = await db.get_client(client_id)
        if not client or not client.active:
            return {"success": False, "message": "This account is no longer active."}

        session = await db.create_client_session(client_id, settings.client_session_ttl_hours)
        return {
            "success": True,
            "session_token": session["token"],
            "client_name": client.name,
            "client_company": client.company,
            "expires_at": session["expires_at"],
            "message": "Signed in.",
        }

    async def client_from_token(self, token: Optional[str]) -> Optional[ClientRecord]:
        if not token:
            return None
        client = await db.get_client_by_session_token(token)
        if client and client.active:
            return client
        return None


client_auth = ClientAuth()


async def require_client(
    authorization: Optional[str] = Header(default=None),
) -> ClientRecord:
    """FastAPI dependency: resolve the authenticated client or reject the request."""
    token = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()

    client = await client_auth.client_from_token(token)
    if not client:
        raise HTTPException(status_code=401, detail="Not signed in, or the session expired")
    return client
