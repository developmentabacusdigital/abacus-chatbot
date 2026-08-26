"""
Abacus Digital Chatbot - Transactional Email (PRD Phase 2 / 9.1)

Free-tier providers (Resend or Brevo). Follow-up emails are drafted by the bot but
queued for human approval by default; nothing reaches a visitor until someone on the
team approves it from the dashboard.
"""

import html
import logging
import re
from datetime import datetime
from typing import Dict, Any, Optional

import httpx

from .config import settings
from .database import db
from .models import EmailRecord, EmailStatus, ProjectBrief

logger = logging.getLogger(__name__)

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[a-zA-Z]{2,}$")


def is_valid_email(address: str) -> bool:
    return bool(address and EMAIL_RE.match(address.strip()))


class EmailService:
    """Drafts, queues, and sends transactional email."""

    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None

    async def initialize(self):
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(20.0))

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    @property
    def configured(self) -> bool:
        if settings.email_provider == "resend":
            return bool(settings.resend_api_key)
        if settings.email_provider == "brevo":
            return bool(settings.brevo_api_key)
        return False

    # --- Queue / send ---

    async def queue(
        self,
        to_email: str,
        subject: str,
        body: str,
        session_id: Optional[str] = None,
        force_send: bool = False,
    ) -> EmailRecord:
        """
        Persist an email. Sends immediately only when approval is not required
        (or the caller explicitly overrides, e.g. a magic link).
        """
        record = EmailRecord(
            session_id=session_id,
            to_email=to_email.strip(),
            subject=subject,
            body=body,
            status=EmailStatus.PENDING_APPROVAL,
        )

        if not is_valid_email(record.to_email):
            record.status = EmailStatus.FAILED
            record.error = "invalid recipient address"
            await db.save_email(record)
            return record

        should_send = force_send or not settings.email_require_approval
        if should_send:
            record.status = EmailStatus.APPROVED

        await db.save_email(record)

        if should_send:
            await self.send(record)

        return record

    async def approve_and_send(
        self, email_id: str, edited_body: Optional[str] = None
    ) -> Optional[EmailRecord]:
        """Human approval path from the dashboard."""
        record = await db.get_email(email_id)
        if not record or record.status not in (EmailStatus.PENDING_APPROVAL, EmailStatus.FAILED):
            return record
        if edited_body:
            record.body = edited_body
        record.status = EmailStatus.APPROVED
        await db.save_email(record)
        return await self.send(record)

    async def reject(self, email_id: str) -> Optional[EmailRecord]:
        record = await db.get_email(email_id)
        if not record:
            return None
        record.status = EmailStatus.REJECTED
        await db.save_email(record)
        return record

    async def send(self, record: EmailRecord) -> EmailRecord:
        """Deliver an approved email through the configured provider."""
        if not self.configured:
            record.status = EmailStatus.FAILED
            record.error = f"email provider '{settings.email_provider}' is not configured"
            logger.warning(f"Email {record.id} not sent: {record.error}")
            await db.save_email(record)
            return record

        await self.initialize()

        try:
            if settings.email_provider == "resend":
                message_id = await self._send_resend(record)
            else:
                message_id = await self._send_brevo(record)

            record.status = EmailStatus.SENT
            record.provider_message_id = message_id
            record.sent_at = datetime.utcnow().isoformat()
            record.error = None
            logger.info(f"Email {record.id} sent to {record.to_email}")
        except Exception as e:
            record.status = EmailStatus.FAILED
            record.error = str(e)[:500]
            logger.error(f"Email {record.id} failed: {e}")

        await db.save_email(record)
        return record

    async def _send_resend(self, record: EmailRecord) -> str:
        resp = await self._client.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {settings.resend_api_key}"},
            json={
                "from": settings.email_from,
                "to": [record.to_email],
                "reply_to": settings.email_reply_to,
                "subject": record.subject,
                "html": self._to_html(record.body),
                "text": record.body,
            },
        )
        resp.raise_for_status()
        return resp.json().get("id", "")

    async def _send_brevo(self, record: EmailRecord) -> str:
        name, address = self._parse_from(settings.email_from)
        resp = await self._client.post(
            "https://api.brevo.com/v3/smtp/email",
            headers={"api-key": settings.brevo_api_key, "accept": "application/json"},
            json={
                "sender": {"name": name, "email": address},
                "replyTo": {"email": settings.email_reply_to},
                "to": [{"email": record.to_email}],
                "subject": record.subject,
                "htmlContent": self._to_html(record.body),
                "textContent": record.body,
            },
        )
        resp.raise_for_status()
        return str(resp.json().get("messageId", ""))

    @staticmethod
    def _parse_from(from_header: str) -> tuple[str, str]:
        match = re.match(r"^\s*(.*?)\s*<([^>]+)>\s*$", from_header)
        if match:
            return match.group(1) or "Abacus Digital", match.group(2)
        return "Abacus Digital", from_header.strip()

    @staticmethod
    def _to_html(body: str) -> str:
        """Minimal, safe text-to-HTML. Visitor-supplied content is escaped."""
        escaped = html.escape(body)
        escaped = re.sub(r"^# (.+)$", r"<h2>\1</h2>", escaped, flags=re.MULTILINE)
        escaped = re.sub(r"^## (.+)$", r"<h3>\1</h3>", escaped, flags=re.MULTILINE)
        escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
        escaped = re.sub(
            r"(https?://[^\s<]+)", r'<a href="\1">\1</a>', escaped
        )
        escaped = escaped.replace("\n", "<br>")
        return (
            '<div style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;'
            'font-size:15px;line-height:1.6;color:#1a1a1a;max-width:600px">'
            f"{escaped}"
            '<hr style="border:none;border-top:1px solid #e5e5e5;margin:24px 0">'
            '<p style="font-size:12px;color:#888">Abacus Digital · '
            '<a href="https://www.abacusdigital.net">abacusdigital.net</a></p>'
            "</div>"
        )

    # --- Drafting ---

    async def draft_intake_followup(
        self,
        to_email: str,
        brief: ProjectBrief,
        session_id: str,
        visitor_name: Optional[str] = None,
    ) -> EmailRecord:
        """Draft the post-discovery summary email for the visitor (PRD 7.6)."""
        greeting = f"Hi {visitor_name}," if visitor_name else "Hi,"
        services = ", ".join(brief.recommended_services) or "the right mix of our services"

        body = f"""{greeting}

Thanks for taking the time to walk me through your project. Here's the summary I've passed to the Abacus Digital team so you can check I've got it right.

{brief.summary}

WHAT YOU'RE TRYING TO ACHIEVE
{chr(10).join('• ' + g for g in brief.goals) or '• (to confirm on the call)'}

WHERE THINGS STAND TODAY
{brief.current_state}

BUDGET AND TIMING
Budget band: {(brief.budget_band or 'to be confirmed').replace('_', ' ')}
Timeline: {(brief.timeline or 'to be confirmed').replace('_', ' ')}

WHAT WE'D SUGGEST
{services}
{brief.bundle_rationale}

If anything above is wrong or incomplete, just reply to this email and we'll correct it before the call.

You can book a time here: {settings.calendly_url}

Speak soon,
The Abacus Digital team"""

        return await self.queue(
            to_email=to_email,
            subject=f"Your project summary — {brief.title or 'Abacus Digital'}",
            body=body,
            session_id=session_id,
        )

    async def draft_booking_confirmation(
        self, to_email: str, session_id: str, visitor_name: Optional[str] = None
    ) -> EmailRecord:
        """Confirm booking details by email (PRD 7.3)."""
        greeting = f"Hi {visitor_name}," if visitor_name else "Hi,"
        body = f"""{greeting}

Thanks for chatting with us. Here's your link to book a free discovery call with the Abacus Digital team:

{settings.calendly_url}

The call runs about 30 minutes. We'll dig into your requirements, talk through the approach we'd take, and give you a clear picture of scope and next steps. No obligation either way.

If none of the times work, reply to this email and we'll find something that does.

The Abacus Digital team"""

        return await self.queue(
            to_email=to_email,
            subject="Book your discovery call with Abacus Digital",
            body=body,
            session_id=session_id,
        )

    async def send_magic_link(self, to_email: str, token: str) -> EmailRecord:
        """Phase 3 login link. Sent immediately — approval would defeat the purpose."""
        link = f"{settings.client_portal_url}?token={token}"
        body = f"""Hi,

Here's your secure sign-in link for the Abacus Digital client portal:

{link}

The link works once and expires in {settings.magic_link_ttl_minutes} minutes. If you didn't request it, you can ignore this email — no one can sign in without it.

The Abacus Digital team"""

        return await self.queue(
            to_email=to_email,
            subject="Your Abacus Digital sign-in link",
            body=body,
            force_send=True,
        )

    async def notify_new_lead(
        self,
        to_email: str,
        lead_data: Dict[str, Any],
        session_id: str,
    ) -> EmailRecord:
        """
        Internal alert fired the moment a visitor's name and email are both known — a
        snapshot of whatever's been gathered so far, not a customer-facing message, so
        it bypasses approval the same way an escalation alert does.
        """
        def field(key: str) -> str:
            value = lead_data.get(key)
            if value in (None, "", "null"):
                return "—"
            if isinstance(value, list):
                value = ", ".join(str(v) for v in value)
            return str(value).replace("_", " ")

        company_suffix = f" ({field('company')})" if lead_data.get("company") else ""

        body = f"""A new lead just shared their contact details in the chatbot.

Name: {field('name')}
Email: {field('email')}
Phone: {field('phone')}
Company: {field('company')}
Industry / business type: {field('business_type')}
Requirement / pain point: {field('pain_point')}
Goals: {field('goals')}
Budget band: {field('budget_band')}
Timeline: {field('timeline')}
Decision maker: {field('decision_maker')}
Service interest: {field('service_interest')}
Qualification score: {field('qualification_score')}

Session: {session_id}"""

        return await self.queue(
            to_email=to_email,
            subject=f"New lead: {field('name')}{company_suffix}",
            body=body,
            session_id=session_id,
            force_send=True,
        )

    async def notify_escalation(
        self,
        manager_email: str,
        client_name: str,
        client_company: str,
        reason: str,
        transcript_summary: str,
        session_id: str,
    ) -> EmailRecord:
        """Alert the account manager that a client conversation needs a human (PRD 7.7)."""
        body = f"""A client conversation has been escalated from the support chat.

Client: {client_name} ({client_company})
Reason: {reason}
Session: {session_id}

Conversation summary:
{transcript_summary}

Please follow up with the client directly."""

        return await self.queue(
            to_email=manager_email,
            subject=f"[Escalation] {client_company or client_name} needs a human",
            body=body,
            session_id=session_id,
            force_send=True,
        )


email_service = EmailService()
