"""
Abacus Digital Chatbot - Data Models
Pydantic models for sessions, leads, messages, briefs, clients, and API contracts.
"""

from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
import uuid


def _now() -> str:
    return datetime.utcnow().isoformat()


def _uuid() -> str:
    return str(uuid.uuid4())


# --- Enums ---

class Intent(str, Enum):
    GREETING = "greeting"
    QUESTION = "question"
    QUALIFICATION = "qualification"
    BOOKING = "booking"
    PROJECT_INTAKE = "project_intake"
    GENERAL = "general"
    FAREWELL = "farewell"
    DATA_DELETION = "data_deletion"
    HUMAN_HANDOFF = "human_handoff"


class SessionStatus(str, Enum):
    ACTIVE = "active"
    QUALIFIED = "qualified"
    BOOKED = "booked"
    ABANDONED = "abandoned"
    COMPLETED = "completed"
    ESCALATED = "escalated"


class SurfaceType(str, Enum):
    """Which chat surface a session belongs to. Phase 3 isolation depends on this."""
    PUBLIC = "public"
    CLIENT = "client"


class BudgetBand(str, Enum):
    UNDER_1K = "under_1k"
    ONE_TO_5K = "1k_to_5k"
    FIVE_TO_15K = "5k_to_15k"
    FIFTEEN_TO_50K = "15k_to_50k"
    OVER_50K = "over_50k"
    NOT_SURE = "not_sure"


class Timeline(str, Enum):
    IMMEDIATE = "immediate"          # Within 2 weeks
    SHORT_TERM = "short_term"        # 1-3 months
    MEDIUM_TERM = "medium_term"      # 3-6 months
    LONG_TERM = "long_term"          # 6+ months
    JUST_EXPLORING = "just_exploring"
    NOT_SURE = "not_sure"


class EmailStatus(str, Enum):
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    SENT = "sent"
    FAILED = "failed"
    REJECTED = "rejected"


# --- Database Models ---

class SessionRecord(BaseModel):
    """Represents a chat session in the database."""
    id: str = Field(default_factory=_uuid)
    started_at: str = Field(default_factory=_now)
    ended_at: Optional[str] = None
    source_page: Optional[str] = None
    consent_given: bool = False
    status: SessionStatus = SessionStatus.ACTIVE
    surface: SurfaceType = SurfaceType.PUBLIC
    client_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    total_messages: int = 0
    total_cost: float = 0.0
    summary: Optional[str] = None
    next_step: Optional[str] = None
    # Browser-scoped identity (not a login) that links this visitor's sessions together
    # so returning visitors get chat history and cross-session memory (not Phase 3 auth).
    visitor_id: Optional[str] = None
    # Short AI-generated label for the chat-list panel, e.g. "Website rebuild inquiry"
    title: Optional[str] = None
    # Coarse classification for the dashboard: "lead" | "query" | "other"
    category: str = "other"


class LeadRecord(BaseModel):
    """Represents a lead extracted from a conversation."""
    id: str = Field(default_factory=_uuid)
    session_id: str
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    company: Optional[str] = None
    business_type: Optional[str] = None
    pain_point: Optional[str] = None
    budget_band: Optional[str] = None
    timeline: Optional[str] = None
    decision_maker: Optional[bool] = None
    service_interest: Optional[str] = None
    qualification_score: float = 0.0
    transcript_summary: Optional[str] = None
    next_step: Optional[str] = None
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)


# Fields on `leads` that may be written by progressive capture. Anything the LLM
# extracts is checked against this set before it reaches a SQL statement.
LEAD_WRITABLE_FIELDS = frozenset({
    "name", "email", "phone", "company", "business_type", "pain_point",
    "budget_band", "timeline", "decision_maker", "service_interest",
    "qualification_score", "transcript_summary", "next_step",
})


class MessageRecord(BaseModel):
    """Represents a single message in a conversation."""
    id: str = Field(default_factory=_uuid)
    session_id: str
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: str = Field(default_factory=_now)
    intent: Optional[str] = None
    model_used: Optional[str] = None
    cost: float = 0.0


class ProjectBrief(BaseModel):
    """Structured discovery brief produced by the Phase 2 intake agent."""
    id: str = Field(default_factory=_uuid)
    session_id: str
    title: str = ""
    summary: str = ""
    goals: List[str] = []
    current_state: str = "Not disclosed"
    constraints: List[str] = []
    budget_band: Optional[str] = None
    timeline: Optional[str] = None
    success_criteria: List[str] = []
    recommended_services: List[str] = []
    bundle_rationale: str = ""
    open_questions: List[str] = []
    risk_flags: List[str] = []
    created_at: str = Field(default_factory=_now)

    def to_markdown(self) -> str:
        """Render the brief for email / dashboard display."""
        def bullets(items: List[str]) -> str:
            return "\n".join(f"- {i}" for i in items) if items else "- (none captured)"

        return f"""# {self.title or 'Project Brief'}

{self.summary}

## Goals
{bullets(self.goals)}

## Current State
{self.current_state or 'Not disclosed'}

## Constraints
{bullets(self.constraints)}

## Budget
{(self.budget_band or 'not_sure').replace('_', ' ')}

## Timeline
{(self.timeline or 'not_sure').replace('_', ' ')}

## Success Criteria
{bullets(self.success_criteria)}

## Recommended Services
{bullets(self.recommended_services)}

{self.bundle_rationale}

## Open Questions for the Team
{bullets(self.open_questions)}

## Risk Flags
{bullets(self.risk_flags)}
"""


class EmailRecord(BaseModel):
    """A queued or sent transactional email."""
    id: str = Field(default_factory=_uuid)
    session_id: Optional[str] = None
    to_email: str
    subject: str
    body: str
    status: EmailStatus = EmailStatus.PENDING_APPROVAL
    provider_message_id: Optional[str] = None
    error: Optional[str] = None
    created_at: str = Field(default_factory=_now)
    sent_at: Optional[str] = None


class ClientRecord(BaseModel):
    """An existing Abacus Digital client, for the Phase 3 authenticated surface."""
    id: str = Field(default_factory=_uuid)
    email: str
    name: Optional[str] = None
    company: Optional[str] = None
    account_manager_email: Optional[str] = None
    active: bool = True
    created_at: str = Field(default_factory=_now)


class ClientProject(BaseModel):
    """A project belonging to a client. Indexed into the access-controlled client KB."""
    id: str = Field(default_factory=_uuid)
    client_id: str
    name: str
    status: str = "active"
    service_line: Optional[str] = None
    started_at: Optional[str] = None
    target_date: Optional[str] = None
    progress_notes: str = ""
    deliverables: List[str] = []
    support_docs: List[str] = []
    updated_at: str = Field(default_factory=_now)


class MagicLinkToken(BaseModel):
    """Single-use login token emailed to a client."""
    token: str
    client_id: str
    expires_at: str
    used: bool = False
    created_at: str = Field(default_factory=_now)


class ClientSessionToken(BaseModel):
    """Bearer token issued after a magic link is redeemed."""
    token: str
    client_id: str
    expires_at: str
    created_at: str = Field(default_factory=_now)


# --- API Request/Response Models ---

class ChatRequest(BaseModel):
    """Incoming chat message from the widget."""
    session_id: Optional[str] = None
    visitor_id: Optional[str] = None
    message: str
    source_page: Optional[str] = None
    consent_given: bool = False


class ChatResponse(BaseModel):
    """Outgoing chat response to the widget."""
    session_id: str
    message: str
    intent: Optional[str] = None
    # A single citation link when the answer draws on a live-crawled page — the widget
    # renders this as one discreet "Learn more" link, not a row of source-name chips.
    source_link: Optional[Dict[str, str]] = None
    suggestions: List[str] = []
    show_booking: bool = False
    booking_url: Optional[str] = None
    qualification_score: Optional[float] = None
    show_consent: bool = False
    brief_ready: bool = False
    escalated: bool = False
    # Present once the chat has an AI-generated title, so the widget can update the
    # chat-list panel without a second round-trip.
    title: Optional[str] = None


class LeadSummary(BaseModel):
    """Summary of a lead for the internal dashboard."""
    session_id: str
    name: Optional[str] = None
    email: Optional[str] = None
    company: Optional[str] = None
    business_type: Optional[str] = None
    pain_point: Optional[str] = None
    service_interest: Optional[str] = None
    budget_band: Optional[str] = None
    timeline: Optional[str] = None
    qualification_score: float = 0.0
    transcript_summary: Optional[str] = None
    next_step: Optional[str] = None
    status: str
    category: str = "other"
    started_at: str
    message_count: int = 0
    has_brief: bool = False


class DataDeletionRequest(BaseModel):
    """Request to delete user data (GDPR compliance)."""
    email: Optional[str] = None
    session_id: Optional[str] = None
    visitor_id: Optional[str] = None


class ChatSummary(BaseModel):
    """One row in the visitor-facing chat list panel."""
    session_id: str
    title: Optional[str] = None
    snippet: Optional[str] = None
    status: str
    category: str = "other"
    started_at: str
    updated_at: Optional[str] = None
    message_count: int = 0


class ChatListResponse(BaseModel):
    chats: List[ChatSummary] = []


class ChatTranscriptResponse(BaseModel):
    session_id: str
    title: Optional[str] = None
    transcript: List[Dict[str, str]] = []


class DataDeletionResponse(BaseModel):
    """Response to data deletion request."""
    success: bool
    records_deleted: int
    message: str


class MagicLinkRequest(BaseModel):
    """Phase 3: request a login link."""
    email: EmailStr


class MagicLinkResponse(BaseModel):
    success: bool
    message: str
    # Only populated when email delivery is disabled (local dev), never in production
    debug_token: Optional[str] = None


class ClientVerifyRequest(BaseModel):
    token: str


class ClientVerifyResponse(BaseModel):
    success: bool
    session_token: Optional[str] = None
    client_name: Optional[str] = None
    client_company: Optional[str] = None
    expires_at: Optional[str] = None
    message: str = ""


class ClientChatRequest(BaseModel):
    """Phase 3: authenticated chat message."""
    session_id: Optional[str] = None
    message: str
    source_page: Optional[str] = None


class ReindexRequest(BaseModel):
    """Trigger a knowledge base re-index (scheduled job or CMS publish webhook)."""
    full: bool = False
    crawl_site: bool = True


class EmailApprovalRequest(BaseModel):
    email_id: str
    approve: bool
    edited_body: Optional[str] = None


# --- Conversation State ---

class ConversationState(BaseModel):
    """Tracks the full state of an ongoing conversation."""
    session_id: str
    messages: List[Dict[str, str]] = []
    current_intent: Intent = Intent.GREETING
    consent_given: bool = False
    qualification_data: Dict[str, Any] = {}
    qualification_score: float = 0.0
    is_qualified: bool = False
    booking_offered: bool = False
    service_interests: List[str] = []
    source_page: Optional[str] = None
    message_count: int = 0
    total_cost: float = 0.0
    consent_shown: bool = False

    # Phase 2: agentic intake
    in_intake: bool = False
    intake_data: Dict[str, Any] = {}
    brief_generated: bool = False

    # Phase 3: authenticated client surface
    surface: SurfaceType = SurfaceType.PUBLIC
    client_id: Optional[str] = None
    client_name: Optional[str] = None
    client_company: Optional[str] = None
    escalated: bool = False

    # Cross-session visitor identity and memory
    visitor_id: Optional[str] = None
    visitor_memory: str = ""

    # Soft email capture, asked once near the start of a session
    email_capture_asked: bool = False
    email_capture_pending: bool = False

    # AI-generated chat title (chat-list panel) and lead/query classification
    title: Optional[str] = None
    title_generated: bool = False
    category: str = "other"
