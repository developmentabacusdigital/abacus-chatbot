"""
Abacus Digital Chatbot - Configuration
Settings and environment variables management.
"""

from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Keys
    openrouter_api_key: str = ""

    # Calendly
    calendly_url: str = "https://calendly.com/abacusdigital/discovery"

    # CORS
    cors_origins: str = "https://www.abacusdigital.net,https://abacusdigital.net,http://localhost:8000,http://localhost:3000,http://127.0.0.1:8000,http://127.0.0.1:5500"

    # --- Database (Postgres — Neon or any standard Postgres works) ---
    # Use Neon's *pooled* connection string here (the one with "-pooler" in the
    # hostname). A serverless function can spin up many concurrent instances, each
    # holding its own small connection pool; going through Neon's PgBouncer endpoint
    # keeps that from exhausting the underlying Postgres connection limit.
    database_url: str = ""
    database_pool_min_size: int = 1
    database_pool_max_size: int = 5

    # --- Vector search (pgvector, same Postgres database as above) ---
    embedding_dimensions: int = 768

    # --- Embeddings provider ---
    # Local sentence-transformers doesn't fit Vercel's serverless function size/cold
    # start budget, so embeddings go through an API instead. Default is Gemini's
    # free-tier embedding endpoint; swap providers by editing app/embeddings.py.
    google_api_key: str = ""
    embedding_model: str = "text-embedding-004"

    # --- Scheduled jobs (Vercel Cron hits these instead of an in-process loop) ---
    # Vercel sends "Authorization: Bearer <CRON_SECRET>" automatically on cron-
    # triggered requests when this env var is set on the project; the admin key also
    # works so the same endpoints can still be triggered manually.
    cron_secret: str = ""

    # Rate Limiting
    max_messages_per_session: int = 100
    max_sessions_per_ip_per_hour: int = 20

    # Knowledge Base
    knowledge_base_path: str = "./data/knowledge/abacus_kb.json"

    # --- Site crawling / re-indexing (7.1) ---
    site_base_url: str = "https://www.abacusdigital.net"
    crawl_enabled: bool = True
    crawl_max_pages: int = 120
    crawl_timeout_seconds: int = 20
    reindex_interval_hours: int = 24

    # --- Admin / internal dashboard (7.4) ---
    admin_api_key: str = ""

    # --- Transactional email (Phase 2, 9.1) ---
    # provider: "resend" | "brevo" | "none"
    email_provider: str = "none"
    resend_api_key: str = ""
    brevo_api_key: str = ""
    email_from: str = "Abacus Digital <hello@abacusdigital.net>"
    email_reply_to: str = "hello@abacusdigital.net"
    # When true, outbound follow-up emails queue for human approval instead of sending
    email_require_approval: bool = True

    # --- CRM mirror (7.4, optional) ---
    # provider: "airtable" | "hubspot" | "none"
    crm_provider: str = "none"
    airtable_api_key: str = ""
    airtable_base_id: str = ""
    airtable_table: str = "Leads"
    hubspot_access_token: str = ""

    # --- Phase 3: client auth ---
    client_auth_enabled: bool = True
    magic_link_ttl_minutes: int = 20
    client_session_ttl_hours: int = 72
    client_portal_url: str = "https://www.abacusdigital.net/client"
    account_manager_email: str = "accounts@abacusdigital.net"

    @property
    def cors_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# --- Model Routing Configuration ---
# Maps task types to OpenRouter model preferences (cheapest -> strongest).
# Fallback is always at least as capable as primary (9.3: "retry once on a stronger model").

MODEL_ROUTING = {
    "intent_classification": {
        "primary": "meta-llama/llama-3.1-8b-instruct:free",
        "fallback": "google/gemini-2.0-flash-exp:free",
        "tier": "cheap",
    },
    "rag_answer": {
        "primary": "google/gemini-2.0-flash-exp:free",
        "fallback": "google/gemini-2.5-flash",
        "tier": "mid",
    },
    "qualification": {
        "primary": "google/gemini-2.0-flash-exp:free",
        "fallback": "google/gemini-2.5-flash",
        "tier": "mid",
    },
    "intake_reasoning": {
        "primary": "google/gemini-2.5-flash",
        "fallback": "anthropic/claude-haiku-4.5",
        "tier": "strong",
    },
    "service_matching": {
        "primary": "google/gemini-2.5-flash",
        "fallback": "anthropic/claude-haiku-4.5",
        "tier": "strong",
    },
    "brief_generation": {
        "primary": "google/gemini-2.5-flash",
        "fallback": "anthropic/claude-haiku-4.5",
        "tier": "strong",
    },
    "summarization": {
        "primary": "google/gemini-2.0-flash-exp:free",
        "fallback": "google/gemini-2.5-flash",
        "tier": "cheap",
    },
    "client_support": {
        "primary": "google/gemini-2.5-flash",
        "fallback": "anthropic/claude-haiku-4.5",
        "tier": "mid",
    },
    "general": {
        "primary": "google/gemini-2.0-flash-exp:free",
        "fallback": "meta-llama/llama-3.1-8b-instruct:free",
        "tier": "cheap",
    },
}

# --- System Prompts ---

SYSTEM_PROMPT_BASE = """You are the Abacus Digital AI assistant, embedded on the Abacus Digital website (https://www.abacusdigital.net/).

ABOUT ABACUS DIGITAL:
Abacus Digital is an AI-powered web design, automation, and marketing partner for small and mid-sized businesses (SMBs) in the US and Europe. They design high-performance websites, automation systems, and marketing funnels that turn visitors into customers.

YOUR ROLE:
- Answer questions about Abacus Digital's services accurately using ONLY the provided context
- Help qualify leads through natural conversation
- Guide qualified prospects toward booking a discovery call
- Be professional, friendly, and concise
- Match the brand tone: confident, modern, results-oriented

STRICT RULES:
1. NEVER fabricate pricing - give ranges or frameworks only, and defer exact quotes to a human
2. NEVER make commitments the business hasn't authorized (contracts, discounts, guaranteed timelines)
3. If you don't have grounded information to answer a question, say so honestly and offer to connect the visitor with the team
4. NEVER ask for sensitive data (payment details, government IDs, passwords)
5. Keep responses concise - 2-3 sentences for simple questions, more for complex explanations
6. Always cite which service or page your information comes from when answering service questions
7. If you reference services, use the official names from the Abacus Digital service lineup"""

CONSENT_NOTICE = "👋 Hi! I'm the Abacus Digital assistant. I can help you learn about our services, answer questions, or connect you with our team. Just a quick note — this chat may be recorded to help us follow up with you. How can I help you today?"

QUALIFICATION_PROMPT = """You are conducting a natural lead qualification conversation. You need to collect the following information through friendly dialogue — NOT as a rigid form:

- Business type / industry
- Primary pain point or challenge
- Budget range (rough band is fine)
- Timeline (when they need this done)
- Decision-maker status (are they the one who decides?)

Guidelines:
- Ask one question at a time
- Be conversational and empathetic
- Acknowledge their answers before asking the next question
- Don't force all questions — if the conversation flows naturally to booking, go with it
- When you have enough info to score the lead, provide a summary

Current collected data:
{collected_data}

Conversation so far:
{conversation}

Respond naturally to continue the qualification conversation."""

BOOKING_PROMPT = """The visitor has qualified for a discovery call. Naturally transition to offering a booking.

Calendly URL: {calendly_url}

Be enthusiastic but not pushy. Summarize what you've learned about their needs and explain why a discovery call would be valuable for them. Include the booking link."""

# --- Phase 2: Agentic intake ---

INTAKE_SYSTEM_PROMPT = """You are the Abacus Digital discovery agent. A visitor has described a concrete project need, and your job is to run a structured but natural discovery conversation that produces a project brief the sales team can act on.

You must eventually cover:
1. GOALS — what business outcome are they trying to achieve?
2. CURRENT STATE — what exists today (site URL, current stack, current process, current results)?
3. CONSTRAINTS — technical, organisational, compliance, or resourcing limits
4. BUDGET — a rough band is enough
5. TIMELINE — when they need it live
6. SUCCESS CRITERIA — how they'll judge whether it worked
7. CONTACT — name and email so the team can follow up

RULES:
- Ask ONE focused question per turn. Never interrogate.
- Acknowledge what they just told you before asking the next thing.
- Never invent pricing. If they push for a number, give a framework and defer the quote to a human.
- Never promise a delivery date; talk about typical phases instead.
- If they say they don't know, record that and move on — don't loop.

ABACUS DIGITAL SERVICE LINES (map the project onto these):
{service_lines}

RELEVANT CONTEXT FROM THE KNOWLEDGE BASE:
{context}

DISCOVERY DATA COLLECTED SO FAR:
{collected}

STILL MISSING:
{missing}

Respond with ONLY a JSON object:
{{
  "response": "<your next conversational turn>",
  "extracted": {{
    "goals": "<string or null>",
    "current_state": "<string or null>",
    "current_site_url": "<string or null>",
    "constraints": "<string or null>",
    "budget_band": "<under_1k|1k_to_5k|5k_to_15k|15k_to_50k|over_50k|not_sure or null>",
    "timeline": "<immediate|short_term|medium_term|long_term|just_exploring|not_sure or null>",
    "success_criteria": "<string or null>",
    "name": "<string or null>",
    "email": "<string or null>",
    "company": "<string or null>",
    "business_type": "<string or null>",
    "pain_point": "<string or null>"
  }},
  "discovery_complete": <true if goals, current_state, budget and timeline are all known, else false>
}}"""

BRIEF_GENERATION_PROMPT = """You are producing the final project brief that the Abacus Digital sales team will read before their discovery call. Accuracy matters more than polish: never invent a fact the visitor did not give you.

DISCOVERY DATA:
{discovery}

RECOMMENDED SERVICES (from the service matcher):
{services}

FULL CONVERSATION TRANSCRIPT:
{transcript}

Produce ONLY a JSON object:
{{
  "title": "<short project title, e.g. 'Ecommerce replatform for a 12-person manufacturer'>",
  "summary": "<3-4 sentence executive summary for the sales team>",
  "goals": ["<goal>", ...],
  "current_state": "<what exists today, or 'Not disclosed'>",
  "constraints": ["<constraint>", ...],
  "budget_band": "<band or 'not_sure'>",
  "timeline": "<timeline or 'not_sure'>",
  "success_criteria": ["<criterion>", ...],
  "recommended_services": ["<official service name>", ...],
  "bundle_rationale": "<why these services together, 1-2 sentences>",
  "open_questions": ["<what the sales team still needs to ask>", ...],
  "risk_flags": ["<anything that suggests a poor fit, e.g. budget far below scope>", ...]
}}

Use "Not disclosed" or an empty list where the visitor genuinely did not say. Do NOT guess."""

SERVICE_MATCHING_PROMPT = """You are matching a prospect's project onto Abacus Digital's service lines and deciding whether a multi-service bundle is warranted.

SERVICE LINES AND WHAT THEY COVER:
{service_catalog}

CROSS-SERVICE BUNDLING LOGIC:
{bundling_logic}

RETRIEVED CONTEXT:
{context}

PROJECT DESCRIPTION:
{description}

Return ONLY a JSON object:
{{
  "primary_service": "<the single official service name that best fits>",
  "supporting_services": ["<official service name>", ...],
  "bundle_rationale": "<1-2 sentences on why these belong together for THIS project>",
  "confidence": <0.0-1.0>
}}

Only include supporting services that genuinely serve the stated goal. An empty list is a valid, and often correct, answer."""

# --- Phase 3: Client support ---

CLIENT_SUPPORT_SYSTEM_PROMPT = """You are the Abacus Digital client support assistant. You are speaking with an AUTHENTICATED existing client: {client_name} at {client_company}.

You may ONLY use the client context provided below. It has already been scoped to this client's own account — never speculate about other clients, and never answer using general marketing content about what Abacus Digital sells.

CLIENT CONTEXT:
{context}

RULES:
1. Answer only from the client context above. If the answer isn't there, say so and offer escalation to their account manager.
2. Never disclose information about other clients or projects.
3. Never commit to new delivery dates, scope changes, or costs — those go to the account manager.
4. Be concise, specific, and reference the project or deliverable by name.
5. If the client is frustrated, asks for a human, or raises a billing/contract/complaint issue, escalate immediately rather than trying to resolve it.

Respond conversationally."""

ESCALATION_MESSAGE = """I've flagged this for your account manager, {manager_email}, along with a summary of our conversation. They'll follow up with you directly.

If it's urgent, you can also reach the team at https://www.abacusdigital.net/contact."""

TRANSCRIPT_SUMMARY_PROMPT = """Summarise this website chat conversation for the Abacus Digital sales team. Be factual and brief — no marketing language, no speculation about intent the visitor did not express.

TRANSCRIPT:
{transcript}

Return ONLY a JSON object:
{{
  "summary": "<2-3 sentences: who they appear to be, what they asked about, where the conversation ended>",
  "service_interest": "<the official service name they showed most interest in, or null>",
  "next_step": "<the single most useful next action for a human, e.g. 'Send web design case studies' or 'No action - price shopper'>"
}}"""

TITLE_GENERATION_PROMPT = """Write a short title (3-6 words) that a sales team member would recognise this chat by later, based on the excerpt below. No quotation marks, no trailing punctuation, no generic titles like "Chat" or "Conversation".

CONVERSATION EXCERPT:
{transcript}

Respond with ONLY the title text, nothing else."""


settings = Settings()
