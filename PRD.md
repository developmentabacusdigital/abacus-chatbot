# PRD: Abacus Digital Website Chatbot

**Owner:** Abhijay
**Site:** https://www.abacusdigital.net/
**Status:** Draft v1
**Last updated:** August 11, 2026

---

## 1. Summary

An AI chatbot embedded on the Abacus Digital website (built on Framer) that qualifies leads, books discovery calls, answers service and pricing questions using the site's own content, performs agentic project intake, collects and stores visitor-submitted data, and supports existing clients through a separate authenticated flow. The bot uses OpenRouter to route across multiple LLMs, favoring cheaper models by default and escalating to stronger models only for reasoning-heavy steps. The full stack is built on free-tier infrastructure (hosting, database, vector store) to keep running costs near zero.

## 2. Problem Statement

Abacus Digital runs a marketing site across 9 service lines with an active blog, 50+ clients, and 200+ collaborative projects. Visitors currently have to self-navigate services, submit a generic contact form, and wait for manual follow-up. There's no automated qualification, no instant answers grounded in the actual site content, and no structured intake for project requests. This creates friction at the top of the funnel and manual overhead for the team triaging inbound leads.

## 3. Goals

- Increase qualified-lead conversion rate from site visitors
- Reduce time-to-first-response for inbound inquiries to near-zero
- Cut manual triage work by having the bot pre-qualify and structure leads before a human sees them
- Ground all answers in real site content to avoid hallucinated pricing or capability claims
- Keep LLM inference cost low by defaulting to cheap models and escalating selectively
- Capture and persist visitor-submitted data (contact info, qualification answers, intent) for every session, not just sessions that convert
- Run the entire system on free-tier infrastructure so hosting/DB/inference costs stay at or near $0 while validating the concept

## 4. Non-Goals (v1)

- Full autonomous deal-closing or contract generation
- Replacing the sales team; the bot hands off, it doesn't close
- Payment collection or invoicing within the chat
- Multi-language support (English only for v1)

## 5. Users / Personas

| Persona | Description | Primary need |
|---|---|---|
| Prospect (cold) | First-time visitor exploring services | Fast, accurate answers; low-friction next step |
| Prospect (warm) | Knows roughly what they want (e.g., "I need a website") | Structured intake, quote path, booking |
| Existing client | Has an active or past project with Abacus | Status updates, support, escalation to account manager |
| Abacus team (internal) | Sales/ops reviewing bot output | Clean, structured leads in CRM; no noise |

## 6. Scope by Phase

### Phase 1: Foundation
- RAG-grounded Q&A over service pages and blog content
- Lead qualification flow: business type, pain point, budget band, timeline
- Calendar booking for qualified leads
- CRM write-back: every session creates/updates a lead record with structured qualification data

### Phase 2: Agentic Intake
- Structured discovery agent for inbound project requests, producing a formatted brief (goals, current state, budget, timeline, constraints)
- Service-matching reasoning across the 9 service lines, including multi-service bundling recommendations
- Auto-drafted follow-up email summarizing the conversation, sent via email tool (human-approved initially)

### Phase 3: Existing Client Support
- Separate authenticated chat surface, gated behind login
- Access to a distinct knowledge base: project status, deliverables, support docs
- Escalation path to a human account manager
- Explicitly isolated from the anonymous prospect flow and its knowledge base to prevent data leakage between a prospect and client-only information

## 7. Functional Requirements

### 7.1 RAG Q&A
- Index all service pages (`/all-services/*`) and blog posts (`/blog/*`)
- Re-index on a schedule (e.g., daily) or via webhook on publish
- Answers cite which service page or post they're drawn from
- If no grounded answer exists, the bot says so and offers to connect with the team rather than guessing

### 7.2 Lead Qualification
- Conversational form, not a rigid multi-step wizard: business type, primary pain point, budget band, timeline, decision-maker status
- Qualification score computed from responses (e.g., budget + timeline + authority)
- Below-threshold leads get self-serve resources; above-threshold leads get routed to booking

### 7.3 Booking
- Integrates with the team's calendar tool (Calendly or equivalent)
- Only offered to qualified leads or on explicit request
- Confirms booking details back in-chat and via email

### 7.4 CRM Integration
- Every conversation produces a structured record: contact info (if given), qualification answers, service interest, transcript summary, qualification score
- v1 uses the app's own free-tier database (Section 9.1) as the system of record, with a simple internal dashboard or export (CSV/API) for the team to review leads, avoiding a paid CRM subscription
- Optional: mirror records into a free-tier CRM (Airtable free plan or HubSpot free tier) if the team wants a familiar UI, via a scheduled sync or webhook rather than a real-time paid integration
- Deduplicates by email/contact where possible

### 7.5 User Data Collection & Storage
- Every session persists to the database regardless of outcome, not just qualified/converted leads, so partial and abandoned conversations are still captured
- Data captured per session: name, email, phone (if given), company, business type, pain point, budget band, timeline, service interest, full transcript, qualification score, timestamp, source page
- A lightweight consent notice is shown at the start of the chat (e.g., "This chat may be recorded to follow up with you") before any personal data is requested
- Visitors can request data deletion; the backend supports a delete-by-email/session-id operation
- No sensitive data categories (payment details, government IDs) are ever requested or stored in chat
- Stored data is the source of truth for CRM write-back, not a duplicate system; CRM (Section 7.4) reads from or mirrors this store rather than maintaining a separate record

### 7.6 Agentic Intake
- Triggered when a visitor describes a concrete project need
- Multi-turn discovery: goals, current situation (e.g., existing site URL if applicable), constraints, budget, timeline
- Outputs a structured brief object, both stored in CRM and optionally emailed to the visitor for confirmation
- Reasons over the 9 service categories to recommend the right service(s)

### 7.7 Client Support (Phase 3)
- Requires authentication (magic link or existing client portal login)
- Retrieves project-specific data from a separate, access-controlled data source
- Never mixes prospect-facing RAG index with client-specific data
- Escalates to a human when the bot can't resolve a request

### 7.8 Guardrails
- No fabricated pricing; the bot gives ranges or frameworks only, and defers exact quotes to a human
- No commitments the business hasn't authorized (contracts, discounts, timelines as guarantees)
- Rate limiting and abuse detection on the public widget

## 8. Non-Functional Requirements

- Response latency: under 3 seconds for cached/simple queries, under 8 seconds for agentic multi-step flows
- Uptime: matches site uptime expectations (widget degrades gracefully to a contact form if backend is down)
- Data privacy: prospect data handled per standard data protection practice; client data access strictly scoped per authenticated user
- Cost tracking: per-conversation LLM cost logged for monitoring spend

## 9. Technical Architecture

### 9.1 Stack (free-tier first)
- **Backend:** FastAPI, deployed on a free-tier host (e.g., Render free web service, Railway free tier, or Fly.io free allowance). Note: most free tiers spin down on inactivity, causing a cold-start delay on the first message after idle time; this is an accepted tradeoff for v1
- **Orchestration:** LangGraph state machine (intent classification → RAG / qualification / booking / intake → CRM write)
- **Database:** A free-tier managed Postgres (e.g., Supabase or Neon free tier) for session/user data storage, or a free-tier MongoDB Atlas cluster if a document store fits better. Supabase is preferred if we also want built-in auth for Phase 3 client login
- **Vector store:** Self-hosted Chroma (runs in-process/on the same free host, no separate cost) or Qdrant Cloud free tier if a managed option is preferred
- **LLM routing:** OpenRouter, multiple models selected per task, prioritizing free/low-cost models on OpenRouter's roster wherever quality allows
- **Frontend widget:** Lightweight JS chat widget embedded on the Framer site via Framer's Custom Code (embed) component, pointing at the hosted backend's API. Framer itself cannot run backend logic, so the widget is a thin client only, all logic lives on the FastAPI backend
- **Integrations:** Cal.com (free tier/self-hosted) for booking, a free-tier CRM or the app's own database doubling as the CRM system of record (see 7.4), free-tier transactional email (e.g., Resend or Brevo free allowance) for follow-ups

### 9.2 Framer Integration Notes
- Framer supports embedding custom HTML/JS via its Custom Code component, which is how the chat widget gets onto the live site
- The widget itself should be a small, dependency-light script/iframe so it doesn't slow down Framer's page load or fight with Framer's own JS
- All API calls from the widget go to the externally hosted FastAPI backend (Framer cannot host the backend itself); CORS must be configured on the backend to allow requests from the Framer domain
- If Framer's plan doesn't allow arbitrary script embeds on all pages, confirm placement (site-wide vs. specific pages) before build

### 9.3 Model Routing Strategy (via OpenRouter)

| Task | Model tier | Rationale |
|---|---|---|
| Intent classification | Cheapest available (e.g., small open-weight model) | High volume, low complexity |
| RAG answer generation | Cheap/mid-tier model | Retrieval does the heavy lifting; generation just needs to synthesize cleanly |
| Lead qualification dialogue | Cheap/mid-tier model | Structured, mostly slot-filling |
| Agentic intake reasoning / service matching | Stronger model | Needs multi-step reasoning and accurate service mapping |
| Brief generation (final structured output) | Stronger model | Output quality directly affects what the sales team sees |
| Client support (Phase 3) | Mid-tier, escalate to strong model on ambiguous/sensitive queries | Balance cost with accuracy on account-specific answers |

Fallback logic: if a cheap model's output fails a confidence/format check, retry once on a stronger model before returning to the user.

### 9.4 Data Flow (Phase 1)
1. Visitor opens widget on the Framer site → consent notice shown → session starts, session record created in the database
2. Intent classifier routes message (question / qualification / booking / general)
3. RAG node retrieves relevant chunks from vector store if it's a question
4. Qualification node runs conversational slot-filling if applicable, writing captured fields to the session record as they're given (not just at the end, so partial data survives an abandoned chat)
5. On qualification threshold met → booking node offers calendar slots
6. Every turn appends to session state in the database; on session end (or key milestones) → CRM sync/export step fires if enabled

### 9.5 Knowledge Base Separation
- **Public index:** service pages, blog posts, testimonials, general FAQs
- **Client index (Phase 3 only):** project status, deliverables, support docs, access-controlled per authenticated client
- These are never queried in the same retrieval call

## 10. Success Metrics

- Lead-to-qualified-lead conversion rate (target: define baseline in first 30 days, then improve)
- Bot-assisted bookings per week
- CRM record completeness (percent of sessions producing a usable structured lead)
- Cost per conversation (track against OpenRouter spend)
- Deflection rate for client support queries (Phase 3) that don't need human escalation
- Answer grounding accuracy (percent of RAG answers correctly citing source content, spot-checked)

## 11. Risks & Open Questions

- **CRM choice:** which system will the team standardize on for lead write-back?
- **Calendar tool:** confirm which booking system is already in use
- **Client auth:** does an existing client portal/login system exist, or does this need to be built from scratch for Phase 3?
- **Content freshness:** need a reliable trigger (webhook or scheduled job) to re-index when blog/service pages change
- **Escalation path:** define exactly when and how the bot hands off to a human (business hours vs. always-on)
- **Rate limiting/abuse:** define thresholds before Phase 1 launch to control cost exposure on the public widget
- **Free-tier limitations:** cold starts on free hosting can cause a multi-second delay on the first message after idle; free DB tiers cap storage and connection count, which needs monitoring as lead volume grows
- **Framer embed constraints:** confirm the Framer plan supports the custom code embed needed, and that CORS/domain restrictions on the backend are set correctly before launch
- **Free-tier scaling ceiling:** if lead volume outgrows free-tier limits (DB storage, host uptime/requests), plan a migration path to paid tiers of the same providers rather than a full re-architecture

## 12. Rollout Plan

1. Build Phase 1 against a staging index of the current site content
2. Internal dogfooding with the Abacus team before public launch
3. Soft launch on a low-traffic page, monitor cost and qualification accuracy
4. Full rollout to homepage and service pages
5. Phase 2 (agentic intake) once Phase 1 metrics are stable
6. Phase 3 (client support) scoped separately, pending auth system decision