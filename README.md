# Abacus Digital Chatbot

AI chatbot for the [Abacus Digital](https://www.abacusdigital.net/) website.

The chatbot answers service questions grounded in live website content, qualifies leads, runs agentic project intake, books discovery calls, and supports existing clients through a separate authenticated surface.

Built against the `PRD.md` across all three phases.

Runs primarily on free tiers:

* **Neon** for PostgreSQL + `pgvector`
* **Vercel** for hosting
* **Google Gemini** free tier for embeddings
* **OpenRouter** free model roster for chat where quality allows

---

## What's Implemented

### Phase 1 — Foundation

* RAG Q&A grounded in the company documentation plus a live crawl of `/all-services/*` and `/blog/*`
* Source citations in generated answers
* Honest `"I don't know"` response when retrieval does not provide sufficient context
* Daily scheduled re-indexing
* `POST /api/admin/reindex` webhook for publish events
* Conversational lead qualification with weighted scoring:

  * Budget: 30%
  * Timeline: 25%
  * Authority: 25%
  * Fit: 20%
* Booking hand-off to Calendly for qualified leads
* Booking confirmation in chat and by email
* Persistent sessions, including abandoned sessions
* Progressive lead-field capture
* Transcript summaries
* Suggested next steps for the sales team

### Phase 2 — Agentic Intake

* Multi-turn discovery agent covering:

  * Goals
  * Current state
  * Constraints
  * Budget
  * Timeline
  * Success criteria
* Service matching across all nine capability areas
* Multi-service bundling with an explicit rationale
* Structured `ProjectBrief` stored as the system of record
* Risk flags and open questions for the sales team
* Auto-drafted follow-up emails
* Human approval required before follow-up emails are sent

### Phase 3 — Client Support

* Magic-link authentication

  * Single-use links
  * 20-minute expiration
  * Opaque server-side session tokens
* Separate `pgvector` table for client project data
* Hard, indexed `client_id` filtering rather than relying only on metadata
* Escalation to a human account manager for:

  * Commercial questions
  * Contractual questions
  * Unresolvable queries
* Public and client knowledge indexes are never queried in the same request

### Cross-Cutting Features

* Internal dashboard at `/dashboard/`
* Lead management
* Project briefs
* Email approval queue
* Escalation management
* Metrics
* CSV export
* Optional CRM mirror:

  * Airtable
  * HubSpot
* The application's own database remains the system of record
* Guardrails:

  * No fabricated pricing
  * Commitment-language disclaimers
  * Sensitive-data rejection
  * Per-session rate limits
  * Per-IP rate limits
* Per-conversation OpenRouter cost tracking

---

## Quick Start

You need three services before the application can operate fully:

1. PostgreSQL with `pgvector`
2. OpenRouter API key
3. Google Gemini API key for embeddings

All three have usable free tiers.

### 1. Database — Neon

Create a free project on [Neon](https://neon.tech/).

Copy the **pooled** connection string from the Neon dashboard.

The hostname should contain `-pooler`.

### 2. Chat — OpenRouter

Create an API key through [OpenRouter](https://openrouter.ai/keys).

### 3. Embeddings — Google AI Studio

Create a free API key through [Google AI Studio](https://aistudio.google.com/apikey).

Without the Gemini key, the application still boots and can answer greetings and general questions. However, it cannot ground answers in the knowledge base until embeddings are available.

### Local Setup

```bash
cd backend

cp .env.example .env

# Configure:
# DATABASE_URL
# OPENROUTER_API_KEY
# GOOGLE_API_KEY
# ADMIN_API_KEY

pip install -r requirements.txt

uvicorn app.main:app --reload
```

Then open:

* Public widget: `http://localhost:8000/widget/`
* Team dashboard: `http://localhost:8000/dashboard/`

On first boot, the application:

1. Creates the required PostgreSQL tables
2. Creates the `pgvector` extension
3. Indexes the static knowledge base
4. Starts crawling the live website approximately 30 seconds later

---

## Client Portal

The client portal is part of Phase 3.

Create a test client:

```bash
cd backend

python -m app.seed_client \
  --email you@example.com \
  --name "You" \
  --company "Acme Ltd"
```

Then open:

```text
http://localhost:8000/widget/client.html
```

Sign in using the seeded email address.

If no email provider is configured, the API returns the magic-link token directly so the flow can be tested locally.

Once `EMAIL_PROVIDER` is configured, the token is no longer returned directly.

---

## Tests

Run the complete test suite:

```bash
cd backend

python -m pytest
```

The project currently contains **129 tests**.

Tests require no network access or API keys.

The test suite uses a real, throwaway embedded PostgreSQL instance through the `pgserver` package instead of mocks.

This is intentional because the application relies heavily on:

* SQL upserts
* PostgreSQL transactions
* `pgvector` queries
* JSONB filters
* Database constraints

LLM calls use a fake router, while embeddings use a deterministic fake implementation so relevance ranking can still be tested.

---

# Architecture

```text
                    ┌────────────────────────────────────────┐
                    │             Framer Website              │
                    │                                        │
                    │  ┌──────────────┐   ┌───────────────┐  │
                    │  │ Public Widget│   │ Client Widget │  │
                    │  │ data-mode=   │   │ data-mode=    │  │
                    │  │ "public"     │   │ "client"      │  │
                    │  └──────┬───────┘   └───────┬───────┘  │
                    └─────────┼───────────────────┼──────────┘
                              │ /api/chat         │ /api/client/*
                              │                   │
                    ┌─────────▼───────────────────▼──────────┐
                    │       FastAPI on Vercel (Serverless)    │
                    │                                        │
                    │  ┌─────────────────────────────────┐   │
                    │  │       Chat Orchestrator          │   │
                    │  │                                 │   │
                    │  │ intent → route → persist → CRM  │   │
                    │  └──┬────┬────┬────┬────┬─────────┘   │
                    │     │    │    │    │    │             │
                    │    RAG  Qual Intake Book Client        │
                    │     │    │    │         Support        │
                    │     │    │    │           │            │
                    │  ┌──▼────▼────▼───────────▼─────────┐  │
                    │  │       Neon PostgreSQL + pgvector │  │
                    │  │                                  │  │
                    │  │ sessions/leads/briefs/emails/... │  │
                    │  │                                  │  │
                    │  │ public_documents                  │  │
                    │  │ client_documents                  │  │
                    │  │                                  │  │
                    │  │   Never queried together         │  │
                    │  └──────────────────────────────────┘  │
                    │                                        │
                    │ Gemini (embeddings) · OpenRouter (chat)│
                    │ Vercel Cron → /api/cron/*             │
                    └────────────────────────────────────────┘
```

---

## Model Routing

| Task                                       | Primary Model                | Fallback                    |
| ------------------------------------------ | ---------------------------- | --------------------------- |
| Intent classification                      | `llama-3.1-8b-instruct:free` | `gemini-2.0-flash-exp:free` |
| RAG answers, qualification                 | `gemini-2.0-flash-exp:free`  | `gemini-2.5-flash`          |
| Intake reasoning, service matching, briefs | `gemini-2.5-flash`           | `claude-haiku-4.5`          |
| Client support                             | `gemini-2.5-flash`           | `claude-haiku-4.5`          |
| Embeddings                                 | `text-embedding-004`         | —                           |

Structured LLM calls that return malformed JSON are retried once using the stronger fallback model.

If the retry also fails, the application uses a deterministic fallback where available.

A missing or failing embedding provider follows the same degradation strategy:

```text
Embedding failure
       ↓
No crash
       ↓
No unsupported answer
       ↓
Honest "I don't know"
```

---

# API

## Public API

| Method   | Path                     | Description                                              |
| -------- | ------------------------ | -------------------------------------------------------- |
| `POST`   | `/api/chat`              | Send a chat message                                      |
| `POST`   | `/api/sessions/{id}/end` | Close a session, summarize it, and sync with CRM         |
| `GET`    | `/api/chats`             | Get visitor's chat list                                  |
| `GET`    | `/api/chats/{id}`        | Get visitor's transcript                                 |
| `DELETE` | `/api/data`              | Delete data by email, session ID, or visitor ID          |
| `GET`    | `/health`                | Health check used by the widget for graceful degradation |

---

## Client Portal API

All client portal requests require a bearer token where applicable.

| Method | Path                 | Description                             |
| ------ | -------------------- | --------------------------------------- |
| `POST` | `/api/client/login`  | Request a magic link                    |
| `POST` | `/api/client/verify` | Redeem a magic link for a session token |
| `POST` | `/api/client/chat`   | Authenticated client support chat       |
| `GET`  | `/api/client/me`     | Get client profile and projects         |
| `POST` | `/api/client/logout` | Revoke the current session token        |

---

## Admin API

Admin endpoints require the `X-Admin-Key` header.

| Method | Path                               | Description                        |
| ------ | ---------------------------------- | ---------------------------------- |
| `GET`  | `/api/admin/leads`                 | Get lead list                      |
| `GET`  | `/api/admin/leads.csv`             | Export leads as CSV                |
| `GET`  | `/api/admin/metrics`               | Get PRD Section 10 metrics         |
| `GET`  | `/api/admin/sessions/{id}`         | Get transcript, lead, and brief    |
| `GET`  | `/api/admin/briefs`                | Get intake briefs                  |
| `GET`  | `/api/admin/emails`                | Get email approval queue           |
| `POST` | `/api/admin/emails/approve`        | Approve a queued email             |
| `GET`  | `/api/admin/escalations`           | Get open human hand-offs           |
| `POST` | `/api/admin/reindex`               | Re-index the knowledge base        |
| `POST` | `/api/admin/crm-sync`              | Mirror leads to the configured CRM |
| `POST` | `/api/admin/clients`               | Create/manage client portal users  |
| `POST` | `/api/admin/clients/{id}/projects` | Add project data for a client      |

If `ADMIN_API_KEY` is not configured, all admin routes return `503`.

This prevents the lead database from being exposed by default.

---

## Scheduled Jobs

Scheduled endpoints require:

```http
Authorization: Bearer <CRON_SECRET>
```

The admin API key can also be used.

| Method | Path                 | Description                                          |
| ------ | -------------------- | ---------------------------------------------------- |
| `GET`  | `/api/cron/reindex`  | Daily public knowledge-base re-index                 |
| `GET`  | `/api/cron/crm-sync` | Mirror leads to the configured CRM                   |
| `GET`  | `/api/cron/sweep`    | Finalize abandoned sessions and purge expired tokens |

Vercel Cron calls these endpoints according to the schedule defined in `vercel.json`.

Once `CRON_SECRET` is configured as a project environment variable, Vercel automatically sends the required bearer token.

On always-on hosts such as Render or Docker-based deployments, these endpoints are redundant.

The same scheduled work also runs through the in-process background loop and automatically disables itself on Vercel when the `VERCEL` environment variable is detected.

---

# Framer Embed

## Public Widget

Add the following script to public Framer pages:

```html
<script
  src="https://YOUR-BACKEND-URL/widget/abacus-chat-widget.js"
  data-api-url="https://YOUR-BACKEND-URL"
  data-calendly-url="https://calendly.com/abacusdigital/discovery">
</script>
```

## Client Portal Widget

Use the client mode only on the dedicated client portal page:

```html
<script
  src="https://YOUR-BACKEND-URL/widget/abacus-chat-widget.js"
  data-api-url="https://YOUR-BACKEND-URL"
  data-mode="client">
</script>
```

Configure the following environment variables:

```env
CORS_ORIGINS=https://your-framer-domain.com
CLIENT_PORTAL_URL=https://your-framer-domain.com/client
```

`CLIENT_PORTAL_URL` determines where magic links redirect users.

---

# Deployment

## Vercel + Neon

This is the recommended deployment configuration.

### 1. Create the Neon Database

Create a Neon project and use the pooled connection string.

The application automatically runs:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

The default Neon role should have permission to create the extension.

### 2. Deploy to Vercel

Import the repository into Vercel.

The repository already contains:

* `vercel.json`
* `api/index.py`

These handle the serverless deployment.

### Environment Variables

| Variable                          | Required    | Description                                        |
| --------------------------------- | ----------- | -------------------------------------------------- |
| `DATABASE_URL`                    | Yes         | Neon pooled PostgreSQL connection string           |
| `OPENROUTER_API_KEY`              | Yes         | Chat generation                                    |
| `GOOGLE_API_KEY`                  | Yes         | Gemini embeddings                                  |
| `ADMIN_API_KEY`                   | Yes         | Protects admin endpoints                           |
| `CRON_SECRET`                     | Recommended | Authenticates Vercel Cron requests                 |
| Everything else in `.env.example` | Optional    | Email, CRM, Calendly, CORS, and other integrations |

### Static Files

Vercel serves:

```text
/widget/
/dashboard/
```

as static assets.

Everything else is routed to:

```text
api/index.py
```

The FastAPI application also mounts the same directories as a fallback.

After deployment, verify that:

```text
https://YOUR-DOMAIN/widget/
```

loads successfully.

---

## Vercel Constraints

These are platform constraints rather than application bugs.

### Function Execution Limits

The Vercel Hobby plan limits function execution to approximately 10 seconds.

The Phase 2 brief-generation flow chains several LLM calls and may approach this limit.

The repository already requests:

```json
{
  "maxDuration": 60
}
```

in `vercel.json`.

This takes effect only on plans that support the requested duration, such as Vercel Pro.

### WebSockets

Vercel serverless functions do not provide traditional WebSocket support.

The previous `/ws/chat` endpoint has been removed.

The widget only used:

```text
POST /api/chat
```

over REST, so no current functionality was lost.

### Filesystem

Vercel's filesystem is read-only except for ephemeral `/tmp` storage.

The application does not depend on persistent local files.

Persistent data is stored in Neon.

---

# Alternative Deployment: Docker + Render

The same application can run on Render or another container host.

This uses the same Neon database while replacing Vercel's serverless runtime with an always-on container while warm.

This can be useful if:

* Vercel execution limits become restrictive
* You do not want to upgrade to Vercel Pro
* You prefer a persistent application process

Build the image from the repository root:

```bash
docker build -t abacus-bot .
```

Run locally:

```bash
docker run \
  -p 8000:8000 \
  --env-file backend/.env \
  abacus-bot
```

The included `render.yaml` can be used for Render deployment.

### Render Free Tier

Free Render instances spin down after approximately 15 minutes of inactivity.

The first request after inactivity may therefore experience a cold start.

The widget displays a typing indicator while waiting and falls back to the contact form if the backend cannot be reached.

Unlike Vercel, in-process background loops run natively on the container:

* Re-indexing
* CRM synchronization
* Session sweeping

Therefore, `/api/cron/*` endpoints are not required for this deployment model.

---

# Project Structure

```text
.
├── api/
│   └── index.py
│
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── models.py
│   │   ├── database.py
│   │   ├── vector_store.py
│   │   ├── embeddings.py
│   │   ├── knowledge_base.py
│   │   ├── site_crawler.py
│   │   ├── indexer.py
│   │   ├── rag_engine.py
│   │   ├── llm_router.py
│   │   ├── intent_classifier.py
│   │   ├── lead_qualifier.py
│   │   ├── intake_agent.py
│   │   ├── booking_handler.py
│   │   ├── client_support.py
│   │   ├── auth.py
│   │   ├── email_service.py
│   │   ├── crm_sync.py
│   │   ├── guardrails.py
│   │   ├── chat_orchestrator.py
│   │   └── seed_client.py
│   │
│   ├── dashboard/
│   │   └── index.html
│   │
│   └── tests/
│
├── widget/
│   └── ...
│
├── vercel.json
├── requirements.txt
├── Dockerfile
├── render.yaml
├── Procfile
├── PRD.md
└── README.md
```

---

# Knowledge & Retrieval Architecture

<<<<<<< Updated upstream
- **Live end-to-end generation against Neon is exercised via a local embedded Postgres,
  not Neon itself.** The SQL, pgvector queries, and full request lifecycle are all
  verified against a real Postgres server (not mocks) — but this session never had a
  live Neon connection string to test against. Neon runs a standard, current Postgres
  with pgvector, so this should be a non-event, but it's the one part of the migration
  that's "should work" rather than "verified working."
- **No live Gemini embedding key was available to test against either.** The embedding
  client, its retry/batching logic, and the exact request shape are implemented per
  Gemini's documented API, but never called for real — verify indexing produces
  reasonable answers once a key is set, and adjust `app/embeddings.py` if the API's
  exact response shape has since changed.
- **Email and CRM mirroring are off by default.** Both need credentials; until they're
  set, follow-ups queue in the dashboard and nothing is mirrored.
=======
The application uses two independent knowledge namespaces.

```text
                         User Request
                              │
                              ▼
                       Intent / Auth Check
                              │
                 ┌────────────┴────────────┐
                 │                         │
            Public User               Client User
                 │                         │
                 ▼                         ▼
       public_documents            client_documents
                 │                         │
                 │                  WHERE client_id = ?
                 │                         │
                 └────────────┬────────────┘
                              │
                              ▼
                        Context Retrieval
                              │
                              ▼
                         LLM Generation
```

The public and client indexes are deliberately separated.

Client project information is never retrieved merely because a metadata field happens to contain a matching client ID.

Instead, the database query applies an explicit indexed:

```sql
client_id
```

filter.

The public and client indexes are also never queried in the same retrieval call.

---

# Lead Qualification

Leads are scored using the following weighted model:

| Factor    |   Weight |
| --------- | -------: |
| Budget    |      30% |
| Timeline  |      25% |
| Authority |      25% |
| Fit       |      20% |
| **Total** | **100%** |

The qualification system progressively captures information during conversation rather than requiring a static form.

Qualified leads can be handed off to Calendly for discovery-call booking.

---

# Agentic Project Intake

The intake agent collects:

* Project goals
* Current state
* Constraints
* Budget
* Timeline
* Success criteria

It then:

1. Matches the project against Abacus Digital's service capabilities.
2. Determines whether multiple services should be bundled.
3. Provides an explicit rationale for the recommendation.
4. Generates a structured `ProjectBrief`.
5. Identifies risk flags.
6. Identifies unresolved questions.
7. Drafts a follow-up email.

Follow-up emails are **not automatically sent**.

They enter the dashboard approval queue for human review.

---

# Client Support & Authentication

The client portal uses passwordless magic-link authentication.

Authentication properties:

* Single-use links
* 20-minute expiration
* Opaque server-side session tokens
* Bearer authentication for protected client endpoints
* Explicit logout/revocation

Client project data is stored separately from public knowledge.

Commercial, contractual, or unresolvable questions are escalated to a human account manager rather than being answered speculatively.

---

# Guardrails

The chatbot implements several safeguards.

### Pricing

The chatbot does not fabricate prices.

If pricing is unavailable from the knowledge base, it should state that pricing requires confirmation from the Abacus Digital team.

### Commitments

The chatbot does not make unauthorized commitments regarding:

* Delivery dates
* Scope
* Contracts
* Guarantees
* Commercial terms

### Sensitive Data

Sensitive information is rejected rather than stored or processed unnecessarily.

### Rate Limiting

Rate limits are applied at both:

* Session level
* IP level

### RAG Failure

When retrieval does not provide sufficient evidence, the system does not invent an answer.

It returns an honest uncertainty response.

---

# CRM Integration

CRM mirroring is optional.

Supported integrations include:

* Airtable
* HubSpot

The application database remains the **system of record**.

CRM synchronization is therefore treated as a mirror rather than the authoritative data source.

If CRM credentials are not configured, leads remain available inside the internal dashboard.

---

# Email Workflow

Email generation supports providers such as:

* Resend
* Brevo

The default workflow is:

```text
Conversation
     ↓
Project Brief
     ↓
Follow-up Email Draft
     ↓
Dashboard Approval Queue
     ↓
Human Approval
     ↓
Email Provider
     ↓
Client
```

Email sending is not automatic by default.

---

# Cost Tracking

The application records actual OpenRouter usage per conversation.

This allows the dashboard to track:

* Model usage
* Token usage
* Estimated generation cost
* Cost per conversation

This is especially important when using multiple model tiers and fallbacks.

---

# Environment Configuration

Copy the example environment file:

```bash
cp backend/.env.example backend/.env
```

At minimum, configure:

```env
DATABASE_URL=
OPENROUTER_API_KEY=
GOOGLE_API_KEY=
ADMIN_API_KEY=
```

Recommended:

```env
CRON_SECRET=
```

Optional integrations may include:

```env
EMAIL_PROVIDER=
RESEND_API_KEY=
BREVO_API_KEY=

CALENDLY_URL=

AIRTABLE_API_KEY=
HUBSPOT_ACCESS_TOKEN=

CORS_ORIGINS=
CLIENT_PORTAL_URL=
```

Refer to `.env.example` for the complete configuration surface.

---

# Known Limitations

## Neon Has Not Been Live-Tested

The complete request lifecycle and SQL behavior have been tested against a real embedded PostgreSQL instance.

However, the application has not been tested against a live Neon connection string.

The expectation is that this should work without modification because Neon provides standard PostgreSQL with `pgvector`.

This remains the primary unverified deployment-specific component.

## Gemini Embeddings Have Not Been Live-Tested

The embedding client, retry logic, batching, and request structure have been implemented against the Gemini API specification.

However, a live Gemini embedding API key was not available during development.

After configuring `GOOGLE_API_KEY`, verify that:

1. Indexing completes successfully.
2. Embeddings are stored correctly.
3. Retrieval returns relevant chunks.
4. RAG responses contain appropriate source citations.

If the Gemini response schema has changed, `backend/app/embeddings.py` may require adjustment.

## Email and CRM Integrations Are Disabled by Default

Email and CRM mirroring require external credentials.

Until those credentials are configured:

* Follow-up emails remain in the dashboard approval queue.
* No emails are sent.
* No leads are mirrored to external CRM systems.

---

# Development Notes

The application is designed to degrade gracefully.

```text
Missing embedding provider
        ↓
RAG unavailable
        ↓
No fabricated answer
        ↓
Honest fallback response
```

Similarly, a weak model failing to produce valid structured output does not necessarily terminate the request:

```text
Primary model
     ↓
Malformed JSON?
     ↓
Retry with stronger model
     ↓
Still invalid?
     ↓
Deterministic fallback
```

This architecture prioritizes reliability and correctness over maximum model autonomy.

---

# Status

| Phase                      | Status      |
| -------------------------- | ----------- |
| Phase 1 — Foundation       | Implemented |
| Phase 2 — Agentic Intake   | Implemented |
| Phase 3 — Client Support   | Implemented |
| Internal Dashboard         | Implemented |
| Public RAG                 | Implemented |
| Live Site Crawler          | Implemented |
| Lead Qualification         | Implemented |
| Calendly Booking           | Implemented |
| Project Brief Generation   | Implemented |
| Client Authentication      | Implemented |
| Client Knowledge Isolation | Implemented |
| Human Escalation           | Implemented |
| Email Approval Queue       | Implemented |
| CRM Mirror                 | Optional    |
| Neon Live Verification     | Pending     |
| Gemini Live Verification   | Pending     |

---

# License

Private project for **Abacus Digital Pvt. Ltd.**

