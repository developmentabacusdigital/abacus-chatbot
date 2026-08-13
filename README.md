# Abacus Digital Chatbot

AI chatbot for the [Abacus Digital](https://www.abacusdigital.net/) website. It answers
service questions grounded in the live site content, qualifies leads, runs agentic project
intake, books discovery calls, and supports existing clients through a separate
authenticated surface. Built against the [PRD](PRD.md), all three phases.

Runs on free tiers throughout: **Neon** for Postgres + vector search (pgvector), **Vercel**
for hosting, Gemini's free tier for embeddings, and OpenRouter's free model roster for
chat where quality allows.

---

## What's implemented

**Phase 1 — Foundation**
- RAG Q&A grounded in the company doc **plus a live crawl** of `/all-services/*` and
  `/blog/*`, with source citations and an honest "I don't know" when retrieval misses
- Daily scheduled re-index, plus a `POST /api/admin/reindex` webhook for publish events
- Conversational lead qualification with weighted scoring (budget 30 / timeline 25 /
  authority 25 / fit 20)
- Booking hand-off to Calendly for qualified leads, confirmed in chat and by email
- Every session persists — including abandoned ones — with progressive field capture,
  a transcript summary, and a suggested next step for the sales team

**Phase 2 — Agentic Intake**
- Multi-turn discovery agent covering goals, current state, constraints, budget, timeline
  and success criteria
- Service matching across all nine capability areas, including multi-service bundling
  with an explicit rationale
- Structured `ProjectBrief` stored as the system of record, with risk flags and open
  questions for the sales team
- Auto-drafted follow-up email, **queued for human approval** in the dashboard by default

**Phase 3 — Client Support**
- Magic-link authentication (single-use, 20-minute links; opaque server-side session tokens)
- A physically separate pgvector table for client project data, retrieved only with a
  hard, indexed `client_id` filter — not just a key inside a metadata blob
- Escalation to a human account manager on commercial, contractual, or unresolvable queries
- The public and client indexes are never queried in the same call

**Cross-cutting**
- Internal dashboard at `/dashboard/` — leads, briefs, email approvals, escalations,
  metrics, CSV export
- Optional CRM mirror to Airtable or HubSpot free tiers (the app's own DB stays the
  system of record)
- Guardrails: no fabricated pricing, commitment-language disclaimers, sensitive-data
  rejection, per-session and per-IP rate limits
- Real per-conversation cost tracking from OpenRouter usage

---

## Quick start

You need three things before the app will fully boot: a Postgres database, an OpenRouter
key, and a Gemini key for embeddings. All three have usable free tiers.

1. **Database — [Neon](https://neon.tech)**: create a free project, then copy the
   **pooled** connection string (hostname contains `-pooler`) from the dashboard.
2. **Chat — [OpenRouter](https://openrouter.ai/keys)**: create a key.
3. **Embeddings — [Google AI Studio](https://aistudio.google.com/apikey)**: create a
   free key. Without it the app still boots and answers greetings/general questions —
   it just can't ground answers in the knowledge base until the key is set.

```bash
cd backend
cp .env.example .env
# set DATABASE_URL, OPENROUTER_API_KEY, GOOGLE_API_KEY, ADMIN_API_KEY
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://localhost:8000/widget/` for the public widget, or
`http://localhost:8000/dashboard/` for the team dashboard. First boot creates the
Postgres tables and pgvector extension automatically, then indexes the static knowledge
base and crawls the live site about 30 seconds later.

### Client portal (Phase 3)

```bash
cd backend
python -m app.seed_client --email you@example.com --name "You" --company "Acme Ltd"
```

Then open `http://localhost:8000/widget/client.html` and sign in with that email. With no
email provider configured the API returns the magic-link token directly so the flow is
testable locally — it never does this once `EMAIL_PROVIDER` is set.

### Tests

```bash
cd backend
python -m pytest          # 129 tests, no network or API keys required
```

Tests run against a real, throwaway embedded Postgres instance (via the `pgserver`
package — no Docker or local Postgres install needed) rather than mocks, since the SQL
itself (upserts, pgvector queries, JSONB filters) is exactly what needs checking. LLM
calls go through a fake router; embeddings go through a deterministic fake so relevance
ranking is still meaningful.

---

## Architecture

```
                    ┌────────────────────────────────────────┐
                    │            Framer website               │
                    │  ┌──────────────┐   ┌───────────────┐  │
                    │  │ public widget│   │ client widget │  │
                    │  │ data-mode=   │   │ data-mode=    │  │
                    │  │ "public"     │   │ "client"      │  │
                    │  └──────┬───────┘   └───────┬───────┘  │
                    └─────────┼───────────────────┼──────────┘
                              │ /api/chat         │ /api/client/*
                    ┌─────────▼───────────────────▼──────────┐
                    │      FastAPI on Vercel (serverless)     │
                    │                                         │
                    │   ┌─────────────────────────────────┐  │
                    │   │       Chat Orchestrator          │  │
                    │   │  intent → route → persist → CRM  │  │
                    │   └──┬────┬────┬────┬────┬──────────┘  │
                    │      │    │    │    │    │              │
                    │    RAG  Qual Intake Book Client         │
                    │      │    │    │         support        │
                    │      │    │    │           │            │
                    │  ┌───▼────▼────▼───────────▼─────────┐ │
                    │  │      Neon Postgres + pgvector      │ │
                    │  │  sessions/leads/briefs/emails/...  │ │
                    │  │  public_documents · client_documents│ │
                    │  │        (never queried together)     │ │
                    │  └─────────────────────────────────────┘ │
                    │                                         │
                    │  Gemini (embeddings) · OpenRouter (chat)│
                    │  Vercel Cron → /api/cron/* (scheduled)  │
                    └─────────────────────────────────────────┘
```

### Model routing

| Task | Primary | Fallback |
|---|---|---|
| Intent classification | `llama-3.1-8b-instruct:free` | `gemini-2.0-flash-exp:free` |
| RAG answers, qualification | `gemini-2.0-flash-exp:free` | `gemini-2.5-flash` |
| Intake reasoning, service matching, briefs | `gemini-2.5-flash` | `claude-haiku-4.5` |
| Client support | `gemini-2.5-flash` | `claude-haiku-4.5` |
| Embeddings (RAG indexing + retrieval) | Gemini `text-embedding-004` | — |

Any structured LLM call that returns malformed JSON is retried once on the stronger model
before the deterministic fallback takes over, so a cheap model failing never dead-ends a
conversation. A missing or failing embedding provider degrades the same way — RAG answers
fall back to an honest "I don't know" instead of crashing the request.

---

## API

### Public
| Method | Path | Description |
|---|---|---|
| `POST` | `/api/chat` | Send a chat message |
| `POST` | `/api/sessions/{id}/end` | Close a session (summarise + CRM sync) |
| `GET` | `/api/chats` · `/chats/{id}` | Visitor's own chat list / transcript |
| `DELETE` | `/api/data` | Delete data by email, session id, or visitor id |
| `GET` | `/health` | Health check; the widget uses this to degrade gracefully |

### Client portal (bearer token)
| Method | Path | Description |
|---|---|---|
| `POST` | `/api/client/login` | Request a magic link |
| `POST` | `/api/client/verify` | Redeem a link for a session token |
| `POST` | `/api/client/chat` | Authenticated support chat |
| `GET` | `/api/client/me` | Profile and projects |
| `POST` | `/api/client/logout` | Revoke the session token |

### Admin (`X-Admin-Key` header)
| Method | Path | Description |
|---|---|---|
| `GET` | `/api/admin/leads` · `/leads.csv` | Lead list and CSV export |
| `GET` | `/api/admin/metrics` | PRD Section 10 metrics |
| `GET` | `/api/admin/sessions/{id}` | Transcript, lead and brief |
| `GET` | `/api/admin/briefs` | Intake briefs |
| `GET` | `/api/admin/emails` · `POST /emails/approve` | Email approval queue |
| `GET` | `/api/admin/escalations` | Open human hand-offs |
| `POST` | `/api/admin/reindex` | Re-index (wire to a publish webhook) |
| `POST` | `/api/admin/crm-sync` | Mirror leads to the configured CRM |
| `POST` | `/api/admin/clients` · `/clients/{id}/projects` | Manage client portal data |

If `ADMIN_API_KEY` is unset, every admin route returns 503 — the lead database is never
left open by default.

### Scheduled jobs (`Authorization: Bearer <CRON_SECRET>`, or the admin key)
| Method | Path | Description |
|---|---|---|
| `GET` | `/api/cron/reindex` | Daily public knowledge base re-index |
| `GET` | `/api/cron/crm-sync` | Mirror leads to the configured CRM |
| `GET` | `/api/cron/sweep` | Finalize abandoned sessions, purge expired tokens |

Vercel Cron hits these on the schedule in `vercel.json` and sends the bearer token
automatically once `CRON_SECRET` is set as a project env var. On any always-on host
(Render, Docker, etc.) these are redundant — the same work also runs as an in-process
background loop there, and self-disables on Vercel (detected via the `VERCEL` env var).

---

## Framer embed

Public pages:

```html
<script
  src="https://YOUR-BACKEND-URL/widget/abacus-chat-widget.js"
  data-api-url="https://YOUR-BACKEND-URL"
  data-calendly-url="https://calendly.com/abacusdigital/discovery">
</script>
```

Client portal page only:

```html
<script
  src="https://YOUR-BACKEND-URL/widget/abacus-chat-widget.js"
  data-api-url="https://YOUR-BACKEND-URL"
  data-mode="client">
</script>
```

Set `CORS_ORIGINS` to the Framer domain and `CLIENT_PORTAL_URL` to the page hosting the
client widget — magic links point there.

---

## Deployment

### Vercel + Neon (recommended)

1. **Neon**: create a free project, enable the `vector` extension is automatic (the app
   runs `CREATE EXTENSION IF NOT EXISTS vector` itself on first connect — the default
   Neon role has permission). Copy the **pooled** connection string.
2. **Vercel**: import the repo, set these project environment variables, then deploy —
   `vercel.json` and `api/index.py` handle the rest.

   | Variable | Required | Notes |
   |---|---|---|
   | `DATABASE_URL` | yes | Neon's pooled connection string |
   | `OPENROUTER_API_KEY` | yes | chat generation |
   | `GOOGLE_API_KEY` | yes | embeddings (RAG grounding) |
   | `ADMIN_API_KEY` | yes | protects `/api/admin/*` |
   | `CRON_SECRET` | recommended | lets Vercel Cron call `/api/cron/*` |
   | everything else in `.env.example` | optional | email/CRM providers, Calendly, CORS, etc. |

3. Vercel serves `widget/` and `dashboard/` as static files directly (fast, free, no
   function invocation) and routes everything else to `api/index.py`. The FastAPI app
   also mounts those same directories itself as a fallback, so it works correctly either
   way — verify `/widget/` loads after your first deploy.

**Known Vercel constraints, not app bugs:**
- **Hobby plan caps function execution at 10 seconds** and Cron Jobs at once-daily.
  The Phase 2 brief-generation flow chains a few LLM calls and can occasionally run
  close to that ceiling — if you hit timeouts, Vercel Pro raises both limits (this repo's
  `vercel.json` already requests `maxDuration: 60`, which only takes effect on Pro).
- **No WebSocket support in serverless functions.** The `/ws/chat` endpoint from earlier
  versions of this app is gone — the widget only ever used the REST `/api/chat` endpoint,
  so nothing was lost, but a future contributor shouldn't be surprised it's missing.
- **No local disk.** Nothing in the app writes anywhere except `/tmp` (ephemeral) and
  Neon — this was true after the Postgres/pgvector migration regardless of host, so
  Vercel's read-only filesystem isn't a special case to work around.

### Alternative: Docker on Render (or any container host)

Same codebase, same Neon database — just a different, always-on-while-warm host instead
of serverless. Useful if you outgrow Vercel's execution-time limits before upgrading to
Pro, or just prefer a persistent process.

```bash
docker build -t abacus-bot .     # build context must be the repo root
docker run -p 8000:8000 --env-file backend/.env abacus-bot
```

`render.yaml` deploys the same image on Render's free tier. Free instances spin down
after ~15 minutes idle, so the first message after a quiet period pays a cold start —
the widget shows a typing indicator throughout, and falls back to the contact form if the
backend is unreachable. Unlike Vercel, the in-process background loops (re-index, CRM
sync, session sweep) run natively here — the `/api/cron/*` endpoints are simply unused.

---

## Project structure

```
├── api/
│   └── index.py                 # Vercel entrypoint — re-exports the FastAPI app
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app: public, client, admin, and cron APIs
│   │   ├── config.py            # Settings, model routing, prompts
│   │   ├── models.py            # Pydantic models + lead field whitelist
│   │   ├── database.py          # Postgres (asyncpg): sessions, leads, briefs, clients
│   │   ├── vector_store.py      # pgvector, namespaced public vs client tables
│   │   ├── embeddings.py        # Gemini embeddings API client
│   │   ├── knowledge_base.py    # Static KB from the company doc
│   │   ├── site_crawler.py      # Live site crawl → RAG chunks
│   │   ├── indexer.py           # Index building + re-index scheduler
│   │   ├── rag_engine.py        # Public and client retrieval (never mixed)
│   │   ├── llm_router.py        # OpenRouter routing, JSON repair, cost tracking
│   │   ├── intent_classifier.py # Keyword fast path + LLM fallback
│   │   ├── lead_qualifier.py    # Slot filling + weighted scoring
│   │   ├── intake_agent.py      # Phase 2 discovery, matching, brief generation
│   │   ├── booking_handler.py   # Calendly hand-off
│   │   ├── client_support.py    # Phase 3 support + escalation
│   │   ├── auth.py              # Magic links, client sessions, admin/cron guards
│   │   ├── email_service.py     # Resend/Brevo + approval queue
│   │   ├── crm_sync.py          # Airtable/HubSpot mirror
│   │   ├── guardrails.py        # Safety + rate limiting
│   │   ├── chat_orchestrator.py # State machine
│   │   └── seed_client.py       # CLI to seed a portal client
│   ├── dashboard/index.html     # Internal team dashboard
│   └── tests/                   # 129 tests, run against a real embedded Postgres
├── widget/                      # Embeddable widget (public + client modes)
├── vercel.json                  # Routing + Cron schedule
├── requirements.txt              # Points at backend/requirements.txt (Vercel needs root)
├── Dockerfile · render.yaml · Procfile   # Alternative: always-on container host
└── PRD.md
```

---

## Known limitations

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
