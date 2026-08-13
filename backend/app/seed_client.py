"""
Seed a client and a sample project into the Phase 3 client knowledge base.

Usage:
    python -m app.seed_client --email you@example.com --name "Jane" --company "Acme Ltd"

Run from the backend/ directory with the virtualenv active. Existing clients are
updated rather than duplicated.
"""

import argparse
import asyncio
from datetime import datetime, timedelta

from .config import settings
from .database import db
from .indexer import indexer
from .models import ClientRecord, ClientProject


async def seed(email: str, name: str, company: str, manager: str, with_project: bool):
    await db.connect()
    try:
        client = await db.upsert_client(ClientRecord(
            email=email,
            name=name,
            company=company,
            account_manager_email=manager,
        ))
        print(f"Client: {client.name or client.email} ({client.id})")

        if with_project:
            existing = await db.get_client_projects(client.id)
            if existing:
                print(f"Client already has {len(existing)} project(s); skipping sample project.")
            else:
                today = datetime.utcnow()
                await db.upsert_project(ClientProject(
                    client_id=client.id,
                    name=f"{company or 'Client'} website rebuild",
                    status="active",
                    service_line="Web Design & Development",
                    started_at=(today - timedelta(days=38)).date().isoformat(),
                    target_date=(today + timedelta(days=24)).date().isoformat(),
                    progress_notes=(
                        "Discovery and IA complete. Design system signed off on "
                        f"{(today - timedelta(days=9)).date().isoformat()}. Front-end build is "
                        "roughly 60% done; templates for home, services and blog are in review. "
                        "Waiting on final product photography from the client before the gallery "
                        "template can be finished."
                    ),
                    deliverables=[
                        "Discovery report and sitemap — delivered",
                        "Design system and page designs — delivered",
                        "Front-end build (10 templates) — in progress",
                        "CRM integration and form routing — not started",
                        "SEO migration plan and 301 map — not started",
                    ],
                    support_docs=[
                        "CMS editing guide (how to publish a blog post)",
                        "Staging environment access and review process",
                        "Change request process and turnaround times",
                    ],
                ))
                print("Sample project created.")

        result = await indexer.index_client(client.id)
        print(f"Indexed into the client knowledge base: {result}")
        print(f"\nSign in at the portal with: {email}")
    finally:
        await db.disconnect()


def main():
    parser = argparse.ArgumentParser(description="Seed a Phase 3 client")
    parser.add_argument("--email", required=True)
    parser.add_argument("--name", default="")
    parser.add_argument("--company", default="")
    parser.add_argument("--manager", default=settings.account_manager_email)
    parser.add_argument("--no-project", action="store_true", help="skip the sample project")
    args = parser.parse_args()

    asyncio.run(seed(
        email=args.email,
        name=args.name,
        company=args.company,
        manager=args.manager,
        with_project=not args.no_project,
    ))


if __name__ == "__main__":
    main()
