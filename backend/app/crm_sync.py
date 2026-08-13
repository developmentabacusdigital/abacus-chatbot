"""
Abacus Digital Chatbot - CRM Mirror (PRD 7.4)

The app's own database is the system of record. This module optionally mirrors lead
records into a free-tier CRM (Airtable or HubSpot) so the team can work in a familiar
UI. Mirroring is best-effort: a CRM outage never blocks or fails a conversation.
"""

import asyncio
import logging
from typing import Optional, Dict, Any

import httpx

from .config import settings
from .database import db
from .models import LeadRecord

logger = logging.getLogger(__name__)


class CRMSync:
    """Pushes lead records to an external CRM, if one is configured."""

    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None
        self._task: Optional[asyncio.Task] = None
        self.last_error: Optional[str] = None
        self.synced_count: int = 0

    @property
    def enabled(self) -> bool:
        if settings.crm_provider == "airtable":
            return bool(settings.airtable_api_key and settings.airtable_base_id)
        if settings.crm_provider == "hubspot":
            return bool(settings.hubspot_access_token)
        return False

    async def initialize(self):
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(20.0))

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    async def sync_lead(self, lead: LeadRecord) -> Dict[str, Any]:
        """Mirror a single lead. Returns a status dict; never raises."""
        if not self.enabled:
            return {"synced": False, "reason": "crm mirroring disabled"}
        if not (lead.email or lead.name):
            return {"synced": False, "reason": "no contact details to sync"}

        await self.initialize()
        try:
            if settings.crm_provider == "airtable":
                await self._sync_airtable(lead)
            else:
                await self._sync_hubspot(lead)
            self.synced_count += 1
            self.last_error = None
            return {"synced": True}
        except Exception as e:
            self.last_error = str(e)[:300]
            logger.error(f"CRM sync failed for lead {lead.id}: {e}")
            return {"synced": False, "reason": self.last_error}

    async def _sync_airtable(self, lead: LeadRecord):
        url = (
            f"https://api.airtable.com/v0/{settings.airtable_base_id}/"
            f"{settings.airtable_table}"
        )
        fields = {
            "Session ID": lead.session_id,
            "Name": lead.name or "",
            "Email": lead.email or "",
            "Company": lead.company or "",
            "Business Type": lead.business_type or "",
            "Pain Point": lead.pain_point or "",
            "Service Interest": lead.service_interest or "",
            "Budget Band": lead.budget_band or "",
            "Timeline": lead.timeline or "",
            "Qualification Score": lead.qualification_score,
            "Summary": lead.transcript_summary or "",
            "Next Step": lead.next_step or "",
        }

        # Upsert on Session ID so re-syncing a session updates rather than duplicates
        resp = await self._client.patch(
            url,
            headers={"Authorization": f"Bearer {settings.airtable_api_key}"},
            json={
                "performUpsert": {"fieldsToMergeOn": ["Session ID"]},
                "records": [{"fields": fields}],
            },
        )
        resp.raise_for_status()

    async def _sync_hubspot(self, lead: LeadRecord):
        headers = {"Authorization": f"Bearer {settings.hubspot_access_token}"}
        properties = {
            "email": lead.email or "",
            "firstname": (lead.name or "").split(" ")[0],
            "lastname": " ".join((lead.name or "").split(" ")[1:]),
            "company": lead.company or "",
            "message": lead.transcript_summary or lead.pain_point or "",
        }
        properties = {k: v for k, v in properties.items() if v}

        # Search first so we update the existing contact rather than erroring on conflict
        contact_id = None
        if lead.email:
            search = await self._client.post(
                "https://api.hubapi.com/crm/v3/objects/contacts/search",
                headers=headers,
                json={
                    "filterGroups": [{
                        "filters": [{
                            "propertyName": "email",
                            "operator": "EQ",
                            "value": lead.email,
                        }]
                    }],
                    "limit": 1,
                },
            )
            if search.status_code == 200:
                results = search.json().get("results") or []
                if results:
                    contact_id = results[0]["id"]

        if contact_id:
            resp = await self._client.patch(
                f"https://api.hubapi.com/crm/v3/objects/contacts/{contact_id}",
                headers=headers,
                json={"properties": properties},
            )
        else:
            resp = await self._client.post(
                "https://api.hubapi.com/crm/v3/objects/contacts",
                headers=headers,
                json={"properties": properties},
            )
        resp.raise_for_status()

    async def sync_session(self, session_id: str) -> Dict[str, Any]:
        """Mirror the lead attached to a session (called at session-end milestones)."""
        lead = await db.get_lead_by_session(session_id)
        if not lead:
            return {"synced": False, "reason": "no lead for session"}
        return await self.sync_lead(lead)

    async def sync_all(self, limit: int = 200) -> Dict[str, Any]:
        """Batch mirror, for the scheduled sync and the manual dashboard button."""
        if not self.enabled:
            return {"synced": 0, "reason": "crm mirroring disabled"}

        summaries = await db.get_all_leads(limit=limit)
        synced = 0
        for summary in summaries:
            lead = await db.get_lead_by_session(summary.session_id)
            if lead:
                result = await self.sync_lead(lead)
                synced += int(result.get("synced", False))
        return {"synced": synced, "considered": len(summaries)}

    async def _schedule_loop(self, interval_hours: int = 6):
        await asyncio.sleep(120)
        while True:
            try:
                result = await self.sync_all()
                logger.info(f"Scheduled CRM sync: {result}")
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Scheduled CRM sync failed: {e}")
            await asyncio.sleep(interval_hours * 3600)

    def start_scheduler(self):
        if self._task or not self.enabled:
            return
        self._task = asyncio.create_task(self._schedule_loop())
        logger.info(f"CRM mirror scheduler started (provider={settings.crm_provider})")

    async def stop_scheduler(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None


crm_sync = CRMSync()
