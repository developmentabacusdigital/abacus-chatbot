"""
Abacus Digital Chatbot - Knowledge Base Parser
Parses the PDF document and website content into structured chunks for RAG indexing.
"""

import json
import os
import re
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from .config import settings

logger = logging.getLogger(__name__)


# --- Structured Knowledge Base from PDF ---
# Pre-parsed from "Abacus chatbot doc.pdf" to avoid runtime PDF dependency

ABACUS_KNOWLEDGE_BASE = {
    "company_overview": {
        "positioning": "Digital Growth & Technology Management Partner / Digital Transformation Partner for SMBs",
        "tagline": "AI-first digital solutions designed to help businesses scale smarter.",
        "elevator_pitch": "We design high-performance websites, automation systems, and marketing funnels that turn visitors into customers. We're an AI-powered web design, automation, and marketing partner for small and mid-sized businesses in the US and Europe.",
        "target_audience": "Small and mid-sized businesses (SMBs) in the US; currently prioritizing US SMBs with manufacturing companies as the primary focus, alongside tech, construction, logistics and others.",
        "core_capabilities": [
            "Web Solutions", "AI & Automation", "Lead Generation", "Brand Identity",
            "Cybersecurity", "Performance Marketing", "Digital Business Transformation",
            "Software Solutions", "Engineering Solutions"
        ],
        "stats": {
            "clients": "50+",
            "projects": "200+",
            "regions": "US and Europe"
        }
    },
    "services": {
        "web_design": {
            "name": "Web Design & Development",
            "tagline": "Design. Develop. Scale. Without Limits.",
            "positioning": "Web design company building high-performance, responsive websites and web applications for B2B, B2C, and ecommerce businesses.",
            "problems_solved": [
                "Website not ranking on Google despite quality products/services",
                "Poor responsive/mobile design driving users away",
                "Slow page speeds hurting UX/SEO",
                "Outdated site hurting credibility",
                "No clear conversion path for leads",
                "B2B vs B2C needing different site strategies",
                "Ecommerce sites not optimized for checkout/sales",
                "No CRM/ERP/marketing tool integration",
                "Content updates needing developer support and causing delays",
                "No analytics/tracking for user behavior insights"
            ],
            "sub_services": [
                {"name": "Custom Website Design & Development", "description": "Tailored responsive sites for B2B/B2C, SEO-ready, performance-optimized"},
                {"name": "Ecommerce Website Development", "description": "End-to-end D2C/B2B store setup, payment gateway integration, conversion-optimized checkout"},
                {"name": "Web Application Development", "description": "Custom business tools, client portals, partner-facing apps, API/ERP/CRM integrations"},
                {"name": "Website Redesign Services", "description": "UX audits, performance gap analysis, conversion-focused redesign without losing SEO"}
            ],
            "process": [
                "Research & Discovery - goals/audience/competitor research, full audit (SEO, performance, UX), scope/timeline, tech stack",
                "Design & Prototyping - wireframes, user journeys, responsive design, information architecture, interactive prototypes",
                "Development & Integration - front/back-end build, CRM/CMS/payment/automation integration, mobile & accessibility, QA testing",
                "Launch & Growth - speed optimization, zero-downtime deployment, GA/Search Console setup, post-launch monitoring"
            ]
        },
        "ai_automation": {
            "name": "AI & Automation",
            "tagline": "Optimize. Scale. Without Limits.",
            "positioning": "AI development + business process automation for B2B/B2C, reducing costs and eliminating inefficiencies.",
            "problems_solved": [
                "Manual processes raising time/costs",
                "No AI strategy while competitors automate",
                "Data silos blocking actionable insights",
                "Decisions made without predictive/real-time analytics",
                "Disconnected software causing workflow bottlenecks",
                "Fear of AI adoption / limited internal expertise",
                "No automation in sales, marketing, or operations",
                "High staffing costs for repetitive manual tasks",
                "No real-time visibility into business performance",
                "Legacy systems incompatible with modern AI"
            ],
            "sub_services": [
                {"name": "AI Development & Machine Learning Solutions", "description": "Custom AI/ML models, predictive analytics, intelligent data processing"},
                {"name": "Business Process Automation", "description": "Workflow automation across sales/marketing/finance/ops, document processing automation"},
                {"name": "AI Chatbots & Intelligent Customer Engagement", "description": "Custom chatbot dev for websites/messaging platforms, conversational AI/NLP, 24/7 support, lead qualification"},
                {"name": "AI Consulting & Digital Transformation Services", "description": "AI readiness assessment, strategy, tech stack recommendation, implementation roadmap"}
            ],
            "process": [
                "AI Readiness Assessment - audit processes/systems, identify high-impact automation areas, define objectives/success metrics",
                "Solution Design & Architecture - design AI models/workflows/data pipelines, integration mapping, POC/feasibility validation",
                "Development & Integration - build/train models, integrate with CRM/ERP/BI tools, testing/benchmarking",
                "Deployment & Continuous Improvement - deployment + team training, performance monitoring, retraining, ongoing scaling support"
            ]
        },
        "engineering": {
            "name": "Engineering Services",
            "tagline": "Elevate. Innovate. Engineer. Without Limits.",
            "positioning": "End-to-end engineering design & analysis for B2B/B2C businesses, the most directly manufacturing-relevant service in the lineup.",
            "problems_solved": [
                "Outdated designs with no digital documentation",
                "Inability to recreate or modify existing products",
                "Lack of precise technical drawings for manufacturing",
                "No structural or fluid flow analysis capability",
                "Complex civil infrastructure planning without expert support",
                "Inefficient mechanical systems slowing production",
                "No in-house 3D modeling"
            ],
            "sub_services": [
                {"name": "Reverse Engineering", "description": "Recreate, analyze, and modify existing products with precision; design enhancement"},
                {"name": "3D Modeling", "description": "High-precision modeling, manufacturing-ready drawings"},
                {"name": "Structural & Flow Analysis", "description": "Finite Element Analysis (FEA), Computational Fluid Dynamics (CFD), thermal & stress testing"},
                {"name": "Civil Design", "description": "Process plant design & project modification, structural load analysis"},
                {"name": "Mechanical Design", "description": "Product & machinery development, manufacturing optimization via reverse engineering & 3D modeling"}
            ],
            "process": []
        },
        "cybersecurity": {
            "name": "Cybersecurity",
            "tagline": "Protect. Prevent. Secure. Without Limits.",
            "positioning": "Enterprise-grade IT security + managed cybersecurity for B2B businesses.",
            "problems_solved": [
                "No cybersecurity strategy, exposing critical business assets",
                "Rising ransomware and phishing attack risks",
                "Remote work creating new security vulnerabilities",
                "No visibility into real-time threats or suspicious activity",
                "Outdated IT infrastructure vulnerable to modern threats",
                "No incident response plan for cyberattacks or recovery",
                "Third-party vendor access increasing security risks",
                "Employees lacking cybersecurity awareness and training"
            ],
            "sub_services": [
                {"name": "Network Security & Infrastructure Protection", "description": "Firewall management, network monitoring, intrusion detection, secure architecture design, vulnerability remediation"},
                {"name": "Managed Cybersecurity Services", "description": "24/7 SOC monitoring, rapid incident response, managed detection & response (MDR)"},
                {"name": "Cyber Risk Assessment & Compliance", "description": "Penetration testing, cyber risk assessment, security audits"},
                {"name": "Cloud Security & Endpoint Protection", "description": "Cloud security architecture/monitoring/access management, endpoint protection for remote/hybrid teams"}
            ],
            "process": [
                "Security Assessment & Discovery - audit IT infrastructure/security posture, identify vulnerabilities/threat vectors/compliance gaps",
                "Security Strategy & Architecture - tailored framework, network architecture/data protection plan, access controls/identity management",
                "Implementation & Integration - deploy network/endpoint/cloud security, integrate with existing IT, configure threat detection/monitoring",
                "Monitoring & Continuous Protection - 24/7 real-time monitoring, incident detection/response/recovery, regular audits/pen testing"
            ]
        },
        "lead_generation": {
            "name": "Lead Generation & Performance Marketing",
            "tagline": "Target. Optimize. Convert. Without Limits.",
            "positioning": "Lead generation agency delivering qualified B2B/B2C leads via data-driven performance marketing.",
            "problems_solved": [
                "Inconsistent sales pipeline with unpredictable lead flow",
                "Marketing spend generating poor ROI and weak tracking",
                "Difficulty identifying high-quality, conversion-ready leads",
                "B2B lead nurturing lacking automation and follow-up",
                "High cost per lead with low campaign efficiency",
                "No clear funnel from awareness to conversion",
                "Overreliance on referrals for business growth",
                "Landing pages not optimized for lead conversions",
                "Ad campaigns wasting budget without measurable results",
                "Sales and marketing teams misaligned on lead handling"
            ],
            "sub_services": [
                {"name": "B2B & B2C Lead Generation Services", "description": "Multi-channel campaigns (search/social/display), lead qualification frameworks, CRM integration"},
                {"name": "Paid Search & Performance Advertising", "description": "Google Ads, Meta Ads, LinkedIn campaign management, keyword strategy, ad copywriting, bid optimization"},
                {"name": "Conversion Rate Optimization (CRO)", "description": "Landing page design/testing, funnel analysis, drop-off point resolution"},
                {"name": "Demand Generation & Lead Nurturing", "description": "Email marketing automation, nurture sequences, content-driven demand gen, appointment setting"}
            ],
            "process": [
                "Audit & Strategy Development - audit existing campaigns/spend/lead quality, competitor analysis, ICP/persona definition",
                "Campaign Planning & Setup - campaign structures per channel, ad creatives/landing pages/lead capture assets, CRM/tracking setup",
                "Launch & Lead Generation - simultaneous multi-channel launch, daily monitoring of lead quality/CPL, real-time optimization",
                "Optimisation & Scaling - weekly performance analysis, continuous creative/landing page/audience testing, scale winners"
            ]
        },
        "social_media": {
            "name": "Social Media Marketing & Management",
            "tagline": "Engage. Grow. Convert. Without Limits.",
            "positioning": "End-to-end social media marketing for B2B/B2C/ecommerce - strategy, content, paid social, community management.",
            "problems_solved": [
                "No clear social media strategy aligned with business growth",
                "Inconsistent posting reducing reach and engagement",
                "Generic content failing to reflect brand identity",
                "Low organic visibility without a paid advertising strategy",
                "Social media followers not converting into leads or sales",
                "No insight into which platforms drive real business results",
                "Competitors gaining stronger visibility and audience attention",
                "Managing multiple platforms without a streamlined process",
                "Social media efforts disconnected from marketing and sales goals"
            ],
            "sub_services": [
                {"name": "Social Media Strategy & Planning", "description": "Platform selection, audience targeting, content calendar, campaign framework"},
                {"name": "Content Creation & Community Management", "description": "Original graphics/reels/carousels/short-form video, daily community engagement, reputation management"},
                {"name": "Paid Social Media Advertising", "description": "Meta/LinkedIn/Instagram ad campaigns for B2B lead gen and ecommerce conversions, audience segmentation, retargeting"},
                {"name": "Social Media Analytics & Reporting", "description": "Monthly performance reports, competitor benchmarking, audience growth analysis"}
            ],
            "process": [
                "Audit & Strategy Development - audit existing presence/content/performance, competitor benchmarking, persona/platform strategy",
                "Content Planning & Creation - monthly content calendar, platform-optimized assets, align with launches/promos/seasons",
                "Publishing & Community Management - scheduling at peak engagement times, daily monitoring, comment/DM management",
                "Performance Tracking & Optimisation - weekly/monthly KPI tracking, testing formats/creatives/schedules, refine strategy"
            ]
        },
        "brand_identity": {
            "name": "Brand Identity",
            "tagline": "Create. Differentiate. Influence. Without Limits.",
            "positioning": "Full-service branding - logo, strategy, visual identity systems for B2B/B2C/ecommerce.",
            "problems_solved": [
                "Inconsistent brand identity across digital and print",
                "Outdated logo and visual identity hurting credibility",
                "Brand messaging not connecting with target audiences",
                "No clear differentiation from competitors",
                "Inconsistent colors, fonts, and brand tone",
                "Rebranding not aligned with business vision",
                "New businesses struggling to build trust quickly",
                "Brand identity not optimized for digital platforms",
                "No brand guidelines for internal consistency",
                "Expanding into new markets without brand strategy"
            ],
            "sub_services": [
                {"name": "Logo Design & Visual Identity", "description": "Custom logo/brand mark, full visual identity system (colors, typography, iconography)"},
                {"name": "Brand Strategy & Positioning", "description": "Positioning framework, messaging strategy, tone of voice, competitive analysis, persona development"},
                {"name": "Brand Guidelines & Identity Systems", "description": "Style guide, asset libraries for digital/print/social/ecommerce"},
                {"name": "Rebranding & Brand Refresh Services", "description": "Full rebrand strategy, visual overhaul, repositioning, brand audit/gap analysis"}
            ],
            "process": [
                "Brand Discovery & Research - brand audit, competitor landscape, audience/persona research, vision/mission/values",
                "Strategy & Concept Development - positioning framework, messaging architecture, mood boards/visual direction, tone of voice",
                "Design & Identity Creation - logo + full visual identity, typography/color/iconography, assets for all channels",
                "Brand Rollout & Guidelines - finalize style guide, deliver assets in all formats, support rollout, ongoing consultancy"
            ]
        }
    },
    "cross_service_positioning": [
        "AI & Automation works best paired with solid web infrastructure",
        "Engineering services connect into AI automation",
        "Lead gen/performance marketing needs strong brand + website behind it to actually convert",
        "Social media drives awareness; performance marketing converts that awareness into revenue",
        "Cybersecurity and web infrastructure are framed together — a secure business needs a strong digital foundation"
    ],
    "website_info": {
        "url": "https://www.abacusdigital.net/",
        "title": "Digital Transformation Partner for SMBs | Abacus Digital",
        "description": "AI-powered web design, automation, and marketing for small and mid-sized businesses in the US and Europe. 50+ clients. Affordable. Reliable. Results-driven.",
        "contact_page": "https://www.abacusdigital.net/contact",
        "services_page": "https://www.abacusdigital.net/all-services",
        "blog_page": "https://www.abacusdigital.net/blog",
        "calendly_integration": True
    }
}


def get_knowledge_chunks() -> List[Dict[str, Any]]:
    """
    Convert the structured knowledge base into chunks suitable for vector indexing.
    Each chunk has: text, metadata (source, service, section).
    """
    chunks = []

    # Company overview chunk
    overview = ABACUS_KNOWLEDGE_BASE["company_overview"]
    chunks.append({
        "id": "company_overview",
        "text": f"""Abacus Digital - Company Overview
Positioning: {overview['positioning']}
Tagline: {overview['tagline']}
{overview['elevator_pitch']}
Target Audience: {overview['target_audience']}
Core Capability Areas: {', '.join(overview['core_capabilities'])}
Stats: {overview['stats']['clients']} clients, {overview['stats']['projects']} collaborative projects across {overview['stats']['regions']}.""",
        "metadata": {
            "source": "company_overview",
            "section": "overview",
            "service": "general"
        }
    })

    # Service chunks - create multiple chunks per service for better retrieval
    for service_key, service in ABACUS_KNOWLEDGE_BASE["services"].items():
        # Main service overview chunk
        chunks.append({
            "id": f"{service_key}_overview",
            "text": f"""{service['name']}
Tagline: {service['tagline']}
{service['positioning']}

Sub-services offered:
{chr(10).join('• ' + s['name'] + ': ' + s['description'] for s in service['sub_services'])}""",
            "metadata": {
                "source": f"service_{service_key}",
                "section": "service_overview",
                "service": service['name']
            }
        })

        # Problems solved chunk
        chunks.append({
            "id": f"{service_key}_problems",
            "text": f"""{service['name']} - Problems We Solve:
{chr(10).join('• ' + p for p in service['problems_solved'])}""",
            "metadata": {
                "source": f"service_{service_key}",
                "section": "problems_solved",
                "service": service['name']
            }
        })

        # Process chunk (if available)
        if service.get("process"):
            chunks.append({
                "id": f"{service_key}_process",
                "text": f"""{service['name']} - Our Process:
{chr(10).join('Step ' + str(i+1) + ': ' + p for i, p in enumerate(service['process']))}""",
                "metadata": {
                    "source": f"service_{service_key}",
                    "section": "process",
                    "service": service['name']
                }
            })

        # Individual sub-service chunks for more granular retrieval
        for sub in service["sub_services"]:
            chunks.append({
                "id": f"{service_key}_{sub['name'].lower().replace(' ', '_')[:30]}",
                "text": f"""{sub['name']} (part of {service['name']})
{sub['description']}
This is a sub-service under Abacus Digital's {service['name']} offering.""",
                "metadata": {
                    "source": f"service_{service_key}",
                    "section": "sub_service",
                    "service": service['name']
                }
            })

    # Cross-service positioning
    chunks.append({
        "id": "cross_service_positioning",
        "text": f"""Abacus Digital - Cross-Service Positioning and Bundling:
{chr(10).join('• ' + p for p in ABACUS_KNOWLEDGE_BASE['cross_service_positioning'])}

These insights help recommend multi-service bundles to clients based on their needs.""",
        "metadata": {
            "source": "cross_service",
            "section": "positioning",
            "service": "general"
        }
    })

    # Website info
    web = ABACUS_KNOWLEDGE_BASE["website_info"]
    chunks.append({
        "id": "website_info",
        "text": f"""Abacus Digital Website Information:
URL: {web['url']}
Title: {web['title']}
Description: {web['description']}
Contact Page: {web['contact_page']}
Services Page: {web['services_page']}
Blog: {web['blog_page']}""",
        "metadata": {
            "source": "website",
            "section": "info",
            "service": "general"
        }
    })

    return chunks


def save_knowledge_base(output_path: str = None):
    """
    Save the structured knowledge base to JSON for reference (debugging aid only —
    nothing reads this back; Postgres/pgvector is the real source of truth).

    Best-effort: a serverless deployment's filesystem is read-only outside /tmp, so
    this silently no-ops there rather than failing startup over a convenience dump.
    """
    path = output_path or settings.knowledge_base_path
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(ABACUS_KNOWLEDGE_BASE, f, indent=2, ensure_ascii=False)
        logger.info(f"Knowledge base saved to {path}")
    except OSError as e:
        logger.info(f"Skipping knowledge base snapshot (read-only filesystem?): {e}")


def get_service_names() -> List[str]:
    """Get all service names for intent matching."""
    return [s["name"] for s in ABACUS_KNOWLEDGE_BASE["services"].values()]


def get_service_by_keyword(keyword: str) -> Optional[Dict]:
    """Find a service by keyword match."""
    keyword_lower = keyword.lower()
    for key, service in ABACUS_KNOWLEDGE_BASE["services"].items():
        if keyword_lower in service["name"].lower():
            return service
        if keyword_lower in key.lower():
            return service
        for sub in service["sub_services"]:
            if keyword_lower in sub["name"].lower():
                return service
    return None


def get_service_catalog_text() -> str:
    """Compact catalogue of every service line, for the service-matching prompt."""
    lines = []
    for service in ABACUS_KNOWLEDGE_BASE["services"].values():
        subs = ", ".join(s["name"] for s in service["sub_services"])
        lines.append(
            f"- {service['name']}: {service['positioning']}\n"
            f"  Sub-services: {subs}"
        )
    # Capability areas without a dedicated section in the source doc
    lines.append(
        "- Software Solutions: custom software and internal tooling, delivered alongside "
        "Web Solutions and AI & Automation."
    )
    lines.append(
        "- Digital Business Transformation: multi-service engagements combining strategy, "
        "automation, and digital infrastructure."
    )
    return "\n".join(lines)


def get_bundling_logic_text() -> str:
    """Cross-service positioning rules used when recommending bundles."""
    return "\n".join("- " + p for p in ABACUS_KNOWLEDGE_BASE["cross_service_positioning"])


def get_service_line_names() -> List[str]:
    """Official names of all nine capability areas (PRD 7.6)."""
    return list(ABACUS_KNOWLEDGE_BASE["company_overview"]["core_capabilities"])


def build_client_chunks(client, projects) -> List[Dict[str, Any]]:
    """
    Build access-controlled chunks for the Phase 3 client knowledge base.

    Every chunk carries client_id in metadata; retrieval filters on it so a client can
    only ever see their own account data (PRD 7.7 / 9.5).
    """
    chunks: List[Dict[str, Any]] = []

    account = f"""Account overview for {client.company or client.name or client.email}
Primary contact: {client.name or 'Unknown'} ({client.email})
Company: {client.company or 'Not recorded'}
Account manager: {client.account_manager_email or 'Unassigned'}
Active projects: {len([p for p in projects if p.status == 'active'])} of {len(projects)} total."""

    chunks.append({
        "id": f"client_{client.id}_account",
        "text": account,
        "metadata": {
            "client_id": client.id,
            "section": "account",
            "source": "client_account",
        },
    })

    for project in projects:
        deliverables = "\n".join(f"• {d}" for d in project.deliverables) or "• None recorded"
        docs = "\n".join(f"• {d}" for d in project.support_docs) or "• None recorded"
        chunks.append({
            "id": f"client_{client.id}_project_{project.id}",
            "text": f"""Project: {project.name}
Status: {project.status}
Service line: {project.service_line or 'Not recorded'}
Started: {project.started_at or 'Not recorded'}
Target date: {project.target_date or 'Not recorded'}

Progress notes:
{project.progress_notes or 'No notes recorded.'}

Deliverables:
{deliverables}

Support documentation:
{docs}""",
            "metadata": {
                "client_id": client.id,
                "project_id": project.id,
                "project_name": project.name,
                "section": "project",
                "source": f"project:{project.name}",
            },
        })

    return chunks


