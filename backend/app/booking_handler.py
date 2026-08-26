"""
Abacus Digital Chatbot - Booking Handler
Calendly integration for qualified leads.
"""

import logging
from typing import Dict, Any

from .config import settings
from .knowledge_base import get_service_url

logger = logging.getLogger(__name__)


class BookingHandler:
    """Manages booking flow for qualified leads."""

    def __init__(self):
        self.calendly_url = settings.calendly_url

    def generate_booking_response(
        self,
        lead_data: Dict[str, Any],
        qualification_score: float,
    ) -> Dict[str, Any]:
        """
        Generate a booking response with Calendly link.

        Returns:
        {
            "message": str,
            "booking_url": str,
            "show_booking": bool,
        }
        """
        name = lead_data.get("name", "")
        pain_point = lead_data.get("pain_point", "your needs")
        service = lead_data.get("service_interest", "our services")

        greeting = f"Great talking with you{', ' + name if name else ''}!"

        message = f"""{greeting} Based on what you've shared about {pain_point}, I think a discovery call with our team would be the perfect next step.

During the call, we'll:
• Dive deeper into your specific requirements
• Discuss how {service if service != 'our services' else 'our services'} can address your challenges
• Outline a tailored approach and timeline
• Give you a clear picture of investment and next steps

👉 **[Book your free discovery call here]({self.calendly_url})**

The call is completely free, no obligations. We just want to make sure we're the right fit for you!"""

        return {
            "message": message,
            "booking_url": self.calendly_url,
            "show_booking": True,
        }

    def generate_self_serve_response(self, lead_data: Dict[str, Any]) -> str:
        """Generate a response for leads that don't yet qualify for booking."""
        # service_interest can be several comma-joined names (chat_orchestrator tracks
        # every RAG source touched in a session) — link each one we recognize to its
        # real page rather than dumping them all onto the generic services listing.
        raw_services = [s.strip() for s in lead_data.get("service_interest", "").split(",") if s.strip()]

        resources = []
        for name in raw_services[:3]:
            url = get_service_url(name) or "https://www.abacusdigital.net/all-services"
            resources.append(f"• Check out our [{name}]({url}) page for detailed information")
        resources.append("• Browse our [blog](https://www.abacusdigital.net/blog) for insights and case studies")
        resources.append("• Explore all our [services](https://www.abacusdigital.net/all-services) to find the right fit")
        resources.append(f"• When you're ready for a deeper conversation, you can always [book a call]({self.calendly_url})")

        return f"""Thanks for sharing that! Here are some resources that might help:

{chr(10).join(resources)}

Feel free to ask me any other questions about our services — I'm here to help! 😊"""


# Singleton
booking_handler = BookingHandler()
