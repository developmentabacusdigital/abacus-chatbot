"""
Vercel entrypoint. Vercel's Python runtime builds each file under api/ as its own
serverless function and expects an ASGI/WSGI `app` object at module level — this file
just re-exports the real FastAPI app from backend/app/main.py so the actual
application code has exactly one home, not two.
"""

import os
import sys

_BACKEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend")
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from app.main import app  # noqa: E402

__all__ = ["app"]
