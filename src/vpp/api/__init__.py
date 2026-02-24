"""FastAPI REST + WebSocket API for the VPP platform."""

from .app import create_app

__all__ = ["create_app"]
