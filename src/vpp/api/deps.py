"""Shared FastAPI dependencies for route handlers."""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from vpp.core import VirtualPowerPlant
from vpp.config import VPPConfig

_vpp_instance: Optional[VirtualPowerPlant] = None


def get_vpp() -> VirtualPowerPlant:
    """Return the singleton VPP instance, creating one lazily if needed."""
    global _vpp_instance
    if _vpp_instance is None:
        _vpp_instance = VirtualPowerPlant(config=VPPConfig())
    return _vpp_instance


def reset_vpp() -> None:
    """Reset the singleton (useful in tests)."""
    global _vpp_instance
    _vpp_instance = None
