"""Event definitions for the Virtual Power Plant."""

from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

class EventType(str, Enum):
    """Types of VPP events."""
    # Resource events
    RESOURCE_ADDED = "resource_added"
    RESOURCE_REMOVED = "resource_removed"
    RESOURCE_ERROR = "resource_error"
    
    # Power events
    POWER_CHANGE = "power_change"
    POWER_LIMIT_REACHED = "power_limit_reached"
    
    # Optimization events
    OPTIMIZATION_START = "optimization_start"
    OPTIMIZATION_COMPLETE = "optimization_complete"
    OPTIMIZATION_FAILED = "optimization_failed"

@dataclass
class Event:
    """Base event class."""
    type: EventType
    timestamp: datetime
    details: Optional[Dict[str, Any]] = None
    resource_id: Optional[str] = None

@dataclass
class ResourceEvent(Event):
    """Resource-related event."""
    resource_type: str
    resource_name: str
    metrics: Optional[Dict[str, float]] = None

@dataclass
class PowerEvent(Event):
    """Power-related event."""
    power_value: float
    target_value: Optional[float] = None
    deviation: Optional[float] = None

@dataclass
class OptimizationEvent(Event):
    """Optimization-related event."""
    target: float
    result: Optional[float] = None
