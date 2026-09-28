"""Battery degradation models (Milestone 1).

Standalone library of degradation/aging models. These models do NOT
modify or depend on the existing :class:`vpp.resources.Battery` state.
They are designed to be wired into the optimizer in M2 as wear-cost
terms, and persisted in M3.

Public API
----------
- :class:`DegradationModel` -- abstract base class
- :class:`ThroughputDegradation`
- :class:`CalendarDegradation`
- :class:`RainflowDegradation`
- :data:`LFP_PRESET`, :data:`NMC_PRESET` -- ready-made parameter sets
"""

from .models import (
    LFP_PRESET,
    NMC_PRESET,
    CalendarDegradation,
    DegradationModel,
    RainflowDegradation,
    ThroughputDegradation,
)
from .optimization import (
    WearCost,
    add_calendar_aging_bias,
    add_dod_constraints,
    add_wear_cost_term,
    wear_cost_hooks_for_telemetry_consistency,
)
from .telemetry import DegradationUpdater, SOHUpdate, TelemetryWindow

__all__ = [
    "LFP_PRESET",
    "NMC_PRESET",
    "CalendarDegradation",
    "DegradationModel",
    "DegradationUpdater",
    "RainflowDegradation",
    "SOHUpdate",
    "TelemetryWindow",
    "ThroughputDegradation",
    "WearCost",
    "add_calendar_aging_bias",
    "add_dod_constraints",
    "add_wear_cost_term",
    "wear_cost_hooks_for_telemetry_consistency",
]
