"""
Concrete solver plugins for the VPP optimization framework.

Currently provides:

- :class:`PyomoPlugin` - MILP plugin backed by Pyomo + HiGHS for the
  deterministic battery dispatch problem (Milestone 1).
- :class:`SimpleBatteryDispatchRules` - rule-based fallback used for fair
  comparison and when the solver stack is unavailable.
"""

from .pyomo_plugin import PyomoPlugin, SimpleBatteryDispatchRules

__all__ = ["PyomoPlugin", "SimpleBatteryDispatchRules"]
