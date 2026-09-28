"""
Concrete solver plugins for the VPP optimization framework.

Currently provides:

- :class:`PyomoPlugin` - MILP plugin backed by Pyomo + HiGHS for the
  deterministic battery dispatch problem (Milestone 1).
- :class:`SimpleBatteryDispatchRules` - rule-based fallback used for fair
  comparison and when the solver stack is unavailable.
- :class:`StochasticCVaRPlugin` - extensive-form stochastic dispatch with
  CVaR risk term (Milestone 2).
- :class:`PowerAllocationPlugin` / :class:`ProportionalAllocationRules` -
  single-interval fleet power allocation LP and its proportional fallback.
"""

from .pyomo_plugin import PyomoPlugin, SimpleBatteryDispatchRules
from .stochastic_plugin import StochasticCVaRPlugin
from .allocation_plugin import PowerAllocationPlugin, ProportionalAllocationRules

__all__ = [
    "PyomoPlugin",
    "SimpleBatteryDispatchRules",
    "StochasticCVaRPlugin",
    "PowerAllocationPlugin",
    "ProportionalAllocationRules",
]
