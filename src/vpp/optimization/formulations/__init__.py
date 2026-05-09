"""
Pyomo formulations for VPP optimization problems.

This subpackage hosts concrete Pyomo model builders. Each builder takes a
plain-dict ``params`` and returns a ``pyomo.ConcreteModel``. Models use
``Param(mutable=True)`` for time-varying inputs (prices, forecasts) so the
same model instance can be re-solved with new data.
"""

from .dispatch import build_battery_dispatch_model

__all__ = ["build_battery_dispatch_model"]
