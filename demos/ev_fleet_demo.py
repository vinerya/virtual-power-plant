"""Deprecated alias: the demos moved into the package as ``vpp.demos.ev_fleet_demo``.

Kept so ``from demos... import run`` keeps working from a source checkout;
the module object is replaced by ``vpp.demos.ev_fleet_demo`` itself.
"""

import importlib
import sys

sys.modules[__name__] = importlib.import_module("vpp.demos.ev_fleet_demo")
