"""Deprecated alias: the benchmark suite moved into the package as ``vpp.benchmarks.metrics``.

Kept so ``from benchmarks... import ...`` keeps working from a source
checkout; the module object is replaced by ``vpp.benchmarks.metrics`` itself.
"""

import importlib
import sys

sys.modules[__name__] = importlib.import_module("vpp.benchmarks.metrics")
