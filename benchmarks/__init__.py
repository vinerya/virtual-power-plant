"""Deprecated alias package: the benchmark suite moved into ``vpp.benchmarks``.

Each ``benchmarks.<name>`` submodule here resolves to
``vpp.benchmarks.<name>``, so existing ``from benchmarks.x import ...`` code
keeps working from a source checkout. Import from ``vpp.benchmarks`` instead.
"""

from vpp.benchmarks import *  # noqa: F403
from vpp.benchmarks import __all__ as __all__
