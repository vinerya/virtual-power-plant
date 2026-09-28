"""Deprecated alias package: the demos moved into ``vpp.demos``.

Each ``demos.<name>`` submodule here resolves to ``vpp.demos.<name>``, so
``from demos.residential_demo import run`` keeps working from a source
checkout. Use ``vpp demo <name>`` or ``vpp.demos`` instead.
"""
