"""Device simulators for exercising the VPP without hardware.

* :mod:`vpp.simulators.sunspec` -- a SunSpec Modbus TCP device (common model
  1, inverter model 103 or 113, controls model 123, storage model 124) with
  simple PV + battery physics. Run it with ``vpp simulate sunspec`` or
  ``python -m vpp.simulators.sunspec``.

A simulator built from the published specification shows that the VPP
speaks the specification; it does not show how any vendor's firmware
behaves.
"""
