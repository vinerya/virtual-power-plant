# Core library

The in-process library: a `VirtualPowerPlant` holds `EnergyResource`
objects (batteries, PV, wind turbines) and splits a target power across
them. It needs no database or API server. The REST API keeps its own fleet
in the database instead (see the [API guide](../../api.md)).

```python
from vpp import VirtualPowerPlant, VPPConfig
from vpp.resources import Battery, Solar, WindTurbine
```

::: vpp.core

::: vpp.resources

::: vpp.config.VPPConfig

::: vpp.exceptions
