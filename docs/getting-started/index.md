# Getting started

Three short pages take you from nothing to a dispatch that changes a
(simulated) inverter's output:

1. **[Installation](installation.md)**: Docker Compose, or the Python
   package with the extras you need.
2. **[First run](first-run.md)**: database migrations, the first admin
   account, the API server and the web console.
3. **[Tutorial: simulated inverter](tutorial.md)**: start the bundled
   SunSpec simulator, register it as a resource, and dispatch it. About ten
   minutes, no hardware.

Then, to run it against your own equipment:
**[Connecting real data and devices](../connecting-real-data.md)** walks
through accounts and API keys, describing your fleet, telemetry over HTTP,
MQTT and Modbus, tariffs, grid signals, EV chargers, and turning on device
control one device at a time.

Prefer to look before installing? `vpp demo residential` and
`vpp benchmark run PEAK_SHAVING` run on synthetic data with no server or
database (see the [CLI reference](../reference/cli.md)).
