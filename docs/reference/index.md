# Reference

Generated from the code each time this site is built, so it matches the
commit the site was built from.

- **[REST API guide](../api.md)**: authentication, roles per route,
  pagination, errors, the WebSocket protocol and examples.
- **[OpenAPI (ReDoc)](rest-api.md)**: every endpoint and schema, from
  `vpp.api.app.create_app()`; also downloadable as
  [openapi.json](openapi.json).
- **[CLI](cli.md)**: `--help` of every `vpp` command.
- **Python API**: the modules a library user imports.
    - [Core](python/core.md): `VirtualPowerPlant`, resources, configuration,
      exceptions.
    - [Optimization](python/optimization.md): MPC controller, backtest
      harness, optimization plugin framework.
    - [Tariffs](python/tariffs.md): tariff model, bill components, meter
      traces, URDB import, bill simulation.
    - [Protocols](python/protocols.md): the protocol adapter base class and
      registry.
    - [Settings](python/settings.md): `VPP_*` settings.
