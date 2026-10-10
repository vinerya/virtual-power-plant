---
hide:
  - navigation
---

# Virtual Power Plant

An open-source, self-hostable platform for aggregating and dispatching
distributed energy resources: batteries, PV, wind and EV chargers. It has a
FastAPI backend, a Next.js operator console and customer portal, OCPP /
OpenADR / IEEE 2030.5 / Modbus / MQTT integrations, and optimization you
can read, test and change. Every dispatch decision is stored with the
inputs that produced it.

<div class="grid cards" markdown>

-   **Get started**

    ---

    Install the package, run the API and walk through a dispatch to a
    simulated SunSpec inverter in about ten minutes.

    [Getting started](getting-started/index.md)

-   **Deploy and operate**

    ---

    Docker Compose or bare metal, PostgreSQL, TLS, backups, upgrades,
    monitoring and capacity numbers.

    [Deployment](deployment.md) · [Operations](operations.md) · [Performance](performance.md)

-   **Connect devices and grid signals**

    ---

    OCPP 1.6-J Central System, OpenADR 2.0b VEN, IEEE 2030.5 client,
    Modbus/SunSpec control, MQTT telemetry, V2G and demand response.

    [Protocols, V2G and DR](protocols.md)

-   **Reference**

    ---

    REST API (OpenAPI), the `vpp` command line, and the Python library.

    [REST API](reference/rest-api.md) · [CLI](reference/cli.md) · [Python API](reference/python/core.md)

</div>

!!! warning "Beta software"

    The project is **beta**. It has a large automated test suite, but it
    has not been certified against any protocol conformance suite or proven
    in a field deployment. Device control and demand-response auto-response
    are **off by default**. Read the status table below before connecting it
    to equipment.

## What works today

<!-- include: README.md#maturity -->

## What it is not

- **Not connected to a real market.** Trading runs against a simulated
  venue only.
- **Not certified.** Protocol support is tested against in-process mock
  peers and the bundled SunSpec simulator, not against conformance suites
  or vendor hardware.
- **Not a hosted service.** You run it yourself, on your own
  infrastructure. (This site, too, loads no analytics or third-party
  assets.)

## Project links

- Source code and issues: [github.com/vinerya/virtual-power-plant](https://github.com/vinerya/virtual-power-plant)
- [Changelog](changelog.md) · [Contributing](contributing.md) · [License (MIT)](license.md)
