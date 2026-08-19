# Changelog

All notable changes to the Virtual Power Plant platform are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

**Combined tiered + TOU rate structures**
- `TimeOfUseRate` gains `period_tiers`: an optional inclining-block tier
  schedule scoped to an individual TOU period, keyed by that period's
  label. When a URDB `energyratestructure` has more than one TOU period
  AND at least one of them also has more than one usage tier, that period
  now bills correctly against its own cumulative kWh for the billing
  cycle (URDB's per-period tier convention) instead of silently using
  only its first (lowest) tier's rate for all usage.
- `urdb.py`'s existing single-period-multiple-tiers shortcut (folds into
  `TieredEnergyRate`) is unchanged and still used when it applies — the
  new machinery only engages for the genuinely multi-period case.

**NEM 2.0 TOU-period-aware export credit**
- URDB `energyratestructure[..].sell` is now parsed into
  `TOUSchedule.sell_rate` and exposed via `TimeOfUseRate.export_rate()`,
  which falls back to the period's import rate when the source tariff
  doesn't define a distinct sell rate.
- Bill simulation's NEM 2.0 export credit now prices each exported
  interval at *its own* TOU period rate (URDB `sell` if defined, else the
  same as import) instead of a single bill-wide blended average — the old
  proxy mis-credited any customer whose export TOU mix differed from
  their import mix. Tariffs without a TOU component (flat or tiered-only)
  still fall back to the blended average, which has no period to be more
  accurate about.
- The dispatch-side optimizer (`tariff_to_opt_params`, `nem="nem2"`) now
  prefers the same explicit `sell_rate` per step, while still deferring to
  a live price-feed override when one is active for that step.

**MQTT Battery Telemetry Ingestion (M5)**
- `MQTTTelemetryIngestor` bridges `MQTTAdapter` messages on
  `vpp/{site_id}/{resource_type}/{resource_id}/{metric}` topics into
  `battery_states` rows — closes the gap where the degradation updater's
  DB-backed telemetry fetch had a correct read path but nothing ever wrote
  to that table in production.
- `MQTTAdapter.receive_forever()` — async generator draining the adapter's
  message queue without busy-polling.
- Ingested telemetry publishes `RESOURCE_UPDATED` on the event bus, so
  connected WebSocket clients on `resource_updates` see live updates.
- New settings: `VPP_MQTT_INGESTION_ENABLED` (default off — dials out to an
  external broker), `VPP_MQTT_BROKER_HOST`, `VPP_MQTT_BROKER_PORT`,
  `VPP_MQTT_TOPIC_PREFIX`, `VPP_MQTT_USERNAME`, `VPP_MQTT_PASSWORD`.
- Ingestion runs as a lifespan-managed background task; a failed initial
  broker connection is retried on a fixed delay instead of crashing
  startup, and the adapter registers into the shared protocol registry so
  `GET /api/v1/protocols` reflects its status.

### Fixed

- EventBus publishes now reach connected WebSocket clients — previously
  two disconnected pub/sub systems.
- Rate limiting middleware (opt-in via `VPP_RATE_LIMIT_ENABLED`, on by
  default) wired into `create_app()`.
- Non-admins can no longer mint an API key with a role higher than their
  own.
- `RiskManager.check_limits()` now checks VaR and position concentration
  in addition to position/loss/drawdown limits, pricing positions from
  live market data when available.
- `wear_cost_hooks_for_telemetry_consistency()` keeps a live dispatch
  model's wear-cost term consistent with the Rainflow + calendar-aging
  model the telemetry updater actually persists as SOH.
- Removed dead flat modules superseded by their package equivalents
  (`analysis.py`, `config.py`, `events.py`, `optimization.py`,
  `simulation.py`, `visualization.py`).

## [2.0.0] - 2025-02-24

### Added

**Production Infrastructure (Phase 1)**
- FastAPI REST + WebSocket API with automatic OpenAPI docs
- JWT authentication with role-based access control and API key support
- SQLAlchemy 2.0 async database layer (SQLite dev, PostgreSQL prod)
- Pydantic v2 request/response schemas for all endpoints
- Click CLI (`vpp serve`, `vpp dispatch`, `vpp benchmark`, `vpp demo`)
- Docker Compose deployment (production + development + monitoring)
- GitHub Actions CI/CD — lint, test matrix, security scan, Docker build, PyPI release
- Pre-commit hooks (ruff, mypy, detect-secrets)
- Structured logging via structlog (JSON prod, colored dev)
- Prometheus metrics collector (`/metrics` endpoint)
- Alert engine with threshold, rate-of-change, and Z-score anomaly rules
- Typed event bus with async publish and WebSocket broadcast

**Protocol Integrations (Phase 2)**
- OpenADR 2.0b adapter — VTN/VEN roles, DR event handling, auto opt-in
- OCPP 1.6 adapter — charge point management, remote start/stop, charging profiles
- MQTT adapter — IoT telemetry pub/sub with hierarchical topic structure
- Modbus TCP/RTU adapter — pre-built register maps for SMA, Fronius, SolarEdge inverters
- IEEE 2030.5 adapter — Smart Energy Profile, DER program control
- Protocol registry with unified connect/disconnect/send/receive interface

**Vehicle-to-Grid & Grid Control (Phase 2)**
- EV battery models with capacity, SOC, charge/discharge rates, V2G capability
- Smart charging scheduler with TOU-aware and solar-priority strategies
- Fleet aggregator for ancillary services flexibility bidding
- Grid-forming inverter models with droop control and virtual synchronous machine (VSM)
- Microgrid controller — island detection, seamless transitions, load priority shedding

**Monitoring & Observability (Phase 3)**
- Prometheus metrics for resources, optimization, trading, protocols, API
- Grafana dashboard configurations (VPP overview, trading performance)
- Prometheus scrape configuration
- Docker Compose monitoring stack (Prometheus + Grafana + node-exporter)

**Research / AI Layer (Phase 3)**
- Gaussian Process forecasting for load, price, and renewable prediction
- PPO reinforcement learning for dispatch optimization (non-production, shadows rule-based)
- Anomaly detection with configurable sensitivity
- Experiment runner with seed management and reproducible comparison tables

**Benchmarking Suite (Phase 4)**
- 4 synthetic datasets: IEEE residential, California ISO, EU multi-zone grid, 50-vehicle EV fleet
- 7 predefined scenarios: peak shaving, frequency response, V2G arbitrage, multi-site coordination, islanding, high renewable penetration, multi-market trading
- 13 standardized metrics: peak reduction, self-consumption, battery cycles, Sharpe ratio, max drawdown, CO2 reduction, uptime, and more
- Benchmark runner with method comparison and markdown report generation
- 3 built-in benchmark methods: NoOp baseline, rule-based peak shaving, simple V2G scheduler

**Demo Applications (Phase 4)**
- Residential VPP demo — 10 homes with solar + battery, peak shaving
- EV fleet V2G demo — 50-vehicle parking garage, smart vs. dumb charging
- Microgrid islanding demo — grid fault, island transition, reconnection
- Trading bot demo — multi-market arbitrage with P&L tracking
- Multi-protocol demo — OpenADR + OCPP + MQTT + Modbus coordination
- Dashboard demo — terminal UI with ASCII progress bars and live panels

### Changed
- Upgraded minimum Python version from 3.8 to 3.10
- Migrated project metadata from setup.py to pyproject.toml (PEP 621)
- Restructured project as installable package with `src/` layout
- Comprehensive README rewrite covering the full platform

### Removed
- Removed redundant `setup.py` (superseded by `pyproject.toml`)

## [1.0.0] - 2025-01-15

### Added
- Core VPP management library with resource models (Battery, Solar, Wind)
- Stochastic optimization with scenario generation
- Real-time grid services with sub-millisecond frequency/voltage response
- Distributed coordination via ADMM for multi-site VPP portfolios
- Model Predictive Control for rolling-horizon dispatch
- Plugin architecture for custom optimization solvers
- Multi-market trading system (day-ahead, real-time, ancillary services, bilateral)
- 5 trading strategies: arbitrage, momentum, mean reversion, ML-based, multi-market
- Portfolio management with P&L tracking and risk controls
- Physics-based battery models with electrochemical accuracy
- Solar PV and wind turbine resource models
- YAML/JSON configuration system with validation and hot reload
- Rule-based expert system for operational decisions
- Comprehensive test suite
