# Changelog

All notable changes to the Virtual Power Plant platform are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
