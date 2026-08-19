<div align="center">

# Virtual Power Plant Platform

The world's first comprehensive, open-source Virtual Power Plant platform.
Production-ready optimization, multi-protocol DER control, V2G-native, with a reproducible benchmarking suite.

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests: 359 passed](https://img.shields.io/badge/tests-359%20passed-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Status: Production Ready](https://img.shields.io/badge/status-production%20ready-green.svg)]()

[Features](#features) · [Quick Start](#quick-start) · [Architecture](#architecture) · [Demos](#demo-applications) · [Benchmarks](#benchmarking-suite) · [API](#rest--websocket-api) · [Contributing](#contributing)

</div>

---

## Why This Exists

The VPP software market is **$6.28B (2025)** growing to **$39.31B by 2034**. Every commercial platform (Tesla Autobidder, Next Kraftwerke, AutoGrid) is a proprietary black box costing **$50K-500K/year**. No comprehensive open-source alternative exists.

This project fills that gap: a **complete, production-ready VPP platform** with transparent algorithms, industry-standard protocol support, and V2G-native design. Built for startups, researchers, and grid operators who need full control over their optimization logic.

**Design principles:** Rule-based production excellence first. AI for research only. Everything is a plugin. Edge-first, cloud-optional.

---

## Features

### Optimization Engine
- **Stochastic optimization** with scenario generation (price, renewable, load uncertainty)
- **Real-time grid services** with sub-millisecond frequency/voltage response
- **Distributed coordination** via ADMM for multi-site VPP portfolios
- **Model Predictive Control** for rolling-horizon dispatch
- **Plugin architecture** — drop in custom solvers with automatic rule-based fallbacks
- **CVaR risk management** for uncertainty-aware scheduling

### Trading System
- **4 market types**: day-ahead auctions, real-time continuous, ancillary services, bilateral contracts
- **5 strategy engines**: arbitrage, momentum, mean reversion, ML-based, multi-market
- **Portfolio management** with real-time P&L, position tracking, and risk metrics
- **Risk controls**: position limits, daily loss caps, VaR, max drawdown, concentration limits

### Protocol Integrations
- **OpenADR 2.0b** — demand response event handling (VTN/VEN roles)
- **OCPP 1.6** — EV charger management, remote start/stop, charging profiles
- **MQTT** — IoT telemetry pub/sub with topic hierarchy (`vpp/{site}/{type}/{id}/{metric}`)
- **Modbus TCP/RTU** — inverter control with pre-built register maps (SMA, Fronius, SolarEdge)
- **IEEE 2030.5** — Smart Energy Profile, DER program control

### Vehicle-to-Grid (V2G)
- **Smart scheduling** — TOU-aware and solar-priority charge/discharge
- **Fleet aggregation** — aggregate flexibility windows for ancillary services bidding
- **Departure SOC guarantees** — constraint-based scheduling respects driver needs
- **Grid-forming inverters** — droop control, virtual synchronous machine (VSM), virtual inertia
- **Microgrid islanding** — fault detection, seamless island/reconnect transitions, load priority shedding

### Production Infrastructure
- **FastAPI REST + WebSocket API** with automatic OpenAPI docs
- **JWT authentication** with role-based access control and API key support
- **SQLAlchemy 2.0 async** database layer (SQLite dev, PostgreSQL prod)
- **Pydantic v2 schemas** for request/response validation
- **Docker Compose** deployment with PostgreSQL and Redis
- **CI/CD pipelines** — GitHub Actions for lint, test, security scan, Docker build, PyPI release

### Monitoring & Observability
- **Prometheus metrics** — resource gauges, optimization histograms, trading counters
- **Grafana dashboards** — VPP overview, trading performance, fleet capacity
- **Structured logging** — JSON (production) / colored (development) via structlog
- **Alert engine** — threshold, rate-of-change, and Z-score anomaly rules with webhook/log/WS channels

### Research / AI Layer *(non-production, clearly separated)*
- **Gaussian Process forecasting** for load, price, and renewable prediction
- **PPO reinforcement learning** for dispatch optimization (shadows rule-based, never controls)
- **Federated learning** for privacy-preserving multi-site model training
- **Digital twin simulation** for what-if scenario analysis
- **Experiment runner** with seed management and reproducible comparison tables

### Benchmarking Suite
- **4 synthetic datasets**: IEEE residential, California ISO, EU multi-zone grid, 50-vehicle EV fleet
- **7 scenarios**: peak shaving, frequency response, V2G arbitrage, multi-site coordination, islanding, high renewable, multi-market trading
- **13 standardized metrics**: peak reduction, self-consumption, battery cycles, Sharpe ratio, max drawdown, CO2 reduction, uptime, and more
- **Benchmark runner** with method comparison, statistical analysis, and markdown report generation

---

## Architecture

```
                          ┌─────────────────────────────┐
                          │     FastAPI REST + WS API    │
                          │   JWT Auth · Pydantic v2     │
                          └─────────────┬───────────────┘
                                        │
              ┌─────────────────────────┼─────────────────────────┐
              │                         │                         │
   ┌──────────▼──────────┐  ┌──────────▼──────────┐  ┌──────────▼──────────┐
   │   Optimization      │  │      Trading        │  │     Resources       │
   │                      │  │                      │  │                      │
   │ Stochastic · MPC     │  │ Day-Ahead · RT      │  │ Battery · Solar     │
   │ Real-Time · ADMM     │  │ Ancillary · Bilateral│  │ Wind · EV Fleet    │
   │ Plugin Architecture  │  │ Risk Management     │  │ Physics Models      │
   └──────────┬──────────┘  └──────────┬──────────┘  └──────────┬──────────┘
              │                         │                         │
              └─────────────────────────┼─────────────────────────┘
                                        │
   ┌────────────────────────────────────┼────────────────────────────────────┐
   │                                    │                                    │
   │  ┌─────────┐ ┌──────┐ ┌──────┐ ┌──▼───┐  ┌───────────┐ ┌───────────┐ │
   │  │ OpenADR │ │ OCPP │ │ MQTT │ │Modbus│  │    V2G    │ │  Grid     │ │
   │  │  2.0b   │ │ 1.6  │ │      │ │TCP/RTU│ │ Scheduler │ │ Forming   │ │
   │  └─────────┘ └──────┘ └──────┘ └──────┘  │ Aggregator│ │ Microgrid │ │
   │         Protocol Layer                     └───────────┘ └───────────┘ │
   └────────────────────────────────────────────────────────────────────────┘
                                        │
              ┌─────────────────────────┼─────────────────────────┐
              │                         │                         │
   ┌──────────▼──────────┐  ┌──────────▼──────────┐  ┌──────────▼──────────┐
   │   Monitoring        │  │     Research (AI)    │  │    Benchmarks       │
   │                      │  │   (non-production)   │  │                      │
   │ Prometheus · Grafana │  │ GP · RL · Federated │  │ 4 Datasets          │
   │ Alerts · Logging    │  │ Digital Twin         │  │ 7 Scenarios         │
   └─────────────────────┘  └─────────────────────┘  │ 13 Metrics          │
                                                       └─────────────────────┘
```

---

## Quick Start

### Installation

```bash
git clone https://github.com/vinerya/virtual-power-plant.git
cd virtual-power-plant
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### Run Tests

```bash
pytest tests/ -v    # 359 tests
```

### Run a Demo

```bash
# 10-home residential VPP with solar + battery
python -c "from demos.residential_demo import run; run()"

# 50-vehicle V2G parking garage
python -c "from demos.ev_fleet_demo import run; run()"

# Grid fault → island transition → reconnection
python -c "from demos.microgrid_demo import run; run()"

# Multi-market arbitrage trading bot
python -c "from demos.trading_demo import run; run()"

# OpenADR + OCPP + MQTT + Modbus coordination
python -c "from demos.protocols_demo import run; run()"

# Terminal dashboard with live ASCII visualization
python -c "from demos.dashboard_demo import run; run()"
```

### Stochastic Optimization

```python
from vpp.optimization import create_stochastic_problem, solve_with_fallback

base_data = {
    "base_prices": [0.1, 0.15, 0.12, 0.08] * 6,
    "renewable_forecast": [100, 150, 200, 120] * 6,
    "battery_capacity": 2000.0,
    "max_power": 500.0,
}

problem = create_stochastic_problem(base_data, num_scenarios=20)
result = solve_with_fallback(problem, timeout_ms=1000)
print(f"Status: {result.status}, Cost: ${result.objective_value:.2f}")
```

### V2G Smart Scheduling

```python
from vpp.v2g import EVFleet, SmartChargingScheduler, TOU_PRIORITY

fleet = EVFleet()
fleet.add_vehicle(capacity_kwh=60, soc=0.3, departure_soc=0.8,
                  max_charge_kw=11, v2g_capable=True)

scheduler = SmartChargingScheduler(strategy=TOU_PRIORITY)
schedule = scheduler.create_schedule(fleet, price_forecast=prices)
```

### Start the API Server

```bash
# Development
docker-compose -f docker-compose.dev.yml up

# Production
docker-compose up -d
```

### Run Benchmarks

```python
from benchmarks import BenchmarkRunner, ScenarioRegistry, DatasetRegistry

runner = BenchmarkRunner()
scenario = ScenarioRegistry.get("PEAK_SHAVING")
result = runner.run(scenario, seed=42)
print(runner.generate_report())
```

---

## CLI Reference

| Command | Description |
|---------|-------------|
| `vpp serve` | Start the FastAPI server |
| `vpp init` | Initialize database and configuration |
| `vpp config show` | Display current configuration |
| `vpp config validate` | Validate configuration file |
| `vpp dispatch <target>` | Run optimization dispatch |
| `vpp status` | Show VPP system status |
| `vpp benchmark list` | List available datasets and scenarios |
| `vpp benchmark run <SCENARIO>` | Run benchmark with all methods |
| `vpp benchmark report` | Generate comparison report |
| `vpp demo [name]` | Run a demo (residential, ev_fleet, microgrid, trading, protocols, dashboard) |

---

## Demo Applications

| Demo | What it shows |
|------|---------------|
| **Residential VPP** | 10 homes with solar + battery, peak shaving optimization, cost savings analysis |
| **EV Fleet V2G** | 50-vehicle parking garage, smart vs. dumb charging, V2G revenue optimization |
| **Microgrid Islanding** | Grid fault detection, seamless island transition, grid-forming inverters, reconnection |
| **Trading Bot** | Multi-market arbitrage (DA/RT/ancillary), P&L tracking, Sharpe ratio, risk controls |
| **Multi-Protocol** | OpenADR DR events + OCPP charger control + MQTT telemetry + Modbus inverters in concert |
| **Dashboard** | Terminal UI with ASCII progress bars, live resource/grid/trading panels, alert feed |

---

## Benchmarking Suite

### Datasets

| Dataset | Description | Resolution |
|---------|-------------|------------|
| IEEE Test Case | 24h residential DER (load, solar, price, temperature) | 15 min (96 steps) |
| California ISO | 7-day grid-scale with duck curve, LMP, regulation prices | 15 min (672 steps) |
| EU Grid Data | 48h European multi-zone (DA/intraday/balancing prices, cross-border) | 15 min (192 steps) |
| EV Fleet | 50-vehicle fleet with arrival/departure patterns, SOC, V2G capability | Per-vehicle |

### Scenarios

Peak Shaving · Frequency Response · V2G Arbitrage · Multi-Site Coordination · Islanding · High Renewable Penetration · Multi-Market Trading

### Metrics

Peak Reduction · Total Cost · Self-Consumption · Renewable Utilization · Battery Cycles · Frequency RMSE · V2G Utilization · Departure SOC Compliance · Curtailment · Sharpe Ratio · Max Drawdown · CO2 Reduction · Uptime

---

## REST + WebSocket API

The FastAPI server auto-generates interactive docs at `/docs` (Swagger UI) and `/redoc`.

### Key Endpoints

| Group | Endpoints |
|-------|-----------|
| **Health** | `GET /health`, `GET /ready`, `GET /version` |
| **Auth** | `POST /auth/token`, `POST /auth/register`, `GET /auth/me` |
| **Resources** | `GET/POST /api/v1/resources`, `GET/PUT/DELETE /api/v1/resources/{id}` |
| **Optimization** | `POST /dispatch`, `POST /stochastic`, `POST /realtime`, `POST /distributed` |
| **Trading** | `POST/GET /orders`, `GET /portfolio`, `GET /trades`, `GET /markets` |
| **Protocols** | `GET /protocols`, `POST /protocols/{name}/connect` |
| **V2G** | `POST /v2g/vehicles`, `GET /v2g/fleet`, `POST /v2g/schedule` |
| **Metrics** | `GET /metrics` (Prometheus format) |

### WebSocket Channels

Connect to `/ws` and subscribe to: `resource_updates`, `optimization_events`, `market_data`, `alerts`

---

## Docker Deployment

```bash
# Full stack: API + PostgreSQL + Redis
docker-compose up -d

# With monitoring: + Prometheus + Grafana
docker-compose -f docker-compose.yml -f monitoring/docker-compose.monitoring.yml up -d

# Development mode with live reload
docker-compose -f docker-compose.dev.yml up
```

---

## Project Structure

```
virtual-power-plant/
├── src/vpp/
│   ├── api/                 # FastAPI REST + WebSocket
│   │   ├── routes/          # Resource, optimization, trading, auth, protocol, V2G endpoints
│   │   └── websocket.py     # Real-time WebSocket manager
│   ├── auth/                # JWT authentication + RBAC
│   ├── cli/                 # Click CLI (vpp serve, dispatch, benchmark, demo)
│   ├── db/                  # SQLAlchemy 2.0 async models + repositories
│   ├── events/              # Typed event bus with async publish
│   ├── grid/                # Grid-forming inverters, microgrid controller
│   ├── optimization/        # Stochastic, real-time, distributed, MPC, ADMM, plugins
│   ├── protocols/           # OpenADR, OCPP, MQTT, Modbus, IEEE 2030.5
│   ├── research/            # GP forecasting, RL dispatch, federated learning, digital twin
│   ├── schemas/             # Pydantic v2 request/response models
│   ├── trading/             # Markets, orders, strategies, portfolio, risk
│   ├── v2g/                 # EV models, smart scheduler, fleet aggregator
│   ├── alerts.py            # Alert rules + manager
│   ├── metrics.py           # Prometheus metrics collector
│   ├── settings.py          # Pydantic BaseSettings (.env)
│   └── resources.py         # Battery, Solar, Wind physics models
├── benchmarks/              # Datasets, scenarios, metrics, runner
├── demos/                   # 6 interactive demo applications
├── tests/                   # 359 tests (pytest)
├── monitoring/              # Prometheus + Grafana configs
├── Dockerfile               # Multi-stage production build
├── docker-compose.yml       # Production deployment
├── docker-compose.dev.yml   # Development with live reload
└── pyproject.toml           # PEP 621 metadata + tool configs
```

---

## Testing

```bash
# Run all 359 tests
pytest tests/ -v

# Run specific test modules
pytest tests/test_protocols.py -v    # Protocol adapters
pytest tests/test_v2g.py -v          # V2G scheduling
pytest tests/test_grid.py -v         # Grid-forming inverters
pytest tests/test_benchmarks.py -v   # Benchmarking suite
pytest tests/test_demos.py -v        # Demo applications
pytest tests/test_trading.py -v      # Trading system (via test_api_trading.py)
```

---

## Contributing

We welcome contributions. This project aims to become the **industry standard** for open-source VPP platforms.

### Priority Areas
- New resource types (fuel cells, pumped hydro, thermal storage)
- Additional protocol adapters (SunSpec, DNP3, IEC 61850)
- Real grid data integration (ISO APIs, utility feeds)
- Advanced optimization algorithms
- Documentation, tutorials, and case studies

### Development Setup

```bash
git clone https://github.com/vinerya/virtual-power-plant.git
cd virtual-power-plant
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
pytest tests/ -v
```

### Guidelines
- Follow PEP 8, use type hints, add docstrings
- Maintain test coverage — add tests for all new features
- Rule-based logic for production; AI/ML in `research/` only
- Open a PR with a clear description and test plan

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Made with care by [Moudather Chelbi](https://www.linkedin.com/in/moudatherchelbi/) & [Mariem Khemir](https://www.linkedin.com/in/mariem-khemir/)

[Star this repo](https://github.com/vinerya/virtual-power-plant) · [Report issues](https://github.com/vinerya/virtual-power-plant/issues) · [Request features](https://github.com/vinerya/virtual-power-plant/issues/new)

</div>
