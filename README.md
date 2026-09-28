<div align="center">

# Virtual Power Plant Platform

An open-source platform for aggregating and dispatching distributed energy
resources: batteries, PV, wind and EV chargers. FastAPI backend, Next.js
operator console, OCPP / OpenADR / IEEE 2030.5 integrations, and
transparent optimization you can read, test and change.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: beta](https://img.shields.io/badge/status-beta-orange.svg)](#what-works-today)

[What works](#what-works-today) · [Quick start](#quick-start) · [Architecture](#architecture) · [API](#api-overview) · [Security](#security-model) · [Docs](docs/) · [Contributing](CONTRIBUTING.md)

</div>

---

## What it is

Commercial VPP software is mostly closed. This project is an open,
self-hostable alternative for startups, researchers, utilities' innovation
teams and anyone who wants to see exactly how a dispatch decision was made.

What you get:

- a **REST + WebSocket API** that stores your fleet (resources, sites,
  customers, telemetry) in PostgreSQL or SQLite, with JWT / API-key auth
  and role-based access;
- **dispatch optimization** (Pyomo + HiGHS with rule-based fallback),
  horizon **MPC schedules** and **closed-loop backtests**, every run
  persisted and explainable;
- **grid and device protocols**: an OCPP 1.6-J Central System, an OpenADR
  2.0b VEN, an IEEE 2030.5 client, MQTT and Modbus telemetry ingestion,
  and a **demand-response orchestrator** that turns grid signals into fleet
  dispatch (off by default);
- **V2G**: persisted vehicles bound to chargers, schedules pushed as OCPP
  charging profiles with honest per-charger delivery results;
- **tariffs**: URDB-based bill engine with TOU, tiers, demand charges, NEM
  2.0 / NEM 3.0 / net billing, bill simulation from CSV or synthetic load;
- a **trading engine** with risk checks and a portfolio, running on a
  **simulated venue** (no real market connection);
- a **web console** for operators and a **customer portal**;
- Prometheus metrics, Grafana dashboards, persisted alerts with signed
  webhooks.

The project is **beta**. It has a substantial automated test suite, but it
has not been certified against any protocol conformance suite or proven in
a field deployment. Read [What works today](#what-works-today) before
connecting it to equipment.

## What works today

Maturity labels used below:

- **production-grade** — complete for its stated scope, covered by tests,
  safe defaults. Still review it for your deployment.
- **beta** — works end to end and is tested, with known gaps documented in
  [docs/](docs/). Not proven against real hardware or third-party peers.
- **simulated** — runs against an in-process simulation only; nothing real
  is connected.
- **research** — library code for experiments; not wired into the
  operational API (or only operating on request-supplied data).

| Area | What it does | Maturity |
|---|---|---|
| Database & migrations | SQLAlchemy 2.0 async, PostgreSQL/SQLite, alembic migrations 0001-0007, model/migration drift test | production-grade |
| Observability | `/metrics` (Prometheus), request ids, structured JSON logs, provisioned Grafana dashboards | production-grade |
| Auth & RBAC | JWT with `aud` claim, API keys, roles admin/operator/viewer/researcher/customer, deny-by-default for customers, authenticated WebSocket | beta — no user-management endpoints beyond register (see [security](docs/security.md)) |
| Resources, sites, telemetry | typed battery/solar/wind resources, sites with live aggregates, telemetry and meter-reading ingest, time-bucketed history | beta |
| Dispatch optimization | single-interval allocation LP (Pyomo + HiGHS) over DB resources with SOC/energy limits and SOH-aware wear cost; proportional fallback; run history + explainer | beta — computes and records; does not command devices by itself |
| MPC schedule & backtest | horizon MPC from prices or a stored tariff; closed-loop backtest vs idle / rule-based / perfect foresight | beta |
| Stochastic, real-time, ADMM endpoints | CVaR scenario optimization, fast rules, multi-site ADMM on request-supplied data | research |
| Battery degradation | rainflow + calendar ageing SOH, periodic updater, wear cost for the optimizer | beta |
| Tariffs & billing | URDB parsing, TOU / tiers / demand / fixed / minimum / taxes, NEM 2.0 / 3.0 / net billing, billing cycles, CSV and synthetic load, presets, OpenEI import | beta |
| Customer portal | customer accounts, own devices, bill from own meter data, DR program enrollment | beta |
| Alerts | rules on live telemetry, persisted alerts, ack/snooze/resolve, HMAC-signed webhooks | beta |
| OCPP 1.6-J Central System | Boot/Heartbeat/Status/Authorize/Start/Stop/MeterValues; RemoteStart/Stop, Set/ClearChargingProfile; Security Profile 1 | beta |
| OpenADR 2.0b VEN | pull-mode registration, polling, event parsing, opt-in/out, mTLS | beta |
| IEEE 2030.5 client | mTLS resource-tree walk to active DER controls | beta |
| DR orchestrator | OpenADR / 2030.5 (incl. DefaultDERControl, Response posting) → fleet target → dispatch → EV and device setpoints, with caps and audit | beta — **auto-response off by default** |
| Device control | dispatch allocations → Modbus setpoints (generic register, SunSpec 123; SunSpec 124 unverified) with clamping, deadband, rate limit, read-back and fallback watchdog | beta — **off by default** (`VPP_CONTROL_ENABLED`), per-device opt-in |
| V2G | persisted vehicles, charger binding, schedules / dispatch via OCPP profiles | beta — discharge uses a vendor extension (negative limits) |
| MQTT / Modbus ingestion | telemetry in from brokers and inverters/meters | beta |
| Trading | order types incl. stop-limit/iceberg/IOC/FOK, pre-trade risk, portfolio, VaR, strategies and backtests | **simulated** venue |
| Protocol adapters without an endpoint | in-memory state machines, status `simulated` | simulated |
| Grid-forming inverters, microgrid islanding | models and demos | simulated |
| Benchmarks | synthetic datasets, scenarios, metrics, runner | research |
| Forecasting, anomaly detection | `vpp.research` | research |
| Web console | operator console + customer portal (see [below](#web-console)) | beta |

## Architecture

```
                       Browser (operators, customers)
                        |                         \
                  HTTPS |                          \ WSS /api/v1/ws
                        v                           \ (60 s socket token)
        +-------------------------------+            \
        | Web console (Next.js 15)      |             \
        | operator UI + customer portal |              \
        | httpOnly session, /api/proxy  |               \
        +---------------+---------------+                \
                        | HTTP + Bearer JWT (server side)   \
                        v                                     v
+------------------------------------------------------------------------------+
| FastAPI API (single process)                                                 |
|  auth (JWT aud / API keys, RBAC) · rate limit · request ids · /metrics       |
|                                                                              |
|  resources · sites · customers/portal · tariffs · alerts · config · V2G      |
|                                                                              |
|  +----------------+  +-----------------+  +----------------+  +------------+ |
|  | Optimization   |  | Trading         |  | Tariff engine  |  | Alerts     | |
|  | Pyomo/HiGHS LP |  | SIMULATED venue |  | URDB, NEM,     |  | rules,     | |
|  | MPC, backtest, |  | risk, portfolio |  | bill cycles    |  | webhooks   | |
|  | fallbacks      |  | strategies      |  |                |  |            | |
|  +-------+--------+  +--------+--------+  +----------------+  +------------+ |
|          ^                    |                                              |
|  +-------+---------------------------------------+                           |
|  | DR orchestrator: grid signal -> target ->       |                           |
|  | dispatch -> EV setpoints (auto-response OFF)    |                           |
|  +-------+-----------------------------+-----------+                           |
|          |                             |                                     |
|  EventBus -> WebSocket channels, Prometheus metrics, alert service           |
|          |                             |                                     |
|  Protocol registry (each adapter LIVE or SIMULATED)                          |
|   OCPP 1.6-J Central System  <-- chargers (wss /ocpp/{id}), V2G bridge        |
|   OpenADR 2.0b VEN           --> utility VTN (HTTPS pull, mTLS)              |
|   IEEE 2030.5 client         --> utility server (mTLS)                       |
|   MQTT / Modbus ingestion    --> broker / inverters (telemetry)              |
+-----------------------------------+------------------------------------------+
                                    |
                     PostgreSQL / SQLite (alembic migrations)

   Prometheus --> Grafana (:3001)          Research & benchmarks (offline)
```

More in [docs/architecture.md](docs/architecture.md).

## Quick start

### Docker Compose (API + PostgreSQL + console)

```bash
git clone https://github.com/vinerya/virtual-power-plant.git
cd virtual-power-plant
export VPP_SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(48))")
export POSTGRES_PASSWORD=$(python3 -c "import secrets; print(secrets.token_urlsafe(24))")
docker compose up -d --build
```

The API runs migrations on start. Create the first admin (there is no
self-registration) with the snippet in
[docs/deployment.md](docs/deployment.md#create-the-first-admin), then open
the console at <http://localhost:3000> and the API docs at
<http://localhost:8000/docs>.

Add Prometheus (:9090) and Grafana (:3001):

```bash
docker compose -f docker-compose.yml -f monitoring/docker-compose.monitoring.yml up -d
```

### Backend without Docker

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[api,db,protocols,solver,degradation,monitoring,cli,dev]"

export VPP_SECRET_KEY=dev-only-secret      # SQLite ./vpp.db by default
vpp migrate                                # from the repository root
# create the first admin: docs/deployment.md#create-the-first-admin
uvicorn vpp.api.app:create_app --factory --reload
```

Try it:

```bash
TOKEN=$(curl -s -X POST localhost:8000/api/v1/auth/token \
  -d username=admin -d password='change-me-now' | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

curl -s -X POST localhost:8000/api/v1/resources -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"name": "home-battery-1", "resource_type": "battery", "rated_power": 5,
       "capacity_kwh": 13.5, "state_of_charge": 0.6, "chemistry": "lfp"}'

# Split a 3 kW export target across online resources (computes and records;
# nothing is sent to devices)
curl -s -X POST localhost:8000/api/v1/optimization/dispatch -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' -d '{"target_power_kw": 3}'
```

### Web console

```bash
cd web
cp .env.example .env.local        # API_BASE_URL=http://localhost:8000
pnpm install
pnpm dev                          # http://localhost:3000
```

See [web/README.md](web/README.md). `NEXT_PUBLIC_USE_MOCKS=1` runs the
console with bundled demo data and no backend.

### Demos and benchmarks (no server needed)

```bash
# from the repository root; PYTHONPATH=. makes the top-level demos/ and
# benchmarks/ packages importable for the `vpp` command
PYTHONPATH=. vpp demo                    # list: residential, ev_fleet, microgrid, trading, protocols, dashboard
PYTHONPATH=. vpp demo residential
PYTHONPATH=. vpp benchmark list
PYTHONPATH=. vpp benchmark run PEAK_SHAVING
```

Demos and benchmarks run on synthetic data.

## Configuration

All API settings are `VPP_*` environment variables (or a `.env` file);
every one is listed with its default in
[docs/configuration.md](docs/configuration.md), and
[.env.example](.env.example) is a ready template. The ones you must set for
anything beyond local development: `VPP_SECRET_KEY`, `VPP_DATABASE_URL`,
`VPP_CORS_ORIGINS`. Protocols and DR auto-response are opt-in.

## API overview

Interactive docs at `/docs`, schema at `/openapi.json`. Full route list with
required roles: [docs/api.md](docs/api.md#routes).

| Group | Routes |
|---|---|
| Health | `GET /health`, `/ready`, `/version`; `GET /metrics` |
| Auth | `POST /api/v1/auth/token` (form or JSON body), `POST /api/v1/auth/register` (admin), `GET /api/v1/auth/me`, `POST /api/v1/auth/api-key` |
| Resources | `GET/POST /api/v1/resources`, `GET/PUT/DELETE /api/v1/resources/{id}`, `GET .../{id}/metrics`, `POST .../{id}/telemetry` |
| Sites | `GET/POST /api/v1/sites`, `GET/PATCH/DELETE /api/v1/sites/{id}`, `GET/POST .../{id}/meter-readings` |
| Customers | portal `GET /api/v1/customer/me`, `/me/bill`, `/me/devices`, `/programs`, `POST/DELETE /enrollments`; staff `/api/v1/customers`, `/api/v1/programs` |
| Optimization | `POST /api/v1/optimization/dispatch`, `/schedule`, `/backtest`, `/stochastic`, `/realtime`, `/distributed`; `GET /runs`, `/runs/{id}`, `/runs/{id}/explain`, `/history`, `/stats`; `GET /api/v1/dispatches[/{id}[/explain]]` |
| Trading (simulated) | `POST/GET /api/v1/trading/orders`, `GET/DELETE /orders/{id}`, `POST /orders/{id}/cancel`, `GET /trades`, `/portfolio`, `/markets`, `POST /markets/tick`, `GET /strategies`, `POST /strategies/{name}/backtest`, `/run` |
| Tariffs | `GET/POST /api/v1/tariffs`, `GET/PUT/DELETE /api/v1/tariffs/{id}`, `POST /{id}/simulate`, `POST /simulate`, `GET /presets[/{id}]`, `GET/POST /import-urdb` |
| V2G | `GET/POST /api/v1/v2g/vehicles`, `GET/PATCH/DELETE /vehicles/{ev_id}`, `PUT/DELETE /vehicles/{ev_id}/binding`, `GET /sessions`, `/fleet`, `/flexibility`, `POST /schedule`, `GET /schedules`, `POST /dispatch`, `POST /bid`, `GET /bids`, `/metrics` |
| Protocols | `GET /api/v1/protocols`, `POST /{name}/connect\|disconnect`, `GET /{name}/metrics`; OCPP charge points, transactions, remote start/stop; OpenADR events + opt override; IEEE 2030.5 controls |
| Demand response | `GET /api/v1/dr/status`, `GET /api/v1/dr/responses` |
| Alerts | `GET /api/v1/alerts[/{id}]`, `POST /{id}/ack\|snooze\|resolve`, rules CRUD under `/api/v1/alerts/rules` |
| Config | `GET/PUT /api/v1/config`, `GET /schema`, `POST /validate` |
| Degradation | `GET /api/v1/batteries/{id}/soh[/history]`, `POST .../soh/update` |
| WebSocket | `POST /api/v1/ws/token`; `WS /api/v1/ws` (alias `/ws`) |
| OCPP | `WS /ocpp/{charge_point_id}` (subprotocol `ocpp1.6`, when enabled) |

### WebSocket channels

Connect to `/api/v1/ws` with a token (see [docs/api.md](docs/api.md#websocket)),
then subscribe via `?channels=...` or `{"action": "subscribe", "channel": ...}`:

| Channel | Events |
|---|---|
| `resource_updates` | resource added/removed/updated/fault, telemetry, OCPP meter values |
| `optimization_events` | optimization started/completed/failed, dispatch executed |
| `market_data` | simulated quotes, order and trade events |
| `alerts` | newly fired alerts only |
| `grid_events` | protocol status, DR events and responses, EV plug-in/out, V2G dispatch, islanding |
| `system` | everything else |
| `*` | all channels |

Close codes: `1008` handshake refused (missing/invalid token, customer
account), `4001` the session behind the socket expired — reconnect with a
fresh token.

## Web console

| Page | What it shows |
|---|---|
| `/` | fleet overview (count, online, rated and current kW, resource table) |
| `/assets`, `/assets/[id]` | resource detail with stored power/SOC history |
| `/sites` | map of sites with health |
| `/alerts` | live alert feed, ack / snooze, bulk ack |
| `/optimization` | schedule planner and closed-loop backtest |
| `/trading`, `/trading/dispatches`, `/trading/strategies` | simulated-venue workspace, dispatch history + explainer, strategy backtests |
| `/tariffs`, `/tariffs/[id]` | tariff browser, TOU heatmaps, bill simulator, URDB import |
| `/protocols` | adapters with LIVE / SIMULATED badges, connect/disconnect |
| `/settings` | platform config YAML editor with schema validation |
| `/portal/*` | customer portal: overview, bill, devices, program enrollment |

Viewers get read-only pages; the backend enforces every permission
regardless of what the UI shows.

## Security model

- **Roles**: `admin`, `operator`, `viewer`, `researcher` (currently the
  same as viewer) and `customer`. Customers are refused on every operator
  endpoint and on the WebSocket; customer routes only return the caller's
  own data. Writes that change device or venue state need operator or
  admin.
- **JWT**: HS256 signed with `VPP_SECRET_KEY`, `aud` = `operator` or
  `customer`, re-checked against the user's current role on each request.
  Credentials go in the request body (query-string login is deprecated).
- **API keys**: `X-API-Key`, stored hashed. A key currently acts with its
  creator's full role — give integrations their own low-privilege users.
- **WebSocket**: authentication required by default; browsers use a 60 s
  socket-only token passed as a subprotocol, and sockets close with `4001`
  when the session expires.
- **Grid safety**: DR auto-response is off by default; utility limits and
  operator caps clamp every DR target; protocols are opt-in and report
  SIMULATED unless really connected.

Details, including what is *not* implemented (MFA, lockout, user
management, key revocation): [docs/security.md](docs/security.md).

## Testing

```bash
pip install -e ".[api,db,protocols,solver,degradation,monitoring,cli,dev]"
pytest                                   # 800+ backend tests
pytest tests/test_alembic_drift.py       # models vs migrations
ruff check src tests && mypy src/vpp

cd web
pnpm lint && pnpm typecheck && pnpm build
pnpm exec playwright install chromium
pnpm test:e2e                            # Playwright, backend stubbed
```

Protocol tests run against in-process mock peers (no certification suites).

## Project structure

```
├── src/vpp/
│   ├── api/            FastAPI app, routes, WebSocket, middleware, observability
│   ├── auth/           JWT / API keys, RBAC, rate limiting
│   ├── db/             SQLAlchemy models, engine, repositories
│   ├── schemas/        Pydantic request/response models
│   ├── optimization/   allocation LP, MPC, backtest, stochastic/real-time/ADMM, fallbacks
│   ├── degradation/    battery SOH model and updater
│   ├── trading/        markets, orders, portfolio, risk, strategies, simulated venue
│   ├── tariffs/        URDB engine, NEM, billing cycles, CSV/synthetic load, presets, feeds
│   ├── protocols/      OCPP 1.6-J, OpenADR 2.0b, IEEE 2030.5, MQTT, Modbus, bootstrap
│   ├── v2g/            EV fleet models, scheduler, store, OCPP bridge
│   ├── dr/             demand-response translation rules and orchestrator
│   ├── portal/         sites, customer scoping, customer billing, telemetry history
│   ├── grid/           grid-forming inverter and microgrid models
│   ├── research/       forecasting, anomaly detection, experiment runner
│   ├── config/         platform configuration document and schema
│   ├── events/         in-process EventBus
│   ├── cli/            `vpp` command
│   └── settings.py     VPP_* settings
├── web/                Next.js console + customer portal (Playwright tests in web/tests)
├── alembic/            database migrations
├── monitoring/         Prometheus + Grafana (compose overlay, dashboards)
├── benchmarks/         synthetic datasets, scenarios, metrics, runner
├── demos/, examples/   runnable scripts
├── docs/               architecture, configuration, deployment, protocols, API, tariffs, security
├── tests/              pytest suite
├── Dockerfile, web/Dockerfile, docker-compose*.yml
└── pyproject.toml
```

## Documentation

- [Architecture](docs/architecture.md)
- [Configuration reference](docs/configuration.md)
- [Deployment](docs/deployment.md)
- [Protocols, V2G and DR](docs/protocols.md)
- [API guide](docs/api.md)
- [Tariffs and billing](docs/tariffs.md)
- [Security](docs/security.md)
- [Library features](ADVANCED_FEATURES.md) · [Changelog](CHANGELOG.md) · [Contributing](CONTRIBUTING.md)

## Contributing

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). Useful
areas: field testing against real chargers/VTNs/2030.5 servers and
inverters/batteries (vendor setpoint profiles), user management, additional protocols
(SunSpec, DNP3, IEC 61850), and real market/ISO integrations.

## License

MIT — see [LICENSE](LICENSE).

<div align="center">

Made by [Moudather Chelbi](https://www.linkedin.com/in/moudatherchelbi/) & [Mariem Khemir](https://www.linkedin.com/in/mariem-khemir/)

[Report issues](https://github.com/vinerya/virtual-power-plant/issues) · [Request features](https://github.com/vinerya/virtual-power-plant/issues/new)

</div>
