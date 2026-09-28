# Architecture

The platform is a single FastAPI process (`vpp.api.app:create_app`) backed by
a SQL database, with a Next.js operator console in front of it and
optional connections to grid peers and devices. Everything that must
survive a restart is in the database; a few things are deliberately
per-process (see [Process model](#process-model)).

## Components

```
                    Browser (operators, customers)
                      |                       \
             HTTPS    |                        \  WSS  /api/v1/ws
                      v                         \  (short-lived socket token)
     +--------------------------------+          \
     |  web console (Next.js 15)      |           \
     |  operator UI + customer portal |            \
     |  httpOnly session cookie       |             \
     |  /api/proxy/* -> FastAPI       |              \
     +---------------+----------------+               \
                     | HTTP (Bearer JWT, server side)   \
                     v                                    v
+-----------------------------------------------------------------------------+
| FastAPI API (one process)                                                   |
|  middleware: request id + access log, Prometheus, rate limit, CORS, slash   |
|  auth: JWT (aud operator|customer) / API keys, RBAC                          |
|                                                                             |
|  routes: resources, sites, customers & portal, optimization, trading,      |
|          tariffs, V2G, protocols, DR, alerts, config, degradation, ws token |
|                                                                             |
|  +-------------+  +--------------+  +-------------+  +-------------------+  |
|  | optimization|  | trading      |  | tariffs     |  | alerts service    |  |
|  | Pyomo/HiGHS |  | simulated    |  | URDB engine |  | rules on telemetry|  |
|  | + fallbacks |  | venue, risk, |  | NEM, cycles |  | webhook (HMAC)    |  |
|  | MPC, backtest| | portfolio    |  +-------------+  +-------------------+  |
|  +------+------+  +------+-------+                                          |
|         |                |          +-----------------------------------+  |
|         |                |          | DR orchestrator (vpp.dr)          |  |
|         |                |          | signals -> target -> dispatch     |  |
|         |                |          +----+---------------------+--------+  |
|         v                v               |                     |           |
|  +---------------------------------------+---------------------+--------+  |
|  | EventBus (in-process pub/sub)  -> WebSocket channels, metrics, alerts |  |
|  +------------------------------------------------------------------------+ |
|                                                                             |
|  protocol registry (LIVE or SIMULATED adapters):                            |
|   OCPP 1.6-J Central System <- chargers  (/ocpp/{cp}, V2G bridge)           |
|   OpenADR 2.0b VEN          -> utility VTN (HTTPS pull)                     |
|   IEEE 2030.5 client        -> utility server (mTLS)                        |
|   MQTT / Modbus ingestion   -> broker / devices (telemetry in)              |
+-----------------------------------+-----------------------------------------+
                                    |
                     SQLAlchemy async (asyncpg / aiosqlite)
                                    v
                   PostgreSQL (production) or SQLite (development)
                   schema managed by alembic (migrations 0001-0007)

  /metrics -> Prometheus -> Grafana (:3001, provisioned dashboards)
```

## Backend packages (`src/vpp`)

| Package | Responsibility |
|---|---|
| `api/` | app factory and lifespan, routes, WebSocket manager, middleware, observability wiring, `optimization_support` (DB-backed dispatch used by the API and the DR orchestrator) |
| `auth/` | password hashing, JWT and API-key authentication, `require_role`, rate-limit middleware |
| `db/` | SQLAlchemy 2.0 async models, engine/session setup, repositories |
| `schemas/` | Pydantic v2 request/response models |
| `events/` | typed in-process EventBus |
| `optimization/` | allocation LP (Pyomo + HiGHS), MPC and multi-resource MPC, closed-loop backtest, stochastic CVaR plugin, real-time and ADMM distributed plugins, rule-based fallbacks |
| `degradation/` | battery SOH model (rainflow cycle counting + calendar ageing), wear cost for the optimizer, periodic updater |
| `trading/` | markets, order books, engine, portfolio, risk manager, strategies, backtests, `service` (API-facing simulated venue) and `simulation` (synthetic liquidity) |
| `tariffs/` | URDB parsing, bill engine, billing cycles, NEM, CSV and synthetic load, presets, tariff-to-optimizer hooks, price feeds (CAISO, ComEd, OpenADR price; library use) |
| `protocols/` | adapters (OCPP 1.6-J, OpenADR 2.0b, IEEE 2030.5, MQTT, Modbus), OCPP-J framing, bootstrap/supervision, telemetry ingestion |
| `v2g/` | EV and fleet models, scheduler, aggregator, persistent store, OCPP bridge |
| `dr/` | DR translation rules and orchestrator |
| `control/` | setpoint actuator: dispatch allocations -> device setpoints (Modbus writer in `protocols/modbus_control.py`), safety rules, fallback watchdog |
| `portal/` | sites aggregation, customer access scoping, customer billing, telemetry history |
| `grid/` | grid-forming inverter and microgrid models (simulation) |
| `research/` | forecasting, anomaly detection, experiment runner (not used by the API) |
| `config/` | platform configuration document (`VPPConfig`) and JSON Schema |
| `alerts.py`, `alert_service.py` | alert rules/manager and the service that evaluates them on telemetry |
| `metrics.py`, `logging.py` | Prometheus metrics, structlog configuration |
| `cli/` | `vpp` command (serve, init, migrate, dispatch, status, config, benchmark, mpc, demo) |

Outside the package: `benchmarks/` (datasets, scenarios, metrics, runner),
`demos/` and `examples/` (scripts), `alembic/` (migrations), `monitoring/`
(Prometheus and Grafana), `web/` (console).

## Key flows

### Telemetry in

MQTT messages, Modbus polls, OCPP `MeterValues` and
`POST /api/v1/resources/{id}/telemetry` update resource state and history
(`battery_states`, `resource_telemetry`) and publish `RESOURCE_UPDATED`.
That event feeds the WebSocket `resource_updates` channel, the Prometheus
gauges and the alert service.

### Dispatch

`POST /api/v1/optimization/dispatch` loads online resources from the
database, bounds each by rated power, SOC/energy limits and configured
charge/discharge limits, and solves a single-interval allocation LP with
Pyomo + HiGHS (battery wear cost from persisted SOH, renewables first). If
the solver is unavailable or fails, a headroom-proportional rule splits the
target. The run is stored in `optimization_runs` and
`OPTIMIZATION_*` events are published. By default the call **computes
and records**; with `"apply": true` the setpoint actuator (`control/`) writes
the stationary allocations to devices that opted in via
`metadata.modbus.control`, behind the `VPP_CONTROL_ENABLED` kill switch and a
fallback watchdog (see [protocols.md](protocols.md#device-control-setpoint-actuator)).

`POST /api/v1/optimization/schedule` plans a horizon (MPC) from prices or a
stored tariff; `POST /api/v1/optimization/backtest` replays MPC closed-loop
against idle, rule-based and perfect-foresight baselines. Every run can be
explained (`/runs/{id}/explain`).

### Demand response

The DR orchestrator watches OpenADR events and IEEE 2030.5 controls,
translates them into a fleet target with limits, and (only when
`VPP_DR_AUTO_RESPONSE_ENABLED`) runs the same DB-backed dispatch over
resources plus plugged-in V2G vehicles, pushes EV setpoints through OCPP,
hands stationary setpoints to the actuator and records the decision. See [protocols.md](protocols.md).

### Trading

Orders go through validation and pre-trade risk checks, then match against
a **simulated** venue with seeded synthetic liquidity. Orders, fills and
portfolio state are persisted and rebuilt on first use after a restart.
No order leaves the process; there is no connection to a real exchange or
ISO market.

### Configuration

`PUT /api/v1/config` (admin) validates a YAML document against the JSON
Schema from `GET /api/v1/config/schema` plus semantic checks, stores it as a
new version and applies it; on startup the newest stored document is
re-applied. Environment settings (`VPP_*`) are separate and documented in
[configuration.md](configuration.md).

## Process model

Run the API as **one** process. The following are per-process:

- WebSocket connections and subscriptions,
- the rate limiter's buckets,
- the simulated trading venue's in-memory books (persisted state is rebuilt
  from the database),
- protocol adapters, including live OCPP charge-point connections,
- the EventBus.

Scale up (bigger machine) rather than out; horizontal scaling would need a
shared event bus and connection routing that the platform does not provide.

## Web console (`web/`)

Next.js 15 App Router, TanStack Query, Tailwind/shadcn-style components,
Recharts, MapLibre. Two layouts:

- **operator** (`/`, `/assets`, `/alerts`, `/sites`, `/tariffs`,
  `/trading`, `/trading/dispatches`, `/trading/strategies`,
  `/optimization`, `/protocols`, `/settings`)
- **customer portal** (`/portal`, `/portal/bill`, `/portal/devices`,
  `/portal/enroll`)

The browser talks only to the Next.js server for HTTP (session in an
httpOnly cookie, forwarded as a bearer token by `/api/proxy/*`) and
directly to FastAPI for the WebSocket, using a short-lived socket token from
`/api/auth/ws-token`. See [web/README.md](../web/README.md).

## Database

Tables: `users`, `api_keys`, `resources`, `battery_states`,
`battery_soh_samples`, `resource_telemetry`, `sites`, `customer_profiles`,
`dr_programs`, `program_enrollments`, `meter_readings`,
`optimization_runs`, `orders`, `trades`, `tariffs`, `alert_rules`,
`alerts`, `config_documents`, `event_log`, `v2g_vehicles`,
`v2g_charging_sessions`, `v2g_schedules`, `dr_event_responses`.

Migrations live in `alembic/versions/` (0001-0007).
`tests/test_alembic_drift.py` upgrades a fresh database to head and fails if
the models and migrations differ, and checks there is a single head.
