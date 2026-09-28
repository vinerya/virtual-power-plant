# Architecture

The platform is a FastAPI application (`vpp.api.app:create_app`, one or
more worker processes) backed by a SQL database, with a Next.js operator
console in front of it and optional connections to grid peers and devices.
Everything that must survive a restart is in the database; singleton work
runs on one worker at a time and a few things stay per-process (see
[Process model](#process-model)).

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
| FastAPI API (1..N worker processes, see Process model)                      |
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
| `cluster/` | multi-worker coordination: DB leases for singleton work, calls forwarded to lease holders, WebSocket relay, startup topology checks |
| `portal/` | sites aggregation, customer access scoping, customer billing, telemetry history |
| `grid/` | grid-forming inverter and microgrid models (simulation) |
| `research/` | forecasting, anomaly detection, experiment runner (not used by the API) |
| `config/` | platform configuration document (`VPPConfig`) and JSON Schema |
| `alerts.py`, `alert_service.py` | alert rules/manager and the service that evaluates them on telemetry |
| `metrics.py`, `logging.py` | Prometheus metrics, structlog configuration |
| `cli/` | `vpp` command (serve, init, migrate, dispatch, status, config, benchmark, mpc, demo) |

Also in the package: `migrations/` (alembic environment and revisions, run
by `vpp migrate`), `benchmarks/` (datasets, scenarios, metrics, runner) and
`demos/` (`vpp demo`). Outside the package: `examples/` (scripts),
`monitoring/` (Prometheus and Grafana), `web/` (console). The repo-root
`demos/` and `benchmarks/` directories are deprecated import aliases for
`vpp.demos` / `vpp.benchmarks`.

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

The API can run as several worker processes (`VPP_API_WORKERS`, used by
`vpp serve`) against one database; PostgreSQL is recommended for more than
one. Coordination goes through the database (`vpp.cluster`):

- **Leases** (`cluster_leases`). Work that must happen once per deployment
  runs only in the process holding its lease: `trading-venue` (simulated
  venue + market-data tick), `degradation-updater`, `alert-evaluator`,
  `mqtt-ingestion`, `modbus-ingestion`, `protocol-adapters` (OCPP /
  OpenADR / IEEE 2030.5 and the DR orchestrator). Acquire/renew is one
  atomic `UPDATE ... WHERE holder = :me OR expires_at < :now`. Followers
  stay idle and take over once the lease expires
  (`VPP_CLUSTER_LEASE_TTL_SECONDS`), or immediately when the holder was a
  process on the same host that no longer exists. A lone process acquires
  every lease during startup, so a single worker behaves exactly as before.
- **Forwarded calls** (`cluster_calls`). Trading requests that touch the
  venue (submit/cancel orders, portfolio, markets, tick, strategy runs) are
  written to the table by the worker that received them and executed by
  the `trading-venue` holder, which writes the result back. If no holder
  claims a call within `VPP_CLUSTER_CALL_TIMEOUT_SECONDS` it is cancelled
  (never executed) and the client gets `503 leader_unavailable`; a claimed
  call that does not finish in time gives `504 leader_timeout` with the
  call id. OpenADR / IEEE 2030.5 / DR orchestrator requests
  (`/api/v1/protocols/openadr/*`, `/ieee2030_5/*`, `/api/v1/dr/status`,
  and the `openadr` / `ieee2030_5` entries of `/api/v1/protocols/`) are
  forwarded the same way to the `protocol-adapters` holder, whose adapters
  hold the VEN's events, programs, DERControls and poll state; reads give
  up after at most 3 s with `503 leader_unavailable`. In a single process
  (or on the holder) they are answered directly, with no database round
  trip. Telemetry events that arrive on a follower (e.g.
  `POST /resources/{id}/telemetry`) are forwarded the same way, without
  waiting, to the `alert-evaluator` holder; so are alert-rule reloads.
  A worker that takes over the venue rebuilds its books and portfolio from
  `orders`/`trades`; simulated prices restart from the base prices.
- **WebSocket relay** (`cluster_events`, only with `VPP_API_WORKERS > 1`).
  Every broadcast is also written to the table and re-sent by the other
  workers to their own clients, so a browser sees market data, fills and
  alerts whichever worker its socket landed on (about one
  `VPP_CLUSTER_POLL_INTERVAL_SECONDS` later).

Still per process:

- the rate limiter's buckets (a client can make up to workers x
  `VPP_RATE_LIMIT_REQUESTS_PER_MINUTE`), and any login throttling state;
- the EventBus itself (only WebSocket broadcasts are relayed);
- **OCPP**: charge-point sockets and the Central System's state live in one
  process, so `VPP_OCPP_ENABLED=true` with `VPP_API_WORKERS > 1` is refused
  at startup (`vpp serve` and the app lifespan). A DB command outbox would
  cover remote start/stop but not the connector/transaction reads, V2G
  schedule pushes and DR setpoints that also depend on that state, and it
  would still need sticky routing for the sockets. Run OCPP on a
  single-worker deployment.
- **Device control**: each device's active setpoint, write rate limiter and
  expiry watchdog live in the process that issued it, so
  `VPP_CONTROL_ENABLED=true` with `VPP_API_WORKERS > 1` is refused at
  startup for the same reason: two workers could send conflicting commands
  to one device.

Each worker logs the effective topology at startup (`API topology: ...`)
and warns about the per-process limits above.

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
`v2g_charging_sessions`, `v2g_schedules`, `v2g_flexibility_bids`,
`dr_event_responses`, and the multi-worker coordination tables
`cluster_leases`, `cluster_calls`, `cluster_events`.

Migrations live in `src/vpp/migrations/versions/` (0001-0010) and ship in
the wheel.
`tests/test_alembic_drift.py` upgrades a fresh database to head and fails if
the models and migrations differ, and checks there is a single head.
