# Changelog

All notable changes to the Virtual Power Plant platform are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

This release turns the library into an operable platform: the API is backed
by the database end to end, grid protocols talk to real peers, and the web
console covers operations, trading, optimization, protocols, tariffs and a
customer portal. Several defaults changed for safety; read **Breaking
changes** before upgrading.

### Breaking changes

- **`VPP_API_WORKERS` is honoured** (it was ignored). `VPP_OCPP_ENABLED`
  with more than one worker is refused at startup: OCPP charge-point
  sessions live in one process.
- **API keys are limited to their own role.** A key now acts with the lesser
  of its `role` and its owner's current role. Integrations that relied on a
  lower-role key minted by an admin having admin rights must be given a key
  with the role they actually need.
- **Production requires a real secret key.** With `VPP_ENV=production` the
  API refuses to start on the default `VPP_SECRET_KEY` or one shorter than
  32 characters.
- **WebSocket requires authentication by default.** `/api/v1/ws` (new
  canonical path) and `/ws` refuse handshakes without a valid JWT with close
  code `1008` (`VPP_WS_AUTH_REQUIRED=true`). Customer accounts are always
  refused. Open sockets are closed with `4001` when their credential
  expires. Browsers should mint a 60 s token with `POST /api/v1/ws/token`.
- **WebSocket channel semantics.** `alerts` now carries only alert payloads
  (newly fired alerts). Grid, DR, EV/V2G and protocol events moved to the new
  `grid_events` channel; every other unmapped event type goes to the new
  `system` channel. Clients that listened for those events on `alerts` must
  subscribe to `grid_events` / `system`.
- **Roles tightened.** Creating/updating/deleting resources, submitting and
  cancelling orders, ticking markets and running strategies now require
  `admin` or `operator` (viewers could do all of these). A new `customer`
  role is refused (`403`) on every operator endpoint.
- **Inactive users** can no longer obtain tokens.
- **Resource create/update is typed.** `POST /api/v1/resources` is a
  discriminated union on `resource_type` (`battery`, `solar`,
  `wind_turbine`); unknown or foreign fields are rejected with `422` instead
  of being silently dropped. `resource_type` is canonicalised (`"wind"` →
  `"wind_turbine"`, case-insensitive) on create and in list filters. `PUT`
  is a partial update re-validated against the type's model.
- **Trading `DELETE /api/v1/trading/orders/{id}`** returns the full order
  (`OrderResponse`) instead of `{"id", "status"}`, and `409` when the order
  is already final.
- **`POST /api/v1/v2g/vehicles` with a duplicate `ev_id` returns `409`**
  instead of overwriting the vehicle. The V2G fleet is now persisted; the
  previous per-process in-memory fleet is not migrated.
- **`GET /api/v1/optimization/history`** returns the `DispatchRun` shape
  (with `start`/`end`/`resource_id`/`offset` filters) used by
  `/optimization/runs`.
- **`POST /api/v1/optimization/dispatch`** allocates over the online
  resources stored in the database instead of the (always empty) in-memory
  VPP singleton.
- **`vpp.api.deps.get_vpp` removed.** The legacy in-memory
  `VirtualPowerPlant` singleton is replaced by an explicit live-config
  holder (`get_live_config` / `set_live_config` / `reset_live_config`).
- **Redis removed.** `VPP_REDIS_URL` and the compose `redis` service are
  gone (nothing used them). A leftover `VPP_REDIS_URL` is ignored.
- **Grafana moved to port 3001** in the monitoring overlay (3000 is the web
  console).
- **Docker Compose**: `VPP_SECRET_KEY` is required; PostgreSQL is no longer
  published on the host; `docker-compose.dev.yml` is an overlay for
  `docker-compose.yml` rather than a standalone file.
- Sub-packages (`vpp.optimization`, `vpp.config`, `vpp.trading`,
  `vpp.models`) no longer define their own `__version__`; use
  `vpp.__version__`.
- **Password policy.** New passwords (register, `POST /api/v1/users`,
  `POST /api/v1/customers`, password change/reset, CLI, bootstrap) must be
  at least 12 characters (`VPP_PASSWORD_MIN_LENGTH`), at most 72 bytes, and
  not common or containing the username; otherwise `422`. Existing
  passwords keep working.
- **Login throttle.** 5 failed logins for a username lock it for 5 minutes
  (`429`; `VPP_LOGIN_MAX_FAILURES`, `VPP_LOGIN_LOCKOUT_SECONDS`).
- **Migration `0009_user_management`** adds `users.token_version`,
  `users.last_login_at`, `api_keys.key_prefix` and `api_keys.last_used_at`.

### Added

**Users and credentials**
- `vpp users create-admin | set-password | list` CLI (password from a
  prompt, stdin, a file or `VPP_ADMIN_PASSWORD`), and a first-boot admin
  bootstrap (`VPP_BOOTSTRAP_ADMIN_USERNAME` +
  `VPP_BOOTSTRAP_ADMIN_PASSWORD_FILE`, only while no user exists), wired
  into `docker-compose.yml`; replaces the database script in the docs.
- Admin user management under `/api/v1/users`: list/get/create, role
  change, activate/deactivate (not yourself, never the last active admin),
  password reset, revoke sessions, per-user API keys.
- Self-service `POST /api/v1/auth/password` and
  `POST /api/v1/auth/logout-all`; `GET /api/v1/auth/api-keys` (own, or
  `?all=true` for admins) with key prefix and last-used time;
  `DELETE /api/v1/auth/api-keys/{id}` (owner or admin).
- Server-side session revocation: JWTs and socket tokens carry the user's
  `token_version` (`ver`); password change/reset, role change,
  deactivation and "log out everywhere" revoke existing tokens and close
  open WebSockets. Tokens without `ver` remain valid until expiry only
  while the user's version is 0.
- Console: **Settings → Account** (change password, log out everywhere,
  own API keys with show-once creation) and **Settings → Users & API keys**
  (admin).

**Optimization**
- DB-backed dispatch: single-interval allocation LP (Pyomo + HiGHS) with
  SOC/energy limits, configured charge/discharge limits, SOH-aware battery
  wear cost and renewables-first merit order; headroom-proportional
  fallback; reports method, shortfall and solve time.
- `POST /api/v1/optimization/schedule` (alias `/mpc`): horizon MPC via
  `MPCController` / `MultiResourceMPCController` from a price series or a
  stored tariff, degradation-aware wear cost, feeder limits, terminal-SOC
  policy.
- `POST /api/v1/optimization/backtest`: closed-loop MPC replay with
  perfect / persistence / noisy forecasts, scored against idle, rule-based
  and perfect-foresight baselines (regret).
- Run history: every run is persisted in `optimization_runs` and publishes
  `OPTIMIZATION_STARTED/COMPLETED/FAILED`; `GET /optimization/runs`,
  `/runs/{id}`, `/runs/{id}/explain`, `/api/v1/dispatches[/{id}[/explain]]`.
- `/optimization/stochastic` builds a real `stochastic_dispatch` problem so
  the CVaR plugin runs (seeded scenarios, `risk_weight`, optional DB battery).

**Trading (simulated venue)**
- `TradingService` wires `vpp.trading` into the API with a
  `SimulatedExchange` (seeded synthetic liquidity); state is rebuilt from
  the `orders`/`trades` tables.
- Order types market, limit, stop, stop-limit, iceberg, IOC, FOK with
  GTC/DAY; pre-trade risk checks (`422 risk_limit_breached` with reasons;
  rejected orders persisted); cancel via `DELETE` or
  `POST /orders/{id}/cancel`.
- `GET /portfolio` (positions, realized/unrealized P&L, fees, exposure,
  parametric VaR, drawdown, limit breaches), `GET /trades`, `GET /markets`
  (+ depth), `POST /markets/tick`, strategy catalogue, backtests and
  dry-run execution.
- `MARKET_DATA`, order and trade events; a lifespan task ticks the venue
  (`VPP_TRADING_MARKET_DATA_*`).

**Protocols, V2G and demand response**
- `ProtocolStatus.SIMULATED` and `ProtocolMode` (live/simulated); adapters
  without a real endpoint never report `connected`; `GET /api/v1/protocols`
  exposes `mode` / `simulated`.
- OCPP 1.6-J Central System at `/ocpp/{charge_point_id}` (subprotocol
  `ocpp1.6`, allow-list, Security Profile 1 Basic auth): Boot, Heartbeat,
  Status, Authorize, Start/StopTransaction, MeterValues, DataTransfer;
  RemoteStart/Stop, Set/ClearChargingProfile.
- OpenADR 2.0b VEN (HTTPS pull): registration, `oadrPoll`,
  `oadrDistributeEvent` parsing with a hardened XML parser,
  `oadrCreatedEvent` opt-in/out, re-registration, optional mTLS, backoff.
- IEEE 2030.5 client (mTLS): DeviceCapability → … → DERControlList,
  default control, time offset, paging, LFDI/SFDI identity, active controls
  by primacy.
- Opt-in settings `VPP_OCPP_*`, `VPP_OPENADR_*`, `VPP_IEEE2030_5_*`; enabled
  adapters are supervised with exponential-backoff reconnects.
- Persistent V2G fleet (`v2g_vehicles`, `v2g_charging_sessions`,
  `v2g_schedules`); `PATCH` vehicle, `PUT/DELETE` binding, `GET` sessions and
  schedules; automatic EV ↔ charger binding by idTag; meter values update
  SOC/power; schedules and dispatches pushed as `SetChargingProfile` with
  per-vehicle delivery results.
- DR orchestrator (`vpp.dr`): OpenADR SIMPLE / LOAD_DISPATCH / LOAD_CONTROL
  and IEEE 2030.5 controls → fleet target and limits → DB-backed dispatch →
  EV setpoints, re-dispatch on change/interval, release when signals end,
  audit in `dr_event_responses`. **Auto-response off by default**
  (`VPP_DR_AUTO_RESPONSE_ENABLED`); caps `VPP_DR_MAX_EXPORT_KW` /
  `VPP_DR_MAX_IMPORT_KW`.
- **Device control (setpoint actuator, `vpp.control`).** Dispatch
  allocations are written to stationary devices that opt in via
  `metadata["modbus"]["control"]`: generic signed-register profile, SunSpec
  model 123 (`WMaxLimPct` / `WMaxLim_Ena` / revert timer) and SunSpec model
  124 storage (**generic/unverified**). Safety: global kill switch
  `VPP_CONTROL_ENABLED` (default off), clamp to resource and per-device
  limits, deadband, rate limit with deferred writes, read-back verification,
  writable-register guard, never writes offline resources, watchdog fallback
  (`safe_setpoint_kw` or release) when a setpoint expires
  (`VPP_CONTROL_WATCHDOG_INTERVAL_S`, `VPP_CONTROL_EXPIRY_GRACE_S`) and
  release on shutdown. Commands are logged in `event_log`
  (`device_setpoint`) and published as `DEVICE_SETPOINT`. Used by
  `POST /api/v1/optimization/dispatch` with `"apply": true` (default false;
  admin/operator only; per-resource `device_deliveries`) and the DR
  orchestrator; `GET /api/v1/optimization/setpoints` shows state and recent
  commands.
- IEEE 2030.5: `DefaultDERControl` applies while no event control is active
  (its limits clamp; an OpenADR event target beats its target), and the
  client POSTs `DERControlResponse` resources (received / started /
  completed / cancelled / superseded) per `responseRequired`.
- Protocol data API: OCPP charge points and transactions, remote start/stop,
  OpenADR events and opt override, IEEE 2030.5 controls, `/api/v1/dr/status`,
  `/api/v1/dr/responses`.
- **Modbus inverter/meter telemetry ingestion.** `ModbusResourcePersister`
  bridges `ModbusAdapter` polls into live resource state; a resource opts in
  via its own `metadata["modbus"]` (host/port/mode/device_profile/
  poll_interval_s/power_register), each device gets its own supervised
  adapter (`modbus:{resource_id}` in the registry), polled power updates
  `current_power` (W → kW) and publishes `RESOURCE_UPDATED`.
  `VPP_MODBUS_INGESTION_ENABLED` (default off); discovery runs once at
  startup.
- **MQTT battery telemetry ingestion.** `MQTTTelemetryIngestor` writes
  `vpp/{site_id}/{resource_type}/{resource_id}/{metric}` messages to
  `battery_states` and publishes `RESOURCE_UPDATED`;
  `MQTTAdapter.receive_forever()`; settings `VPP_MQTT_INGESTION_ENABLED`
  (default off), `VPP_MQTT_BROKER_HOST/PORT`, `VPP_MQTT_TOPIC_PREFIX`,
  `VPP_MQTT_USERNAME/PASSWORD`; a failed broker connection is retried
  instead of crashing startup.

**Sites, customers and resources**
- `customer` role; tokens carry `aud` (`operator` | `customer`) re-checked
  against the user's current role; `/auth/me` returns `audience`.
- Sites (`/api/v1/sites`) with lat/lon, region, IANA timezone and owner,
  live aggregates (power, capacity, SOC, online count, health, active
  alerts from persisted alerts); revenue-meter interval ingest and read.
- Customer portal (`/api/v1/customer/*`: me, devices, bill, DR programs,
  enrollments) and staff admin (`/api/v1/customers`, `/api/v1/programs`);
  the bill uses the tariff engine on the customer's own meter data and
  answers `409` instead of inventing a bill when no tariff is assigned.
- Resource metrics history (`/resources/{id}/metrics`) from
  `battery_states` and the new `resource_telemetry` table, and a telemetry
  ingest endpoint; Modbus polls append history.
- Typed resource responses: `capacity_kwh`, `state_of_charge` (+ source),
  `current_charge_kwh`, `state_of_health`, `equivalent_full_cycles`.

**Tariffs**
- `TariffRead` gains derived components, weekday/weekend TOU heatmaps,
  sector, source, description, `is_tou`, `nem_regime`/`nem_source`,
  `parse_error`; create/update reject tariffs the engine cannot bill.
- `GET /tariffs/presets[/{id}]` (presets ship in the wheel, two new
  illustrative ones); `GET /tariffs/import-urdb` reports whether
  `OPENEI_API_KEY` is set; OpenEI network errors map to `502`.
- Simulation load sources `meter_trace`, `synthetic` (residential /
  commercial shape, optional PV) or `csv`; `timezone`, `billing_cycle`
  (monthly cycles for long windows), `compare_to`, NEM regimes
  `none|nem2|nem3|net_billing`; response adds cycles, export totals, notes,
  load summary, comparison.
- `vpp.tariffs.nem` shared by the simulator and the customer bill; regime
  from the tariff's `nem` extension key or URDB `dgrules`; `?nem=` what-if
  for staff.
- **Combined tiered + TOU rate structures.** `TimeOfUseRate.period_tiers`
  bills a multi-period URDB tariff whose periods have usage tiers against
  each period's own cumulative kWh (previously only the first tier's rate).
- **NEM 2.0 TOU-period-aware export credit.** URDB `sell` rates are parsed
  into `TOUSchedule.sell_rate` (`TimeOfUseRate.export_rate()` falls back to
  the import rate); each exported interval is credited at its own period's
  rate instead of a bill-wide blended average; the dispatch-side optimizer
  (`tariff_to_opt_params`, `nem="nem2"`) uses the same per-step sell rate.

**Observability and alerts**
- `GET /metrics` (when `VPP_METRICS_ENABLED` and `prometheus_client` is
  installed; optional `VPP_METRICS_BEARER_TOKEN`), Prometheus middleware by
  route template, EventBus-driven resource/optimization/trading/protocol
  metrics.
- Request ids (`X-Request-ID`, bound into structlog contextvars) and one
  `vpp.access` line per request; `VPP_LOG_JSON`, `VPP_ACCESS_LOG_ENABLED`.
- Persisted alerts and rules (`/api/v1/alerts`, ack/snooze/resolve, rules
  CRUD); `AlertService` evaluates rules on `RESOURCE_UPDATED`, de-duplicates,
  auto-resolves, broadcasts on `alerts`; three default rules seeded.
- Real webhook delivery with retries/backoff and HMAC-SHA256 signing
  (`X-VPP-Timestamp`, `X-VPP-Signature`).
- Grafana datasource/dashboard provisioning with "VPP Overview", "VPP
  Trading" and "VPP Fleet" dashboards (metric names guarded by a test).

**Configuration and API plumbing**
- `GET /api/v1/config` (live YAML + hash/version), `GET /config/schema`
  (strict JSON Schema), `PUT /config` (admin; validated, versioned,
  applied, optimistic concurrency via `base_hash`), re-applied on startup.
- `POST /api/v1/auth/token` accepts an OAuth2 form body or JSON.
- Collection routes are served with or without a trailing slash (no `307`).
- `vpp migrate` runs `alembic upgrade head`; migrations 0004 (tariffs),
  0005 (alerts), 0006 (sites, customers, metering, telemetry, config
  documents), 0007 (V2G and protocols); a drift test compares models and
  migrations and checks for a single head.

**Web console (`web/`)**
- Authenticated live updates: `/api/auth/ws-token` exchanges the session
  cookie for a socket token; the client connects to `/api/v1/ws` directly
  with backoff, resubscription and connection status.
- New pages: trading workspace (markets, order ticket, orders, trades,
  portfolio) and strategies; optimization planner (schedule + backtest);
  protocols status (LIVE / SIMULATED); tariffs console against the real
  API (heatmaps, simulator with synthetic/CSV/compare, presets, URDB
  import); customer portal with real device energy flow.
- Role-aware session (viewers get read-only views), readable API errors,
  accessible tabs and fields, server config errors shown inline in the YAML
  editor.
- Mock data is opt-in (`NEXT_PUBLIC_USE_MOCKS=1`); failures render error
  states instead of fake data.
- Monaco is self-hosted; ESLint config; Playwright end-to-end tests in CI.
- `web/Dockerfile` (standalone output) and a `vpp-web` compose service.

**Multi-worker deployments**
- `VPP_API_WORKERS` is read: `vpp serve` starts that many workers
  (`--workers` overrides) and each logs the effective topology at startup.
  OCPP and device control (`VPP_CONTROL_ENABLED`) keep per-process state and
  are refused when more than one worker is configured.
- DB-backed leadership leases (`cluster_leases`): the market-data tick,
  degradation updater, alert evaluation, MQTT/Modbus ingestion and protocol
  adapters + DR orchestrator run once per deployment, on the lease holder;
  another worker takes over when it dies.
- Trading venue calls (orders, cancel, portfolio, markets, tick, strategy
  runs) received by a non-holder are forwarded through `cluster_calls` and
  answered with the holder's result, or `503 leader_unavailable` /
  `504 leader_timeout`. Follower telemetry is forwarded to the alert
  evaluator.
- WebSocket broadcasts are relayed between workers (`cluster_events`).
- The HTTP rate limit and the failed-login throttle hold across workers:
  with `VPP_API_WORKERS > 1` they count in the new `shared_rate_limits`
  table (migration `0011_shared_rate_limits`) with atomic upserts on SQLite
  and PostgreSQL, instead of giving a client N x the request limit and a
  password guesser N x the attempts. `VPP_RATE_LIMIT_BACKEND`
  (`auto` | `memory` | `database`, default `auto`) picks the store; a
  single worker stays in memory. On a database error the rate limiter fails
  open and the login throttle falls back to per-worker memory, both with a
  warning.
- Settings `VPP_CLUSTER_LEASE_TTL_SECONDS`,
  `VPP_CLUSTER_CALL_TIMEOUT_SECONDS`, `VPP_CLUSTER_POLL_INTERVAL_SECONDS`,
  `VPP_RATE_LIMIT_BACKEND`.

**Docs**
- `docs/`: architecture, configuration reference (every `VPP_*` setting),
  deployment, protocols/V2G/DR, API guide, tariffs, security.

### Changed

- Pyomo and HiGHS are now core dependencies. `import vpp` (and so the API
  and `vpp` CLI) always needed them, so a plain install without the
  `solver` extra could not start; `solver` remains as an empty extra.
- `VPP_API_HOST` defaults to `127.0.0.1`, so a bare `vpp serve` is not
  reachable from the network by accident. The Docker image and compose
  files already pass `--host 0.0.0.0` explicitly.
- V2G flexibility bids (`v2g_flexibility_bids`) and the aggregator's
  dispatch counters are persisted instead of kept per process; bids carry a
  `bid_id`.
- Schema bootstrap at startup (`create_all` or `VPP_USE_ALEMBIC`) runs
  under a PostgreSQL advisory lock, so several workers can start at once.
- The API lifespan applies the stored config document after `init_db()`.
- `configure_logging` is called by `create_app`, is idempotent and stamps
  stdlib records too.
- `vpp.__version__` comes from the installed distribution metadata (falls
  back to `pyproject.toml` in a source checkout); the app, `/version` and
  `vpp_info` use it.
- `AlertManager` keeps per-source rule state, supports per-rule resource
  scoping and splits `check()` / `dispatch()`.
- The OCPP, OpenADR and IEEE 2030.5 adapters' receive buffers drop the
  oldest message on overflow instead of counting each overflow as an error.
- `POST /api/v1/optimization/dispatch` and the DR orchestrator share one
  implementation (`execute_dispatch`); the route no longer duplicates the
  run/status bookkeeping.
- The console login sends a form body; mock fallbacks and silent 404
  handling were removed from sites, alerts and the portal.
- API Docker image installs the `protocols`, `solver` and `degradation`
  extras and psycopg2, ships alembic, and runs as a non-root user with a
  writable `/app`.
- README and docs rewritten to describe maturity honestly (production-grade
  / beta / simulated / research).
- Alembic migrations moved from `alembic/` into the package
  (`src/vpp/migrations/`) and ship in the wheel; `vpp migrate` works from any
  directory and from an installed wheel. `alembic.ini` at the repo root
  points at the new location. The API image now installs the wheel instead
  of an editable source checkout.
- `vpp.trading.markets.MarketData` is now the same class as
  `vpp.trading.data.MarketData` (the duplicate dataclass and the cast that
  bridged them are gone); API responses are unchanged.
- Demos and benchmarks moved into the package as `vpp.demos` and
  `vpp.benchmarks`, so `vpp demo` / `vpp benchmark` work from an installed
  command. The repo-root `demos/` and `benchmarks/` remain as deprecated
  import aliases.

### Deprecated

- Credentials as query parameters on `POST /api/v1/auth/token` (still
  accepted; responses carry `Deprecation` and `Warning` headers).

### Fixed

- Python 3.10 (the declared minimum) is supported again:
  - The WebSocket loop caught the builtin `TimeoutError`, which only aliases
    `asyncio.TimeoutError` from 3.11. On 3.10 the token-expiry and
    session-revocation checks never ran and the loop crashed instead.
  - The version fallback for uninstalled checkouts needed `tomllib` (3.11+);
    it now uses `tomli` or a minimal `[project]` parser.
- The V2G LP scheduler works on PuLP 4: it solves with HiGHS (a core
  dependency) instead of the no-longer-bundled CBC, creates variables with
  `prob.add_variable`, and reads the solve status in a version-independent
  way. Previously it silently fell back to the rule-based schedule; the
  fallback now logs the underlying exception.
- Per-user rate limiting behind the console: the Next.js proxy and auth
  routes forward the client address (`X-Forwarded-For` / `X-Real-IP`), and
  the API honours those headers only from `VPP_TRUSTED_PROXIES` (new, empty
  by default), walking `X-Forwarded-For` from the right so forged entries
  are ignored. Compose pins `vpp-web` to a fixed address on its own subnet,
  trusts only it, and sets the per-client limit back to 120/min (was 600
  shared by every console user).
- NEM 3.0 avoided cost supports full-year vectors: 24 (hour of day),
  12 x 24 (month x hour, nested or flat), 8760 and 8784 (hour of year,
  leap-day aware), indexed by local time in both the bill credit and the
  optimizer. Previously only the hour of day was used for bills and the
  optimizer indexed long vectors by horizon offset. Other lengths are now
  rejected (422) instead of being indexed modulo their length.
- `VPP_API_KEY_HEADER` now sets the header API keys are read from (it was
  ignored in favour of a hard-coded `X-API-Key`).
- `VPP_CONFIG_PATH` is now loaded at startup when no configuration document
  has been stored; a stored document still wins, and a missing or invalid
  file fails startup. A test fails if a new setting is never read.
- EventBus publishes now reach connected WebSocket clients (previously two
  disconnected pub/sub systems), and non-alert events no longer land on the
  `alerts` channel.
- Rate-limiting middleware (`VPP_RATE_LIMIT_ENABLED`, on by default) is
  wired into `create_app()`.
- `PUT /api/v1/resources/{id}` no longer 500s (`MissingGreenlet` on the
  expired `updated_at`).
- Typed resource fields (capacity, SOC, chemistry, limits, nameplates) were
  silently dropped on create; the optimizer now reads telemetry SOC and
  `max_charge_kw` / `max_discharge_kw`.
- Sites always reported `active_alerts = 0`.
- Config documents applied with `PUT /api/v1/config` were lost on restart.
- Trading: portfolio equity double-counted purchase cost; realized P&L
  attribution, buy-to-cover and fees in daily P&L, partial short covers;
  historical VaR sign and quantile; risk checks skipped the position limit
  for new positions; order-book levels stayed inflated after fills; FOK
  mutated the book before cancelling; stop-limit never filled at its
  trigger; float-modulo tick/lot checks; markets without sessions;
  real-time market partial remainders; every market order rejected;
  zero-quantity fills booked as filled; simulated prices compounding
  seasonally; strategies replay with data timestamps; arbitrage emitting
  opportunities twice; the ML strategy no longer claims to load a model.
- `RiskManager.check_limits()` also checks VaR and concentration, pricing
  positions from live market data when available.
- Alembic: `tariffs` and `alerts` tables had no migration; `fileConfig()`
  disabled every app logger when migrations ran in-process; timestamp
  columns tightened to `NOT NULL` to match the ORM.
- `vpp migrate` was a stub.
- Modbus reads/writes failed on pymodbus >= 3.10 (`slave=` renamed to
  `device_id=`, `count` keyword-only); the adapter now adapts to either.
  Custom registers no longer leak into the shared vendor register maps.
- Protocols demo no longer depends on a current event loop and no longer
  calls the OpenADR handler twice.
- Console: WebSocket client dialled a proxy path that could never upgrade;
  asset metrics chart lagged one poll; two-decimal price axis.
- `wear_cost_hooks_for_telemetry_consistency()` keeps the dispatch model's
  wear cost consistent with the SOH the telemetry updater persists.
- Flaky tests: `test_mpc_warm_start_speedup` compares deterministic HiGHS
  iteration counts (`solver_iterations` /
  `cumulative_solver_iterations`); the optimization benchmark consistency
  check uses thread CPU time; `tests/test_db.py` runs standalone.
- API Dockerfile copied a non-existent `setup.py` and tried to download the
  package from PyPI.

### Removed

- Redis setting and compose service.
- `VPP_METRICS_PREFIX`: it was never read (metric names are fixed to
  `vpp_*`, which the Grafana dashboards rely on). Setting it is harmless.
- Dead modules shadowed by packages: `vpp/models.py`, and the flat
  `analysis.py`, `config.py`, `events.py`, `optimization.py`,
  `simulation.py`, `visualization.py`.
- The console's fake `app/api/tariff-presets` route (presets now come from
  the backend).
- Tracked `web/tsconfig.tsbuildinfo`.

### Security

- The failed-login lockout and the per-IP rate limit are no longer
  multiplied by the number of API workers (shared through the database).
- Login no longer reveals whether a username exists through response
  timing (a dummy bcrypt check runs for unknown users) and passwords over
  72 bytes answer `401` instead of `500`.
- Deactivating a user also revokes their API keys.
- WebSocket authentication (see Breaking changes); socket-only tokens are
  rejected by the HTTP API; sockets close when their session expires.
- Customer accounts are denied by default on operator endpoints and the
  WebSocket; customer routes scope every response to the caller (foreign
  ids → `404`).
- Viewers can no longer create, modify or delete resources or trade.
- Non-admins can no longer mint an API key with a role higher than their
  own.
- Login credentials move from the query string to the request body.
- `PUT /api/v1/config` refuses `monitoring.log_file` (server-side file
  write).
- Alert webhooks can be HMAC-signed; `/metrics` can require a bearer token.
- OpenADR / IEEE 2030.5 XML is parsed with entity resolution and network
  access disabled.
- API keys are now limited to their own `role`: a request made with a key
  runs with the lesser of the key's role and its owner's current role.
  Previously the key's role was ignored, so a `viewer` key minted by an
  admin carried full admin rights.
- A production instance (`VPP_ENV=production`) refuses to start with the
  public default `VPP_SECRET_KEY` or a key shorter than 32 characters.
- `GET /ready` probes the database and returns 503 when it is unreachable
  (it previously always reported ready).

## [2.0.0] - 2025-02-24

> Note (added later): some entries below describe components that are not
> in the code base — Gaussian-process forecasting and PPO reinforcement
> learning. `vpp.research` contains baseline forecasters, anomaly
> detectors and an experiment runner; see ADVANCED_FEATURES.md.

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
- Real-time grid services with fast rule-based frequency/voltage response
- Distributed coordination via ADMM for multi-site VPP portfolios
- Model Predictive Control for rolling-horizon dispatch
- Plugin architecture for custom optimization solvers
- Multi-market trading system (day-ahead, real-time, ancillary services, bilateral)
- 5 trading strategies: arbitrage, momentum, mean reversion, ML-based, multi-market
- Portfolio management with P&L tracking and risk controls
- Physics-based battery models (equivalent-circuit and electrochemical)
- Solar PV and wind turbine resource models
- YAML/JSON configuration system with validation and hot reload
- Rule-based expert system for operational decisions
- Comprehensive test suite
