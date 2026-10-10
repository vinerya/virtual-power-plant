# Changelog

All notable changes to the Virtual Power Plant platform are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Documentation site** (MkDocs + Material), to be published at
  <https://vinerya.github.io/virtual-power-plant/> once GitHub Pages is
  enabled (one-time steps in [releasing](docs/releasing.md#one-time-setup)).
  It serves the existing guides in `docs/` plus a home page with the
  maturity table, a getting-started section (installation, first run, and a
  ten-minute tutorial that dispatches the SunSpec simulator), and
  references generated at build time: REST API (OpenAPI schema from
  `create_app()` rendered with a self-hosted ReDoc), CLI (`--help` of every
  `vpp` command) and Python API (mkdocstrings). Changelog, contributing
  guide and license are included from the repository root. The site loads
  no analytics, fonts or other third-party assets. Build it with
  `pip install -e ".[docs]"` and `mkdocs serve` (new `docs` extra).
  `.github/workflows/docs.yml` runs `mkdocs build --strict` on pull
  requests and deploys from `main` only when the repository variable
  `DOCS_PAGES_ENABLED` is `true`.

## [3.0.0] - 2026-10-10

This release turns the library into an operable platform: the API is backed
by the database end to end, grid protocols talk to real peers, devices can
be controlled through Modbus/SunSpec, and the web console covers operations,
trading, optimization, protocols, tariffs, user administration and a
customer portal. Several defaults changed for safety; read **Upgrading from
2.x** and **Breaking changes** before upgrading.

### Upgrading from 2.x

1. **Back up the database** (`pg_dump`, or copy the SQLite file) and stop
   every API worker.
2. **Run the migrations with `vpp migrate`** (`alembic upgrade head`; the
   revisions ship inside the package, so it works from an installed wheel
   and any directory; Docker Compose runs it before starting the API). Do not
   skip this step: the API's default startup (`create_all`) creates missing
   tables but never adds columns to existing ones.
   - A database created by 2.0.0 has no `alembic_version` table (2.0.0 built
     its schema with `create_all`). Stamp it at the baseline once, then
     migrate. From a source checkout (repository root):
     `alembic -x url="$VPP_DATABASE_URL" stamp 0001_baseline`; from an
     installed wheel:
     `python -c "from alembic import command; from vpp.db.engine import alembic_config; from vpp.settings import get_settings; command.stamp(alembic_config(get_settings().database_url), '0001_baseline')"`.
     Then `vpp migrate` applies 0002-0014.
   - A database already under alembic (a pre-release build of 3.0) just
     runs `vpp migrate`.
   - New revisions: `0002` battery degradation columns and
     `battery_soh_samples`, `0003` `resources.nominal_energy_kwh`, `0004`
     tariffs, `0005` alerts and alert rules, `0006` sites, customers,
     metering, resource telemetry and config documents, `0007` V2G and
     protocol tables, `0008` drops UNIQUE constraints duplicated by unique
     indexes (PostgreSQL only), `0009` `users.token_version`,
     `users.last_login_at`, `api_keys.key_prefix`, `api_keys.last_used_at`,
     `0010` `cluster_leases`, `cluster_calls`, `cluster_events` and
     `v2g_flexibility_bids`, `0011` `shared_rate_limits`, `0012`
     `audit_log`, `0013` `output_caps`, `0014` an index on
     `resource_telemetry` (index only).
   - Optionally set `VPP_USE_ALEMBIC=1` so the API upgrades the schema at
     startup instead of running `create_all`.
3. **Settings you must review:**
   - `VPP_SECRET_KEY`: with `VPP_ENV=production` the API refuses to start
     on the default key or one shorter than 32 characters. Docker Compose
     now requires it to be set.
   - `VPP_API_HOST` now defaults to `127.0.0.1`. A bare `vpp serve` that
     must be reachable from the network needs `VPP_API_HOST=0.0.0.0` (or
     `--host`). The Docker image and compose files already bind `0.0.0.0`.
   - `VPP_API_WORKERS` is now honoured. With more than one worker,
     `VPP_OCPP_ENABLED` and `VPP_CONTROL_ENABLED` are refused at startup.
   - Rate limiting is now active (`VPP_RATE_LIMIT_ENABLED`, 120 requests
     per minute per client). Behind a reverse proxy or the web console, set
     `VPP_TRUSTED_PROXIES` to the proxy addresses so clients are told apart
     by their forwarded address.
   - `VPP_REDIS_URL` and `VPP_METRICS_PREFIX` are ignored and can be
     removed. In the monitoring overlay Grafana moved to port 3001.
   - Create the first admin with `vpp users create-admin` or the first-boot
     bootstrap (`VPP_BOOTSTRAP_ADMIN_USERNAME` +
     `VPP_BOOTSTRAP_ADMIN_PASSWORD_FILE`).
4. **API clients:** list endpoints return at most their default page size
   (100 for users and API keys) and report the total in `X-Total-Count`;
   the WebSocket requires a JWT and some events moved from `alerts` to
   `grid_events` / `system`; viewers can no longer write; API keys act with
   at most their own role; resource create/update is strictly typed; a few
   response shapes and status codes changed (trading order cancel, V2G
   duplicate vehicle, optimization history). See **Breaking changes**.
5. **Behaviour changes to know about:**
   - Audit log rows older than 365 days are pruned
     (`VPP_AUDIT_RETENTION_DAYS`); every other table is kept forever unless
     you set its retention.
   - New passwords need at least 12 characters; five failed logins lock a
     username for five minutes; inactive users cannot log in; password,
     role or activation changes revoke the user's existing tokens.
   - `VPP_CONFIG_PATH` is now loaded at startup when no configuration
     document is stored, and a missing or invalid file fails startup.
   - `GET /ready` returns `503` when the database is unreachable.
   - The V2G fleet is persisted; vehicles held in memory by 2.x are not
     migrated and must be registered again.
   - Device control (`VPP_CONTROL_ENABLED`), DR auto-response
     (`VPP_DR_AUTO_RESPONSE_ENABLED`) and Modbus/MQTT ingestion are off
     until you enable them.

### Breaking changes

- **Production requires a real secret key.** With `VPP_ENV=production` the
  API refuses to start on the public default `VPP_SECRET_KEY` or one
  shorter than 32 characters.
- **`VPP_API_HOST` defaults to `127.0.0.1`**, so a bare `vpp serve` is not
  reachable from the network by accident. The Docker image and compose
  files already pass `--host 0.0.0.0` explicitly.
- **WebSocket requires authentication by default.** `/api/v1/ws` (new
  canonical path) and `/ws` refuse handshakes without a valid JWT with close
  code `1008` (`VPP_WS_AUTH_REQUIRED=true`). Customer accounts are always
  refused. Open sockets are closed with `4001` when their credential
  expires or the session is revoked. Browsers should mint a 60 s token with
  `POST /api/v1/ws/token`; socket-only tokens are rejected by the HTTP API.
- **WebSocket channel semantics.** `alerts` now carries only alert payloads
  (newly fired alerts). Grid, DR, EV/V2G and protocol events moved to the new
  `grid_events` channel; every other unmapped event type goes to the new
  `system` channel. Clients that listened for those events on `alerts` must
  subscribe to `grid_events` / `system`.
- **Roles tightened.** Creating/updating/deleting resources, submitting and
  cancelling orders, ticking markets and running strategies now require
  `admin` or `operator` (viewers could do all of these). The new `customer`
  role is refused (`403`) on every operator endpoint.
- **API keys are limited to their own role.** A request made with a key
  runs with the lesser of the key's `role` and its owner's current role.
  Previously the key's role was ignored, so a `viewer` key minted by an
  admin carried full admin rights. Integrations that relied on that must be
  given a key with the role they actually need. Non-admins can no longer
  mint a key with a role higher than their own.
- **Password policy.** New passwords (register, `POST /api/v1/users`,
  `POST /api/v1/customers`, password change/reset, CLI, bootstrap) must be
  at least 12 characters (`VPP_PASSWORD_MIN_LENGTH`), at most 72 bytes, and
  not common or containing the username; otherwise `422`. Existing
  passwords keep working.
- **Login throttle.** 5 failed logins for a username lock it for 5 minutes
  (`429`; `VPP_LOGIN_MAX_FAILURES`, `VPP_LOGIN_LOCKOUT_SECONDS`).
- **Inactive users** can no longer obtain tokens.
- **List endpoints are capped.** `GET /api/v1/users`,
  `/api/v1/auth/api-keys` and `/api/v1/users/{id}/api-keys` now return at
  most 100 rows by default (`limit` up to 500; they returned everything),
  and every paginated list (see **Added → API**) applies its default page
  size. `/api/v1/dr/responses`, `/api/v1/alerts` and
  `/api/v1/optimization/setpoints` accept `offset`. A `limit` above an
  endpoint's maximum is `422`. Page with `limit` / `offset` and read the
  total from `X-Total-Count`.
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
- **`VPP_API_WORKERS` is honoured** (it was ignored). `VPP_OCPP_ENABLED` or
  `VPP_CONTROL_ENABLED` with more than one worker is refused at startup:
  OCPP charge-point sessions and device-control state live in one process.
- **Redis removed.** `VPP_REDIS_URL` and the compose `redis` service are
  gone (nothing used them). A leftover `VPP_REDIS_URL` is ignored.
- **Docker Compose**: `VPP_SECRET_KEY` is required; PostgreSQL is no longer
  published on the host; `docker-compose.dev.yml` is an overlay for
  `docker-compose.yml` rather than a standalone file; Grafana moved to port
  3001 in the monitoring overlay (3000 is the web console).
- **Python API.** `vpp.api.deps.get_vpp` is removed: the legacy in-memory
  `VirtualPowerPlant` singleton is replaced by an explicit live-config
  holder (`get_live_config` / `set_live_config` / `reset_live_config`).
  Sub-packages (`vpp.optimization`, `vpp.config`, `vpp.trading`,
  `vpp.models`) no longer define their own `__version__`; use
  `vpp.__version__`. `BaseConfig.merge()` raises `TypeError` for a config
  of another class (see **Fixed**).

### Added

**Capacity and soak testing**
- `vpp loadtest`: starts a real `vpp serve` (fresh SQLite file or any
  `--database-url`) and SunSpec simulators, then concurrently runs MQTT or
  HTTP telemetry ingestion (latency publish -> persisted -> WebSocket
  broadcast), authenticated dashboard reads (JWT and API key), WebSocket
  subscribers, periodic dispatch over K resources and Modbus polling.
  Reports throughput, p50/p95/p99, errors, server RSS/CPU, database growth
  and latency creep to `report.json` / `report.md`; exits 1 on errors. The
  CI job `perf` runs `vpp loadtest --smoke` and fails only on errors.
- [docs/performance.md](docs/performance.md): measured capacity of one API
  worker on SQLite and PostgreSQL 16 (15- and 60-minute soaks, 60-second
  ramps, one 4-vCPU host), bottlenecks found, reproduction commands and
  sizing guidance.
- [docs/operations.md](docs/operations.md): SQLite online backups and
  PostgreSQL `pg_dump`/`pg_restore` (executed end to end), verifying a
  backup, upgrades and rollback by restore (with the lossy downgrade
  migrations listed), rotating `VPP_SECRET_KEY` and API keys, revoking
  sessions, retention, probes and monitoring.
- `/metrics` exposes the standard process metrics
  (`process_resident_memory_bytes`, `process_cpu_seconds_total`, ...) and
  `vpp_alert_evaluation_dropped_total`.
- Migration `0014_telemetry_soc_index`: covering partial index on
  `resource_telemetry (resource_id, timestamp, state_of_charge) WHERE
  state_of_charge IS NOT NULL`.

**HTTP security headers and dependency updates**
- API: `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`,
  `Referrer-Policy: no-referrer` and `Content-Security-Policy: default-src
  'none'; frame-ancestors 'none'` on every response (a looser CSP for
  `/docs` and `/redoc`, which load Swagger UI / ReDoc from jsDelivr),
  `Cache-Control: no-store` on `/api/*`, and `Strict-Transport-Security` on
  HTTPS requests (direct TLS or `X-Forwarded-Proto: https` from a trusted
  proxy) or on every response with `VPP_HSTS_ENABLED=true`. Settings:
  `VPP_SECURITY_HEADERS_ENABLED`, `VPP_CONTENT_SECURITY_POLICY`,
  `VPP_DOCS_CONTENT_SECURITY_POLICY`, `VPP_HSTS_*`. WebSocket handshakes and
  CORS are unaffected. See docs/security.md#http-security-headers.
- Console: nonce-based Content-Security-Policy (`'strict-dynamic'`, no
  `'unsafe-eval'`, no un-nonced inline scripts) built per request by the
  middleware, plus `nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy` and
  `Permissions-Policy`; `X-Powered-By` removed. `CSP_CONNECT_SRC` adds
  origins, `HSTS_ENABLED=1` sends HSTS. Every page is now rendered per
  request (it must carry its nonce). The Playwright suite fails on any CSP
  violation.
- `.github/dependabot.yml`: weekly updates for pip, the console's npm/pnpm
  packages, GitHub Actions and the Docker base images, minor/patch grouped
  per ecosystem (see CONTRIBUTING.md).

**Users, credentials and audit**
- Admin user management under `/api/v1/users`: list/get/create, role
  change, activate/deactivate (not yourself, never the last active admin),
  password reset, revoke sessions, per-user API keys, and
  `DELETE /api/v1/users/{id}`, which deletes a user with their API keys,
  customer profile and programme enrolments (owned sites and saved config
  versions are kept, unlinked; not your own account, never the last active
  admin).
- Self-service `POST /api/v1/auth/password`, `POST /api/v1/auth/logout`
  (recorded in the audit log; the console's sign-out calls it) and
  `POST /api/v1/auth/logout-all`; `GET /api/v1/auth/api-keys` (own, or
  `?all=true` for admins) with key prefix and last-used time;
  `DELETE /api/v1/auth/api-keys/{id}` (owner or admin).
- Server-side session revocation: JWTs and socket tokens carry the user's
  `token_version` (`ver`); password change/reset, role change,
  deactivation and "log out everywhere" revoke existing tokens and close
  open WebSockets. Tokens without `ver` remain valid until expiry only
  while the user's version is 0.
- `vpp users create-admin | set-password | list` CLI (password from a
  prompt, stdin, a file or `VPP_ADMIN_PASSWORD`), and a first-boot admin
  bootstrap (`VPP_BOOTSTRAP_ADMIN_USERNAME` +
  `VPP_BOOTSTRAP_ADMIN_PASSWORD_FILE`, only while no user exists), wired
  into `docker-compose.yml`; replaces the database script in the docs.
- **Audit log** (`audit_log` table): logins (success, failure, throttled),
  logouts, session revocation, API-key creation/revocation, user creation /
  role change / (de)activation / password reset / deletion, password
  changes, and privileged control actions (device dispatch with `apply`,
  V2G schedules, dispatches and bids, OCPP remote start/stop, OpenADR opt
  overrides, protocol connect/disconnect, configuration changes, market
  orders, cancellations and live strategy runs) are recorded with actor,
  target, client IP, outcome and secret-free details. Admins read it with
  `GET /api/v1/audit` (filters: actor, action or action prefix, outcome,
  target, since/until; paginated). A failing audit insert never fails the
  request.
- **Data retention**: `VPP_AUDIT_RETENTION_DAYS`,
  `VPP_EVENT_LOG_RETENTION_DAYS`, `VPP_ALERT_RETENTION_DAYS` (resolved
  alerts), `VPP_DR_RESPONSE_RETENTION_DAYS`, `VPP_TELEMETRY_RETENTION_DAYS`
  (`0` = forever), pruned every `VPP_RETENTION_INTERVAL_MINUTES` in batches
  of `VPP_RETENTION_BATCH_SIZE` by the `data-retention` lease holder, with
  counts logged. Only the audit log is pruned by default (365 days); the
  other tables are opt-in, so upgrading deletes no existing history.
  `vpp prune [--dry-run]` runs the same pass once. Trades, orders and
  billing data are never pruned.
- `customer` role; tokens carry `aud` (`operator` | `customer`) re-checked
  against the user's current role; `/auth/me` returns `audience`.

**Device control and telemetry (Modbus, SunSpec, MQTT)**
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
- **Modbus inverter/meter telemetry ingestion.** `ModbusResourcePersister`
  bridges `ModbusAdapter` polls into live resource state; a resource opts in
  via its own `metadata["modbus"]` (host/port/mode/device_profile/
  poll_interval_s/power_register), each device gets its own supervised
  adapter (`modbus:{resource_id}` in the registry), polled power updates
  `current_power` (W → kW) and publishes `RESOURCE_UPDATED`, and polls
  append resource metrics history. An optional `soc_register` records a
  polled state of charge (percent, e.g. SunSpec `ChaState`) in the resource
  telemetry, where the dispatch optimiser reads a battery's SoC (before,
  Modbus-polled batteries never reported SoC and dispatch fell back to the
  configured or assumed value). `VPP_MODBUS_INGESTION_ENABLED` (default
  off); discovery runs once at startup.
- **SunSpec discovery.** `vpp.protocols.modbus.discover_sunspec_models()`
  walks a device's SunSpec chain (the `"SunS"` marker at 40000, 0 or 50000,
  `ID`/`L` model headers, end model `0xFFFF`), and the SunSpec setpoint
  profiles accept `"model_base": "auto"` to locate model 123/124 that way.
  `sunspec_model_124_registers()` also returns the read-only `WChaMax`,
  `ChaState` and `ChaSt` points (with their scale factors) for polling;
  `SUNSPEC_NOT_IMPLEMENTED` is public.
- **SunSpec device simulator.** `vpp simulate sunspec` /
  `python -m vpp.simulators.sunspec`: a pymodbus Modbus TCP server serving
  a SunSpec PV + battery hybrid inverter ("SunS" at 40000, common model 1,
  inverter model 103 or 113, controls model 123, storage model 124, end
  model), with realistic scale factors and "not implemented" values,
  strict write validation (read-only points and out-of-range set points
  answer Modbus exceptions) and simple physics: `WMaxLimPct`/`WMaxLim_Ena`
  curtail the PV output, `StorCtl_Mod` with `OutWRte`/`InWRte` charges or
  discharges the battery, SoC integrates between the reserve and 100 %,
  and the `*_RvrtTms` revert timers fire. Point layouts are checked against
  the SunSpec model definitions and the project's own register maps. Built
  from the specification: it does not show how vendor firmware behaves.
  Docker Compose profile `sim` runs it as `sunspec-sim`; see
  docs/protocols.md "Testing without hardware".
  `tests/test_sunspec_simulator.py` runs discovery, polling (ingestion
  loop) and `POST /api/v1/optimization/dispatch` with `apply=true` through
  the real actuator against the simulator on an ephemeral port.
- **MQTT battery telemetry ingestion.** `MQTTTelemetryIngestor` writes
  `vpp/{site_id}/{resource_type}/{resource_id}/{metric}` messages to
  `battery_states` and publishes `RESOURCE_UPDATED`;
  `MQTTAdapter.receive_forever()`; settings `VPP_MQTT_INGESTION_ENABLED`
  (default off), `VPP_MQTT_BROKER_HOST/PORT`, `VPP_MQTT_TOPIC_PREFIX`,
  `VPP_MQTT_USERNAME/PASSWORD`; a failed broker connection is retried
  instead of crashing startup.
- `VPP_MODBUS_INGESTION_ENABLED` and `VPP_CONTROL_ENABLED` are passed
  through by `docker-compose.yml`.

**Protocols, V2G and demand response**
- OCPP 1.6-J Central System at `/ocpp/{charge_point_id}` (subprotocol
  `ocpp1.6`, allow-list, Security Profile 1 Basic auth): Boot, Heartbeat,
  Status, Authorize, Start/StopTransaction, MeterValues, DataTransfer;
  RemoteStart/Stop, Set/ClearChargingProfile.
- OpenADR 2.0b VEN (HTTPS pull): registration, `oadrPoll`,
  `oadrDistributeEvent` parsing with a hardened XML parser,
  `oadrCreatedEvent` opt-in/out, re-registration, optional mTLS, backoff.
- IEEE 2030.5 client (mTLS): DeviceCapability → … → DERControlList,
  default control, time offset, paging, LFDI/SFDI identity, active controls
  by primacy. `DefaultDERControl` applies while no event control is active
  (its limits clamp; an OpenADR event target beats its target), and the
  client POSTs `DERControlResponse` resources (received / started /
  completed / cancelled / superseded) per `responseRequired`.
- Opt-in settings `VPP_OCPP_*`, `VPP_OPENADR_*`, `VPP_IEEE2030_5_*`; enabled
  adapters are supervised with exponential-backoff reconnects.
- `ProtocolStatus.SIMULATED` and `ProtocolMode` (live/simulated); adapters
  without a real endpoint never report `connected`; `GET /api/v1/protocols`
  exposes `mode` / `simulated`.
- Protocol data API: OCPP charge points and transactions, remote start/stop,
  OpenADR events and opt override, IEEE 2030.5 controls, `/api/v1/dr/status`,
  `/api/v1/dr/responses`.
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
- Library: Pyomo + HiGHS MILP solver plugin (`PyomoPlugin`), stochastic
  CVaR plugin with pluggable objective hooks, fleet dispatch model with an
  ADMM decomposition (`admm_fleet_solve`), and the rolling-horizon backtest
  harness (`vpp.optimization.backtest`).

**Battery degradation**
- `vpp.degradation`: throughput, calendar and rainflow degradation models
  with LFP / NMC presets, and wear-cost / depth-of-discharge hooks for the
  dispatch optimizer.
- State of health is persisted (resource SOH and throughput columns, the
  `battery_soh_samples` history) and updated from telemetry by a periodic
  updater (`VPP_DEGRADATION_UPDATER_ENABLED`, default on, every
  `VPP_DEGRADATION_UPDATER_INTERVAL_MINUTES` = 60);
  `GET /api/v1/batteries/{id}/soh`, `/soh/history` and
  `POST /api/v1/batteries/{id}/soh/update`.

**Tariffs**
- Tariff engine with a URDB loader (TOU / tiers / demand / fixed / minimum
  charges, adders and taxes), the `/api/v1/tariffs` REST API with URDB
  import, tariff-aware dispatch (tier-aware MILP) and pluggable real-time
  price feeds (`vpp.tariffs.feeds`: CAISO LMP, ComEd hourly, OpenADR price,
  synthetic).
- `TariffRead` gains derived components, weekday/weekend TOU heatmaps,
  sector, source, description, `is_tou`, `nem_regime`/`nem_source`,
  `parse_error`; create/update reject tariffs the engine cannot bill.
- `GET /tariffs/presets[/{id}]` (presets ship in the wheel: PG&E E-TOU-C,
  SCE TOU-D-PRIME and two illustrative ones); `GET /tariffs/import-urdb`
  reports whether `OPENEI_API_KEY` is set; OpenEI network errors map to
  `502`.
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

**Sites, customers and resources**
- Sites (`/api/v1/sites`) with lat/lon, region, IANA timezone and owner,
  live aggregates (power, capacity, SOC, online count, health, active
  alerts from persisted alerts); revenue-meter interval ingest and read.
- Customer portal (`/api/v1/customer/*`: me, devices, bill, DR programs,
  enrollments) and staff admin (`/api/v1/customers`, `/api/v1/programs`);
  the bill uses the tariff engine on the customer's own meter data and
  answers `409` instead of inventing a bill when no tariff is assigned.
- Resource metrics history (`/resources/{id}/metrics`) from
  `battery_states` and the new `resource_telemetry` table, and a telemetry
  ingest endpoint.
- Typed resource responses: `capacity_kwh`, `state_of_charge` (+ source),
  `current_charge_kwh`, `state_of_health`, `equivalent_full_cycles`.

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

**Multi-worker deployments**
- `vpp serve` starts `VPP_API_WORKERS` workers (`--workers` overrides) and
  each logs the effective topology at startup.
- DB-backed leadership leases (`cluster_leases`): the market-data tick,
  degradation updater, alert evaluation, data retention, MQTT/Modbus
  ingestion and protocol adapters + DR orchestrator run once per
  deployment, on the lease holder; another worker takes over when it dies.
- Trading venue calls (orders, cancel, portfolio, markets, tick, strategy
  runs) received by a non-holder are forwarded through `cluster_calls` and
  answered with the holder's result, or `503 leader_unavailable` /
  `504 leader_timeout`. Follower telemetry is forwarded to the alert
  evaluator.
- OpenADR / IEEE 2030.5 / DR views are complete on every worker: events,
  opt state, DERControls, programs, adapter metrics and `/api/v1/dr/status`
  (and the opt-in/out override, connect/disconnect) received by a worker
  that does not hold the `protocol-adapters` lease are forwarded to the
  holder through `cluster_calls`; `/api/v1/protocols/` includes the holder's
  adapters. Reads wait at most 3 s and answer `503 leader_unavailable` when
  the holder does not respond (they previously answered `404` "not
  running"). A single worker still answers them directly, with no database
  round trip.
- WebSocket broadcasts are relayed between workers (`cluster_events`).
- The HTTP rate limit and the failed-login throttle hold across workers:
  with `VPP_API_WORKERS > 1` they count in the new `shared_rate_limits`
  table with atomic upserts on SQLite and PostgreSQL, instead of giving a
  client N x the request limit and a password guesser N x the attempts.
  `VPP_RATE_LIMIT_BACKEND` (`auto` | `memory` | `database`, default
  `auto`) picks the store; a single worker stays in memory. On a database
  error the rate limiter fails open and the login throttle falls back to
  per-worker memory, both with a warning.
- Settings `VPP_CLUSTER_LEASE_TTL_SECONDS`,
  `VPP_CLUSTER_CALL_TIMEOUT_SECONDS`, `VPP_CLUSTER_POLL_INTERVAL_SECONDS`,
  `VPP_RATE_LIMIT_BACKEND`.

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
- Optional OpenTelemetry tracing: `tracing` extra and `VPP_TRACING_ENABLED`
  (default `false`; exporter, sampler and service name from the standard
  `OTEL_*` variables, OTLP over HTTP by default). HTTP requests, SQL,
  outbound httpx calls and spans for dispatch, optimisation solves,
  per-device setpoint writes, DR dispatch/release, Modbus polls/writes, OCPP
  calls, OpenADR / IEEE 2030.5 polls, market orders and forwarded cluster
  calls (which continue the caller's trace); `trace_id`/`span_id` in log
  lines. `docker compose --profile tracing` adds Jaeger. See
  [docs/observability.md](docs/observability.md).

**Configuration, API and database**
- `GET /api/v1/config` (live YAML + hash/version), `GET /config/schema`
  (strict JSON Schema), `PUT /config` (admin; validated, versioned,
  applied, optimistic concurrency via `base_hash`), re-applied on startup.
- **Pagination** for list endpoints that grow without bound: users, API
  keys, audit log, resources, alerts, orders, trades, optimization runs /
  dispatches, device commands, DR decisions, tariffs and customers take
  `limit` (capped per endpoint) and `offset` (`skip` stays an alias where it
  existed), keep returning a plain array and report the unpaginated total in
  `X-Total-Count` (exposed through CORS and the console proxy).
- `POST /api/v1/auth/token` accepts an OAuth2 form body or JSON.
- Collection routes are served with or without a trailing slash (no `307`).
- Alembic migrations (`0001_baseline` to `0014_telemetry_soc_index`, see
  **Upgrading from 2.x**) and a working `vpp migrate` (`alembic upgrade
  head`); a drift test compares models and migrations against SQLite and,
  in CI, PostgreSQL and checks for a single head.

**Web console (`web/`)**
- Next.js operator console: fleet overview, asset detail, dispatch history
  and explainer, sites map, alerts feed, YAML settings editor.
- New pages: trading workspace (markets, order ticket, orders, trades,
  portfolio) and strategies; optimization planner (schedule + backtest);
  protocols status (LIVE / SIMULATED); tariffs console against the real
  API (heatmaps, simulator with synthetic/CSV/compare, presets, URDB
  import); customer portal with real device energy flow.
- **Settings → Account** (change password, log out everywhere, own API keys
  with show-once creation), **Settings → Users & API keys** (admin, with a
  Delete action) and **Settings → Audit log**.
- Authenticated live updates: `/api/auth/ws-token` exchanges the session
  cookie for a socket token; the client connects to `/api/v1/ws` directly
  with backoff, resubscription and connection status.
- Role-aware session (viewers get read-only views), readable API errors,
  accessible tabs and fields, server config errors shown inline in the YAML
  editor.
- Mock data is opt-in (`NEXT_PUBLIC_USE_MOCKS=1`); failures render error
  states instead of fake data.
- Monaco is self-hosted; ESLint config; Playwright end-to-end tests in CI.
- `web/Dockerfile` (standalone output) and a `vpp-web` compose service.

**Docs**
- `docs/`: architecture, configuration reference (every `VPP_*` setting),
  deployment, protocols/V2G/DR, API guide, tariffs, security,
  [operations runbook](docs/operations.md),
  [performance](docs/performance.md), [observability](docs/observability.md)
  and [releasing](docs/releasing.md).

### Changed

- The config editor validates against the JSON Schema with
  `@cfworker/json-schema` instead of Ajv, which compiles schemas with
  `new Function` and so cannot run under the console's CSP. Messages read
  differently (e.g. `Instance type "string" is invalid. Expected
  "integer".`).
- Dependency floors raised to what the code needs: `pulp>=2.8` (the
  highspy-backed `HiGHS` solver used by the V2G scheduler; 2.7 only has
  `HiGHS_CMD`) and `paho-mqtt>=2.0` (`CallbackAPIVersion.VERSION2`).
- Pyomo and HiGHS are now core dependencies. `import vpp` (and so the API
  and `vpp` CLI) always needed them, so a plain install without the
  `solver` extra could not start; `solver` remains as an empty extra.
- Alembic migrations moved from `alembic/` into the package
  (`src/vpp/migrations/`) and ship in the wheel; `vpp migrate` works from any
  directory and from an installed wheel. `alembic.ini` at the repo root
  points at the new location.
- API Docker image: installs the package from its wheel (not an editable
  source checkout) with the `protocols`, `solver` and `degradation`
  extras and psycopg2, ships the migrations, and runs as a non-root user
  with a writable `/app`.
- Schema bootstrap at startup (`create_all` or `VPP_USE_ALEMBIC`) runs
  under a PostgreSQL advisory lock, so several workers can start at once.
  The API lifespan applies the stored config document after `init_db()`.
- V2G flexibility bids (`v2g_flexibility_bids`) and the aggregator's
  dispatch counters are persisted instead of kept per process; bids carry a
  `bid_id`.
- `POST /api/v1/optimization/dispatch` and the DR orchestrator share one
  implementation (`execute_dispatch`); the route no longer duplicates the
  run/status bookkeeping.
- `vpp.__version__` is the single version source: the checkout's
  `pyproject.toml` in a source checkout (an editable install's metadata
  lags a version bump), else the installed distribution's metadata. The
  app, `/version`, `/health`, `vpp_info` and `vpp --version` use it.
- `configure_logging` is called by `create_app`, is idempotent and stamps
  stdlib records too.
- `AlertManager` keeps per-source rule state, supports per-rule resource
  scoping and splits `check()` / `dispatch()`.
- The OCPP, OpenADR and IEEE 2030.5 adapters' receive buffers drop the
  oldest message on overflow instead of counting each overflow as an error.
- `vpp.trading.markets.MarketData` is now the same class as
  `vpp.trading.data.MarketData` (the duplicate dataclass and the cast that
  bridged them are gone); API responses are unchanged.
- Demos and benchmarks moved into the package as `vpp.demos` and
  `vpp.benchmarks`, so `vpp demo` / `vpp benchmark` work from an installed
  command.
- The console login sends a form body; mock fallbacks and silent 404
  handling were removed from sites, alerts and the portal.
- **`examples/` are linted and tested.** CI's lint job (and CONTRIBUTING's
  checks) run `ruff check` / `ruff format --check` on `examples/`, and
  `tests/test_examples.py` runs every example as a subprocess from an empty
  working directory (exit code 0, no files left behind; ~5 s in total).
  `examples/advanced_usage.py`, which imported APIs removed long ago
  (`EventHandler`, `WebhookEventHandler`, `create_strategy`,
  `forecast_production`, ...), is rewritten as a walkthrough of the current
  core API: `VPPConfig`, `Battery`/`Solar`/`WindTurbine`,
  `VirtualPowerPlant` dispatch and the `EventBus`. The other examples exit
  non-zero on failure instead of printing the error and exiting 0;
  `advanced_configuration_demo.py` writes into `--output-dir` (default: a
  temporary directory) instead of the repo's `output/` directory and the
  current directory, and drops its "hot reload (simulated)" section (there
  is no hot reload); `trading_demo.py`'s engine section now actually
  submits orders against seeded liquidity. The stale generated files in
  `output/` are removed.
- The test suite keeps its SQLite databases in temporary directories, so a
  run leaves the working tree clean and concurrent runs do not share a
  database.
- README and docs rewritten to describe maturity honestly (production-grade
  / beta / simulated / research).

### Deprecated

- Credentials as query parameters on `POST /api/v1/auth/token` (still
  accepted; responses carry `Deprecation` and `Warning` headers). Send a
  form or JSON body.
- The repo-root `demos/` and `benchmarks/` packages: deprecated import
  aliases of `vpp.demos` / `vpp.benchmarks`.
- The `skip` query parameter on list endpoints that had it: an alias of
  `offset`.

### Removed

- Redis setting and compose service.
- `VPP_METRICS_PREFIX`: it was never read (metric names are fixed to
  `vpp_*`, which the Grafana dashboards rely on). Setting it is harmless.
- `vpp.api.deps.get_vpp` and the in-memory `VirtualPowerPlant` API
  singleton; sub-package `__version__` attributes.
- Dead modules shadowed by packages: `vpp/models.py`, and the flat
  `analysis.py`, `config.py`, `events.py`, `optimization.py`,
  `simulation.py`, `visualization.py`.
- The console's fake `app/api/tariff-presets` route (presets now come from
  the backend).
- Tracked `web/tsconfig.tsbuildinfo`.

### Fixed

Found with `vpp loadtest` (numbers in docs/performance.md):

- **The API could deadlock and die under load when a dispatch solved
  while the event loop logged.** Pyomo's HiGHS interface redirected the
  process stdout/stderr file descriptors and re-pointed every log handler
  for each solve, from the solver's worker thread; a concurrent log line
  could deadlock the solve and leave the root handler writing to a closed
  stream, after which every log call raised. The API now switches Pyomo to
  stream-only capture and turns HiGHS's console output off.
- **MQTT telemetry waited for an unrelated timer before it was
  processed.** The paho callback put messages on an `asyncio.Queue` from
  its own thread without waking the event loop: 79 ms median / 285 ms p95
  publish-to-persist at 1 message/s, now 8 ms / 11 ms.
- **MQTT ingestion topped out at about 90 messages/s on SQLite and then
  dropped telemetry.** Each message was its own lookup and commit; queued
  messages are now written in batches: 800 messages/s with no loss and a
  p95 under 60 ms on one worker. Rows are stamped with the time the message
  was received rather than the commit time.
- **`GET /api/v1/resources` (and resource detail, sites, dispatch) slowed
  down linearly as telemetry accumulated.** The newest-SOC lookup read every
  sample of every listed battery (~190 ms at 1M rows on SQLite, 84 ms on
  PostgreSQL); it now seeks the newest row per resource (3 ms / 6 ms).
- **Alert rules saw only part of the telemetry above ~200 events/s.** The
  evaluator now evaluates everything queued in one transaction (retrying
  event by event if that fails) and skips the auto-resolve query for
  rule/resource pairs with no open alert: at 800 messages/s every event is
  evaluated (before: about a third).
- Every HTTP request walked the whole route table in the trailing-slash
  middleware (~0.8 ms); the per-path answer is now cached.
- A full MQTT or alert-evaluation queue logged one warning per dropped
  message; both now log at most every 10 s with a count, and the drops are
  counted in metrics.
- The API's own MQTT disconnect at shutdown was logged as "disconnected
  unexpectedly".

**API and platform**
- EventBus publishes now reach connected WebSocket clients (previously two
  disconnected pub/sub systems), and non-alert events no longer land on the
  `alerts` channel.
- Rate-limiting middleware (`VPP_RATE_LIMIT_ENABLED`, on by default) is
  wired into `create_app()`; it was never installed.
- Per-user rate limiting behind the console: the Next.js proxy and auth
  routes forward the client address (`X-Forwarded-For` / `X-Real-IP`), and
  the API honours those headers only from `VPP_TRUSTED_PROXIES` (new, empty
  by default), walking `X-Forwarded-For` from the right so forged entries
  are ignored. Compose pins `vpp-web` to a fixed address on its own subnet,
  trusts only it, and sets the per-client limit back to 120/min (was 600
  shared by every console user).
- Typed resource fields (capacity, SOC, chemistry, limits, nameplates) were
  silently dropped on create; the optimizer now reads telemetry SOC and
  `max_charge_kw` / `max_discharge_kw`.
- `PUT /api/v1/resources/{id}` no longer 500s (`MissingGreenlet` on the
  expired `updated_at`).
- Config documents applied with `PUT /api/v1/config` were lost on restart.
- `VPP_CONFIG_PATH` is now loaded at startup when no configuration document
  has been stored; a stored document still wins, and a missing or invalid
  file fails startup. A test fails if a new setting is never read.
- `VPP_API_KEY_HEADER` now sets the header API keys are read from (it was
  ignored in favour of a hard-coded `X-API-Key`).
- Sites always reported `active_alerts = 0`.
- `GET /ready` probes the database and returns `503` when it is unreachable
  (it previously always reported ready).
- `vpp migrate` was a stub. Alembic: `tariffs` and `alerts` tables had no
  migration; `fileConfig()` disabled every app logger when migrations ran
  in-process; timestamp columns tightened to `NOT NULL` to match the ORM.
- Python 3.10 (the declared minimum) is supported again:
  - The WebSocket loop caught the builtin `TimeoutError`, which only aliases
    `asyncio.TimeoutError` from 3.11. On 3.10 the token-expiry and
    session-revocation checks never ran and the loop crashed instead.
  - The version fallback for uninstalled checkouts needed `tomllib` (3.11+);
    it now uses `tomli` or a minimal `[project]` parser.
- API Dockerfile copied a non-existent `setup.py` and tried to download the
  package from PyPI.

**Device control and Modbus**
- **Curtailed PV can be un-curtailed by dispatch.** A generation resource's
  availability was its polled output, so once the VPP had capped it (e.g.
  `WMaxLimPct` 40 % -> 4 kW) every later dispatch saw only 4 kW and never
  raised the limit until the device's revert timer expired. The actuator
  now tracks the output caps it holds (accepted, read-back-verified writes),
  and while a resource's output sits at such a cap the optimiser estimates
  its availability from the newest reading taken before the cap (at most
  15 min old, `availability_basis` `pre_curtailment_telemetry`) or else its
  nameplate (`nameplate_curtailed`). A generation resource allocated all of
  its estimated availability gets its upper limit written instead of the
  estimate (`"uncurtailed": true` in the delivery), and `sunspec_123`
  treats a setpoint of 100 % or more as no limit and writes only
  `WMaxLim_Ena` = 0. The run's asset summary records `output_limit_kw`. See
  [docs/protocols.md](docs/protocols.md) for the limitations.
- **Curtailment survives an API restart.** The output caps are persisted in
  the new `output_caps` table (migration `0013_output_caps`): written when a
  cap is set or rewritten, deleted when it is lifted or released (including
  on a clean shutdown), and loaded at startup when
  `VPP_CONTROL_ENABLED=true`. Rows whose SunSpec revert timer
  (`revert_timeout_s`) has run out since the last write, or whose resource
  was deleted, are dropped on load. A database error while saving or
  loading caps is logged and never fails a dispatch.
- **SunSpec register maps verified against the official SunSpec model
  definitions** (the sunspec/models JSON bundled with pysunspec2 1.3.6);
  still untested on physical hardware. `tests/test_sunspec_models.py`
  checks every shipped SunSpec register's offset, size, type, units, write
  access and scale-factor pairing against the model JSON (skipped without
  `pysunspec2`, now in the `dev` extra). Discrepancies fixed:
  - `solaredge_se` (model 101/103, ID at 40069) was off by one (1-based
    register numbers used as wire addresses): `ac_power` 40084 -> 40083,
    `ac_power_scale` 40085 -> 40084, `dc_power` 40101 -> 40100,
    `temperature` 40104 -> 40103 (`TmpSnk`), `ac_energy` 40094 -> 40093
    and typed `acc32` instead of `uint32`.
  - `solaredge_se` values ignored their SunSpec scale factors (`ac_power`
    was the raw count; `temperature` hard-coded 0.01). Each value is now
    paired with its `*_SF` register (`W_SF`, `Hz_SF`, `WH_SF`, `DCW_SF`,
    `Tmp_SF`, all read) and polled values are in engineering units.
    `frequency` was added.
  - `fronius_symo` (float model 113, ID at 40069): `float32` points read one
    register instead of two, and three addresses pointed at other points:
    `ac_power` 40092 -> 40091 (`W`), `frequency` 40086 (`PhVphA`) -> 40093
    (`Hz`), `ac_energy` 40094 -> 40101 (`WH`), `dc_power` 40101 -> 40107
    (`DCW`).
  - Model 123/124 control offsets were already correct; they now record
    their SunSpec points and scale-factor pairing so the test covers them.
- **Modbus scale-factor tearing.** Polling read each value and its SunSpec
  scale-factor register in separate requests, so a device changing a scale
  factor in between yielded a value off by a power of ten. The polled
  registers of one SunSpec model are now read in one contiguous request
  (e.g. `solaredge_se`: one 24-register read per poll instead of ten), any
  other value together with its scale factor, and each value is scaled only
  with the scale factor from its own request. A rejected block read falls
  back to one request per value + scale-factor pair.
- SunSpec conventions are honoured when polling: "not implemented" values
  (0x8000 int16/sunssf, 0xFFFF uint16/enum16/bitfield16, 0x80000000 int32,
  0xFFFFFFFF uint32, 0 acc32, NaN float32) are dropped instead of being
  reported as readings, and so is any value whose scale factor is not
  implemented. Float NaN readings are dropped for every map.
- The SunSpec setpoint profiles read the model's `ID`/`L` header at
  `model_base` before the first write and refuse a mismatch, instead of
  writing to whatever sits at a wrong address.
- Modbus reads/writes failed on pymodbus >= 3.10 (`slave=` renamed to
  `device_id=`, `count` keyword-only); the adapter now adapts to either.
  Custom registers no longer leak into the shared vendor register maps.
- **Cancelled background writers could leave SQLite locked.** Cancelling a
  task while it was writing interrupted SQLAlchemy mid-statement; the
  connection was invalidated without a rollback and kept SQLite's write
  lock until it was garbage collected ("database is locked" for every other
  writer). This hit Modbus polling when ingestion stopped, and the same
  pattern existed, when cancelled at shutdown or on lease loss, in MQTT
  telemetry ingestion, the alert evaluator, the DR orchestrator's
  dispatch/release records, OCPP vehicle sync, the degradation updater,
  retention pruning, the trading market-data tick, device-setpoint audit
  records and output-cap persistence, the cluster lease renewal,
  forwarded-call executor and WebSocket relay, and the shared rate limit
  store (also the request commit and post-commit audit write). Each write
  transaction now goes through one shared helper
  (`vpp.db.engine.run_in_session` / `run_shielded`) that finishes it
  (commit or rollback, session closed) before the cancellation proceeds;
  the loops around them still stop promptly. A forwarded call the executor
  claimed now always gets its final status written instead of staying
  `running`.

**Models, optimization and tariffs**
- **Battery model SoC did not track the charge moved.**
  `AdvancedElectrochemicalModel` reported a normalised "bulk
  concentration" as SoC, driven by a dimensionally wrong flux term and
  lagging the surface concentration: an hour at 100 kW into the sample
  2000 Ah / 400 V battery moved SoC from 0.50 to 0.83 (Coulomb counting
  gives about 0.62) and SoC kept rising during a following discharge. Its
  OCV, a single-cell curve clipped to the pack's voltage limits, sat at
  `min_voltage`, which blocked discharge entirely. SoC now follows Coulomb
  counting in both battery models; the advanced model's OCV is a pack-level
  curve rising with SoC and its concentrations start at the initial SoC.
  Both models also applied the charge/discharge efficiencies the wrong way
  round, so a charge/discharge cycle created energy. Charging now stores
  `charge_efficiency` of the delivered power and discharging draws
  `power / discharge_efficiency`. The voltage power limits now use the
  current that brings the terminal voltage to `max_voltage`/`min_voltage`,
  which no longer caps the sample battery near 10 kW, and a step no longer
  overshoots `max_soc`/`min_soc`. The simple model's terminal voltage now
  rises while charging.
- **`AdvancedElectrochemicalModel` aged the caller's `BatteryParameters`.**
  Resistance growth multiplied `parameters.internal_resistance` in place,
  so models sharing one parameters object (or a reused config) aged each
  other, and the per-step re-multiplication compounded with the step count.
  Battery models no longer modify `parameters` (the nominal values): the
  advanced model tracks its cumulative aging and computes the resistance as
  `nominal * (1 + resistance_growth_rate * cumulative_aging)`, which the
  terminal voltage, heat and power limits all use. The effective value is
  available as `BatteryModel.internal_resistance`.
- **`AdvancedElectrochemicalModel` crashed on YAML-loaded parameters.**
  PyYAML reads `1e-14` (no dot) as a string, so the model built from
  `configs/advanced_vpp_config.yaml` raised `TypeError` on the first
  `update()`. Numeric parameters are now coerced with `float()`, and the
  sample config spells those values YAML-1.1-safely (`1.0e-14`).
- **Unbounded history on long-running objects.** The advanced battery
  model's power/SoC history and `EnergyResource`'s state history (appended
  on every `set_power`) grew forever. Both are now capped (default 1000
  samples, most recent kept): battery models take `max_history` as a
  constructor/`create_battery_model` argument or a `max_history` resource
  parameter, and `EnergyResource` takes a `max_history` argument. New
  `BatteryModel.get_soc_history()` / `get_power_history()` return lists;
  `get_state_history()` still returns a list (now a copy).
- **`BaseConfig.merge(other)` returned `other`.** It deep-merged the two
  configs' `to_dict()` output, which includes every field, defaults too, so
  merging a config that set a single field wiped the base's resources,
  name, objectives and so on. Config dataclasses now record which fields
  were explicitly set (passed to the constructor, assigned afterwards, or
  present in the dict given to `from_dict`/`load_from_file`), and `merge`
  applies only those: nested configs are merged field by field, dict
  fields key by key, and lists and scalars are replaced. `merge` now
  raises `TypeError` for a config of another class. Loading a
  `VPPConfig` whose `monitoring.log_file` is set no longer fails with
  `UnboundLocalError` when the `vpp` logger already has a handler, and no
  longer adds a second file handler for the same file.
- **Rule-based stochastic fallback never dispatched.**
  `SimpleStochasticRules` (the fallback for stochastic problems, also used
  by `optimize_trading_schedule` and by the API's stochastic dispatch when
  the Pyomo CVaR plugin cannot solve) compared each period's
  10th-percentile price with the same period's 90th percentile, which can
  never trigger, so it always returned an all-idle schedule with zero cost.
  It now charges when even the pessimistic (90th-percentile) price is below
  the horizon's average expected price and discharges when even the
  pessimistic (10th-percentile) price is above it.
- The V2G LP scheduler works on PuLP 4: it solves with HiGHS (a core
  dependency) instead of the no-longer-bundled CBC, creates variables with
  `prob.add_variable`, and reads the solve status in a version-independent
  way. Previously it silently fell back to the rule-based schedule; the
  fallback now logs the underlying exception.
- NEM 3.0 avoided cost supports full-year vectors: 24 (hour of day),
  12 x 24 (month x hour, nested or flat), 8760 and 8784 (hour of year,
  leap-day aware), indexed by local time in both the bill credit and the
  optimizer. Previously only the hour of day was used for bills and the
  optimizer indexed long vectors by horizon offset. Other lengths are now
  rejected (`422`) instead of being indexed modulo their length.
- `wear_cost_hooks_for_telemetry_consistency()` keeps the dispatch model's
  wear cost consistent with the SOH the telemetry updater persists.

**Trading**
- Portfolio equity double-counted purchase cost; realized P&L attribution,
  buy-to-cover and fees in daily P&L, partial short covers; historical VaR
  sign and quantile; risk checks skipped the position limit for new
  positions; order-book levels stayed inflated after fills; FOK mutated the
  book before cancelling; stop-limit never filled at its trigger;
  float-modulo tick/lot checks; markets without sessions; real-time market
  partial remainders; every market order rejected; zero-quantity fills
  booked as filled; simulated prices compounding seasonally; strategies
  replay with data timestamps; arbitrage emitting opportunities twice; the
  ML strategy no longer claims to load a model.
- `RiskManager.check_limits()` also checks VaR and concentration, pricing
  positions from live market data when available.
- An order's volume-weighted average fill price was accumulated
  incrementally in floating point and could land one ulp below the lowest
  fill price (two fills at 46.18 averaged 46.179999…), which made
  `test_market_buy_fills_and_persists` fail intermittently.
  `Order.update_fill` now keeps the average inside the range of the prices
  it averages.
- **`TradingEngine.stop()` took ~20 s and left threads running.** Its
  worker loops slept with `time.sleep(60)`, so `stop()` waited out
  `join(timeout=10)` per thread; they now wait on the stop event and exit
  immediately. Same for `MarketDataProvider.stop_streaming()`.

**Console, demos and tests**
- The console's API proxy failed on `204 No Content` answers (it built a
  response with a body for a null-body status); it now forwards them as-is.
- Console: WebSocket client dialled a proxy path that could never upgrade;
  asset metrics chart lagged one poll; two-decimal price axis.
- Protocols demo no longer depends on a current event loop and no longer
  calls the OpenADR handler twice.
- Flaky tests: `test_mpc_warm_start_speedup` compares deterministic HiGHS
  iteration counts (`solver_iterations` /
  `cumulative_solver_iterations`); the optimization benchmark consistency
  check uses thread CPU time; `tests/test_db.py` runs standalone.

### Security

- Security response headers on the API and the console (see Added), and
  `Cache-Control: no-store` on API and proxied API responses so tokens, API
  keys and per-user data are not cached by browsers or proxies.
Hardening that changes behaviour (production secret-key check, WebSocket
authentication, role tightening, API-key role cap, password policy, login
throttle, inactive users) is listed under **Breaking changes**. In
addition:

- Customer accounts are denied by default on operator endpoints and the
  WebSocket; customer routes scope every response to the caller (foreign
  ids → `404`).
- Login no longer reveals whether a username exists through response
  timing (a dummy bcrypt check runs for unknown users) and passwords over
  72 bytes answer `401` instead of `500`.
- The failed-login lockout and the per-IP rate limit are no longer
  multiplied by the number of API workers (shared through the database).
- Forwarded client addresses are trusted only from `VPP_TRUSTED_PROXIES`.
- Deactivating a user also revokes their API keys; password, role and
  activation changes revoke the user's sessions.
- Login credentials move from the query string to the request body.
- `PUT /api/v1/config` refuses `monitoring.log_file` (server-side file
  write).
- Alert webhooks can be HMAC-signed; `/metrics` can require a bearer token.
- OpenADR / IEEE 2030.5 XML is parsed with entity resolution and network
  access disabled.
- Privileged actions are recorded in the audit log.

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
