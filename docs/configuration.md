# Configuration reference

The API is configured through environment variables prefixed with `VPP_`
(case-insensitive), read by [`vpp.settings.Settings`](../src/vpp/settings.py)
(pydantic-settings). A `.env` file in the **working directory** is read as a
fallback; real environment variables win. Unknown `VPP_*` variables are
ignored, so an old deployment that still sets e.g. `VPP_REDIS_URL` keeps
starting.

Value formats:

- booleans: `true`/`false`, `1`/`0`, `yes`/`no`
- lists: JSON, e.g. `VPP_CORS_ORIGINS='["https://vpp.example.com"]'`
- optional values: leave unset for "none"

A copy-and-edit template lives in [`.env.example`](../.env.example). The web
console has its own variables, listed at the end of this page.

The tables below list **every** field of `Settings` with its default as
defined in code.

## Core

| Variable | Default | Meaning |
|---|---|---|
| `VPP_ENV` | `development` | `production` turns on JSON logs (unless `VPP_LOG_JSON` says otherwise). Also reported by `GET /api/v1/config`. `development` and `testing` are the other recognised values. |
| `VPP_DEBUG` | `false` | Echo SQL statements (SQLAlchemy `echo`) and more verbose logging in a few modules. Keep off in production. |
| `VPP_LOG_LEVEL` | `INFO` | One of `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` (validated). |
| `VPP_LOG_JSON` | unset | `true` = JSON log lines, `false` = coloured console logs. Unset = JSON only when `VPP_ENV=production`. |
| `VPP_ACCESS_LOG_ENABLED` | `true` | One `vpp.access` log line per HTTP request, carrying the request id. |

## API server

| Variable | Default | Meaning |
|---|---|---|
| `VPP_API_HOST` | `127.0.0.1` | Bind address for `vpp serve` (`--host` overrides); plain uvicorn uses its own `--host`. Reported by `GET /api/v1/config`. |
| `VPP_API_PORT` | `8000` | Bind port for `vpp serve` (`--port` overrides), as above. |
| `VPP_API_WORKERS` | `1` | Worker processes started by `vpp serve` (overridden by `--workers`). With more than one, singleton work runs on DB-lease holders and WebSocket broadcasts are relayed between workers; see [architecture](architecture.md#process-model). Refused together with `VPP_OCPP_ENABLED`. If you start uvicorn/gunicorn with several workers yourself, set this to the same number. |
| `VPP_CLUSTER_LEASE_TTL_SECONDS` | `15` | Leadership lease lifetime (renewed every third of it). A crashed leader's work moves to another worker within about this long (at once on the same host). Host clocks must agree to well within it. |
| `VPP_CLUSTER_CALL_TIMEOUT_SECONDS` | `10` | How long a worker waits for the lease holder to answer a forwarded call (trading; OpenADR / IEEE 2030.5 / DR views, which also cap it at 3 s) before returning `503 leader_unavailable` (not executed) or `504 leader_timeout` (outcome unknown). |
| `VPP_CLUSTER_POLL_INTERVAL_SECONDS` | `0.25` | How often lease holders poll for forwarded calls and workers poll the WebSocket relay. |
| `VPP_CORS_ORIGINS` | `["http://localhost:3000"]` | Allowed browser origins (credentials allowed). The web console calls the API server-side, so this only matters for browsers calling the API directly. |

## Security

| Variable | Default | Meaning |
|---|---|---|
| `VPP_SECRET_KEY` | `change-me-to-a-real-secret-key` | HMAC key for every JWT (sessions and WebSocket tokens). **Must** be replaced with a long random value; the default is public. `docker-compose.yml` refuses to start without it. Rotating it invalidates all sessions. |
| `VPP_JWT_ALGORITHM` | `HS256` | JWT signing algorithm (symmetric). |
| `VPP_JWT_EXPIRE_MINUTES` | `60` | Lifetime of access tokens from `POST /api/v1/auth/token`, and of WebSocket sessions opened with an API key. |
| `VPP_API_KEY_HEADER` | `X-API-Key` | Request header that carries API keys (`POST /api/v1/auth/api-key`). The OpenAPI security scheme shows the name configured at startup. |
| `VPP_RATE_LIMIT_ENABLED` | `true` | Per-client-IP rate limit on every HTTP route. |
| `VPP_RATE_LIMIT_REQUESTS_PER_MINUTE` | `120` | Requests per minute per client, across all workers. Requests over the limit get `429`. |
| `VPP_RATE_LIMIT_BACKEND` | `auto` | Where the rate limiter and the login throttle count: `memory` (per process; a token bucket, no database round trip), `database` (table `shared_rate_limits`, shared by every worker and replica on the database; a one-minute sliding window) or `auto` (`database` when `VPP_API_WORKERS > 1`, else `memory`). Set `database` for several single-worker replicas behind a load balancer. On a database error requests are let through and logins fall back to in-memory counting, with a warning. See [architecture](architecture.md#process-model). |
| `VPP_TRUSTED_PROXIES` | `[]` | JSON list of CIDRs/addresses (e.g. `'["172.29.0.10/32"]'`) whose `X-Forwarded-For` / `X-Real-IP` headers identify the client for rate limiting. The web console forwards them, so trust its address to give each console user their own bucket. Empty: the headers are ignored (every console user shares the Next.js server's bucket). `X-Forwarded-For` is walked from the right past trusted hops, so clients cannot choose their address by sending the header themselves. See [deployment](deployment.md#rate-limiting-behind-the-console). |
| `VPP_WS_AUTH_REQUIRED` | `true` | Refuse WebSocket handshakes without a valid token (close code 1008). Setting `false` lets anonymous sockets receive fleet-wide data; only for isolated development. |
| `VPP_WS_TOKEN_EXPIRE_SECONDS` | `60` | How long a token from `POST /api/v1/ws/token` may be used to *open* a socket. The socket itself lives until the underlying session expires (see [api.md](api.md#websocket)). |
| `VPP_PASSWORD_MIN_LENGTH` | `12` | Minimum password length; the rest of the policy is fixed (see [security.md](security.md#authentication)). |
| `VPP_LOGIN_MAX_FAILURES` | `5` | Failed logins per username before it is locked out (`429`); counted across workers when `VPP_RATE_LIMIT_BACKEND` resolves to `database`. `0` disables. |
| `VPP_LOGIN_LOCKOUT_SECONDS` | `300` | Lockout duration, and the window in which failures are counted. |
| `VPP_BOOTSTRAP_ADMIN_USERNAME` | unset | With `VPP_BOOTSTRAP_ADMIN_PASSWORD_FILE`: create this admin at startup while the users table is empty. |
| `VPP_BOOTSTRAP_ADMIN_PASSWORD_FILE` | unset | File holding the bootstrap admin's password (e.g. `/run/secrets/...`); read only when the bootstrap runs. |

## Database

| Variable | Default | Meaning |
|---|---|---|
| `VPP_DATABASE_URL` | `sqlite+aiosqlite:///./vpp.db` | SQLAlchemy async URL. PostgreSQL: `postgresql+asyncpg://user:pass@host:5432/vpp`. Migrations rewrite the driver to a sync one (`sqlite`, `postgresql+psycopg2`), so `psycopg2` must be installed to migrate PostgreSQL (the Docker image includes it). |
| `VPP_USE_ALEMBIC` | `false` | `false`: tables are created with `create_all` on startup (quick for development, no schema upgrades). `true`: run `alembic upgrade head` on startup. For production run `vpp migrate` before starting the API instead (compose does this). |

## Monitoring and alerts

| Variable | Default | Meaning |
|---|---|---|
| `VPP_METRICS_ENABLED` | `true` | Mount `GET /metrics` (Prometheus text format). Requires `prometheus-client` (`monitoring` extra); otherwise the route is not mounted. |
| `VPP_METRICS_BEARER_TOKEN` | unset | When set, `/metrics` requires `Authorization: Bearer <token>`. Unset = unauthenticated; restrict it at the network/proxy level. |
| `VPP_ALERTS_ENABLED` | `true` | Evaluate persisted alert rules against `RESOURCE_UPDATED` telemetry. |
| `VPP_ALERTS_SEED_DEFAULT_RULES` | `true` | On startup, insert three default rules (SOC low, over-temperature, SOH degraded) if the `alert_rules` table is empty. |
| `VPP_ALERT_WEBHOOK_URL` | unset | POST every fired alert as JSON to this URL (retries with capped exponential backoff on network errors, 429 and 5xx). |
| `VPP_ALERT_WEBHOOK_SECRET` | unset | Sign webhook requests: `X-VPP-Timestamp: <unix seconds>` and `X-VPP-Signature: sha256=<hex HMAC-SHA256 of "<timestamp>.<body>">`. |
| `VPP_ALERT_WEBHOOK_TIMEOUT_SECONDS` | `5.0` | Per-attempt HTTP timeout. |
| `VPP_ALERT_WEBHOOK_MAX_RETRIES` | `3` | Retries after the first attempt. |

## Battery degradation

| Variable | Default | Meaning |
|---|---|---|
| `VPP_DEGRADATION_UPDATER_ENABLED` | `true` | Background task that recomputes battery state of health from stored SOC history (rainflow cycle counting + calendar ageing). A no-op until telemetry is stored. |
| `VPP_DEGRADATION_UPDATER_INTERVAL_MINUTES` | `60` | How often it runs. |

## Telemetry ingestion (MQTT, Modbus)

Both are off by default because they dial out to external systems.

| Variable | Default | Meaning |
|---|---|---|
| `VPP_MQTT_INGESTION_ENABLED` | `false` | Subscribe to an MQTT broker and store battery telemetry from `vpp/{site_id}/{resource_type}/{resource_id}/{metric}` topics; publishes `RESOURCE_UPDATED`. |
| `VPP_MQTT_BROKER_HOST` | `localhost` | Broker host. |
| `VPP_MQTT_BROKER_PORT` | `1883` | Broker port. |
| `VPP_MQTT_TOPIC_PREFIX` | `vpp/#` | Subscription filter. |
| `VPP_MQTT_USERNAME` | unset | Broker username. |
| `VPP_MQTT_PASSWORD` | unset | Broker password. |
| `VPP_MODBUS_INGESTION_ENABLED` | `false` | Poll Modbus TCP/RTU devices for every resource whose `metadata.modbus` block configures one (host, port, mode, device profile, poll interval, power register). Discovery runs once at startup. See `src/vpp/protocols/modbus_ingestion.py`. |

## OCPP 1.6-J Central System

See [protocols.md](protocols.md#ocpp-16-j-central-system).

| Variable | Default | Meaning |
|---|---|---|
| `VPP_OCPP_ENABLED` | `false` | Start the Central System; charge points connect to `ws(s)://<host>/ocpp/{charge_point_id}` with subprotocol `ocpp1.6`. While disabled, that route refuses every connection. |
| `VPP_OCPP_HEARTBEAT_INTERVAL_S` | `300` | Heartbeat interval returned in `BootNotification.conf`. |
| `VPP_OCPP_CALL_TIMEOUT_S` | `30.0` | Timeout for CALLs the Central System sends (RemoteStart, SetChargingProfile, ...). |
| `VPP_OCPP_AUTO_ACCEPT_BOOT` | `true` | Accept `BootNotification` from charge points not seen before; `false` rejects unknown charge points. |
| `VPP_OCPP_ALLOWED_CHARGE_POINTS` | `[]` | Allow-list of charge point ids. Empty = any id may connect (set this, or Basic auth, in production). |
| `VPP_OCPP_BASIC_AUTH_PASSWORD` | unset | OCPP Security Profile 1: require HTTP Basic auth, username = charge point id, this password. Only safe over `wss://`. |
| `VPP_OCPP_AUTHORIZED_ID_TAGS` | unset | Accepted idTags for `Authorize` / `StartTransaction`. Unset = every idTag is accepted. |
| `VPP_OCPP_REMOTE_ID_TAG` | `VPP` | idTag used for `RemoteStartTransaction`. |
| `VPP_V2G_MAX_PROFILE_PERIODS` | `48` | Maximum `ChargingSchedulePeriod`s per `SetChargingProfile` (periods are merged to fit). |

## OpenADR 2.0b VEN

See [protocols.md](protocols.md#openadr-20b-ven).

| Variable | Default | Meaning |
|---|---|---|
| `VPP_OPENADR_ENABLED` | `false` | Run the VEN (simple-HTTP pull). Without `VPP_OPENADR_VTN_URL` it runs SIMULATED. |
| `VPP_OPENADR_VTN_URL` | unset | VTN base URL, e.g. `https://vtn.example.com/OpenADR2/Simple/2.0b`. |
| `VPP_OPENADR_VEN_NAME` | `vpp-ven` | VEN name used at registration. |
| `VPP_OPENADR_VEN_ID` | unset | VEN id; normally assigned by the VTN at registration. |
| `VPP_OPENADR_POLL_INTERVAL_S` | unset | Poll period; unset = the frequency the VTN requests. |
| `VPP_OPENADR_AUTO_OPT_IN` | `true` | Opt in to events automatically when the DR orchestrator's auto-response is off. When auto-response is on, the orchestrator decides. |
| `VPP_OPENADR_TIMEOUT_S` | `10.0` | HTTP timeout. |
| `VPP_OPENADR_VERIFY_TLS` | `true` | Verify the VTN certificate. Do not disable outside a lab. |
| `VPP_OPENADR_CA_PATH` | unset | CA bundle for the VTN certificate. |
| `VPP_OPENADR_CERT_PATH` | unset | VEN client certificate (mutual TLS). |
| `VPP_OPENADR_KEY_PATH` | unset | VEN client key. |

## IEEE 2030.5 client

See [protocols.md](protocols.md#ieee-20305-client).

| Variable | Default | Meaning |
|---|---|---|
| `VPP_IEEE2030_5_ENABLED` | `false` | Run the client. Without `VPP_IEEE2030_5_SERVER_URL` it runs SIMULATED. |
| `VPP_IEEE2030_5_SERVER_URL` | unset | Utility server base URL, e.g. `https://utility.example.com:8443`. |
| `VPP_IEEE2030_5_DCAP_PATH` | `/dcap` | DeviceCapability path. |
| `VPP_IEEE2030_5_LFDI` | unset | Device LFDI; unset = derived from the client certificate. |
| `VPP_IEEE2030_5_POLL_INTERVAL_S` | unset | Poll period; unset = the server's `pollRate`. |
| `VPP_IEEE2030_5_TIMEOUT_S` | `10.0` | HTTP timeout. |
| `VPP_IEEE2030_5_VERIFY_TLS` | `true` | Verify the server certificate. |
| `VPP_IEEE2030_5_CA_PATH` | unset | CA bundle for the server certificate. |
| `VPP_IEEE2030_5_CERT_PATH` | unset | Device client certificate (required for mutual TLS). |
| `VPP_IEEE2030_5_KEY_PATH` | unset | Device client key. |
| `VPP_IEEE2030_5_TLS_CIPHERS` | unset | OpenSSL cipher string, e.g. `ECDHE-ECDSA-AES128-CCM8` as the standard requires. |

## Demand-response orchestrator

See [protocols.md](protocols.md#demand-response-orchestrator). Only active when
OpenADR or IEEE 2030.5 is enabled.

| Variable | Default | Meaning |
|---|---|---|
| `VPP_DR_AUTO_RESPONSE_ENABLED` | `false` | Dispatch the fleet in response to grid signals. Off: signals are recorded and shown, nothing is dispatched. |
| `VPP_DR_TICK_INTERVAL_S` | `5.0` | How often the orchestrator re-evaluates active signals. |
| `VPP_DR_REDISPATCH_INTERVAL_S` | `900.0` | Re-plan an ongoing event at least this often (fresh SOC). |
| `VPP_DR_MAX_EXPORT_KW` | unset | Hard cap on any DR export target (kW), applied last. |
| `VPP_DR_MAX_IMPORT_KW` | unset | Hard cap on any DR absorb target (kW), applied last. |
| `VPP_DR_SIMPLE_LEVEL_FRACTIONS` | `[0.0, 0.5, 0.75, 1.0]` | OpenADR `SIMPLE` level 0..3 as a fraction of the fleet's export capability. |
| `VPP_DR_MIN_OPT_IN_FRACTION` | `0.5` | Opt out of an OpenADR event when the fleet can cover less than this fraction of the request. |
| `VPP_DR_INCLUDE_V2G` | `true` | Include plugged-in, V2G-capable vehicles (setpoints sent via OCPP). |
| `VPP_DR_IEEE2030_5_SET_MAX_W` | unset | `setMaxW` used for percentage controls (`opModFixedW`, `opModMaxLimW`); unset = the fleet's current export capability. |

## Device control (setpoint actuator)

See [protocols.md](protocols.md#device-control-setpoint-actuator). Per-device
settings live in each resource's `metadata.modbus.control`.

| Variable | Default | Meaning |
|---|---|---|
| `VPP_CONTROL_ENABLED` | `false` | Global kill switch. While false nothing is written to any device (dispatch reports `disabled`). |
| `VPP_CONTROL_WATCHDOG_INTERVAL_S` | `5.0` | How often expiry, deferred writes and keep-alives are checked. |
| `VPP_CONTROL_EXPIRY_GRACE_S` | `30.0` | A setpoint lives for its dispatch interval plus this, then falls back to `safe_setpoint_kw` or is released. |

## Simulated trading venue

| Variable | Default | Meaning |
|---|---|---|
| `VPP_TRADING_MARKET_DATA_ENABLED` | `true` | Background task that advances simulated prices, matches resting orders and publishes `market_data` events. Everything is labelled `source="simulated"`; no order leaves the process. |
| `VPP_TRADING_MARKET_DATA_INTERVAL_SECONDS` | `5.0` | Tick period. |

## Platform configuration document

| Variable | Default | Meaning |
|---|---|---|
| `VPP_CONFIG_PATH` | unset | YAML platform configuration document (validated against `GET /api/v1/config/schema`) applied at startup **only while no document is stored**. Precedence: newest document stored with `PUT /api/v1/config` (versioned in the database, re-applied on every start) > `VPP_CONFIG_PATH` > built-in defaults. The file is not copied into the database; the first `PUT` supersedes it. A missing or invalid file fails startup. |
| `VPP_DEFAULT_TIMEZONE` | `UTC` | Reported in `GET /api/v1/config`. Tariff simulations take an explicit `timezone`; sites carry their own IANA timezone. |

## Non-`VPP_` variables read by the API

| Variable | Meaning |
|---|---|
| `OPENEI_API_KEY` | API key for importing tariffs from the OpenEI Utility Rate Database (`POST /api/v1/tariffs/import-urdb`). `GET /api/v1/tariffs/import-urdb` reports whether it is set. |
| `FORWARDED_ALLOW_IPS` | Read by uvicorn: addresses whose `X-Forwarded-*` headers are trusted. Set it to your reverse proxy's address. |

## Web console variables

Set in `web/.env.local` for development or as container environment. See
[`web/.env.example`](../web/.env.example).

| Variable | When read | Meaning |
|---|---|---|
| `API_BASE_URL` | runtime (server) | URL the Next.js server uses to reach FastAPI, e.g. `http://vpp-api:8000`. |
| `NEXT_PUBLIC_API_BASE_URL` | build | Legacy fallback when `API_BASE_URL` is unset. |
| `AUTH_COOKIE_NAME` | runtime | Session cookie name (default `vpp_session`). |
| `WS_PUBLIC_URL` | runtime | WebSocket URL the **browser** dials, e.g. `wss://vpp.example.com/api/v1/ws`. |
| `NEXT_PUBLIC_WS_URL` | build | Same, inlined at build time; used when `WS_PUBLIC_URL` is unset. If neither is set it is derived from `API_BASE_URL`. |
| `NEXT_PUBLIC_USE_MOCKS` | build | `1` = demo mode with bundled fake data when API calls fail. Never set in real deployments. |
| `E2E_BASE_URL`, `E2E_PORT`, `E2E_NO_SERVER`, `PLAYWRIGHT_CHROMIUM_EXECUTABLE` | tests | Playwright knobs, see `web/README.md`. |
