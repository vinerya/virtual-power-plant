# API guide

The FastAPI server documents every route interactively at `/docs` (Swagger
UI) and `/redoc`; the machine-readable schema is `/openapi.json`. This page
covers what the schema does not: how to authenticate, who may call what,
the WebSocket protocol, and conventions shared by all routes.

## Conventions

- **Base path.** Everything except health, metrics and the OCPP socket lives
  under `/api/v1`.
- **Trailing slashes.** `/api/v1/resources` and `/api/v1/resources/` are
  served identically (no `307` redirect, so clients that drop
  `Authorization` on redirects keep working).
- **Request ids.** Every response carries `X-Request-ID`. A sane incoming
  `X-Request-ID` is reused; otherwise one is generated. The id is bound to
  every log line written while handling the request.
- **Errors.** FastAPI's shape: `{"detail": ...}`. `detail` is a string, a
  list of validation errors (`422`), or an object for domain errors, e.g.
  a trading risk rejection:
  `422 {"detail": {"code": "risk_limit_breached", "message": "...", "reasons": [...], "order_id": "..."}}`.
- **Rate limiting.** When `VPP_RATE_LIMIT_ENABLED` (default), each client IP
  gets `VPP_RATE_LIMIT_REQUESTS_PER_MINUTE` requests per minute; beyond that
  the API answers `429`.
- **Units and signs.** Power in kW, energy in kWh, prices in $/kWh for
  tariffs and $/MWh on the trading venue. Dispatch targets are
  **export-positive**: a positive `target_power_kw` delivers power to the
  grid, a negative one absorbs (batteries charge).

## Authentication

### Users and roles

| Role | Intended for | Can |
|---|---|---|
| `admin` | platform administrators | everything, including users, tariffs, config, customers/programs, deleting sites |
| `operator` | control-room staff | read everything operator-side; create/update/delete resources; trade on the simulated venue; V2G and protocol actions; ack/snooze/resolve alerts; sites and meter data |
| `viewer` | read-only staff | read operator-side data; run computations that change no device or venue state (dispatch/schedule/backtest solves, strategy backtests, bill simulations), create their own API key |
| `researcher` | analysts | same effective permissions as `viewer` today |
| `customer` | a household or C&I site owner | only `/api/v1/customer/*` and reads of their own sites, devices and metering; refused on every other route (`403`) and on the WebSocket |

"Operator-side" below means any role except `customer`.

There is no self-registration. `POST /api/v1/auth/register` is admin-only,
so the first admin is created out of band, see
[deployment.md](deployment.md#create-the-first-admin).

### Access tokens (JWT)

```bash
# OAuth2 password grant (form body) ...
curl -s -X POST http://localhost:8000/api/v1/auth/token \
  -d username=admin -d password='change-me-now'
# ... or JSON
curl -s -X POST http://localhost:8000/api/v1/auth/token \
  -H 'Content-Type: application/json' \
  -d '{"username": "admin", "password": "change-me-now"}'
# -> {"access_token": "eyJ...", "token_type": "bearer", "expires_in": 3600}
```

Send it as `Authorization: Bearer <access_token>`. Tokens are HS256 JWTs
signed with `VPP_SECRET_KEY` and expire after `VPP_JWT_EXPIRE_MINUTES`.
There are no refresh tokens: log in again.

Claims: `sub` (user id), `username`, `role`, `exp`, and `aud` —
`"customer"` for customer accounts, `"operator"` for everyone else. The web
console's middleware routes on `aud`; the API re-checks it against the
user's *current* role on every request, so a token minted before a role
change that moves the user between consoles is rejected (`401`). Inactive
users cannot log in and their existing tokens stop working.

Passing `username`/`password` as query parameters still works but is
**deprecated** (credentials end up in access logs): the response carries
`Deprecation: true` and a `Warning` header.

`GET /api/v1/auth/me` returns the caller, including `audience`.

### API keys

```bash
curl -s -X POST http://localhost:8000/api/v1/auth/api-key \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"name": "scada-bridge", "role": "operator"}'
# -> {"key": "vpp_...", ...}   shown once; only its SHA-256 is stored
curl -s http://localhost:8000/api/v1/resources -H "X-API-Key: vpp_..."
```

A non-admin can only mint a key with their own role. Requests made with a
key act as the user who created it, limited to the lesser of the key's
`role` and that user's current role, so an admin can hand an integration
a `viewer` key without giving away admin rights. There is no endpoint to list or revoke keys yet; setting the
user inactive (`users.is_active`, database only) disables all of them.

### Metrics

`GET /metrics` is unauthenticated unless `VPP_METRICS_BEARER_TOKEN` is set,
in which case it requires `Authorization: Bearer <that token>`.

## WebSocket

Live events are pushed on `ws(s)://<api>/api/v1/ws` (`/ws` is a legacy
alias with the same protocol).

### Authenticating the handshake

Browsers cannot set headers on a WebSocket upgrade, so the token is
accepted from (first match wins):

1. the `token` query parameter,
2. the `Sec-WebSocket-Protocol` header as the pair `bearer, <token>` (the
   server then selects the `bearer` subprotocol),
3. an `Authorization: Bearer <token>` header (non-browser clients).

Use a short-lived socket token rather than the session JWT, so the latter
never reaches browser JavaScript or URLs:

```bash
curl -s -X POST http://localhost:8000/api/v1/ws/token -H "Authorization: Bearer $TOKEN"
# -> {"token": "eyJ...", "expires_in": 60, "channels": ["*", "alerts", ...]}
```

The socket token is only accepted by the WebSocket (the HTTP API rejects
it), must be used within `VPP_WS_TOKEN_EXPIRE_SECONDS` (default 60 s), and
carries the expiry of the session it was minted from. API keys can mint
socket tokens too; the session then lasts `VPP_JWT_EXPIRE_MINUTES`.

With `VPP_WS_AUTH_REQUIRED=true` (default) a handshake without a valid
token is refused. Customer accounts are always refused: every channel
carries fleet-wide data.

### Close codes

| Code | When | Client action |
|---|---|---|
| `1008` | handshake refused: missing, invalid or expired token, inactive user, customer account (ASGI servers surface this as HTTP `403` on the upgrade) | get a new token; if that fails with `401`, log in again |
| `4001` | the credential behind an open socket expired (reason `"Token expired"`) | reconnect with a fresh socket token (the console does this automatically) |

### Messages

Subscribe on connect with `?channels=resource_updates,alerts`, or send:

```json
{"action": "subscribe", "channel": "resource_updates"}
{"action": "unsubscribe", "channel": "resource_updates"}
{"action": "ping"}
```

The server answers `{"ack": "subscribed:<channel>"}`,
`{"ack": "unsubscribed:<channel>"}`, `{"pong": "<iso time>"}` or
`{"error": "..."}` (unknown channel/action, invalid JSON). Broadcasts look
like:

```json
{
  "channel": "resource_updates",
  "data": {
    "event_id": "...", "event_type": "resource_updated",
    "data": {"resource_id": "...", "power_kw": 3.2, "soc": 0.61},
    "source": "...", "severity": "info", "timestamp": "..."
  },
  "timestamp": "2026-01-01T12:00:00+00:00"
}
```

For event-bus broadcasts `data` is the event envelope shown above. On the
`alerts` channel `data` is the newly fired alert as stored (same shape as
`GET /api/v1/alerts/{id}`); acknowledging or resolving an alert is not
broadcast.

### Channels

| Channel | Carries |
|---|---|
| `resource_updates` | resource added/removed/updated/fault (telemetry, OCPP meter values, SOC) |
| `optimization_events` | optimization started/completed/failed, dispatch executed |
| `market_data` | simulated venue quotes, order submitted/filled/cancelled/rejected, trades |
| `alerts` | newly fired alerts only |
| `grid_events` | protocol connected/disconnected/error, DR event received / response sent, EV connected/disconnected, V2G dispatch/schedule, islanding, load shed |
| `system` | every other event type (tariff, config, ...) |
| `*` | everything |

A client should expect event types it does not know on `system` and ignore
them.

## Routes

Generated from the application (`create_app()` routes and their auth
dependencies). Trailing-slash variants are omitted.
`POST /api/v1/optimization/mpc` (alias of `/schedule`),
`GET /api/v1/optimization/explain/{run_id}` and `PUT /api/v1/alerts/rules/{rule_id}`
exist but are hidden from the OpenAPI schema. `GET /metrics` needs a bearer
token only when `VPP_METRICS_BEARER_TOKEN` is set. The OCPP charge-point
socket `/ocpp/{charge_point_id}` is described in
[protocols.md](protocols.md#ocpp-16-j-central-system).

| Method | Path | Who |
|---|---|---|
| GET | `/metrics` | public, or bearer token (`VPP_METRICS_BEARER_TOKEN`) |
| GET | `/api/v1/alerts/rules` | any operator-side role |
| POST | `/api/v1/alerts/rules` | admin |
| GET | `/api/v1/alerts/rules/{rule_id}` | any operator-side role |
| PUT | `/api/v1/alerts/rules/{rule_id}` | admin |
| PATCH | `/api/v1/alerts/rules/{rule_id}` | admin |
| DELETE | `/api/v1/alerts/rules/{rule_id}` | admin |
| GET | `/api/v1/alerts` | any operator-side role |
| GET | `/api/v1/alerts/{alert_id}` | any operator-side role |
| POST | `/api/v1/alerts/{alert_id}/ack` | admin, operator |
| POST | `/api/v1/alerts/{alert_id}/snooze` | admin, operator |
| POST | `/api/v1/alerts/{alert_id}/resolve` | admin, operator |
| GET | `/health` | public |
| GET | `/ready` | public |
| GET | `/version` | public |
| POST | `/api/v1/auth/token` | public |
| POST | `/api/v1/auth/register` | admin |
| GET | `/api/v1/auth/me` | any role (customers: own data only) |
| POST | `/api/v1/auth/api-key` | any operator-side role |
| GET | `/api/v1/resources` | any operator-side role |
| POST | `/api/v1/resources` | admin, operator |
| GET | `/api/v1/resources/{resource_id}` | any operator-side role |
| PUT | `/api/v1/resources/{resource_id}` | admin, operator |
| DELETE | `/api/v1/resources/{resource_id}` | admin, operator |
| POST | `/api/v1/optimization/dispatch` | any operator-side role; `"apply": true` (write device setpoints) needs admin, operator |
| GET | `/api/v1/optimization/setpoints` | any operator-side role |
| POST | `/api/v1/optimization/stochastic` | any operator-side role |
| POST | `/api/v1/optimization/realtime` | any operator-side role |
| POST | `/api/v1/optimization/distributed` | any operator-side role |
| POST | `/api/v1/optimization/mpc` | any operator-side role |
| POST | `/api/v1/optimization/schedule` | any operator-side role |
| POST | `/api/v1/optimization/backtest` | any operator-side role |
| GET | `/api/v1/optimization/history` | any operator-side role |
| GET | `/api/v1/optimization/runs` | any operator-side role |
| GET | `/api/v1/optimization/runs/{run_id}` | any operator-side role |
| GET | `/api/v1/optimization/explain/{run_id}` | any operator-side role |
| GET | `/api/v1/optimization/runs/{run_id}/explain` | any operator-side role |
| GET | `/api/v1/optimization/stats` | any operator-side role |
| GET | `/api/v1/dispatches` | any operator-side role |
| GET | `/api/v1/dispatches/{run_id}` | any operator-side role |
| GET | `/api/v1/dispatches/{run_id}/explain` | any operator-side role |
| POST | `/api/v1/trading/orders` | admin, operator |
| GET | `/api/v1/trading/orders` | any operator-side role |
| GET | `/api/v1/trading/orders/{order_id}` | any operator-side role |
| DELETE | `/api/v1/trading/orders/{order_id}` | admin, operator |
| POST | `/api/v1/trading/orders/{order_id}/cancel` | admin, operator |
| GET | `/api/v1/trading/trades` | any operator-side role |
| GET | `/api/v1/trading/portfolio` | any operator-side role |
| GET | `/api/v1/trading/markets` | any operator-side role |
| POST | `/api/v1/trading/markets/tick` | admin, operator |
| GET | `/api/v1/trading/strategies` | any operator-side role |
| POST | `/api/v1/trading/strategies/{name}/backtest` | any operator-side role |
| POST | `/api/v1/trading/strategies/{name}/run` | admin, operator |
| GET | `/api/v1/config` | any operator-side role |
| GET | `/api/v1/config/schema` | any operator-side role |
| PUT | `/api/v1/config` | admin |
| POST | `/api/v1/config/validate` | any operator-side role |
| POST | `/api/v1/protocols/{name}/connect` | admin, operator |
| POST | `/api/v1/protocols/{name}/disconnect` | admin, operator |
| GET | `/api/v1/protocols/{name}/metrics` | any operator-side role |
| POST | `/api/v1/v2g/vehicles` | admin, operator |
| GET | `/api/v1/v2g/vehicles` | any operator-side role |
| GET | `/api/v1/v2g/vehicles/{ev_id}` | any operator-side role |
| PATCH | `/api/v1/v2g/vehicles/{ev_id}` | admin, operator |
| DELETE | `/api/v1/v2g/vehicles/{ev_id}` | admin, operator |
| PUT | `/api/v1/v2g/vehicles/{ev_id}/binding` | admin, operator |
| DELETE | `/api/v1/v2g/vehicles/{ev_id}/binding` | admin, operator |
| GET | `/api/v1/v2g/sessions` | any operator-side role |
| GET | `/api/v1/v2g/fleet` | any operator-side role |
| GET | `/api/v1/v2g/flexibility` | any operator-side role |
| POST | `/api/v1/v2g/schedule` | admin, operator |
| GET | `/api/v1/v2g/schedules` | any operator-side role |
| POST | `/api/v1/v2g/dispatch` | admin, operator |
| POST | `/api/v1/v2g/bid` | admin, operator |
| GET | `/api/v1/v2g/bids` | any operator-side role |
| GET | `/api/v1/v2g/metrics` | any operator-side role |
| POST | `/api/v1/tariffs` | admin |
| GET | `/api/v1/tariffs` | any operator-side role |
| GET | `/api/v1/tariffs/presets` | any operator-side role |
| GET | `/api/v1/tariffs/presets/{preset_id}` | any operator-side role |
| GET | `/api/v1/tariffs/import-urdb` | any operator-side role |
| GET | `/api/v1/tariffs/{tariff_id}` | any operator-side role |
| PUT | `/api/v1/tariffs/{tariff_id}` | admin |
| DELETE | `/api/v1/tariffs/{tariff_id}` | admin |
| POST | `/api/v1/tariffs/{tariff_id}/simulate` | any operator-side role |
| POST | `/api/v1/tariffs/simulate` | any operator-side role |
| POST | `/api/v1/tariffs/import-urdb` | admin |
| GET | `/api/v1/batteries/{battery_id}/soh` | any operator-side role |
| GET | `/api/v1/batteries/{battery_id}/soh/history` | any operator-side role |
| POST | `/api/v1/batteries/{battery_id}/soh/update` | admin |
| GET | `/api/v1/protocols/ocpp/charge-points` | any operator-side role |
| GET | `/api/v1/protocols/ocpp/charge-points/{cp_id}` | any operator-side role |
| GET | `/api/v1/protocols/ocpp/transactions` | any operator-side role |
| POST | `/api/v1/protocols/ocpp/charge-points/{cp_id}/remote-start` | admin, operator |
| POST | `/api/v1/protocols/ocpp/charge-points/{cp_id}/remote-stop` | admin, operator |
| GET | `/api/v1/protocols/openadr/events` | any operator-side role |
| GET | `/api/v1/protocols/openadr/events/{event_id}` | any operator-side role |
| POST | `/api/v1/protocols/openadr/events/{event_id}/opt` | admin, operator |
| GET | `/api/v1/protocols/ieee2030_5/controls` | any operator-side role |
| GET | `/api/v1/dr/status` | any operator-side role |
| GET | `/api/v1/dr/responses` | any operator-side role |
| GET | `/api/v1/resources/{resource_id}/metrics` | any role (customers: own data only) |
| POST | `/api/v1/resources/{resource_id}/telemetry` | admin, operator |
| GET | `/api/v1/sites` | any role (customers: own data only) |
| POST | `/api/v1/sites` | admin, operator |
| GET | `/api/v1/sites/{site_id}` | any role (customers: own data only) |
| PATCH | `/api/v1/sites/{site_id}` | admin, operator |
| DELETE | `/api/v1/sites/{site_id}` | admin |
| POST | `/api/v1/sites/{site_id}/meter-readings` | admin, operator |
| GET | `/api/v1/sites/{site_id}/meter-readings` | any role (customers: own data only) |
| GET | `/api/v1/customer/me` | customer |
| GET | `/api/v1/customer/me/bill` | customer |
| GET | `/api/v1/customer/me/devices` | customer |
| GET | `/api/v1/customer/programs` | customer |
| POST | `/api/v1/customer/enrollments` | customer |
| DELETE | `/api/v1/customer/enrollments/{program_id}` | customer |
| POST | `/api/v1/customers` | admin |
| GET | `/api/v1/customers` | admin, operator |
| GET | `/api/v1/customers/{customer_id}` | admin, operator |
| PATCH | `/api/v1/customers/{customer_id}` | admin |
| GET | `/api/v1/customers/{customer_id}/bill` | admin, operator |
| GET | `/api/v1/customers/{customer_id}/devices` | admin, operator |
| GET | `/api/v1/programs` | admin, operator |
| POST | `/api/v1/programs` | admin |
| PATCH | `/api/v1/programs/{program_id}` | admin |
| POST | `/api/v1/ws/token` | any operator-side role |

"Customers: own data only" means a customer gets their own sites, devices
and meter readings; other ids return `404` rather than `403`.
