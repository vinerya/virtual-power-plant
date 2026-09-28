# Security

This page describes what the platform protects against, how, and what it
leaves to the operator. It is not a certification; if you connect the
platform to equipment that can move real power, have your deployment
reviewed.

To report a vulnerability, open a GitHub security advisory on the
repository (or contact the maintainers privately) rather than a public
issue.

## Assets and threats

| Asset | Main threats | Mitigations in the platform |
|---|---|---|
| Control of physical devices (chargers via OCPP, DR dispatch) | unauthorised commands; spoofed grid signals; runaway automation | role checks on every write; DR auto-response **off** by default; operator caps (`VPP_DR_MAX_*_KW`) and utility limits clamp every DR target; per-resource limits are hard bounds; OCPP allow-list + Basic auth; OpenADR / IEEE 2030.5 over verified TLS with client certificates |
| Fleet and customer data | cross-tenant reads by customer accounts; token theft | deny-by-default for the `customer` role; customer routes scope every query to the caller (foreign ids → `404`); httpOnly session cookie; short-lived socket tokens |
| Credentials | password / key disclosure | bcrypt password hashes; API keys stored as SHA-256; credentials accepted in request bodies (query-string login is deprecated and flagged) |
| Platform availability | request floods; slow external peers | per-IP rate limit; external peers (VTN, utility server, brokers, devices) run in supervised tasks with backoff and never block startup |
| Server host | file writes through configuration; XML attacks | `PUT /api/v1/config` refuses `monitoring.log_file`; OpenADR and IEEE 2030.5 XML is parsed with entity resolution and network access disabled |

## Authentication

- **Passwords** are hashed with bcrypt. There is no self-registration: only
  admins create users (`POST /api/v1/auth/register`).
- **JWTs** are HS256, signed with `VPP_SECRET_KEY`, and expire after
  `VPP_JWT_EXPIRE_MINUTES` (default 60). Every request re-loads the user, so
  deactivating a user (`users.is_active`; there is no API for it yet) takes
  effect immediately. There are no refresh
  tokens and no server-side revocation list: to invalidate all sessions,
  rotate `VPP_SECRET_KEY`.
- **Audience.** Tokens carry `aud` = `operator` or `customer`, re-checked
  against the user's current role on every request.
- **API keys** (`X-API-Key`) are random 32-byte URL-safe strings with a
  `vpp_` prefix; only their SHA-256 hash is stored and the key is shown
  once. A key acts as the user who created it, narrowed to the *lesser* of
  the key's `role` and that user's current role: an admin can mint a
  `viewer` key that carries only viewer rights, and demoting a user
  immediately narrows all of their keys. Non-admins cannot mint a key for
  a higher role. There is no key listing/revocation endpoint yet;
  deactivating the user disables all of their keys.
- **WebSocket.** Handshakes require a token by default
  (`VPP_WS_AUTH_REQUIRED`). Browsers should use the 60-second socket-only
  token from `POST /api/v1/ws/token`, sent as a `Sec-WebSocket-Protocol`
  value rather than in the URL (query-string tokens end up in proxy logs).
  Socket tokens are rejected by the HTTP API, and open sockets are closed
  (`4001`) when the session behind them expires. Customer accounts cannot
  open the fleet WebSocket.

Not implemented: MFA, password reset, account lockout after failed logins
(the rate limiter is the only brake), SSO/OIDC.

## Authorization (RBAC)

Roles: `admin`, `operator`, `viewer`, `researcher`, `customer`. The full
per-route matrix is in [api.md](api.md#routes); in short:

- **admin** — users, tariffs, platform config, customers and DR programs,
  site deletion, battery SOH recomputation, alert-rule changes.
- **operator** — everything that changes operational state: resources,
  sites and meter data, orders on the simulated venue, V2G vehicles /
  bindings / schedules / dispatch, OCPP remote start/stop, OpenADR opt
  overrides, protocol connect/disconnect, alert ack/snooze/resolve.
- **viewer**, **researcher** — read operator-side data and run computations
  that do not change device or venue state (optimization solves,
  backtests, bill simulations).
- **customer** — only the member portal (`/api/v1/customer/*`) and their
  own sites, devices and meter readings. `get_current_user`, used by every
  operator endpoint, rejects customers with `403`, so a new operator route
  cannot leak fleet data to customers by accident.

`POST /api/v1/optimization/dispatch` computes and records an allocation; it
does not command devices. Only the DR orchestrator (when auto-response is
enabled) and the V2G/OCPP routes send setpoints to equipment.

## Web console

- The login route sets the JWT in an **httpOnly**, `SameSite=Lax` cookie
  (`Secure` in production builds); client JavaScript never sees it. Serve
  the console over HTTPS — browsers drop `Secure` cookies on plain HTTP
  except on `localhost`.
- All API calls go through the Next.js server (`/api/proxy/*`), which adds
  the bearer token server-side.
- The middleware decodes the JWT only to route users between the operator
  console and the portal; the backend enforces every permission, and write
  actions hidden from viewers in the UI are also refused by the API.
- Monaco (the config editor) is self-hosted; the console loads no scripts
  from third-party CDNs. Map tiles come from OpenFreeMap.

## Secrets and configuration

- **`VPP_SECRET_KEY` must be set** to a long random value. The code default
  is public, and the API does not refuse to start with it; the provided
  `docker-compose.yml` does refuse. Generate one with
  `python -c "import secrets; print(secrets.token_urlsafe(48))"`.
- Keep secrets (`VPP_SECRET_KEY`, database password,
  `VPP_OCPP_BASIC_AUTH_PASSWORD`, `VPP_ALERT_WEBHOOK_SECRET`,
  `VPP_METRICS_BEARER_TOKEN`, `VPP_MQTT_PASSWORD`, `OPENEI_API_KEY`) in
  environment variables or your orchestrator's secret store, never in the
  repository. `.env` files are git-ignored; the pre-commit hooks include
  `detect-private-key`.
- Client certificates and keys for OpenADR / IEEE 2030.5 should be mounted
  read-only into the container (e.g. `/etc/vpp/certs`), readable only by the
  service user.
- Leave `VPP_OPENADR_VERIFY_TLS` and `VPP_IEEE2030_5_VERIFY_TLS` on.

## Network exposure

- Terminate TLS at a reverse proxy for the API, the console and the OCPP
  endpoint (`wss://`). OCPP Basic auth over plain `ws://` sends the password
  in clear text.
- `/metrics` is unauthenticated unless `VPP_METRICS_BEARER_TOKEN` is set;
  either set it or keep `/metrics` off the public proxy.
- `/docs`, `/redoc` and `/openapi.json` are public; hide them at the proxy
  if you do not want the API surface advertised.
- Do not publish PostgreSQL. The compose file keeps it on the internal
  network.
- Set `VPP_CORS_ORIGINS` to the exact console origin(s).
- Set uvicorn's `FORWARDED_ALLOW_IPS` to your proxy's address so client IPs
  (used by the rate limiter and logs) are taken from `X-Forwarded-For` only
  when the proxy set it.

## Outbound integrations

- **Alert webhooks** are signed when `VPP_ALERT_WEBHOOK_SECRET` is set:
  `X-VPP-Signature: sha256=<hex>` is the HMAC-SHA256 of
  `"<X-VPP-Timestamp>.<raw body>"`. Receivers should verify the signature
  with a constant-time comparison and reject stale timestamps to prevent
  replay.
- **OpenADR** and **IEEE 2030.5** clients support mutual TLS; IEEE 2030.5
  requires it in practice. OpenADR XML signatures are not implemented.

## Auditability

- Optimization runs (including DR dispatches), trading orders and fills,
  V2G schedules/dispatches with each charger's answer, DR decisions
  (`dr_event_responses`), alerts, and every applied platform configuration
  document are persisted.
- Every HTTP request is logged once (`vpp.access`) with its request id.
- There is no general audit log of *who* changed users, tariffs or
  resources.

## Supply chain

CI runs `bandit` over `src/vpp`. Python dependencies use lower-bound pins in
`pyproject.toml`; the web console installs from a frozen `pnpm-lock.yaml`.
