# Deployment

This guide covers running the platform with Docker Compose, without
Docker, behind a TLS reverse proxy, and with the monitoring stack. Read
[security.md](security.md) before exposing anything beyond localhost.

## Docker Compose

`docker-compose.yml` runs three services:

| Service | Image | Port | Notes |
|---|---|---|---|
| `vpp-api` | `Dockerfile` | 8000 | runs `vpp migrate` (alembic upgrade head) then uvicorn |
| `vpp-web` | `web/Dockerfile` | 3000 | Next.js standalone server |
| `postgres` | `postgres:16-alpine` | internal only | volume `postgres-data` |

```bash
export VPP_SECRET_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(48))")
export POSTGRES_PASSWORD=$(python -c "import secrets; print(secrets.token_urlsafe(24))")
docker compose up -d --build
docker compose logs -f vpp-api
```

Keep those two values: the database password is baked into the volume on
first start, and changing the secret key logs everybody out.

Compose variables:

| Variable | Default | Purpose |
|---|---|---|
| `VPP_SECRET_KEY` | *(required)* | JWT signing key |
| `POSTGRES_PASSWORD` | `vpppass` | database password (set your own) |
| `VPP_API_PORT` | `8000` | host port for the API |
| `VPP_WEB_PORT` | `3000` | host port for the console |
| `VPP_WS_PUBLIC_URL` | `ws://localhost:8000/api/v1/ws` | WebSocket URL the **browser** dials; set to `wss://<host>/api/v1/ws` behind TLS |
| `VPP_RATE_LIMIT_REQUESTS_PER_MINUTE` | `120` | per client; see [rate limiting](#rate-limiting-behind-the-console) |
| `VPP_WEB_IPV4`, `VPP_NETWORK_SUBNET`, `VPP_NETWORK_IP_RANGE` | `172.29.0.10`, `172.29.0.0/24`, `172.29.0.128/25` | fixed console address the API trusts for forwarded client IPs; change together if the subnet collides with a local network |
| `VPP_LOG_LEVEL` | `INFO` | |
| `OPENEI_API_KEY` | empty | enables URDB tariff import |

Any other `VPP_*` setting ([configuration.md](configuration.md)) can be added
under `vpp-api.environment`, e.g. to enable OCPP.

Development overlay (reload on the mounted `src/`, debug logging, Postgres
on `127.0.0.1:5432`):

```bash
export VPP_SECRET_KEY=dev-only-secret
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

### Create the first admin

There is no self-registration. Create the first admin with the CLI, which
prompts for the password (twice, hidden):

```bash
docker compose exec vpp-api vpp users create-admin admin
```

Non-interactively, pipe the password in (`--password-stdin`), point at a
file (`--password-file`) or set `VPP_ADMIN_PASSWORD` for the command. Weak
passwords are refused (see [security.md](security.md#authentication)).
`vpp users set-password <name>` resets a password (and re-activates the
account) if every admin is locked out; `vpp users list` shows accounts.

**Or bootstrap it on first boot.** When `VPP_BOOTSTRAP_ADMIN_USERNAME` and
`VPP_BOOTSTRAP_ADMIN_PASSWORD_FILE` are set and the `users` table is empty,
the API creates that admin at startup. The password is read from the file
(e.g. a Docker/Kubernetes secret), never from the environment, and is never
logged; startup fails if the file is missing or the password is too weak.
Once any user exists both settings are ignored and the file is not read. With
the provided compose file:

```bash
printf '%s' 'a-long-unique-passphrase' > admin_password
chmod 644 admin_password   # readable by the container's non-root user
VPP_BOOTSTRAP_ADMIN_USERNAME=admin VPP_ADMIN_PASSWORD_FILE=./admin_password \
  docker compose up -d
```

Then log in at <http://localhost:3000>. Further users, roles, password
resets, deactivation and API keys are managed in the console under
**Settings → Users & API keys**, or with the `/api/v1/users` API
([api.md](api.md)). Without Docker, run `vpp users create-admin admin` from
the repository root with the same `VPP_DATABASE_URL` (and `VPP_USE_ALEMBIC`)
as the API.

## Without Docker

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[api,db,protocols,solver,degradation,monitoring,cli]"
pip install psycopg2-binary            # only for migrating PostgreSQL

export VPP_DATABASE_URL=postgresql+asyncpg://vpp:<password>@db-host:5432/vpp
export VPP_SECRET_KEY=...
vpp migrate                            # works from any directory
uvicorn vpp.api.app:create_app --factory --host 127.0.0.1 --port 8000
```

- `vpp migrate` uses the migrations packaged in `vpp/migrations`, so an
  installed wheel can migrate from any working directory; no source
  checkout or `alembic.ini` is needed.
- Migrations rewrite `postgresql+asyncpg` to `postgresql+psycopg2`, hence the
  extra `psycopg2-binary`.
- For a quick SQLite setup, skip `vpp migrate`: with the default
  `VPP_USE_ALEMBIC=false` the API creates missing tables on startup (no
  schema upgrades). Set `VPP_USE_ALEMBIC=true` to run migrations at startup
  instead.
- Several workers are supported: `VPP_API_WORKERS=4 vpp serve` (or pass
  `--workers` to uvicorn *and* set `VPP_API_WORKERS` to the same number, so
  the WebSocket relay starts and the startup checks see it). Use PostgreSQL,
  keep host clocks NTP-synchronised, and note that OCPP and device control
  (`VPP_CONTROL_ENABLED`) require a single worker. The rate limit and the
  login throttle are counted in the database for all workers; for several
  single-worker replicas set `VPP_RATE_LIMIT_BACKEND=database`. See
  [architecture](architecture.md#process-model).

Web console:

```bash
cd web
pnpm install --frozen-lockfile
pnpm build
cp -r public .next/standalone/ && cp -r .next/static .next/standalone/.next/
API_BASE_URL=http://127.0.0.1:8000 WS_PUBLIC_URL=wss://vpp.example.com/api/v1/ws \
  PORT=3000 node .next/standalone/server.js
```

(`pnpm start` also works but prints a warning because the build uses
`output: "standalone"`.)

## Database and migrations

- PostgreSQL 14+ is recommended for production (compose uses 16); SQLite
  is fine for development and demos.
- Schema changes ship as alembic migrations in `src/vpp/migrations/versions/`. Upgrade
  procedure: back up, deploy the new image, let `vpp migrate` run (compose
  does it on every start; it is idempotent), check the logs.
- `alembic downgrade` is available from the repository root
  (`alembic downgrade -1`) but data-bearing downgrades are not tested;
  restore from backup instead.
- Back up with `pg_dump` (e.g.
  `docker compose exec postgres pg_dump -U vpp vpp > vpp.sql`).

## Reverse proxy and TLS

Terminate TLS in front of both services. One hostname can serve
everything, because the console only uses `/api/auth/*` and `/api/proxy/*`
under `/api`:

| Path | Upstream | Notes |
|---|---|---|
| `/api/v1/ws` | `vpp-api:8000` | WebSocket upgrade; long read timeout |
| `/ocpp/` | `vpp-api:8000` | OCPP 1.6-J WebSocket upgrade (only when `VPP_OCPP_ENABLED`) |
| `/` | `vpp-web:3000` | console, including its `/api/auth` and `/api/proxy` routes |

Expose the REST API itself (e.g. on `api.example.com` or under a separate
location) only if external clients need it; the console reaches it on the
internal network.

nginx example:

```nginx
map $http_upgrade $connection_upgrade { default upgrade; '' close; }

server {
    listen 443 ssl;
    http2 on;
    server_name vpp.example.com;
    ssl_certificate     /etc/letsencrypt/live/vpp.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/vpp.example.com/privkey.pem;

    # Live updates for the console.
    location = /api/v1/ws {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection $connection_upgrade;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 3600s;
    }

    # OCPP charge points: wss://vpp.example.com/ocpp/<charge_point_id>
    location /ocpp/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection $connection_upgrade;
        proxy_set_header Host $host;
        proxy_set_header Authorization $http_authorization;   # Security Profile 1
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 3600s;                             # > heartbeat interval
    }

    # Keep metrics internal (or set VPP_METRICS_BEARER_TOKEN).
    location = /metrics { return 404; }

    location / {
        proxy_pass http://127.0.0.1:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

With this layout set `VPP_WS_PUBLIC_URL=wss://vpp.example.com/api/v1/ws`
and `VPP_CORS_ORIGINS='["https://vpp.example.com"]'`, and set uvicorn's
`FORWARDED_ALLOW_IPS` (environment variable) to the proxy's address so the
API trusts `X-Forwarded-For` only from it.

OCPP notes:

- Chargers must offer the `ocpp1.6` subprotocol; nginx passes
  `Sec-WebSocket-Protocol` through unchanged.
- Some chargers cannot validate modern certificate chains or SNI; test
  with the actual hardware.
- Keep `proxy_read_timeout` above `VPP_OCPP_HEARTBEAT_INTERVAL_S`.

The console's session cookie is marked `Secure` in production builds, so
the console must be served over HTTPS (browsers make an exception only for
`http://localhost`).

## Rate limiting behind the console

The rate limiter is keyed by client IP. The console calls the API from its
own server and forwards the user's address in `X-Forwarded-For` /
`X-Real-IP`; the API honours those headers only from addresses in
`VPP_TRUSTED_PROXIES` (default: none, so every console user would share the
console server's bucket). The compose file gives `vpp-web` a fixed address
on its own subnet, trusts exactly that address, and keeps the per-client
limit at 120 requests per minute.

For other deployments set `VPP_TRUSTED_PROXIES` to the console server's
address as the API sees it (plus any reverse proxy in front of the API).
Trust only addresses that cannot be reached by clients directly, e.g. not a
Docker bridge gateway that published ports are forwarded through.

The console takes the client address from the `X-Forwarded-For` it receives
(Next.js fills it from the connection when absent). If the console is
exposed without a reverse proxy, a client can put an arbitrary address in
that header and be rate-limited under it; to prevent that, put a proxy in
front that overwrites the header (nginx:
`proxy_set_header X-Forwarded-For $remote_addr;`), or leave
`VPP_TRUSTED_PROXIES` empty and raise the limit. uvicorn's own
`FORWARDED_ALLOW_IPS` (default `127.0.0.1`) also rewrites the client address
before the rate limiter sees it.

## Monitoring

```bash
docker compose -f docker-compose.yml -f monitoring/docker-compose.monitoring.yml up -d
```

- Prometheus on <http://localhost:9090> scrapes `vpp-api:8000/metrics` and
  node-exporter (`monitoring/prometheus/prometheus.yml`). If you set
  `VPP_METRICS_BEARER_TOKEN`, uncomment the `authorization` block there.
- Grafana on <http://localhost:3001> (port 3001 avoids the console on 3000;
  user `admin`, password `$GRAFANA_ADMIN_PASSWORD`, default `vpp-grafana` —
  change it). Dashboards "VPP Overview", "VPP Trading" and "VPP Fleet" are
  provisioned from `monitoring/grafana/dashboards/`.
- Metrics include HTTP request count/latency by route template,
  per-resource power/SOC gauges and telemetry freshness, optimization and
  trading counters, protocol errors and alerts fired. See
  `src/vpp/metrics.py`.
- Logs: set `VPP_LOG_JSON=true` (the default when `VPP_ENV=production`) and
  ship stdout to your log stack; correlate with `X-Request-ID`.
- `GET /health` is a liveness check. `GET /ready` runs `SELECT 1` against
  the database and returns 503 when it can't, so use it as the readiness
  probe.

## Enabling integrations

Protocols are opt-in; add their settings to the API environment and restart.
Recommended order for a new site:

1. Resources and sites via the API or console; telemetry via MQTT, Modbus
   or `POST /api/v1/resources/{id}/telemetry`.
2. OCPP with an allow-list and Basic auth; verify chargers appear under
   `/protocols` as LIVE.
3. OpenADR / IEEE 2030.5 with certificates; watch
   `GET /api/v1/dr/status` and `/api/v1/dr/responses` with auto-response
   **off**.
4. Only then consider `VPP_DR_AUTO_RESPONSE_ENABLED=true`, with
   `VPP_DR_MAX_EXPORT_KW` / `VPP_DR_MAX_IMPORT_KW` caps. Stationary
   assets are only driven once `VPP_CONTROL_ENABLED=true` and each device has
   a `metadata.modbus.control` block — try `"simulate": true` first (see
   [protocols.md](protocols.md#device-control-setpoint-actuator)).
