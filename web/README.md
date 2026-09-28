# VPP web console (`web/`)

Next.js 15 (App Router) console for the Virtual Power Plant platform: an
operator UI and a customer portal on top of the FastAPI backend.

## Prerequisites

- Node 20+ and pnpm 9 (`corepack enable`; the version is pinned in
  `package.json` → `packageManager`).
- A FastAPI backend, by default on `http://localhost:8000` — see the root
  [README](../README.md#quick-start). Or run without one in mock mode
  (`NEXT_PUBLIC_USE_MOCKS=1`).

## Run

```bash
cd web
cp .env.example .env.local
pnpm install
pnpm dev                  # http://localhost:3000
```

You are redirected to `/login`. Accounts are created by an admin with
`POST /api/v1/auth/register`; the very first admin is created as described
in [docs/deployment.md](../docs/deployment.md#create-the-first-admin).
Customer accounts (role `customer`) land on `/portal`, everyone else on the
operator console.

Backend for local development (from the repository root):

```bash
export VPP_SECRET_KEY=dev-only-secret
vpp migrate
uvicorn vpp.api.app:create_app --factory --reload
```

or with Docker: `docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build`
(needs `VPP_SECRET_KEY` exported).

## Environment variables

See `.env.example` for the annotated list; also in
[docs/configuration.md](../docs/configuration.md#web-console-variables).

| Var | Default | Purpose |
|---|---|---|
| `API_BASE_URL` | falls back to `NEXT_PUBLIC_API_BASE_URL`, then `http://localhost:8000` | Server-side FastAPI URL, e.g. `http://vpp-api:8000` in Docker. |
| `AUTH_COOKIE_NAME` | `vpp_session` | Name of the httpOnly JWT cookie. |
| `WS_PUBLIC_URL` (runtime) / `NEXT_PUBLIC_WS_URL` (build time) | derived from `API_BASE_URL` → `ws(s)://…/api/v1/ws` | WebSocket URL the **browser** dials. Set it whenever `API_BASE_URL` is not reachable from the browser. |
| `NEXT_PUBLIC_USE_MOCKS` | unset | `1` enables demo mode: failed API calls for the customer portal, alerts and sites fall back to bundled demo data and live updates are disabled. **Never** set in real deployments. |

`NEXT_PUBLIC_*` values are inlined at build time.

The server-side proxy (`/api/proxy/*`) and the auth routes forward the
user's address to FastAPI as `X-Forwarded-For` / `X-Real-IP`, so the API can
rate-limit each user separately once this server's address is in the API's
`VPP_TRUSTED_PROXIES` (see
[deployment](../docs/deployment.md#rate-limiting-behind-the-console)).

## Pages

| Route | What it does | Backend |
|---|---|---|
| `/` | Fleet overview: count, online, rated and current kW, sortable resources table (5 s polling) | `/api/v1/resources` |
| `/assets`, `/assets/[id]` | Asset detail: stats, stored 24 h power/SOC history, type-specific panels (battery SOC gauge, solar, wind) | `/api/v1/resources/{id}`, `/metrics` |
| `/alerts` | Live alert feed: severity filters, 24 h volume sparkline, ack / snooze / bulk ack | `/api/v1/alerts`, `alerts` channel |
| `/sites` | MapLibre map (OpenFreeMap tiles, no API key) with site health and a keyboard-accessible list | `/api/v1/sites` |
| `/tariffs`, `/tariffs/[id]` | Tariff list; detail with weekday/weekend TOU heatmaps, components, NEM badge, raw URDB JSON; bill simulator (synthetic load or CSV upload, timezone, NEM regime, compare-to); admin create (from backend presets), edit, delete, URDB import | `/api/v1/tariffs/*` |
| `/trading` | Simulated-venue workspace: markets with live quotes, order ticket (all order types), open orders, history, trades, portfolio and risk; "Simulated venue" badge | `/api/v1/trading/*`, `market_data` channel |
| `/trading/strategies` | Strategy catalogue and backtests | `/api/v1/trading/strategies` |
| `/trading/dispatches` | Dispatch history with filters and a side sheet (inputs, solution, diagnostics, counterfactual explainer) | `/api/v1/dispatches`, `/explain` |
| `/optimization` | Horizon schedule planner (prices or tariff, degradation-aware) and closed-loop MPC backtest | `/api/v1/optimization/schedule`, `/backtest` |
| `/protocols` | Protocol adapters with LIVE / SIMULATED badges, counters, connect / disconnect | `/api/v1/protocols` |
| `/settings` | Platform config YAML editor (self-hosted Monaco), schema validation, server errors inline, diff, apply | `/api/v1/config` |
| `/settings/account` | Change password, log out everywhere, own API keys (new keys shown once) | `/api/auth/password`, `/api/auth/logout-all`, `/api/v1/auth/api-keys` |
| `/settings/users` | Admin: create users, change roles, activate/deactivate, reset passwords, revoke sessions, every API key | `/api/v1/users*`, `/api/v1/auth/api-keys?all=true` |
| `/portal` | Customer overview: energy flow from their live devices, bill summary | `/api/v1/customer/me*` |
| `/portal/account` | Change password, log out everywhere | `/api/auth/password`, `/api/auth/logout-all` |
| `/portal/bill`, `/portal/devices`, `/portal/enroll` | Bill breakdown (or the server's explanation when no bill is possible), devices, DR program enrollment | `/api/v1/customer/*` |

Write actions are hidden from viewers (role from `/api/auth/me`); the
backend enforces every permission anyway.

## Architecture

- **App Router** with three layouts: `app/layout.tsx` (root, providers),
  `app/(operator)/layout.tsx` (sidebar shell) and `app/(customer)/layout.tsx`
  (portal).
- **Server-side auth.** The login form posts to `app/api/auth/login/route.ts`,
  which calls `POST /api/v1/auth/token` with a form body and sets an
  `httpOnly`, `SameSite=Lax` cookie (`Secure` in production builds, so serve
  the console over HTTPS). The token is never exposed to client JS.
  Changing the password (`app/api/auth/password/route.ts`) replaces the
  cookie with the fresh token the backend returns (every older token is
  revoked); "log out everywhere" (`app/api/auth/logout-all/route.ts`)
  revokes all sessions and clears the cookie.
- **API proxy.** Authenticated calls go through
  `app/api/proxy/[...path]/route.ts`, which reads the cookie server-side and
  adds `Authorization: Bearer …`. Client components use an axios instance
  pointed at `/api/proxy`. When FastAPI is unreachable the proxy answers a
  JSON `502`.
- **Middleware.** `middleware.ts` redirects unauthenticated requests to
  `/login` and routes users by the JWT `aud` claim (`customer` → `/portal`).
  That decode is advisory; `/api/auth/me` (backend-validated) is
  authoritative.
- **Live updates.** Route handlers cannot proxy a WebSocket upgrade, so the
  browser connects to FastAPI's `/api/v1/ws` directly. Before each
  (re)connect `lib/ws/client.ts` calls `GET /api/auth/ws-token`, which
  exchanges the session cookie for a 60 s socket-only token
  (`POST /api/v1/ws/token`). The token is sent as the `bearer, <token>`
  subprotocol pair (never in the URL); channels go in the `channels` query
  parameter so reconnects resubscribe. Exponential backoff with jitter, a
  25 s ping, reconnect on close code `4001` (session expired), stop on
  `401`. The topbar badge shows the connection state.
- **API contracts.** `lib/api/*.ts` mirror the backend schemas and validate
  responses with zod; `lib/api/errors.ts` turns FastAPI `detail` payloads
  into readable messages.
- **No silent mocks.** A failing endpoint renders an error state
  (`components/ui/error-state.tsx`) with the HTTP status and Retry. Demo
  data only with `NEXT_PUBLIC_USE_MOCKS=1`.
- **Monaco is self-hosted.** `scripts/copy-monaco.mjs` (run by
  `predev`/`prebuild`) copies `monaco-editor/min/vs` to `public/monaco`.
- **Standalone output.** `next.config.ts` sets `output: "standalone"` for the
  Docker image (`web/Dockerfile`). `pnpm start` still works (Next prints a
  warning); in production run `node .next/standalone/server.js` after
  copying `public/` and `.next/static/` next to it, as the Dockerfile does.

## Verification

```bash
pnpm install
pnpm typecheck
pnpm lint
pnpm build
pnpm exec playwright install chromium   # once
pnpm test:e2e
```

The Playwright specs (`tests/*.spec.ts`) stub every backend call with
`page.route()` / `page.routeWebSocket()`, so no FastAPI server is needed;
`proxy.spec.ts` starts a tiny fake API on `E2E_BACKEND_PORT` (default
`E2E_PORT + 1`, passed to the web server as `API_BASE_URL`) to check the
headers the server-side proxy sends.
They start `pnpm dev` (or `pnpm start` when `CI` is set — run `pnpm build`
first). Knobs: `E2E_PORT`, `E2E_BASE_URL` (reuse a running app),
`E2E_NO_SERVER`, `PLAYWRIGHT_CHROMIUM_EXECUTABLE` (use a preinstalled
Chromium).

## Docker

```bash
docker build -t vpp-web web/
docker run -p 3000:3000 -e API_BASE_URL=http://vpp-api:8000 \
  -e WS_PUBLIC_URL=ws://localhost:8000/api/v1/ws vpp-web
```

`docker-compose.yml` at the repository root runs it as the `vpp-web`
service.

## CI

`.github/workflows/web-ci.yml` runs on changes under `web/**`: a `build` job
(lint, typecheck, build) and an `e2e` job (Playwright/Chromium against a
production build; the HTML report is uploaded on failure).

## Not yet built

Per-site detail pages, an alert-rule editor in the UI (rules are managed
through the API), and user management screens.
