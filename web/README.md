# VPP Operator Console (web/)

Next.js 15 operator console for the open-source Virtual Power Plant. Milestone 1: project scaffold, JWT auth, and read-only Fleet Overview.

## Prerequisites

- Node 20+
- pnpm 9+ (chosen over npm for fast installs and a deterministic
  `pnpm-lock.yaml`. If you don't have pnpm: `corepack enable && corepack prepare pnpm@9 --activate`. npm works too but you'll have to delete `pnpm-lock.yaml` and regenerate `package-lock.json`.)
- A running FastAPI backend on `http://localhost:8000` (see below).

## Run the backend

From the repo root:

```bash
docker-compose -f docker-compose.dev.yml up
```

Or, without Docker:

```bash
pip install -e .[dev]
uvicorn vpp.api.app:app --reload
```

## Run the web app

```bash
cd web
cp .env.example .env.local
pnpm install
pnpm dev
```

Open <http://localhost:3000>. You'll be redirected to `/login`. Use any user
created via `POST /api/v1/auth/register` (admin only) — the dev compose stack
seeds an `operator` user by default.

## Environment variables

| Var | Default | Purpose |
|---|---|---|
| `NEXT_PUBLIC_API_BASE_URL` | `http://localhost:8000` | Public-facing FastAPI URL (used in errors / debug). |
| `API_BASE_URL` | falls back to `NEXT_PUBLIC_API_BASE_URL` | Server-side URL — useful when the Next.js server is in Docker and needs `http://api:8000`. |
| `AUTH_COOKIE_NAME` | `vpp_session` | Name of the httpOnly JWT cookie. |

## Architecture

- **App Router** with two layouts: `app/layout.tsx` (root, providers) and
  `app/(operator)/layout.tsx` (sidebar shell).
- **Server-side auth.** Login form POSTs to `app/api/auth/login/route.ts`
  which calls `POST /api/v1/auth/token` and sets an `httpOnly`, `SameSite=Lax`
  cookie. The token is **never** exposed to client JS.
- **API proxy.** All authenticated calls go through
  `app/api/proxy/[...path]/route.ts`. The proxy reads the cookie server-side
  and attaches `Authorization: Bearer …` before forwarding to FastAPI. Client
  components use a thin axios instance pointed at `/api/proxy`.
- **Middleware.** `middleware.ts` redirects unauthenticated requests to
  `/login` (the `/login` and `/api/auth/*` paths are public).
- **TanStack Query** for caching and 5s polling on the Fleet page.

## What's implemented (M1)

- Login + logout, httpOnly cookie session, middleware-enforced auth.
- Operator shell with side nav (Fleet wired up; Assets / Trading / Tariffs /
  Alerts / Settings stubbed as "Coming soon").
- Fleet Overview: 4-stat header (count, online, rated kW, current kW),
  sortable resources table, last-refresh indicator, polling at 5s.
- Health pill in topbar reading `/health`.
- Playwright smoke test that mocks the proxy and exercises login + fleet
  render.

## What's implemented (M2)

- **Asset detail** at `/assets/[id]` — header (name, type, online pill),
  stats (rated, current, efficiency, last update), 24h time-series chart
  (lazy-loaded Recharts), and subtype-specific panels (battery SOC gauge,
  solar irradiance + DC/AC capacity, wind speed + cut-in/out, default
  key/value table).
- **Dispatch history** at `/trading/dispatches` — date-range filter
  (24h/7d/30d/custom), multi-resource filter, paginated table, and a
  side sheet with Inputs / Solution / Diagnostics / Counterfactual tabs.
  The sheet is keyboard-accessible (Esc closes, focus trap, focus
  restoration on close).
- **Live updates** via the WebSocket at `/ws`. The singleton client in
  `lib/ws/client.ts` subscribes to `resource_updates`,
  `optimization_events`, and `alerts`; messages invalidate TanStack
  Query caches and pop toasts via `sonner`. Mounted at the operator
  layout level so all child pages benefit.
- shadcn primitives: `sheet`, `tabs`, `select` (hand-authored, no extra
  Radix deps).

### Stubs / fallbacks documented for M3

- **Resource metrics endpoint** (`GET /api/v1/resources/{id}/metrics`) —
  not yet exposed by the backend. The asset page falls back to building a
  rolling 24h client-side buffer from the polled `/resources/{id}`
  snapshot (one point per 10s). When the backend lands the endpoint,
  delete the buffer in `components/asset/asset-detail.tsx`.
- **Dispatch history endpoint** — `GET /api/v1/optimization/history`
  exists upstream but does *not* yet return per-run inputs/solution
  payloads needed by the side sheet. When the response is missing or
  404s, the page reads from a small `localStorage`-backed log
  (`lib/dispatch/local-log.ts`) populated by UI-driven submissions.
  This is a temporary stub awaiting the M3 backend work.
- **WebSocket auth.** The current FastAPI `/ws` endpoint is open. The
  Next.js JWT cookie is `httpOnly` and not readable from the client, so
  if/when the backend adds auth the recommended exchange is a one-shot
  ticket route under `app/api/auth/ws-ticket` (server reads the cookie
  and returns a short-lived signed token), then
  `new WebSocket('/ws?ticket=...')`. Tracked for M3.

## Coming later

- M3: dispatch explainer (counterfactuals), tariff bill simulator UI,
  settings YAML editor, customer portal.

## Verification

```bash
pnpm install
pnpm typecheck
pnpm lint
pnpm build
pnpm test:e2e   # requires Playwright browsers: pnpm exec playwright install
```

If `pnpm install` fails offline, the source still builds cleanly for anyone
with network access — versions in `package.json` are pinned to current
stable releases.

## CI

The GitHub Actions workflow lives at the repo root in
`.github/workflows/web-ci.yml` (the project already has top-level workflows,
so it's colocated rather than duplicated under `web/.github/workflows/`).
It runs on PRs that touch `web/**`.
