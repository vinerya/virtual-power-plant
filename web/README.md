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

See `.env.example` for the annotated list.

| Var | Default | Purpose |
|---|---|---|
| `API_BASE_URL` | falls back to `NEXT_PUBLIC_API_BASE_URL`, then `http://localhost:8000` | Server-side FastAPI URL — useful when the Next.js server is in Docker and needs `http://api:8000`. |
| `AUTH_COOKIE_NAME` | `vpp_session` | Name of the httpOnly JWT cookie. |
| `WS_PUBLIC_URL` (runtime) / `NEXT_PUBLIC_WS_URL` (build time) | derived from `API_BASE_URL` → `ws(s)://…/api/v1/ws` | WebSocket URL the **browser** dials. Set it whenever `API_BASE_URL` is not reachable from the browser. |
| `NEXT_PUBLIC_USE_MOCKS` | unset | `1` enables demo mode: failed API calls for the customer portal, alerts and sites fall back to bundled demo data and live updates are disabled. **Never** set in real deployments. |

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
- **Live updates (WebSocket).** Route handlers cannot proxy a WebSocket
  upgrade, so the browser connects to FastAPI's `/api/v1/ws` directly.
  Before each (re)connect `lib/ws/client.ts` calls `GET /api/auth/ws-token`,
  which exchanges the httpOnly session cookie for a short-lived, socket-only
  token (`POST /api/v1/ws/token` on FastAPI, 60s TTL, rejected by the HTTP
  API). The token is sent as the `bearer, <token>` WebSocket subprotocol
  pair (never in the URL); initial channels go in the `channels` query
  param so reconnects resubscribe automatically. Reconnects use
  exponential backoff with jitter; a 25s `ping` keeps idle proxies from
  dropping the socket. The topbar badge shows the connection state.
- **No silent mocks.** A missing/failing endpoint renders an error state
  (`components/ui/error-state.tsx`) with the HTTP status and a Retry
  button. Demo data is only used with `NEXT_PUBLIC_USE_MOCKS=1`.
- **Monaco is self-hosted.** `scripts/copy-monaco.mjs` (run by
  `predev`/`prebuild`) copies `monaco-editor/min/vs` to `public/monaco`, so
  the settings editor never loads code from a third-party CDN.
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
- **Live updates** via the authenticated WebSocket at `/api/v1/ws` (see
  Architecture). The singleton client in
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
  payloads needed by the side sheet. Runs without those fields render
  with the corresponding tabs empty; a failing request shows an error
  state (there is no client-side substitute log).
- **WebSocket auth.** Done: see "Live updates (WebSocket)" under
  Architecture (`app/api/auth/ws-token` + FastAPI `POST /api/v1/ws/token`).

## What's implemented (M3)

- **Dispatch explainer** wired into the existing dispatch side-sheet
  ("Counterfactual" tab). Side-by-side cost bar, per-step net-power
  overlay, free-text rationale, and a binding-constraints list. Calls
  `GET /api/v1/dispatches/{run_id}/explain`; on 404, shows a clear
  empty state.
- **Tariffs page** (`/tariffs` and `/tariffs/[id]`). Left list with
  search and utility filter; detail panel with three tabs:
  - **Schedule**: 12 × 24 calendar heatmap of TOU rates (hand-rendered
    with CSS grid).
  - **Components**: list of energy/demand/fixed/min-bill components.
  - **Simulate**: file upload (CSV `timestamp,kw`) or "Use synthetic
    30-day load" toggle, with optional "compare to" picker. Renders a
    horizontal stacked bar of line items via Recharts plus a
    copy-to-clipboard JSON button.
  If `/api/v1/tariffs` fails the page shows an error state; bundled
  starter tariffs are available separately via `app/api/tariff-presets`.
- **Settings YAML editor** (`/settings`). Monaco lazy-loaded via
  `next/dynamic` (`ssr:false`). Schema-driven client-side validation
  (Ajv against `/api/v1/config/schema`); falls back to YAML-syntax-only
  checks when the schema endpoint 404s. "Validate" surfaces errors
  inline; "Apply" requires a confirmation modal and shows a sonner
  toast. Hand-rolled line-by-line LCS diff (no `react-diff-view`
  dependency) for unsaved changes.

### Backend contracts assumed (mock at the proxy if not yet shipped)

| Endpoint | Used by | Fallback |
|---|---|---|
| `GET /api/v1/dispatches/{run_id}/explain` | Explainer tab | Empty state on 404 |
| `GET /api/v1/tariffs` | Tariff list | Error state |
| `GET /api/v1/tariffs/{id}` | Tariff detail | Error state |
| `POST /api/v1/tariffs/{id}/simulate` | Bill simulator | Toast error |
| `GET /api/v1/config` | Settings live config | hard error |
| `PUT /api/v1/config` | Apply config | hard error |
| `GET /api/v1/config/schema` | Client-side validation | YAML-only checks |
| `POST /api/v1/config/validate` (optional) | Server-side dry-run | Ajv only |

### M4 additions

- **Alerts feed** (`/alerts`). Live-tailed list backed by `GET /api/v1/alerts`
  + WebSocket `alerts` channel. Severity filter chips, 24-hour stacked
  sparkline of alert volume, per-row Acknowledge / Snooze (15m/1h/4h/24h)
  / Open Source actions, bulk-ack via checkbox selection (Space toggles
  the focused row). High-severity alerts continue to fire a `sonner`
  toast through `live-updates.tsx`. WS-pushed alerts pulse briefly when
  they appear at the top of the list. Empty state is "All clear".
- **Sites map** (`/sites`). MapLibre GL JS via `react-map-gl/maplibre`,
  lazy-loaded with `next/dynamic` (`ssr:false`). Tiles come from
  [OpenFreeMap](https://openfreemap.org) (`tiles.openfreemap.org/styles/positron`)
  — OSM-derived, no API key, free for unlimited use, recommended drop-in
  for Mapbox. In mock mode only (`NEXT_PUBLIC_USE_MOCKS=1`) a failing
  `GET /api/v1/sites` falls back to a synthesized site list (group
  resources by `metadata.site_id`, geocode by
  `metadata.location.{lat,lon}`, scatter the rest on a deterministic
  continental-US grid); otherwise the error is shown. Markers are color-coded by aggregate
  health (green/yellow/red based on alert volume, SOH < 0.85, and
  offline-resource count). Hover shows a popover; the synced
  click-to-select sidebar list is the keyboard fallback for screen
  readers (markers are buttons but the list is the canonical input
  device for non-pointer users).
- **Customer portal** (`/portal`, `/portal/bill`, `/portal/devices`,
  `/portal/enroll`). Stripped-down layout under
  `app/(customer)/layout.tsx` — minimal top bar, no operator nav,
  lighter background. Uses the same shadcn primitives but with more
  whitespace. The `/portal/bill` page reuses the operator's
  `BillBreakdown` component in read-only mode.
- **Customer audience JWT model.** We assume FastAPI mints tokens with
  an `aud` claim of either `"operator"` or `"customer"`.
  `middleware.ts` does an *advisory* base64 decode of the JWT payload
  to redirect customers to `/portal` and keep operators out of the
  customer routes; the authoritative check is `GET /api/auth/me` which
  proxies to FastAPI for full signature + audience validation. If your
  backend doesn't ship per-audience tokens yet, treat all logins as
  operator and reach the portal directly via `/portal`.

### M4 backend contracts

The "Mock mode fallback" column applies **only** with
`NEXT_PUBLIC_USE_MOCKS=1`; otherwise a failure renders an error state.

| Endpoint | Used by | Mock mode fallback |
|---|---|---|
| `GET /api/v1/alerts?since=…` | Alerts feed | Demo dataset in `lib/api/alerts.ts` |
| `POST /api/v1/alerts/{id}/ack` | Ack | optimistic-only |
| `POST /api/v1/alerts/{id}/snooze` | Snooze | optimistic-only |
| `GET /api/v1/sites` | Sites map | Synthesized from `/resources` + grid |
| `GET /api/v1/customer/me` | Portal layout | Demo customer |
| `GET /api/v1/customer/me/bill` | Portal bill | Demo bill + baseline |
| `GET /api/v1/customer/me/devices` | Portal devices | Demo devices |
| `GET /api/v1/customer/programs` | Enrollment | Three demo programs |
| `POST /api/v1/customer/enrollments` | Enrollment | Echoes request |
| `GET /api/v1/auth/me` | Audience verify | 401 |

## Coming later

- M5: per-site detail page (`/sites/[id]`), advanced alert-rules editor
  (threshold/condition builder for SOH, latency, dispatch failures),
  ML-driven savings forecasting on the customer portal, and a
  customer-side device override / opt-out-of-event flow.

## Verification

```bash
pnpm install
pnpm typecheck
pnpm lint
pnpm build
pnpm test:e2e   # requires Playwright browsers: pnpm exec playwright install chromium
```

The Playwright specs stub every backend call with `page.route()` /
`page.routeWebSocket()`, so no FastAPI server is needed. They start
`pnpm dev` (or `pnpm start` when `CI` is set — run `pnpm build` first).
Useful knobs: `E2E_PORT`, `E2E_BASE_URL` (reuse a running app) and
`PLAYWRIGHT_CHROMIUM_EXECUTABLE` (use a preinstalled Chromium whose revision
differs from the one the installed Playwright expects).

If `pnpm install` fails offline, the source still builds cleanly for anyone
with network access — versions in `package.json` are pinned to current
stable releases.

## CI

The GitHub Actions workflow lives at the repo root in
`.github/workflows/web-ci.yml` (the project already has top-level workflows,
so it's colocated rather than duplicated under `web/.github/workflows/`).
It runs on PRs that touch `web/**`: a `build` job (lint, typecheck, build)
and an `e2e` job (Playwright/Chromium against a production build; the HTML
report is uploaded on failure).
