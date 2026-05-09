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

## Coming later

- M2: asset detail page, dispatch history view, charts.
- Customer portal, dispatch explainer, trading, settings.

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
