# Contributing

Thanks for helping. Bug reports, fixes, docs, field-test reports against
real chargers / VTNs / IEEE 2030.5 servers and new features are all
welcome.

## Workflow

We use GitHub Flow:

1. Fork the repository and branch from `main`.
2. Make your change with tests. If you change an API, a setting or a
   behaviour, update the docs (`README.md`, `docs/`, `CHANGELOG.md` under
   `[Unreleased]`) in the same pull request.
3. Make sure the checks below pass locally.
4. Open a pull request describing the change and how you tested it.

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/)
(`feat(tariffs): ...`, `fix(ws): ...`, `docs: ...`).

## Backend setup

Python 3.10+.

```bash
git clone https://github.com/vinerya/virtual-power-plant.git
cd virtual-power-plant
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[api,db,protocols,solver,degradation,monitoring,cli,dev]"
pre-commit install
```

Run the API against a local SQLite database:

```bash
export VPP_SECRET_KEY=dev-only-secret
vpp migrate                                   # from the repository root
uvicorn vpp.api.app:create_app --factory --reload
```

Create a first admin as described in
[docs/deployment.md](docs/deployment.md#create-the-first-admin).

## Web console setup

Node 20+ (CI uses 20) and pnpm 9 (`corepack enable`).

```bash
cd web
cp .env.example .env.local        # API_BASE_URL=http://localhost:8000
pnpm install
pnpm dev                          # http://localhost:3000
```

`NEXT_PUBLIC_USE_MOCKS=1` in `.env.local` runs the console against bundled
demo data without a backend. See [web/README.md](web/README.md).

## Checks

Backend:

```bash
pytest                                        # whole suite
pytest tests/test_api_tariffs.py -q           # one module
ruff check src tests
ruff format --check src tests
mypy src/vpp --ignore-missing-imports
```

Pre-commit runs ruff (with `--fix`), ruff-format, whitespace/YAML/JSON
checks, `detect-private-key` and mypy on every commit;
`pre-commit run --all-files` runs them on demand.

Web console:

```bash
cd web
pnpm lint
pnpm typecheck
pnpm build
```

### Playwright end-to-end tests

The specs in `web/tests/*.spec.ts` stub every backend call with
`page.route()` / `page.routeWebSocket()`, so no FastAPI server is needed.

```bash
cd web
pnpm exec playwright install chromium         # once
pnpm test:e2e                                 # starts `pnpm dev` on :3000
CI=1 pnpm build && CI=1 pnpm test:e2e         # as CI does: against `pnpm start`
pnpm exec playwright test tests/trading.spec.ts --headed
```

Knobs: `E2E_PORT` (port of the auto-started server), `E2E_BASE_URL` (test an
already running app), `E2E_NO_SERVER=1`, `PLAYWRIGHT_CHROMIUM_EXECUTABLE`
(use a preinstalled Chromium).

## Code style

- Python: ruff for linting, import sorting and formatting (line length
  99, target Python 3.10). Type hints on public functions; docstrings on
  modules and non-trivial functions.
- TypeScript: ESLint (`next/core-web-vitals`, `next/typescript`), strict
  TypeScript, validate API responses with zod in `web/lib/api/*`.
- Prefer honest behaviour over optimistic UI: report what actually
  happened (e.g. a charger's answer), label simulated data as simulated,
  and surface errors instead of falling back to fake data.

## Database migrations

The ORM models (`src/vpp/db/models.py`) and the alembic migrations
(`src/vpp/migrations/versions/`) must stay in sync; `tests/test_alembic_drift.py`
fails when they differ or when there is more than one head.

When you change a model:

```bash
# 1. Start from a database at the current head
export VPP_DATABASE_URL=sqlite+aiosqlite:///./migrate-dev.db
rm -f migrate-dev.db && vpp migrate

# 2. Generate a revision (run from the repository root; alembic.ini points
#    at src/vpp/migrations)
alembic revision --autogenerate -m "add widgets" --rev-id 0008_add_widgets
#    alembic names the file 0008_add_widgets_add_widgets.py; rename it to
#    0008_add_widgets.py so file name and revision id match the others.

# 3. Review and edit the generated file: autogenerate misses renames,
#    server defaults and data migrations. down_revision must be the
#    previous head.

# 4. Verify
vpp migrate
pytest tests/test_alembic_drift.py tests/test_alembic_migrations.py
```

Migrations must work on both SQLite and PostgreSQL (use `batch_alter_table`
for SQLite column changes). Never edit a migration that has been released;
add a new one.

## Architecture principles

- **Safe by default.** Anything that dials out or commands equipment is
  opt-in (`VPP_*_ENABLED`), and DR auto-response is off unless an operator
  turns it on.
- **Honest status.** Adapters without a real endpoint report `simulated`;
  APIs return what really happened.
- **Rules first, research separate.** Operational code paths use
  deterministic optimization and rules; experimental ML lives in
  `src/vpp/research/` and is not called by the API.
- **Deny by default.** New operator endpoints depend on `get_current_user`
  (which rejects customers) or `require_role(...)`; customer-facing routes
  must scope every query to the caller.
- **Plugins.** New optimization methods implement `OptimizationPlugin`;
  new protocols implement `ProtocolAdapter` and register in the protocol
  registry.

## Project layout

| Directory | Contents |
|---|---|
| `src/vpp/api/` | FastAPI app, routes, WebSocket, middleware |
| `src/vpp/optimization/` | allocation LP, MPC, backtest, plugins, fallbacks |
| `src/vpp/protocols/` | OCPP, OpenADR, IEEE 2030.5, MQTT, Modbus |
| `src/vpp/dr/`, `src/vpp/v2g/` | DR orchestrator, V2G fleet and OCPP bridge |
| `src/vpp/tariffs/` | URDB engine, NEM, billing |
| `src/vpp/trading/` | simulated venue, portfolio, risk, strategies |
| `src/vpp/research/` | non-operational ML experiments |
| `web/` | Next.js console and customer portal |
| `src/vpp/migrations/` | alembic environment and migrations (`alembic.ini` at the root points here) |
| `docs/` | user and operator documentation |
| `tests/` | pytest suite |

More in [docs/architecture.md](docs/architecture.md).

## Reporting bugs

Open an issue at <https://github.com/vinerya/virtual-power-plant/issues>
with what you did, what you expected, what happened, versions (`GET
/version`), and logs (include the `X-Request-ID` of a failing request).
Report security issues privately (see [docs/security.md](docs/security.md)).

## License

By contributing you agree that your contributions are licensed under the
project's [MIT License](LICENSE).
