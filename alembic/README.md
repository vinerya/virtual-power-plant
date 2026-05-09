# Alembic migrations for the VPP platform

Alembic manages the SQL schema for the persistence layer.  Migration scripts
live in `alembic/versions/`.  Configuration is in `alembic.ini` at the repo
root, and the environment in `alembic/env.py`.

## Configuration source

`alembic/env.py` reads the database URL from
`vpp.settings.Settings.database_url` (i.e. the `VPP_DATABASE_URL` environment
variable, falling back to the `.env` default of
`sqlite+aiosqlite:///./vpp.db`).

The application uses an *async* SQLAlchemy engine, but alembic operates
synchronously — `env.py` strips async driver suffixes:

| Runtime URL                       | Alembic URL                  |
|-----------------------------------|------------------------------|
| `sqlite+aiosqlite:///./vpp.db`    | `sqlite:///./vpp.db`         |
| `postgresql+asyncpg://host/db`    | `postgresql+psycopg2://host/db` |

You can override the URL on the CLI: `alembic -x url=sqlite:///tmp.db upgrade head`.

## Day-to-day commands

```bash
alembic upgrade head      # apply all pending migrations
alembic current           # show current head
alembic history --verbose # full revision graph
alembic downgrade -1      # roll back one revision
alembic revision --autogenerate -m "describe change"
```

## Migration sequence

| Revision                          | Purpose                                         |
|-----------------------------------|-------------------------------------------------|
| `0001_baseline`                   | Pre-M3 schema (resources, battery_states, optimization_runs, orders, trades, users, api_keys, event_log) |
| `0002_add_battery_degradation`    | M3: SOH/throughput columns + `battery_soh_samples` time-series table |
| `0003_add_nominal_energy`         | M4: `resources.nominal_energy_kwh` for accurate throughput accounting |

## Fresh init vs. existing data

* **Fresh database** — run `alembic upgrade head` and you get the full
  schema in one go.  This is the path the test suite exercises.
* **Existing M2-era database** (no degradation columns yet) — stamp at the
  pre-M3 baseline and upgrade:
  ```bash
  alembic stamp 0001_baseline
  alembic upgrade head
  ```
  `0002` and `0003` use `op.batch_alter_table` so they work on SQLite as
  well as PostgreSQL.
* **Existing M3 database** (already has the degradation columns from
  `Base.metadata.create_all`) — stamp at `0002` to skip the column-add
  migration and pick up M4:
  ```bash
  alembic stamp 0002_add_battery_degradation
  alembic upgrade head
  ```

## Switching `init_db` between `create_all` and alembic

Set `VPP_USE_ALEMBIC=1` (or `degradation_updater_enabled` setting) to make
`init_db` invoke `alembic upgrade head` instead of `Base.metadata.create_all`.
The default is `create_all` for backwards compatibility with existing dev
workflows; production deployments should set `VPP_USE_ALEMBIC=1`.
