# Operations runbook

Day-2 tasks for a running installation: backups and restores, upgrades,
credential rotation, data retention, probes and monitoring. Deployment
itself is in [deployment.md](deployment.md), settings in
[configuration.md](configuration.md), capacity numbers in
[performance.md](performance.md).

The SQLite and PostgreSQL commands below were run end to end, as written
(with only the paths and host names changed), against PostgreSQL 16.13 and
SQLite 3.45 on Linux, including a restore of a database taken while the API
was ingesting 100 telemetry messages/s. The `docker compose` variants run the
same client programs inside the `postgres` container; they were not
executed for this document.

## What to back up

| What | Where | Notes |
|---|---|---|
| The database | `VPP_DATABASE_URL` | Everything the platform knows: users, API-key hashes, resources, sites, telemetry, alerts, optimisation runs, orders and trades, tariffs, the audit log, stored configuration documents, output caps on curtailed devices. |
| `VPP_SECRET_KEY` | your secret store | Not in the database. Restoring with a different key only logs everybody out (see [rotation](#rotating-vpp_secret_key)). |
| Database password, `VPP_OCPP_BASIC_AUTH_PASSWORD`, `VPP_ALERT_WEBHOOK_SECRET`, `VPP_METRICS_BEARER_TOKEN`, `VPP_MQTT_PASSWORD`, client certificates | your secret store | Needed to bring a restored instance back on line. |
| `VPP_CONFIG_PATH` file, compose overrides | your config repository | Only used when no configuration document is stored in the database. |

The coordination tables `cluster_leases`, `cluster_calls`, `cluster_events`
and `shared_rate_limits` hold seconds-to-minutes of state; restoring stale
rows is harmless (leases expire, events are purged after 5 minutes).

## SQLite

### Back up a live database

Never `cp` the database file while the API runs: a copy taken mid-commit
can be torn (pages from before and after a transaction), and it misses
whatever is still in a `-journal` or `-wal` file. Use SQLite's online
backup API, which copies a consistent snapshot while the API keeps running.
The container image has Python but no `sqlite3` shell, so the Python form
works everywhere:

```bash
DB=/var/lib/vpp/vpp.db                 # the file named in VPP_DATABASE_URL
BACKUP_DIR=/var/backups/vpp
mkdir -p "$BACKUP_DIR"
BACKUP="$BACKUP_DIR/vpp-$(date -u +%Y%m%dT%H%M%SZ).db"
python3 - "$DB" "$BACKUP" <<'EOF'
import sqlite3
import sys

src = sqlite3.connect(sys.argv[1])
dst = sqlite3.connect(sys.argv[2])
src.backup(dst)  # consistent snapshot, even while the API writes
print(dst.execute("PRAGMA integrity_check").fetchone()[0])
dst.close()
src.close()
EOF
```

It prints `ok` when the copy is sound. With the `sqlite3` shell installed,
the equivalent is:

```bash
sqlite3 "$DB" ".backup '$BACKUP'"
sqlite3 "$BACKUP" "PRAGMA integrity_check"
```

The backup reads the whole database under a shared lock; the API's writers
wait for it (up to SQLite's 5-second busy timeout, after which a write
fails with "database is locked"). An 11 MB database copied in 0.15 s while
the API ingested 100 messages/s, with no failed writes. For databases of
several GB, run backups off-peak, or move to PostgreSQL (see
[performance.md](performance.md#when-to-move-from-sqlite-to-postgresql)).

In Docker, run the same Python against the mounted volume, e.g.
`docker compose exec -T vpp-api python3 - /app/data/vpp.db /app/data/backup.db <<'EOF' ...`,
and copy the result out of the volume.

Keep several generations and prune old ones, e.g. from cron:

```bash
find "$BACKUP_DIR" -name 'vpp-*.db' -mtime +14 -delete
```

### Restore

1. Stop the API (every worker: a restore under a running process is
   overwritten by its next write, or corrupts it).
2. Move the current database **and** its side files aside, put the backup
   in place, and fix ownership for the service user:

   ```bash
   DB=/var/lib/vpp/vpp.db
   BACKUP=/var/backups/vpp/vpp-20261010T011900Z.db
   mkdir -p "$DB.before-restore"
   for f in "$DB" "$DB-wal" "$DB-shm" "$DB-journal"; do
     if [ -e "$f" ]; then mv "$f" "$DB.before-restore/"; fi
   done
   cp "$BACKUP" "$DB"
   chown vpp:vpp "$DB"    # the user the API runs as
   ```

   A stale `-wal` or `-journal` file left next to a restored database
   would be replayed into it.
3. If the backup is from an older release than the one you run, apply its
   migrations: `VPP_DATABASE_URL=sqlite+aiosqlite:///$DB vpp migrate`.
4. Start the API and run the [smoke check](#smoke-check) against it.

## PostgreSQL

### Back up

`pg_dump` takes a consistent snapshot of a running database without
blocking the API. Use the custom format (`-Fc`): compressed, and
`pg_restore` can restore it selectively.

```bash
export PGHOST=db.internal PGPORT=5432 PGUSER=vpp PGDATABASE=vpp   # password via ~/.pgpass or PGPASSWORD
BACKUP_DIR=/var/backups/vpp
mkdir -p "$BACKUP_DIR"
pg_dump --format=custom --file="$BACKUP_DIR/vpp-$(date -u +%Y%m%dT%H%M%SZ).dump" "$PGDATABASE"
```

With the provided `docker-compose.yml` (the database is not published on
the host):

```bash
docker compose exec -T postgres pg_dump -U vpp --format=custom vpp > "vpp-$(date -u +%Y%m%dT%H%M%SZ).dump"
```

The dump contains the schema, data and the `alembic_version` row, not
roles: the restore below re-creates objects owned by the connecting role.

### Restore

Restore into an **empty, newly created** database. Do not `pg_restore
--clean` over a database that a newer release has migrated: tables that
release added are not in the dump, survive the restore, and make the next
`vpp migrate` fail.

1. Stop the API (every worker and replica).
2. Re-create the database and restore into it:

   ```bash
   export PGHOST=db.internal PGPORT=5432 PGUSER=vpp
   DUMP=/var/backups/vpp/vpp-20261010T014128Z.dump
   dropdb --if-exists vpp
   createdb --owner=vpp vpp
   pg_restore --exit-on-error --no-owner --role=vpp --dbname=vpp "$DUMP"
   ```

   `dropdb` needs no other session connected to `vpp`; the role needs
   `CREATEDB` (or ask your DBA / use your provider's console). With
   compose:

   ```bash
   docker compose stop vpp-api
   docker compose exec -T postgres dropdb -U vpp --if-exists vpp
   docker compose exec -T postgres createdb -U vpp --owner=vpp vpp
   docker compose exec -T postgres pg_restore -U vpp --exit-on-error --no-owner --role=vpp --dbname=vpp < vpp-20261010T014128Z.dump
   docker compose start vpp-api    # runs `vpp migrate` before serving
   ```
3. Bare metal: `vpp migrate` (a no-op unless the dump predates the release
   you run), start the API, run the [smoke check](#smoke-check).

### Point-in-time recovery

`pg_dump` restores to the moment of the dump. To lose at most a few
seconds instead, run PostgreSQL with continuous WAL archiving
(`archive_mode = on` plus an `archive_command`, or a tool such as pgBackRest
or WAL-G) and periodic base backups (`pg_basebackup`), then recover to a
timestamp with `recovery_target_time`; see "Continuous Archiving and
Point-in-Time Recovery (PITR)" in the PostgreSQL manual. Managed services
(RDS, Cloud SQL, Azure) offer the same as a setting. PITR restores the whole
cluster; keep the logical `pg_dump`s as well for restoring one database
elsewhere and for the verification below.

## Verify a backup

A backup you have not restored is a hope. Regularly (e.g. weekly, and
before every upgrade) restore the newest one into a scratch database, bring
its schema to the current release and smoke-check it. Nothing here touches
the production database.

SQLite:

```bash
BACKUP=/var/backups/vpp/vpp-20261010T011900Z.db
SCRATCH=/tmp/vpp-restore-check.db
rm -f "$SCRATCH" "$SCRATCH-wal" "$SCRATCH-shm" "$SCRATCH-journal"
cp "$BACKUP" "$SCRATCH"
export VPP_DATABASE_URL="sqlite+aiosqlite:///$SCRATCH"
```

PostgreSQL (needs a role allowed to create databases):

```bash
export PGHOST=db.internal PGPORT=5432 PGUSER=vpp    # password via ~/.pgpass or PGPASSWORD
DUMP=/var/backups/vpp/vpp-20261010T014128Z.dump
SCRATCH=vpp_restore_check
dropdb --if-exists "$SCRATCH"
createdb --owner="$PGUSER" "$SCRATCH"
pg_restore --exit-on-error --no-owner --role="$PGUSER" --dbname="$SCRATCH" "$DUMP"
export VPP_DATABASE_URL="postgresql+asyncpg://$PGUSER:$PGPASSWORD@$PGHOST:$PGPORT/$SCRATCH"
```

### Smoke check

With `VPP_DATABASE_URL` pointing at the restored copy:

```bash
vpp migrate                 # upgrades the copy to this release's schema
vpp users list              # accounts are there
vpp prune --dry-run         # reads every pruned table; changes nothing
# Start an API on the copy with every outbound integration off, then probe it.
env VPP_SECRET_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(48))')" \
    VPP_MQTT_INGESTION_ENABLED=false VPP_MODBUS_INGESTION_ENABLED=false \
    VPP_CONTROL_ENABLED=false VPP_OCPP_ENABLED=false VPP_OPENADR_ENABLED=false \
    VPP_IEEE2030_5_ENABLED=false VPP_DR_AUTO_RESPONSE_ENABLED=false \
    VPP_ALERT_WEBHOOK_URL= VPP_TRADING_MARKET_DATA_ENABLED=false \
    vpp serve --host 127.0.0.1 --port 8099 --workers 1 > /tmp/vpp-restore-check.log 2>&1 &
API_PID=$!
for i in $(seq 60); do curl -fs http://127.0.0.1:8099/ready && break; sleep 1; done
echo
curl -fsS http://127.0.0.1:8099/version
echo
kill "$API_PID"
wait "$API_PID" || true     # exits with the signal's status
```

Expect `vpp migrate` to finish, your admin accounts in `vpp users list`,
a line per table from `vpp prune --dry-run` (no error), and
`{"status":"ready",...}` from `/ready`. To compare contents, count a few
tables in production and in the copy (e.g. `SELECT count(*) FROM resources`,
`... FROM users`, `... FROM trades`); a copy made at time T has every row
written before T. Then remove the copy (`rm -f "$SCRATCH"` /
`dropdb "$SCRATCH"`) and `unset VPP_DATABASE_URL`.

## Upgrading

1. Read the release's **Breaking changes** in [CHANGELOG.md](../CHANGELOG.md).
2. Take a backup and [verify](#verify-a-backup) it. This backup is your
   rollback.
3. Stop the API: every worker and replica, so no process of the old release
   writes to the new schema.
4. Deploy the new version and run `vpp migrate` once (compose does it on
   every start; it is idempotent). Read its output: each applied revision
   is logged.
5. Start the API, watch the logs, and run the [smoke check](#smoke-check)
   against production (`/ready`, a login in the console, a telemetry
   update arriving).

Schema migrations are forward-only in practice and not designed for a
mixed-version fleet: do not run old and new releases against one database
at the same time.

To see which revision a database is at: `SELECT version_num FROM
alembic_version;` (SQLite and PostgreSQL alike).

### Rolling back

Roll back by **restoring the pre-upgrade backup** and deploying the previous
release: stop the API, restore (SQLite: replace the file; PostgreSQL:
re-create the database, as above), start the old version. Data written
since the upgrade is lost either way; restore keeps everything that existed
before it.

Do not roll back with `alembic downgrade`. Most downgrades drop the tables
or columns their upgrade added, together with their data, and the
data-bearing downgrades are not exercised against real data (CI only runs
them on an empty PostgreSQL database):

| Revision | Downgrade drops | Lossy |
|---|---|---|
| `0014_telemetry_soc_index` | an index | no |
| `0013_output_caps` | `output_caps` (caps on curtailed devices: a restarted API no longer knows its own curtailment) | yes |
| `0012_audit_log` | `audit_log` (the security audit trail) | yes |
| `0011_shared_rate_limits` | `shared_rate_limits` (transient counters) | no (counters only) |
| `0010_cluster_and_v2g_bids` | `cluster_*` (transient) and `v2g_flexibility_bids` | yes (bids) |
| `0009_user_management` | `users.token_version`, `users.last_login_at`, `api_keys.key_prefix`, `api_keys.last_used_at` | yes; dropping `token_version` also makes tokens revoked before the downgrade valid again until they expire |
| `0008_drop_redundant_unique_constraints` | re-creates constraints (PostgreSQL) | no |
| `0007_v2g_protocols` | V2G vehicles, OCPP charging sessions, V2G schedules, DR event responses | yes |
| `0006_add_sites_customers` | sites, customer profiles, DR programs and enrollments, meter readings, `resource_telemetry`, configuration documents, `resources.site_id` | yes |
| `0005_add_alerts` | alert rules and alerts | yes |
| `0004_add_tariffs` | tariffs (and relaxes timestamp NOT NULLs) | yes |
| `0003_add_nominal_energy` | `resources.nominal_energy_kwh` | yes |
| `0002_add_battery_degradation` | SOH columns and `battery_soh_samples` | yes |
| `0001_baseline` | every table | yes |

## Credentials

### Rotating `VPP_SECRET_KEY`

The key signs JWTs: console and API sessions (`POST /api/v1/auth/token`)
and the short-lived WebSocket tokens. Changing it:

- **invalidates** every session and WebSocket token at once: every user,
  including customers, must log in again; scripts that obtained a JWT with
  a username and password must log in again;
- **does not affect** API keys (stored as SHA-256 hashes, not signed),
  passwords (bcrypt), the database, the metrics bearer token or webhook
  secrets.

Procedure: generate a new key
(`python3 -c "import secrets; print(secrets.token_urlsafe(48))"`), put it
in the secret store, and restart **all** workers and replicas together.
Every process must sign and verify with the same key: during a rolling
restart, requests reaching a process with the other key fail with 401.
There is no grace period with two keys. Rotate on a schedule, and at once
if the key may have leaked: anyone holding it can mint a token for any
user.

### Rotating API keys

API keys never expire on their own. To rotate one without downtime:

```bash
API=https://vpp.example.com
TOKEN=...   # a session JWT of the key's owner (or an existing key: use -H "X-API-Key: ..." instead)
# 1. Create the replacement (same role as the old key); the key is shown only here.
curl -fsS -X POST "$API/api/v1/auth/api-key" -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' -d '{"name": "scada-gateway-2026-10", "role": "operator"}'
# 2. Deploy the new key to the integration. Its last_used_at appears here once it is in use:
curl -fsS "$API/api/v1/auth/api-keys" -H "Authorization: Bearer $TOKEN"
# 3. Revoke the old key by id.
curl -fsS -X DELETE "$API/api/v1/auth/api-keys/<old-key-id>" -H "Authorization: Bearer $TOKEN"
```

Admins see any user's keys at `GET /api/v1/users/{id}/api-keys` and in the
console (**Settings → Users & API keys**). Deactivating a user revokes all
of their keys.

### Revoking sessions

| Situation | Action |
|---|---|
| A user's token or device may be compromised | `POST /api/v1/users/{id}/revoke-sessions` (admin) or the user's own `POST /api/v1/auth/logout-all`: every JWT and socket token issued so far stops working, open WebSockets close within 30 s. API keys are not affected: revoke them separately. |
| A user leaves | `PATCH /api/v1/users/{id}` with `{"is_active": false}`: sessions and API keys stop working immediately. |
| Locked out / forgotten admin password | `vpp users set-password <username>` on the host (also revokes that user's sessions and re-activates the account). |
| Every session of every user | Rotate `VPP_SECRET_KEY`. |

Password changes and role changes also revoke the user's sessions. See
[security.md](security.md#authentication) for the mechanism.

### Other secrets

- **Database password** (compose: `POSTGRES_PASSWORD`). The postgres image
  only uses it when it initialises an empty volume; afterwards change it in
  the database (`ALTER ROLE vpp PASSWORD '...'`), then update
  `POSTGRES_PASSWORD` / `VPP_DATABASE_URL` and restart the API.
- `VPP_METRICS_BEARER_TOKEN`, `VPP_ALERT_WEBHOOK_SECRET`,
  `VPP_OCPP_BASIC_AUTH_PASSWORD`, `VPP_MQTT_PASSWORD`: change on both ends
  (Prometheus scrape config, webhook receiver, charge points, broker) and
  restart the API.

## Data retention

Telemetry tables grow without bound unless a retention is set:
`VPP_TELEMETRY_RETENTION_DAYS` defaults to `0` (keep forever). Each
telemetry sample costs about 370 bytes on disk, indexes included, on SQLite
and PostgreSQL alike (measured in the [soak tests](performance.md#soak-tests)).
100 batteries reporting every 5 s add about 650 MB a day; every second,
about 3.2 GB.

| Setting | Default | Suggested |
|---|---|---|
| `VPP_TELEMETRY_RETENTION_DAYS` | 0 (forever) | 30-90 at a few-second resolution; the metrics API serves up to 366 days |
| `VPP_AUDIT_RETENTION_DAYS` | 365 | your compliance requirement |
| `VPP_EVENT_LOG_RETENTION_DAYS` | 0 | 90 |
| `VPP_ALERT_RETENTION_DAYS` | 0 | 180 (resolved alerts only) |
| `VPP_DR_RESPONSE_RETENTION_DAYS` | 0 | per your DR program's settlement period |

`vpp prune --dry-run` shows what a pass would delete; the API prunes every
`VPP_RETENTION_INTERVAL_MINUTES` on one worker (details and the tables
never pruned are in [security.md](security.md#data-retention)). Turning a
retention on for a large, old table deletes the backlog in batches of
`VPP_RETENTION_BATCH_SIZE` per transaction; run `vpp prune` once off-peak
first.

Deleting rows does not shrink the files: SQLite reuses the free pages
(`VACUUM` returns them, but rewrites the whole file under an exclusive
lock; stop the API first), PostgreSQL's autovacuum makes the space reusable
(`VACUUM FULL` or `pg_repack` returns it to the OS).

## Health and readiness probes

| Endpoint | Meaning | Use as |
|---|---|---|
| `GET /health` | the process answers HTTP | liveness probe (the image's `HEALTHCHECK` uses it) |
| `GET /ready` | the database answers `SELECT 1`; `503` otherwise | readiness probe: stop routing to an instance that lost its database |
| `GET /version` | release and API version | after an upgrade |

Neither probe checks MQTT, Modbus or protocol peers; watch those with the
metrics below. Kubernetes example:

```yaml
livenessProbe:
  httpGet: {path: /health, port: 8000}
  periodSeconds: 10
  failureThreshold: 6
readinessProbe:
  httpGet: {path: /ready, port: 8000}
  periodSeconds: 5
  failureThreshold: 2
```

Don't point liveness at `/ready`: a database outage would then restart
every API process in a loop instead of waiting for the database.

## What to monitor

`/metrics` (Prometheus format, see [deployment.md](deployment.md#monitoring)
for the bundled Prometheus and Grafana) exposes:

| Metric | Watch for | Example alert |
|---|---|---|
| `vpp_api_requests_total{status}` | 5xx rate | `sum(rate(vpp_api_requests_total{status=~"5.."}[5m])) / sum(rate(vpp_api_requests_total[5m])) > 0.01` |
| `vpp_api_request_duration_seconds` | p95 per route | `histogram_quantile(0.95, sum by (le, endpoint) (rate(vpp_api_request_duration_seconds_bucket[5m]))) > 1` |
| `vpp_api_requests_in_progress` | requests piling up (saturation) | `> 50` for 5 m |
| `vpp_resource_last_update_timestamp_seconds` | stale telemetry per resource | `time() - vpp_resource_last_update_timestamp_seconds > 300` |
| `vpp_telemetry_events_total{source}` | ingest rate per path (MQTT, Modbus, HTTP) | rate drops to 0 |
| `vpp_protocol_errors_total{protocol}` | protocol errors; `protocol="mqtt"` also counts messages dropped because ingestion fell behind | `increase(...[10m]) > 0` |
| `vpp_alert_evaluation_dropped_total` | telemetry not evaluated against alert rules (evaluator overloaded) | `increase(...[10m]) > 0` |
| `vpp_optimization_runs_total{status}`, `vpp_optimization_duration_seconds` | failing or slow dispatch | error rate, p95 above your dispatch interval |
| `vpp_alerts_fired_total` | operational alerts | per your rules |
| `process_resident_memory_bytes`, `process_cpu_seconds_total` | memory growth, CPU saturation of a worker | RSS growing steadily over days; CPU rate near 1.0 (one core) |

With several workers each process exposes its own `process_*` and request
metrics; scrape each worker or compare them per instance. Watch the
database too (disk free, connections, replication lag) with your usual
exporter: the most common failure of a long-running installation is a full
disk from telemetry kept forever.

Log lines worth an alert (`VPP_LOG_JSON=true`): `MQTT message queue full`
and `Alert evaluation queue full` (ingestion overloaded; see
[performance.md](performance.md)), `Modbus connect failed`, `lease`
warnings (leadership flapping between workers), and any `level: error`.
