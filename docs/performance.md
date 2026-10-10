# Performance and capacity

What one API worker sustains, measured with the `vpp loadtest` harness on a
single small machine, what limited it, what was fixed, and how to size a
deployment. Read the [scope](#scope-and-honesty) first: these are
single-host numbers from short runs, not a benchmark of your hardware.

## Scope and honesty

- **One machine** for everything: the API, the database, the MQTT broker,
  the SunSpec simulators and the load generator compete for the same
  4 vCPUs. The harness's own CPU is reported with every run; its HTTP
  client saturates one core at about 130 requests/s, so HTTP capacity above
  that was not measured.
- **Short runs**: 15-minute soaks on each database (before and after the
  fixes), one 60-minute soak (PostgreSQL) and 60-second ramps. Nothing here shows behaviour over days (index bloat,
  autovacuum, a year of telemetry), across machines, behind TLS or a
  reverse proxy, or with real devices on real networks.
- **One API worker** (`VPP_API_WORKERS=1`) unless stated; the multi-worker
  topology was not load-tested.
- Default settings everywhere: PostgreSQL 16 untuned, SQLite in its default
  rollback-journal mode, the rate limiter off (all load comes from one
  address), access logging on, alerts and the default alert rules on.

Machine: Intel Xeon @ 2.80 GHz (4 vCPU, AVX-512), 15.7 GiB RAM, virtio
disk, Linux 6.18. Python 3.11.15, uvicorn 0.54 (uvloop, httptools), FastAPI
0.141, SQLAlchemy 2.1, aiosqlite 0.22 / SQLite 3.45, asyncpg 0.31 /
PostgreSQL 16.13, paho-mqtt 2.1 / Mosquitto 2.0.18, pymodbus 3.15.

## The harness

`vpp loadtest` starts a real `vpp serve` process (and `vpp simulate
sunspec` processes) as subprocesses, registers a fleet through the API,
restarts the server with ingestion enabled, and then runs these scenarios
concurrently:

| | Scenario | What is measured |
|---|---|---|
| (a) | telemetry from N batteries x M points/s, over **MQTT** (`--ingest mqtt`, published to a broker the server subscribes to) or the **HTTP** endpoint `POST /api/v1/resources/{id}/telemetry` (`--ingest http`) | MQTT: end to end, publish -> persisted -> WebSocket broadcast, matched per message (unmatched messages after a drain period are errors); HTTP: request latency |
| (b) | authenticated reads of the dashboard endpoints (`/resources`, `/resources/{id}`, `/resources/{id}/metrics?window=1h` and `24h`, `/sites`, `/alerts`, `/dispatches`, `/auth/me`, `/trading/portfolio`, `/protocols`), alternating JWT and API-key auth | latency per endpoint |
| (c) | S WebSocket subscribers on `resource_updates`, `optimization_events`, `market_data`, `alerts` | event -> client delivery latency, disconnects |
| (d) | `POST /api/v1/optimization/dispatch` over K resources every D seconds | latency, solver time, method |
| (e) | P SunSpec simulators polled by the server's Modbus ingestion | interval between updates per device; a device silent for 5 poll intervals is an error |

Load is open loop (a fixed rate, independent of response times) with a cap
of `--max-inflight` outstanding operations per scenario; ticks over the cap
are counted as **skipped** (the server is not keeping up). Every
`--sample-interval` seconds it records the server's process-tree RSS and
CPU, database size (file size / `pg_database_size`) and table row counts,
and each scenario's throughput, p50/p95/p99 and errors. The report
(`report.json`, `report.md`, plus the server log, in `--output-dir`) adds:
the RSS and database growth slopes after the first 20 % of the run, a
**latency-creep ratio** per scenario and endpoint (median p95 of the last
third of the run over the first third; around 1.0 means no creep), and the
server log's error and warning lines. The command exits 1 when there were
errors (client-side, or `error` lines in the server log).

`vpp loadtest --smoke` is a 30-second small run (HTTP ingest, one Modbus
device, about 45 s in all) that the CI job `perf` runs on every push; it
fails only on errors, never on latency.

### Reproduce

```bash
pip install -e ".[api,db,cli,protocols,monitoring,migrations]"
mosquitto -p 1883 -d                       # or: docker run -d -p 1883:1883 eclipse-mosquitto:2 mosquitto -c /mosquitto-no-auth.conf

# 15-minute soak on a fresh SQLite file
vpp loadtest --duration 900 --sample-interval 15 --resources 100 --rate 1 \
  --api-rps 20 --ws-clients 20 --dispatch-interval 15 --dispatch-resources 50 \
  --modbus-devices 4 --output-dir soak-sqlite

# The same on a scratch PostgreSQL database (the harness migrates it)
createdb vpp_loadtest
vpp loadtest --database-url postgresql+asyncpg://vpp:vpp@localhost:5432/vpp_loadtest \
  --duration 900 --sample-interval 15 --resources 100 --rate 1 --api-rps 20 \
  --ws-clients 20 --dispatch-interval 15 --dispatch-resources 50 --modbus-devices 4 \
  --output-dir soak-pg

# Ramps (60 s each); vary one dimension, keep the rest idle
vpp loadtest --duration 60 --resources 100 --rate 8 --api-rps 0 --ws-clients 1 \
  --dispatch-interval 0 --modbus-devices 0 --output-dir mqtt-800       # MQTT ingest
vpp loadtest --duration 60 --ingest http --resources 100 --rate 1 --api-rps 0 \
  --ws-clients 1 --dispatch-interval 0 --modbus-devices 0 --max-inflight 128 --output-dir http-100
vpp loadtest --duration 60 --ingest http --resources 20 --rate 0.5 --api-rps 100 \
  --ws-clients 1 --dispatch-interval 0 --modbus-devices 0 --max-inflight 128 --output-dir api-100
vpp loadtest --duration 60 --resources 100 --rate 1 --api-rps 0 --ws-clients 100 \
  --dispatch-interval 0 --modbus-devices 0 --output-dir ws-100           # 10,000 WS msgs/s
vpp loadtest --duration 60 --resources 200 --rate 0.1 --api-rps 0 --ws-clients 1 \
  --dispatch-interval 5 --dispatch-resources 200 --modbus-devices 0 --output-dir dispatch-200
vpp loadtest --duration 60 --ingest none --resources 10 --api-rps 0 --ws-clients 1 \
  --dispatch-interval 0 --modbus-devices 20 --output-dir modbus-20
```

`vpp loadtest --help` lists every option; `--server-env KEY=VALUE` passes
settings to the server (e.g. `VPP_TELEMETRY_RETENTION_DAYS=7`), `--workers`
sets its worker count. Use a scratch database: the harness adds users,
resources and telemetry. It reads `/proc`, so it runs on Linux.

## Soak tests

Mixed load for 15 minutes, one worker: 100 batteries publishing over MQTT
once a second (100 messages/s), 20 dashboard reads/s, 20 WebSocket
subscribers (about 2,090 messages/s delivered), a dispatch over 50 resources
every 15 s, 4 SunSpec simulators polled every second.

Final code (every fix below), 15 minutes each:

| Latency, ms (p50 / p95 / p99) | SQLite | PostgreSQL 16 |
|---|---|---|
| (a) MQTT publish -> persisted -> broadcast | 12 / 33 / 84 | 7 / 23 / 98 |
| (b) API reads, all endpoints | 17 / 40 / 65 | 12 / 33 / 69 |
| `GET /resources?limit=50` | 25 / 40 / 72 | 16 / 23 / 46 |
| `GET /resources/{id}/metrics?window=24h` | 21 / 33 / 49 | 14 / 23 / 99 |
| `GET /dispatches?limit=20` | 37 / 57 / 109 | 32 / 44 / 60 |
| (c) WebSocket event -> client | 1.8 / 3.1 / 4.6 | 1.7 / 3.1 / 4.8 |
| (d) dispatch, 50 resources (request) | 365 / 525 / 700 | 208 / 505 / 629 |
| (d) of which solver time (p50 / p95) | 74 / 93 | 76 / 226 |
| (e) Modbus update interval (target 1,000) | 1,016 / 1,038 / 1,073 | 1,011 / 1,025 / 1,096 |
| Errors (client + server log), warnings in the log | 0, 0 | 0, 0 |
| Server CPU (one core = 100 %) | 78 % (75-83 %) | 61 % (57-67 %) |
| Server RSS | 177 MB at 15 s, 192 MB at 2 min, 206 MB at 8 min, then 205-210 MB | 185 MB at 15 s, 211 MB at 8 min, then 215-218 MB |
| Database growth | 34 MB in 15 min (137 MB/h) | 36 MB in 15 min (142 MB/h) |
| Latency creep (p95, last third / first third) | 0.92-1.19 | 1.01-1.26 (ingestion p95 20 -> 28 ms) |

The same soak before the alert-evaluation and trailing-slash fixes (but
after the others) ran the SQLite worker at 100 % CPU, with ingestion at
22 / 57 / 135 ms and reads at 23 / 55 / 101 ms; on PostgreSQL at 78 % CPU,
9 / 30 / 155 and 13 / 35 / 83 ms. No errors in either.

Each telemetry sample costs about **370 bytes** on disk including indexes,
on both databases (34.2 MB for 93,500 samples on SQLite, 35.6 MB on
PostgreSQL). Dispatch spends most of its time outside the solver: its
database reads and the threadpool hand-off wait for the busy event loop.

A 60-minute PostgreSQL soak with the same load is in
[60-minute soak](#60-minute-soak).

No leak was found: RSS grows by 30-50 MB over the first 8-15 minutes
(caches filling: SQLAlchemy's compiled-statement cache, asyncpg's prepared
statements, the event-bus history of 1,000 events, the alert evaluator's
per-resource state, the trailing-slash route cache) and then stays flat
while telemetry keeps flowing; in the 60-minute run it moved by less than
5 MB over the last 45 minutes. The slopes the reports print ("after
warm-up") still include the end of that warm-up in 15-minute runs, which
is why they read 75-80 MB/h there and 6 MB/h over an hour. Latency did not
creep beyond the run-to-run noise (ratios 0.9-1.3 on p95s of tens of
milliseconds).

## Capacity ramps

60 seconds each, one worker, one dimension at a time (the others idle).
"Sustained" means no errors, no skipped ticks and no growing backlog. The
MQTT rows were measured with the final code; the HTTP and API rows before
the trailing-slash cache (about 5 % less CPU per request since), the other
rows before the alert-evaluation changes (which do not affect them).

| Load | SQLite | PostgreSQL 16 |
|---|---|---|
| MQTT 400 messages/s (alert rules evaluated on every message) | sustained: p50 13 / p95 22 ms, CPU 87 % | sustained: p50 9 / p95 14 ms, CPU 85 % |
| MQTT 800 messages/s | sustained: p50 31 / p95 53 ms, CPU 97 % | sustained: p50 16 / p95 27 ms, CPU 97 % |
| MQTT 1,600 messages/s offered | **saturated**: 1,440/s processed, latency ~1.2-1.5 s, 9,800 messages dropped (queue full) | **saturated**: 1,290/s processed, 18,400 dropped |
| HTTP telemetry, 100 requests/s | **saturated** (server CPU 100 %): p50 830 ms, 1 `database is locked` 500 | sustained: p50 9 / p95 19 ms |
| HTTP telemetry, 200 requests/s offered | ~98/s served | ~130-170/s served, seconds of latency |
| API reads, 50 requests/s | sustained: p50 8 / p95 20 ms | sustained: p50 9 / p95 17 ms |
| API reads, 100 requests/s | **saturated**: p50 1.2 s | sustained: p50 8 / p95 16-21 ms at ~60 % CPU |
| API reads, 150 requests/s offered | - | not measurable: the harness's client saturates its core first |
| WebSocket: 100 clients x 100 events/s (10,000 messages/s) | sustained: delivery p50 7 / p95 13 ms | sustained: p50 6 / p95 9 ms (server CPU ~80 %) |
| Dispatch over 200 resources every 5 s | p50 554 / p95 784 ms | p50 501 / p95 746 ms |
| Modbus: 20 simulators polled every 1 s | sustained: interval p50 1,009 / p95 1,016 ms | sustained, server CPU 10-15 % |

Every telemetry message is also evaluated against the alert rules. Before
the [fix](#alert-evaluation) the evaluator managed only 190-270 events/s
and dropped the rest (now counted in `vpp_alert_evaluation_dropped_total`),
so above that rate threshold rules saw only a sample of the telemetry; it
now keeps up with everything ingestion does.

On SQLite every write transaction takes the database lock: at saturation
writers queue behind each other and some exceed SQLite's 5-second busy
timeout (`database is locked`, HTTP 500). PostgreSQL has no such limit; its
ceiling for HTTP telemetry is the worker's CPU.

## Bottlenecks found (and fixed)

All found by this harness; each fix has a regression test.

| Problem | Effect | Fix | Before -> after |
|---|---|---|---|
| **Pyomo redirected the process stdout/stderr during every solve**, from the solver's worker thread, re-pointing the log handlers | a log line from the event loop during a dispatch could deadlock the solve and leave the root log handler on a closed stream; at 400 telemetry messages/s with dispatch every 15 s **the API process died** | `create_app()` switches Pyomo to stream-only capture (`disable_solver_fd_capture()`) and turns HiGHS's console output off | API died -> no crash or log error in any later run (including saturation at 1,600 messages/s) |
| **MQTT callback on paho's thread did not wake the event loop** (`asyncio.Queue.put_nowait` from another thread) | messages waited for an unrelated timer | `loop.call_soon_threadsafe` | 1 message/s: p50 79 / p95 285 ms -> 8 / 11 ms |
| **One transaction per MQTT message** (lookup + insert + commit) | ingestion topped out at ~90 messages/s on SQLite, then the 1,000-message queue overflowed and telemetry was dropped | the ingestor writes whatever is queued (up to 500) with one lookup and one commit | 100 messages/s: p50 2.6 s and growing -> 22 ms; ceiling ~90 -> ~1,300-1,450 messages/s (with the alert fix) |
| **Newest-SOC lookup read every sample** of the listed batteries (`GROUP BY max(timestamp)`) | `GET /resources`, resource detail, sites and dispatch slowed down linearly with telemetry volume | correlated `ORDER BY timestamp DESC LIMIT 1` lookups + covering partial index (migration 0014) | 50 batteries, 1M samples per table: SQLite 187 ms -> 3.1 ms, PostgreSQL 84 ms -> 6.0 ms |
| **Alert evaluation: one transaction and one auto-resolve query per event** | evaluator capped at ~190-270 events/s; above that, telemetry was not evaluated | evaluates what is queued in one transaction, and skips the auto-resolve query for (rule, resource) pairs known to have no open alert | see [alert evaluation](#alert-evaluation) |
| **Trailing-slash middleware matched every route on every request** | ~0.8 ms per request, ~10 % of a worker's CPU at 100 requests/s (py-spy) | bounded LRU of the per-path answer | ~63 % -> ~59 % CPU at 100 reads/s on PostgreSQL |
| **Log floods**: one warning per dropped MQTT message / alert event | thousands of lines a minute exactly when the worker has no CPU to spare | one line per 10 s with a count, plus Prometheus counters | - |
| `/metrics` had no process metrics | RSS growth invisible from Prometheus | `process_resident_memory_bytes` etc. registered | - |

### Alert evaluation

Telemetry events evaluated against the default rules, 60 s of MQTT
ingestion, one worker:

| Offered | Before (one transaction + query per event) | Batched only | Batched + "known clear" cache (final) |
|---|---|---|---|
| SQLite, 400/s | 11,400 of 24,000 evaluated (12,600 dropped) | 20,800 (3,200 dropped) | all; ingest p95 22 ms |
| SQLite, 800/s | 13,100 of 48,000 (34,800 dropped) | 18,900 (29,100 dropped) | all; ingest p95 53 ms |
| PostgreSQL, 400/s | 14,600 of 24,000 (9,400 dropped) | all | all; ingest p95 14 ms |
| PostgreSQL, 800/s | 16,400 of 48,000 (31,600 dropped) | all, but the worker's CPU ran out: ingestion fell behind (1.8 s, 4,400 messages dropped) | all; ingest p95 27 ms |

Batching alone moved the work from waiting on round trips to burning CPU,
which on PostgreSQL starved ingestion; skipping the per-sample lookup made
the evaluator cheap enough that both keep up.

## What still limits one worker

- **One CPU core.** A worker is one Python process with one event loop:
  HTTP handling, WebSocket fan-out, JSON encoding, telemetry ingestion and
  alert evaluation share it. In the mixed soak it was 61 % (PostgreSQL) to
  78 % (SQLite) busy. Per request the API spends roughly 6 ms of CPU on
  PostgreSQL (authentication, ORM, serialisation, access log, metrics), so
  a worker serves on the order of 150 simple reads/s (estimated: the
  harness could not offer more than ~130/s).
- **WebSocket fan-out** costs CPU per client per message: 10,000
  messages/s used ~80 % of a core. Turning off per-message compression
  (`uvicorn ... --ws-per-message-deflate false`) cut CPU by about 13 points
  of a core at 5,000 messages/s (and delivery p50 from 4.1 to 2.6 ms), at
  the cost of more bandwidth.
- **Dispatch latency** is mostly waiting for the busy worker; the HiGHS
  solve itself is 80-220 ms for 50-200 resources.
- **SQLite's single writer**: fine for one worker up to about a hundred
  writes/s from HTTP, more through batched MQTT ingestion, but it cannot
  scale out to several workers or hosts.
- **Access logs**: one JSON line (~250 bytes) per request, ~430 MB/day at
  20 requests/s; size your log pipeline accordingly or set
  `VPP_ACCESS_LOG_ENABLED=false` and rely on the metrics.

## Sizing guidance

### When to move from SQLite to PostgreSQL

SQLite carried the mixed soak (100 MQTT messages/s plus reads, sockets,
dispatch and Modbus) at 78 % of a core, and MQTT ingestion alone up to 800
messages/s. Its limits are elsewhere:

- one writer at a time: HTTP writes saturate a worker near 100/s and then
  fail with `database is locked` after the 5-second busy timeout;
- one process on one host: no second worker or replica;
- backups copy the whole file under a read lock, and there is no
  point-in-time recovery.

Use it for development, demos and small single-site installations: one
worker, telemetry up to a few hundred messages/s over MQTT, HTTP writes
well below 100/s, a handful of dashboard users, a database of at most a
few GB. Move to PostgreSQL when any of these is exceeded, when you need a
second worker or host, or point-in-time recovery. PostgreSQL was faster in
every measurement here, on the same host (soak p95: ingestion 23 vs 33 ms,
reads 33 vs 40 ms, at 61 % instead of 78 % CPU), and does not fail writes
under contention.

### Workers

- Start with **one worker per core you can dedicate**, on PostgreSQL
  (`VPP_API_WORKERS`, see [deployment.md](deployment.md#without-docker)).
  A single worker handled the mixed soak (100 messages/s, 20 reads/s, 20
  WebSocket clients, dispatch every 15 s, 4 Modbus devices) at 61 % of a
  core on PostgreSQL: plan a second worker well before a worker's
  `process_cpu_seconds_total` rate stays above ~0.7.
- Singleton work (MQTT ingestion, Modbus polling, alert evaluation,
  retention, the trading venue) runs on one lease holder regardless of the
  worker count, so **extra workers add HTTP and WebSocket capacity, not
  ingestion capacity**. With several workers each WebSocket broadcast is
  also written to `cluster_events` for the other workers to relay.
- OCPP and device control (`VPP_CONTROL_ENABLED`) require a single worker.
- Not load-tested here: several workers, several replicas.

### Ingestion

- Prefer MQTT for fleets: batched writes reach 800 messages/s with p95
  under 60 ms on one worker (alerts included). The HTTP endpoint costs a
  request per call; batch samples in one request (`samples` takes up to
  5,000) instead of one call per point.
- Alert evaluation keeps up with ingestion on one worker up to the rates
  above ([alert evaluation](#alert-evaluation)); watch
  `vpp_alert_evaluation_dropped_total` and
  `vpp_protocol_errors_total{protocol="mqtt"}` for the point where a
  worker runs out of CPU.
- Modbus polling is cheap (20 devices at 1 s used 10-15 % of a core);
  devices on slow links or with long timeouts are the practical limit.

### Retention and disk

At ~370 bytes per sample, telemetry needs `devices x samples/day x 370 B`:
100 devices every 5 s is ~650 MB/day, every second ~3.2 GB/day. Set
`VPP_TELEMETRY_RETENTION_DAYS` (default: keep forever) and size the disk
for retention x daily volume x 1.5 (indexes, bloat, backups in flight). See
[operations.md](operations.md#data-retention).

## 60-minute soak

PostgreSQL, the soak load above for 3,600 s (code before the
trailing-slash and alert-evaluation commits; the leak and creep checks do
not depend on them):

| | Result |
|---|---|
| Throughput | 100 messages/s ingested (360,000, all persisted), 20 reads/s (71,990), 2,089 WebSocket messages/s (7.5 million), 240 dispatches, 14,187 Modbus updates |
| Latency p50 / p95 / p99, ms | ingestion 9 / 34 / 183; reads 14 / 37 / 105; WebSocket 2.0 / 3.6 / 5.6; dispatch 250 / 563 / 746; Modbus interval 1,012 / 1,035 / 1,149 |
| Errors | none (the run reported 5: messages the probe socket missed while reconnecting at the 60-minute session expiry, all persisted; the harness now checks the database before counting a message as lost) |
| Server RSS | 188 MB at 30 s, 216 MB at 10 min, 224 MB at 15 min, then 221-226 MB for the remaining 45 min: **5.8 MB/h** slope after warm-up, i.e. flat |
| Server CPU | 74-83 % of one core throughout |
| Database | 9 MB -> 151 MB (142 MB/h, linear with the rows: 374,000 samples) |
| Latency creep (p95, last third / first third) | 0.99-1.13 |

One stall: at minute 40 every request, broadcast and Modbus update paused
for about 3.3 s (max latencies 3.5-4.2 s in that 30-s sample) while the
broker kept delivering (received-at timestamps continued at 100/s). No
server log line, PostgreSQL checkpoint or autovacuum coincides with it; it
did not recur in the run. Treat p99.9 latencies of a few seconds as
possible on a shared VM until this is explained; a longer run with
`py-spy` attached would be the next step.
