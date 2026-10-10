# Observability

The platform has three complementary signals:

| Signal | What it answers | Where |
|---|---|---|
| Prometheus metrics | How much, how fast, how often (rates, latencies, fleet gauges) | `GET /metrics`, `src/vpp/metrics.py`, `monitoring/` |
| Structured logs | What happened, with request id and (when tracing is on) trace id | stderr (JSON with `VPP_LOG_JSON=true` or `VPP_ENV=production`), `src/vpp/logging.py` |
| OpenTelemetry traces | Where the time went in *one* operation, end to end (request -> optimisation -> device writes, across workers) | OTLP exporter -> Jaeger / any OTLP backend, `src/vpp/tracing.py` |

## Metrics (Prometheus + Grafana)

`GET /metrics` serves the Prometheus text format when `VPP_METRICS_ENABLED=true`
(default) and `prometheus-client` is installed (`monitoring` extra). Set
`VPP_METRICS_BEARER_TOKEN` to require `Authorization: Bearer <token>`.

Main series (full list in `src/vpp/metrics.py`):

- HTTP: `vpp_api_requests_total`, `vpp_api_request_duration_seconds`, `vpp_api_requests_in_progress`
- Fleet: `vpp_resource_count`, `vpp_resource_power_kw`, `vpp_battery_soc`, `vpp_resource_last_update_timestamp_seconds`, `vpp_telemetry_events_total`
- Optimisation: `vpp_optimization_runs_total`, `vpp_optimization_duration_seconds`
- Trading: `vpp_trading_orders_total`, `vpp_trading_trades_total`, `vpp_trading_volume_mwh_total`, `vpp_trading_pnl_total`
- Protocols and alerts: `vpp_protocol_messages_total`, `vpp_protocol_errors_total`, `vpp_alerts_fired_total`, plus V2G fleet gauges

Prometheus (:9090) and Grafana (:3001, dashboards "VPP Overview", "VPP Trading"
and "VPP Fleet" provisioned from `monitoring/grafana/dashboards`) run as an
overlay:

```bash
docker compose -f docker-compose.yml -f monitoring/docker-compose.monitoring.yml up -d
# Grafana: http://localhost:3001  (admin / $GRAFANA_ADMIN_PASSWORD, default "vpp-grafana")
# Prometheus: http://localhost:9090/targets  (vpp-api should be UP)
```

If you set `VPP_METRICS_BEARER_TOKEN`, put the same token in
`monitoring/prometheus/prometheus.yml` (`authorization:` block).

## Logs

Logs are structured (`structlog`). Every line carries `request_id` inside an
HTTP request (also returned as the `X-Request-ID` header), and with tracing
enabled every line logged inside a span also carries `trace_id` and
`span_id`, for plain `logging` calls as well as structlog ones. Paste a
`trace_id` from a log line into Jaeger's search box to open the whole trace.

```json
{"method": "POST", "route": "/api/v1/optimization/dispatch", "status": 200,
 "event": "http_request", "request_id": "76599aaa...", "trace_id": "d49e767f9273907f4c239a9f1706c14f",
 "span_id": "88ff584590c9d5c4", "level": "info", "logger": "vpp.access"}
```

## Tracing (OpenTelemetry)

Tracing is optional and off by default. It needs the `tracing` extra:

```bash
pip install "virtual-power-plant[tracing]"
export VPP_TRACING_ENABLED=true
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318   # OTLP/HTTP collector
vpp serve
```

With `VPP_TRACING_ENABLED=true` but the extra missing, the API logs a warning
("OpenTelemetry is not installed ... continuing without tracing") and runs
normally. Without the flag, OpenTelemetry is never imported and every span
call site is a shared no-op (a global lookup; well under a microsecond).

### Configuration

Only `VPP_TRACING_ENABLED` is VPP-specific; everything else uses the standard
OpenTelemetry environment variables. They must be real environment variables
(the app's own `.env` loader only reads `VPP_*` settings; `docker compose`
`environment:`/`env_file:` entries are fine).

| Variable | Default | Meaning |
|---|---|---|
| `VPP_TRACING_ENABLED` | `false` | Turn tracing on. |
| `OTEL_SERVICE_NAME` | `vpp` | `service.name` of the spans (`service.version` is the package version). |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:4318` | Collector base URL (`http://host:4317` with gRPC). `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`, `..._HEADERS`, `..._TIMEOUT`, `..._CERTIFICATE` work as specified by OpenTelemetry. |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | `http/protobuf` | `http/protobuf` or `grpc` (`OTEL_EXPORTER_OTLP_TRACES_PROTOCOL` overrides it). `grpc` needs `pip install opentelemetry-exporter-otlp-proto-grpc`; the `tracing` extra ships only the HTTP exporter. |
| `OTEL_TRACES_EXPORTER` | `otlp` | `otlp`, `console` (print spans to stdout, handy locally) or `none`. |
| `OTEL_TRACES_SAMPLER` | `parentbased_always_on` | e.g. `parentbased_traceidratio` ... |
| `OTEL_TRACES_SAMPLER_ARG` | unset | ... with `0.1` to keep 10% of new traces (children follow their parent's decision). |
| `OTEL_RESOURCE_ATTRIBUTES` | unset | Extra resource attributes, e.g. `deployment.environment=prod,site=north`. |

### What is traced

Automatic instrumentation:

- **HTTP requests and WebSockets** (FastAPI): one server span per request,
  the root of everything the request does. `/metrics` and `/health` are excluded.
- **SQL** (SQLAlchemy engine): one span per statement, nested under whatever
  issued it.
- **Outbound HTTP** (httpx): OpenADR VTN and IEEE 2030.5 server calls,
  tariff feeds, alert webhooks; the W3C `traceparent` header is sent along.

Manual spans (attributes are prefixed `vpp.`):

| Span | Where | Key attributes |
|---|---|---|
| `vpp.dispatch.execute` | Single-interval dispatch (`POST /api/v1/optimization/dispatch`, DR orchestrator) | `problem_type`, `source`, `target_kw`, `resources`, `apply`, `run_id`, `status`, `method`, `delivered_kw`, `shortfall_kw`, `devices` |
| `vpp.optimization.solve` | Every solve through the plugin engine (dispatch, stochastic, realtime, distributed...) | `strategy` (problem type), `horizon`, `time_step`, `solver` (plugin or `fallback`), `status`, `objective`, `fallback_used`, `solve_time_s` |
| `vpp.optimization.plan_schedule` | Horizon schedule / MPC (`/schedule`, backtests) | `resources`, `horizon`, `interval_minutes`, `solver`, `objective`, `fallback_used` |
| `vpp.actuator.setpoint` | One per device setpoint write (dispatch, deferred write, keepalive) | `resource_id`, `profile`, `requested_kw`, `setpoint_kw`, `status`, `verified`, `failed`, `clamped`, `run_id` |
| `vpp.actuator.release` | Fallback / release of a device (expiry, DR release, shutdown) | `resource_id`, `profile`, `action`, `status`, `verified`, `failed` |
| `vpp.dr.dispatch` / `vpp.dr.release` | DR orchestrator acting on an OpenADR event / IEEE 2030.5 control | `protocol`, `source_id`, `signal_type`, `target_kw`, `run_id`, `status` |
| `vpp.modbus.poll` / `vpp.modbus.write` | Each Modbus poll cycle / register write | `adapter`, `device_profile`, `unit_id`, `values`, `errors`, `address` |
| `vpp.ocpp.call` / `vpp.ocpp.handle` | CALLs to a charge point / CALLs from it | `action`, `charge_point_id` |
| `vpp.openadr.poll` / `vpp.ieee2030_5.poll` | Each VTN / 2030.5 server poll | `ven_id`, `messages`, `programs`, `active_controls` |
| `vpp.market.order.submit` / `vpp.market.order.fill` | Order placement on the trading venue / each fill booked | `market`, `side`, `order_type`, `quantity`, `price`, `order_id`, `status`, `fills` |
| `vpp.cluster.call` / `vpp.cluster.execute` | A call forwarded to the lease holder (multi-worker) / its execution there | `target`, `method`, `call_id`, `caller`, `status` |

A failed device write, a failed solve or an error escaping a span marks the
span as failed (status `ERROR`), so "show me failing traces" in Jaeger finds them.

### Following one dispatch

`POST /api/v1/optimization/dispatch` with `"apply": true` produces:

```
POST /api/v1/optimization/dispatch                 (server, FastAPI)
├── SELECT resources ... / INSERT optimization_runs   (SQLAlchemy)
└── vpp.dispatch.execute
    ├── vpp.optimization.solve                     strategy=power_allocation solver=pyomo_highs_allocation status=success
    ├── vpp.actuator.setpoint  resource_id=bat-1    status=accepted verified=true
    │   └── vpp.modbus.write
    ├── vpp.actuator.setpoint  resource_id=bat-2    status=failed failed=true   (ERROR)
    └── UPDATE optimization_runs ...
```

With several API workers (`VPP_API_WORKERS>1`), calls that must run on the
lease holder (trading venue, alert evaluation) are forwarded through the
`cluster_calls` table. The caller's trace context travels in the call's
payload (`_trace_context: {"traceparent": "00-..."}`, stripped before the
handler runs), so the executing worker's `vpp.cluster.execute` span, and the
`vpp.market.order.submit` under it, are children of the caller's
`vpp.cluster.call` span: one trace across both processes.

### Local Jaeger with docker compose

The `tracing` compose profile adds Jaeger (OTLP on 4317/4318, UI on 16686,
in-memory storage); the API container exports to it by default:

```bash
export VPP_SECRET_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(48))")
VPP_TRACING_ENABLED=true docker compose --profile tracing up -d --build
# make a request, e.g. a dispatch, then open the UI:
#   http://localhost:16686  -> Service "vpp" -> Find Traces
```

To sample 10% of traces: `OTEL_TRACES_SAMPLER=parentbased_traceidratio
OTEL_TRACES_SAMPLER_ARG=0.1 VPP_TRACING_ENABLED=true docker compose --profile tracing up -d`.
To send to your own collector instead, set `OTEL_EXPORTER_OTLP_ENDPOINT`
(e.g. `http://otel-collector:4318`) and leave the profile off.

Running the API outside Docker against the same Jaeger:

```bash
docker compose --profile tracing up -d jaeger
pip install -e ".[api,db,cli,tracing]"
VPP_TRACING_ENABLED=true OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 vpp serve
```

Quick check without any backend: `OTEL_TRACES_EXPORTER=console
VPP_TRACING_ENABLED=true vpp serve` prints every finished span as JSON.

### Adding spans

```python
from vpp import tracing

with tracing.span("vpp.my.operation", resource_id=rid, kw=kw) as sp:
    result = await do_it()
    sp.set(status=result.status)        # recorded as vpp.status
    if not result.ok:
        sp.error(result.error)          # marks the span failed
```

`None` attributes are dropped and non-primitive values are stringified. Guard
expensive attribute computation with `if sp.recording:`. To carry a trace
across a process boundary, store `tracing.inject()` with the message and wrap
the receiving side in `with tracing.continue_trace(carrier):`.
