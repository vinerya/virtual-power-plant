"""Render a load-test report dict as Markdown."""

from __future__ import annotations

from typing import Any

_SCENARIO_LABELS = {
    "ingest_mqtt_publish": "(a) MQTT publish (client side)",
    "ingest_mqtt_e2e": "(a) MQTT publish -> persisted -> WS broadcast",
    "ingest_http": "(a) POST /resources/{id}/telemetry",
    "api_read": "(b) API reads (all endpoints)",
    "ws_broadcast": "(c) WS broadcast (event -> client)",
    "ws_token": "(c) POST /ws/token",
    "dispatch": "(d) POST /optimization/dispatch",
    "modbus_update_interval": "(e) Modbus update interval per device",
}


def _f(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:,.2f}".rstrip("0").rstrip(".") if abs(v) < 1e6 else f"{v:,.0f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def _row(name: str, s: dict[str, Any]) -> str:
    return (
        f"| {name} | {_f(s.get('throughput_per_s'))} | {_f(s.get('p50_ms'))} | "
        f"{_f(s.get('p95_ms'))} | {_f(s.get('p99_ms'))} | {_f(s.get('max_ms'))} | "
        f"{_f(s.get('ok'))} | {_f(s.get('errors'))} | {_f(s.get('skipped'))} |"
    )


def render_markdown(report: dict[str, Any]) -> str:
    cfg = report["config"]
    m = report["machine"]
    lines = [
        f"# Load test report ({cfg['dialect']}, {report['duration_s']} s)",
        "",
        f"Machine: {m['cpu']}, {m['logical_cpus']} logical CPUs, {m['memory_gib']} GiB, "
        f"{m['os']}, Python {m['python']}, SQLite {m['sqlite']}.",
        "",
        f"Load: {cfg['resources']} batteries x {cfg['points_per_s']} points/s over "
        f"`{cfg['ingest']}`; {cfg['api_rps']} API reads/s; {cfg['ws_clients']} WebSocket "
        f"clients; dispatch over {cfg['dispatch_resources']} resources every "
        f"{cfg['dispatch_interval_s']} s; {cfg['modbus_devices']} Modbus simulators polled "
        f"every {cfg['modbus_poll_s']} s; {cfg['workers']} API worker(s).",
        "",
        "| Scenario | ops/s | p50 ms | p95 ms | p99 ms | max ms | ok | errors | skipped |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, s in report["scenarios"].items():
        lines.append(_row(_SCENARIO_LABELS.get(name, name), s))
    if report.get("endpoints"):
        lines += [
            "",
            "| GET endpoint | ops/s | p50 ms | p95 ms | p99 ms | max ms | ok | errors | skipped |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for name, s in sorted(report["endpoints"].items()):
            lines.append(_row(name, s))
    t = report["trends"]
    db = report["db"]
    log = report["server_log"]
    lines += [
        "",
        f"Server RSS: start {_f(t.get('server_rss_mb_start'))} MB, peak "
        f"{_f(t.get('server_rss_mb_peak'))} MB, end {_f(t.get('server_rss_mb_end'))} MB; "
        f"slope after warm-up {_f(t.get('server_rss_mb_per_hour'))} MB/h.",
        f"Database: {_f(db.get('start_mb'))} MB -> {_f(db.get('end_mb'))} MB "
        f"({_f(t.get('db_mb_per_hour'))} MB/h); rows added: "
        + ", ".join(f"{k} {_f(v)}" for k, v in db.get("row_growth", {}).items() if v),
        f"WebSocket: {_f(report['websocket']['messages'])} messages received, "
        f"{_f(report['websocket']['reconnects'])} reconnects.",
        f"Dispatch: methods {report['dispatch']['methods']}, solver p50 "
        f"{_f(report['dispatch']['solve_ms_p50'])} ms / p95 "
        f"{_f(report['dispatch']['solve_ms_p95'])} ms.",
        f"Server log: {log['errors']} error lines, {log['warnings']} warning lines.",
        "",
        "Latency creep (median p95 of the last third of the run / first third):",
        "",
    ]
    for name, ratio in t.get("p95_last_third_over_first_third", {}).items():
        lines.append(f"- {name}: {_f(ratio)}")
    if log.get("top"):
        lines += ["", "Most frequent server log warnings/errors:", ""]
        lines += [f"- {n} x {msg}" for msg, n in log["top"].items()]
    errs = {n: s["error_kinds"] for n, s in report["scenarios"].items() if s.get("error_kinds")}
    if errs:
        lines += ["", "Client-side error kinds:", ""]
        lines += [f"- {n}: {k}" for n, k in errs.items()]
    lines += ["", f"Total errors: {report['total_errors']}", ""]
    return "\n".join(lines)
