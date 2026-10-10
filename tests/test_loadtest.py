"""Unit tests for the load / soak harness (``vpp loadtest``).

The harness itself is exercised end to end by the CI ``perf`` job
(``vpp loadtest --smoke``); these cover its bookkeeping and reporting.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from click.testing import CliRunner

from vpp.cli.main import cli
from vpp.loadtest import LoadConfig, procinfo
from vpp.loadtest.report import render_markdown
from vpp.loadtest.runner import _redact, _scan_log, _trends, db_file_bytes, sqlite_path
from vpp.loadtest.stats import Recorder, linear_slope, percentile, thirds_ratio


def test_percentile_nearest_rank():
    values = list(range(1, 101))
    assert percentile(values, 50) == 50
    assert percentile(values, 95) == 95
    assert percentile(values, 99) == 99
    assert percentile(values, 100) == 100
    assert percentile([], 50) is None
    assert percentile([7], 99) == 7


def test_recorder_windows_and_summary():
    rec = Recorder("x")
    for ms in (10, 20, 30, 40):
        rec.ok(ms / 1000)
    rec.error("HTTP 500")
    rec.skip()
    w = rec.window(2.0)
    assert w["ops"] == 4 and w["rate"] == 2.0 and w["errors"] == 1
    assert w["p50_ms"] == 20.0 and w["max_ms"] == 40.0
    # The window resets; the run totals do not.
    assert rec.window(1.0)["ops"] == 0
    s = rec.summary(4.0)
    assert s["ok"] == 4 and s["errors"] == 1 and s["skipped"] == 1
    assert s["error_rate"] == pytest.approx(0.2)
    assert s["throughput_per_s"] == 1.0
    assert s["error_kinds"] == {"HTTP 500": 1}


def test_trend_helpers():
    assert linear_slope([0, 1, 2, 3], [10, 12, 14, 16]) == pytest.approx(2.0)
    assert linear_slope([0, 1], [1, 2]) is None
    assert thirds_ratio([10, 10, 10, 20, 20, 20, 30, 30, 30]) == 3.0
    assert thirds_ratio([1, 2]) is None


def test_trends_ignore_warmup_and_drain():
    timeline = [
        {"t": 0.0, "phase": "load", "server_rss_mb": 100.0, "db_mb": 1.0, "scenarios": {}},
        *(
            {
                "t": float(t),
                "phase": "load",
                "server_rss_mb": 200.0,
                "db_mb": t / 100,
                "scenarios": {"api_read": {"p95_ms": 10.0}},
            }
            for t in range(20, 101, 10)
        ),
        # The drain sample (load stopped) must not count as latency creep.
        {"t": 110.0, "phase": "drain", "server_rss_mb": 900.0, "scenarios": {}},
    ]
    trends = _trends(timeline, 100.0)
    assert trends["server_rss_mb_per_hour"] == pytest.approx(0.0)
    assert trends["p95_last_third_over_first_third"]["api_read"] == 1.0
    assert trends["server_rss_mb_peak"] == 900.0


def test_scan_log_counts_errors_and_warnings(tmp_path: Path):
    log = tmp_path / "server.log"
    log.write_text(
        '{"event": "fine", "level": "info"}\n'
        '{"event": "queue full, dropped 12", "level": "warning"}\n'
        '{"event": "queue full, dropped 30", "level": "warning"}\n'
        '{"event": "boom", "level": "error"}\n'
        "Traceback (most recent call last):\n"
        "ERROR:    Exception in ASGI application\n"
    )
    stats = _scan_log(log)
    assert stats["errors"] == 3
    assert stats["warnings"] == 2
    assert stats["top"]["WARNING queue full, dropped N"] == 2


def test_sqlite_url_helpers(tmp_path: Path):
    db = tmp_path / "x.db"
    db.write_bytes(b"a" * 100)
    (tmp_path / "x.db-wal").write_bytes(b"b" * 20)
    url = f"sqlite+aiosqlite:///{db}"
    assert sqlite_path(url) == db
    assert db_file_bytes(url) == 120
    assert sqlite_path("postgresql+asyncpg://u:p@h/db") is None
    assert _redact("postgresql+asyncpg://vpp:secret@h:5432/db") == (
        "postgresql+asyncpg://vpp:***@h:5432/db"
    )


@pytest.mark.skipif(not Path("/proc/self/status").exists(), reason="Linux /proc only")
def test_procinfo_reads_own_process():
    pids = procinfo.process_tree(os.getpid())
    assert pids[0] == os.getpid()
    assert (procinfo.rss_bytes(pids) or 0) > 10 * 1024 * 1024
    assert (procinfo.cpu_seconds(pids) or 0) > 0
    info = procinfo.machine_info()
    assert info["logical_cpus"] and info["python"]


def test_config_validation(tmp_path: Path):
    LoadConfig(output_dir=tmp_path).validate()
    with pytest.raises(ValueError):
        LoadConfig(output_dir=tmp_path, ingest="carrier-pigeon").validate()
    with pytest.raises(ValueError):
        LoadConfig(output_dir=tmp_path, resources=0).validate()


def test_render_markdown_from_minimal_report():
    scen = {
        "ok": 10,
        "errors": 0,
        "skipped": 0,
        "error_rate": 0.0,
        "throughput_per_s": 1.0,
        "p50_ms": 1.0,
        "p95_ms": 2.0,
        "p99_ms": 3.0,
        "max_ms": 4.0,
        "error_kinds": {},
    }
    report = {
        "config": {
            **vars(LoadConfig(output_dir=Path("x"))),
            "dialect": "sqlite",
        },
        "machine": procinfo.machine_info(),
        "duration_s": 10.0,
        "scenarios": {"api_read": scen},
        "endpoints": {"resources": scen},
        "trends": {"p95_last_third_over_first_third": {"api_read": 1.1}},
        "db": {"start_mb": 1.0, "end_mb": 2.0, "row_growth": {"battery_states": 5}},
        "server_log": {"errors": 0, "warnings": 0, "top": {}},
        "websocket": {"messages": 3, "reconnects": 0},
        "dispatch": {"methods": {}, "solve_ms_p50": None, "solve_ms_p95": None},
        "total_errors": 0,
    }
    md = render_markdown(report)
    assert "(b) API reads (all endpoints)" in md
    assert "| resources |" in md
    assert "Total errors: 0" in md


def test_cli_lists_the_command():
    result = CliRunner().invoke(cli, ["loadtest", "--help"])
    assert result.exit_code == 0
    assert "--smoke" in result.output
    bad = CliRunner().invoke(cli, ["loadtest", "--server-env", "NOEQUALS"])
    assert bad.exit_code != 0
    assert "KEY=VALUE" in bad.output


def test_mqtt_messages_missed_by_the_probe_but_persisted_are_not_errors(tmp_path: Path):
    from vpp.loadtest.runner import LoadRun

    run = LoadRun(LoadConfig(output_dir=tmp_path))
    run.published = 100
    run.pending = {("r", float(i)): 0.0 for i in range(7)}  # never seen on the socket
    run.timeline = [{"rows": {"battery_states": 10}}]
    run._count_lost_mqtt({"rows": {"battery_states": 107}})  # 97 of 100 persisted
    assert run.mqtt_lost == 3
    assert run.mqtt_unobserved == 4
    rec = run.rec["ingest_mqtt_e2e"]
    assert rec.errors == 3
    assert rec.error_kinds == {"not persisted": 3}
