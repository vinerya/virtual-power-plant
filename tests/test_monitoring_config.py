"""Grafana/Prometheus config must stay in sync with the exported metrics."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

from vpp.metrics import REGISTRY  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "monitoring"
DASHBOARDS = sorted((ROOT / "grafana" / "dashboards").glob("*.json"))
_SUFFIXES = ("", "_total", "_bucket", "_count", "_sum", "_created", "_info")


def _exported_names() -> set[str]:
    names: set[str] = set()
    for family in REGISTRY.collect():
        names.update(family.name + s for s in _SUFFIXES)
    return names


def test_expected_dashboards_exist():
    assert {p.stem for p in DASHBOARDS} == {"vpp-overview", "vpp-trading", "vpp-fleet"}


@pytest.mark.parametrize("path", DASHBOARDS, ids=lambda p: p.stem)
def test_dashboard_queries_reference_real_metrics(path: Path):
    dash = json.loads(path.read_text())
    exported = _exported_names()
    exprs = [t["expr"] for p in dash["panels"] for t in p.get("targets", [])]
    assert exprs
    used = {m for e in exprs for m in re.findall(r"\bvpp_[a-z0-9_]+", e)}
    missing = used - exported
    assert not missing, f"{path.name} queries unknown metrics: {sorted(missing)}"
    for panel in dash["panels"]:
        assert panel["datasource"]["uid"] == "prometheus"
    ids = [p["id"] for p in dash["panels"]]
    assert len(ids) == len(set(ids))


def test_provisioning_files_are_consistent():
    ds = yaml.safe_load((ROOT / "grafana/provisioning/datasources/prometheus.yml").read_text())
    assert ds["datasources"][0]["uid"] == "prometheus"
    prov = yaml.safe_load((ROOT / "grafana/provisioning/dashboards/vpp.yml").read_text())
    dash_path = prov["providers"][0]["options"]["path"]

    compose = yaml.safe_load((ROOT / "docker-compose.monitoring.yml").read_text())
    grafana_volumes = compose["services"]["grafana"]["volumes"]
    assert f"./monitoring/grafana/dashboards:{dash_path}:ro" in grafana_volumes
    assert "./monitoring/grafana/provisioning:/etc/grafana/provisioning:ro" in grafana_volumes

    prom = yaml.safe_load((ROOT / "prometheus/prometheus.yml").read_text())
    job = next(j for j in prom["scrape_configs"] if j["job_name"] == "vpp-api")
    assert job["metrics_path"] == "/metrics"
