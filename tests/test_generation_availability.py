"""Availability of uncurtailed generation (solar, wind) for dispatch.

Fresh telemetry is taken as measured (0 kW at night is 0 kW available);
missing or stale telemetry falls back per
``VPP_GENERATION_AVAILABILITY_FALLBACK`` (``zero`` by default, never the
nameplate unless asked for). Curtailed resources keep their own estimate
(see tests/test_device_control.py).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from _v2g_helpers import tmp_session_factory

from vpp.api.optimization_support import resource_to_asset, telemetry_is_fresh
from vpp.control.actuator import OutputLimit
from vpp.db.models import ResourceModel
from vpp.portal.telemetry import record_samples
from vpp.settings import get_settings

GENERATORS = ("solar", "wind_turbine")


@pytest.fixture
def db(tmp_path):
    return tmp_session_factory(tmp_path / "avail.db")


@pytest.fixture
def setting(monkeypatch):
    """Set VPP_* environment variables for the cached settings."""

    def _set(**env: str) -> None:
        for key, value in env.items():
            monkeypatch.setenv(f"VPP_{key.upper()}", value)
        get_settings.cache_clear()

    get_settings.cache_clear()
    yield _set
    monkeypatch.undo()
    get_settings.cache_clear()


async def _asset(
    db,
    rtype: str,
    *,
    current_kw: float = 0.0,
    sample_age_s: float | None = None,
    meta: dict | None = None,
    limit: OutputLimit | None = None,
):
    now = datetime.now(timezone.utc)
    async with db() as s:
        row = ResourceModel(
            id="g",
            name="g",
            resource_type=rtype,
            rated_power=10.0,
            current_power=current_kw,
            metadata_json=json.dumps(meta or {}),
        )
        s.add(row)
        await s.flush()
        if sample_age_s is not None:
            sample = (now - timedelta(seconds=sample_age_s), current_kw, None)
            await record_samples(s, "g", [sample], source="t")
        await s.commit()
        return await resource_to_asset(s, row, {"g": limit} if limit else None)


@pytest.mark.parametrize("rtype", GENERATORS)
async def test_fresh_zero_output_is_zero_available(db, setting, rtype):
    setting(generation_availability_fallback="nameplate")  # does not apply
    asset = await _asset(db, rtype, current_kw=0.0, sample_age_s=10)
    assert (asset.available_kw, asset.availability_basis) == (0.0, "telemetry")


@pytest.mark.parametrize("rtype", GENERATORS)
async def test_fresh_output_is_available(db, setting, rtype):
    setting()
    asset = await _asset(db, rtype, current_kw=3.5, sample_age_s=10)
    assert (asset.available_kw, asset.availability_basis) == (3.5, "telemetry")
    assert asset.summary()["availability_basis"] == "telemetry"


@pytest.mark.parametrize("rtype", GENERATORS)
@pytest.mark.parametrize("sample_age_s", [None, 3600])  # no telemetry / stale
async def test_missing_or_stale_telemetry_defaults_to_no_data(db, setting, rtype, sample_age_s):
    setting()
    assert get_settings().generation_availability_fallback == "zero"
    asset = await _asset(db, rtype, current_kw=4.0, sample_age_s=sample_age_s)
    assert (asset.available_kw, asset.availability_basis) == (0.0, "no_data")


@pytest.mark.parametrize("rtype", GENERATORS)
@pytest.mark.parametrize("sample_age_s", [None, 3600])
async def test_nameplate_fallback_is_opt_in(db, setting, rtype, sample_age_s):
    setting(generation_availability_fallback="Nameplate")
    asset = await _asset(db, rtype, sample_age_s=sample_age_s)
    assert (asset.available_kw, asset.availability_basis) == (10.0, "nameplate_fallback")


@pytest.mark.parametrize("rtype", GENERATORS)
async def test_forecast_fallback(db, setting, rtype):
    setting(generation_availability_fallback="forecast")
    now = datetime.now(timezone.utc)
    forecast = [
        {"timestamp": (now - timedelta(minutes=90)).isoformat(), "power_kw": 9.0},
        {"timestamp": (now - timedelta(minutes=20)).isoformat(), "power_kw": 6.5},
        {"timestamp": (now + timedelta(minutes=40)).isoformat(), "power_kw": 1.0},
        {"timestamp": "not a date", "power_kw": 3.0},
    ]
    asset = await _asset(db, rtype, meta={"generation_forecast": forecast})
    assert (asset.available_kw, asset.availability_basis) == (6.5, "forecast")


async def test_forecast_is_clamped_and_needs_a_current_point(db, setting, tmp_path):
    setting(generation_availability_fallback="forecast")
    now = datetime.now(timezone.utc)
    too_big = [{"timestamp": now.isoformat(), "power_kw": 25.0}]
    asset = await _asset(db, "solar", meta={"generation_forecast": too_big})
    assert (asset.available_kw, asset.availability_basis) == (10.0, "forecast")

    old = [{"timestamp": (now - timedelta(hours=2)).isoformat(), "power_kw": 5.0}]
    other = tmp_session_factory(tmp_path / "old.db")
    asset = await _asset(other, "solar", meta={"generation_forecast": old})
    assert (asset.available_kw, asset.availability_basis) == (0.0, "no_data")


async def test_forecast_mode_without_forecast_is_no_data(db, setting):
    setting(generation_availability_fallback="forecast")
    asset = await _asset(db, "wind_turbine", sample_age_s=None)
    assert (asset.available_kw, asset.availability_basis) == (0.0, "no_data")


async def test_forecast_is_ignored_in_zero_mode_and_with_fresh_telemetry(db, setting, tmp_path):
    now = datetime.now(timezone.utc)
    forecast = [{"timestamp": now.isoformat(), "power_kw": 7.0}]
    setting()
    asset = await _asset(db, "solar", meta={"generation_forecast": forecast})
    assert asset.availability_basis == "no_data"
    setting(generation_availability_fallback="forecast")
    other = tmp_session_factory(tmp_path / "fresh.db")
    asset = await _asset(
        other, "solar", current_kw=0.0, sample_age_s=5, meta={"generation_forecast": forecast}
    )
    assert (asset.available_kw, asset.availability_basis) == (0.0, "telemetry")


async def test_reported_available_kw_still_wins(db, setting):
    setting()
    asset = await _asset(db, "solar", meta={"available_kw": 5.5})
    assert (asset.available_kw, asset.availability_basis) == (5.5, "reported")


async def test_max_age_setting(db, setting):
    setting(generation_telemetry_max_age_s="30")
    asset = await _asset(db, "solar", current_kw=4.0, sample_age_s=60)
    assert asset.availability_basis == "no_data"
    now = datetime.now(timezone.utc)
    assert telemetry_is_fresh(now - timedelta(seconds=29), now=now)
    assert not telemetry_is_fresh(now - timedelta(seconds=31), now=now)
    assert not telemetry_is_fresh(None, now=now)
    assert telemetry_is_fresh(now + timedelta(seconds=5), now=now)  # clock skew


async def test_curtailed_resource_without_telemetry_keeps_its_estimate(db, setting):
    """The curtailment rules do not depend on the fallback setting."""
    setting()
    since = (datetime.now(timezone.utc) - timedelta(seconds=30)).timestamp()
    asset = await _asset(db, "solar", current_kw=4.0, limit=OutputLimit("g", 4.0, since))
    assert (asset.available_kw, asset.availability_basis) == (10.0, "nameplate_curtailed")


def test_invalid_fallback_is_rejected(setting):
    setting(generation_availability_fallback="always")
    with pytest.raises(ValueError, match="generation_availability_fallback"):
        get_settings()


async def test_dispatch_reports_the_basis_and_plans_no_export_without_data(client, auth_headers):
    """A PV with no telemetry is planned at 0 kW; the run records why."""
    rids = []
    for rtype, rated, meta in (
        ("battery", 10.0, {"capacity_kwh": 40, "soc": 0.8}),
        ("solar", 20.0, {}),
    ):
        resp = await client.post(
            "/api/v1/resources/",
            json={
                "name": f"avail-{rtype}",
                "resource_type": rtype,
                "rated_power": rated,
                "metadata": meta,
            },
            headers=auth_headers,
        )
        assert resp.status_code == 201, resp.text
        rids.append(resp.json()["id"])
    battery, pv = rids

    resp = await client.post(
        "/api/v1/optimization/dispatch",
        json={"target_power_kw": 8.0, "resource_ids": rids},
        headers=auth_headers,
    )
    assert resp.status_code == 200, resp.text
    alloc = {a["resource_id"]: a for a in resp.json()["allocations"]}
    assert alloc[pv]["availability_basis"] == "no_data"
    assert alloc[pv]["available_power_kw"] == 0.0
    assert alloc[pv]["allocated_power_kw"] == pytest.approx(0.0, abs=1e-6)
    assert alloc[battery]["allocated_power_kw"] == pytest.approx(8.0, abs=1e-3)

    run_id = resp.json()["run_id"]
    run = (await client.get(f"/api/v1/optimization/runs/{run_id}", headers=auth_headers)).json()
    summary = {r["id"]: r for r in run["inputs"]["resources"]}
    assert summary[pv]["availability_basis"] == "no_data"
