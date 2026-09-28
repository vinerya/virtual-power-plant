"""NEM 3.0 avoided-cost vector shapes: 24, 12x24, 8760 and 8784 (and flat)."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest
from httpx import AsyncClient

from vpp.tariffs.meter import MeterTrace
from vpp.tariffs.nem import (
    NEMConfigError,
    avoided_cost_at,
    compute_export_credit,
    nem_config_from_urdb,
    normalize_avoided_cost,
)
from vpp.tariffs.optimization import tariff_to_opt_params
from vpp.tariffs.preset_library import get_preset
from vpp.tariffs.urdb import load_urdb_json


def _index_vector(n: int) -> list[float]:
    """Vector whose value at position i is i, so lookups reveal the index."""
    return [float(i) for i in range(n)]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 24, 288, 8760, 8784])
def test_accepted_flat_lengths(n):
    assert normalize_avoided_cost(_index_vector(n)) == tuple(_index_vector(n))


def test_nested_month_by_hour_is_flattened_month_major():
    nested = [[m * 100 + h for h in range(24)] for m in range(12)]
    flat = normalize_avoided_cost(nested)
    assert len(flat) == 288
    assert flat[0] == 0.0 and flat[24] == 100.0 and flat[-1] == 1123.0


def test_empty_and_none_mean_no_vector():
    assert normalize_avoided_cost(None) == ()
    assert normalize_avoided_cost([]) == ()


@pytest.mark.parametrize(
    "raw",
    [
        [0.1] * 23,
        [0.1] * 48,  # not a supported shape (used to be indexed modulo length)
        [0.1] * 8761,
        [[0.1] * 24] * 11,
        [[0.1] * 23] * 12,
        [0.1, "x"],
        [0.1, math.nan] + [0.1] * 22,
        [0.1, math.inf] + [0.1] * 22,
        "0.1",
        {"a": 1},
    ],
)
def test_rejected_shapes(raw):
    with pytest.raises(NEMConfigError):
        normalize_avoided_cost(raw)


def test_invalid_stored_vector_is_ignored_not_raised():
    cfg = nem_config_from_urdb({"nem": "nem3", "nem3_avoided_cost": [0.1] * 25})
    assert cfg.regime == "nem3" and cfg.avoided_cost == ()


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------


def test_hour_of_day_vector():
    acc = _index_vector(24)
    assert avoided_cost_at(acc, datetime(2023, 3, 5, 17, 30)) == 17


def test_month_by_hour_vector():
    acc = _index_vector(288)
    assert avoided_cost_at(acc, datetime(2023, 1, 1, 0)) == 0
    assert avoided_cost_at(acc, datetime(2023, 7, 20, 18)) == 6 * 24 + 18
    assert avoided_cost_at(acc, datetime(2023, 12, 31, 23)) == 287


def test_hour_of_year_8760():
    acc = _index_vector(8760)
    assert avoided_cost_at(acc, datetime(2023, 1, 1, 0)) == 0
    assert avoided_cost_at(acc, datetime(2023, 2, 1, 5)) == 31 * 24 + 5
    assert avoided_cost_at(acc, datetime(2023, 12, 31, 23)) == 8759
    # Leap year: Feb 29 reuses Feb 28, and the rest of the year stays aligned
    # with the calendar date (no overflow on Dec 31).
    feb28 = avoided_cost_at(acc, datetime(2024, 2, 28, 12))
    assert avoided_cost_at(acc, datetime(2024, 2, 29, 12)) == feb28 == 58 * 24 + 12
    assert avoided_cost_at(acc, datetime(2024, 3, 1, 0)) == 59 * 24
    assert avoided_cost_at(acc, datetime(2024, 12, 31, 23)) == 8759


def test_hour_of_year_8784():
    acc = _index_vector(8784)
    assert avoided_cost_at(acc, datetime(2024, 2, 29, 0)) == 59 * 24
    assert avoided_cost_at(acc, datetime(2024, 12, 31, 23)) == 8783
    # Non-leap year: the Feb 29 block is skipped, so dates stay aligned.
    assert avoided_cost_at(acc, datetime(2023, 3, 1, 0)) == 60 * 24
    assert avoided_cost_at(acc, datetime(2023, 12, 31, 23)) == 8783


def test_uses_local_wall_clock_time():
    tz = ZoneInfo("America/Los_Angeles")
    utc = datetime(2024, 7, 1, 2, tzinfo=timezone.utc)  # 19:00 PDT on June 30
    local = utc.astimezone(tz)
    assert avoided_cost_at(_index_vector(24), local) == 19
    assert avoided_cost_at(_index_vector(288), local) == 5 * 24 + 19


# ---------------------------------------------------------------------------
# Bill credit and optimizer projection
# ---------------------------------------------------------------------------


def _tariff():
    return load_urdb_json(get_preset("pge_etouc"))


def test_export_credit_with_month_by_hour_vector():
    tz = ZoneInfo("America/Los_Angeles")
    start = datetime(2024, 7, 1, tzinfo=tz)
    n = 24
    exports = [0.0] * n
    exports[18] = 10.0
    trace = MeterTrace(
        timestamps=[start + timedelta(hours=i) for i in range(n)],
        import_kwh=[0.0] * n,
        export_kwh=exports,
        interval_minutes=60,
        tz=tz,
    )
    acc = [[0.01] * 24 for _ in range(12)]
    acc[6][18] = 0.30  # July, 18:00
    result = compute_export_credit(
        _tariff(),
        trace,
        start.astimezone(timezone.utc),
        (start + timedelta(hours=n)).astimezone(timezone.utc),
        regime="nem3",
        avoided_cost=acc,
    )
    assert result.amount == pytest.approx(3.0)


def test_export_credit_rejects_bad_shape():
    start = datetime(2024, 7, 1, tzinfo=timezone.utc)
    trace = MeterTrace(
        timestamps=[start],
        import_kwh=[0.0],
        export_kwh=[1.0],
        interval_minutes=60,
    )
    with pytest.raises(NEMConfigError):
        compute_export_credit(
            _tariff(),
            trace,
            start,
            start + timedelta(hours=1),
            regime="nem3",
            avoided_cost=[0.1] * 48,
        )


def test_opt_params_index_8760_by_calendar_hour_not_horizon_step():
    """A full-year vector is looked up by hour of year, not horizon offset."""
    horizon_start = datetime(2023, 7, 17, 0, tzinfo=timezone.utc)
    p = tariff_to_opt_params(
        _tariff(), horizon_start, 6, 60, nem="nem3", nem3_avoided_cost=_index_vector(8760)
    )
    first = (horizon_start.timetuple().tm_yday - 1) * 24
    assert p.energy_sell_per_kwh == [float(first + h) for h in range(6)]


def test_opt_params_month_by_hour_in_local_tz_with_sub_hourly_steps():
    tz = ZoneInfo("America/Los_Angeles")
    horizon_start = datetime(2024, 1, 1, 7, tzinfo=timezone.utc)  # 23:00 PST Dec 31
    nested = [[m * 100 + h for h in range(24)] for m in range(12)]
    p = tariff_to_opt_params(
        _tariff(), horizon_start, 1, 15, nem="nem3", nem3_avoided_cost=nested, tz=tz
    )
    assert p.energy_sell_per_kwh == [1123.0] * 4  # December, 23:00 local


def test_opt_params_reject_bad_shape():
    with pytest.raises(ValueError):
        tariff_to_opt_params(
            _tariff(),
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            24,
            60,
            nem="nem3",
            nem3_avoided_cost=[0.1] * 48,
        )


# ---------------------------------------------------------------------------
# API validation
# ---------------------------------------------------------------------------


def _simulate_body(avoided_cost) -> dict:
    start = datetime(2024, 7, 1, tzinfo=timezone.utc)
    n = 24
    exports = [0.0] * n
    exports[18] = 50.0
    return {
        "urdb_json": get_preset("pge_etouc"),
        "meter_trace": {
            "timestamps": [(start + timedelta(hours=i)).isoformat() for i in range(n)],
            "import_kwh": [1.0] * n,
            "export_kwh": exports,
            "interval_minutes": 60,
        },
        "billing_period_start": start.isoformat(),
        "billing_period_end": (start + timedelta(hours=n)).isoformat(),
        "nem": "nem3",
        "nem3_avoided_cost": avoided_cost,
    }


@pytest.mark.asyncio
async def test_simulate_accepts_month_by_hour(client: AsyncClient, auth_headers: dict):
    acc = [[0.05] * 24 for _ in range(12)]
    acc[6][18] = 0.20
    resp = await client.post(
        "/api/v1/tariffs/simulate", json=_simulate_body(acc), headers=auth_headers
    )
    assert resp.status_code == 200, resp.text
    credits = [li for li in resp.json()["line_items"] if li["kind"] == "credit"]
    assert credits[0]["amount"] == pytest.approx(-10.0, abs=1e-2)


@pytest.mark.asyncio
async def test_simulate_rejects_bad_shape(client: AsyncClient, auth_headers: dict):
    resp = await client.post(
        "/api/v1/tariffs/simulate", json=_simulate_body([0.1] * 48), headers=auth_headers
    )
    assert resp.status_code == 422
    assert "nem3_avoided_cost" in resp.text


@pytest.mark.asyncio
async def test_create_tariff_rejects_bad_avoided_cost(client: AsyncClient, auth_headers: dict):
    urdb = {**get_preset("pge_etouc"), "nem": "nem3", "nem3_avoided_cost": [0.1] * 25}
    resp = await client.post(
        "/api/v1/tariffs",
        json={"name": "bad avoided cost", "urdb_json": urdb},
        headers=auth_headers,
    )
    assert resp.status_code == 422
    assert "nem3_avoided_cost" in resp.json()["detail"]
