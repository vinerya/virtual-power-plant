"""Typed resource create/update: fields are validated, persisted and used by
the optimizer's resource loader."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from _portal_helpers import uid

from vpp.api.optimization_support import load_fleet_assets
from vpp.db.models import ResourceModel

if TYPE_CHECKING:
    from httpx import AsyncClient

BATTERY = {
    "resource_type": "battery",
    "rated_power": 50.0,
    "capacity_kwh": 200.0,
    "current_charge_kwh": 50.0,
    "chemistry": "NMC",
    "nominal_voltage": 800.0,
    "charge_efficiency": 0.96,
    "discharge_efficiency": 0.97,
    "max_charge_kw": 40.0,
    "max_discharge_kw": 45.0,
    "soc_min": 0.1,
    "soc_max": 0.9,
    "efficiency": 0.93,
}


async def _create(client: AsyncClient, headers: dict, body: dict) -> dict:
    resp = await client.post(
        "/api/v1/resources", json={"name": uid("res"), **body}, headers=headers
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


@pytest.mark.asyncio
async def test_battery_fields_persisted_and_returned(
    client: AsyncClient, auth_headers: dict, db_session
):
    body = await _create(client, auth_headers, BATTERY)
    assert body["capacity_kwh"] == 200.0
    assert body["state_of_charge"] == pytest.approx(0.25)
    assert body["state_of_charge_source"] == "configured"
    assert body["current_charge_kwh"] == pytest.approx(50.0)
    assert body["state_of_health"] == 1.0
    assert body["equivalent_full_cycles"] == 0.0
    assert body["chemistry"] == "nmc"
    assert body["max_charge_kw"] == 40.0 and body["max_discharge_kw"] == 45.0
    assert body["soc_min"] == 0.1 and body["soc_max"] == 0.9
    assert body["efficiency"] == 0.93
    assert body["nominal_voltage"] == 800.0
    # Solar/wind fields are present but null on a battery.
    assert body["dc_capacity_kw"] is None and body["cut_in_speed_ms"] is None

    row = await db_session.get(ResourceModel, body["id"])
    assert row.nominal_energy_kwh == 200.0
    assert row.chemistry == "nmc"
    assert row.efficiency == 0.93
    cfg = json.loads(row.config_json)
    assert cfg["state_of_charge"] == pytest.approx(0.25)
    assert cfg["max_charge_kw"] == 40.0
    assert "capacity_kwh" not in cfg and "current_charge_kwh" not in cfg

    # GET returns the same shape.
    got = (await client.get(f"/api/v1/resources/{body['id']}", headers=auth_headers)).json()
    assert got["capacity_kwh"] == 200.0 and got["state_of_charge"] == pytest.approx(0.25)


@pytest.mark.asyncio
async def test_optimizer_loader_uses_typed_fields(
    client: AsyncClient, auth_headers: dict, db_session
):
    body = await _create(client, auth_headers, BATTERY)
    assets, missing = await load_fleet_assets(db_session, [body["id"]])
    assert missing == []
    (a,) = assets
    assert a.capacity_kwh == 200.0 and a.capacity_source == "recorded"
    assert a.soc == pytest.approx(0.25) and a.soc_source == "config"
    assert a.chemistry == "nmc"
    assert (a.eta_charge, a.eta_discharge) == (0.96, 0.97)
    assert (a.soc_min, a.soc_max) == (0.1, 0.9)
    assert a.charge_limit_kw == 40.0 and a.discharge_limit_kw == 45.0
    params = a.battery_params()
    assert params["max_charge_kw"] == 40.0 and params["max_discharge_kw"] == 45.0


@pytest.mark.asyncio
async def test_dispatch_respects_discharge_limit(client: AsyncClient, auth_headers: dict):
    body = await _create(
        client,
        auth_headers,
        {
            "resource_type": "battery",
            "rated_power": 100.0,
            "capacity_kwh": 400.0,
            "state_of_charge": 0.8,
            "max_discharge_kw": 30.0,
        },
    )
    resp = await client.post(
        "/api/v1/optimization/dispatch",
        json={"target_power_kw": 80.0, "resource_ids": [body["id"]]},
        headers=auth_headers,
    )
    assert resp.status_code == 200, resp.text
    (alloc,) = resp.json()["allocations"]
    assert alloc["available_power_kw"] <= 30.0 + 1e-6
    assert alloc["allocated_power_kw"] <= 30.0 + 1e-6


@pytest.mark.asyncio
async def test_telemetry_soc_overrides_configured(client: AsyncClient, auth_headers: dict):
    body = await _create(client, auth_headers, BATTERY)
    resp = await client.post(
        f"/api/v1/resources/{body['id']}/telemetry",
        json={"samples": [{"power_kw": 1.0, "state_of_charge": 0.6}]},
        headers=auth_headers,
    )
    assert resp.status_code == 202, resp.text
    got = (await client.get(f"/api/v1/resources/{body['id']}", headers=auth_headers)).json()
    assert got["state_of_charge"] == pytest.approx(0.6)
    assert got["state_of_charge_source"] == "telemetry"
    listed = (await client.get("/api/v1/resources?limit=200", headers=auth_headers)).json()
    assert next(r for r in listed if r["id"] == body["id"])["state_of_charge"] == pytest.approx(
        0.6
    )


@pytest.mark.asyncio
async def test_solar_and_wind_alias(client: AsyncClient, auth_headers: dict):
    solar = await _create(
        client,
        auth_headers,
        {
            "resource_type": "solar",
            "rated_power": 8.0,
            "dc_capacity_kw": 10.0,
            "ac_capacity_kw": 8.0,
            "panel_efficiency": 0.21,
        },
    )
    assert solar["dc_capacity_kw"] == 10.0 and solar["ac_capacity_kw"] == 8.0
    assert solar["state_of_charge"] is None and solar["capacity_kwh"] is None

    wind = await _create(
        client,
        auth_headers,
        {
            "resource_type": "wind",
            "rated_power": 2000.0,
            "cut_in_speed_ms": 3.0,
            "rated_speed_ms": 12.0,
            "cut_out_speed_ms": 25.0,
            "rotor_diameter_m": 90.0,
        },
    )
    assert wind["resource_type"] == "wind_turbine"
    assert wind["cut_in_speed_ms"] == 3.0 and wind["rotor_diameter_m"] == 90.0

    only_wind = (
        await client.get("/api/v1/resources?resource_type=wind&limit=200", headers=auth_headers)
    ).json()
    assert wind["id"] in {r["id"] for r in only_wind}
    assert {r["resource_type"] for r in only_wind} == {"wind_turbine"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        {"resource_type": "hydro", "rated_power": 1.0},
        {"rated_power": 1.0},
        {"resource_type": "solar", "rated_power": 1.0, "capacity_kwh": 5.0},  # wrong type
        {"resource_type": "battery", "rated_power": 1.0, "bogus": 1},
        {"resource_type": "battery", "rated_power": 1.0, "current_charge_kwh": 5.0},
        {
            "resource_type": "battery",
            "rated_power": 1.0,
            "capacity_kwh": 5,
            "current_charge_kwh": 6,
        },
        {"resource_type": "battery", "rated_power": 1.0, "soc_min": 0.9, "soc_max": 0.1},
        {"resource_type": "battery", "rated_power": 1.0, "max_charge_kw": 2.0},
        {"resource_type": "battery", "rated_power": 1.0, "chemistry": "unobtainium"},
        {"resource_type": "battery", "rated_power": 1.0, "state_of_charge": 55},
        {
            "resource_type": "wind_turbine",
            "rated_power": 1.0,
            "cut_in_speed_ms": 5,
            "cut_out_speed_ms": 4,
        },
    ],
)
async def test_create_rejects_invalid(client: AsyncClient, auth_headers: dict, body: dict):
    resp = await client.post(
        "/api/v1/resources", json={"name": uid("bad"), **body}, headers=auth_headers
    )
    assert resp.status_code == 422, resp.text


@pytest.mark.asyncio
async def test_update_merges_and_revalidates(client: AsyncClient, auth_headers: dict, db_session):
    body = await _create(client, auth_headers, BATTERY)
    url = f"/api/v1/resources/{body['id']}"

    resp = await client.put(
        url, json={"capacity_kwh": 400.0, "state_of_charge": 0.5}, headers=auth_headers
    )
    assert resp.status_code == 200, resp.text
    got = resp.json()
    assert got["capacity_kwh"] == 400.0 and got["state_of_charge"] == pytest.approx(0.5)
    assert got["max_charge_kw"] == 40.0  # untouched fields survive
    assert got["chemistry"] == "nmc"

    resp = await client.put(url, json={"current_charge_kwh": 100.0}, headers=auth_headers)
    assert resp.status_code == 200, resp.text
    assert resp.json()["state_of_charge"] == pytest.approx(0.25)

    # Clearing an optional field.
    resp = await client.put(url, json={"max_charge_kw": None}, headers=auth_headers)
    assert resp.status_code == 200 and resp.json()["max_charge_kw"] is None

    # Cross-field rules apply to the merged result.
    for bad in (
        {"current_charge_kwh": 500.0},  # > capacity
        {"soc_min": 0.95},  # >= soc_max
        {"rated_power": 10.0},  # below max_discharge_kw
        {"panel_efficiency": 0.2},  # not a battery field
        {"resource_type": "solar"},  # type is immutable
        {"name": None},
    ):
        resp = await client.put(url, json=bad, headers=auth_headers)
        assert resp.status_code == 422, (bad, resp.text)

    row = await db_session.get(ResourceModel, body["id"])
    await db_session.refresh(row)
    assert row.nominal_energy_kwh == 400.0

    resp = await client.put(url, json={"online": False}, headers=auth_headers)
    assert resp.status_code == 200 and resp.json()["online"] is False


@pytest.mark.asyncio
async def test_update_name_conflict(client: AsyncClient, auth_headers: dict):
    a = await _create(client, auth_headers, {"resource_type": "solar", "rated_power": 1.0})
    b = await _create(client, auth_headers, {"resource_type": "solar", "rated_power": 1.0})
    resp = await client.put(
        f"/api/v1/resources/{b['id']}", json={"name": a["name"]}, headers=auth_headers
    )
    assert resp.status_code == 409
