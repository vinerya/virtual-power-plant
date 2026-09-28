"""The tariff API contract the web console relies on.

Derived presentational fields on ``TariffRead``, presets, URDB-import status,
and the simulate flows the UI offers (synthetic load, CSV upload, compare to
another tariff, billing cycles, timezone, NEM derived from the tariff), plus
NEM export credit on the customer-portal bill.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
import pytest_asyncio
from _iso import parse_iso
from _portal_helpers import (
    FLAT_TARIFF_URDB,
    create_customer,
    create_site,
    isolated_client,
    uid,
    user_headers,
)

from vpp.tariffs.preset_library import get_preset

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def client(app):
    async with isolated_client(app) as c:
        yield c


async def _create(client, headers, urdb_json, name=None) -> dict:
    resp = await client.post(
        "/api/v1/tariffs",
        json={"name": name or uid("tariff"), "utility": "Test", "urdb_json": urdb_json},
        headers=headers,
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


# ---------------------------------------------------------------------------
# Read model
# ---------------------------------------------------------------------------


async def test_read_has_derived_components_and_heatmaps(client, auth_headers):
    t = await _create(client, auth_headers, get_preset("pge_etouc"))
    # Stored fields are unchanged.
    assert t["urdb_json"]["name"] == "PG&E E-TOU-C"
    assert t["sector"] == "Residential"
    assert t["is_tou"] is True
    assert len(t["tou_heatmap"]) == 12 and all(len(r) == 24 for r in t["tou_heatmap"])
    assert len(t["tou_heatmap_weekend"]) == 12
    # July 17:00 is summer peak (0.50), July 03:00 summer off-peak (0.39).
    assert t["tou_heatmap"][6][17] == pytest.approx(0.50)
    assert t["tou_heatmap"][6][3] == pytest.approx(0.39)
    kinds = [c["kind"] for c in t["components"]]
    assert kinds.count("energy") == 4
    assert {"minimum", "adder", "tax"} <= set(kinds)
    peak = next(c for c in t["components"] if c["name"] == "Energy — period 1")
    assert peak["unit"] == "$/kWh" and peak["rate"] == pytest.approx(0.50)
    assert peak["schedule"] == ["Jun-Sep, every day 16:00-21:00"]
    assert t["nem_regime"] == "none" and t["parse_error"] is None

    # GET returns the same derived view; list too.
    one = (await client.get(f"/api/v1/tariffs/{t['id']}", headers=auth_headers)).json()
    assert one["components"] == t["components"]
    listed = (await client.get("/api/v1/tariffs?limit=200", headers=auth_headers)).json()
    assert any(x["id"] == t["id"] and x["tou_heatmap"] for x in listed)


async def test_tiered_and_demand_components(client, auth_headers):
    res = await _create(client, auth_headers, get_preset("illustrative_residential_tiered"))
    tier = next(c for c in res["components"] if c["kind"] == "tier")
    assert tier["rates"] == [0.29, 0.37]
    assert tier["tiers"] == [{"max_kwh": 350.0, "rate": 0.29}, {"max_kwh": None, "rate": 0.37}]
    assert res["is_tou"] is False
    assert res["nem_regime"] == "nem2" and res["nem_source"] == "urdb_dgrules"

    com = await _create(client, auth_headers, get_preset("illustrative_commercial_tou_demand"))
    demand = [c for c in com["components"] if c["kind"] == "demand"]
    assert {d["rate"] for d in demand} == {12.5, 9.0}
    assert com["sector"] == "Commercial"


async def test_create_and_update_reject_unbillable_urdb(client, auth_headers):
    bad = {
        "energyratestructure": [[{"rate": "not-a-number"}]],
        "energyweekdayschedule": [[0] * 24] * 12,
        "energyweekendschedule": [[0] * 24] * 12,
    }
    resp = await client.post(
        "/api/v1/tariffs",
        json={"name": uid("bad"), "urdb_json": bad},
        headers=auth_headers,
    )
    assert resp.status_code == 422
    assert "Invalid URDB JSON" in resp.json()["detail"]

    t = await _create(client, auth_headers, FLAT_TARIFF_URDB)
    resp = await client.put(
        f"/api/v1/tariffs/{t['id']}", json={"urdb_json": bad}, headers=auth_headers
    )
    assert resp.status_code == 422
    resp = await client.put(
        f"/api/v1/tariffs/{t['id']}",
        json={"name": "Renamed", "urdb_json": {**FLAT_TARIFF_URDB, "nem": "nem2"}},
        headers=auth_headers,
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["name"] == "Renamed"
    assert resp.json()["nem_regime"] == "nem2" and resp.json()["nem_source"] == "tariff"


async def test_crud_is_admin_only(client, auth_headers, db_session):
    viewer = await user_headers(db_session, "viewer")
    t = await _create(client, auth_headers, FLAT_TARIFF_URDB)
    assert (await client.get(f"/api/v1/tariffs/{t['id']}", headers=viewer)).status_code == 200
    for method, url, body in [
        ("post", "/api/v1/tariffs", {"name": "x", "urdb_json": FLAT_TARIFF_URDB}),
        ("put", f"/api/v1/tariffs/{t['id']}", {"name": "x"}),
        ("delete", f"/api/v1/tariffs/{t['id']}", None),
    ]:
        kwargs = {"headers": viewer}
        if body is not None:
            kwargs["json"] = body
        resp = await getattr(client, method)(url, **kwargs)
        assert resp.status_code == 403, (method, resp.text)
    resp = await client.delete(f"/api/v1/tariffs/{t['id']}", headers=auth_headers)
    assert resp.status_code == 204
    assert (await client.get(f"/api/v1/tariffs/{t['id']}", headers=viewer)).status_code == 404


# ---------------------------------------------------------------------------
# Presets / URDB status
# ---------------------------------------------------------------------------


async def test_presets(client, auth_headers):
    resp = await client.get("/api/v1/tariffs/presets", headers=auth_headers)
    assert resp.status_code == 200
    ids = {p["id"]: p for p in resp.json()}
    assert {"pge_etouc", "sce_toudprime", "illustrative_residential_tiered"} <= set(ids)
    assert ids["illustrative_residential_tiered"]["illustrative"] is True
    assert ids["pge_etouc"]["illustrative"] is False

    one = await client.get("/api/v1/tariffs/presets/pge_etouc", headers=auth_headers)
    assert one.status_code == 200
    assert one.json()["urdb_json"]["name"] == "PG&E E-TOU-C"
    missing = await client.get("/api/v1/tariffs/presets/nope", headers=auth_headers)
    assert missing.status_code == 404


async def test_urdb_import_status_and_missing_key(client, auth_headers, monkeypatch):
    monkeypatch.delenv("OPENEI_API_KEY", raising=False)
    resp = await client.get("/api/v1/tariffs/import-urdb", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["configured"] is False
    assert "OPENEI_API_KEY" in resp.json()["detail"]
    resp = await client.post(
        "/api/v1/tariffs/import-urdb", json={"urdb_label": "abc"}, headers=auth_headers
    )
    assert resp.status_code == 503

    monkeypatch.setenv("OPENEI_API_KEY", "k")
    resp = await client.get("/api/v1/tariffs/import-urdb", headers=auth_headers)
    assert resp.json()["configured"] is True


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


async def _simulate(client, headers, tariff_id, **body):
    return await client.post(f"/api/v1/tariffs/{tariff_id}/simulate", json=body, headers=headers)


async def test_simulate_synthetic_is_deterministic_and_labelled(client, auth_headers):
    t = await _create(client, auth_headers, get_preset("pge_etouc"))
    body = {
        "synthetic": True,
        "period_days": 30,
        "billing_period_start": "2024-07-01T00:00:00",
        "timezone": "America/Los_Angeles",
    }
    r1 = await _simulate(client, auth_headers, t["id"], **body)
    r2 = await _simulate(client, auth_headers, t["id"], **body)
    assert r1.status_code == 200, r1.text
    a = r1.json()
    assert a["total"] == r2.json()["total"] > 0
    assert a["tariff_id"] == t["id"] and a["currency"] == "USD"
    summary = a["load_summary"]
    assert summary["source"] == "synthetic"
    assert "illustrative" in summary["method"] and "residential" in summary["method"]
    assert summary["intervals"] == 720
    # 0.8 kW average residential shape (+/- the day-to-day modulation).
    assert 500 < summary["import_kwh"] < 660
    assert parse_iso(a["period_start"]) == datetime(
        2024, 7, 1, tzinfo=ZoneInfo("America/Los_Angeles")
    )
    assert len(a["cycles"]) == 1
    assert {li["label"] for li in a["line_items"]} >= {"TOU period_0", "TOU period_1"}


async def test_simulate_synthetic_pv_with_tariff_nem(client, auth_headers):
    t = await _create(client, auth_headers, get_preset("illustrative_residential_tiered"))
    resp = await _simulate(
        client,
        auth_headers,
        t["id"],
        synthetic={"profile": "residential", "pv_kw": 6},
        billing_period_start="2024-06-01T00:00:00",
        timezone="America/Denver",
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["nem_regime"] == "nem2" and body["nem_source"] == "urdb_dgrules"
    assert body["load_summary"]["export_kwh"] > 0
    credit = [li for li in body["line_items"] if li["kind"] == "credit"]
    assert len(credit) == 1 and credit[0]["amount"] < 0
    assert body["export_credit"] == pytest.approx(-credit[0]["amount"], abs=1e-3)

    # An explicit request regime wins over the tariff's.
    resp = await _simulate(
        client,
        auth_headers,
        t["id"],
        synthetic={"profile": "residential", "pv_kw": 6},
        billing_period_start="2024-06-01T00:00:00",
        nem="none",
    )
    assert resp.json()["nem_source"] == "request"
    assert not [li for li in resp.json()["line_items"] if li["kind"] == "credit"]


async def test_simulate_compare_to(client, auth_headers):
    a = await _create(client, auth_headers, get_preset("pge_etouc"))
    b = await _create(client, auth_headers, get_preset("sce_toudprime"))
    resp = await _simulate(
        client,
        auth_headers,
        a["id"],
        synthetic=True,
        billing_period_start="2024-07-01T00:00:00",
        compare_to=b["id"],
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    cmp = body["comparison"]
    assert cmp["tariff_id"] == b["id"] and cmp["tariff_name"] == "SCE TOU-D-PRIME"
    assert cmp["total"] > 0 and cmp["total"] != body["total"]
    assert cmp["comparison"] is None

    resp = await _simulate(
        client, auth_headers, a["id"], synthetic=True, compare_to="does-not-exist"
    )
    assert resp.status_code == 404


async def test_simulate_csv_matches_meter_trace(client, auth_headers):
    t = await _create(client, auth_headers, get_preset("pge_etouc"))
    start = datetime(2024, 7, 1)
    rows = ["timestamp,kw"] + [
        f"{(start + timedelta(minutes=15 * i)).isoformat()},{2.0 if i % 96 >= 64 else 1.0}"
        for i in range(96 * 3)
    ]
    csv_resp = await _simulate(
        client, auth_headers, t["id"], csv="\n".join(rows), timezone="America/Los_Angeles"
    )
    assert csv_resp.status_code == 200, csv_resp.text
    csv_body = csv_resp.json()
    assert csv_body["load_summary"]["source"] == "csv"
    assert csv_body["load_summary"]["interval_minutes"] == 15
    # kW * 0.25 h -> kWh: 2 days x (64 x 0.25 + 32 x 0.5) = 32 kWh/day
    assert csv_body["load_summary"]["import_kwh"] == pytest.approx(96.0)
    assert csv_body["load_summary"]["peak_kw"] == pytest.approx(2.0)

    trace = {
        "timestamps": [(start + timedelta(minutes=15 * i)).isoformat() for i in range(96 * 3)],
        "import_kwh": [0.5 if i % 96 >= 64 else 0.25 for i in range(96 * 3)],
        "interval_minutes": 15,
    }
    mt = await _simulate(
        client,
        auth_headers,
        t["id"],
        meter_trace=trace,
        billing_period_start=start.isoformat(),
        billing_period_end=(start + timedelta(days=3)).isoformat(),
        timezone="America/Los_Angeles",
    )
    assert mt.status_code == 200, mt.text
    assert mt.json()["total"] == pytest.approx(csv_body["total"])


@pytest.mark.parametrize(
    ("csv", "fragment"),
    [
        ("", "empty"),
        ("when,kw\n2024-01-01T00:00,1", "timestamp column"),
        ("timestamp,volts\n2024-01-01T00:00,1", "energy column"),
        ("timestamp,kw\nyesterday,1\n", "cannot parse timestamp"),
        ("timestamp,kw\n2024-01-01T00:00,1\n2024-01-01T00:00,2", "duplicate"),
        ("timestamp,kw\n2024-01-01T00:00,1\n2024-01-01T00:07,1", "unsupported"),
    ],
)
async def test_simulate_csv_errors(client, auth_headers, csv, fragment):
    t = await _create(client, auth_headers, FLAT_TARIFF_URDB)
    resp = await _simulate(client, auth_headers, t["id"], csv=csv)
    assert resp.status_code == 400, resp.text
    assert fragment in resp.json()["detail"]


async def test_simulate_timezone_drives_tou(client, auth_headers):
    t = await _create(client, auth_headers, get_preset("pge_etouc"))
    body = {
        "meter_trace": {
            "timestamps": ["2024-07-01T17:00:00"],
            "import_kwh": [10.0],
            "interval_minutes": 60,
        },
        "billing_period_start": "2024-07-01T00:00:00",
        "billing_period_end": "2024-07-02T00:00:00",
    }
    utc = (await _simulate(client, auth_headers, t["id"], **body)).json()
    la = (
        await _simulate(client, auth_headers, t["id"], timezone="America/Los_Angeles", **body)
    ).json()
    # 17:00 is on-peak (period_1) in either zone when the timestamp is naive,
    # because naive times are read in the request timezone.
    assert [li["label"] for li in utc["line_items"] if li["kind"] == "energy"] == ["TOU period_1"]
    assert [li["label"] for li in la["line_items"] if li["kind"] == "energy"] == ["TOU period_1"]
    # An explicit UTC instant at 17:00Z is 10:00 in Los Angeles: off-peak.
    body["meter_trace"]["timestamps"] = ["2024-07-01T17:00:00+00:00"]
    la2 = (
        await _simulate(client, auth_headers, t["id"], timezone="America/Los_Angeles", **body)
    ).json()
    assert [li["label"] for li in la2["line_items"] if li["kind"] == "energy"] == ["TOU period_0"]


async def test_simulate_long_window_bills_monthly_cycles(client, auth_headers):
    t = await _create(client, auth_headers, get_preset("illustrative_commercial_tou_demand"))
    resp = await _simulate(
        client,
        auth_headers,
        t["id"],
        synthetic=True,
        period_days=91,
        billing_period_start="2024-01-01T00:00:00",
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert [c["period_start"][:10] for c in body["cycles"]] == [
        "2024-01-01",
        "2024-02-01",
        "2024-03-01",
    ]
    assert body["total"] == pytest.approx(sum(c["total"] for c in body["cycles"]), abs=1e-3)
    fixed = next(li for li in body["line_items"] if li["kind"] == "fixed")
    assert fixed["quantity"] == 3 and fixed["amount"] == pytest.approx(450.0)
    assert body["load_summary"]["source"] == "synthetic"
    assert "commercial" in body["load_summary"]["method"]  # from the tariff sector

    single = await _simulate(
        client,
        auth_headers,
        t["id"],
        synthetic=True,
        period_days=91,
        billing_period_start="2024-01-01T00:00:00",
        billing_cycle="single",
    )
    assert len(single.json()["cycles"]) == 1


async def test_simulate_nem3_from_tariff_config(client, auth_headers):
    urdb = {**get_preset("pge_etouc"), "nem": "nem3", "nem3_avoided_cost": [0.05] * 24}
    t = await _create(client, auth_headers, urdb)
    assert t["nem_regime"] == "nem3"
    resp = await _simulate(
        client,
        auth_headers,
        t["id"],
        meter_trace={
            "timestamps": ["2024-07-01T12:00:00Z", "2024-07-01T13:00:00Z"],
            "import_kwh": [0.0, 1.0],
            "export_kwh": [10.0, 0.0],
        },
        billing_period_start="2024-07-01T00:00:00Z",
        billing_period_end="2024-07-02T00:00:00Z",
    )
    assert resp.status_code == 200, resp.text
    credit = next(li for li in resp.json()["line_items"] if li["kind"] == "credit")
    assert credit["amount"] == pytest.approx(-0.5)

    # nem3 without any avoided-cost vector is a 400, not a silent zero.
    t2 = await _create(client, auth_headers, get_preset("pge_etouc"))
    resp = await _simulate(client, auth_headers, t2["id"], synthetic={"pv_kw": 5}, nem="nem3")
    assert resp.status_code == 400
    assert "avoided-cost" in resp.json()["detail"]


@pytest.mark.parametrize(
    "body",
    [
        {},  # no load source
        {"synthetic": True, "csv": "timestamp,kw"},  # two load sources
        {"synthetic": True, "nem": "nem9"},
        {"synthetic": True, "timezone": "Mars/Olympus"},
        {"meter_trace": {"timestamps": [], "import_kwh": []}},  # no billing window
    ],
)
async def test_simulate_validation(client, auth_headers, body):
    t = await _create(client, auth_headers, FLAT_TARIFF_URDB)
    resp = await _simulate(client, auth_headers, t["id"], **body)
    assert resp.status_code == 422, resp.text


async def test_inline_simulate_requires_a_tariff(client, auth_headers):
    resp = await client.post(
        "/api/v1/tariffs/simulate", json={"synthetic": True}, headers=auth_headers
    )
    assert resp.status_code == 422
    resp = await client.post(
        "/api/v1/tariffs/simulate",
        json={"synthetic": True, "urdb_json": FLAT_TARIFF_URDB},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    assert resp.json()["tariff_id"] is None


# ---------------------------------------------------------------------------
# Customer bill: NEM from the assigned tariff
# ---------------------------------------------------------------------------


async def test_customer_bill_credits_exports_under_tariff_nem(client, auth_headers):
    chi = ZoneInfo("America/Chicago")
    t = await _create(client, auth_headers, {**FLAT_TARIFF_URDB, "dgrules": "Net Metering"})
    cid, headers = await create_customer(client, auth_headers, tariff_id=t["id"])
    site = await create_site(client, auth_headers, owner_id=cid, timezone="America/Chicago")
    july = datetime(2026, 7, 1, tzinfo=chi)
    readings = [
        {
            "timestamp": (july + timedelta(hours=i)).isoformat(),
            "import_kwh": 1.0,
            "export_kwh": 0.5,
        }
        for i in range(48)
    ]
    resp = await client.post(
        f"/api/v1/sites/{site['id']}/meter-readings",
        json={"interval_minutes": 60, "readings": readings},
        headers=auth_headers,
    )
    assert resp.status_code == 200, resp.text

    resp = await client.get(
        "/api/v1/customer/me/bill", params={"month": "2026-07"}, headers=headers
    )
    assert resp.status_code == 200, resp.text
    bill = resp.json()["bill"]
    meta = bill["metadata"]
    assert meta["export_credit_applied"] is True
    assert meta["nem_regime"] == "nem2" and meta["nem_source"] == "urdb_dgrules"
    # NEM 2.0 on a flat tariff: 24 kWh exported x $0.20 retail.
    assert meta["export_credit"] == pytest.approx(4.8)
    credit = next(li for li in bill["line_items"] if li["kind"] == "credit")
    assert credit["amount"] == pytest.approx(-4.8)
    assert bill["total"] == pytest.approx(10.0 + 48 * 0.2 - 4.8)

    # Staff what-if override; customers have no such knob.
    resp = await client.get(
        f"/api/v1/customers/{cid}/bill",
        params={"month": "2026-07", "nem": "none"},
        headers=auth_headers,
    )
    assert resp.status_code == 200, resp.text
    meta = resp.json()["bill"]["metadata"]
    assert meta["export_credit_applied"] is False and meta["nem_source"] == "override"
    resp = await client.get(
        f"/api/v1/customers/{cid}/bill", params={"nem": "bogus"}, headers=auth_headers
    )
    assert resp.status_code == 422


async def test_customer_bill_nem3_without_avoided_cost_is_reported(client, auth_headers):
    t = await _create(client, auth_headers, {**FLAT_TARIFF_URDB, "nem": "nem3"})
    cid, headers = await create_customer(client, auth_headers, tariff_id=t["id"])
    site = await create_site(client, auth_headers, owner_id=cid, timezone="UTC")
    resp = await client.post(
        f"/api/v1/sites/{site['id']}/meter-readings",
        json={
            "interval_minutes": 60,
            "readings": [
                {"timestamp": "2026-07-01T12:00:00+00:00", "import_kwh": 0, "export_kwh": 3}
            ],
        },
        headers=auth_headers,
    )
    assert resp.status_code == 200, resp.text
    resp = await client.get(
        "/api/v1/customer/me/bill", params={"month": "2026-07"}, headers=headers
    )
    meta = resp.json()["bill"]["metadata"]
    assert meta["export_credit_applied"] is False
    assert meta["nem_regime"] == "nem3"
    assert "avoided-cost" in meta["export_credit_note"]
