"""Webhook events on tariff PUT/DELETE (M4)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from httpx import AsyncClient

from vpp.events import EventType, get_event_bus, reset_event_bus


PRESET = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "vpp"
    / "tariffs"
    / "presets"
    / "pge_etouc.json"
)


def _preset() -> dict:
    with open(PRESET) as f:
        return json.load(f)


@pytest.mark.asyncio
async def test_put_tariff_emits_event(client: AsyncClient, auth_headers: dict):
    reset_event_bus()
    bus = get_event_bus()
    received = []

    async def cb(ev):
        received.append(ev)

    bus.subscribe(cb, event_types={EventType.TARIFF_UPDATED})

    create = await client.post(
        "/api/v1/tariffs",
        json={"name": "wh-update", "utility": "X", "urdb_json": _preset()},
        headers=auth_headers,
    )
    assert create.status_code == 201
    tid = create.json()["id"]

    update = await client.put(
        f"/api/v1/tariffs/{tid}",
        json={"name": "wh-update-renamed"},
        headers=auth_headers,
    )
    assert update.status_code == 200

    assert len(received) == 1
    assert received[0].data["tariff_id"] == tid
    assert "name" in received[0].data["fields"]


@pytest.mark.asyncio
async def test_delete_tariff_emits_event(client: AsyncClient, auth_headers: dict):
    reset_event_bus()
    bus = get_event_bus()
    received = []

    async def cb(ev):
        received.append(ev)

    bus.subscribe(cb, event_types={EventType.TARIFF_DELETED})

    create = await client.post(
        "/api/v1/tariffs",
        json={"name": "wh-del", "utility": "X", "urdb_json": _preset()},
        headers=auth_headers,
    )
    tid = create.json()["id"]

    resp = await client.delete(f"/api/v1/tariffs/{tid}", headers=auth_headers)
    assert resp.status_code == 204

    assert len(received) == 1
    assert received[0].data["tariff_id"] == tid
