"""Tests for /api/v1/config (document, schema, apply, validate)."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import pytest
import pytest_asyncio
import yaml
from _portal_helpers import isolated_client, user_headers

from vpp.api.deps import get_vpp, reset_vpp
from vpp.config import VPPConfig, vpp_config
from vpp.config import base as config_base
from vpp.config.schema import VPPConfigDocument, vpp_config_json_schema

if TYPE_CHECKING:
    from httpx import AsyncClient


@pytest_asyncio.fixture
async def client(app):
    """Per-test client IP so these API-heavy tests don't drain the shared rate limit."""
    async with isolated_client(app) as c:
        yield c


@pytest.fixture(autouse=True)
def _fresh_vpp():
    reset_vpp()
    yield
    reset_vpp()


def _doc(**overrides) -> str:
    data = VPPConfig().to_dict()
    for dotted, value in overrides.items():
        target = data
        *parents, leaf = dotted.split(".")
        for p in parents:
            target = target[p]
        target[leaf] = value
    return yaml.safe_dump(data, sort_keys=False)


# ---------------------------------------------------------------------------
# Schema <-> dataclass sync (guards drift)
# ---------------------------------------------------------------------------

_PAIRS = [
    (VPPConfig, VPPConfigDocument),
    (config_base.OptimizationConfig, "OptimizationDocument"),
    (config_base.OptimizationObjective, "ObjectiveDocument"),
    (config_base.ConstraintConfig, "ConstraintDocument"),
    (config_base.HeuristicConfig, "HeuristicDocument"),
    (config_base.RuleEngineConfig, "RuleEngineDocument"),
    (config_base.RuleConfig, "RuleDocument"),
    (vpp_config.MonitoringConfig, "MonitoringDocument"),
    (vpp_config.SimulationConfig, "SimulationDocument"),
    (vpp_config.SecurityConfig, "SecurityDocument"),
    (vpp_config.ResourceConfig, "ResourceDocument"),
]


@pytest.mark.parametrize("dc, model", _PAIRS)
def test_schema_mirrors_dataclass_fields(dc, model):
    from vpp.config import schema as schema_mod

    model_cls = getattr(schema_mod, model) if isinstance(model, str) else model
    assert {f.name for f in dataclasses.fields(dc)} == set(model_cls.model_fields)


def test_schema_mirrors_to_dict_output():
    # Every key VPPConfig serialises is accepted by the strict schema.
    VPPConfigDocument.model_validate(VPPConfig().to_dict())


def test_json_schema_is_draft07_compatible():
    schema = vpp_config_json_schema()
    assert "$schema" not in schema
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert "optimization" in schema["properties"]


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_config_document_shape(client: AsyncClient, auth_headers: dict):
    for path in ("/api/v1/config", "/api/v1/config/"):
        resp = await client.get(path, headers=auth_headers)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert isinstance(body["yaml"], str) and body["yaml"]
        assert len(body["hash"]) == 64
        assert "version" in body and "updated_at" in body
        assert yaml.safe_load(body["yaml"])["name"]
        # Pre-existing read-only settings are still reported.
        assert body["database_backend"] in ("sqlite", "postgresql")


@pytest.mark.asyncio
async def test_get_schema(client: AsyncClient, auth_headers: dict, db_session):
    viewer = await user_headers(db_session, "viewer")
    resp = await client.get("/api/v1/config/schema", headers=viewer)
    assert resp.status_code == 200
    assert resp.json()["properties"]["monitoring"]
    assert (await client.get("/api/v1/config/schema")).status_code == 401


@pytest.mark.asyncio
async def test_apply_config_round_trip(client: AsyncClient, auth_headers: dict):
    before = (await client.get("/api/v1/config", headers=auth_headers)).json()
    text = "# operator comment survives\n" + _doc(
        name="Austin VPP", **{"optimization.time_horizon": 48}
    )
    resp = await client.put("/api/v1/config", json={"yaml": text}, headers=auth_headers)
    assert resp.status_code == 200, resp.text
    applied = resp.json()
    assert applied["yaml"] == text
    assert applied["version"] == before["version"] + 1
    assert applied["updated_at"] is not None
    assert (
        applied["updated_by"]
        == (await client.get("/api/v1/auth/me", headers=auth_headers)).json()["id"]
    )
    assert get_vpp().config.name == "Austin VPP"
    assert get_vpp().config.optimization.time_horizon == 48

    live = (await client.get("/api/v1/config", headers=auth_headers)).json()
    assert live["yaml"] == text and live["hash"] == applied["hash"]

    # Re-applying the same document is idempotent (no new version).
    same = (await client.put("/api/v1/config", json={"yaml": text}, headers=auth_headers)).json()
    assert same["version"] == applied["version"]

    # Optimistic concurrency.
    stale = await client.put(
        "/api/v1/config",
        json={"yaml": _doc(name="Other"), "base_hash": before["hash"]},
        headers=auth_headers,
    )
    assert stale.status_code == 409
    fresh = await client.put(
        "/api/v1/config",
        json={"yaml": _doc(name="Other"), "base_hash": live["hash"]},
        headers=auth_headers,
    )
    assert fresh.status_code == 200
    assert fresh.json()["version"] == applied["version"] + 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text, path",
    [
        ("name: [unclosed", "$"),
        ("- just\n- a list\n", "$"),
        ("name: x\nbogus_key: 1\n", "/bogus_key"),
        ("optimization:\n  time_horizn: 48\n", "/optimization/time_horizn"),
        ("optimization:\n  time_horizon: 0\n", "/optimization/time_horizon"),
        ("monitoring:\n  log_level: LOUD\n", "/monitoring/log_level"),
        ("monitoring:\n  log_file: /etc/passwd\n", "/monitoring/log_file"),
        ("rules:\n  inference_method: guessing\n", "/rules/inference_method"),
        ("resources:\n  - {name: a, type: battery}\n  - {name: a, type: solar}\n", "$"),
    ],
)
async def test_apply_config_rejects_invalid(client: AsyncClient, auth_headers: dict, text, path):
    before = (await client.get("/api/v1/config", headers=auth_headers)).json()
    resp = await client.put("/api/v1/config", json={"yaml": text}, headers=auth_headers)
    assert resp.status_code == 422, resp.text
    detail = resp.json()["detail"]
    assert detail["errors"], detail
    assert path in {e["path"] for e in detail["errors"]}, detail
    after = (await client.get("/api/v1/config", headers=auth_headers)).json()
    assert after["version"] == before["version"]


@pytest.mark.asyncio
async def test_apply_config_admin_only(client: AsyncClient, db_session):
    operator = await user_headers(db_session, "operator")
    resp = await client.put("/api/v1/config", json={"yaml": _doc()}, headers=operator)
    assert resp.status_code == 403
    assert (await client.put("/api/v1/config", json={"yaml": _doc()})).status_code == 401


@pytest.mark.asyncio
async def test_apply_config_size_limit(client: AsyncClient, auth_headers: dict):
    huge = "description: '" + "x" * (300 * 1024) + "'\n"
    resp = await client.put("/api/v1/config", json={"yaml": huge}, headers=auth_headers)
    assert resp.status_code == 413


@pytest.mark.asyncio
async def test_validate_endpoint_uses_same_rules(client: AsyncClient, auth_headers: dict):
    ok = await client.post(
        "/api/v1/config/validate",
        json={"optimization": {"time_horizon": 200}},
        headers=auth_headers,
    )
    assert ok.json()["valid"] is True
    assert "time_horizon > 168h may be slow" in ok.json()["warnings"]
    bad = await client.post(
        "/api/v1/config/validate", json={"optimization": {"time_step": -1}}, headers=auth_headers
    )
    assert bad.json()["valid"] is False
    assert any("/optimization/time_step" in e for e in bad.json()["errors"])
