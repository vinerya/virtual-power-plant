"""Tests for health and version endpoints (no auth required)."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health(client: AsyncClient):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_readiness(client: AsyncClient):
    resp = await client.get("/ready")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ready"


@pytest.mark.asyncio
async def test_version(client: AsyncClient):
    resp = await client.get("/version")
    assert resp.status_code == 200
    body = resp.json()
    assert "version" in body
    assert body["api_version"] == "v1"


@pytest.mark.asyncio
async def test_version_single_source_of_truth(client: AsyncClient):
    """/version, the OpenAPI schema and pyproject.toml all agree."""
    import vpp
    from vpp._version import _version_from_pyproject

    pyproject_version = _version_from_pyproject()
    assert pyproject_version is not None
    assert vpp.__version__ == pyproject_version

    assert (await client.get("/version")).json()["version"] == vpp.__version__
    openapi = (await client.get("/openapi.json")).json()
    assert openapi["info"]["version"] == vpp.__version__


def test_version_falls_back_to_pyproject(monkeypatch):
    """Without installed metadata, the version is read from pyproject.toml."""
    from importlib.metadata import PackageNotFoundError

    from vpp import _version

    def _missing(name: str) -> str:
        raise PackageNotFoundError(name)

    monkeypatch.setattr(_version, "version", _missing)
    assert _version.get_version() == _version._version_from_pyproject()


def test_version_unknown_when_no_source(monkeypatch, tmp_path):
    from importlib.metadata import PackageNotFoundError

    from vpp import _version

    def _missing(name: str) -> str:
        raise PackageNotFoundError(name)

    monkeypatch.setattr(_version, "version", _missing)
    monkeypatch.setattr(_version, "_PYPROJECT", tmp_path / "missing.toml")
    assert _version.get_version() == _version.UNKNOWN_VERSION


@pytest.mark.asyncio
async def test_readiness_reports_503_when_database_is_unreachable(
    client: AsyncClient, monkeypatch
):
    from vpp.api.routes import health as health_routes

    def _broken_factory():
        raise RuntimeError("Database not initialised")

    monkeypatch.setattr(health_routes, "get_session_factory", _broken_factory)
    resp = await client.get("/ready")
    assert resp.status_code == 503
    assert resp.json()["subsystems"]["database"] is False


def test_production_refuses_the_default_or_short_secret_key():
    from pydantic import ValidationError

    from vpp.settings import Settings

    with pytest.raises(ValidationError, match="VPP_SECRET_KEY"):
        Settings(env="production", secret_key="change-me-to-a-real-secret-key")
    with pytest.raises(ValidationError, match="VPP_SECRET_KEY"):
        Settings(env="production", secret_key="too-short")
    assert Settings(env="production", secret_key="x" * 48).is_production
    # Development keeps working out of the box.
    assert Settings(env="development").secret_key
