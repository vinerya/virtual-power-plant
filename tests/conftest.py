"""Shared pytest fixtures for the VPP test suite."""

from __future__ import annotations

import asyncio
import os
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

# Force testing settings before any VPP imports
os.environ["VPP_ENV"] = "testing"
os.environ["VPP_DATABASE_URL"] = "sqlite+aiosqlite:///./test_vpp.db"
os.environ["VPP_SECRET_KEY"] = "test-secret-key-not-for-production"

from vpp.api.app import create_app
from vpp.auth.security import create_access_token, get_password_hash
from vpp.db.engine import init_db, close_db, get_db
from vpp.db.repositories import UserRepository
from vpp.resources import Battery, Solar, WindTurbine
from vpp.settings import Settings


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def app():
    """Create the FastAPI app with a test database."""
    _app = create_app()
    # Initialise test database
    await init_db("sqlite+aiosqlite:///./test_vpp.db")
    yield _app
    await close_db()
    # Clean up test database
    try:
        os.remove("test_vpp.db")
    except FileNotFoundError:
        pass


@pytest_asyncio.fixture
async def client(app) -> AsyncGenerator[AsyncClient, None]:
    """HTTP test client using httpx."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest_asyncio.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Provide a database session for direct repository testing."""
    async for session in get_db():
        yield session


@pytest_asyncio.fixture
async def admin_user(db_session: AsyncSession):
    """Create an admin user for auth tests."""
    user = await UserRepository.get_by_username(db_session, "testadmin")
    if user is None:
        user = await UserRepository.create_user(
            db_session,
            username="testadmin",
            hashed_password=get_password_hash("adminpassword123"),
            role="admin",
        )
        await db_session.commit()
    return user


@pytest_asyncio.fixture
async def auth_headers(admin_user) -> dict[str, str]:
    """JWT auth headers for an admin user."""
    token = create_access_token({
        "sub": admin_user.id,
        "username": admin_user.username,
        "role": admin_user.role,
    })
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# Resource fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def battery() -> Battery:
    return Battery(capacity=100.0, current_charge=50.0, max_power=50.0, nominal_voltage=48.0)


@pytest.fixture
def solar() -> Solar:
    return Solar(peak_power=10.0, panel_area=20.0, efficiency=0.2)


@pytest.fixture
def wind_turbine() -> WindTurbine:
    return WindTurbine(
        rated_power=100.0,
        rotor_diameter=20.0,
        hub_height=30.0,
        cut_in_speed=3.0,
        cut_out_speed=25.0,
        rated_speed=12.0,
    )
