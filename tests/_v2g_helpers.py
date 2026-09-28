"""Shared helpers for the V2G / OCPP bridge / DR orchestrator tests.

``tmp_session_factory`` builds an isolated SQLite database with a
``NullPool`` async engine: every session opens a fresh aiosqlite connection
in the *current* event loop, so the same factory works from pytest-asyncio
tests and from inside a Starlette ``TestClient`` portal loop.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from vpp.db import models as _models  # noqa: F401  (register tables)
from vpp.db.base import Base

if TYPE_CHECKING:
    from pathlib import Path

SUBPROTOCOL = ["ocpp1.6"]


def tmp_session_factory(path: Path) -> async_sessionmaker:
    sync = create_engine(f"sqlite:///{path}")
    Base.metadata.create_all(sync)
    sync.dispose()
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}", poolclass=NullPool)
    return async_sessionmaker(engine, expire_on_commit=False)


def fake_user(role: str = "admin") -> SimpleNamespace:
    return SimpleNamespace(id="user-1", username=f"test-{role}", role=role, is_active=True)


def build_app(factory: async_sessionmaker, registry: Any, *, role: str = "admin") -> FastAPI:
    """A minimal app with the V2G, OCPP and protocol-ops routers on a tmp DB."""
    from vpp.api.routes import ocpp as ocpp_routes
    from vpp.api.routes import protocol_ops, v2g
    from vpp.api.routes.protocols import get_registry
    from vpp.auth.security import get_current_principal
    from vpp.db.engine import get_db

    app = FastAPI()
    app.include_router(v2g.router)
    app.include_router(ocpp_routes.router)
    app.include_router(protocol_ops.router)
    app.include_router(protocol_ops.dr_router)

    async def _db():
        async with factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    app.dependency_overrides[get_db] = _db
    app.dependency_overrides[get_registry] = lambda: registry
    app.dependency_overrides[get_current_principal] = lambda: fake_user(role)
    return app


def live_ocpp_adapter(**config):
    """A started live OCPP Central System adapter with the vehicle bridge attached."""
    from vpp.protocols.ocpp import OCPPAdapter

    adapter = OCPPAdapter()
    adapter.configure(central_system_enabled=True, call_timeout_s=5, **config)
    asyncio.run(adapter.connect())
    return adapter


class ChargePointSim:
    """Minimal OCPP 1.6-J charge point over a TestClient websocket."""

    def __init__(self, ws) -> None:
        self.ws = ws
        self._n = 0

    def call(self, action: str, payload: dict) -> list:
        self._n += 1
        uid = f"cp-{self._n}"
        self.ws.send_text(json.dumps([2, uid, action, payload]))
        reply = json.loads(self.ws.receive_text())
        assert reply[1] == uid, reply
        return reply

    def expect_call(self, action: str) -> tuple[str, dict]:
        frame = json.loads(self.ws.receive_text())
        assert frame[0] == 2, frame
        assert frame[2] == action, frame
        return frame[1], frame[3]

    def reply(self, uid: str, payload: dict) -> None:
        self.ws.send_text(json.dumps([3, uid, payload]))

    def boot(self) -> None:
        self.call("BootNotification", {"chargePointVendor": "V", "chargePointModel": "M"})

    def status(self, connector_id: int, status: str) -> None:
        self.call(
            "StatusNotification",
            {"connectorId": connector_id, "errorCode": "NoError", "status": status},
        )

    def start(self, connector_id: int, id_tag: str, meter_start: int = 0) -> int:
        reply = self.call(
            "StartTransaction",
            {
                "connectorId": connector_id,
                "idTag": id_tag,
                "meterStart": meter_start,
                "timestamp": "2026-09-28T10:00:00Z",
            },
        )
        return reply[2]["transactionId"]

    def meter(self, connector_id: int, tx_id: int | None, *, power_w: float, soc: float) -> None:
        payload: dict[str, Any] = {
            "connectorId": connector_id,
            "meterValue": [
                {
                    "timestamp": "2026-09-28T10:05:00Z",
                    "sampledValue": [
                        {"value": str(power_w), "measurand": "Power.Active.Import", "unit": "W"},
                        {"value": str(soc), "measurand": "SoC", "unit": "Percent"},
                    ],
                }
            ],
        }
        if tx_id is not None:
            payload["transactionId"] = tx_id
        self.call("MeterValues", payload)

    def stop(self, tx_id: int, meter_stop: int) -> None:
        self.call(
            "StopTransaction",
            {"transactionId": tx_id, "meterStop": meter_stop, "timestamp": "2026-09-28T11:00:00Z"},
        )
