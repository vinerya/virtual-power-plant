"""OCPP 1.6-J Central System WebSocket endpoint.

Charge points connect to ``ws(s)://<host>/ocpp/{charge_point_id}`` offering
the ``ocpp1.6`` subprotocol. The route is a thin transport shim: framing,
correlation and action handling live in :mod:`vpp.protocols.ocpp_j` and
:class:`vpp.protocols.ocpp.OCPPAdapter`.

Connections are refused (HTTP 403 during the handshake) when:

* no live OCPP adapter is registered/started (``VPP_OCPP_ENABLED=false``,
  the default -- there is no Central System to talk to),
* the client does not offer the ``ocpp1.6`` subprotocol,
* the charge point id is not allow-listed or fails Basic auth.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from vpp.api.routes.protocols import get_registry
from vpp.protocols.base import ProtocolRegistry
from vpp.protocols.ocpp import OCPPAdapter
from vpp.protocols.ocpp_j import OCPP16_SUBPROTOCOL

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ocpp"])

# OCPP frames are small; anything this large is abuse or a broken client.
MAX_FRAME_CHARS = 65536


@router.websocket("/ocpp/{charge_point_id}")
async def ocpp_central_system(
    websocket: WebSocket,
    charge_point_id: str,
    registry: ProtocolRegistry = Depends(get_registry),
) -> None:
    adapter = registry.get("ocpp")
    if not isinstance(adapter, OCPPAdapter) or not adapter.is_connected:
        logger.warning(
            "Refusing OCPP connection from %s: Central System not running", charge_point_id
        )
        await websocket.close(code=1013)
        return

    if OCPP16_SUBPROTOCOL not in websocket.scope.get("subprotocols", []):
        logger.warning(
            "Refusing OCPP connection from %s: subprotocol %s not offered",
            charge_point_id,
            OCPP16_SUBPROTOCOL,
        )
        await websocket.close(code=1002)
        return

    if not adapter.authenticate(charge_point_id, websocket.headers.get("authorization")):
        logger.warning("Refusing OCPP connection from %s: authentication failed", charge_point_id)
        await websocket.close(code=1008)
        return

    async def _close() -> None:
        await websocket.close(code=1000)

    # Attach before accepting so the session is registered (and any stale
    # session for this id closed) by the time the handshake completes.
    session = await adapter.attach_session(charge_point_id, websocket.send_text, _close)
    try:
        await websocket.accept(subprotocol=OCPP16_SUBPROTOCOL)
        while True:
            text = await websocket.receive_text()
            if len(text) > MAX_FRAME_CHARS:
                logger.warning(
                    "OCPP frame from %s too large (%d chars); closing", charge_point_id, len(text)
                )
                await websocket.close(code=1009)
                break
            await session.handle_text(text)
    except WebSocketDisconnect:
        pass
    except RuntimeError:
        # Socket was closed from our side (session replaced / shutdown).
        pass
    finally:
        await adapter.detach_session(charge_point_id, session)
