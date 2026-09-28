"""OCPP-J (OCPP over JSON/WebSocket) RPC framing.

Implements the transport-independent half of OCPP 1.6-J section 4:

* ``[2, "<uniqueId>", "<Action>", {payload}]``   -- CALL
* ``[3, "<uniqueId>", {payload}]``               -- CALLRESULT
* ``[4, "<uniqueId>", "<errorCode>", "<errorDescription>", {errorDetails}]`` -- CALLERROR

:class:`OCPPJSession` wraps one charge point's WebSocket (abstracted as an
async ``send_text`` callable, so FastAPI, ``websockets`` or a test harness
can all drive it). It correlates server-initiated CALLs with their
CALLRESULT/CALLERROR by unique id, enforces a per-call timeout, enforces
OCPP's "at most one outstanding CALL per direction" rule, and turns
exceptions raised by inbound-CALL handlers into well-formed CALLERRORs.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any

logger = logging.getLogger(__name__)

OCPP16_SUBPROTOCOL = "ocpp1.6"


class MessageTypeId(IntEnum):
    CALL = 2
    CALLRESULT = 3
    CALLERROR = 4


class OCPPErrorCode(str, Enum):
    """OCPP-J 1.6 CALLERROR error codes (spelling per the spec, incl. "Occurence")."""

    NOT_IMPLEMENTED = "NotImplemented"
    NOT_SUPPORTED = "NotSupported"
    INTERNAL_ERROR = "InternalError"
    PROTOCOL_ERROR = "ProtocolError"
    SECURITY_ERROR = "SecurityError"
    FORMATION_VIOLATION = "FormationViolation"
    PROPERTY_CONSTRAINT_VIOLATION = "PropertyConstraintViolation"
    OCCURENCE_CONSTRAINT_VIOLATION = "OccurenceConstraintViolation"
    TYPE_CONSTRAINT_VIOLATION = "TypeConstraintViolation"
    GENERIC_ERROR = "GenericError"


class OCPPError(Exception):
    """Raised by inbound-CALL handlers; serialised back as a CALLERROR."""

    def __init__(
        self,
        code: OCPPErrorCode,
        description: str = "",
        details: dict[str, Any] | None = None,
        *,
        unique_id: str | None = None,
    ) -> None:
        super().__init__(f"{code.value}: {description}")
        self.code = code
        self.description = description
        self.details = details or {}
        self.unique_id = unique_id


class OCPPCallError(Exception):
    """The charge point answered one of our CALLs with a CALLERROR."""

    def __init__(self, action: str, code: str, description: str, details: dict[str, Any]) -> None:
        super().__init__(f"{action} rejected with {code}: {description}")
        self.action = action
        self.code = code
        self.description = description
        self.details = details


class OCPPTimeoutError(TimeoutError):
    """The charge point did not answer one of our CALLs in time."""


@dataclass(frozen=True)
class Call:
    unique_id: str
    action: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class CallResult:
    unique_id: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class CallErrorFrame:
    unique_id: str
    code: str
    description: str
    details: dict[str, Any] = field(default_factory=dict)


Frame = Call | CallResult | CallErrorFrame

MAX_UNIQUE_ID_LENGTH = 36


def parse_frame(text: str | bytes) -> Frame:
    """Parse one OCPP-J frame, raising :class:`OCPPError` if malformed.

    When the unique id could be recovered, it is attached to the raised
    error so the caller can still answer with a CALLERROR.
    """
    try:
        data = json.loads(text)
    except (TypeError, ValueError) as exc:
        raise OCPPError(OCPPErrorCode.FORMATION_VIOLATION, f"Invalid JSON: {exc}") from exc

    if not isinstance(data, list) or len(data) < 3:
        raise OCPPError(OCPPErrorCode.FORMATION_VIOLATION, "Frame must be a JSON array")

    type_id, unique_id = data[0], data[1]
    if not isinstance(unique_id, str) or not unique_id or len(unique_id) > MAX_UNIQUE_ID_LENGTH:
        raise OCPPError(OCPPErrorCode.FORMATION_VIOLATION, "Invalid uniqueId")

    if type_id == MessageTypeId.CALL:
        if len(data) != 4 or not isinstance(data[2], str):
            raise OCPPError(
                OCPPErrorCode.FORMATION_VIOLATION,
                "CALL must be [2, id, action, payload]",
                unique_id=unique_id,
            )
        if not isinstance(data[3], dict):
            raise OCPPError(
                OCPPErrorCode.FORMATION_VIOLATION,
                "CALL payload must be an object",
                unique_id=unique_id,
            )
        return Call(unique_id, data[2], data[3])
    if type_id == MessageTypeId.CALLRESULT:
        if len(data) != 3 or not isinstance(data[2], dict):
            raise OCPPError(OCPPErrorCode.FORMATION_VIOLATION, "Malformed CALLRESULT")
        return CallResult(unique_id, data[2])
    if type_id == MessageTypeId.CALLERROR:
        if len(data) != 5 or not isinstance(data[2], str) or not isinstance(data[3], str):
            raise OCPPError(OCPPErrorCode.FORMATION_VIOLATION, "Malformed CALLERROR")
        details = data[4] if isinstance(data[4], dict) else {}
        return CallErrorFrame(unique_id, data[2], data[3], details)
    raise OCPPError(
        OCPPErrorCode.PROTOCOL_ERROR, f"Unknown MessageTypeId {type_id!r}", unique_id=unique_id
    )


def encode_call(unique_id: str, action: str, payload: dict[str, Any]) -> str:
    return json.dumps([int(MessageTypeId.CALL), unique_id, action, payload], separators=(",", ":"))


def encode_call_result(unique_id: str, payload: dict[str, Any]) -> str:
    return json.dumps([int(MessageTypeId.CALLRESULT), unique_id, payload], separators=(",", ":"))


def encode_call_error(
    unique_id: str,
    code: OCPPErrorCode | str,
    description: str = "",
    details: dict[str, Any] | None = None,
) -> str:
    code_str = code.value if isinstance(code, OCPPErrorCode) else code
    return json.dumps(
        [int(MessageTypeId.CALLERROR), unique_id, code_str, description, details or {}],
        separators=(",", ":"),
    )


SendText = Callable[[str], Awaitable[None]]
# (charge_point_id, action, payload) -> response payload
CallHandler = Callable[[str, str, dict[str, Any]], Awaitable[dict[str, Any]]]


class OCPPJSession:
    """One charge point's OCPP-J RPC session."""

    def __init__(
        self,
        charge_point_id: str,
        send_text: SendText,
        handler: CallHandler,
        *,
        default_timeout: float = 30.0,
    ) -> None:
        self.charge_point_id = charge_point_id
        self._send_text = send_text
        self._handler = handler
        self._default_timeout = default_timeout
        self._pending: dict[str, tuple[str, asyncio.Future[dict[str, Any]]]] = {}
        self._call_lock = asyncio.Lock()
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    async def call(
        self, action: str, payload: dict[str, Any], *, timeout: float | None = None
    ) -> dict[str, Any]:
        """Send a CALL and wait for its CALLRESULT payload.

        Raises :class:`OCPPCallError` on CALLERROR, :class:`OCPPTimeoutError`
        on timeout and :class:`ConnectionError` if the session is closed.
        """
        if self._closed:
            raise ConnectionError(f"Charge point {self.charge_point_id} is not connected")
        wait = self._default_timeout if timeout is None else timeout
        # OCPP-J: a sender must not send a new CALL until the previous one
        # was answered (or timed out).
        async with self._call_lock:
            unique_id = str(uuid.uuid4())
            future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
            self._pending[unique_id] = (action, future)
            try:
                await self._send_text(encode_call(unique_id, action, payload))
                return await asyncio.wait_for(future, timeout=wait)
            except asyncio.TimeoutError as exc:
                raise OCPPTimeoutError(
                    f"{action} to {self.charge_point_id} timed out after {wait:.1f}s"
                ) from exc
            finally:
                self._pending.pop(unique_id, None)

    async def handle_text(self, text: str | bytes) -> None:
        """Process one inbound WebSocket text frame."""
        try:
            frame = parse_frame(text)
        except OCPPError as exc:
            logger.warning("Malformed OCPP frame from %s: %s", self.charge_point_id, exc)
            if exc.unique_id is not None:
                await self._send_text(
                    encode_call_error(exc.unique_id, exc.code, exc.description, exc.details)
                )
            return

        if isinstance(frame, Call):
            await self._handle_call(frame)
        elif isinstance(frame, CallResult):
            self._resolve(frame.unique_id, result=frame.payload)
        else:
            self._resolve(frame.unique_id, error=frame)

    async def _handle_call(self, call: Call) -> None:
        try:
            response = await self._handler(self.charge_point_id, call.action, call.payload)
        except OCPPError as exc:
            reply = encode_call_error(call.unique_id, exc.code, exc.description, exc.details)
        except Exception:
            logger.exception(
                "OCPP handler for %s from %s crashed", call.action, self.charge_point_id
            )
            reply = encode_call_error(
                call.unique_id, OCPPErrorCode.INTERNAL_ERROR, "Internal error"
            )
        else:
            reply = encode_call_result(call.unique_id, response)
        await self._send_text(reply)

    def _resolve(
        self,
        unique_id: str,
        *,
        result: dict[str, Any] | None = None,
        error: CallErrorFrame | None = None,
    ) -> None:
        entry = self._pending.get(unique_id)
        if entry is None:
            logger.warning(
                "Dropping OCPP response with unknown uniqueId %s from %s",
                unique_id,
                self.charge_point_id,
            )
            return
        action, future = entry
        if future.done():
            return
        if error is not None:
            future.set_exception(
                OCPPCallError(action, error.code, error.description, error.details)
            )
        else:
            future.set_result(result or {})

    def close(self) -> None:
        """Fail every outstanding CALL; further calls raise ConnectionError."""
        self._closed = True
        for action, future in list(self._pending.values()):
            if not future.done():
                future.set_exception(
                    ConnectionError(
                        f"Charge point {self.charge_point_id} disconnected during {action}"
                    )
                )
        self._pending.clear()
