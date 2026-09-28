"""OCPP 1.6-J Central System adapter for EV charger management.

Two modes (see :class:`~vpp.protocols.base.ProtocolMode`):

* **Live** (``central_system_enabled=True``): the adapter is a real OCPP
  1.6-J Central System. Charge points open a WebSocket to
  ``/ocpp/{charge_point_id}`` (subprotocol ``ocpp1.6``, served by
  :mod:`vpp.api.routes.ocpp`) and the adapter handles BootNotification,
  Heartbeat, StatusNotification, Authorize, StartTransaction,
  StopTransaction and MeterValues, and sends RemoteStartTransaction,
  RemoteStopTransaction, SetChargingProfile and ClearChargingProfile over
  the wire, correlating responses by unique id with timeouts. Status is
  ``CONNECTED`` (accepting charge points).
* **Simulated** (default, no Central System configured): charge points live
  in an in-memory registry and remote operations mutate that registry
  directly. Status is ``SIMULATED`` -- never ``CONNECTED`` -- so the API
  and dashboards can't mistake it for real hardware control.

TLS: terminate ``wss://`` at the reverse proxy / ingress in front of the
API. OCPP 1.6 Security Profile 1 (HTTP Basic auth, username = charge point
id) is supported via the ``basic_auth_password`` configuration key.

V2G note: OCPP 1.6 has no standard representation of discharge. Negative
``limit`` values in charging profiles (as produced by
:meth:`OCPPAdapter.set_v2g_discharge_profile`) are a widespread vendor
extension; chargers that don't support it will answer ``Rejected``, which
is reported back to the caller rather than hidden.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import hmac
import logging
import time
import uuid
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, ClassVar

from vpp.protocols.base import (
    ProtocolAdapter,
    ProtocolMessage,
    ProtocolMode,
    ProtocolStatus,
)
from vpp.protocols.ocpp_j import (
    OCPPCallError,
    OCPPError,
    OCPPErrorCode,
    OCPPJSession,
    OCPPTimeoutError,
    SendText,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OCPP value objects
# ---------------------------------------------------------------------------


class ChargePointStatus(str, Enum):
    AVAILABLE = "Available"
    PREPARING = "Preparing"
    CHARGING = "Charging"
    SUSPENDED_EVSE = "SuspendedEVSE"
    SUSPENDED_EV = "SuspendedEV"
    FINISHING = "Finishing"
    RESERVED = "Reserved"
    UNAVAILABLE = "Unavailable"
    FAULTED = "Faulted"


class ChargingProfilePurpose(str, Enum):
    CHARGE_POINT_MAX = "ChargePointMaxProfile"
    TX_DEFAULT = "TxDefaultProfile"
    TX_PROFILE = "TxProfile"


class ChargingRateUnit(str, Enum):
    WATTS = "W"
    AMPS = "A"


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class ChargingSchedulePeriod:
    """One period within a charging profile."""

    start_period: int  # seconds from profile start
    limit: float  # power in W or current in A
    number_phases: int = 3


@dataclass
class ChargingProfile:
    """An OCPP charging profile that controls charger output."""

    profile_id: int
    stack_level: int = 0
    purpose: ChargingProfilePurpose = ChargingProfilePurpose.TX_PROFILE
    kind: str = "Absolute"  # Absolute | Recurring | Relative
    rate_unit: ChargingRateUnit = ChargingRateUnit.WATTS
    schedule: list[ChargingSchedulePeriod] = field(default_factory=list)
    valid_from: float | None = None
    valid_to: float | None = None
    start_schedule: float | None = None  # Unix ts; required by OCPP for kind=Absolute
    duration_s: int | None = None
    transaction_id: int | None = None
    recurrency_kind: str | None = None  # Daily | Weekly (kind=Recurring only)
    min_charging_rate: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise as an OCPP 1.6 ``csChargingProfiles`` object."""
        schedule: dict[str, Any] = {
            "chargingRateUnit": self.rate_unit.value,
            "chargingSchedulePeriod": [
                {
                    "startPeriod": int(p.start_period),
                    "limit": round(float(p.limit), 1),
                    "numberPhases": p.number_phases,
                }
                for p in self.schedule
            ],
        }
        if self.duration_s is not None:
            schedule["duration"] = int(self.duration_s)
        if self.start_schedule is not None:
            schedule["startSchedule"] = _iso(self.start_schedule)
        if self.min_charging_rate is not None:
            schedule["minChargingRate"] = round(self.min_charging_rate, 1)

        out: dict[str, Any] = {
            "chargingProfileId": self.profile_id,
            "stackLevel": self.stack_level,
            "chargingProfilePurpose": self.purpose.value,
            "chargingProfileKind": self.kind,
            "chargingSchedule": schedule,
        }
        if self.transaction_id is not None:
            out["transactionId"] = self.transaction_id
        if self.recurrency_kind is not None:
            out["recurrencyKind"] = self.recurrency_kind
        if self.valid_from is not None:
            out["validFrom"] = _iso(self.valid_from)
        if self.valid_to is not None:
            out["validTo"] = _iso(self.valid_to)
        return out


@dataclass
class ChargePoint:
    """Representation of an OCPP ChargePoint (EV charger)."""

    charge_point_id: str
    status: ChargePointStatus = ChargePointStatus.AVAILABLE
    vendor: str = ""
    model: str = ""
    serial_number: str = ""
    firmware_version: str = ""
    num_connectors: int = 1
    max_power_kw: float = 22.0
    current_power_kw: float = 0.0
    current_soc: float | None = None
    v2g_capable: bool = False
    active_transaction_id: str | None = None
    active_profile: ChargingProfile | None = None
    last_heartbeat: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    connected: bool = False  # live WebSocket session open
    connector_status: dict[int, ChargePointStatus] = field(default_factory=dict)
    energy_import_kwh: float | None = None
    # Latest MeterValues readings per connector (power_kw, soc, energy_import_kwh, at)
    connector_meter: dict[int, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "charge_point_id": self.charge_point_id,
            "status": self.status.value,
            "vendor": self.vendor,
            "model": self.model,
            "max_power_kw": self.max_power_kw,
            "current_power_kw": self.current_power_kw,
            "current_soc": self.current_soc,
            "v2g_capable": self.v2g_capable,
            "active_transaction_id": self.active_transaction_id,
            "num_connectors": self.num_connectors,
            "connected": self.connected,
            "energy_import_kwh": self.energy_import_kwh,
            "last_heartbeat": self.last_heartbeat,
        }


# Callback type for charge-point status changes
CPStatusCallback = Callable[[ChargePoint, ChargePointStatus], Awaitable[None]]

# Aggregate connector statuses into one charge-point status: the first
# status (in this order) that any connector reports wins.
_STATUS_PRIORITY = (
    ChargePointStatus.CHARGING,
    ChargePointStatus.SUSPENDED_EV,
    ChargePointStatus.SUSPENDED_EVSE,
    ChargePointStatus.PREPARING,
    ChargePointStatus.FINISHING,
    ChargePointStatus.RESERVED,
    ChargePointStatus.FAULTED,
    ChargePointStatus.AVAILABLE,
    ChargePointStatus.UNAVAILABLE,
)

_MAX_CHARGE_POINT_ID_LENGTH = 64

# push_charging_profile() outcomes besides the charger's own status strings.
PROFILE_SIMULATED = "simulated"
PROFILE_NOT_CONNECTED = "not_connected"
PROFILE_ERROR = "error"


def _require(payload: dict[str, Any], *keys: str) -> None:
    missing = [k for k in keys if k not in payload]
    if missing:
        raise OCPPError(
            OCPPErrorCode.OCCURENCE_CONSTRAINT_VIOLATION,
            f"Missing required field(s): {', '.join(missing)}",
        )


def _as_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise OCPPError(OCPPErrorCode.TYPE_CONSTRAINT_VIOLATION, f"{key} must be an integer")
    return value


def _slot_value(slot: Any, key: str) -> float:
    if isinstance(slot, dict):
        return float(slot[key])
    return float(getattr(slot, key))


def schedule_to_charging_profile(
    slots: Iterable[Any],
    *,
    profile_id: int = 1,
    stack_level: int = 0,
    purpose: ChargingProfilePurpose = ChargingProfilePurpose.TX_PROFILE,
    number_phases: int = 3,
) -> ChargingProfile:
    """Convert V2G scheduler slots into an absolute OCPP charging profile.

    *slots* are :class:`vpp.v2g.scheduler.ScheduleSlot` objects (or dicts)
    with ``start_time``/``end_time`` (Unix seconds) and ``power_kw``
    (positive = charge, negative = discharge). Limits are emitted in watts.
    """
    ordered = sorted(slots, key=lambda s: _slot_value(s, "start_time"))
    if not ordered:
        raise ValueError("Cannot build a charging profile from an empty schedule")
    origin = _slot_value(ordered[0], "start_time")
    end = max(_slot_value(s, "end_time") for s in ordered)
    periods = [
        ChargingSchedulePeriod(
            start_period=round(_slot_value(s, "start_time") - origin),
            limit=_slot_value(s, "power_kw") * 1000.0,
            number_phases=number_phases,
        )
        for s in ordered
    ]
    return ChargingProfile(
        profile_id=profile_id,
        stack_level=stack_level,
        purpose=purpose,
        kind="Absolute",
        rate_unit=ChargingRateUnit.WATTS,
        schedule=periods,
        start_schedule=origin,
        duration_s=round(end - origin),
    )


class OCPPAdapter(ProtocolAdapter):
    """OCPP 1.6-J Central System adapter.

    Configuration keys:
        central_system_enabled (bool, default False -> simulated mode),
        heartbeat_interval_s (int, default 300),
        call_timeout_s (float, default 30),
        auto_accept_boot (bool, default True: unknown charge points that
            send BootNotification are registered; False rejects them),
        allowed_charge_points (list[str] | None: WebSocket allow-list),
        basic_auth_password (str | None: OCPP Security Profile 1),
        authorized_id_tags (list[str] | None: None accepts every idTag),
        remote_id_tag (str, default "VPP"),
        publish_events (bool, default True: MeterValues -> RESOURCE_UPDATED)
    """

    def __init__(self) -> None:
        super().__init__("ocpp", "1.6")
        self._charge_points: dict[str, ChargePoint] = {}
        self._status_callbacks: list[CPStatusCallback] = []
        self._server: Any = None
        self._message_queue: asyncio.Queue[ProtocolMessage] = asyncio.Queue(maxsize=500)
        self._sessions: dict[str, OCPPJSession] = {}
        self._session_closers: dict[str, Callable[[], Awaitable[None]]] = {}
        self._transactions: dict[int, dict[str, Any]] = {}
        self._next_transaction_id = int(time.time()) % 1_000_000_000

    # -- Mode ----------------------------------------------------------------

    @property
    def live(self) -> bool:
        return bool(self._config.get("central_system_enabled", False))

    @property
    def mode(self) -> ProtocolMode:
        return ProtocolMode.LIVE if self.live else ProtocolMode.SIMULATED

    # -- Lifecycle -----------------------------------------------------------

    async def connect(self) -> None:
        """Start accepting charge points (live) or enter simulated mode."""
        self._status = ProtocolStatus.CONNECTING
        self.version = "1.6"
        if not self.live:
            self._enter_simulated_mode("no OCPP Central System configured")
            return
        self._status = ProtocolStatus.CONNECTED
        self._metrics.connected_since = time.time()
        logger.info("OCPP 1.6-J Central System accepting charge points at /ocpp/{id}")

    async def disconnect(self) -> None:
        for cp_id in list(self._sessions):
            closer = self._session_closers.get(cp_id)
            if closer is not None:
                try:
                    await closer()
                except Exception:
                    logger.debug("Error closing OCPP session %s", cp_id, exc_info=True)
            self._drop_session(cp_id)
        if self._server is not None:
            self._server.close()
            self._server = None
        self._status = ProtocolStatus.DISCONNECTED
        self._metrics.connected_since = None
        logger.info("OCPP central system stopped")

    async def send(self, message: ProtocolMessage) -> None:
        if not self.is_operational:
            raise ConnectionError("OCPP adapter not started")

        cp_id = str(message.payload.get("charge_point_id") or "")
        if not cp_id:
            raise ValueError("OCPP message payload needs a charge_point_id")
        action = message.payload.get("action", "")

        if action == "RemoteStartTransaction":
            await self.remote_start(cp_id)
        elif action == "RemoteStopTransaction":
            await self.remote_stop(cp_id)
        elif action == "SetChargingProfile":
            profile_data = message.payload.get("profile", {})
            await self.set_charging_profile(cp_id, profile_data)
        else:
            logger.warning("Unknown OCPP action: %s", action)

        self._metrics.messages_sent += 1
        self._metrics.last_message_at = time.time()

    async def receive(self) -> ProtocolMessage | None:
        try:
            return self._message_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    # -- WebSocket transport (called by vpp.api.routes.ocpp) ------------------

    def authenticate(self, charge_point_id: str, authorization: str | None) -> bool:
        """Decide whether a charge point may open a session.

        Enforces the id allow-list and, when ``basic_auth_password`` is
        set, OCPP Security Profile 1 HTTP Basic auth (username must equal
        the charge point id).
        """
        if not charge_point_id or len(charge_point_id) > _MAX_CHARGE_POINT_ID_LENGTH:
            return False
        allowed = self._config.get("allowed_charge_points")
        if allowed and charge_point_id not in allowed:
            logger.warning(
                "OCPP connection from non-allow-listed charge point %s", charge_point_id
            )
            return False
        password = self._config.get("basic_auth_password")
        if not password:
            return True
        if not authorization or not authorization.lower().startswith("basic "):
            return False
        try:
            decoded = base64.b64decode(authorization[6:].strip(), validate=True).decode()
        except (binascii.Error, UnicodeDecodeError):
            return False
        user, _, supplied = decoded.partition(":")
        return hmac.compare_digest(user, charge_point_id) and hmac.compare_digest(
            supplied.encode(), str(password).encode()
        )

    async def attach_session(
        self,
        charge_point_id: str,
        send_text: SendText,
        close: Callable[[], Awaitable[None]] | None = None,
    ) -> OCPPJSession:
        """Bind a newly-opened WebSocket to *charge_point_id*.

        A second connection for the same id replaces (and closes) the old one,
        which is what a charger does after a network blip.
        """
        if charge_point_id in self._sessions:
            logger.warning("Charge point %s reconnected; replacing old session", charge_point_id)
            old_close = self._session_closers.get(charge_point_id)
            self._drop_session(charge_point_id)
            if old_close is not None:
                try:
                    await old_close()
                except Exception:
                    logger.debug("Error closing stale session", exc_info=True)

        session = OCPPJSession(
            charge_point_id,
            send_text,
            self._handle_call,
            default_timeout=float(self._config.get("call_timeout_s", 30.0)),
        )
        self._sessions[charge_point_id] = session
        if close is not None:
            self._session_closers[charge_point_id] = close
        cp = self._charge_points.get(charge_point_id)
        if cp is not None:
            cp.connected = True
        logger.info("Charge point %s connected", charge_point_id)
        return session

    async def detach_session(self, charge_point_id: str, session: OCPPJSession) -> None:
        """Unbind a closed WebSocket (no-op if it was already replaced)."""
        if self._sessions.get(charge_point_id) is session:
            self._drop_session(charge_point_id)
            logger.info("Charge point %s disconnected", charge_point_id)
        else:
            session.close()

    def _drop_session(self, charge_point_id: str) -> None:
        session = self._sessions.pop(charge_point_id, None)
        self._session_closers.pop(charge_point_id, None)
        if session is not None:
            session.close()
        cp = self._charge_points.get(charge_point_id)
        if cp is not None:
            cp.connected = False

    def is_charge_point_connected(self, charge_point_id: str) -> bool:
        return charge_point_id in self._sessions

    def connected_charge_points(self) -> list[str]:
        return list(self._sessions)

    async def call(
        self,
        charge_point_id: str,
        action: str,
        payload: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Send an arbitrary OCPP CALL to a connected charge point."""
        session = self._sessions.get(charge_point_id)
        if session is None:
            raise ConnectionError(f"Charge point {charge_point_id} is not connected")
        self._metrics.messages_sent += 1
        self._metrics.last_message_at = time.time()
        try:
            return await session.call(action, payload, timeout=timeout)
        except (OCPPCallError, OCPPTimeoutError, ConnectionError):
            self._metrics.errors += 1
            raise

    # -- ChargePoint management ----------------------------------------------

    def register_charge_point(self, cp: ChargePoint) -> None:
        """Register a charge point in the central system."""
        cp.connected = cp.charge_point_id in self._sessions
        self._charge_points[cp.charge_point_id] = cp
        logger.info("Registered charge point: %s", cp.charge_point_id)

    def unregister_charge_point(self, cp_id: str) -> None:
        self._charge_points.pop(cp_id, None)

    def get_charge_point(self, cp_id: str) -> ChargePoint | None:
        return self._charge_points.get(cp_id)

    def list_charge_points(self) -> list[ChargePoint]:
        return list(self._charge_points.values())

    def on_status_change(self, callback: CPStatusCallback) -> None:
        """Register a callback for charge-point status changes."""
        self._status_callbacks.append(callback)

    # -- OCPP operations (Central System -> Charge Point) --------------------

    def _route(self, cp_id: str) -> str:
        """'live' if a session exists, 'sim' if simulated, '' if unreachable."""
        if cp_id in self._sessions:
            return "live"
        if self.is_simulated:
            return "sim"
        return ""

    async def _live_call(self, cp_id: str, action: str, payload: dict[str, Any]) -> str | None:
        """Send *action*; return the response ``status`` or None on failure."""
        try:
            response = await self.call(cp_id, action, payload)
        except (OCPPCallError, OCPPTimeoutError, ConnectionError) as exc:
            logger.warning("%s to %s failed: %s", action, cp_id, exc)
            return None
        status = response.get("status")
        if status != "Accepted":
            logger.warning("%s rejected by %s: %s", action, cp_id, status)
        return status if isinstance(status, str) else None

    async def remote_start(
        self,
        cp_id: str,
        connector_id: int = 1,
        *,
        id_tag: str | None = None,
        charging_profile: ChargingProfile | None = None,
    ) -> bool:
        """Send RemoteStartTransaction to a charge point.

        Live: returns True when the charger answers ``Accepted``; the
        transaction itself is then reported by the charger's own
        StartTransaction / StatusNotification.
        """
        cp = self._charge_points.get(cp_id)
        route = self._route(cp_id)
        if route == "live":
            payload: dict[str, Any] = {
                "connectorId": connector_id,
                "idTag": id_tag or str(self._config.get("remote_id_tag", "VPP")),
            }
            if charging_profile is not None:
                payload["chargingProfile"] = charging_profile.to_dict()
            return await self._live_call(cp_id, "RemoteStartTransaction", payload) == "Accepted"

        if cp is None:
            logger.warning("Unknown charge point: %s", cp_id)
            return False
        if route != "sim":
            logger.warning("Charge point %s is not connected", cp_id)
            return False
        if cp.status != ChargePointStatus.AVAILABLE:
            logger.warning("Charge point %s is %s, cannot start", cp_id, cp.status.value)
            return False

        cp.active_transaction_id = str(uuid.uuid4())
        await self._update_cp_status(cp, ChargePointStatus.CHARGING)
        logger.info("[simulated] Remote start on %s (tx=%s)", cp_id, cp.active_transaction_id)
        return True

    async def remote_stop(self, cp_id: str, transaction_id: int | None = None) -> bool:
        """Send RemoteStopTransaction to a charge point.

        *transaction_id* selects the transaction on multi-connector chargers;
        by default the charge point's most recent transaction is stopped.
        """
        cp = self._charge_points.get(cp_id)
        if cp is None:
            return False
        if transaction_id is None and cp.active_transaction_id is None:
            logger.warning("No active transaction on %s", cp_id)
            return False

        route = self._route(cp_id)
        if route == "live":
            try:
                tx_id = (
                    int(transaction_id)
                    if transaction_id is not None
                    else int(str(cp.active_transaction_id))
                )
            except ValueError:
                logger.warning(
                    "Transaction id %r on %s is not an OCPP id", cp.active_transaction_id, cp_id
                )
                return False
            status = await self._live_call(
                cp_id, "RemoteStopTransaction", {"transactionId": tx_id}
            )
            return status == "Accepted"
        if route != "sim":
            logger.warning("Charge point %s is not connected", cp_id)
            return False

        cp.active_transaction_id = None
        cp.current_power_kw = 0.0
        await self._update_cp_status(cp, ChargePointStatus.AVAILABLE)
        logger.info("[simulated] Remote stop on %s", cp_id)
        return True

    async def set_charging_profile(
        self,
        cp_id: str,
        profile: ChargingProfile | dict[str, Any],
        connector_id: int | None = None,
    ) -> bool:
        """Apply a charging profile to a charge point (SetChargingProfile)."""
        outcome = await self.push_charging_profile(cp_id, profile, connector_id)
        return outcome in ("Accepted", PROFILE_SIMULATED)

    async def push_charging_profile(
        self,
        cp_id: str,
        profile: ChargingProfile | dict[str, Any],
        connector_id: int | None = None,
    ) -> str:
        """SetChargingProfile, reporting *what happened* rather than a bool.

        Returns the charger's answer (``"Accepted"``, ``"Rejected"``,
        ``"NotSupported"``), :data:`PROFILE_SIMULATED` when the adapter is
        in simulated mode (applied to the in-memory registry only),
        :data:`PROFILE_NOT_CONNECTED` when the charge point has no live
        session / is unknown, or :data:`PROFILE_ERROR` when the call timed
        out or failed.
        """
        cp = self._charge_points.get(cp_id)
        route = self._route(cp_id)
        if cp is None and route != "live":
            return PROFILE_NOT_CONNECTED

        if isinstance(profile, dict):
            periods = [ChargingSchedulePeriod(**p) for p in profile.get("schedule", [])]
            profile = ChargingProfile(
                profile_id=profile.get("profile_id", 1),
                schedule=periods,
            )

        if route == "live":
            if (
                profile.purpose == ChargingProfilePurpose.TX_PROFILE
                and profile.transaction_id is None
                and cp is not None
                and cp.active_transaction_id is not None
                and cp.active_transaction_id.isdigit()
            ):
                profile.transaction_id = int(cp.active_transaction_id)
            if connector_id is None:
                connector_id = (
                    0 if profile.purpose == ChargingProfilePurpose.CHARGE_POINT_MAX else 1
                )
            status = await self._live_call(
                cp_id,
                "SetChargingProfile",
                {"connectorId": connector_id, "csChargingProfiles": profile.to_dict()},
            )
            if status != "Accepted":
                return status or PROFILE_ERROR
            outcome = "Accepted"
        elif route != "sim":
            logger.warning("Charge point %s is not connected", cp_id)
            return PROFILE_NOT_CONNECTED
        else:
            outcome = PROFILE_SIMULATED

        if cp is not None:
            cp.active_profile = profile
        logger.info("Set charging profile on %s (id=%d)", cp_id, profile.profile_id)

        msg = ProtocolMessage(
            topic=f"ocpp/profile/{cp_id}",
            payload={"charge_point_id": cp_id, "profile": profile.to_dict()},
            source="ocpp",
        )
        await self._dispatch(msg)
        return outcome

    async def clear_charging_profile(
        self,
        cp_id: str,
        *,
        profile_id: int | None = None,
        connector_id: int | None = None,
    ) -> bool:
        """Send ClearChargingProfile (all profiles when no filter is given)."""
        cp = self._charge_points.get(cp_id)
        route = self._route(cp_id)
        if route == "live":
            payload: dict[str, Any] = {}
            if profile_id is not None:
                payload["id"] = profile_id
            if connector_id is not None:
                payload["connectorId"] = connector_id
            if await self._live_call(cp_id, "ClearChargingProfile", payload) != "Accepted":
                return False
        elif route != "sim" or cp is None:
            return False
        if cp is not None and (
            profile_id is None
            or (cp.active_profile is not None and cp.active_profile.profile_id == profile_id)
        ):
            cp.active_profile = None
        return True

    async def apply_v2g_schedule(
        self,
        cp_id: str,
        slots: Iterable[Any],
        *,
        profile_id: int = 200,
        connector_id: int | None = None,
    ) -> bool:
        """Push a V2G scheduler result to a charger as an absolute TxProfile."""
        profile = schedule_to_charging_profile(slots, profile_id=profile_id)
        return await self.set_charging_profile(cp_id, profile, connector_id)

    async def push_v2g_schedule(
        self,
        cp_id: str,
        slots: Iterable[Any],
        *,
        profile_id: int = 200,
        stack_level: int = 0,
        connector_id: int | None = None,
        transaction_id: int | None = None,
    ) -> str:
        """Like :meth:`apply_v2g_schedule` but returns the detailed outcome.

        With a known *transaction_id* the schedule is sent as a ``TxProfile``
        bound to that transaction; without one it is sent as a
        ``TxDefaultProfile`` on the connector (OCPP 1.6 only accepts
        ``TxProfile`` while a transaction is running), which the charger
        applies to the next transaction.
        """
        purpose = (
            ChargingProfilePurpose.TX_PROFILE
            if transaction_id is not None
            else ChargingProfilePurpose.TX_DEFAULT
        )
        profile = schedule_to_charging_profile(
            slots, profile_id=profile_id, stack_level=stack_level, purpose=purpose
        )
        profile.transaction_id = transaction_id
        return await self.push_charging_profile(cp_id, profile, connector_id)

    async def set_v2g_discharge_profile(
        self,
        cp_id: str,
        discharge_power_kw: float,
        duration_minutes: int = 60,
    ) -> bool:
        """Create a V2G discharge profile for a V2G-capable charger."""
        cp = self._charge_points.get(cp_id)
        if cp is None or not cp.v2g_capable:
            logger.warning("Charge point %s not V2G capable", cp_id)
            return False

        profile = ChargingProfile(
            profile_id=100,
            purpose=ChargingProfilePurpose.TX_PROFILE,
            schedule=[
                ChargingSchedulePeriod(
                    start_period=0,
                    limit=-discharge_power_kw * 1000,  # negative = discharge, in W
                    number_phases=3,
                ),
                ChargingSchedulePeriod(
                    start_period=duration_minutes * 60,
                    limit=0.0,
                    number_phases=3,
                ),
            ],
        )
        return await self.set_charging_profile(cp_id, profile)

    # -- Fleet-level queries -------------------------------------------------

    def list_transactions(self) -> list[dict[str, Any]]:
        """Open transactions reported by live charge points (StartTransaction seen,
        StopTransaction not yet)."""
        return [{"transaction_id": tx_id, **tx} for tx_id, tx in self._transactions.items()]

    def total_charging_power(self) -> float:
        """Sum of all active charging power in kW."""
        return sum(cp.current_power_kw for cp in self._charge_points.values())

    def available_v2g_capacity(self) -> float:
        """Total V2G discharge capacity from available V2G chargers in kW."""
        return sum(
            cp.max_power_kw
            for cp in self._charge_points.values()
            if cp.v2g_capable and cp.status == ChargePointStatus.CHARGING
        )

    # -- Inbound CALL handlers (Charge Point -> Central System) --------------

    async def _handle_call(
        self, cp_id: str, action: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        # One inbound CALL counts as one received message, however many
        # internal ProtocolMessages its handler fans out via _dispatch().
        received_before = self._metrics.messages_received
        self._metrics.last_message_at = time.time()
        try:
            handler = self._CALL_HANDLERS.get(action)
            if handler is None:
                raise OCPPError(OCPPErrorCode.NOT_IMPLEMENTED, f"Action {action} not implemented")
            cp = self._charge_points.get(cp_id)
            if cp is not None:
                cp.last_heartbeat = time.time()
            return await handler(self, cp_id, payload)
        finally:
            self._metrics.messages_received = received_before + 1

    def _ensure_cp(self, cp_id: str) -> ChargePoint:
        cp = self._charge_points.get(cp_id)
        if cp is None:
            cp = ChargePoint(charge_point_id=cp_id, connected=cp_id in self._sessions)
            self._charge_points[cp_id] = cp
        return cp

    def _id_tag_status(self, id_tag: str) -> str:
        allowed = self._config.get("authorized_id_tags")
        if allowed is None:
            return "Accepted"
        return "Accepted" if id_tag in allowed else "Invalid"

    async def _on_boot_notification(self, cp_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        _require(payload, "chargePointVendor", "chargePointModel")
        interval = int(self._config.get("heartbeat_interval_s", 300))
        now = _iso(time.time())
        known = cp_id in self._charge_points
        if not known and not self._config.get("auto_accept_boot", True):
            logger.warning("Rejected BootNotification from unknown charge point %s", cp_id)
            return {"status": "Rejected", "currentTime": now, "interval": interval}
        cp = self._ensure_cp(cp_id)
        cp.vendor = str(payload["chargePointVendor"])
        cp.model = str(payload["chargePointModel"])
        cp.serial_number = str(payload.get("chargePointSerialNumber", cp.serial_number))
        cp.firmware_version = str(payload.get("firmwareVersion", cp.firmware_version))
        cp.last_heartbeat = time.time()
        await self._publish(f"ocpp/boot/{cp_id}", {"charge_point_id": cp_id, **payload})
        logger.info("BootNotification accepted: %s (%s %s)", cp_id, cp.vendor, cp.model)
        return {"status": "Accepted", "currentTime": now, "interval": interval}

    async def _on_heartbeat(self, cp_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {"currentTime": _iso(time.time())}

    async def _on_status_notification(self, cp_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        _require(payload, "connectorId", "errorCode", "status")
        connector_id = _as_int(payload, "connectorId")
        try:
            status = ChargePointStatus(payload["status"])
        except ValueError as exc:
            raise OCPPError(
                OCPPErrorCode.PROPERTY_CONSTRAINT_VIOLATION,
                f"Unknown status {payload['status']!r}",
            ) from exc
        cp = self._ensure_cp(cp_id)
        cp.connector_status[connector_id] = status
        if connector_id > cp.num_connectors:
            cp.num_connectors = connector_id
        if payload.get("errorCode") not in (None, "NoError"):
            cp.metadata["last_error"] = {
                "connector_id": connector_id,
                "error_code": payload.get("errorCode"),
                "info": payload.get("info"),
                "vendor_error_code": payload.get("vendorErrorCode"),
            }
        await self._update_cp_status(cp, self._aggregate_status(cp, connector_id, status))
        await self._publish(
            f"ocpp/connector/{cp_id}",
            {
                "charge_point_id": cp_id,
                "connector_id": connector_id,
                "status": status.value,
                "error_code": payload.get("errorCode"),
            },
        )
        return {}

    @staticmethod
    def _aggregate_status(
        cp: ChargePoint, connector_id: int, status: ChargePointStatus
    ) -> ChargePointStatus:
        if connector_id == 0:
            return status
        reported = {s for cid, s in cp.connector_status.items() if cid > 0}
        for candidate in _STATUS_PRIORITY:
            if candidate in reported:
                return candidate
        return status

    async def _on_authorize(self, cp_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        _require(payload, "idTag")
        return {"idTagInfo": {"status": self._id_tag_status(str(payload["idTag"]))}}

    async def _on_start_transaction(self, cp_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        _require(payload, "connectorId", "idTag", "meterStart", "timestamp")
        connector_id = _as_int(payload, "connectorId")
        meter_start = _as_int(payload, "meterStart")
        id_status = self._id_tag_status(str(payload["idTag"]))
        self._next_transaction_id += 1
        tx_id = self._next_transaction_id
        cp = self._ensure_cp(cp_id)
        if id_status == "Accepted":
            cp.active_transaction_id = str(tx_id)
            self._transactions[tx_id] = {
                "charge_point_id": cp_id,
                "connector_id": connector_id,
                "id_tag": payload["idTag"],
                "meter_start_wh": meter_start,
                "started_at": payload["timestamp"],
            }
        await self._publish(
            f"ocpp/transaction/{cp_id}",
            {
                "charge_point_id": cp_id,
                "event": "started",
                "transaction_id": tx_id,
                "connector_id": connector_id,
                "id_tag": str(payload["idTag"]),
                "id_tag_status": id_status,
                "meter_start_wh": meter_start,
                "timestamp": payload["timestamp"],
            },
        )
        return {"transactionId": tx_id, "idTagInfo": {"status": id_status}}

    async def _on_stop_transaction(self, cp_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        _require(payload, "transactionId", "meterStop", "timestamp")
        tx_id = _as_int(payload, "transactionId")
        meter_stop = _as_int(payload, "meterStop")
        cp = self._ensure_cp(cp_id)
        tx = self._transactions.pop(tx_id, None)
        energy_kwh = None
        if tx is not None:
            energy_kwh = max(0, meter_stop - tx["meter_start_wh"]) / 1000.0
            cp.metadata["last_session_kwh"] = energy_kwh
        if cp.active_transaction_id == str(tx_id):
            cp.active_transaction_id = None
            cp.current_power_kw = 0.0
        await self._publish(
            f"ocpp/transaction/{cp_id}",
            {
                "charge_point_id": cp_id,
                "event": "stopped",
                "transaction_id": tx_id,
                "connector_id": tx["connector_id"] if tx is not None else None,
                "reason": payload.get("reason", "Local"),
                "energy_kwh": energy_kwh,
                "meter_stop_wh": meter_stop,
                "timestamp": payload["timestamp"],
            },
        )
        response: dict[str, Any] = {}
        if "idTag" in payload:
            response["idTagInfo"] = {"status": self._id_tag_status(str(payload["idTag"]))}
        return response

    async def _on_meter_values(self, cp_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        _require(payload, "connectorId", "meterValue")
        if not isinstance(payload["meterValue"], list):
            raise OCPPError(OCPPErrorCode.TYPE_CONSTRAINT_VIOLATION, "meterValue must be a list")
        cp = self._ensure_cp(cp_id)
        readings = parse_meter_values(payload["meterValue"])
        if "power_kw" in readings:
            cp.current_power_kw = readings["power_kw"]
        if "soc" in readings:
            cp.current_soc = readings["soc"]
        if "energy_import_kwh" in readings:
            cp.energy_import_kwh = readings["energy_import_kwh"]
        connector_id = payload.get("connectorId")
        if readings and isinstance(connector_id, int) and not isinstance(connector_id, bool):
            cp.connector_meter.setdefault(connector_id, {}).update(readings, at=time.time())
        data = {
            "charge_point_id": cp_id,
            "connector_id": payload.get("connectorId"),
            "transaction_id": payload.get("transactionId"),
            **readings,
        }
        await self._publish(f"ocpp/meter/{cp_id}", data)
        if readings and self._config.get("publish_events", True):
            await self._publish_resource_update(cp, readings)
        return {}

    async def _on_data_transfer(self, cp_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        _require(payload, "vendorId")
        return {"status": "UnknownVendorId"}

    async def _on_ack_only(self, cp_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {}

    _CALL_HANDLERS: ClassVar[
        dict[str, Callable[[OCPPAdapter, str, dict[str, Any]], Awaitable[dict[str, Any]]]]
    ] = {
        "BootNotification": _on_boot_notification,
        "Heartbeat": _on_heartbeat,
        "StatusNotification": _on_status_notification,
        "Authorize": _on_authorize,
        "StartTransaction": _on_start_transaction,
        "StopTransaction": _on_stop_transaction,
        "MeterValues": _on_meter_values,
        "DataTransfer": _on_data_transfer,
        "DiagnosticsStatusNotification": _on_ack_only,
        "FirmwareStatusNotification": _on_ack_only,
    }

    # -- Internal ------------------------------------------------------------

    async def _publish(self, topic: str, payload: dict[str, Any]) -> None:
        msg = ProtocolMessage(topic=topic, payload=payload, source="ocpp")
        await self._dispatch(msg)
        self._enqueue(msg)

    def _enqueue(self, msg: ProtocolMessage) -> None:
        """Buffer *msg* for :meth:`receive`, dropping the oldest when full.

        Nothing is required to drain this queue (subscribers get every
        message via ``_dispatch``), so a full buffer is not an error.
        """
        if self._message_queue.full():
            with contextlib.suppress(asyncio.QueueEmpty):
                self._message_queue.get_nowait()
        self._message_queue.put_nowait(msg)

    async def _publish_resource_update(self, cp: ChargePoint, readings: dict[str, Any]) -> None:
        from vpp.events import Event, EventType, get_event_bus

        resource_id = str(cp.metadata.get("resource_id", cp.charge_point_id))
        try:
            await get_event_bus().publish(
                Event(
                    event_type=EventType.RESOURCE_UPDATED,
                    data={
                        "resource_id": resource_id,
                        "charge_point_id": cp.charge_point_id,
                        **readings,
                    },
                    source="ocpp.meter_values",
                )
            )
        except Exception:
            logger.exception("Failed to publish RESOURCE_UPDATED for %s", cp.charge_point_id)

    async def _update_cp_status(self, cp: ChargePoint, new_status: ChargePointStatus) -> None:
        old = cp.status
        cp.status = new_status
        if old == new_status:
            return
        for cb in self._status_callbacks:
            try:
                await cb(cp, old)
            except Exception:
                logger.exception("Status callback error for %s", cp.charge_point_id)

        msg = ProtocolMessage(
            topic=f"ocpp/status/{cp.charge_point_id}",
            payload={
                "charge_point_id": cp.charge_point_id,
                "old": old.value,
                "new": new_status.value,
            },
            source="ocpp",
        )
        await self._dispatch(msg)
        self._enqueue(msg)


_POWER_UNITS = {"W": 0.001, "kW": 1.0}
_ENERGY_UNITS = {"Wh": 0.001, "kWh": 1.0}


def parse_meter_values(meter_values: list[Any]) -> dict[str, float]:
    """Extract power (kW, + import / - export), SoC (0-1) and energy (kWh).

    Uses the most recent ``meterValue`` entry that carries each measurand.
    Phase-level readings are summed when no total is reported. Signed-data
    values (``format: SignedData``) are skipped.
    """
    out: dict[str, float] = {}
    for entry in meter_values:
        if not isinstance(entry, dict):
            continue
        totals: dict[str, float] = {}
        phases: dict[str, float] = {}
        for sv in entry.get("sampledValue") or []:
            if not isinstance(sv, dict) or sv.get("format") == "SignedData":
                continue
            measurand = sv.get("measurand", "Energy.Active.Import.Register")
            try:
                value = float(sv["value"])
            except (KeyError, TypeError, ValueError):
                continue
            if measurand.startswith("Power.Active."):
                scale = _POWER_UNITS.get(sv.get("unit", "W"))
                key = "power_import_kw" if measurand.endswith("Import") else "power_export_kw"
            elif measurand == "Energy.Active.Import.Register":
                scale = _ENERGY_UNITS.get(sv.get("unit", "Wh"))
                key = "energy_import_kwh"
            elif measurand == "SoC":
                scale, key = 0.01, "soc"
            else:
                continue
            if scale is None:
                continue
            target = phases if sv.get("phase") and key != "soc" else totals
            target[key] = target.get(key, 0.0) + value * scale
        merged = {**phases, **totals}
        if "power_import_kw" in merged or "power_export_kw" in merged:
            out["power_kw"] = merged.get("power_import_kw", 0.0) - merged.get(
                "power_export_kw", 0.0
            )
        for key in ("energy_import_kwh", "soc"):
            if key in merged:
                out[key] = merged[key]
    return out
