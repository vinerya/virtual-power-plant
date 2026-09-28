"""Bridge between the persistent V2G fleet and the OCPP Central System.

Inbound (charger -> fleet), via :class:`OCPPVehicleBridge` subscribed to the
:class:`~vpp.protocols.ocpp.OCPPAdapter` message stream:

* ``StartTransaction`` whose ``idTag`` matches a vehicle's ``id_tag`` binds
  that vehicle to the (charge point, connector) automatically
  (``binding_source="id_tag"``); a vehicle bound to the connector by an
  operator (``binding_source="manual"``) picks the transaction up too.
  Every accepted transaction is recorded in ``v2g_charging_sessions``.
* ``StopTransaction`` closes the session row and the vehicle's transaction.
* ``MeterValues`` update the bound vehicle's SOC (``soc_source="ocpp"``),
  power and charging/discharging state, and publish ``RESOURCE_UPDATED``
  (``resource_id="ev:<ev_id>"``).
* ``StatusNotification`` per connector drives plug-in / unplug:
  ``Available`` means nothing is plugged in (the vehicle becomes
  ``disconnected`` and an automatic id-tag binding is released);
  ``Preparing``/``Suspended*``/``Finishing`` mean plugged in but idle;
  ``Charging`` means energy is flowing. ``EV_CONNECTED`` /
  ``EV_DISCONNECTED`` are published on transitions.

Outbound (fleet -> charger), :func:`deliver_slots` pushes per-vehicle
schedules / setpoints as OCPP ``SetChargingProfile`` and reports per-vehicle
what actually happened -- the charger's own answer when live, ``simulated``
when the Central System is simulated, never a silent success.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from vpp.db.models import V2GChargingSessionModel
from vpp.events import Event, EventType, get_event_bus
from vpp.protocols.ocpp import (
    PROFILE_ERROR,
    PROFILE_NOT_CONNECTED,
    PROFILE_SIMULATED,
    ChargePointStatus,
    OCPPAdapter,
)
from vpp.v2g.models import EVConnectionState
from vpp.v2g.store import EV_ASSET_PREFIX, V2GRepository, utc

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from vpp.db.models import V2GVehicleModel
    from vpp.protocols.base import ProtocolAdapter, ProtocolMessage

logger = logging.getLogger(__name__)

# Power (kW) below which a connector is considered idle rather than charging.
_IDLE_POWER_KW = 0.05

# Profile ids / stack levels used by the platform. DR setpoints stack above
# the V2G schedule so they override it for the event window only.
SCHEDULE_PROFILE_ID = 200
SCHEDULE_STACK_LEVEL = 0
DISPATCH_PROFILE_ID = 250  # operator V2G dispatch (POST /api/v1/v2g/dispatch)
DISPATCH_STACK_LEVEL = 1
DR_PROFILE_ID = 300
DR_STACK_LEVEL = 2

# Delivery status values reported per vehicle.
DELIVERY_ACCEPTED = "accepted"
DELIVERY_REJECTED = "rejected"
DELIVERY_NOT_SUPPORTED = "not_supported"
DELIVERY_SIMULATED = "simulated"
DELIVERY_NOT_CONNECTED = "not_connected"
DELIVERY_ERROR = "error"
DELIVERY_NOT_BOUND = "not_bound"
DELIVERY_NO_OCPP = "no_ocpp"
DELIVERY_NO_SLOTS = "no_slots"
DELIVERY_SKIPPED = "skipped"

_OUTCOME_TO_DELIVERY = {
    "Accepted": DELIVERY_ACCEPTED,
    "Rejected": DELIVERY_REJECTED,
    "NotSupported": DELIVERY_NOT_SUPPORTED,
    PROFILE_SIMULATED: DELIVERY_SIMULATED,
    PROFILE_NOT_CONNECTED: DELIVERY_NOT_CONNECTED,
    PROFILE_ERROR: DELIVERY_ERROR,
}

_PLUGGED_IDLE = {
    ChargePointStatus.PREPARING.value,
    ChargePointStatus.SUSPENDED_EV.value,
    ChargePointStatus.SUSPENDED_EVSE.value,
    ChargePointStatus.FINISHING.value,
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_ocpp_time(value: Any) -> datetime:
    if isinstance(value, str):
        try:
            return utc(datetime.fromisoformat(value.replace("Z", "+00:00"))) or _now()
        except ValueError:
            pass
    return _now()


async def publish(event_type: EventType, data: dict[str, Any], source: str) -> None:
    """Publish on the EventBus; failures are logged, never raised."""
    try:
        await get_event_bus().publish(Event(event_type=event_type, data=data, source=source))
    except Exception:
        logger.warning("failed to publish %s", event_type, exc_info=True)


def ocpp_adapter_from(adapter: ProtocolAdapter | None) -> OCPPAdapter | None:
    """The OCPP adapter if it is registered and operational (live or simulated)."""
    if isinstance(adapter, OCPPAdapter) and adapter.is_operational:
        return adapter
    return None


# ---------------------------------------------------------------------------
# Inbound: charger -> fleet
# ---------------------------------------------------------------------------


class OCPPVehicleBridge:
    """Keeps ``v2g_vehicles`` in sync with what the OCPP chargers report."""

    def __init__(
        self,
        adapter: OCPPAdapter,
        session_factory: async_sessionmaker[AsyncSession]
        | Callable[[], async_sessionmaker[AsyncSession]]
        | None = None,
    ) -> None:
        self.adapter = adapter
        self._factory = session_factory

    def _sessions(self) -> async_sessionmaker[AsyncSession]:
        from sqlalchemy.ext.asyncio import async_sessionmaker

        if isinstance(self._factory, async_sessionmaker):
            return self._factory
        if self._factory is not None:
            return self._factory()
        from vpp.db.engine import get_session_factory

        return get_session_factory()

    def attach(self) -> None:
        self.adapter.subscribe("*", self.handle_message)

    def detach(self) -> None:
        self.adapter.unsubscribe("*", self.handle_message)

    async def handle_message(self, msg: ProtocolMessage) -> None:
        kind = msg.topic.split("/", 2)[1] if msg.topic.count("/") >= 2 else ""
        handler = {
            "transaction": self._on_transaction,
            "meter": self._on_meter,
            "connector": self._on_connector_status,
        }.get(kind)
        if handler is None:
            return
        async with self._sessions()() as session:
            events = await handler(session, msg.payload)
            await session.commit()
        for event_type, data in events:
            await publish(event_type, data, "v2g.ocpp")

    # -- Handlers (return events to publish after commit) --------------------

    async def _on_transaction(
        self, session: AsyncSession, p: dict[str, Any]
    ) -> list[tuple[EventType, dict[str, Any]]]:
        if p.get("event") == "started":
            return await self._on_started(session, p)
        if p.get("event") == "stopped":
            return await self._on_stopped(session, p)
        return []

    async def _on_started(
        self, session: AsyncSession, p: dict[str, Any]
    ) -> list[tuple[EventType, dict[str, Any]]]:
        if p.get("id_tag_status") != "Accepted":
            return []
        cp_id = str(p["charge_point_id"])
        connector_id = int(p.get("connector_id") or 1)
        tx_id = int(p["transaction_id"])
        id_tag = p.get("id_tag")
        events: list[tuple[EventType, dict[str, Any]]] = []

        vehicle = await V2GRepository.by_id_tag(session, str(id_tag)) if id_tag else None
        if vehicle is not None and (
            vehicle.charge_point_id != cp_id or vehicle.connector_id != connector_id
        ):
            occupant = await V2GRepository.by_connector(session, cp_id, connector_id)
            if occupant is not None and occupant.id != vehicle.id:
                logger.warning(
                    "idTag %s identifies vehicle %s on %s/%s; releasing previous binding of %s",
                    id_tag,
                    vehicle.id,
                    cp_id,
                    connector_id,
                    occupant.id,
                )
                events.append(self._disconnected_event(occupant, "rebound"))
                self._unbind(occupant)
                await session.flush()
            vehicle.charge_point_id = cp_id
            vehicle.connector_id = connector_id
            vehicle.binding_source = "id_tag"
            await session.flush()
        elif vehicle is None:
            vehicle = await V2GRepository.by_connector(session, cp_id, connector_id)

        session.add(
            V2GChargingSessionModel(
                created_at=_now(),
                vehicle_id=vehicle.id if vehicle is not None else None,
                charge_point_id=cp_id,
                connector_id=connector_id,
                transaction_id=tx_id,
                id_tag=str(id_tag) if id_tag else None,
                status="active",
                started_at=_parse_ocpp_time(p.get("timestamp")),
                meter_start_wh=p.get("meter_start_wh"),
            )
        )
        if vehicle is None:
            return events

        was_disconnected = vehicle.connection_state == EVConnectionState.DISCONNECTED.value
        vehicle.active_transaction_id = tx_id
        if was_disconnected:
            vehicle.connection_state = EVConnectionState.CONNECTED_IDLE.value
            vehicle.connected_at = _now()
            events.append(
                (
                    EventType.EV_CONNECTED,
                    {
                        "ev_id": vehicle.id,
                        "charge_point_id": cp_id,
                        "connector_id": connector_id,
                        "transaction_id": tx_id,
                        "binding_source": vehicle.binding_source,
                    },
                )
            )
        return events

    async def _on_stopped(
        self, session: AsyncSession, p: dict[str, Any]
    ) -> list[tuple[EventType, dict[str, Any]]]:
        cp_id = str(p["charge_point_id"])
        tx_id = int(p["transaction_id"])
        row = await V2GRepository.open_session(session, cp_id, tx_id)
        if row is not None:
            row.status = "completed"
            row.stopped_at = _parse_ocpp_time(p.get("timestamp"))
            row.meter_stop_wh = p.get("meter_stop_wh")
            row.energy_kwh = p.get("energy_kwh")
            row.stop_reason = str(p.get("reason") or "Local")[:64]
        vehicle = await V2GRepository.by_transaction(session, cp_id, tx_id)
        if vehicle is not None:
            vehicle.active_transaction_id = None
            vehicle.current_power_kw = 0.0
            if vehicle.connection_state != EVConnectionState.DISCONNECTED.value:
                vehicle.connection_state = EVConnectionState.CONNECTED_IDLE.value
        return []

    async def _on_meter(
        self, session: AsyncSession, p: dict[str, Any]
    ) -> list[tuple[EventType, dict[str, Any]]]:
        cp_id = str(p["charge_point_id"])
        vehicle = None
        tx_id = p.get("transaction_id")
        if isinstance(tx_id, int) and not isinstance(tx_id, bool):
            vehicle = await V2GRepository.by_transaction(session, cp_id, tx_id)
        connector_id = p.get("connector_id")
        if vehicle is None and isinstance(connector_id, int) and connector_id > 0:
            vehicle = await V2GRepository.by_connector(session, cp_id, connector_id)
        if vehicle is None:
            return []

        data: dict[str, Any] = {
            "resource_id": f"{EV_ASSET_PREFIX}{vehicle.id}",
            "resource_type": "ev",
            "ev_id": vehicle.id,
            "charge_point_id": cp_id,
            "connector_id": vehicle.connector_id,
        }
        if "soc" in p:
            vehicle.current_soc = max(0.0, min(1.0, float(p["soc"])))
            vehicle.soc_source = "ocpp"
            vehicle.soc_updated_at = _now()
            data["soc"] = vehicle.current_soc
        if "power_kw" in p:
            power = float(p["power_kw"])
            vehicle.current_power_kw = power
            data["power_kw"] = power
            if power > _IDLE_POWER_KW:
                vehicle.connection_state = EVConnectionState.CHARGING.value
            elif power < -_IDLE_POWER_KW:
                vehicle.connection_state = EVConnectionState.DISCHARGING.value
            else:
                vehicle.connection_state = EVConnectionState.CONNECTED_IDLE.value
            if vehicle.connected_at is None:
                vehicle.connected_at = _now()
        if "soc" not in data and "power_kw" not in data:
            return []
        data["connection_state"] = vehicle.connection_state
        return [(EventType.RESOURCE_UPDATED, data)]

    async def _on_connector_status(
        self, session: AsyncSession, p: dict[str, Any]
    ) -> list[tuple[EventType, dict[str, Any]]]:
        cp_id = str(p["charge_point_id"])
        connector_id = p.get("connector_id")
        if not isinstance(connector_id, int) or connector_id <= 0:
            return []
        vehicle = await V2GRepository.by_connector(session, cp_id, connector_id)
        if vehicle is None:
            return []
        status = str(p.get("status"))
        vehicle.charger_status = status
        was_disconnected = vehicle.connection_state == EVConnectionState.DISCONNECTED.value

        if status == ChargePointStatus.AVAILABLE.value:
            if vehicle.active_transaction_id is not None:
                return []  # stale ordering; StopTransaction will follow
            event = self._disconnected_event(vehicle, "unplugged")
            vehicle.connection_state = EVConnectionState.DISCONNECTED.value
            vehicle.current_power_kw = 0.0
            vehicle.connected_at = None
            if vehicle.binding_source == "id_tag":
                self._unbind(vehicle)
            return [] if was_disconnected else [event]

        if status in _PLUGGED_IDLE:
            vehicle.connection_state = EVConnectionState.CONNECTED_IDLE.value
        elif status == ChargePointStatus.CHARGING.value:
            vehicle.connection_state = (
                EVConnectionState.DISCHARGING.value
                if vehicle.current_power_kw < -_IDLE_POWER_KW
                else EVConnectionState.CHARGING.value
            )
        else:  # Reserved / Unavailable / Faulted: no information about the car
            return []
        if was_disconnected:
            vehicle.connected_at = _now()
            return [
                (
                    EventType.EV_CONNECTED,
                    {
                        "ev_id": vehicle.id,
                        "charge_point_id": cp_id,
                        "connector_id": connector_id,
                        "charger_status": status,
                    },
                )
            ]
        return []

    # -- Helpers -------------------------------------------------------------

    @staticmethod
    def _unbind(vehicle: V2GVehicleModel) -> None:
        vehicle.charge_point_id = None
        vehicle.connector_id = None
        vehicle.binding_source = None
        vehicle.active_transaction_id = None
        vehicle.connection_state = EVConnectionState.DISCONNECTED.value
        vehicle.current_power_kw = 0.0

    @staticmethod
    def _disconnected_event(
        vehicle: V2GVehicleModel, reason: str
    ) -> tuple[EventType, dict[str, Any]]:
        return (
            EventType.EV_DISCONNECTED,
            {
                "ev_id": vehicle.id,
                "charge_point_id": vehicle.charge_point_id,
                "connector_id": vehicle.connector_id,
                "reason": reason,
            },
        )


# ---------------------------------------------------------------------------
# Outbound: fleet -> charger
# ---------------------------------------------------------------------------


def _slot(s: Any, key: str) -> float:
    return float(s[key] if isinstance(s, dict) else getattr(s, key))


def compact_slots(slots: Iterable[Any], max_periods: int) -> tuple[list[dict[str, float]], bool]:
    """Fill gaps with 0 kW, merge contiguous equal-power slots; cap at *max_periods*.

    Chargers advertise a small ``ChargingScheduleMaxPeriods`` (often 24-48),
    so a 96-slot scheduler result is merged first and, if still too long,
    truncated (the second return value says so).
    """
    ordered = sorted(
        (
            {
                "start_time": _slot(s, "start_time"),
                "end_time": _slot(s, "end_time"),
                "power_kw": round(_slot(s, "power_kw"), 2),
            }
            for s in slots
        ),
        key=lambda d: d["start_time"],
    )
    # An OCPP period lasts until the next one starts, so gaps between the
    # scheduler's (sparse) slots must become explicit 0 kW periods --
    # otherwise the charger would keep charging/discharging through them.
    filled: list[dict[str, float]] = []
    for s in ordered:
        if filled and s["start_time"] - filled[-1]["end_time"] > 1e-6:
            filled.append(
                {
                    "start_time": filled[-1]["end_time"],
                    "end_time": s["start_time"],
                    "power_kw": 0.0,
                }
            )
        filled.append(s)
    merged: list[dict[str, float]] = []
    for s in filled:
        last = merged[-1] if merged else None
        if (
            last is not None
            and last["power_kw"] == s["power_kw"]
            and abs(last["end_time"] - s["start_time"]) < 1e-6
        ):
            last["end_time"] = s["end_time"]
        else:
            merged.append(dict(s))
    truncated = len(merged) > max_periods
    return merged[: max(1, max_periods)], truncated


async def _deliver_one(
    adapter: OCPPAdapter,
    vehicle: V2GVehicleModel,
    slots: list[dict[str, float]],
    *,
    profile_id: int,
    stack_level: int,
) -> str:
    return await adapter.push_v2g_schedule(
        str(vehicle.charge_point_id),
        slots,
        profile_id=profile_id,
        stack_level=stack_level,
        connector_id=vehicle.connector_id,
        transaction_id=vehicle.active_transaction_id,
    )


async def deliver_slots(
    adapter: OCPPAdapter | None,
    vehicles: Iterable[V2GVehicleModel],
    slots_by_ev: dict[str, list[Any]],
    *,
    profile_id: int = SCHEDULE_PROFILE_ID,
    stack_level: int = SCHEDULE_STACK_LEVEL,
    max_periods: int = 48,
    push: bool = True,
) -> list[dict[str, Any]]:
    """Push each vehicle's slots to its bound charger; one result per vehicle.

    ``slots`` use the scheduler's sign convention (``power_kw`` > 0 charges
    the vehicle, < 0 discharges it to the grid); OCPP limits are emitted in
    watts with the same sign (negative = V2G discharge, a vendor extension
    in OCPP 1.6 -- chargers that don't support it answer ``Rejected``).
    """
    results: list[dict[str, Any]] = []
    pending: list[tuple[dict[str, Any], V2GVehicleModel, list[dict[str, float]]]] = []
    for vehicle in vehicles:
        result: dict[str, Any] = {
            "ev_id": vehicle.id,
            "charge_point_id": vehicle.charge_point_id,
            "connector_id": vehicle.connector_id,
            "transaction_id": vehicle.active_transaction_id,
            "status": "",
            "periods": 0,
            "detail": "",
        }
        results.append(result)
        raw = slots_by_ev.get(vehicle.id) or []
        if not raw:
            result["status"] = DELIVERY_NO_SLOTS
            result["detail"] = "the scheduler produced no slots for this vehicle"
            continue
        slots, truncated = compact_slots(raw, max_periods)
        result["periods"] = len(slots)
        if truncated:
            result["detail"] = f"schedule truncated to {max_periods} periods"
        if not push:
            result["status"] = DELIVERY_SKIPPED
            result["detail"] = "push_to_chargers=false"
        elif not vehicle.charge_point_id or vehicle.connector_id is None:
            result["status"] = DELIVERY_NOT_BOUND
            result["detail"] = "vehicle is not bound to an OCPP charge point/connector"
        elif adapter is None:
            result["status"] = DELIVERY_NO_OCPP
            result["detail"] = "OCPP Central System is not running (VPP_OCPP_ENABLED=false)"
        else:
            pending.append((result, vehicle, slots))

    if pending and adapter is not None:
        outcomes = await asyncio.gather(
            *(
                _deliver_one(adapter, v, s, profile_id=profile_id, stack_level=stack_level)
                for _, v, s in pending
            ),
            return_exceptions=True,
        )
        for (result, _v, _s), outcome in zip(pending, outcomes, strict=True):
            if isinstance(outcome, BaseException):
                result["status"] = DELIVERY_ERROR
                result["detail"] = f"{type(outcome).__name__}: {outcome}"
                continue
            result["status"] = _OUTCOME_TO_DELIVERY.get(str(outcome), DELIVERY_REJECTED)
            if result["status"] == DELIVERY_SIMULATED:
                result["detail"] = "OCPP adapter is simulated: applied in memory only"
            elif result["status"] == DELIVERY_NOT_CONNECTED:
                result["detail"] = "charge point has no live OCPP session"
            elif result["status"] not in (DELIVERY_ACCEPTED,):
                result["detail"] = result["detail"] or f"charger answered {outcome}"
    return results


def setpoint_slots(power_kw: float, duration_s: float, *, start: float | None = None) -> list:
    """One-period slot list holding *power_kw* (charge-positive) for *duration_s*."""
    t0 = time.time() if start is None else start
    return [{"start_time": t0, "end_time": t0 + max(60.0, duration_s), "power_kw": power_kw}]


async def clear_profiles(
    adapter: OCPPAdapter | None,
    vehicles: Iterable[V2GVehicleModel],
    *,
    profile_id: int = DR_PROFILE_ID,
) -> list[dict[str, Any]]:
    """ClearChargingProfile(*profile_id*) on each bound vehicle's charger."""
    out: list[dict[str, Any]] = []
    for vehicle in vehicles:
        if adapter is None or not vehicle.charge_point_id:
            continue
        ok = await adapter.clear_charging_profile(
            vehicle.charge_point_id, profile_id=profile_id, connector_id=vehicle.connector_id
        )
        out.append(
            {
                "ev_id": vehicle.id,
                "charge_point_id": vehicle.charge_point_id,
                "status": "cleared" if ok else "failed",
            }
        )
    return out


def summarize_deliveries(deliveries: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for d in deliveries:
        counts[d["status"]] = counts.get(d["status"], 0) + 1
    return counts


__all__ = [
    "DELIVERY_ACCEPTED",
    "DELIVERY_NOT_BOUND",
    "DELIVERY_NOT_CONNECTED",
    "DELIVERY_NO_OCPP",
    "DELIVERY_REJECTED",
    "DELIVERY_SIMULATED",
    "DR_PROFILE_ID",
    "DR_STACK_LEVEL",
    "SCHEDULE_PROFILE_ID",
    "OCPPVehicleBridge",
    "clear_profiles",
    "compact_slots",
    "deliver_slots",
    "ocpp_adapter_from",
    "publish",
    "setpoint_slots",
    "summarize_deliveries",
]
