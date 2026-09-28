"""DR orchestrator: grid signals (OpenADR / IEEE 2030.5) -> fleet dispatch.

This is what closes the VPP control loop::

    VTN event / utility DERControl
        -> translate (vpp.dr.translate: target kW + limits, safety caps)
        -> DB-backed dispatch (vpp.api.optimization_support.execute_dispatch:
           Pyomo/HiGHS LP over the persisted resources + connected V2G EVs,
           per-resource limits are hard bounds; recorded in optimization_runs)
        -> EV setpoints pushed to chargers (OCPP SetChargingProfile, stack
           level above the V2G schedule, for the dispatch interval)
        -> audit row in dr_event_responses + EventBus events

Rules
-----
* **Auto-response is off by default** (``VPP_DR_AUTO_RESPONSE_ENABLED``).
  While off, signals are still observed, recorded and published, and the VEN
  answers opt-in/out per ``VPP_OPENADR_AUTO_OPT_IN`` exactly as before; no
  dispatch is run.
* When on, a new OpenADR event is opted **in** only if its translated target
  is supported and the fleet's capability over the event window covers at
  least ``dr_min_opt_in_fraction`` of the request; otherwise **out**.
* During an event window the orchestrator dispatches the effective target
  (IEEE 2030.5 target > OpenADR target; IEEE 2030.5 limits and operator caps
  always clamp), re-dispatching when the target changes or every
  ``dr_redispatch_interval_s`` (fresh SOC each time). Opted-out, test,
  cancelled and completed events are never dispatched. When no signal is
  active any more, EV DR profiles are cleared and a release is recorded.
* Resource limits are never exceeded: the allocation bounds each resource by
  its rated power and the energy available over the interval; EVs only
  export when V2G-capable, plugged in and flexible w.r.t. their departure
  target; an unreachable target is reported as a shortfall, not forced.

Delivery honesty: stationary resources receive their allocation through the
``DISPATCH_EXECUTED`` event (device drivers subscribe to it); EV setpoints are
sent to chargers and each charger's answer is recorded.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from sqlalchemy import select

from vpp.db.models import DREventResponseModel, V2GVehicleModel
from vpp.dr.translate import (
    DRDirective,
    DRPolicy,
    FleetCapability,
    combine,
    opt_decision,
    translate_ieee2030_5,
    translate_openadr,
)
from vpp.events import EventType
from vpp.optimization.planning import FleetAsset, build_allocation_problem
from vpp.v2g.models import EVConnectionState
from vpp.v2g.ocpp_bridge import (
    DR_PROFILE_ID,
    DR_STACK_LEVEL,
    clear_profiles,
    deliver_slots,
    ocpp_adapter_from,
    publish,
    setpoint_slots,
    summarize_deliveries,
)
from vpp.v2g.store import EV_ASSET_PREFIX, row_to_ev, ts_to_dt

if TYPE_CHECKING:
    from collections.abc import Callable

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from vpp.protocols.base import ProtocolRegistry
    from vpp.protocols.openadr import DREvent, DRResponse

logger = logging.getLogger(__name__)

_SOURCE = "dr.orchestrator"
_MAX_INTERVAL_MIN = 240


def _clean(obj: Any) -> Any:
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


class DROrchestrator:
    """Turns active grid signals into recorded, limit-respecting dispatches."""

    def __init__(
        self,
        policy: DRPolicy,
        registry: ProtocolRegistry,
        *,
        session_factory: async_sessionmaker[AsyncSession]
        | Callable[[], async_sessionmaker[AsyncSession]]
        | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.policy = policy
        self.registry = registry
        self._factory = session_factory
        self._clock = clock
        self._lock = asyncio.Lock()
        self._active: dict[str, Any] | None = None
        self._seen_controls: set[str] = set()
        self.last_tick_at: float | None = None
        self.last_error: str | None = None
        self.dispatch_count = 0

    # -- plumbing ------------------------------------------------------------

    def _sessions(self) -> async_sessionmaker[AsyncSession]:
        from sqlalchemy.ext.asyncio import async_sessionmaker

        if isinstance(self._factory, async_sessionmaker):
            return self._factory
        if self._factory is not None:
            return self._factory()
        from vpp.db.engine import get_session_factory

        return get_session_factory()

    def _openadr(self) -> Any:
        from vpp.protocols.openadr import OpenADRAdapter

        a = self.registry.get("openadr")
        return a if isinstance(a, OpenADRAdapter) and a.is_operational else None

    def _ieee(self) -> Any:
        from vpp.protocols.ieee2030_5 import IEEE2030_5Adapter

        a = self.registry.get("ieee2030_5")
        return a if isinstance(a, IEEE2030_5Adapter) and a.is_operational else None

    def attach_openadr(self, adapter: Any) -> None:
        adapter.register_event_handler(self.on_openadr_event)

    async def _record(self, session: AsyncSession, **fields: Any) -> DREventResponseModel:
        details = fields.pop("details", None) or {}
        row = DREventResponseModel(
            created_at=datetime.now(timezone.utc),
            details_json=json.dumps(_clean(details), default=str),
            **fields,
        )
        session.add(row)
        await session.flush()
        return row

    async def _record_directive(
        self,
        directive: DRDirective,
        action: str,
        *,
        opt_type: str | None = None,
        reason: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        try:
            async with self._sessions()() as session:
                await self._record(
                    session,
                    protocol=directive.protocol,
                    source_id=directive.source_id,
                    revision=directive.revision,
                    action=action,
                    opt_type=opt_type,
                    signal_type=directive.signal_type,
                    signal_level=directive.signal_level,
                    target_kw=directive.target_kw,
                    window_start=ts_to_dt(directive.start) if directive.start else None,
                    window_end=ts_to_dt(directive.end) if directive.end else None,
                    reason=reason if reason is not None else directive.reason,
                    details=details or {"directive": directive.to_dict()},
                )
                await session.commit()
        except Exception:
            logger.exception("failed to record DR %s for %s", action, directive.source_id)

    # -- fleet ---------------------------------------------------------------

    async def _load(
        self, session: AsyncSession
    ) -> tuple[list[FleetAsset], dict[str, dict[str, Any]], list[V2GVehicleModel]]:
        """Online resources + plugged-in V2G vehicles as dispatchable assets."""
        from vpp.api.optimization_support import load_fleet_assets

        assets, _ = await load_fleet_assets(session)
        constraints: dict[str, dict[str, Any]] = {}
        vehicles: list[V2GVehicleModel] = []
        if self.policy.include_v2g:
            rows = (await session.execute(select(V2GVehicleModel))).scalars().all()
            for row in rows:
                if row.connection_state == EVConnectionState.DISCONNECTED.value:
                    continue
                ev = row_to_ev(row)
                aid = f"{EV_ASSET_PREFIX}{row.id}"
                soc_min = max(0.0, min(0.99, ev.min_soc))
                assets.append(
                    FleetAsset(
                        id=aid,
                        name=row.name or aid,
                        resource_type="battery",
                        rated_power_kw=max(ev.max_charge_kw, ev.max_discharge_kw),
                        capacity_kwh=ev.capacity_kwh,
                        capacity_source="recorded",
                        soc=ev.current_soc,
                        soc_source=row.soc_source,
                        soc_min=soc_min,
                        soc_max=1.0,
                        eta_charge=ev.charge_efficiency,
                        eta_discharge=ev.discharge_efficiency,
                        metadata={"ev_id": row.id},
                    )
                )
                can_export = ev.v2g_capable and ev.has_flexibility
                constraints[aid] = {
                    "max_kw": ev.max_discharge_kw if can_export else 0.0,
                    "min_kw": -ev.max_charge_kw,
                }
                vehicles.append(row)
        return assets, constraints, vehicles

    @staticmethod
    def capability(
        assets: list[FleetAsset],
        constraints: dict[str, dict[str, Any]],
        window_s: float,
        vehicles: int = 0,
    ) -> FleetCapability:
        """Sustainable export/absorb over *window_s* (energy-limited, per resource)."""
        dt_h = max(0.25, min(4.0, window_s / 3600.0))
        _, info = build_allocation_problem(assets, 0.0, dt_hours=dt_h, constraints=constraints)
        return FleetCapability(
            export_kw=sum(max(0.0, r["hi_kw"]) for r in info.values()),
            import_kw=sum(max(0.0, -r["lo_kw"]) for r in info.values()),
            resources=len(info),
            vehicles=vehicles,
        )

    async def fleet_capability(self, window_s: float) -> FleetCapability:
        async with self._sessions()() as session:
            assets, constraints, vehicles = await self._load(session)
        return self.capability(assets, constraints, window_s, len(vehicles))

    # -- OpenADR: event receipt ---------------------------------------------

    async def on_openadr_event(self, event: DREvent) -> DRResponse:
        """VEN event handler: decide and record the opt-in/out, publish events."""
        from vpp.protocols.openadr import DRResponse

        now = self._clock()
        meta = event.metadata or {}
        await publish(
            EventType.DR_EVENT_RECEIVED,
            {
                "protocol": "openadr",
                "event_id": event.event_id,
                "modification_number": meta.get("modification_number"),
                "signal_type": event.signal_type.value,
                "signal_level": event.signal_level,
                "start_time": event.start_time,
                "duration_seconds": event.duration_seconds,
                "status": event.status.value,
                "test_event": bool(meta.get("test_event")),
                "market_context": event.market_context,
            },
            _SOURCE,
        )
        try:
            window = max(60.0, event.end_time - max(now, event.start_time))
            cap = await self.fleet_capability(window)
        except Exception:
            logger.exception("could not assess fleet capability for DR event %s", event.event_id)
            cap = FleetCapability()
        directive = translate_openadr(event, cap, self.policy, now=now)
        opt_type, action, reason = opt_decision(
            directive, cap, self.policy, test_event=bool(meta.get("test_event"))
        )
        await self._record_directive(
            directive,
            "received",
            opt_type=opt_type,
            reason=reason,
            details={
                "directive": directive.to_dict(),
                "decision": action,
                "capability": cap.__dict__,
                "response_required": meta.get("response_required"),
            },
        )
        adapter = self._openadr()
        await publish(
            EventType.DR_RESPONSE_SENT,
            {
                "protocol": "openadr",
                "event_id": event.event_id,
                "opt_type": opt_type,
                "decision": action,
                "reason": reason,
                "automatic": True,
                # The adapter sends oadrCreatedEvent itself when the VTN asked.
                "delivery": "vtn" if adapter is not None and adapter.is_connected else "local",
            },
            _SOURCE,
        )
        return DRResponse(event_id=event.event_id, opt_type=opt_type)

    async def override_opt(
        self, event_id: str, opt_type: str, *, user: str | None = None, reason: str = ""
    ) -> dict[str, Any]:
        """Operator opt-in/out override for an OpenADR event."""
        adapter = self._openadr()
        if adapter is None:
            raise LookupError("OpenADR adapter is not running")
        event = adapter.get_event(event_id)
        if event is None:
            raise KeyError(event_id)
        sent_to_vtn = await adapter.set_opt(event_id, opt_type)
        directive = translate_openadr(event, FleetCapability(), self.policy, now=self._clock())
        note = reason or f"operator override by {user or 'unknown'}"
        await self._record_directive(
            directive,
            "opt_override",
            opt_type=opt_type,
            reason=note,
            details={"user": user, "sent_to_vtn": sent_to_vtn},
        )
        await publish(
            EventType.DR_RESPONSE_SENT,
            {
                "protocol": "openadr",
                "event_id": event_id,
                "opt_type": opt_type,
                "automatic": False,
                "user": user,
                "reason": note,
                "delivery": "vtn" if sent_to_vtn else "local",
            },
            _SOURCE,
        )
        released = False
        if opt_type == "optOut":
            async with self._lock:
                if self._active is not None and any(
                    s.get("protocol") == "openadr" and s.get("id") == event_id
                    for s in self._active["directive"].sources
                ):
                    await self._release(f"operator opted out of {event_id}")
                    released = True
        return {
            "event_id": event_id,
            "opt_type": opt_type,
            "sent_to_vtn": sent_to_vtn,
            "released_dispatch": released,
        }

    # -- Periodic evaluation -------------------------------------------------

    def _openadr_candidate(self, now: float, cap: FleetCapability) -> DRDirective | None:
        from vpp.protocols.openadr import DREventStatus

        adapter = self._openadr()
        if adapter is None:
            return None
        best: tuple[tuple[int, float], DRDirective] | None = None
        for event in adapter.list_events():
            if event.status in (DREventStatus.CANCELLED, DREventStatus.COMPLETED):
                continue
            if not (event.start_time <= now < event.end_time):
                continue
            meta = event.metadata or {}
            if meta.get("test_event"):
                continue
            response = adapter.get_response(event.event_id)
            if response is not None and response.opt_type == "optOut":
                continue
            directive = translate_openadr(event, cap, self.policy, now=now)
            if not directive.supported or directive.target_kw is None:
                continue
            priority = meta.get("priority")
            rank = (int(priority) if isinstance(priority, int) else 1_000_000, -event.start_time)
            if best is None or rank < best[0]:
                best = (rank, directive)
        return best[1] if best else None

    async def _observe_ieee(self, controls: list[Any], directive: DRDirective | None) -> None:
        keys = {f"{c.program_id}/{c.control_id}" for c in controls}
        new = keys - self._seen_controls
        self._seen_controls = keys
        if not new or directive is None:
            return
        await publish(
            EventType.DR_EVENT_RECEIVED,
            {
                "protocol": "ieee2030_5",
                "control_ids": sorted(new),
                "target_kw": directive.target_kw,
                "max_export_kw": directive.max_export_kw,
                "max_import_kw": directive.max_import_kw,
            },
            _SOURCE,
        )
        await self._record_directive(
            directive,
            "received" if self.policy.auto_response else "observed",
            details={"directive": directive.to_dict(), "new_controls": sorted(new)},
        )

    async def tick(self) -> dict[str, Any] | None:
        """Evaluate active signals once; dispatch/release as needed.

        Returns a summary of what was done (``None`` when nothing changed).
        """
        async with self._lock:
            now = self._clock()
            self.last_tick_at = now
            try:
                return await self._tick(now)
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                logger.exception("DR orchestrator tick failed")
                return None

    async def _tick(self, now: float) -> dict[str, Any] | None:
        ieee = self._ieee()
        controls = ieee.get_active_controls() if ieee is not None else []
        openadr = self._openadr()
        has_openadr_window = openadr is not None and any(
            e.start_time <= now < e.end_time for e in openadr.list_events()
        )
        if not controls and not has_openadr_window and self._active is None:
            self._seen_controls = set()
            return None

        async with self._sessions()() as session:
            assets, constraints, vehicles = await self._load(session)
        window = self.policy.redispatch_interval_s
        cap = self.capability(assets, constraints, window, len(vehicles))
        ieee_directive = translate_ieee2030_5(controls, cap, self.policy)
        await self._observe_ieee(controls, ieee_directive)

        if not self.policy.auto_response:
            return None

        effective = combine(
            ieee_directive, self._openadr_candidate(now, cap) if openadr else None, self.policy
        )
        if effective is None:
            if self._active is not None:
                return await self._release("no active DR signal")
            return None

        active = self._active
        if (
            active is not None
            and active["key"] == effective.key()
            and now - active["dispatched_at"] < self.policy.redispatch_interval_s
        ):
            return None
        return await self._dispatch(effective, now, redispatch=active is not None)

    async def _dispatch(
        self, directive: DRDirective, now: float, *, redispatch: bool
    ) -> dict[str, Any]:
        from vpp.api.optimization_support import execute_dispatch

        horizon_s = self.policy.redispatch_interval_s
        if directive.end and directive.end > now:
            horizon_s = min(horizon_s, directive.end - now)
        interval_min = max(1, min(_MAX_INTERVAL_MIN, math.ceil(horizon_s / 60.0)))
        target = float(directive.target_kw or 0.0)

        async with self._sessions()() as session:
            assets, constraints, vehicles = await self._load(session)
            result = await execute_dispatch(
                session,
                assets,
                target,
                interval_minutes=interval_min,
                constraints=constraints,
                problem_type="dr_dispatch",
                extra_inputs={"dr": directive.to_dict()},
            )
            ev_alloc = {
                rid[len(EV_ASSET_PREFIX) :]: float(kw)
                for rid, kw in result["allocations"].items()
                if rid.startswith(EV_ASSET_PREFIX)
            }
            participants = [v for v in vehicles if v.id in ev_alloc]
            # Allocation is export-positive; OCPP/scheduler power is charge-positive.
            slots = {
                ev_id: setpoint_slots(-kw, interval_min * 60.0, start=now)
                for ev_id, kw in ev_alloc.items()
            }
            deliveries = (
                await deliver_slots(
                    ocpp_adapter_from(self.registry.get("ocpp")),
                    participants,
                    slots,
                    profile_id=DR_PROFILE_ID,
                    stack_level=DR_STACK_LEVEL,
                    max_periods=self.policy.max_profile_periods,
                )
                if participants
                else []
            )
            failed = result["status"] == "failed"
            row = await self._record(
                session,
                protocol=directive.protocol,
                source_id=directive.source_id,
                revision=directive.revision,
                action="failed" if failed else "dispatched",
                signal_type=directive.signal_type,
                signal_level=directive.signal_level,
                target_kw=target,
                delivered_kw=result["delivered_kw"],
                window_start=ts_to_dt(now),
                window_end=ts_to_dt(now + interval_min * 60.0),
                run_id=result["run_id"],
                reason=directive.reason if not failed else result["message"],
                details={
                    "directive": directive.to_dict(),
                    "redispatch": redispatch,
                    "status": result["status"],
                    "method": result["method"],
                    "shortfall_kw": result["shortfall_kw"],
                    "allocations": result["allocations"],
                    "ev_deliveries": deliveries,
                },
            )
            await session.commit()

        self.dispatch_count += 1
        self._active = {
            "key": directive.key(),
            "directive": directive,
            "dispatched_at": now,
            "run_id": result["run_id"],
            "record_id": row.id,
            "ev_ids": [v.id for v in participants],
            "target_kw": target,
            "delivered_kw": result["delivered_kw"],
            "status": result["status"],
        }
        summary = {
            "source": "dr",
            "action": "dispatched",
            "protocol": directive.protocol,
            "source_id": directive.source_id,
            "sources": directive.sources,
            "run_id": result["run_id"],
            "status": result["status"],
            "method": result["method"],
            "target_kw": target,
            "delivered_kw": result["delivered_kw"],
            "shortfall_kw": result["shortfall_kw"],
            "interval_minutes": interval_min,
            "allocations": result["allocations"],
            "ev_deliveries": summarize_deliveries(deliveries),
            "redispatch": redispatch,
            "reason": directive.reason,
        }
        await publish(EventType.DISPATCH_EXECUTED, _clean(summary), _SOURCE)
        if participants:
            await publish(
                EventType.V2G_DISPATCH,
                _clean(
                    {
                        "source": "dr",
                        "run_id": result["run_id"],
                        "dispatch_kw": sum(ev_alloc.values()),
                        "connected_evs": len(vehicles),
                        "avg_soc": sum(v.current_soc for v in vehicles) / len(vehicles),
                        "deliveries": deliveries,
                    }
                ),
                _SOURCE,
            )
        logger.info(
            "DR dispatch (%s %s): target %.1f kW, delivered %.1f kW [%s]",
            directive.protocol,
            directive.source_id,
            target,
            result["delivered_kw"],
            result["status"],
        )
        return summary

    async def _release(self, reason: str) -> dict[str, Any]:
        active = self._active
        self._active = None
        if active is None:
            return {"action": "noop"}
        cleared: list[dict[str, Any]] = []
        adapter = ocpp_adapter_from(self.registry.get("ocpp"))
        try:
            async with self._sessions()() as session:
                vehicles = []
                if active["ev_ids"]:
                    vehicles = list(
                        (
                            await session.execute(
                                select(V2GVehicleModel).where(
                                    V2GVehicleModel.id.in_(active["ev_ids"])
                                )
                            )
                        )
                        .scalars()
                        .all()
                    )
                cleared = await clear_profiles(adapter, vehicles, profile_id=DR_PROFILE_ID)
                directive: DRDirective = active["directive"]
                await self._record(
                    session,
                    protocol=directive.protocol,
                    source_id=directive.source_id,
                    revision=directive.revision,
                    action="released",
                    signal_type=directive.signal_type,
                    target_kw=None,
                    window_end=ts_to_dt(self._clock()),
                    run_id=active["run_id"],
                    reason=reason,
                    details={"cleared_ev_profiles": cleared},
                )
                await session.commit()
        except Exception:
            logger.exception("failed to release DR dispatch")
        summary = {
            "source": "dr",
            "action": "released",
            "protocol": active["directive"].protocol,
            "source_id": active["directive"].source_id,
            "run_id": active["run_id"],
            "reason": reason,
            "cleared_ev_profiles": cleared,
        }
        await publish(EventType.DISPATCH_EXECUTED, summary, _SOURCE)
        return summary

    async def run_forever(self) -> None:
        while True:
            await self.tick()
            await asyncio.sleep(max(0.5, self.policy.tick_interval_s))

    def status(self) -> dict[str, Any]:
        active = None
        if self._active is not None:
            active = {k: v for k, v in self._active.items() if k not in ("key", "directive")} | {
                "directive": self._active["directive"].to_dict()
            }
        out: dict[str, Any] = _clean(
            {
                "running": True,
                "auto_response_enabled": self.policy.auto_response,
                "policy": self.policy.to_dict(),
                "active": active,
                "dispatch_count": self.dispatch_count,
                "last_tick_at": self.last_tick_at,
                "last_error": self.last_error,
                "protocols": {
                    "openadr": self._openadr() is not None,
                    "ieee2030_5": self._ieee() is not None,
                    "ocpp": ocpp_adapter_from(self.registry.get("ocpp")) is not None,
                },
            }
        )
        return out


# ---------------------------------------------------------------------------
# Process-wide instance (created by vpp.protocols.bootstrap)
# ---------------------------------------------------------------------------

_orchestrator: DROrchestrator | None = None


def get_dr_orchestrator() -> DROrchestrator | None:
    return _orchestrator


def set_dr_orchestrator(orchestrator: DROrchestrator | None) -> None:
    global _orchestrator
    _orchestrator = orchestrator


def response_to_dict(row: DREventResponseModel) -> dict[str, Any]:
    from vpp.v2g.store import dt_to_ts, loads

    return {
        "id": row.id,
        "created_at": dt_to_ts(row.created_at),
        "protocol": row.protocol,
        "source_id": row.source_id,
        "revision": row.revision,
        "action": row.action,
        "opt_type": row.opt_type,
        "signal_type": row.signal_type,
        "signal_level": row.signal_level,
        "target_kw": row.target_kw,
        "delivered_kw": row.delivered_kw,
        "window_start": dt_to_ts(row.window_start),
        "window_end": dt_to_ts(row.window_end),
        "run_id": row.run_id,
        "reason": row.reason,
        "details": loads(row.details_json, {}),
    }


__all__ = [
    "DROrchestrator",
    "get_dr_orchestrator",
    "response_to_dict",
    "set_dr_orchestrator",
]
