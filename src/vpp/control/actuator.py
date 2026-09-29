"""Setpoint actuator: the consumer of dispatch allocations for stationary devices.

EV setpoints reach chargers over OCPP (:mod:`vpp.v2g.ocpp_bridge`). This
module does the same for batteries and inverters: it takes the per-resource
allocation of a dispatch (``POST /api/v1/optimization/dispatch`` with
``apply=true``, the DR orchestrator, and IEEE 2030.5 controls through the
orchestrator) and writes a power setpoint to each device that opted in.

Opt-in is two-level:

* ``VPP_CONTROL_ENABLED`` (default **false**) is the global kill switch.
  While off, nothing is ever written; deliveries report ``disabled``.
* Each resource opts in with ``metadata["modbus"]["control"]`` (see
  :mod:`vpp.protocols.modbus_control`). Without it: ``not_configured``;
  with ``"enabled": false``: ``disabled``.

Safety rules, in the order applied:

1. **Offline** resources are never written (``offline``).
2. **Clamp** to the resource's limits (battery: -charge .. +discharge limit,
   anything else: 0 .. rated power) and the control block's
   ``min_kw``/``max_kw``. ``clamped`` is reported. A generation resource
   allocated all of its estimated availability is *uncurtailed*: its upper
   limit is written instead of the estimate (``uncurtailed``), so a cap
   never pins the output at a stale estimate.
3. **Deadband**: a setpoint within ``deadband_kw`` of the one in force is
   not rewritten (``unchanged``), only its expiry is extended.
4. **Rate limit**: at most one write per ``min_interval_s``; a newer
   setpoint arriving sooner is held and written by the watchdog
   (``deferred``). Releases are never rate-limited.
5. **Read-back** verification (``verify``, default on): every written
   register is read back; a mismatch is a failure.
6. **Watchdog**: every setpoint carries an expiry (the dispatch interval
   plus ``VPP_CONTROL_EXPIRY_GRACE_S``). When it passes -- the dispatch
   ended, the orchestrator stopped re-dispatching, the API stopped calling
   -- the device falls back to ``safe_setpoint_kw`` if configured, else
   control is released (profile-specific, e.g. ``WMaxLim_Ena=0``). On API
   shutdown every active setpoint is released the same way. Profiles with
   a device-side revert timer (``revert_timeout_s``) also cover a crashed
   process; the watchdog refreshes the setpoint (``keepalive_s``) so that
   timer only fires when the VPP really is gone.

The output caps currently held on generation resources are exposed by
:meth:`SetpointActuator.output_limits` / :func:`active_output_limits`: the
optimiser needs them to tell a curtailed reading from the real availability
(:func:`vpp.api.optimization_support.resource_to_asset`).

Every command that touches (or would touch) a device is recorded in the
``event_log`` table (``event_type="device_setpoint"``, one row per
resource) and announced as a ``DEVICE_SETPOINT`` event on the EventBus.
Delivery statuses: ``accepted``, ``unchanged``, ``deferred``, ``simulated``
(``"simulate": true`` in the control block: full safety pipeline, no I/O),
``failed``, ``offline``, ``disabled``, ``not_configured``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from vpp.events import Event, EventType, get_event_bus
from vpp.protocols.modbus_control import (
    ControlConfigError,
    ModbusControlConfig,
    ModbusSetpointWriter,
    WriteResult,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from vpp.optimization.planning import FleetAsset
    from vpp.protocols.base import ProtocolRegistry
    from vpp.protocols.modbus import ModbusAdapter

logger = logging.getLogger(__name__)

EVENT_LOG_TYPE = "device_setpoint"
EV_PREFIX = "ev:"  # vpp.v2g.store.EV_ASSET_PREFIX (EVs are driven over OCPP)
_MAX_RELEASE_ATTEMPTS = 3
# Setpoints closer than this to a bound are treated as at the bound.
_LIMIT_EPS_KW = 1e-3

ACCEPTED = "accepted"
UNCHANGED = "unchanged"
DEFERRED = "deferred"
SIMULATED = "simulated"
FAILED = "failed"
OFFLINE = "offline"
DISABLED = "disabled"
NOT_CONFIGURED = "not_configured"


class SetpointWriter(Protocol):
    async def write(self, kw: float) -> WriteResult: ...

    async def release(self) -> WriteResult: ...


@dataclass
class _Active:
    resource_id: str
    name: str
    config: ModbusControlConfig
    writer: SetpointWriter | None
    lo_kw: float
    hi_kw: float
    setpoint_kw: float
    written_at: float
    expires_at: float
    source: str
    run_id: str | None
    pending_kw: float | None = None
    release_failures: int = 0
    # Generation (non-battery) resource whose output this setpoint caps.
    curtailable: bool = False
    # When the VPP started holding this resource's output below its upper
    # limit (continuously, across rewrites); ``None`` while not limiting.
    limit_since: float | None = None

    def limiting(self, kw: float) -> bool:
        return self.curtailable and kw < self.hi_kw - _LIMIT_EPS_KW

    def note_written(self, kw: float, now: float) -> None:
        """Record that *kw* is now in force on the device."""
        if not self.limiting(kw):
            self.limit_since = None
        elif self.limit_since is None:
            self.limit_since = now
        self.setpoint_kw, self.written_at = kw, now

    def to_dict(self) -> dict[str, Any]:
        return {
            "resource_id": self.resource_id,
            "resource_name": self.name,
            "setpoint_kw": self.setpoint_kw,
            "pending_kw": self.pending_kw,
            "written_at": self.written_at,
            "expires_at": self.expires_at,
            "source": self.source,
            "run_id": self.run_id,
            "profile": self.config.profile,
            "simulated": self.config.simulate,
            "limit_since": self.limit_since,
        }


@dataclass(frozen=True)
class OutputLimit:
    """A generation resource's output cap currently held by the VPP.

    Only limits the device accepted (``accepted``, read back when ``verify``
    is on) are reported; simulated control blocks write nothing and are not.
    """

    resource_id: str
    limit_kw: float
    since: float  # epoch seconds: when the VPP started limiting (continuously)


@dataclass
class _Target:
    """What the actuator needs to know about a resource."""

    id: str
    name: str
    resource_type: str
    online: bool
    lo_kw: float
    hi_kw: float
    rated_kw: float
    metadata: dict[str, Any] = field(default_factory=dict)
    is_battery: bool = False
    # Generation resources: the output the optimiser believed available.
    available_kw: float | None = None

    @classmethod
    def from_asset(cls, asset: FleetAsset) -> _Target:
        if asset.is_battery:
            lo, hi = -asset.charge_limit_kw, asset.discharge_limit_kw
        else:
            lo, hi = 0.0, max(0.0, float(asset.rated_power_kw))
        return cls(
            id=asset.id,
            name=asset.name,
            resource_type=asset.resource_type,
            online=bool(asset.online),
            lo_kw=lo,
            hi_kw=hi,
            rated_kw=max(0.0, float(asset.rated_power_kw)),
            metadata=asset.metadata or {},
            is_battery=asset.is_battery,
            available_kw=None if asset.is_battery else asset.available_kw,
        )


class SetpointActuator:
    """Writes dispatch setpoints to opted-in devices, with safety rules + watchdog."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        watchdog_interval_s: float = 5.0,
        expiry_grace_s: float = 30.0,
        session_factory: async_sessionmaker[AsyncSession]
        | Callable[[], async_sessionmaker[AsyncSession]]
        | None = None,
        registry: ProtocolRegistry | None = None,
        writer_factory: Any = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.enabled = bool(enabled)
        self.watchdog_interval_s = max(0.5, float(watchdog_interval_s))
        self.expiry_grace_s = max(0.0, float(expiry_grace_s))
        self._factory = session_factory
        self._registry = registry
        self._writer_factory = writer_factory or self._modbus_writer
        self._clock = clock
        self._lock = asyncio.Lock()
        self._active: dict[str, _Active] = {}
        self._writers: dict[tuple[str, str], SetpointWriter] = {}
        self._own_adapters: dict[str, ModbusAdapter] = {}
        self.last_tick_at: float | None = None
        self.commands = 0
        self.failures = 0

    # -- plumbing ------------------------------------------------------------

    def _sessions(self) -> async_sessionmaker[AsyncSession] | None:
        from sqlalchemy.ext.asyncio import async_sessionmaker

        if isinstance(self._factory, async_sessionmaker):
            return self._factory
        if self._factory is not None:
            return self._factory()
        try:
            from vpp.db.engine import get_session_factory

            return get_session_factory()
        except Exception:
            return None

    def _modbus_writer(
        self, resource_id: str, modbus: dict[str, Any], control: ModbusControlConfig, rated: float
    ) -> SetpointWriter:
        async def provider() -> Any:
            return await self._adapter_for(resource_id, modbus)

        return ModbusSetpointWriter(control, provider, reference_kw=rated)

    async def _adapter_for(self, resource_id: str, modbus: dict[str, Any]) -> Any:
        """The ingestion loop's connected adapter if any, else a private one."""
        from vpp.protocols.modbus import ModbusAdapter
        from vpp.protocols.modbus_ingestion import NON_ADAPTER_KEYS

        if self._registry is not None:
            shared = self._registry.get(f"modbus:{resource_id}")
            if isinstance(shared, ModbusAdapter) and shared.is_connected:
                return shared
        own = self._own_adapters.get(resource_id)
        if own is not None and own.is_connected:
            return own
        adapter = ModbusAdapter()
        adapter.name = f"modbus-control:{resource_id}"
        cfg = {k: v for k, v in modbus.items() if k not in NON_ADAPTER_KEYS}
        cfg["poll_interval_s"] = 0  # writes only; telemetry is the ingestion loop's job
        adapter.configure(**cfg)
        await adapter.connect()
        self._own_adapters[resource_id] = adapter
        return adapter

    def _writer(
        self, target: _Target, modbus: dict[str, Any], control: ModbusControlConfig
    ) -> SetpointWriter:
        key = (target.id, json.dumps(modbus, sort_keys=True, default=str))
        writer = self._writers.get(key)
        if writer is None:
            for k in [k for k in self._writers if k[0] == target.id]:
                del self._writers[k]  # config changed
            writer = self._writer_factory(target.id, modbus, control, target.rated_kw)
            self._writers[key] = writer
        return writer

    # -- dispatch entry point --------------------------------------------------

    async def apply(
        self,
        assets: list[FleetAsset],
        allocations: dict[str, float],
        *,
        source: str,
        run_id: str | None = None,
        ttl_s: float = 900.0,
        session: AsyncSession | None = None,
    ) -> list[dict[str, Any]]:
        """Write each stationary resource's allocation (kW, export-positive).

        Returns one delivery dict per allocated stationary resource. Never
        raises for device errors. When *session* is given the audit rows are
        added to it (the caller commits), else they are committed here.
        """
        deliveries: list[dict[str, Any]] = []
        async with self._lock:
            now = self._clock()
            for asset in assets:
                if asset.id.startswith(EV_PREFIX) or asset.id not in allocations:
                    continue
                target = _Target.from_asset(asset)
                kw = float(allocations[asset.id])
                deliveries.append(
                    await self._apply_one(target, kw, now, source=source, run_id=run_id, ttl=ttl_s)
                )
        await self._record(deliveries, source=source, run_id=run_id, session=session)
        return deliveries

    async def _apply_one(
        self, target: _Target, requested: float, now: float, *, source: str, run_id, ttl: float
    ) -> dict[str, Any]:
        d: dict[str, Any] = {
            "resource_id": target.id,
            "resource_name": target.name,
            "resource_type": target.resource_type,
            "action": "setpoint",
            "requested_kw": requested,
            "setpoint_kw": None,
            "clamped": False,
            "status": NOT_CONFIGURED,
            "reason": "",
            "verified": None,
            "writes": [],
            "expires_at": None,
        }
        modbus = target.metadata.get("modbus")
        try:
            control = ModbusControlConfig.from_modbus_config(
                modbus if isinstance(modbus, dict) else None
            )
        except (ControlConfigError, TypeError, ValueError) as exc:
            d.update(status=FAILED, reason=f"invalid modbus.control config: {exc}")
            return d
        if control is None or not isinstance(modbus, dict):
            d["reason"] = "no modbus.control block on this resource"
            return d
        d["profile"] = control.profile
        if control.unverified:
            d["unverified_profile"] = True
        if not self.enabled:
            d.update(status=DISABLED, reason="global kill switch off (VPP_CONTROL_ENABLED=false)")
            return d
        if not control.enabled:
            d.update(status=DISABLED, reason="modbus.control.enabled is false")
            return d
        if not target.online:
            d.update(status=OFFLINE, reason="resource offline; nothing written")
            return d

        lo, hi = target.lo_kw, target.hi_kw
        if control.max_kw is not None:
            hi = min(hi, control.max_kw)
        if control.min_kw is not None:
            lo = max(lo, control.min_kw)
        if lo > hi:
            d.update(status=FAILED, reason=f"empty setpoint range [{lo:g}, {hi:g}] kW")
            return d
        setpoint = max(lo, min(hi, requested))
        d["clamped"] = abs(setpoint - requested) > 1e-9
        if (
            not target.is_battery
            and target.available_kw is not None
            and target.available_kw > _LIMIT_EPS_KW
            and requested >= target.available_kw - _LIMIT_EPS_KW
            and setpoint < hi
        ):
            # The allocation is everything the resource was believed to have:
            # it is not to be curtailed. Capping it at that estimate would pin
            # the output there (the next poll could never show more), so the
            # cap goes to the upper limit instead (sunspec_123: lifted).
            setpoint = hi
            d["uncurtailed"] = True
        d["setpoint_kw"] = setpoint
        expires = now + max(0.0, ttl) + self.expiry_grace_s
        d["expires_at"] = expires

        active = self._active.get(target.id)
        if active is not None and active.config.raw == control.raw:
            if active.pending_kw is None and abs(setpoint - active.setpoint_kw) <= max(
                control.deadband_kw, 1e-9
            ):
                active.expires_at, active.source, active.run_id = expires, source, run_id
                d.update(status=UNCHANGED, reason=f"within deadband of {active.setpoint_kw:g} kW")
                return d
            wait = control.min_interval_s - (now - active.written_at)
            if wait > 0:
                active.pending_kw = setpoint
                active.expires_at, active.source, active.run_id = expires, source, run_id
                active.lo_kw, active.hi_kw = lo, hi
                d.update(
                    status=DEFERRED, reason=f"rate limit: written by the watchdog in {wait:.1f}s"
                )
                return d

        writer = None
        if control.simulate:
            d.update(status=SIMULATED, reason="simulate=true: no device I/O")
        else:
            writer = self._writer(target, modbus, control)
            result = await writer.write(setpoint)
            self.commands += 1
            d.update(writes=result.writes, verified=result.verified)
            if not result.ok:
                self.failures += 1
                d.update(status=FAILED, reason=result.error or "write failed")
                return d
            d.update(
                status=ACCEPTED, reason="written" + (" and verified" if result.verified else "")
            )
        prev = self._active.get(target.id)
        entry = _Active(
            resource_id=target.id,
            name=target.name,
            config=control,
            writer=writer,
            lo_kw=lo,
            hi_kw=hi,
            setpoint_kw=setpoint,
            written_at=now,
            expires_at=expires,
            source=source,
            run_id=run_id,
            curtailable=not target.is_battery,
            limit_since=prev.limit_since if prev is not None else None,
        )
        entry.note_written(setpoint, now)
        self._active[target.id] = entry
        return d

    # -- fallback / release ----------------------------------------------------

    async def _fallback(self, a: _Active, reason: str, *, action: str) -> dict[str, Any]:
        d: dict[str, Any] = {
            "resource_id": a.resource_id,
            "resource_name": a.name,
            "action": action,
            "requested_kw": None,
            "setpoint_kw": None,
            "status": ACCEPTED,
            "reason": reason,
            "verified": None,
            "writes": [],
            "run_id": a.run_id,
        }
        if a.config.simulate or a.writer is None:
            self._active.pop(a.resource_id, None)
            d.update(status=SIMULATED)
            return d
        if a.config.safe_setpoint_kw is not None:
            safe = max(a.lo_kw, min(a.hi_kw, a.config.safe_setpoint_kw))
            d["setpoint_kw"] = safe
            result = await a.writer.write(safe)
            d["mode"] = "safe_setpoint"
        else:
            result = await a.writer.release()
            d["mode"] = "release"
        self.commands += 1
        d.update(writes=result.writes, verified=result.verified)
        if result.ok:
            self._active.pop(a.resource_id, None)
            return d
        self.failures += 1
        a.release_failures += 1
        d.update(status=FAILED, reason=f"{reason}; {result.error}")
        if a.release_failures >= _MAX_RELEASE_ATTEMPTS:
            self._active.pop(a.resource_id, None)
            d["reason"] += f"; gave up after {a.release_failures} attempts"
            logger.error("could not release control of %s: %s", a.resource_id, result.error)
        return d

    async def release(
        self,
        resource_ids: list[str] | None = None,
        *,
        reason: str,
        source: str = "control",
        run_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Fall back / release *resource_ids* (default: every active setpoint)."""
        async with self._lock:
            ids = list(self._active) if resource_ids is None else resource_ids
            out = [
                await self._fallback(self._active[rid], reason, action="release")
                for rid in ids
                if rid in self._active
            ]
        await self._record(out, source=source, run_id=run_id)
        return out

    # -- watchdog ----------------------------------------------------------------

    async def _online(self, ids: list[str]) -> dict[str, bool]:
        factory = self._sessions()
        if factory is None or not ids:
            return {}
        from sqlalchemy import select

        from vpp.db.models import ResourceModel

        try:
            async with factory() as session:
                rows = await session.execute(
                    select(ResourceModel.id, ResourceModel.online).where(ResourceModel.id.in_(ids))
                )
                return {rid: bool(online) for rid, online in rows.all()}
        except Exception:
            logger.warning("control watchdog could not read resource status", exc_info=True)
            return {}

    async def tick(self) -> list[dict[str, Any]]:
        """Expire, flush deferred and keep-alive setpoints once."""
        async with self._lock:
            now = self._clock()
            self.last_tick_at = now
            if not self._active:
                return []
            online = await self._online(list(self._active))
            out: list[dict[str, Any]] = []
            for rid, a in list(self._active.items()):
                if online.get(rid) is False:
                    # Never write to an offline device; drop local control state.
                    self._active.pop(rid, None)
                    out.append(
                        {
                            "resource_id": rid,
                            "resource_name": a.name,
                            "action": "abandon",
                            "status": OFFLINE,
                            "reason": "resource went offline; control state dropped, "
                            "nothing written",
                            "run_id": a.run_id,
                        }
                    )
                elif now >= a.expires_at:
                    out.append(await self._fallback(a, "dispatch expired", action="expire"))
                elif a.pending_kw is not None and now - a.written_at >= a.config.min_interval_s:
                    out.append(await self._rewrite(a, a.pending_kw, now, action="deferred_write"))
                elif (
                    a.config.keepalive_s is not None
                    and a.writer is not None
                    and now - a.written_at >= a.config.keepalive_s
                ):
                    out.append(await self._rewrite(a, a.setpoint_kw, now, action="keepalive"))
        if out:
            await self._record(out, source="control.watchdog")
        return out

    async def _rewrite(self, a: _Active, kw: float, now: float, *, action: str) -> dict[str, Any]:
        d: dict[str, Any] = {
            "resource_id": a.resource_id,
            "resource_name": a.name,
            "action": action,
            "requested_kw": kw,
            "setpoint_kw": kw,
            "status": SIMULATED,
            "reason": "",
            "verified": None,
            "writes": [],
            "run_id": a.run_id,
        }
        a.pending_kw = None
        if a.writer is not None and not a.config.simulate:
            result = await a.writer.write(kw)
            self.commands += 1
            d.update(writes=result.writes, verified=result.verified)
            if not result.ok:
                self.failures += 1
                d.update(status=FAILED, reason=result.error or "write failed")
                return d
            d["status"] = ACCEPTED
        a.note_written(kw, now)
        return d

    async def run_forever(self) -> None:
        while True:
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("control watchdog tick failed")
            await asyncio.sleep(self.watchdog_interval_s)

    async def shutdown(self) -> list[dict[str, Any]]:
        """Release every active setpoint and close private connections."""
        out: list[dict[str, Any]] = []
        try:
            out = await self.release(reason="API shutting down", source="control.shutdown")
        finally:
            for adapter in self._own_adapters.values():
                with contextlib.suppress(Exception):
                    await adapter.disconnect()
            self._own_adapters.clear()
            self._writers.clear()
        return out

    # -- audit -------------------------------------------------------------------

    async def _record(
        self,
        entries: list[dict[str, Any]],
        *,
        source: str,
        run_id: str | None = None,
        session: AsyncSession | None = None,
    ) -> None:
        logged = [e for e in entries if e.get("status") != NOT_CONFIGURED]
        if not logged:
            return
        at = self._clock()
        rows = [
            (e, {**e, "source": source, "run_id": e.get("run_id") or run_id, "at": at})
            for e in logged
        ]
        try:
            from vpp.db.repositories import EventLogRepository

            async def _write(s: AsyncSession) -> None:
                for e, details in rows:
                    await EventLogRepository.log(
                        s,
                        event_type=EVENT_LOG_TYPE,
                        resource_id=str(e["resource_id"])[:36],
                        details=json.loads(json.dumps(details, default=str)),
                        severity="warning" if e.get("status") == FAILED else "info",
                    )

            if session is not None:
                await _write(session)
            else:
                factory = self._sessions()
                if factory is not None:
                    async with factory() as own:
                        await _write(own)
                        await own.commit()
        except Exception:
            logger.warning("failed to record device setpoint commands", exc_info=True)
        counts: dict[str, int] = {}
        for e in entries:
            counts[e["status"]] = counts.get(e["status"], 0) + 1
        failed = counts.get(FAILED, 0) > 0
        try:
            await get_event_bus().publish(
                Event(
                    event_type=EventType.DEVICE_SETPOINT,
                    data={
                        "source": source,
                        "run_id": run_id,
                        "summary": counts,
                        "deliveries": [d for _, d in rows],
                    },
                    source="control.actuator",
                    severity="warning" if failed else "info",
                )
            )
        except Exception:
            logger.warning("failed to publish DEVICE_SETPOINT", exc_info=True)

    def output_limits(self) -> dict[str, OutputLimit]:
        """Generation resources whose output the VPP currently caps, by id.

        The dispatch optimiser uses this to tell a curtailed reading from the
        resource's real availability (see
        :func:`vpp.api.optimization_support.resource_to_asset`).
        """
        return {
            rid: OutputLimit(rid, a.setpoint_kw, a.limit_since)
            for rid, a in self._active.items()
            if a.limit_since is not None and a.writer is not None and not a.config.simulate
        }

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "watchdog_interval_s": self.watchdog_interval_s,
            "expiry_grace_s": self.expiry_grace_s,
            "last_tick_at": self.last_tick_at,
            "commands": self.commands,
            "failures": self.failures,
            "active": [a.to_dict() for a in self._active.values()],
        }


def summarize(deliveries: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for d in deliveries:
        counts[d["status"]] = counts.get(d["status"], 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Process-wide instance
# ---------------------------------------------------------------------------

_actuator: SetpointActuator | None = None


def _registry_or_none() -> ProtocolRegistry | None:
    try:
        from vpp.api.routes.protocols import get_registry

        return get_registry()
    except Exception:
        return None


def build_actuator(settings: Any) -> SetpointActuator:
    return SetpointActuator(
        enabled=bool(getattr(settings, "control_enabled", False)),
        watchdog_interval_s=float(getattr(settings, "control_watchdog_interval_s", 5.0)),
        expiry_grace_s=float(getattr(settings, "control_expiry_grace_s", 30.0)),
        registry=_registry_or_none(),
    )


def get_setpoint_actuator() -> SetpointActuator:
    """The process-wide actuator (created from settings on first use)."""
    global _actuator
    if _actuator is None:
        from vpp.settings import get_settings

        _actuator = build_actuator(get_settings())
    return _actuator


def active_output_limits() -> dict[str, OutputLimit]:
    """Output caps held by this process's actuator (``{}`` when there is none).

    Never creates the actuator. The state is per process: another API worker
    or a restarted process does not see these limits (on a clean shutdown
    they are released; after a crash the device's revert timer applies).
    """
    return _actuator.output_limits() if _actuator is not None else {}


def set_setpoint_actuator(actuator: SetpointActuator | None) -> None:
    global _actuator
    _actuator = actuator


def start_control(settings: Any) -> asyncio.Task | None:
    """Install the actuator and, when enabled, start its watchdog task."""
    actuator = build_actuator(settings)
    set_setpoint_actuator(actuator)
    if not actuator.enabled:
        return None
    return asyncio.create_task(actuator.run_forever(), name="vpp-control-watchdog")


async def stop_control(task: asyncio.Task | None) -> None:
    """Stop the watchdog and release every active setpoint (fallback on shutdown)."""
    if task is not None:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    actuator = _actuator
    if actuator is not None:
        try:
            await actuator.shutdown()
        except Exception:
            logger.exception("control actuator shutdown failed")


__all__ = [
    "ACCEPTED",
    "DEFERRED",
    "DISABLED",
    "EVENT_LOG_TYPE",
    "FAILED",
    "NOT_CONFIGURED",
    "OFFLINE",
    "SIMULATED",
    "UNCHANGED",
    "OutputLimit",
    "SetpointActuator",
    "active_output_limits",
    "build_actuator",
    "get_setpoint_actuator",
    "set_setpoint_actuator",
    "start_control",
    "stop_control",
    "summarize",
]
