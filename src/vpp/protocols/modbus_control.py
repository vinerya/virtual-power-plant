"""Modbus setpoint writer: turn a dispatch allocation into register writes.

This is the device side of :mod:`vpp.control.actuator`. A resource opts in
per device by adding a ``control`` block to the Modbus config it already
keeps in its ``metadata`` (see :mod:`vpp.protocols.modbus_ingestion`)::

    {
      "modbus": {
        "mode": "tcp", "host": "192.168.1.50", "port": 502, "unit_id": 1,
        "device_profile": "fronius_symo",
        "control": {
          "enabled": true,
          "profile": "sunspec_123",       # or "register" / "sunspec_124"
          "model_base": 40236,            # address of the model's ID register
          "revert_timeout_s": 900,        # device-side fallback (WMaxLimPct_RvrtTms)
          "min_kw": 0, "max_kw": 8,       # extra clamp inside the resource limits
          "deadband_kw": 0.2, "min_interval_s": 10,
          "verify": true, "safe_setpoint_kw": null
        }
      }
    }

Profiles
--------
``register`` (generic)
    One signed setpoint register. ``register`` names a map/custom register
    that is flagged ``writable`` (or give ``address`` + ``data_type``);
    ``unit`` is ``"W"``, ``"kW"`` or ``"pct"`` (percent of
    ``reference_kw``, default the resource's rated power); ``scale`` is the
    engineering value of one raw count (``raw = value / scale``) or
    ``scale_factor_register`` names/addresses a SunSpec-style int16
    exponent read from the device; ``sign`` is ``"export_positive"``
    (default: positive = discharge/generate) or ``"import_positive"``.
    Optional ``enable_register`` (+ ``enable_value``, ``disable_value``) is
    written before each setpoint and set to ``disable_value`` on release;
    without it, release writes ``release_value`` (default 100 for ``pct``,
    otherwise 0).
``sunspec_123``
    SunSpec model 123 Immediate Controls: ``WMaxLimPct`` = setpoint as % of
    ``reference_kw`` (curtailment; negative setpoints clamp to 0 %),
    ``WMaxLim_Ena`` = 1; release sets ``WMaxLim_Ena`` = 0. The scale factor
    is read from ``WMaxLimPct_SF``. ``revert_timeout_s`` programs
    ``WMaxLimPct_RvrtTms`` so the inverter reverts on its own if the VPP
    goes silent.
``sunspec_124`` (**generic/unverified**)
    SunSpec model 124 Storage: ``OutWRte``/``InWRte`` as % of
    ``reference_kw`` (forced discharge: ``OutWRte`` = p, ``InWRte`` = -p;
    forced charge the reverse), ``StorCtl_Mod`` = 3; release sets
    ``StorCtl_Mod`` = 0. Vendors interpret these registers differently;
    validate against your device before enabling.

Register addresses are what pymodbus sends on the wire (0-based). Vendor
documentation often lists 1-based register *numbers* -- subtract one.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from vpp.protocols.modbus import (
    RegisterDefinition,
    RegisterType,
    encode_value,
    register_words,
    sunspec_model_123_registers,
    sunspec_model_124_registers,
)

logger = logging.getLogger(__name__)

PROFILES = ("register", "sunspec_123", "sunspec_124")
UNVERIFIED_PROFILES = frozenset({"sunspec_124"})


class ControlConfigError(ValueError):
    """The resource's ``modbus.control`` block is invalid."""


class RegisterIO(Protocol):
    """What the writer needs from a (connected) Modbus adapter."""

    @property
    def is_connected(self) -> bool: ...

    def register(self, name: str) -> RegisterDefinition | None: ...

    async def write_registers(
        self, address: int, values: list[int], *, unit: int | None = None
    ) -> None: ...

    async def read_holding(
        self, address: int, count: int = 1, *, unit: int | None = None
    ) -> list[int]: ...


def _opt_float(cfg: dict[str, Any], key: str) -> float | None:
    v = cfg.get(key)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError) as exc:
        raise ControlConfigError(f"control.{key} must be a number") from exc


@dataclass
class ModbusControlConfig:
    """Parsed ``metadata["modbus"]["control"]``."""

    enabled: bool = False
    profile: str = "register"
    simulate: bool = False
    # generic register profile
    register: str | None = None
    address: int | None = None
    data_type: str = "int16"
    unit: str = "W"
    scale: float = 1.0
    scale_factor_register: str | int | None = None
    sign: str = "export_positive"
    enable_register: str | int | None = None
    enable_value: int = 1
    disable_value: int = 0
    release_value: float | None = None
    # SunSpec profiles
    model_base: int | None = None
    revert_timeout_s: int | None = None
    reference_kw: float | None = None
    unit_id: int | None = None
    # safety (enforced by vpp.control.actuator)
    min_kw: float | None = None
    max_kw: float | None = None
    deadband_kw: float = 0.1
    min_interval_s: float = 5.0
    verify: bool = True
    safe_setpoint_kw: float | None = None
    keepalive_s: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def unverified(self) -> bool:
        return self.profile in UNVERIFIED_PROFILES

    @classmethod
    def from_modbus_config(cls, modbus: dict[str, Any] | None) -> ModbusControlConfig | None:
        """Parse the ``control`` block of a resource's Modbus config (``None`` if absent)."""
        if not isinstance(modbus, dict):
            return None
        cfg = modbus.get("control")
        if cfg is None:
            return None
        if not isinstance(cfg, dict):
            raise ControlConfigError("modbus.control must be an object")
        profile = str(cfg.get("profile", "register"))
        if profile not in PROFILES:
            raise ControlConfigError(f"control.profile must be one of {PROFILES}")
        out = cls(
            enabled=bool(cfg.get("enabled", False)),
            profile=profile,
            simulate=bool(cfg.get("simulate", False)),
            register=cfg.get("register"),
            address=int(cfg["address"]) if cfg.get("address") is not None else None,
            data_type=str(cfg.get("data_type", "int16")),
            unit=str(cfg.get("unit", "W")),
            scale=float(cfg.get("scale", 1.0)),
            scale_factor_register=cfg.get("scale_factor_register"),
            sign=str(cfg.get("sign", "export_positive")),
            enable_register=cfg.get("enable_register"),
            enable_value=int(cfg.get("enable_value", 1)),
            disable_value=int(cfg.get("disable_value", 0)),
            release_value=_opt_float(cfg, "release_value"),
            model_base=int(cfg["model_base"]) if cfg.get("model_base") is not None else None,
            revert_timeout_s=int(cfg["revert_timeout_s"])
            if cfg.get("revert_timeout_s") is not None
            else None,
            reference_kw=_opt_float(cfg, "reference_kw"),
            unit_id=int(cfg["unit_id"])
            if cfg.get("unit_id") is not None
            else (int(modbus["unit_id"]) if modbus.get("unit_id") is not None else None),
            min_kw=_opt_float(cfg, "min_kw"),
            max_kw=_opt_float(cfg, "max_kw"),
            deadband_kw=max(0.0, float(cfg.get("deadband_kw", 0.1))),
            min_interval_s=max(0.0, float(cfg.get("min_interval_s", 5.0))),
            verify=bool(cfg.get("verify", True)),
            safe_setpoint_kw=_opt_float(cfg, "safe_setpoint_kw"),
            keepalive_s=_opt_float(cfg, "keepalive_s"),
            raw=dict(cfg),
        )
        if out.unit not in ("W", "kW", "pct"):
            raise ControlConfigError("control.unit must be W, kW or pct")
        if out.sign not in ("export_positive", "import_positive"):
            raise ControlConfigError("control.sign must be export_positive or import_positive")
        if out.scale == 0:
            raise ControlConfigError("control.scale must be non-zero")
        if profile == "register" and out.register is None and out.address is None:
            raise ControlConfigError("control.register or control.address is required")
        if profile != "register" and out.model_base is None:
            raise ControlConfigError(f"control.model_base is required for {profile}")
        if out.revert_timeout_s is not None and out.keepalive_s is None:
            # Refresh the device-side revert timer well before it fires.
            out.keepalive_s = max(1.0, out.revert_timeout_s / 2.0)
        return out


@dataclass
class WriteResult:
    ok: bool
    writes: list[dict[str, Any]] = field(default_factory=list)
    verified: bool | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "writes": self.writes,
            "verified": self.verified,
            "error": self.error,
        }


@dataclass
class _Write:
    label: str
    address: int
    data_type: str
    raw_value: float  # already scaled (register counts)
    value: float  # engineering value, for the audit record


AdapterProvider = Callable[[], Awaitable[RegisterIO]]


class ModbusSetpointWriter:
    """Writes one resource's power setpoint (kW, export-positive) to its device."""

    def __init__(
        self,
        config: ModbusControlConfig,
        adapter_provider: AdapterProvider,
        *,
        reference_kw: float,
    ) -> None:
        self.config = config
        self._provider = adapter_provider
        self.reference_kw = float(config.reference_kw or reference_kw or 0.0)
        self._scale: float | None = None
        base = config.model_base or 0
        if config.profile == "sunspec_123":
            self._regs = sunspec_model_123_registers(base)
        elif config.profile == "sunspec_124":
            self._regs = sunspec_model_124_registers(base)
        else:
            self._regs = {}

    # -- register resolution -------------------------------------------------

    def _resolve(self, io: RegisterIO, ref: str | int, *, write: bool) -> RegisterDefinition:
        if isinstance(ref, int) or (isinstance(ref, str) and ref.isdigit()):
            return RegisterDefinition(
                int(ref), 1, RegisterType.HOLDING, str(ref), "", 1.0, "uint16", writable=True
            )
        reg = self._regs.get(ref) or io.register(ref)
        if reg is None:
            raise ControlConfigError(f"unknown Modbus register {ref!r}")
        if write and not reg.writable:
            raise ControlConfigError(f"register {ref!r} is not flagged writable")
        return reg

    async def _read_sf(self, io: RegisterIO, ref: str | int) -> float:
        reg = self._resolve(io, ref, write=False)
        [raw] = await io.read_holding(reg.address, 1, unit=self.config.unit_id)
        sf = raw - 0x10000 if raw >= 0x8000 else raw
        if sf == -0x8000 or not -10 <= sf <= 10:  # 0x8000 = SunSpec "not implemented"
            raise ControlConfigError(f"invalid scale factor {sf} in register {ref!r}")
        return float(10.0**sf)

    async def _scale_for(self, io: RegisterIO) -> float:
        if self._scale is not None:
            return self._scale
        cfg = self.config
        if cfg.profile == "sunspec_123":
            self._scale = await self._read_sf(io, "wmax_lim_pct_sf")
        elif cfg.profile == "sunspec_124":
            self._scale = await self._read_sf(io, "in_out_w_rte_sf")
        elif cfg.scale_factor_register is not None:
            self._scale = await self._read_sf(io, cfg.scale_factor_register)
        else:
            self._scale = cfg.scale
        return self._scale

    def _pct(self, kw: float) -> float:
        if self.reference_kw <= 0:
            raise ControlConfigError("reference_kw (or the resource's rated power) must be > 0")
        return max(-100.0, min(100.0, kw / self.reference_kw * 100.0))

    # -- plans ---------------------------------------------------------------

    async def _plan(self, io: RegisterIO, kw: float | None) -> list[_Write]:
        """Writes for setpoint *kw*; ``None`` plans the release sequence."""
        cfg = self.config
        scale = await self._scale_for(io)
        writes: list[_Write] = []

        def add(ref: str | int, value: float, *, scaled: bool) -> None:
            reg = self._resolve(io, ref, write=True)
            raw = value / scale if scaled else value
            writes.append(_Write(reg.name or str(ref), reg.address, reg.data_type, raw, value))

        if cfg.profile == "sunspec_123":
            if kw is None:
                add("wmax_lim_ena", 0, scaled=False)
                return writes
            if cfg.revert_timeout_s is not None:
                add("wmax_lim_pct_rvrt_tms", cfg.revert_timeout_s, scaled=False)
            add("wmax_lim_pct", max(0.0, self._pct(kw)), scaled=True)
            add("wmax_lim_ena", 1, scaled=False)
            return writes

        if cfg.profile == "sunspec_124":
            if kw is None:
                add("stor_ctl_mod", 0, scaled=False)
                return writes
            pct = abs(self._pct(kw))
            out_rte, in_rte = (pct, -pct) if kw > 0 else ((-pct, pct) if kw < 0 else (0.0, 0.0))
            if cfg.revert_timeout_s is not None:
                add("in_out_w_rte_rvrt_tms", cfg.revert_timeout_s, scaled=False)
            add("out_w_rte", out_rte, scaled=True)
            add("in_w_rte", in_rte, scaled=True)
            add("stor_ctl_mod", 3, scaled=False)
            return writes

        # generic single-register profile
        if kw is None:
            if cfg.enable_register is not None:
                add(cfg.enable_register, cfg.disable_value, scaled=False)
                return writes
            release = cfg.release_value
            if release is None:
                release = 100.0 if cfg.unit == "pct" else 0.0
            self._add_setpoint(writes, io, release, scale)
            return writes
        sign = 1.0 if cfg.sign == "export_positive" else -1.0
        if cfg.unit == "W":
            value = kw * 1000.0 * sign
        elif cfg.unit == "kW":
            value = kw * sign
        else:
            value = self._pct(kw) * sign
        if cfg.enable_register is not None:
            add(cfg.enable_register, cfg.enable_value, scaled=False)
        self._add_setpoint(writes, io, value, scale)
        return writes

    def _add_setpoint(
        self, writes: list[_Write], io: RegisterIO, value: float, scale: float
    ) -> None:
        cfg = self.config
        if cfg.register is not None:
            reg = self._resolve(io, cfg.register, write=True)
            label, address, dtype = reg.name or cfg.register, reg.address, reg.data_type
        else:
            label, address, dtype = f"register {cfg.address}", int(cfg.address or 0), cfg.data_type
        writes.append(_Write(label, address, dtype, value / scale, value))

    # -- execution -----------------------------------------------------------

    async def _execute(self, kw: float | None) -> WriteResult:
        record: list[dict[str, Any]] = []
        try:
            io = await self._provider()
            if not io.is_connected:
                return WriteResult(False, error="Modbus device not connected")
            plan = await self._plan(io, kw)
            encoded = [(w, encode_value(w.raw_value, w.data_type)) for w in plan]
            for w, regs in encoded:
                await io.write_registers(w.address, regs, unit=self.config.unit_id)
                record.append(
                    {"register": w.label, "address": w.address, "value": w.value, "raw": regs}
                )
            verified: bool | None = None
            if self.config.verify:
                for w, regs in encoded:
                    back = await io.read_holding(
                        w.address, register_words(w.data_type), unit=self.config.unit_id
                    )
                    if list(back) != regs:
                        return WriteResult(
                            False,
                            record,
                            verified=False,
                            error=(
                                f"read-back mismatch at {w.label} ({w.address}): "
                                f"wrote {regs}, read {list(back)}"
                            ),
                        )
                verified = True
            return WriteResult(True, record, verified=verified)
        except Exception as exc:  # device I/O must never take the dispatch down
            return WriteResult(False, record, error=f"{type(exc).__name__}: {exc}")

    async def write(self, kw: float) -> WriteResult:
        """Write setpoint *kw* (export-positive, already clamped by the caller)."""
        return await self._execute(float(kw))

    async def release(self) -> WriteResult:
        """Hand control back to the device (profile-specific release sequence)."""
        return await self._execute(None)


__all__ = [
    "PROFILES",
    "UNVERIFIED_PROFILES",
    "ControlConfigError",
    "ModbusControlConfig",
    "ModbusSetpointWriter",
    "RegisterIO",
    "WriteResult",
]
