"""SunSpec Modbus TCP device simulator (PV hybrid inverter + battery).

Serves a SunSpec register map over Modbus TCP (pymodbus) so the read and
control paths of the VPP can be exercised end to end without hardware::

    vpp simulate sunspec --port 5020                  # or:
    python -m vpp.simulators.sunspec --port 5020 --inverter-model 113

Register map (defaults; addresses are 0-based wire addresses)::

    40000  "SunS" marker
    40002  model 1    common (Mn/Md/Opt/Vr/SN strings, DA)   L = 65 (or 66)
    40069  model 103  three-phase inverter, integer + scale factors  L = 50
           (or model 113, float32, L = 60)
    40121  model 123  immediate controls (WMaxLimPct, WMaxLim_Ena, Conn, ...)
    40147  model 124  basic storage (StorCtl_Mod, OutWRte/InWRte, ChaState, ...)
    40173  end model  0xFFFF, L = 0

(with model 113 the controls start at 40131 and storage at 40157). Common
model length 65 is what SolarEdge and Fronius devices report (the Pad
register of model 1 omitted), which is what the ``solaredge_se`` /
``fronius_symo`` polling maps assume; ``--common-length 66`` serves the full
model including Pad, which moves every following model by one register --
use ``model_base: "auto"`` discovery then.

The point layouts below are transcribed from the SunSpec information model
(the JSON definitions bundled with pysunspec2, checked point by point in
``tests/test_sunspec_simulator.py`` when pysunspec2 is installed); nothing
here needs pysunspec2 at runtime.

Behaviour (deliberately simple physics):

* PV: ``pv_available_w`` (constant, settable at runtime) is produced unless
  curtailed. Model 123 ``WMaxLimPct`` (% of ``wmax_w``) caps the AC output
  while ``WMaxLim_Ena`` = 1; PV is curtailed first. ``Conn`` = 0 disconnects
  (no output, battery idle).
* Battery: model 124 ``StorCtl_Mod`` bit 0 makes ``InWRte`` (% of
  ``WChaMax``) the charge limit, bit 1 makes ``OutWRte`` (% of the discharge
  maximum, taken to equal ``WChaMax``; model 124 has no ``WDisChaMax``) the
  discharge limit. The battery power is the idle point (0 W) clamped into
  ``[-OutWRte, +InWRte]``, so negative rates force the opposite direction:
  ``OutWRte = p, InWRte = -p`` discharges at ``p`` %, ``OutWRte = -p,
  InWRte = p`` charges at ``p`` %. With ``StorCtl_Mod`` = 0 the battery idles.
  ``ChaGriSet`` = 0 (PV) limits charging to the available PV power.
* SoC integrates battery power with a one-way efficiency, stops at 100 %
  and at the ``MinRsvPct`` reserve.
* Revert timers: ``WMaxLimPct_RvrtTms`` clears ``WMaxLim_Ena``,
  ``InOutWRte_RvrtTms`` clears ``StorCtl_Mod`` and ``Conn_RvrtTms``
  reconnects, each counted from the last write to that control. Window
  (``*_WinTms``) and ramp (``*_RmpTms``) times are stored but not modelled:
  set points take effect on the next simulation step.
* Writes are validated like a strict device: read-only points and addresses
  outside the map answer ILLEGAL DATA ADDRESS, out-of-range values (e.g.
  ``WMaxLimPct`` > 100 %, ``StorCtl_Mod`` > 3, rates beyond +/-100 %)
  ILLEGAL DATA VALUE; nothing of a rejected request is applied. (Checked
  with pymodbus 3.6, 3.9 and 3.11-3.15. Before 3.10 the server cannot answer
  a rejected *write* with an exception: it is dropped silently, which
  read-back verification shows.)
* Optional points the simulator does not model hold the SunSpec "not
  implemented" value (0x8000 int16/sunssf, 0xFFFF uint16/enum16,
  0xFFFFFFFF bitfield32, NaN float32).

What this does **not** prove: vendor firmware differs from the spec in
exactly the places that matter for control (how forced charge/discharge via
model 124 is interpreted, ramping, which points are implemented, write
access and persistence, unit IDs, dynamic scale factors). Passing against
this simulator shows that the VPP implements the specification; validate
against the real device before enabling control on it.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import importlib
import inspect
import logging
import math
import struct
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vpp.protocols.modbus import (
    SUNSPEC_BASE_ADDRESSES,
    SUNSPEC_END_MODEL_ID,
    SUNSPEC_MARKER,
    SUNSPEC_NOT_IMPLEMENTED,
    encode_value,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

# Modbus exception codes returned for rejected requests.
ILLEGAL_ADDRESS = 0x02
ILLEGAL_VALUE = 0x03

# ---------------------------------------------------------------------------
# SunSpec point layouts (name, SunSpec type, size, scale-factor point, RW)
# ---------------------------------------------------------------------------

Point = tuple[str, str, int, str | None, bool]

_HEADER: tuple[Point, ...] = (("ID", "uint16", 1, None, False), ("L", "uint16", 1, None, False))

_EVENTS: tuple[Point, ...] = (
    ("Evt1", "bitfield32", 2, None, False),
    ("Evt2", "bitfield32", 2, None, False),
    ("EvtVnd1", "bitfield32", 2, None, False),
    ("EvtVnd2", "bitfield32", 2, None, False),
    ("EvtVnd3", "bitfield32", 2, None, False),
    ("EvtVnd4", "bitfield32", 2, None, False),
)


def _float_points(names: Sequence[str]) -> tuple[Point, ...]:
    return tuple((n, "float32", 2, None, False) for n in names)


MODEL_POINTS: dict[int, tuple[Point, ...]] = {
    1: (
        *_HEADER,
        ("Mn", "string", 16, None, False),
        ("Md", "string", 16, None, False),
        ("Opt", "string", 8, None, False),
        ("Vr", "string", 8, None, False),
        ("SN", "string", 16, None, False),
        ("DA", "uint16", 1, None, True),
        ("Pad", "pad", 1, None, False),
    ),
    103: (
        *_HEADER,
        ("A", "uint16", 1, "A_SF", False),
        ("AphA", "uint16", 1, "A_SF", False),
        ("AphB", "uint16", 1, "A_SF", False),
        ("AphC", "uint16", 1, "A_SF", False),
        ("A_SF", "sunssf", 1, None, False),
        ("PPVphAB", "uint16", 1, "V_SF", False),
        ("PPVphBC", "uint16", 1, "V_SF", False),
        ("PPVphCA", "uint16", 1, "V_SF", False),
        ("PhVphA", "uint16", 1, "V_SF", False),
        ("PhVphB", "uint16", 1, "V_SF", False),
        ("PhVphC", "uint16", 1, "V_SF", False),
        ("V_SF", "sunssf", 1, None, False),
        ("W", "int16", 1, "W_SF", False),
        ("W_SF", "sunssf", 1, None, False),
        ("Hz", "uint16", 1, "Hz_SF", False),
        ("Hz_SF", "sunssf", 1, None, False),
        ("VA", "int16", 1, "VA_SF", False),
        ("VA_SF", "sunssf", 1, None, False),
        ("VAr", "int16", 1, "VAr_SF", False),
        ("VAr_SF", "sunssf", 1, None, False),
        ("PF", "int16", 1, "PF_SF", False),
        ("PF_SF", "sunssf", 1, None, False),
        ("WH", "acc32", 2, "WH_SF", False),
        ("WH_SF", "sunssf", 1, None, False),
        ("DCA", "uint16", 1, "DCA_SF", False),
        ("DCA_SF", "sunssf", 1, None, False),
        ("DCV", "uint16", 1, "DCV_SF", False),
        ("DCV_SF", "sunssf", 1, None, False),
        ("DCW", "int16", 1, "DCW_SF", False),
        ("DCW_SF", "sunssf", 1, None, False),
        ("TmpCab", "int16", 1, "Tmp_SF", False),
        ("TmpSnk", "int16", 1, "Tmp_SF", False),
        ("TmpTrns", "int16", 1, "Tmp_SF", False),
        ("TmpOt", "int16", 1, "Tmp_SF", False),
        ("Tmp_SF", "sunssf", 1, None, False),
        ("St", "enum16", 1, None, False),
        ("StVnd", "enum16", 1, None, False),
        *_EVENTS,
    ),
    113: (
        *_HEADER,
        *_float_points(
            (
                "A",
                "AphA",
                "AphB",
                "AphC",
                "PPVphAB",
                "PPVphBC",
                "PPVphCA",
                "PhVphA",
                "PhVphB",
                "PhVphC",
                "W",
                "Hz",
                "VA",
                "VAr",
                "PF",
                "WH",
                "DCA",
                "DCV",
                "DCW",
                "TmpCab",
                "TmpSnk",
                "TmpTrns",
                "TmpOt",
            )
        ),
        ("St", "enum16", 1, None, False),
        ("StVnd", "enum16", 1, None, False),
        *_EVENTS,
    ),
    123: (
        *_HEADER,
        ("Conn_WinTms", "uint16", 1, None, True),
        ("Conn_RvrtTms", "uint16", 1, None, True),
        ("Conn", "enum16", 1, None, True),
        ("WMaxLimPct", "uint16", 1, "WMaxLimPct_SF", True),
        ("WMaxLimPct_WinTms", "uint16", 1, None, True),
        ("WMaxLimPct_RvrtTms", "uint16", 1, None, True),
        ("WMaxLimPct_RmpTms", "uint16", 1, None, True),
        ("WMaxLim_Ena", "enum16", 1, None, True),
        ("OutPFSet", "int16", 1, "OutPFSet_SF", True),
        ("OutPFSet_WinTms", "uint16", 1, None, True),
        ("OutPFSet_RvrtTms", "uint16", 1, None, True),
        ("OutPFSet_RmpTms", "uint16", 1, None, True),
        ("OutPFSet_Ena", "enum16", 1, None, True),
        ("VArWMaxPct", "int16", 1, "VArPct_SF", True),
        ("VArMaxPct", "int16", 1, "VArPct_SF", True),
        ("VArAvalPct", "int16", 1, "VArPct_SF", True),
        ("VArPct_WinTms", "uint16", 1, None, True),
        ("VArPct_RvrtTms", "uint16", 1, None, True),
        ("VArPct_RmpTms", "uint16", 1, None, True),
        ("VArPct_Mod", "enum16", 1, None, True),
        ("VArPct_Ena", "enum16", 1, None, True),
        ("WMaxLimPct_SF", "sunssf", 1, None, False),
        ("OutPFSet_SF", "sunssf", 1, None, False),
        ("VArPct_SF", "sunssf", 1, None, False),
    ),
    124: (
        *_HEADER,
        ("WChaMax", "uint16", 1, "WChaMax_SF", True),
        ("WChaGra", "uint16", 1, "WChaDisChaGra_SF", True),
        ("WDisChaGra", "uint16", 1, "WChaDisChaGra_SF", True),
        ("StorCtl_Mod", "bitfield16", 1, None, True),
        ("VAChaMax", "uint16", 1, "VAChaMax_SF", True),
        ("MinRsvPct", "uint16", 1, "MinRsvPct_SF", True),
        ("ChaState", "uint16", 1, "ChaState_SF", False),
        ("StorAval", "uint16", 1, "StorAval_SF", False),
        ("InBatV", "uint16", 1, "InBatV_SF", False),
        ("ChaSt", "enum16", 1, None, False),
        ("OutWRte", "int16", 1, "InOutWRte_SF", True),
        ("InWRte", "int16", 1, "InOutWRte_SF", True),
        ("InOutWRte_WinTms", "uint16", 1, None, True),
        ("InOutWRte_RvrtTms", "uint16", 1, None, True),
        ("InOutWRte_RmpTms", "uint16", 1, None, True),
        ("ChaGriSet", "enum16", 1, None, True),
        ("WChaMax_SF", "sunssf", 1, None, False),
        ("WChaDisChaGra_SF", "sunssf", 1, None, False),
        ("VAChaMax_SF", "sunssf", 1, None, False),
        ("MinRsvPct_SF", "sunssf", 1, None, False),
        ("ChaState_SF", "sunssf", 1, None, False),
        ("StorAval_SF", "sunssf", 1, None, False),
        ("InBatV_SF", "sunssf", 1, None, False),
        ("InOutWRte_SF", "sunssf", 1, None, False),
    ),
}

INVERTER_MODELS = (103, 113)

# SunSpec type -> the project's wire data type (vpp.protocols.modbus).
_WIRE_TYPE = {
    "uint16": "uint16",
    "enum16": "uint16",
    "bitfield16": "uint16",
    "int16": "int16",
    "sunssf": "int16",
    "pad": "int16",
    "acc32": "acc32",
    "uint32": "uint32",
    "bitfield32": "uint32",
    "float32": "float32",
}

# Scale factors served by the simulator (exponent; None = not implemented).
SCALE_FACTORS: dict[int, dict[str, int | None]] = {
    103: {
        "A_SF": -2,
        "V_SF": -1,
        "W_SF": 0,
        "Hz_SF": -2,
        "VA_SF": 0,
        "VAr_SF": 0,
        "PF_SF": -1,
        "WH_SF": 0,
        "DCA_SF": -2,
        "DCV_SF": -1,
        "DCW_SF": 0,
        "Tmp_SF": -1,
    },
    123: {"WMaxLimPct_SF": -1, "OutPFSet_SF": -3, "VArPct_SF": None},
    124: {
        "WChaMax_SF": 0,
        "WChaDisChaGra_SF": 0,
        "VAChaMax_SF": None,
        "MinRsvPct_SF": 0,
        "ChaState_SF": -1,
        "StorAval_SF": -2,
        "InBatV_SF": -1,
        "InOutWRte_SF": -2,
    },
}

# Inverter operating states (model 103/113 ``St``) and storage ``ChaSt``.
ST_SLEEPING, ST_MPPT, ST_THROTTLED, ST_STANDBY = 2, 4, 5, 8
CHA_EMPTY, CHA_DISCHARGING, CHA_CHARGING, CHA_FULL, CHA_HOLDING = 2, 3, 4, 5, 6


def model_length(model_id: int, *, common_length: int = 65) -> int:
    """``L`` of *model_id* as served (points after ``ID`` and ``L``)."""
    if model_id == 1:
        return common_length
    return sum(p[2] for p in MODEL_POINTS[model_id]) - 2


def _not_implemented(sunspec_type: str, size: int) -> list[int]:
    if sunspec_type == "string":
        return [0] * size
    if sunspec_type == "float32":
        return [0x7FC0, 0x0000]  # NaN
    wire = _WIRE_TYPE[sunspec_type]
    raw = SUNSPEC_NOT_IMPLEMENTED["uint32" if sunspec_type == "bitfield32" else wire]
    if sunspec_type == "pad":
        raw = SUNSPEC_NOT_IMPLEMENTED["int16"]
    return [(raw >> (16 * (size - 1 - i))) & 0xFFFF for i in range(size)]


def _encode_string(text: str, size: int) -> list[int]:
    data = text.encode("ascii", "replace")[: size * 2].ljust(size * 2, b"\0")
    return [(data[2 * i] << 8) | data[2 * i + 1] for i in range(size)]


def _decode(regs: list[int], sunspec_type: str) -> float:
    if sunspec_type == "float32":
        return float(struct.unpack(">f", struct.pack(">HH", regs[0], regs[1]))[0])
    raw = 0
    for r in regs:
        raw = (raw << 16) | r
    if _WIRE_TYPE.get(sunspec_type) == "int16" and raw >= 0x8000:
        raw -= 0x10000
    return float(raw)


# ---------------------------------------------------------------------------
# Configuration + device model (no I/O)
# ---------------------------------------------------------------------------


@dataclass
class SimulatorConfig:
    """What the simulated device looks like."""

    inverter_model: int = 103  # 103 (integer + scale factors) or 113 (float32)
    unit_id: int = 1
    base_address: int = SUNSPEC_BASE_ADDRESSES[0]  # where "SunS" sits
    common_length: int = 65  # 65 (as SolarEdge/Fronius report) or 66 (with Pad)
    manufacturer: str = "VPP Simulator"
    model_name: str = ""  # default: "SunSpec sim <inverter model>"
    options: str = ""
    version: str = "1.0"
    serial: str = "SIM-0001"
    wmax_w: float = 10_000.0  # inverter AC rating (WMaxLimPct is % of this)
    pv_available_w: float = 6_000.0  # PV power available before curtailment
    battery_capacity_wh: float = 10_000.0
    battery_max_w: float = 5_000.0  # WChaMax (charge = discharge maximum)
    soc_pct: float = 50.0
    min_reserve_pct: float = 10.0  # MinRsvPct
    battery_efficiency: float = 0.95  # one way
    grid_v: float = 230.0  # phase-to-neutral
    grid_hz: float = 50.0
    energy_wh: float = 1_000_000.0  # lifetime export counter at start

    def validate(self) -> None:
        if self.inverter_model not in INVERTER_MODELS:
            raise ValueError(f"inverter_model must be one of {INVERTER_MODELS}")
        if self.common_length not in (65, 66):
            raise ValueError("common_length must be 65 or 66")
        if not 1 <= self.unit_id <= 247:
            raise ValueError("unit_id must be 1..247")
        if not 0 < self.wmax_w <= 32767:  # model 103 W is int16 at W_SF 0
            raise ValueError("wmax_w must be in (0, 32767]")
        if not 0 < self.battery_max_w <= 65535 or self.battery_capacity_wh <= 0:
            raise ValueError("battery_max_w must be in (0, 65535] and capacity > 0")
        if not 0 <= self.soc_pct <= 100 or not 0 <= self.min_reserve_pct < 100:
            raise ValueError("soc_pct must be 0..100 and min_reserve_pct 0..<100")
        if not 0 < self.battery_efficiency <= 1:
            raise ValueError("battery_efficiency must be in (0, 1]")


@dataclass(frozen=True)
class ModelLayout:
    model_id: int
    address: int  # wire address of the model's ID register
    length: int


class SunSpecDevice:
    """Register image + physics of the simulated device (transport independent).

    ``read``/``write`` take 0-based wire addresses and return a Modbus
    exception code on rejection. ``step(dt)`` advances the simulation by
    *dt* seconds of device time.
    """

    def __init__(self, config: SimulatorConfig | None = None) -> None:
        cfg = config or SimulatorConfig()
        cfg.validate()
        self.config = cfg
        self.inverter_model = cfg.inverter_model
        self.now = 0.0  # simulated seconds since start
        self.pv_available_w = float(cfg.pv_available_w)
        self.ac_power_w = 0.0
        self.pv_power_w = 0.0
        self.battery_power_w = 0.0  # discharge positive
        self.battery_energy_wh = cfg.battery_capacity_wh * cfg.soc_pct / 100.0
        self.energy_wh = float(cfg.energy_wh)
        self.writes = 0
        self.reverts: list[tuple[float, str]] = []
        self._written_at: dict[str, float] = {}

        # Layout: marker, models, end model.
        self.models: list[ModelLayout] = []
        addr = cfg.base_address + 2
        for mid in (1, cfg.inverter_model, 123, 124):
            length = model_length(mid, common_length=cfg.common_length)
            self.models.append(ModelLayout(mid, addr, length))
            addr += length + 2
        self.end_address = addr  # the end model's ID register
        self.size = addr + 2 - cfg.base_address
        self._regs = [0] * self.size
        self._regs[0:2] = list(SUNSPEC_MARKER)
        self._regs[addr - cfg.base_address : addr - cfg.base_address + 2] = [
            SUNSPEC_END_MODEL_ID,
            0,
        ]
        # point lookup: (model, point) -> (address, type, size, sf point, writable)
        self._points: dict[tuple[int, str], tuple[int, str, int, str | None, bool]] = {}
        # register -> (model, point) for write validation
        self._owner: dict[int, tuple[int, str]] = {}
        for layout in self.models:
            offset = 0
            for name, stype, size, sf, rw in MODEL_POINTS[layout.model_id]:
                if offset >= layout.length + 2:
                    break  # model 1 at L = 65: no Pad
                address = layout.address + offset
                self._points[(layout.model_id, name)] = (address, stype, size, sf, rw)
                for i in range(size):
                    self._owner[address + i] = (layout.model_id, name)
                self._set_raw(layout.model_id, name, _not_implemented(stype, size))
                offset += size
            self._set_raw(layout.model_id, "ID", [layout.model_id])
            self._set_raw(layout.model_id, "L", [layout.length])
        self._init_values()
        self.step(0.0)

    # -- layout -------------------------------------------------------------

    def model_address(self, model_id: int) -> int:
        """Wire address of *model_id*'s ``ID`` register."""
        for m in self.models:
            if m.model_id == model_id:
                return m.address
        raise KeyError(model_id)

    def point_address(self, model_id: int, point: str) -> int:
        return self._points[(model_id, point)][0]

    # -- raw register access --------------------------------------------------

    def _idx(self, address: int) -> int:
        return address - self.config.base_address

    def _set_raw(self, model_id: int, point: str, regs: list[int]) -> None:
        address, _stype, size, _sf, _rw = self._points[(model_id, point)]
        if len(regs) != size:
            raise ValueError(f"{model_id}.{point}: {len(regs)} registers, expected {size}")
        i = self._idx(address)
        self._regs[i : i + size] = [r & 0xFFFF for r in regs]

    def raw(self, model_id: int, point: str) -> list[int]:
        address, _stype, size, _sf, _rw = self._points[(model_id, point)]
        i = self._idx(address)
        return list(self._regs[i : i + size])

    def read(self, address: int, count: int = 1) -> list[int] | int:
        """Holding registers ``address .. address+count-1`` or an exception code."""
        i = self._idx(address)
        if count < 1 or i < 0 or i + count > self.size:
            return ILLEGAL_ADDRESS
        return list(self._regs[i : i + count])

    # -- engineering values ---------------------------------------------------

    def _sf(self, model_id: int, sf_point: str | None) -> int | None:
        if sf_point is None:
            return 0
        [raw] = self.raw(model_id, sf_point)
        return None if raw == 0x8000 else (raw - 0x10000 if raw >= 0x8000 else raw)

    def value(self, model_id: int, point: str) -> float | None:
        """Engineering value of a point (scale factor applied); None if not implemented."""
        _address, stype, size, sf_point, _rw = self._points[(model_id, point)]
        regs = self.raw(model_id, point)
        if regs == _not_implemented(stype, size) and stype not in ("acc32", "string"):
            return None
        v = _decode(regs, stype)
        if math.isnan(v):
            return None
        sf = self._sf(model_id, sf_point)
        return None if sf is None else v * 10.0**sf

    def set_value(self, model_id: int, point: str, value: float | None) -> None:
        """Store an engineering value (scaled by the point's SF); None = not implemented."""
        _address, stype, size, sf_point, _rw = self._points[(model_id, point)]
        if value is None:
            self._set_raw(model_id, point, _not_implemented(stype, size))
            return
        if stype == "float32":
            self._set_raw(model_id, point, encode_value(value, "float32"))
            return
        sf = self._sf(model_id, sf_point)
        if sf is None:
            raise ValueError(f"{model_id}.{point}: scale factor {sf_point} not implemented")
        raw = round(value / 10.0**sf)
        wire = _WIRE_TYPE[stype]
        if wire in ("uint16", "uint32", "acc32"):
            hi = 0xFFFF if wire == "uint16" else 0xFFFFFFFF
            raw = max(0, min(hi - 1, raw))  # never emit the "not implemented" value
        elif wire == "int16":
            raw = max(-0x7FFF, min(0x7FFF, raw))
        if wire == "acc32":
            raw = max(1, raw)  # 0 means "not implemented" for accumulators
        self._set_raw(model_id, point, encode_value(raw, wire))

    # -- initial values ---------------------------------------------------------

    def _init_values(self) -> None:
        cfg = self.config
        inv = cfg.inverter_model
        strings = {
            "Mn": cfg.manufacturer,
            "Md": cfg.model_name or f"SunSpec sim {inv}",
            "Opt": cfg.options,
            "Vr": cfg.version,
            "SN": cfg.serial,
        }
        for point, text in strings.items():
            self._set_raw(1, point, _encode_string(text, self._points[(1, point)][2]))
        self._set_raw(1, "DA", [cfg.unit_id])
        for model_id, sfs in SCALE_FACTORS.items():
            if model_id in (103, 113) and model_id != inv:
                continue
            for point, sf in sfs.items():
                self._set_raw(model_id, point, [0x8000 if sf is None else sf & 0xFFFF])
        # Inverter: events clear (Evt1/Evt2 mandatory), vendor events not implemented.
        for point in ("Evt1", "Evt2"):
            self._set_raw(inv, point, [0, 0])
        # Model 123 defaults: connected, 100 % limit disabled, PF 1.000 disabled.
        for point, v in (
            ("Conn_WinTms", 0),
            ("Conn_RvrtTms", 0),
            ("Conn", 1),
            ("WMaxLimPct_WinTms", 0),
            ("WMaxLimPct_RvrtTms", 0),
            ("WMaxLimPct_RmpTms", 0),
            ("WMaxLim_Ena", 0),
            ("OutPFSet_WinTms", 0),
            ("OutPFSet_RvrtTms", 0),
            ("OutPFSet_RmpTms", 0),
            ("OutPFSet_Ena", 0),
            ("VArPct_Ena", 0),
        ):
            self._set_raw(123, point, [v])
        self.set_value(123, "WMaxLimPct", 100.0)
        self.set_value(123, "OutPFSet", 1.0)
        # Model 124 defaults: limits inactive, rates 100 %, charging from grid allowed.
        self.set_value(124, "WChaMax", cfg.battery_max_w)
        self.set_value(124, "WChaGra", 100.0)
        self.set_value(124, "WDisChaGra", 100.0)
        self._set_raw(124, "StorCtl_Mod", [0])
        self.set_value(124, "MinRsvPct", cfg.min_reserve_pct)
        self.set_value(124, "OutWRte", 100.0)
        self.set_value(124, "InWRte", 100.0)
        for point in ("InOutWRte_WinTms", "InOutWRte_RvrtTms", "InOutWRte_RmpTms"):
            self._set_raw(124, point, [0])
        self._set_raw(124, "ChaGriSet", [1])

    # -- writes -----------------------------------------------------------------

    def _check(self, model_id: int, point: str, value: float) -> bool:
        cfg = self.config
        if point == "DA":
            return 1 <= value <= 247
        if point in ("Conn", "WMaxLim_Ena", "OutPFSet_Ena", "VArPct_Ena", "ChaGriSet"):
            return value in (0, 1)
        if point == "VArPct_Mod":
            return 0 <= value <= 3
        if point == "WMaxLimPct":
            return 0 <= value <= 100
        if point == "OutPFSet":
            return 0.8 <= abs(value) <= 1.0
        if point == "StorCtl_Mod":
            return 0 <= value <= 3
        if point in ("OutWRte", "InWRte"):
            return -100 <= value <= 100
        if point == "MinRsvPct":
            return 0 <= value < 100
        if point == "WChaMax":
            return 0 <= value <= cfg.battery_max_w
        return True

    def write(self, address: int, values: Sequence[int]) -> int | None:
        """Write holding registers; returns a Modbus exception code on rejection.

        The request is applied atomically: either every register is stored
        or none.
        """
        values = [int(v) & 0xFFFF for v in values]
        if not values:
            return ILLEGAL_VALUE
        i = self._idx(address)
        if i < 0 or i + len(values) > self.size:
            return ILLEGAL_ADDRESS
        staged = list(self._regs)
        staged[i : i + len(values)] = values
        touched: dict[tuple[int, str], None] = {}
        for a in range(address, address + len(values)):
            owner = self._owner.get(a)
            if owner is None or not self._points[owner][4]:
                return ILLEGAL_ADDRESS  # marker, headers, read-only points
            touched[owner] = None
        for model_id, point in touched:
            p_address, stype, size, sf_point, _rw = self._points[(model_id, point)]
            if p_address < address or p_address + size > address + len(values):
                return ILLEGAL_ADDRESS  # partial write of a multi-register point
            regs = staged[self._idx(p_address) : self._idx(p_address) + size]
            sf = self._sf(model_id, sf_point)
            if sf is None:
                return ILLEGAL_ADDRESS  # scale factor not implemented: point unusable
            if not self._check(model_id, point, _decode(regs, stype) * 10.0**sf):
                return ILLEGAL_VALUE
        self._regs = staged
        self.writes += 1
        for _model_id, point in touched:
            self._written_at[point] = self.now
        logger.info(
            "write %d..%d: %s",
            address,
            address + len(values) - 1,
            ", ".join(f"{m}.{p}={self.value(m, p)}" for m, p in touched),
        )
        return None

    # -- physics ----------------------------------------------------------------

    def _since_write(self, *points: str) -> float:
        last = max((self._written_at.get(p, 0.0) for p in points), default=0.0)
        return self.now - last

    def _revert(self) -> None:
        conn_rvrt = self.value(123, "Conn_RvrtTms") or 0.0
        if (
            self.value(123, "Conn") == 0
            and conn_rvrt > 0
            and self._since_write("Conn", "Conn_RvrtTms") >= conn_rvrt
        ):
            self._set_raw(123, "Conn", [1])
            self.reverts.append((self.now, "Conn"))
        lim_rvrt = self.value(123, "WMaxLimPct_RvrtTms") or 0.0
        if (
            self.value(123, "WMaxLim_Ena") == 1
            and lim_rvrt > 0
            and self._since_write("WMaxLimPct", "WMaxLim_Ena", "WMaxLimPct_RvrtTms") >= lim_rvrt
        ):
            self._set_raw(123, "WMaxLim_Ena", [0])
            self.reverts.append((self.now, "WMaxLim_Ena"))
        rte_rvrt = self.value(124, "InOutWRte_RvrtTms") or 0.0
        if (
            (self.value(124, "StorCtl_Mod") or 0) != 0
            and rte_rvrt > 0
            and self._since_write("OutWRte", "InWRte", "StorCtl_Mod", "InOutWRte_RvrtTms")
            >= rte_rvrt
        ):
            self._set_raw(124, "StorCtl_Mod", [0])
            self.reverts.append((self.now, "StorCtl_Mod"))

    @property
    def soc_pct(self) -> float:
        return 100.0 * self.battery_energy_wh / self.config.battery_capacity_wh

    @property
    def power_limit_w(self) -> float:
        """AC output cap currently in force (model 123)."""
        if self.value(123, "WMaxLim_Ena") == 1:
            return self.config.wmax_w * (self.value(123, "WMaxLimPct") or 0.0) / 100.0
        return self.config.wmax_w

    def _battery_charge_w(self, dt: float) -> float:
        """Battery charge power (W, negative = discharge) requested by model 124."""
        cfg = self.config
        cmax = self.value(124, "WChaMax") or 0.0
        mode = int(self.value(124, "StorCtl_Mod") or 0)
        hi = cmax * (self.value(124, "InWRte") or 0.0) / 100.0 if mode & 1 else cmax
        lo = -cmax * (self.value(124, "OutWRte") or 0.0) / 100.0 if mode & 2 else -cmax
        charge = 0.0 if lo > hi else min(max(0.0, lo), hi)
        if charge > 0 and self.value(124, "ChaGriSet") == 0:
            charge = min(charge, self.pv_available_w)
        # Energy limits: full / reserve.
        eta = cfg.battery_efficiency
        if charge > 0:
            room = cfg.battery_capacity_wh - self.battery_energy_wh
            if room <= 1e-9:
                charge = 0.0
            elif dt > 0:
                charge = min(charge, room * 3600.0 / dt / eta)
        elif charge < 0:
            reserve = cfg.battery_capacity_wh * (self.value(124, "MinRsvPct") or 0.0) / 100.0
            avail = self.battery_energy_wh - reserve
            if avail <= 1e-9:
                charge = 0.0
            elif dt > 0:
                charge = max(charge, -avail * 3600.0 / dt * eta)
        return charge

    def step(self, dt: float) -> None:
        """Advance the simulation by *dt* seconds and refresh the register image."""
        cfg = self.config
        dt = max(0.0, float(dt))
        self.now += dt
        self._revert()
        connected = self.value(123, "Conn") != 0
        pv_avail = max(0.0, self.pv_available_w)
        limit = self.power_limit_w
        charge = self._battery_charge_w(dt) if connected else 0.0
        batt_ac = -charge + 0.0  # no negative zero
        if batt_ac > limit:  # discharge alone exceeds the export cap
            batt_ac = limit
        pv_out = min(pv_avail, max(0.0, limit - batt_ac)) if connected else 0.0
        ac = pv_out + batt_ac
        if ac < -cfg.wmax_w:  # grid charging beyond the inverter rating
            batt_ac = -cfg.wmax_w - pv_out
            ac = -cfg.wmax_w
        charge = -batt_ac
        eta = cfg.battery_efficiency
        delta = charge * dt / 3600.0
        self.battery_energy_wh += delta * eta if delta > 0 else delta / eta
        self.battery_energy_wh = max(0.0, min(cfg.battery_capacity_wh, self.battery_energy_wh))
        self.energy_wh += max(0.0, ac) * dt / 3600.0
        self.ac_power_w, self.pv_power_w, self.battery_power_w = ac, pv_out, batt_ac

        if not connected:
            st = ST_STANDBY
        elif pv_out < pv_avail - 0.5:
            st = ST_THROTTLED
        elif abs(ac) > 0.5 or pv_out > 0.5:
            st = ST_MPPT
        else:
            st = ST_SLEEPING
        self._update_inverter(st)
        self._update_storage(charge)

    def _update_inverter(self, st: int) -> None:
        cfg = self.config
        inv = self.inverter_model
        ac = self.ac_power_w
        amps = abs(ac) / (3.0 * cfg.grid_v)
        load = abs(ac) / cfg.wmax_w
        dcv = 400.0
        dcw = self.pv_power_w / 0.97
        values: dict[str, float | None] = {
            "A": 3 * amps,
            "AphA": amps,
            "AphB": amps,
            "AphC": amps,
            "PPVphAB": cfg.grid_v * math.sqrt(3),
            "PPVphBC": cfg.grid_v * math.sqrt(3),
            "PPVphCA": cfg.grid_v * math.sqrt(3),
            "PhVphA": cfg.grid_v,
            "PhVphB": cfg.grid_v,
            "PhVphC": cfg.grid_v,
            "W": ac,
            "Hz": cfg.grid_hz,
            "VA": abs(ac),
            "VAr": 0.0,
            "PF": 100.0,
            "WH": self.energy_wh,
            "DCA": dcw / dcv,
            "DCV": dcv,
            "DCW": dcw,
            "TmpCab": 30.0 + 10.0 * load,
            "TmpSnk": 35.0 + 25.0 * load,
            "TmpTrns": None,
            "TmpOt": None,
        }
        for point, v in values.items():
            self.set_value(inv, point, v)
        self._set_raw(inv, "St", [st])

    def _update_storage(self, charge: float) -> None:
        cfg = self.config
        soc = self.soc_pct
        reserve = self.value(124, "MinRsvPct") or 0.0
        volts = 380.0 + 40.0 * soc / 100.0
        self.set_value(124, "ChaState", soc)
        self.set_value(124, "InBatV", volts)
        usable_wh = max(0.0, self.battery_energy_wh - cfg.battery_capacity_wh * reserve / 100.0)
        self.set_value(124, "StorAval", usable_wh / volts)
        if charge > 0.5:
            st = CHA_CHARGING
        elif charge < -0.5:
            st = CHA_DISCHARGING
        elif soc >= 99.95:
            st = CHA_FULL
        elif soc <= reserve + 0.05:
            st = CHA_EMPTY
        else:
            st = CHA_HOLDING
        self._set_raw(124, "ChaSt", [st])

    def summary(self) -> str:
        lim = (
            f"{self.value(123, 'WMaxLimPct'):.1f}%"
            if self.value(123, "WMaxLim_Ena") == 1
            else "off"
        )
        return (
            f"t={self.now:.0f}s ac={self.ac_power_w:.0f}W pv={self.pv_power_w:.0f}/"
            f"{self.pv_available_w:.0f}W battery={self.battery_power_w:+.0f}W "
            f"soc={self.soc_pct:.1f}% limit={lim} stor_ctl={self.raw(124, 'StorCtl_Mod')[0]}"
        )

    def describe(self) -> list[str]:
        lines = [f"{self.config.base_address}: SunS marker"]
        for m in self.models:
            lines.append(f"{m.address}: model {m.model_id} (L={m.length})")
        lines.append(f"{self.end_address}: end model (0xFFFF, L=0)")
        return lines


# ---------------------------------------------------------------------------
# pymodbus TCP server (pymodbus 3.6+)
# ---------------------------------------------------------------------------


def _exc_code(code: int) -> Any:
    try:
        from pymodbus.constants import ExcCodes

        return ExcCodes(code)
    except ImportError:  # pymodbus < 3.10: plain ints
        return code


def _new_style_context(device: SunSpecDevice) -> Any | None:
    """A ``SimDevice`` with an access hook (pymodbus >= 3.13), else None.

    pymodbus 3.12 has an experimental ``SimDevice`` with a different action
    signature; it (and older versions) use :func:`_legacy_context`.
    """
    try:
        import pymodbus

        server_mod = importlib.import_module("pymodbus.server")
        sim_mod = importlib.import_module("pymodbus.simulator")
    except ImportError:
        return None
    try:
        version = tuple(int(x) for x in str(pymodbus.__version__).split(".")[:2])
    except ValueError:
        version = (0, 0)
    if version < (3, 13):
        return None
    sim_device = getattr(sim_mod, "SimDevice", None)
    sim_data = getattr(sim_mod, "SimData", None)
    data_type = getattr(sim_mod, "DataType", None)
    if sim_device is None or sim_data is None or data_type is None:
        return None
    fields = getattr(sim_device, "__dataclass_fields__", {})
    ctx_param = inspect.signature(server_mod.ModbusTcpServer.__init__).parameters.get("context")
    if "action" not in fields or ctx_param is None or "SimDevice" not in str(ctx_param.annotation):
        return None

    async def action(
        func_code: int,
        start_address: int,
        address: int,
        count: int,
        registers: list[int],
        set_values: list[int] | list[bool] | None,
    ) -> Any:
        if func_code not in (3, 4, 6, 16, 23):
            return _exc_code(0x01)
        if set_values is not None:
            code = device.write(address, [int(v) for v in set_values])
            return None if code is None else _exc_code(code)
        got = device.read(address, count)
        if isinstance(got, int):
            return _exc_code(got)
        offset = address - start_address
        registers[offset : offset + count] = got
        return None

    base = device.config.base_address
    return sim_device(
        id=device.config.unit_id,
        simdata=[sim_data(base, count=device.size, values=0, datatype=data_type.REGISTERS)],
        action=action,
    )


def _legacy_context(device: SunSpecDevice) -> Any:
    """Datablock-backed server context for pymodbus 3.6 .. 3.11."""
    store = importlib.import_module("pymodbus.datastore.store")
    ds = importlib.import_module("pymodbus.datastore")
    base_block: Any = store.BaseModbusDataBlock
    # The device context adds one to every wire address before calling the block.
    shift = 1

    class _Block(base_block):  # type: ignore[misc, valid-type]
        def __init__(self) -> None:
            self.address = device.config.base_address + shift
            self.default_value = 0
            self.values = [0] * device.size

        def validate(self, address: int, count: int = 1) -> bool:  # pymodbus < 3.9
            return not isinstance(device.read(address - shift, count), int)

        def getValues(self, address: int, count: int = 1) -> Any:
            got = device.read(address - shift, count)
            return _exc_code(got) if isinstance(got, int) else got

        def setValues(self, address: int, values: Any) -> Any:
            vals = values if isinstance(values, list) else [values]
            code = device.write(address - shift, [int(v) for v in vals])
            return None if code is None else _exc_code(code)

        def reset(self) -> None:
            return None

    device_ctx_cls = getattr(ds, "ModbusDeviceContext", None) or ds.ModbusSlaveContext
    hr = _Block()
    # di/co passed explicitly: pymodbus 3.9 drops hr/ir when di is None.
    bits = ds.ModbusSequentialDataBlock(1, [0])
    ctx = device_ctx_cls(di=bits, co=bits, hr=hr, ir=hr)
    return ds.ModbusServerContext({device.config.unit_id: ctx}, single=False)


class SunSpecSimulator:
    """A :class:`SunSpecDevice` served over Modbus TCP, advancing in real time.

    *tick_s* is the real-time step interval (None: no background stepping,
    the caller drives :meth:`SunSpecDevice.step`); *speed* multiplies
    simulated time (e.g. 60 = one simulated minute per second).
    """

    def __init__(
        self,
        config: SimulatorConfig | None = None,
        *,
        tick_s: float | None = 1.0,
        speed: float = 1.0,
    ) -> None:
        self.device = SunSpecDevice(config)
        self.tick_s = tick_s
        self.speed = float(speed)
        self.host = "127.0.0.1"
        self.port = 0
        self._server: Any = None
        self._serve_task: asyncio.Task[Any] | None = None
        self._tick_task: asyncio.Task[None] | None = None

    async def start(self, host: str = "127.0.0.1", port: int = 0) -> SunSpecSimulator:
        """Listen on *host*:*port* (0 = an ephemeral port, see :attr:`port`)."""
        from pymodbus.server import ModbusTcpServer

        context = _new_style_context(self.device) or _legacy_context(self.device)
        server_cls: Any = ModbusTcpServer
        self._server = server_cls(context, address=(host, int(port)))
        self._serve_task = asyncio.create_task(self._server.serve_forever())
        for _ in range(200):
            await asyncio.sleep(0.01)
            if self._serve_task.done():
                self._serve_task.result()  # raises the startup error
                raise RuntimeError("Modbus server stopped during startup")
            listener = getattr(self._server, "transport", None)
            sockets = getattr(listener, "sockets", None)
            if sockets:
                self.host, self.port = host, int(sockets[0].getsockname()[1])
                break
        else:
            await self.stop()
            raise RuntimeError(f"Modbus server did not start listening on {host}:{port}")
        if self.tick_s is not None and self.tick_s > 0:
            self._tick_task = asyncio.create_task(self._tick_loop(float(self.tick_s)))
        logger.info("SunSpec simulator listening on %s:%d", self.host, self.port)
        return self

    async def _tick_loop(self, interval: float) -> None:
        last = time.monotonic()
        while True:
            await asyncio.sleep(interval)
            now = time.monotonic()
            self.device.step((now - last) * self.speed)
            last = now

    async def stop(self) -> None:
        for task in (self._tick_task,):
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        self._tick_task = None
        if self._server is not None:
            with contextlib.suppress(Exception):
                await self._server.shutdown()
        if self._serve_task is not None:
            self._serve_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._serve_task
        self._server = self._serve_task = None

    async def __aenter__(self) -> SunSpecSimulator:
        if self._server is None:
            await self.start(self.host, self.port)
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    defaults = SimulatorConfig()
    p = argparse.ArgumentParser(
        prog=prog,
        description="Serve a simulated SunSpec PV + battery inverter over Modbus TCP "
        "(models 1, 103/113, 123, 124). Not a model of any vendor's firmware.",
    )
    p.add_argument("--host", default="127.0.0.1", help="bind address (default 127.0.0.1)")
    p.add_argument("--port", type=int, default=5020, help="TCP port (default 5020; 0 = any)")
    p.add_argument("--unit-id", type=int, default=defaults.unit_id)
    p.add_argument("--inverter-model", type=int, choices=INVERTER_MODELS, default=103)
    p.add_argument(
        "--common-length",
        type=int,
        choices=(65, 66),
        default=defaults.common_length,
        help="L of model 1: 65 as SolarEdge/Fronius report (default), 66 with Pad",
    )
    p.add_argument("--base-address", type=int, choices=SUNSPEC_BASE_ADDRESSES, default=40000)
    p.add_argument("--wmax", type=float, default=defaults.wmax_w, help="AC rating in W")
    p.add_argument("--pv", type=float, default=defaults.pv_available_w, help="available PV W")
    p.add_argument("--battery-capacity", type=float, default=defaults.battery_capacity_wh)
    p.add_argument("--battery-power", type=float, default=defaults.battery_max_w)
    p.add_argument("--soc", type=float, default=defaults.soc_pct, help="initial SoC in %%")
    p.add_argument("--min-reserve", type=float, default=defaults.min_reserve_pct)
    p.add_argument("--serial", default=defaults.serial)
    p.add_argument("--tick", type=float, default=1.0, help="real seconds per step")
    p.add_argument("--speed", type=float, default=1.0, help="simulated seconds per real second")
    p.add_argument(
        "--status-interval", type=float, default=10.0, help="seconds between status lines"
    )
    p.add_argument("--log-level", default="INFO")
    return p


def config_from_args(args: argparse.Namespace) -> SimulatorConfig:
    return SimulatorConfig(
        inverter_model=args.inverter_model,
        unit_id=args.unit_id,
        base_address=args.base_address,
        common_length=args.common_length,
        serial=args.serial,
        wmax_w=args.wmax,
        pv_available_w=args.pv,
        battery_capacity_wh=args.battery_capacity,
        battery_max_w=args.battery_power,
        soc_pct=args.soc,
        min_reserve_pct=args.min_reserve,
    )


async def serve(
    config: SimulatorConfig,
    *,
    host: str,
    port: int,
    tick_s: float = 1.0,
    speed: float = 1.0,
    status_interval_s: float = 10.0,
) -> None:
    """Run the simulator until cancelled, logging a status line periodically."""
    sim = SunSpecSimulator(config, tick_s=tick_s, speed=speed)
    await sim.start(host, port)
    try:
        for line in sim.device.describe():
            logger.info("  %s", line)
        while True:
            await asyncio.sleep(max(0.5, status_interval_s))
            logger.info(sim.device.summary())
    finally:
        await sim.stop()


def main(argv: Sequence[str] | None = None, *, prog: str | None = None) -> int:
    args = build_parser(prog).parse_args(argv)
    logging.basicConfig(
        level=str(args.log_level).upper(), format="%(asctime)s %(levelname)s %(message)s"
    )
    # pymodbus logs every request at INFO in some versions; keep it quiet.
    logging.getLogger("pymodbus").setLevel(logging.WARNING)
    try:
        config = config_from_args(args)
        config.validate()
    except ValueError as exc:
        logger.error("invalid configuration: %s", exc)
        return 2
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(
            serve(
                config,
                host=args.host,
                port=args.port,
                tick_s=args.tick,
                speed=args.speed,
                status_interval_s=args.status_interval,
            )
        )
    return 0


__all__ = [
    "INVERTER_MODELS",
    "MODEL_POINTS",
    "SCALE_FACTORS",
    "SimulatorConfig",
    "SunSpecDevice",
    "SunSpecSimulator",
    "build_parser",
    "main",
    "model_length",
    "serve",
]

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
