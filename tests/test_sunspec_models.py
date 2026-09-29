"""The shipped SunSpec register maps against the official SunSpec model definitions.

Every register that records a SunSpec point (``RegisterDefinition.sunspec``)
is compared with the model JSON bundled with ``pysunspec2`` (the SunSpec
Alliance's machine-readable copy of https://github.com/sunspec/models):
offset from the model's ``ID`` register, size, type, units, write access and
scale-factor pairing. The conformance test is skipped when ``pysunspec2`` is
not installed (it is in the ``dev`` extra); the hard-coded address checks
below run everywhere.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

import pytest

from vpp.protocols.modbus import (
    INVERTER_MAPS,
    SUNSPEC_INVERTER_BASE,
    SUNSPEC_MODEL_123_LENGTH,
    SUNSPEC_MODEL_124_LENGTH,
    RegisterDefinition,
    sunspec_model_123_registers,
    sunspec_model_124_registers,
    sunspec_not_implemented,
)

BASE = 1000  # arbitrary model_base for the model-relative control blocks

MAPS: dict[str, dict[str, RegisterDefinition]] = {
    "model_123": sunspec_model_123_registers(BASE),
    "model_124": sunspec_model_124_registers(BASE),
    "fronius_symo": INVERTER_MAPS["fronius_symo"].registers,
    "solaredge_se": INVERTER_MAPS["solaredge_se"].registers,
}

# SunSpec point type -> data types this project may read/write it as.
COMPATIBLE_TYPES: dict[str, set[str]] = {
    "uint16": {"uint16"},
    "enum16": {"uint16"},
    "bitfield16": {"uint16"},
    "int16": {"int16"},
    "sunssf": {"int16"},
    "acc32": {"acc32"},
    "uint32": {"uint32"},
    "int32": {"int32"},
    "float32": {"float32"},
}

# SunSpec units -> the unit string used in the maps.
UNITS: dict[str, str] = {
    "": "",
    "W": "W",
    "Wh": "Wh",
    "Hz": "Hz",
    "C": "°C",
    "Secs": "s",
    "% WMax": "%",
    "% WDisChaMax": "%",
    "% WChaMax": "%",
    "% AhrRtg": "%",
}


@cache
def _model(model_id: int) -> dict[str, Any]:
    sunspec2 = pytest.importorskip(
        "sunspec2", reason="pysunspec2 not installed (pip install -e '.[dev]')"
    )
    path = Path(sunspec2.__file__).parent / "models" / "json" / f"model_{model_id}.json"
    return json.loads(path.read_text())


def _points(model_id: int) -> dict[str, dict[str, Any]]:
    """Top-level points of *model_id* with their offset from the ``ID`` register."""
    out: dict[str, dict[str, Any]] = {}
    offset = 0
    for point in _model(model_id)["group"]["points"]:
        out[point["name"]] = {**point, "offset": offset}
        offset += int(point["size"])
    return out


def _cases() -> list[tuple[str, str]]:
    return [
        (map_name, key)
        for map_name, regs in MAPS.items()
        for key, reg in regs.items()
        if reg.sunspec is not None
    ]


@pytest.mark.parametrize(("map_name", "key"), _cases())
def test_register_matches_sunspec_model_definition(map_name: str, key: str) -> None:
    regs = MAPS[map_name]
    reg = regs[key]
    ref = reg.sunspec
    assert ref is not None
    assert _model(ref.model)["id"] == ref.model
    points = _points(ref.model)
    assert ref.point in points, f"{ref.point} is not a point of model {ref.model}"
    point = points[ref.point]
    where = f"{map_name}.{key} -> model {ref.model} {ref.point}"

    assert reg.address - ref.base == point["offset"], f"{where}: offset"
    assert reg.count == point["size"], f"{where}: size"
    assert reg.data_type in COMPATIBLE_TYPES[point["type"]], f"{where}: type {point['type']}"
    units = point.get("units", "").strip()
    assert UNITS[units] == reg.unit, f"{where}: units {units!r}"
    if reg.writable:
        assert point.get("access") == "RW", f"{where}: written but read-only in the spec"

    # Scale factors: paired exactly as the spec says, never hard-coded.
    sf_point = point.get("sf")
    if sf_point is None:
        assert reg.scale_factor is None, f"{where}: spec has no scale factor"
    else:
        assert reg.scale_factor is not None, f"{where}: missing scale factor {sf_point}"
        sf_reg = regs[reg.scale_factor]
        assert sf_reg.sunspec is not None and sf_reg.sunspec.point == sf_point, where
        assert sf_reg.sunspec.base == ref.base, where
        assert reg.scale == 1.0, f"{where}: scale must come from {sf_point}"


def test_model_lengths_and_shared_inverter_layout() -> None:
    for model_id, length in ((123, SUNSPEC_MODEL_123_LENGTH), (124, SUNSPEC_MODEL_124_LENGTH)):
        points = _model(model_id)["group"]["points"]
        assert sum(int(p["size"]) for p in points) - 2 == length
    # SolarEdge devices report model 101, 102 or 103; the map is valid for all.
    for key, reg in INVERTER_MAPS["solaredge_se"].registers.items():
        assert reg.sunspec is not None
        for model_id in (101, 102):
            point = _points(model_id)[reg.sunspec.point]
            assert point["offset"] == reg.address - reg.sunspec.base, (model_id, key)
            assert point["type"] == _points(103)[reg.sunspec.point]["type"], (model_id, key)
    # Fronius single/split-phase float models share the model 113 layout.
    for key, reg in INVERTER_MAPS["fronius_symo"].registers.items():
        assert reg.sunspec is not None
        for model_id in (111, 112):
            point = _points(model_id)[reg.sunspec.point]
            assert point["offset"] == reg.address - reg.sunspec.base, (model_id, key)


def test_hard_coded_addresses() -> None:
    """Verified offsets, spelled out (these run without pysunspec2).

    Model 123: WMaxLimPct +5, WMaxLimPct_RvrtTms +7, WMaxLim_Ena +9,
    WMaxLimPct_SF +23. Model 124: StorCtl_Mod +5, OutWRte +12, InWRte +13,
    InOutWRte_RvrtTms +15, InOutWRte_SF +25. Model 103 (ID at 40069):
    W +14, W_SF +15, Hz +16, Hz_SF +17, WH +24 (acc32), WH_SF +26, DCW +31,
    DCW_SF +32, TmpSnk +34, Tmp_SF +37. Model 113: W +22, Hz +24, WH +32,
    DCW +38 (float32, 2 registers each).
    """
    m123 = sunspec_model_123_registers(0)
    assert [m123[k].address for k in ("wmax_lim_pct", "wmax_lim_pct_rvrt_tms")] == [5, 7]
    assert [m123[k].address for k in ("wmax_lim_ena", "wmax_lim_pct_sf")] == [9, 23]
    m124 = sunspec_model_124_registers(0)
    assert [
        m124[k].address for k in ("stor_ctl_mod", "out_w_rte", "in_w_rte", "in_out_w_rte_rvrt_tms")
    ] == [5, 12, 13, 15]
    assert m124["in_out_w_rte_sf"].address == 25

    assert SUNSPEC_INVERTER_BASE == 40069
    se = {k: r.address for k, r in INVERTER_MAPS["solaredge_se"].registers.items()}
    assert se == {
        "ac_power": 40083,
        "ac_power_scale": 40084,
        "frequency": 40085,
        "frequency_scale": 40086,
        "ac_energy": 40093,
        "ac_energy_scale": 40095,
        "dc_power": 40100,
        "dc_power_scale": 40101,
        "temperature": 40103,
        "temperature_scale": 40106,
    }
    fr = {k: (r.address, r.count) for k, r in INVERTER_MAPS["fronius_symo"].registers.items()}
    assert fr == {
        "ac_power": (40091, 2),
        "frequency": (40093, 2),
        "ac_energy": (40101, 2),
        "dc_power": (40107, 2),
    }


@pytest.mark.parametrize(
    ("regs", "data_type", "expected"),
    [
        ([0x8000], "int16", True),  # also sunssf
        ([0x7FFF], "int16", False),
        ([0xFFFF], "uint16", True),  # also enum16 / bitfield16
        ([0xFFFE], "uint16", False),
        ([0x8000, 0x0000], "int32", True),
        ([0xFFFF, 0xFFFF], "uint32", True),
        ([0x0000, 0x0000], "acc32", True),
        ([0x0000, 0x0001], "acc32", False),
        ([0x7FC0, 0x0000], "float32", True),  # NaN
        ([0x4248, 0x0000], "float32", False),  # 50.0
    ],
)
def test_sunspec_not_implemented_sentinels(regs, data_type, expected) -> None:
    assert sunspec_not_implemented(regs, data_type) is expected
